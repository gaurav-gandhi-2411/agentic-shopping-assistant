import logging
import pickle
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Bounded LRU of raw BM25 score vectors, keyed on the exact tokenised query.  An outfit turn
# re-issues the same few slot queries across its base/colour/formality compositions and
# rank_bm25's pure-Python get_scores is ~68% of a search, so reusing the vector removes the
# repeats.  Only the RAW vector is cached (a pure function of the immutable index and the
# tokens); allowed_ids masking and top-k selection run afterwards on every call, so
# exclusion/filter arguments never see a stale result.  0 disables the cache (bypass).
SCORE_CACHE_SIZE = 16


class _CscBm25:
    """Column-compressed (term -> postings) copy of a BM25Okapi's term frequencies that
    reproduces ``BM25Okapi.get_scores`` without its per-query Python loop over every document.

    rank_bm25 evaluates, per query token ``q`` (repeats included, in order)::

        score += idf[q] * (tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl)))

    over ALL documents, where tf is 0 for documents lacking ``q`` (adding exactly 0.0). Here the
    identical expression is evaluated only on the documents that contain ``q`` and accumulated in
    the same token order, using the same float64 operations in the same order, so the resulting
    vector is bit-identical to rank_bm25's (asserted in tests and on the real index).
    """

    def __init__(self, bm25: Any) -> None:
        doc_freqs = bm25.doc_freqs
        n = len(doc_freqs)
        self.corpus_size: int = n
        self.k1: float = bm25.k1
        self.idf: dict[str, float] = bm25.idf
        # Exactly rank_bm25's `self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)`.
        doc_len = np.array(bm25.doc_len)
        self.norm: np.ndarray = bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)
        self.term_id: dict[str, int] = {t: i for i, t in enumerate(self.idf)}
        lens = np.fromiter((len(d) for d in doc_freqs), dtype=np.int64, count=n)
        nnz = int(lens.sum())
        tid = self.term_id
        cols = np.fromiter((tid[t] for d in doc_freqs for t in d), dtype=np.int32, count=nnz)
        vals = np.fromiter(
            (v for d in doc_freqs for v in d.values()), dtype=np.float64, count=nnz
        )
        rows = np.repeat(np.arange(n, dtype=np.int32), lens)
        order = np.argsort(cols, kind="stable")  # stable: rows stay ascending inside a column
        self.rows: np.ndarray = rows[order]
        self.tf: np.ndarray = vals[order]
        self.indptr: np.ndarray = np.concatenate(
            ([0], np.cumsum(np.bincount(cols, minlength=len(tid))))
        )

    def get_scores(self, tokens: list[str]) -> np.ndarray:
        score = np.zeros(self.corpus_size)
        k1p1 = self.k1 + 1
        for q in tokens:
            idf = self.idf.get(q) or 0
            t = self.term_id.get(q)
            if t is None or idf == 0:
                continue  # rank_bm25 adds an all-zero vector here
            lo, hi = self.indptr[t], self.indptr[t + 1]
            rows = self.rows[lo:hi]
            tf = self.tf[lo:hi]
            score[rows] += idf * (tf * k1p1 / (tf + self.norm[rows]))
        return score


class SparseRetriever:
    def __init__(self, config: dict):
        self.config = config
        self.bm25: BM25Okapi | None = None
        self.article_ids: np.ndarray | None = None
        # Hash-based id -> position lookup, built once so per-search allowed_ids
        # filtering doesn't pay an O(n) np.isin scan over the full (68k+) object
        # dtype article_ids array on every call.
        self._id_to_pos: dict[str, int] | None = None
        self._init_score_cache()

    def _init_score_cache(self, size: int = SCORE_CACHE_SIZE) -> None:
        self._score_cache_size = size
        self._score_cache: OrderedDict[tuple[str, ...], np.ndarray] = OrderedDict()
        # Requests run on a threadpool; the lookup and move_to_end/popitem must not interleave.
        self._score_cache_lock = threading.Lock()
        # Strong reference to the bm25 the cached vectors were computed from. Compared with
        # `is`, never id(): a freed object's id can be reused by its replacement, which would
        # let stale vectors survive a swap. Holding the reference keeps the id unique.
        self._score_cache_owner: object | None = None

    def _csc_for(self, bm25: Any) -> _CscBm25 | None:
        """Postings structure for ``bm25`` (built once per bm25 object, rebuilt on swap), or
        None when the vectorised path is off or ``bm25`` is not a BM25Okapi-shaped object."""
        if not getattr(self, "_vectorised", True) or not hasattr(bm25, "doc_freqs"):
            return None
        lock = self.__dict__.setdefault("_csc_lock", threading.Lock())
        with lock:
            csc = self.__dict__.get("_csc")
            if csc is None or self.__dict__.get("_csc_owner") is not bm25:
                t0 = time.perf_counter()
                csc = _CscBm25(bm25)
                self.csc_build_seconds = time.perf_counter() - t0
                logger.info("BM25 postings built in %.2fs", self.csc_build_seconds)
                self._csc, self._csc_owner = csc, bm25  # strong ref: see _score_cache_owner
        return csc

    def _compute_scores(self, tokens: list[str]) -> np.ndarray:
        csc = self._csc_for(self.bm25)
        if csc is not None:
            return csc.get_scores(tokens)
        return self.bm25.get_scores(tokens)

    def _get_scores(self, tokens: list[str]) -> np.ndarray:
        """BM25 score vector for ``tokens``, memoised per exact token tuple (LRU).

        The returned array is a read-only VIEW of the cached vector: a caller can neither write
        to it nor re-enable writing (numpy refuses writeable=True on a view of a read-only base).
        """
        # getattr: instances built without __init__/load() (tests, tooling) simply bypass.
        cache_on = getattr(self, "_score_cache_size", 0) > 0
        key = tuple(tokens)
        owner = self.bm25
        if cache_on:
            with self._score_cache_lock:
                if self._score_cache_owner is not owner:  # index swapped: drop stale vectors
                    self._score_cache.clear()
                    self._score_cache_owner = owner
                cached = self._score_cache.get(key)
                if cached is not None:
                    self._score_cache.move_to_end(key)
                    return cached.view()
        scores = np.asarray(self._compute_scores(tokens))
        if cache_on:
            scores.flags.writeable = False
            with self._score_cache_lock:
                if self._score_cache_owner is owner:
                    self._score_cache[key] = scores
                    self._score_cache.move_to_end(key)
                    while len(self._score_cache) > self._score_cache_size:
                        self._score_cache.popitem(last=False)
            return scores.view()
        return scores

    def build_index(self, catalogue_df: pd.DataFrame, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Building BM25 index over %d articles…", len(catalogue_df))
        tokenized = [self._tokenize(t) for t in catalogue_df["search_text"].tolist()]
        self.bm25 = BM25Okapi(tokenized)
        self.article_ids = catalogue_df["article_id"].values.astype(str)
        self._id_to_pos = {aid: i for i, aid in enumerate(self.article_ids)}

        with open(save_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        np.save(str(save_dir / "bm25_article_ids.npy"), self.article_ids)
        logger.info("BM25 index saved.")

    @classmethod
    def load(cls, config: dict, save_dir: Path) -> "SparseRetriever":
        save_dir = Path(save_dir)
        instance = cls.__new__(cls)
        instance.config = config
        with open(save_dir / "bm25.pkl", "rb") as f:
            instance.bm25 = pickle.load(f)
        instance.article_ids = np.load(
            str(save_dir / "bm25_article_ids.npy"), allow_pickle=True
        )
        instance._id_to_pos = {aid: i for i, aid in enumerate(instance.article_ids)}
        instance._init_score_cache()
        instance._csc_for(instance.bm25)  # build at startup, not on the first user query
        return instance

    def search(
        self,
        query: str,
        top_k: int = 20,
        allowed_ids: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        tokens = self._tokenize(query)
        scores = self._get_scores(tokens)
        if allowed_ids is not None:
            # Zero out scores for items not in the allowed set so only pre-filtered
            # items compete for the top-k slots.  np.isin over the full (68k+)
            # object-dtype article_ids array is O(n) with no hash fast-path and
            # got expensive as the catalogue grew; instead build the boolean mask
            # by hashing the (much smaller) allowed_ids into positions via the
            # precomputed id->position dict.  Mask semantics are bit-identical to
            # np.isin(self.article_ids, allowed_ids).
            mask = np.zeros(len(self.article_ids), dtype=bool)
            allowed_str = np.asarray(allowed_ids).astype(str)
            positions = [self._id_to_pos[aid] for aid in allowed_str if aid in self._id_to_pos]
            mask[positions] = True
            scores = scores * mask
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (str(self.article_ids[i]), float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def has_any_known_token(self, query: str) -> bool:
        """True if ANY query token appears in the indexed catalogue vocabulary.

        Ground truth for the gibberish guard: a query sharing zero tokens with
        60k+ products' search text ("asdfgh qwerty zxcvb") cannot be answered by
        lexical or semantic search — dense similarity over noise still ranks
        SOMETHING first, which is how a keyboard-mash got a confident product
        recommendation live (defect sweep 2026-07-10, P0-4). Returns True when
        no index is loaded — never block search on a missing vocabulary.
        """
        if self.bm25 is None:
            return True
        idf: dict = getattr(self.bm25, "idf", {}) or {}
        return any(t in idf for t in self._tokenize(query))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if len(t) >= 2]
