import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class SparseRetriever:
    def __init__(self, config: dict):
        self.config = config
        self.bm25: BM25Okapi | None = None
        self.article_ids: np.ndarray | None = None
        # Hash-based id -> position lookup, built once so per-search allowed_ids
        # filtering doesn't pay an O(n) np.isin scan over the full (68k+) object
        # dtype article_ids array on every call.
        self._id_to_pos: dict[str, int] | None = None

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
        return instance

    def search(
        self,
        query: str,
        top_k: int = 20,
        allowed_ids: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
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
