import logging
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import tqdm

# tqdm's TMonitor background thread bootstraps/tears down a lock on every
# progress-bar instantiation, and SentenceTransformer.encode() creates one
# internally even when show_progress_bar=False.  Setting monitor_interval=0
# disables the monitor thread entirely (documented tqdm switch) — this must
# happen before sentence_transformers is imported/used so its internal tqdm
# usage picks up the disabled monitor.
tqdm.tqdm.monitor_interval = 0

from src.retrieval.onnx_encoders import OnnxTextEncoder, encoder_backend  # noqa: E402
from src.utils.timing_log import first_call  # noqa: E402

logger = logging.getLogger(__name__)

_QUERY_VEC_LOCK = threading.Lock()
_QUERY_VEC_CACHE_MAX = 256 # ~1.5 kB per 384-d float32 vector: <0.5 MB at the cap


def load_text_encoder(model_name: str) -> Any:
    """Query/document encoder: ONNX Runtime by default (no torch in the production image);
    ``ENCODER_BACKEND=torch`` loads sentence-transformers for local dev (needs
    requirements-export.txt). Both expose ``encode(texts, ..., normalize_embeddings=True)``."""
    if encoder_backend() == "torch":
        from sentence_transformers import SentenceTransformer  # lazy: absent from the runtime image

        return SentenceTransformer(model_name, device="cpu")
    return OnnxTextEncoder(model_name)


class DenseRetriever:
    def __init__(self, config: dict):
        self.config = config
        model_name = config["retrieval"]["dense_model"]
        self.model = load_text_encoder(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.article_ids: np.ndarray | None = None

    def build_index(self, catalogue_df: pd.DataFrame, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        texts = catalogue_df["search_text"].tolist()
        batch_size = self.config["retrieval"]["dense_batch_size"]

        logger.info("Encoding %d articles with %s…", len(texts), self.config["retrieval"]["dense_model"])
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.article_ids = catalogue_df["article_id"].values.astype(str)

        faiss.write_index(self.index, str(save_dir / "dense.faiss"))
        np.save(str(save_dir / "dense_article_ids.npy"), self.article_ids)
        logger.info("Dense index saved: %d vectors, dim=%d", self.index.ntotal, dim)

    @classmethod
    def load(cls, config: dict, save_dir: Path, model: object = None) -> "DenseRetriever":
        """Load the index files; ``model`` reuses an already-loaded embedder (hot reload)
        instead of instantiating a second SentenceTransformer."""
        save_dir = Path(save_dir)
        instance = cls.__new__(cls)
        instance.config = config
        model_name = config["retrieval"]["dense_model"]
        instance.model = model if model is not None else load_text_encoder(model_name)
        instance.index = faiss.read_index(str(save_dir / "dense.faiss"))
        instance.article_ids = np.load(str(save_dir / "dense_article_ids.npy"), allow_pickle=True)
        return instance

    def search_by_id(self, article_id: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return items similar to article_id using its stored FAISS embedding."""
        pos_arr = np.where(self.article_ids == article_id)[0]
        if len(pos_arr) == 0:
            return []
        pos = int(pos_arr[0])
        vec = self.index.reconstruct(pos).reshape(1, -1)
        scores, indices = self.index.search(vec, top_k + 1)  # +1 to skip seed itself
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                aid = str(self.article_ids[idx])
                if aid != article_id:
                    results.append((aid, float(score)))
        return results[:top_k]

    def search(
        self, query: str, top_k: int = 20, params: "faiss.SearchParameters | None" = None
    ) -> list[tuple[str, float]]:
        """`params` restricts the search to a subset of the index (e.g.
        `faiss.SearchParameters(sel=faiss.IDSelectorArray(allowed_positions))`) —
        used by HybridRetriever to push gender/colour/price filters into fetch
        instead of applying them after. IndexFlatIP already scores every vector
        in the index per call regardless of restriction (exact brute-force, no
        approximation structure to prune), so a restricted call costs the same
        as an unrestricted one — confirmed via eval/diagnose_pushdown_mechanism.py
        (30ms restricted vs 32ms unfiltered, full 112k-item index).
        """
        # Query-embedding memo: encoding is ~90 ms of a ~150 ms search, and one outfit turn
        # re-runs the same query text under different price caps (probe + variants, several
        # anchors). The vector is a pure function of the text (CPU, fixed model) and is never
        # mutated downstream, so the memo cannot change a result. Created lazily because
        # ``load`` builds the instance via ``cls.__new__`` (no __init__); FIFO-bounded.
        # Thread-safe: sync FastAPI endpoints run in a threadpool, so lookup, insert and FIFO
        # eviction happen under _QUERY_VEC_LOCK (the slow encode runs outside it).
        cache: dict[str, np.ndarray] = self.__dict__.setdefault("_query_vec_cache", {})
        with _QUERY_VEC_LOCK:
            query_vec = cache.get(query)
        if query_vec is None:
            with first_call("dense_encode"):
                query_vec = self.model.encode(
                    [query], normalize_embeddings=True, show_progress_bar=False
                ).astype(np.float32)
            with _QUERY_VEC_LOCK:
                if query not in cache and len(cache) >= _QUERY_VEC_CACHE_MAX:
                    cache.pop(next(iter(cache)))
                cache[query] = query_vec
        with first_call("faiss_search_masked" if params is not None else "faiss_search"):
            scores, indices = self.index.search(query_vec, top_k, params=params)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((str(self.article_ids[idx]), float(score)))
        return results
