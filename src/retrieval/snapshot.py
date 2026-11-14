"""Immutable index snapshot: everything a request needs from one index build.

Design (Wave 1b V3(b), hot reload)
----------------------------------
All state that depends on the loaded index lives in ONE frozen :class:`IndexSnapshot`, held in a
single reference (``api.deps._snapshot``). A request takes the reference once at the start of a
turn and uses it throughout, so a turn that began on build N finishes on build N even if build
N+1 is swapped in meanwhile; new turns get N+1. Swapping is a single attribute assignment
(atomic under the GIL); nothing is ever mutated in place.

The compiled LangGraph agents capture the retriever and the catalogue DataFrame in closures
(``build_graph`` also precomputes facet vocabularies from the DataFrame), so they belong to the
snapshot: they are compiled from the NEW retriever BEFORE the swap and swapped together with it.
The alternative, making the graph read the retriever through a getter, would touch ~30 closure
sites in a 4,000-line module and still leave the precomputed vocabularies stale.

Public view: the snapshot's DataFrame holds PUBLIC stores only (``public_catalogue``), so every
DataFrame-based path (item lookup, compare, outfit composition, brand/store counts, facet
vocabularies) is public-only by construction. The FAISS/BM25 indices still hold the hidden
stores' vectors, but ``HybridRetriever.search`` restricts both fetches to the rows of its own
(public) DataFrame, so hidden items cannot occupy fetch-window slots (no recall starvation);
over an unfiltered DataFrame (evals) the same method applies the registry mask itself.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.config.store_registry import include_non_public, public_stores

if TYPE_CHECKING:
    from src.retrieval.build_manifest import BuildManifest

logger = logging.getLogger(__name__)

LEGACY_BUILD_ID = "legacy"

ClipEntry = tuple[Any, np.ndarray]  # (faiss index, article_ids)


@dataclass(frozen=True, eq=False)
class IndexSnapshot:
    """One consistent view of the index. ``eq=False``: identity semantics (it holds DataFrames)."""

    retriever: Any
    catalogue_df: pd.DataFrame
    manifest: BuildManifest | None = None
    build_id: str = LEGACY_BUILD_ID
    clip: ClipEntry | None = None
    agent_sync: Any = None
    agent_streaming: Any = None
    source: str = "flat"  # "flat" (legacy layout) | "build" (builds/<id>/)
    loaded_at: float = field(default_factory=time.time)

    @functools.cached_property
    def public_article_ids(self) -> frozenset[str]:
        """Ids of the snapshot's (public) catalogue; the allow-list for the image-anchor path.

        Computed once on first use (a few ms for 90k ids) and stored in the instance dict, which
        works on a frozen dataclass because cached_property bypasses ``__setattr__``.
        """
        if "article_id" in self.catalogue_df.columns:
            ids = self.catalogue_df["article_id"]
        else:
            ids = self.catalogue_df.index.to_series()
        return frozenset(ids.astype(str))


def public_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``df`` that may reach users (store in the registry's public set).

    No-op when ``INCLUDE_NON_PUBLIC_STORES`` is set (local development and evals) or when the
    frame has no ``store`` column (legacy per-brand indices). A store missing from the registry
    is dropped (fail closed). Returns ``df`` itself when nothing is hidden (no copy).
    """
    if include_non_public() or "store" not in df.columns:
        return df
    keep = df["store"].fillna("").astype(str).str.lower().isin(public_stores())
    if bool(keep.all()):
        return df
    return df[keep].reset_index(drop=True)


def load_clip_entry(build_dir: Path) -> ClipEntry | None:
    """Load ``clip.faiss`` + ``clip_article_ids.npy`` from a build dir, or None when absent."""
    import faiss

    faiss_path = Path(build_dir) / "clip.faiss"
    ids_path = Path(build_dir) / "clip_article_ids.npy"
    if not (faiss_path.is_file() and ids_path.is_file()):
        return None
    return faiss.read_index(str(faiss_path)), np.load(str(ids_path), allow_pickle=True)


def load_index_parts(
    index_dir: Path,
    config: dict,
    *,
    model: Any = None,
) -> tuple[Any, pd.DataFrame]:
    """Load dense + BM25 + catalogue from ``index_dir``; return ``(retriever, public_df)``.

    Single loading path for the startup (flat or build directory) and for hot reload.
    ``model`` lets a reload reuse the already-loaded embedding model (MiniLM, ~90 MB) instead
    of instantiating a second copy. The returned retriever is built over the PUBLIC view.
    """
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    index_dir = Path(index_dir)
    df = public_catalogue(pd.read_parquet(index_dir / "catalogue.parquet"))
    dense = DenseRetriever.load(config, index_dir, model=model)
    sparse = SparseRetriever.load(config, index_dir)
    return HybridRetriever(dense, sparse, df, config), df
