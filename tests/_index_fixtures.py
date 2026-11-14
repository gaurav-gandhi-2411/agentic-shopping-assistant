"""Synthetic, network-free index builds for the V3(b) tests (registry filter, manifest, reload).

Real FAISS + real BM25 over a tiny catalogue, with a deterministic hashing "embedding model"
standing in for MiniLM, so tests exercise the real HybridRetriever / loader code paths.
Everything is written under a caller-supplied ``tmp_path`` (hermetic by construction).
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np
import pandas as pd

from src.retrieval.dense_search import DenseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.snapshot import IndexSnapshot, load_index_parts, public_catalogue
from src.retrieval.sparse_search import SparseRetriever

if TYPE_CHECKING:
    from src.retrieval.build_manifest import BuildManifest

DIM = 16
CONFIG: dict[str, Any] = {
    "retrieval": {
        "dense_model": "fake",
        "dense_batch_size": 8,
        "final_k": 10,
        "top_k": 10,
        "rrf_k": 60,
        "store_diversity": 0.0,
    }
}


class FakeModel:
    """Deterministic bag-of-words hashing embedder (stands in for SentenceTransformer)."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], **_: Any) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
            norm = float(np.linalg.norm(out[i])) or 1.0
            out[i] /= norm
        return out


def make_rows(stores: dict[str, int], *, noun: str = "kurta") -> pd.DataFrame:
    """Catalogue rows: ``stores`` maps slug -> item count. Every row mentions ``noun``."""
    rows: list[dict[str, Any]] = []
    for slug, n in stores.items():
        for i in range(n):
            aid = f"{slug}-{i}"
            rows.append(
                {
                    "article_id": aid,
                    "store": slug,
                    "prod_name": f"{noun} {slug} {i}",
                    "display_name": f"{noun} {slug} {i}",
                    "detail_desc": f"a {noun} from {slug}",
                    "search_text": f"{noun} {slug} item{i}",
                    "image_url": None,
                    "price_inr": 500.0 + i,
                    "pdp_handle": f"{slug}-{i}",
                    "pdp_live": True,
                    "gender": "women",
                    "product_type_name": noun,
                    "colour_group_name": "Black",
                    "facets": {
                        "colour_group_name": "Black",
                        "product_type_name": noun,
                        "department_name": "Ethnic",
                    },
                }
            )
    return pd.DataFrame(rows)


def write_index_files(
    df: pd.DataFrame, out: Path, *, dim: int = DIM, with_clip: bool = False
) -> None:
    """Write dense.faiss, bm25.pkl, the two id files and catalogue.parquet (+ optional CLIP)."""
    out.mkdir(parents=True, exist_ok=True)
    dense = DenseRetriever.__new__(DenseRetriever)
    dense.config = CONFIG
    dense.model = FakeModel(dim)
    dense.build_index(df, out)
    SparseRetriever(CONFIG).build_index(df, out)
    df.to_parquet(out / "catalogue.parquet")
    if with_clip:
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(df), 8)).astype(np.float32)
        idx = faiss.IndexFlatIP(8)
        idx.add(vecs)
        faiss.write_index(idx, str(out / "clip.faiss"))
        np.save(str(out / "clip_article_ids.npy"), df["article_id"].values.astype(str))


def write_build(
    store_root: Path,
    build_id: str,
    stores: dict[str, int],
    *,
    dim: int = DIM,
    with_clip: bool = False,
    previous: BuildManifest | None = None,
    publish_pointer: bool = True,
    built_at: datetime | None = None,
) -> BuildManifest:
    """Write ``builds/<id>/`` (files + manifest) and, by default, ``pointers/<id>.json``."""
    from src.retrieval.build_manifest import (  # lazy: keeps this module importable standalone
        StoreEntry,
        build_manifest_for_dir,
        write_manifest,
    )

    df = make_rows(stores)
    bdir = store_root / "builds" / build_id
    write_index_files(df, bdir, dim=dim, with_clip=with_clip)
    counts = df["store"].value_counts().to_dict()
    manifest = build_manifest_for_dir(
        bdir,
        build_id=build_id,
        git_sha="deadbeef",
        stores=[StoreEntry(slug=s, items=int(n)) for s, n in counts.items()],
        built_at=built_at or datetime.now(timezone.utc),
        dense_dim=dim,
        clip_dim=8 if with_clip else None,
        previous=previous,
    )
    write_manifest(bdir, manifest)
    if publish_pointer:
        publish(store_root, build_id)
    return manifest


def publish(store_root: Path, build_id: str) -> None:
    """Create ``pointers/<id>.json`` naming ``build_id``."""
    d = store_root / "pointers"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{build_id}.json").write_text(f'{{"build_id": "{build_id}"}}', encoding="utf-8")


def make_snapshot(
    index_dir: Path,
    *,
    manifest: BuildManifest | None = None,
    model: Any = None,
    source: str = "build",
) -> IndexSnapshot:
    """Load an index directory into an IndexSnapshot (no agent graphs: not needed in tests)."""
    retriever, df = load_index_parts(index_dir, CONFIG, model=model or FakeModel())
    from src.retrieval.snapshot import load_clip_entry

    return IndexSnapshot(
        retriever=retriever,
        catalogue_df=df,
        manifest=manifest,
        build_id=manifest.build_id if manifest else "legacy",
        clip=load_clip_entry(index_dir),
        source=source,
    )


def load_build_fn(
    build_dir: Path, manifest: BuildManifest, current: IndexSnapshot | None
) -> IndexSnapshot:
    """``LoadBuild`` implementation for tests: reuses the current snapshot's model like prod."""
    model = current.retriever.dense.model if current is not None else FakeModel()
    return make_snapshot(build_dir, manifest=manifest, model=model)


__all__ = [
    "CONFIG",
    "DIM",
    "FakeModel",
    "HybridRetriever",
    "load_build_fn",
    "make_rows",
    "make_snapshot",
    "public_catalogue",
    "publish",
    "write_build",
    "write_index_files",
]
