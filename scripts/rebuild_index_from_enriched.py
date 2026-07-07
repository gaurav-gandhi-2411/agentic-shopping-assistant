"""Build script: rebuild BM25 + dense retrieval indices from the enriched catalogue.

Background
----------
scripts/enrich_attributes.py --full writes a COMPLETE enriched catalogue
(all rows, all original columns, with season/occasion_tag/style_tag/fabric
facets merged in and appended to ``search_text``) to
``data/processed/enrichment_full/catalogue_enriched.parquet``. That parquet
by itself is NOT enough for the new facet tags to actually be retrievable —
the live app never reads catalogue.parquet's ``search_text``/``facets``
columns at query time for scoring; it queries prebuilt BM25 and dense
(FAISS) indices (see src/retrieval/hybrid_search.py, sparse_search.py,
dense_search.py). Those indices were built from the OLD ``search_text`` and
must be rebuilt from the NEW (enriched) ``search_text`` before "summer
dress"/"boho"/"office" queries can benefit from the new facet tokens.

What needs rebuilding — BM25 vs dense vs both
----------------------------------------------
Both.

- **BM25 (must rebuild):** BM25 is an exact-token-overlap ranker; the new
  facet tokens (e.g. "boho", "office wear", "summer") are only literally
  searchable once they exist in the corpus BM25 was built from. BM25's IDF
  is also corpus-global (rank_bm25.BM25Okapi), so it is always REBUILT from
  the full corpus, never patched/concatenated — this matches
  scripts/build_unified_index.py::build_bm25_unified, which rebuilds BM25 on
  every unified-index build for the same reason (see that file's module
  docstring: "BM25: IDF is corpus-global so we REBUILD from the merged
  corpus").
- **Dense/FAISS (must rebuild too):** scripts/build_unified_index.py already
  established the precedent that changed ``search_text`` requires a full
  MiniLM re-embed (see build_dense_unified_reembed's docstring: Phase A
  cleaning changed search_text, so "concatenating the stale per-brand FAISS
  vectors... would leave dense retrieval blind to those fixes"). The facet
  enrichment changes search_text the same way (new tokens appended via
  append_enrichment_to_search_text), so the same reasoning applies: the old
  dense.faiss was embedded from pre-enrichment search_text and must be
  re-embedded. MiniLM (384-d) encodes ~62k short texts in well under a
  minute on CPU (local, free, deterministic — no network calls), so this is
  cheap.
- **CLIP (does NOT need touching):** CLIP embeds product IMAGES, not
  ``search_text`` — enrichment never touches images, and enrichment does not
  add/drop/reorder rows (article_id set is identical before/after), so the
  existing gs://.../clip/unified/{clip.faiss,clip_article_ids.npy} remain
  valid as-is. This script does not read, write, or copy any CLIP artifacts.

Output layout
-------------
Mirrors data/processed/unified/ exactly (the layout src/retrieval/index_store.py
resolves for BRAND=unified), but written to a STAGING directory so nothing
promotes to the live path automatically:

    data/processed/unified_enriched/dense.faiss
    data/processed/unified_enriched/dense_article_ids.npy
    data/processed/unified_enriched/bm25.pkl
    data/processed/unified_enriched/bm25_article_ids.npy
    data/processed/unified_enriched/catalogue.parquet

Promotion (NOT done by this script): once reviewed, these 5 files replace
the corresponding files under data/processed/unified/, and (per the
project's GCS-deploy convention) get uploaded to the INDEX_STORE_URI bucket
under unified/ before the next Cloud Run deploy — see
feedback_gcs_index_deploy.md: "always upload rebuilt index to GCS before
deploy."

This script does NOT:
    - touch data/processed/unified/ (the live-serving directory) — writes
      only under data/processed/unified_enriched/.
    - touch GCS / Cloud Run.
    - rebuild or copy CLIP artifacts (see above — unaffected by enrichment).
    - re-run the per-brand load/clean/merge pipeline in build_unified_index.py
      — it takes the already-merged, already-enriched catalogue parquet as
      its single input.

Usage
-----
    python scripts/rebuild_index_from_enriched.py                 # full rebuild
    python scripts/rebuild_index_from_enriched.py --dry-run        # counts only, no writes
    python scripts/rebuild_index_from_enriched.py \\
        --input data/processed/enrichment_full/catalogue_enriched.parquet \\
        --output-dir data/processed/unified_enriched
"""
from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Repo root on sys.path so ``src.*`` imports work when run as a script.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("rebuild_index_from_enriched")

_DEFAULT_INPUT = (
    _REPO_ROOT / "data" / "processed" / "enrichment_full" / "catalogue_enriched.parquet"
)
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "processed" / "unified_enriched"

# Must match src.retrieval.sparse_search.SparseRetriever._tokenize AND
# scripts/build_unified_index.py::_tokenize exactly, so a BM25 index built
# here loads and scores identically to one built by build_unified_index.py.
# Duplicated (not imported) deliberately: this script must stay runnable
# standing alone from just the enriched parquet, without importing
# build_unified_index.py's per-brand pipeline.
_DENSE_DIM = 384


def _tokenize(text: str) -> list[str]:
    """Tokenise *text* exactly as SparseRetriever._tokenize / build_unified_index._tokenize do."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# Core build functions
# ---------------------------------------------------------------------------


def build_bm25_from_enriched(df: pd.DataFrame, output_dir: Path) -> int:
    """Rebuild BM25 from the enriched catalogue's (facet-appended) search_text.

    Writes bm25.pkl + bm25_article_ids.npy to *output_dir*, in the exact
    format src/retrieval/sparse_search.py::SparseRetriever.load expects.
    Returns the document count.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = df["search_text"].fillna("").tolist()
    article_ids = df["article_id"].astype(str).values

    logger.info("Building BM25 over %d documents (enriched corpus)...", len(texts))
    t0 = time.perf_counter()
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    elapsed = time.perf_counter() - t0
    logger.info("BM25 built in %.1fs", elapsed)

    with open(output_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    np.save(str(output_dir / "bm25_article_ids.npy"), article_ids)
    logger.info("BM25 saved: %d documents -> %s", len(texts), output_dir / "bm25.pkl")
    return len(texts)


def build_dense_from_enriched(df: pd.DataFrame, config: dict, output_dir: Path) -> int:
    """Re-embed the enriched catalogue's search_text with MiniLM.

    Reuses DenseRetriever.build_index (src/retrieval/dense_search.py) so the
    output is byte-for-byte the same format as the live-serving
    data/processed/unified/dense.faiss + dense_article_ids.npy.
    Returns the total vector count.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dupes = len(df["article_id"]) - df["article_id"].nunique()
    if dupes:
        raise ValueError(f"{dupes} duplicate article_ids in enriched catalogue. Cannot continue.")

    dense = DenseRetriever(config)
    dense.build_index(df, output_dir)

    total = dense.index.ntotal
    logger.info("Dense (re-embedded from enriched search_text): %d vectors (dim=%d)", total, _DENSE_DIM)
    return total


def _verify_alignment(label: str, expected_total: int, ids_path: Path) -> None:
    """Assert that the saved ids array and (if present) FAISS index both have length *expected_total*."""
    ids = np.load(str(ids_path), allow_pickle=True)
    assert len(ids) == expected_total, f"{label}: ids length {len(ids)} != expected {expected_total}"
    stem = ids_path.name.replace("_article_ids.npy", "")
    faiss_path = ids_path.parent / f"{stem}.faiss"
    if faiss_path.exists():
        idx = faiss.read_index(str(faiss_path))
        assert idx.ntotal == expected_total, f"{label}: faiss ntotal {idx.ntotal} != expected {expected_total}"


def _report_sizes(output_dir: Path) -> None:
    """Print file sizes for all output files.

    Displays paths relative to the repo root when *output_dir* is inside it
    (the common case); falls back to the absolute path otherwise (e.g. a
    custom --output-dir outside the repo, as used in ad-hoc/staging runs).
    """
    print("\n=== Output file sizes ===")
    if output_dir.exists():
        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / 1024 / 1024
                try:
                    label = str(f.relative_to(_REPO_ROOT))
                except ValueError:
                    label = str(f)
                print(f"  {label:<55}  {size_mb:>8.2f} MB")
    print("=========================\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild BM25 + dense (FAISS) indices from the enriched catalogue parquet, "
            "into a staging directory. Does NOT touch data/processed/unified/ or GCS."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(_DEFAULT_INPUT),
        help="Path to the enriched catalogue parquet (default: enrichment_full/catalogue_enriched.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Staging output directory (default: data/processed/unified_enriched/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print row/column counts only; do not build or write any index files.",
    )
    return parser.parse_args()


def main() -> None:
    """Rebuild BM25 + dense indices from the enriched catalogue into a staging directory.

    Steps:
    1. Load the enriched catalogue parquet (all rows, updated facets/search_text).
    2. Rebuild BM25 over the full enriched corpus (IDF is corpus-global — always rebuilt).
    3. Re-embed dense FAISS from the enriched search_text (MiniLM, CPU, ~1 min for ~62k rows).
    4. Write catalogue.parquet (a copy of the enriched input) alongside the rebuilt indices.
    5. Verify alignment (FAISS ntotal == ids array length == catalogue row count).
    6. Print a file-size report.

    CLIP is intentionally NOT touched — see module docstring.
    """
    args = _parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print("\n=== Rebuild indices from enriched catalogue ===")
    print(f"Input:      {input_path}")
    print(f"Output dir: {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Enriched catalogue not found: {input_path}. "
            "Run scripts/enrich_attributes.py --full first."
        )

    df = pd.read_parquet(input_path)
    n = len(df)
    print(f"Loaded enriched catalogue: {n:,} rows, {len(df.columns)} columns")

    required_cols = {"article_id", "search_text", "facets"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Enriched catalogue is missing required column(s): {sorted(missing)}")

    if args.dry_run:
        print("\n=== DRY RUN — no files written ===")
        print(f"  rows:    {n:,}")
        print(f"  columns: {list(df.columns)}")
        dupes = n - df["article_id"].nunique()
        print(f"  duplicate article_ids: {dupes}")
        print("===================================\n")
        return

    config = load_config()
    t_start = time.perf_counter()

    print("\n[1/3] Rebuilding BM25...")
    bm25_total = build_bm25_from_enriched(df, output_dir)

    print("\n[2/3] Re-embedding dense FAISS...")
    dense_total = build_dense_from_enriched(df, config, output_dir)

    print("\n[3/3] Writing catalogue.parquet...")
    df.to_parquet(str(output_dir / "catalogue.parquet"), index=False)
    logger.info("Catalogue written: %d rows -> %s", n, output_dir / "catalogue.parquet")

    print("\n=== Verifying alignment ===")
    _verify_alignment("dense", dense_total, output_dir / "dense_article_ids.npy")
    _verify_alignment("BM25", bm25_total, output_dir / "bm25_article_ids.npy")
    assert dense_total == n, f"Dense total {dense_total} != catalogue rows {n}"
    assert bm25_total == n, f"BM25 total {bm25_total} != catalogue rows {n}"
    print("  All alignment checks passed.")

    elapsed = time.perf_counter() - t_start
    print(f"\n=== Rebuild complete ({elapsed:.1f}s) ===")
    print(f"  Rows:   {n:,}")
    print(f"  Dense:  {dense_total:,} vectors (dim={_DENSE_DIM})")
    print(f"  BM25:   {bm25_total:,} documents")
    print("  CLIP:   untouched (image embeddings unaffected by text enrichment — see module docstring)")

    _report_sizes(output_dir)

    print(
        "Staged only — nothing was written to data/processed/unified/ or uploaded to GCS.\n"
        "Review the staged files, then (separately, manually) promote by copying these 5\n"
        "files over data/processed/unified/ and re-uploading unified/ to the INDEX_STORE_URI\n"
        "bucket before the next deploy."
    )


if __name__ == "__main__":
    main()
