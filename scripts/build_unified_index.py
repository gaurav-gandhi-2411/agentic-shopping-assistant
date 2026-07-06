"""Build script: merge all live per-brand indices into a single cross-store unified index.

Outputs
-------
    data/processed/unified/dense.faiss
    data/processed/unified/dense_article_ids.npy
    data/processed/unified/bm25.pkl
    data/processed/unified/bm25_article_ids.npy
    data/processed/unified/catalogue.parquet
    data/processed/clip/unified/clip.faiss
    data/processed/clip/unified/clip_article_ids.npy

Design decisions
----------------
- article_id is globally unique across all brands (zero cross-brand collisions verified).
  Intra-brand duplicates exist in the Myntra catalogue (106 duplicate/junk rows); these are
  deduped at load time.  Raw article_ids are kept unchanged — no store:: namespacing needed.
- Phase A index-quality cleaning (2026-07-06, src/catalogue/cleaning.py) runs per-brand
  BEFORE the merge: saree reclassification, true-fabric-bolt row drop, colour backfill,
  mojibake cleanup, then search_text/display_name/facets are recomputed from the fixed
  columns. See _clean_brand_catalogue.
- Dense FAISS: RE-EMBEDDED from the cleaned+merged catalogue's search_text (not
  concatenated from stale per-brand vectors) — colour backfill and mojibake cleanup change
  search_text, so the old "concatenate, no re-embedding needed" design no longer holds.
  MiniLM (384-d) encodes ~68k rows in well under a minute on CPU (local, free, deterministic
  inference — no network calls), so re-embedding the whole merged corpus is cheap.
- CLIP FAISS: image embeddings are unaffected by the text cleaning above, so CLIP vectors
  are still concatenated from per-brand IndexFlatIP (via faiss.get_xb()) and filtered to the
  surviving (post-drop) id set — no re-embedding.
- BM25: IDF is corpus-global so we REBUILD from the merged corpus (NOT concatenate pickles).
  Tokenisation matches SparseRetriever._tokenize exactly.
- Catalogue: union of all cleaned per-brand frames; `store` column added from the brand slug
  where missing; pdp_live preserved where present, left absent (NaN) otherwise.
  HybridRetriever treats NaN pdp_live as "unknown" — not moved to the dead pile.
- berrylush is excluded entirely at build time (store inactive; see EXCLUDED_STORES) —
  re-enabling it requires re-running this build script, not just flipping its
  STORE_CONFIG "active" flag (src/config/stores.py).
- Deterministic: numpy seed=42 (no shuffling; guard on any future stochastic steps).
- Idempotent: re-running overwrites outputs safely.

Usage
-----
    python scripts/build_unified_index.py                    # build all live brands
    python scripts/build_unified_index.py --dry-run          # print counts only
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

from src.catalogue.cleaning import (  # noqa: E402
    backfill_colours,
    clean_mojibake_columns,
    drop_true_fabric_material,
    reclassify_finished_sarees,
    recompute_derived_columns,
)
from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("build_unified_index")

# Seed — no randomness in core path; guard for reproducibility of any future stochastic steps.
_RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Brand → index directory layout
# ---------------------------------------------------------------------------
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_CLIP_DIR = _DATA_DIR / "clip"
_UNIFIED_DIR = _DATA_DIR / "unified"
_CLIP_UNIFIED_DIR = _CLIP_DIR / "unified"

# hm: archival Kaggle data, no live PDP/image; requeue for partner-API phase.
# berrylush: store inactive (password-walled, src/config/stores.py) — dropped entirely
# at build time (2026-07-06 Phase A) rather than merely filtered at query time, so it
# stops occupying FAISS/BM25 candidate-window slots. Re-enabling requires a rebuild.
EXCLUDED_STORES: frozenset[str] = frozenset({"hm", "berrylush"})

# Canonical set of live stores included in the unified index (all have working deep-links + images).
UNIFIED_STORES: tuple[str, ...] = (
    "myntra", "flipkart", "snitch", "fashor", "powerlook", "virgio",
    "globalrepublic", "libas",
)

# Directory layout: all brands live in data/processed/<brand>/
_BRAND_DIRS: dict[str, Path] = {brand: _DATA_DIR / brand for brand in UNIFIED_STORES}

_CLIP_BRAND_DIRS: dict[str, Path] = {
    brand: _CLIP_DIR / brand for brand in UNIFIED_STORES
}

# Must match what 01_build_retrieval.py / build_clip_index.py embedded with.
_DENSE_DIM = 384
_CLIP_DIM = 512


# ---------------------------------------------------------------------------
# BM25 tokeniser — must be identical to SparseRetriever._tokenize
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Tokenise *text* exactly as SparseRetriever._tokenize does."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# Catalogue loading with dedup
# ---------------------------------------------------------------------------

def _load_clean_catalogue(brand: str) -> pd.DataFrame:
    """Load and clean the catalogue parquet for *brand*.

    Cleaning steps (applied before merging):
    1. Drop rows with null/empty/sentinel-zero article_id (corrupted Myntra rows).
    2. Deduplicate by article_id within the brand, keeping the first occurrence.
       (The Myntra feed contains 106 duplicate rows — identical content, safe to drop.)
    3. Inject `store` column from brand slug if absent.

    Returns a clean DataFrame aligned with the brand's dense index positions.
    """
    brand_dir = _BRAND_DIRS[brand]
    path = brand_dir / "catalogue.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Catalogue not found: {path}")

    df = pd.read_parquet(path)
    before = len(df)

    # Drop corrupted rows (article_id is null, empty string, or the sentinel "0")
    df = df[df["article_id"].notna()].copy()
    df["article_id"] = df["article_id"].astype(str).str.strip()
    df = df[(df["article_id"] != "") & (df["article_id"] != "0")]

    # Deduplicate within brand — keep first, so positional alignment with FAISS is preserved
    df = df.drop_duplicates(subset="article_id", keep="first").reset_index(drop=True)

    after = len(df)
    if after < before:
        logger.info(
            "brand=%-12s: dropped %d duplicate/junk rows (%d → %d)",
            brand,
            before - after,
            before,
            after,
        )

    if "store" not in df.columns:
        df["store"] = brand

    return df


# ---------------------------------------------------------------------------
# Phase A index-quality cleaning (2026-07-06) — applied per brand, before merge.
# See src/catalogue/cleaning.py for the underlying rule-based functions.
# ---------------------------------------------------------------------------


def _clean_brand_catalogue(brand: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply Phase A cleaning to a single brand's (deduped) catalogue DataFrame.

    Order matters:
        1. Mojibake cleanup on prod_name/detail_desc (feeds the saree-name regex
           and colour extraction below with clean text).
        2. Saree reclassification (fabric_material -> saree for finished sarees).
        3. Drop remaining true fabric_material rows (fabric bolts).
        4. Colour backfill for null/empty colour_group_name.
        5. Recompute search_text/display_name/facets from the fixed columns.

    Returns (df, stats) where stats records per-step counts for the build report.
    """
    df, mojibake_stats = clean_mojibake_columns(df)
    df, n_saree_reclassified = reclassify_finished_sarees(df)
    df, n_fabric_dropped = drop_true_fabric_material(df)
    df, n_colour_filled = backfill_colours(df)
    df = recompute_derived_columns(df)

    stats = {
        "brand": brand,
        "mojibake_prod_name": mojibake_stats.get("prod_name", 0),
        "mojibake_detail_desc": mojibake_stats.get("detail_desc", 0),
        "saree_reclassified": n_saree_reclassified,
        "fabric_material_dropped": n_fabric_dropped,
        "colour_backfilled": n_colour_filled,
        "rows_after_cleaning": len(df),
    }
    return df, stats


def _load_and_clean_brand(brand: str) -> tuple[pd.DataFrame, dict]:
    """Load, dedup, and apply Phase A cleaning to a single brand's catalogue."""
    df = _load_clean_catalogue(brand)
    return _clean_brand_catalogue(brand, df)


def _print_cleaning_report(all_stats: list[dict]) -> None:
    """Print a per-brand Phase A cleaning report."""
    print("\n=== Phase A cleaning report ===")
    header = (
        f"  {'brand':<14} {'mojibake(name/desc)':<20} {'saree_reclass':>13} "
        f"{'fabric_dropped':>14} {'colour_filled':>13} {'rows_after':>10}"
    )
    print(header)
    totals = {
        "mojibake_prod_name": 0, "mojibake_detail_desc": 0, "saree_reclassified": 0,
        "fabric_material_dropped": 0, "colour_backfilled": 0, "rows_after_cleaning": 0,
    }
    for s in all_stats:
        mojibake = f"{s['mojibake_prod_name']}/{s['mojibake_detail_desc']}"
        print(
            f"  {s['brand']:<14} {mojibake:<20} {s['saree_reclassified']:>13} "
            f"{s['fabric_material_dropped']:>14} {s['colour_backfilled']:>13} "
            f"{s['rows_after_cleaning']:>10}"
        )
        for k in totals:
            totals[k] += s[k]
    total_mojibake = f"{totals['mojibake_prod_name']}/{totals['mojibake_detail_desc']}"
    print(
        f"  {'TOTAL':<14} {total_mojibake:<20} "
        f"{totals['saree_reclassified']:>13} {totals['fabric_material_dropped']:>14} "
        f"{totals['colour_backfilled']:>13} {totals['rows_after_cleaning']:>10}"
    )
    print("================================\n")


# ---------------------------------------------------------------------------
# FAISS vector extraction helper
# ---------------------------------------------------------------------------

def _extract_vectors_for_ids(
    index: faiss.IndexFlatIP,
    index_ids: np.ndarray,
    keep_ids: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract rows from *index* whose ids are in *keep_ids*.

    Args:
        index:     FAISS IndexFlatIP to read vectors from.
        index_ids: article_id array aligned with *index* (same order).
        keep_ids:  Set of article_ids to keep (after dedup).

    Returns:
        (vecs, ids) — float32 array (M, D) and 1-D str array of length M,
        in the original positional order of the FAISS index.
    """
    n = index.ntotal
    if n == 0:
        return np.zeros((0, index.d), dtype=np.float32), np.array([], dtype=str)

    # Read all vectors at once — IndexFlatIP stores them in a flat array
    all_vecs: np.ndarray = faiss.rev_swig_ptr(index.get_xb(), n * index.d).reshape(n, index.d).copy()

    # Build a mask; we also track which ids we've seen to avoid taking the same
    # id twice even if the FAISS index itself contains duplicate positions.
    mask: list[bool] = []
    seen: set[str] = set()
    for aid in index_ids.astype(str):
        if aid in keep_ids and aid not in seen:
            mask.append(True)
            seen.add(aid)
        else:
            mask.append(False)

    mask_arr = np.array(mask, dtype=bool)
    return all_vecs[mask_arr].astype(np.float32), index_ids.astype(str)[mask_arr]


# ---------------------------------------------------------------------------
# Core merge functions
# ---------------------------------------------------------------------------

def build_catalogue_unified(
    brands: list[str], clean_catalogues: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Union the (already loaded, deduped, Phase-A-cleaned) brand catalogues into one.

    Returns the merged DataFrame (also written to _UNIFIED_DIR/catalogue.parquet).
    """
    _UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    per_brand_counts: dict[str, int] = {}

    for brand in brands:
        df = clean_catalogues[brand]
        per_brand_counts[brand] = len(df)
        frames.append(df)
        logger.info("  catalogue brand=%-12s  rows=%d", brand, len(df))

    df_merged = pd.concat(frames, ignore_index=True)

    # Final cross-brand uniqueness check (should be guaranteed at this point)
    dupes = df_merged["article_id"].duplicated().sum()
    if dupes:
        dup_ids = df_merged[df_merged["article_id"].duplicated(keep=False)]["article_id"].unique()
        raise ValueError(
            f"{dupes} cross-brand duplicate article_ids remain after per-brand dedup. "
            f"Sample: {dup_ids[:5].tolist()}. "
            "article_ids are not globally unique — store:: namespacing is required."
        )

    df_merged.to_parquet(str(_UNIFIED_DIR / "catalogue.parquet"), index=False)
    logger.info("Catalogue unified: %d total rows", len(df_merged))
    for brand, count in per_brand_counts.items():
        logger.info("  %-12s %d", brand, count)
    return df_merged


def build_dense_unified_reembed(df_merged: pd.DataFrame, config: dict) -> int:
    """Re-embed the merged, Phase-A-cleaned catalogue's search_text with MiniLM.

    Phase A cleaning (colour backfill, mojibake cleanup, saree reclassification)
    changes search_text for many rows, so concatenating the stale per-brand FAISS
    vectors (the pre-Phase-A design) would leave dense retrieval blind to those
    fixes. MiniLM is local/free/deterministic and encodes ~70k short texts in well
    under a minute on CPU, so a full re-embed is cheap and correct.

    Returns total vector count. Writes to _UNIFIED_DIR/dense.faiss and
    _UNIFIED_DIR/dense_article_ids.npy via DenseRetriever.build_index.
    """
    _UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

    dupes = len(df_merged["article_id"]) - df_merged["article_id"].nunique()
    if dupes:
        raise ValueError(f"{dupes} duplicate article_ids in merged catalogue. Cannot continue.")

    dense = DenseRetriever(config)
    dense.build_index(df_merged, _UNIFIED_DIR)

    total = dense.index.ntotal
    logger.info("Dense unified (re-embedded): %d total vectors (dim=%d)", total, _DENSE_DIM)
    return total


def build_clip_unified(brands: list[str], clean_catalogues: dict[str, pd.DataFrame]) -> int:
    """Concatenate CLIP FAISS indices for *brands* into a single IndexFlatIP.

    Uses *clean_catalogues* to filter to the deduped id set per brand.

    Returns total vector count.
    Writes to _CLIP_UNIFIED_DIR/clip.faiss and _CLIP_UNIFIED_DIR/clip_article_ids.npy.
    """
    _CLIP_UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

    all_vecs: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    per_brand_counts: dict[str, int] = {}

    for brand in brands:
        clip_dir = _CLIP_BRAND_DIRS[brand]
        faiss_path = clip_dir / "clip.faiss"
        ids_path = clip_dir / "clip_article_ids.npy"

        if not faiss_path.exists():
            logger.warning("SKIP clip brand=%s: %s not found", brand, faiss_path)
            continue

        idx = faiss.read_index(str(faiss_path))
        if idx.d != _CLIP_DIM:
            raise ValueError(
                f"brand={brand} CLIP dim={idx.d} != expected {_CLIP_DIM}. "
                "All brands must share the same CLIP model."
            )

        raw_ids = np.load(str(ids_path), allow_pickle=True).astype(str)
        if idx.ntotal != len(raw_ids):
            raise ValueError(
                f"brand={brand}: clip ntotal={idx.ntotal} != ids len={len(raw_ids)}"
            )

        clean_ids = set(clean_catalogues[brand]["article_id"].tolist())
        vecs, ids = _extract_vectors_for_ids(idx, raw_ids, clean_ids)

        all_vecs.append(vecs)
        all_ids.append(ids)
        per_brand_counts[brand] = len(ids)
        logger.info("  clip  brand=%-12s  vectors=%d", brand, len(ids))

    merged_vecs = np.vstack(all_vecs).astype(np.float32)
    merged_ids = np.concatenate(all_ids).astype(str)

    unified_index = faiss.IndexFlatIP(_CLIP_DIM)
    unified_index.add(merged_vecs)

    faiss.write_index(unified_index, str(_CLIP_UNIFIED_DIR / "clip.faiss"))
    np.save(str(_CLIP_UNIFIED_DIR / "clip_article_ids.npy"), merged_ids)

    total = unified_index.ntotal
    logger.info("CLIP  unified: %d total vectors (dim=%d)", total, _CLIP_DIM)
    for brand, count in per_brand_counts.items():
        logger.info("  %-12s %d", brand, count)
    return total


def build_bm25_unified(df_merged: pd.DataFrame) -> None:
    """Rebuild BM25 from the merged corpus and save to _UNIFIED_DIR.

    BM25 IDF is corpus-global so we MUST rebuild (NOT concatenate pickles).
    Tokenisation matches SparseRetriever._tokenize exactly.
    """
    _UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

    texts = df_merged["search_text"].fillna("").tolist()
    article_ids = df_merged["article_id"].astype(str).values

    logger.info("Building BM25 over %d documents (merged corpus)...", len(texts))
    t0 = time.perf_counter()
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    elapsed = time.perf_counter() - t0
    logger.info("BM25 built in %.1fs", elapsed)

    with open(_UNIFIED_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    np.save(str(_UNIFIED_DIR / "bm25_article_ids.npy"), article_ids)
    logger.info("BM25 saved: %d documents", len(texts))


# ---------------------------------------------------------------------------
# Alignment verification
# ---------------------------------------------------------------------------

def _verify_alignment(label: str, expected_total: int, ids_path: Path) -> None:
    """Assert that the saved ids array and FAISS index both have length *expected_total*."""
    ids = np.load(str(ids_path), allow_pickle=True)
    assert len(ids) == expected_total, (
        f"{label}: ids length {len(ids)} != expected {expected_total}"
    )
    # Also check the FAISS file if it exists alongside the ids
    stem = ids_path.name.replace("_article_ids.npy", "")
    faiss_path = ids_path.parent / f"{stem}.faiss"
    if faiss_path.exists():
        idx = faiss.read_index(str(faiss_path))
        assert idx.ntotal == expected_total, (
            f"{label}: faiss ntotal {idx.ntotal} != expected {expected_total}"
        )


# ---------------------------------------------------------------------------
# File-size report
# ---------------------------------------------------------------------------

def _report_sizes(output_dirs: list[Path]) -> None:
    """Print file sizes for all output files."""
    print("\n=== Output file sizes ===")
    for d in output_dirs:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  {str(f.relative_to(_REPO_ROOT)):<55}  {size_mb:>8.2f} MB")
    print("=========================\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge live per-brand indices into a single unified cross-store index.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-brand counts only; do not write any files.",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip CLIP index concatenation and alignment check (use when new stores "
             "have no CLIP index yet).",
    )
    return parser.parse_args()


def main() -> None:
    """Build the unified cross-store index from the live stores (UNIFIED_STORES).

    H&M and berrylush are excluded (see EXCLUDED_STORES).

    Steps:
    1. Load, dedup, and Phase-A-clean per-brand catalogues (saree reclassification,
       fabric-bolt row drop, colour backfill, mojibake cleanup, derived-column recompute).
    2. Union catalogues → data/processed/unified/catalogue.parquet.
    3. Re-embed dense FAISS from the cleaned merged catalogue → dense.faiss + ids.
    4. Concatenate CLIP FAISS → data/processed/clip/unified/clip.faiss + ids.
    5. Rebuild BM25 over merged corpus → data/processed/unified/bm25.pkl + ids.
    6. Verify alignment (FAISS ntotal == ids array length for all outputs).
    7. Print the Phase A cleaning report, per-store + total counts, and file sizes.
    """
    args = _parse_args()
    config = load_config()

    brands = list(UNIFIED_STORES)  # canonical order; EXCLUDED_STORES (hm, berrylush) are omitted

    if args.dry_run:
        print("\n=== DRY RUN — no files written ===")
        total = 0
        all_stats: list[dict] = []
        for brand in brands:
            d = _BRAND_DIRS[brand]
            p = d / "catalogue.parquet"
            if p.exists():
                df, stats = _load_and_clean_brand(brand)
                n = len(df)
                print(f"  {brand:<12}  {n:>6} rows")
                total += n
                all_stats.append(stats)
        print(f"  {'TOTAL':<12}  {total:>6} rows")
        print("===================================\n")
        _print_cleaning_report(all_stats)
        return

    print("\n=== Building unified cross-store index ===")
    t_start = time.perf_counter()

    # Step 1 — load, dedup, and Phase-A-clean per-brand catalogues
    print("\n[1/5] Loading and cleaning per-brand catalogues...")
    clean_catalogues: dict[str, pd.DataFrame] = {}
    all_stats: list[dict] = []
    for brand in brands:
        df, stats = _load_and_clean_brand(brand)
        clean_catalogues[brand] = df
        all_stats.append(stats)

    # Step 2 — catalogue union
    print("\n[2/5] Building unified catalogue...")
    df_merged = build_catalogue_unified(brands, clean_catalogues)
    cat_total = len(df_merged)

    # Step 3 — dense FAISS (re-embedded from the cleaned merged catalogue)
    print("\n[3/5] Re-embedding dense FAISS index...")
    dense_total = build_dense_unified_reembed(df_merged, config)

    # Step 4 — CLIP FAISS (skippable when new stores have no CLIP index)
    print("\n[4/5] Concatenating CLIP FAISS indices...")
    if args.skip_clip:
        clip_total = -1
        print("  --skip-clip: CLIP concatenation skipped.")
    else:
        clip_total = build_clip_unified(brands, clean_catalogues)

    # Step 5 — BM25 (rebuild from merged corpus)
    print("\n[5/5] Rebuilding BM25 over merged corpus...")
    build_bm25_unified(df_merged)

    # Verification
    print("\n=== Verifying alignment ===")
    _verify_alignment("dense", dense_total, _UNIFIED_DIR / "dense_article_ids.npy")
    _verify_alignment("BM25", cat_total, _UNIFIED_DIR / "bm25_article_ids.npy")
    assert dense_total == cat_total, (
        f"Dense total {dense_total} != catalogue total {cat_total}"
    )
    if not args.skip_clip:
        _verify_alignment("CLIP", clip_total, _CLIP_UNIFIED_DIR / "clip_article_ids.npy")
        assert clip_total == cat_total, (
            f"CLIP total {clip_total} != catalogue total {cat_total}"
        )
    print("  All alignment checks passed.")

    # Summary
    elapsed = time.perf_counter() - t_start
    print(f"\n=== Unified index build complete ({elapsed:.1f}s) ===")
    print(f"  Total items across all stores: {cat_total:,}")
    print(f"  Dense vectors: {dense_total:,}  (dim={_DENSE_DIM})")
    if clip_total >= 0:
        print(f"  CLIP  vectors: {clip_total:,}  (dim={_CLIP_DIM})")
    else:
        print("  CLIP  vectors: skipped (--skip-clip)")
    print(f"  BM25  docs:    {cat_total:,}")

    # Per-store summary from merged catalogue
    print("\n  Per-store row counts (from unified catalogue):")
    for store, count in df_merged["store"].value_counts().items():
        print(f"    {store:<14}  {count:>6}")

    _print_cleaning_report(all_stats)
    _report_sizes([_UNIFIED_DIR, _CLIP_UNIFIED_DIR])


if __name__ == "__main__":
    main()
