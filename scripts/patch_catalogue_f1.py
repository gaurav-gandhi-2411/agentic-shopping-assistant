"""
F1 normalization patch — apply garment-type normalization to all per-brand catalogues.

Reads each brand's existing catalogue.parquet, runs the F1 normalizer on every row,
overwrites product_type_name with the canonical garment_type (for high/medium confidence),
adds a type_confidence column, recomputes search_text/display_name/facets, and saves back.
Then rebuilds the BM25 index for each brand (FAISS dense vectors are preserved as-is).

Run BEFORE build_unified_index.py so the unified merge picks up the corrected values.

Usage:
    python scripts/patch_catalogue_f1.py
    python scripts/patch_catalogue_f1.py --dry-run   # print change counts, no writes
"""
from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.catalogue.normalizer import normalize_garment_type  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("patch_catalogue_f1")

# ---------------------------------------------------------------------------
# Stores to patch (same as UNIFIED_STORES in build_unified_index.py)
# ---------------------------------------------------------------------------
STORES: tuple[str, ...] = ("myntra", "flipkart", "snitch", "fashor", "powerlook", "virgio")
_DATA_DIR = _REPO_ROOT / "data" / "processed"


def _tokenize(text: str) -> list[str]:
    """BM25 tokeniser — identical to SparseRetriever._tokenize."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


def _rebuild_search_text(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute search_text, display_name, and facets columns from current product_type_name."""
    df = df.copy()

    df["search_text"] = (
        df["prod_name"].fillna("") + ". "
        + df["product_type_name"].fillna("") + ". "
        + df["colour_group_name"].fillna("") + ". "
        + df["department_name"].fillna("") + ". "
        + df["detail_desc"].fillna("")
    )

    df["display_name"] = (
        df["prod_name"].fillna("").str.strip()
        + " ("
        + df["colour_group_name"].fillna("").str.strip()
        + " "
        + df["product_type_name"].fillna("").str.strip()
        + ")"
    )

    df["facets"] = df.apply(
        lambda r: {
            "colour_group_name": r["colour_group_name"],
            "product_type_name": r["product_type_name"],
            "department_name": r["department_name"],
            "index_group_name": r["index_group_name"],
            "garment_group_name": r["garment_group_name"],
        },
        axis=1,
    )

    return df


def patch_brand(brand: str, *, dry_run: bool = False) -> dict:
    """Patch the catalogue for one brand.

    Returns a summary dict with counts: total, changed, by_confidence.
    """
    brand_dir = _DATA_DIR / brand
    cat_path = brand_dir / "catalogue.parquet"
    bm25_path = brand_dir / "bm25.pkl"
    bm25_ids_path = brand_dir / "bm25_article_ids.npy"

    if not cat_path.exists():
        logger.warning("brand=%s: catalogue.parquet not found — skipping", brand)
        return {"brand": brand, "status": "skipped"}

    df = pd.read_parquet(cat_path)
    total = len(df)
    logger.info("brand=%-12s: %d rows loaded", brand, total)

    # Run F1 normalizer on each row.
    # We do NOT pass `brand` to normalize_garment_type — the brand column in these
    # parquets is not per-item (it's the store name), so the brand-prefix strip would
    # misfire. Without brand stripping, the rightmost-noun rule still correctly resolves
    # "DressBerry Women Black Shorts" → shorts (brand "DressBerry" is not a garment noun).
    norm_results = [
        normalize_garment_type(
            str(row["prod_name"]) if pd.notna(row.get("prod_name")) else "",
            str(row["product_type_name"]) if pd.notna(row.get("product_type_name")) else None,
        )
        for _, row in df.iterrows()
    ]

    # Build updated product_type_name and type_confidence columns.
    old_types = df["product_type_name"].fillna("").tolist()
    new_types: list[str] = []
    confidences: list[str] = []
    changed = 0
    by_confidence: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "unknown": 0}

    for i, (nr, orig) in enumerate(zip(norm_results, old_types)):
        conf = nr.type_confidence
        confidences.append(conf)
        by_confidence[conf] = by_confidence.get(conf, 0) + 1

        if conf in ("high", "medium") and nr.garment_type is not None:
            new_val = nr.garment_type  # canonical lowercase F1 value
            new_types.append(new_val)
            if new_val.lower() != orig.lower():
                changed += 1
                if changed <= 5 and not dry_run:
                    logger.info(
                        "  CHANGE: %r  '%s' -> '%s'  [%s]",
                        str(df.iloc[i]["prod_name"])[:50],
                        orig,
                        new_val,
                        conf,
                    )
        else:
            new_types.append(orig)  # keep original store label for unknowns

    df["product_type_name"] = new_types
    df["type_confidence"] = confidences

    # Per-type summary
    type_counts = {}
    for nt in new_types:
        type_counts[nt] = type_counts.get(nt, 0) + 1
    top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:10]

    logger.info(
        "brand=%-12s: %d/%d rows changed  |  confidence: high=%d med=%d unk=%d",
        brand,
        changed,
        total,
        by_confidence.get("high", 0),
        by_confidence.get("medium", 0),
        by_confidence.get("unknown", 0),
    )
    logger.info("  Top types after patch: %s", top_types[:6])

    if dry_run:
        return {
            "brand": brand,
            "total": total,
            "changed": changed,
            "by_confidence": by_confidence,
            "top_types": top_types,
        }

    # Recompute search_text, display_name, facets
    df = _rebuild_search_text(df)

    # Save updated catalogue
    df.to_parquet(str(cat_path), index=False)
    logger.info("brand=%-12s: catalogue.parquet saved (%d rows)", brand, len(df))

    # Rebuild BM25 from new search_text
    texts = df["search_text"].fillna("").tolist()
    article_ids = df["article_id"].astype(str).values

    t0 = time.perf_counter()
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    elapsed = time.perf_counter() - t0

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    np.save(str(bm25_ids_path), article_ids)
    logger.info(
        "brand=%-12s: BM25 rebuilt in %.1fs (%d docs)", brand, elapsed, len(texts)
    )

    return {
        "brand": brand,
        "total": total,
        "changed": changed,
        "by_confidence": by_confidence,
        "top_types": top_types,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch all per-brand catalogue.parquets with F1 garment normalization."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print change counts without writing any files.",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    if args.dry_run:
        print("\n=== DRY RUN — no files written ===")

    results = []
    for brand in STORES:
        result = patch_brand(brand, dry_run=args.dry_run)
        results.append(result)

    elapsed = time.perf_counter() - t_start
    print(f"\n=== F1 patch complete ({elapsed:.1f}s) ===")
    total_changed = sum(r.get("changed", 0) for r in results)
    total_rows = sum(r.get("total", 0) for r in results)
    print(f"  Rows changed: {total_changed:,} / {total_rows:,}")
    for r in results:
        if r.get("status") == "skipped":
            print(f"  {r['brand']:<14}  SKIPPED")
        else:
            print(
                f"  {r['brand']:<14}  changed={r.get('changed', 0):>5}/{r.get('total', 0):>6}"
                f"  high={r.get('by_confidence', {}).get('high', 0):>5}"
                f"  unk={r.get('by_confidence', {}).get('unknown', 0):>5}"
            )

    if not args.dry_run:
        print("\nNext step: python scripts/build_unified_index.py")
        print("Then:      python -m eval.f2_calibration")


if __name__ == "__main__":
    main()
