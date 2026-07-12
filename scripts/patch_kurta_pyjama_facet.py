#!/usr/bin/env python
"""Surgical facet patch for kurta-pyjama SET rows mistagged product_type_name="nightwear".

Follow-up to the BUG 4 fix in src/catalogue/normalizer.py's `_COMPOUND_TERMS`
(2026-07-13): "kurta and pyjama"/"kurta pajama" SET titles (e.g. "Men Kurta and
Pyjama Set Dupion Silk") were resolving to garment_type="nightwear" because the
bare "pyjama"/"pajama" garment rule matched further right than "kurta" in the
rightmost-match-wins position scan (see normalizer.py's algorithm docstring).
This mistagged 40 real rows in data/processed/unified/catalogue.parquet at
ingest time — confirmed by direct query below.

Fixing normalize_garment_type() does NOT retroactively fix already-built
catalogue rows, so this script re-runs the now-fixed normalizer against the
identified rows and patches product_type_name, the `facets` dict, and
search_text/display_name (kept consistent with the build formula in
src/catalogue/loader.py::build_searchable_text) in place.

Scope note: this patches the CATALOGUE PARQUET ONLY. The FAISS dense index and
BM25 sparse index were built by embedding/tokenizing the OLD search_text
(which already contained the literal words "kurta"/"pyjama" — only the
product_type_name facet field differs) — see hybrid_search.py's
`product_type_name` hard-filter, which reads catalogue_df live off the
dataframe every query, NOT from a pre-trained index. So this patch is
sufficient to fix the retrieval filter/facet-gate behaviour without a FAISS/
BM25 rebuild. A full re-embed would only microscopically refine semantic
ranking order (search_text's product_type_name token changing from
"nightwear" to "kurta"), not correctness — not done here per the same
"two-row correction doesn't need a full pipeline re-run" reasoning as
patch_thin_category_facets.py.

Usage:
    python scripts/patch_kurta_pyjama_facet.py --dry-run
    python scripts/patch_kurta_pyjama_facet.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.catalogue.normalizer import normalize_garment_type  # noqa: E402

# Rows this patch is scoped to: kurta+pyjama/pajama combo titles currently
# tagged nightwear. This mirrors the exact "kurta and pyjama"/"kurta pajama"
# compound phrases added to normalizer.py's _COMPOUND_TERMS — NOT a blanket
# "contains kurta and contains pyjama" match, so genuinely distinct titles
# (if any existed) would not be silently swept up here too.
_MASK_QUERY = (
    r'prod_name.str.contains("kurta", case=False, na=False) '
    r'and prod_name.str.contains("pyjama|pajama", case=False, na=False, regex=True) '
    r'and product_type_name == "nightwear"'
)

_EXPECTED_ROW_COUNT = 40


def _rebuild_search_text_row(row: pd.Series) -> str:
    """Recompute search_text for one row — mirrors loader.py::build_searchable_text."""
    return (
        str(row["prod_name"] or "") + ". "
        + str(row["product_type_name"] or "") + ". "
        + str(row["colour_group_name"] or "") + ". "
        + str(row["department_name"] or "") + ". "
        + str(row["detail_desc"] or "")
    )


def _rebuild_display_name_row(row: pd.Series) -> str:
    """Recompute display_name for one row — mirrors loader.py::build_searchable_text."""
    return (
        str(row["prod_name"] or "").strip()
        + " ("
        + str(row["colour_group_name"] or "").strip()
        + " "
        + str(row["product_type_name"] or "").strip()
        + ")"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", default=str(_ROOT / "data" / "processed" / "unified")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cat_path = Path(args.data_dir) / "catalogue.parquet"
    df = pd.read_parquet(cat_path)

    mask = (
        df["prod_name"].str.contains("kurta", case=False, na=False)
        & df["prod_name"].str.contains("pyjama|pajama", case=False, na=False, regex=True)
        & (df["product_type_name"] == "nightwear")
    )
    n = int(mask.sum())
    if n != _EXPECTED_ROW_COUNT:
        raise SystemExit(
            f"FAIL: expected exactly {_EXPECTED_ROW_COUNT} mistagged rows, found {n} — "
            "catalogue has changed since this patch was written; re-verify before applying."
        )

    matched = df.loc[mask]
    for article_id, prod_name in zip(matched["article_id"], matched["prod_name"], strict=True):
        result = normalize_garment_type(prod_name)
        if result.garment_type != "kurta":
            raise SystemExit(
                f"FAIL: article_id={article_id} prod_name={prod_name!r} — fixed normalizer "
                f"returned garment_type={result.garment_type!r}, expected 'kurta'. "
                "Re-verify the normalizer fix before applying this patch."
            )
        print(f"{article_id}: product_type_name 'nightwear' -> 'kurta'  ({prod_name!r})")

    if args.dry_run:
        print(f"\n--dry-run: {n} rows would be patched; no file written")
        return

    df.loc[mask, "product_type_name"] = "kurta"
    df.loc[mask, "type_confidence"] = "high"
    df.loc[mask, "facets"] = df.loc[mask, "facets"].apply(
        lambda f: {**f, "product_type_name": "kurta"}
    )
    df.loc[mask, "search_text"] = df.loc[mask].apply(_rebuild_search_text_row, axis=1)
    df.loc[mask, "display_name"] = df.loc[mask].apply(_rebuild_display_name_row, axis=1)

    df.to_parquet(cat_path, index=False)
    print(f"\nwrote {cat_path} ({n} rows patched)")
    print(
        "Next: restart/redeploy the backend so it reloads this catalogue.parquet, "
        "then re-upload to GCS per DEPLOY.md if the deployed index lives there. "
        "FAISS/BM25 rebuild NOT required for filter correctness — see module docstring."
    )


if __name__ == "__main__":
    main()
