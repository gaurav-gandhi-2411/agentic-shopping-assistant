#!/usr/bin/env python
"""Surgical facet patch for two specific, hand-verified catalogue rows.

Follow-up to reports/catalogue_gap_report.md (2026-07-11): give the single
real sherwani a dedicated, filterable product_type_name facet (it was
dumped in Flipkart's generic "Kurtas, Ethnic Sets and Bottoms" bucket, so a
hard product_type_name="sherwani" filter could never find it), and correct
one women's item that was actively MISLABELED product_type_name="footwear"
(it's a boot-cut trouser, not footwear — this also corrects the earlier gap
report's "women's footwear: 1" to the true count of 0).

Deliberately minimal: only the `product_type_name` column and the matching
key inside the `facets` dict change (search_text/display_name/BM25/FAISS
untouched — the retrieval hard-filter reads product_type_name/facets live
off the dataframe every query, not from a pre-trained index, and lexical
relevance for "sherwani" already works via prod_name). No bulk
reclassification, no F1 renormalization pass — see the module docstring in
patch_catalogue_f1.py for why a full pipeline re-run is a much bigger,
riskier operation than this two-row correction needs.

Usage:
    python scripts/patch_thin_category_facets.py --dry-run
    python scripts/patch_thin_category_facets.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent

# (article_id, old_product_type_name, new_product_type_name, reason)
_PATCHES: tuple[tuple[str, str, str, str], ...] = (
    (
        "de4c2aca-d13b-5954-8757-4bb20b40fd93",
        "Kurtas, Ethnic Sets and Bottoms",
        "sherwani",
        "the catalogue's only sherwani (Mods Western Star Self Design Sherwani) — "
        "give it a dedicated facet so a hard product_type_name filter can find it",
    ),
    (
        "7891313656030",
        "footwear",
        "trousers",
        "'Wine High Waisted Boot Cut Cotton Blend Lower' was mislabeled footwear — "
        "it's a bottom garment; corrects the gap report's 'women's footwear: 1' to 0",
    ),
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

    for article_id, old_val, new_val, reason in _PATCHES:
        mask = df["article_id"] == article_id
        n = int(mask.sum())
        if n != 1:
            raise SystemExit(
                f"FAIL: expected exactly 1 row for article_id={article_id}, found {n} — "
                "catalogue has changed since this patch was written; re-verify before applying."
            )
        current = df.loc[mask, "product_type_name"].iloc[0]
        if current != old_val:
            raise SystemExit(
                f"FAIL: article_id={article_id} product_type_name is {current!r}, "
                f"expected {old_val!r} — catalogue has changed; re-verify before applying."
            )
        print(f"{article_id}: product_type_name {old_val!r} -> {new_val!r}  ({reason})")
        if not args.dry_run:
            df.loc[mask, "product_type_name"] = new_val
            df.loc[mask, "facets"] = df.loc[mask, "facets"].apply(
                lambda f, nv=new_val: {**f, "product_type_name": nv}
            )

    if args.dry_run:
        print("\n--dry-run: no file written")
        return

    df.to_parquet(cat_path, index=False)
    print(f"\nwrote {cat_path}")
    print("Next: restart/redeploy the backend so it reloads this catalogue.parquet, "
          "then re-upload to GCS per DEPLOY.md if the deployed index lives there.")


if __name__ == "__main__":
    main()
