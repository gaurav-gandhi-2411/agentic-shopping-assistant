#!/usr/bin/env python
"""Catalogue inventory gap audit — quantify categories/attributes with
genuinely thin or zero inventory.

Every number is a direct pandas count against the live catalogue parquet —
no LLM, no retrieval, fully reproducible, zero cost. This is the source data
for reports/catalogue_gap_report.md: the honest "what we can't serve yet"
list, and exactly the inventory a data-partnership/funding conversation
would need to fill.

Usage:
    python scripts/catalogue_gap_audit.py
    python scripts/catalogue_gap_audit.py --data-dir data/processed/unified
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", default=str(_ROOT / "data" / "processed" / "unified")
    )
    args = parser.parse_args()

    df = pd.read_parquet(Path(args.data_dir) / "catalogue.parquet")
    print(f"catalogue total: {len(df)} rows\n")

    women = df[df["gender"] == "women"]
    men = df[df["gender"] == "men"]
    print(f"women rows: {len(women)}  |  men rows: {len(men)}\n")

    section("MEN'S OCCASIONWEAR")

    sherwani = men[men["prod_name"].str.contains("sherwani", case=False, na=False)]
    print(f"sherwanis (men): {len(sherwani)}")
    print(f"  colours: {sherwani['colour_group_name'].value_counts().to_dict()}")
    print(f"  product_type_name values: {sherwani['product_type_name'].unique().tolist()}")

    bandhgala = men[men["prod_name"].str.contains("bandhgala", case=False, na=False)]
    print(f"\nbandhgalas (men): {len(bandhgala)}")

    blazer = men[men["product_type_name"] == "blazer"]
    print(f"\nblazers (men), total: {len(blazer)}")
    print(f"  under 5000: {len(blazer[blazer['price_inr'] <= 5000])}")
    print(f"  colours: {blazer['colour_group_name'].value_counts().to_dict()}")

    eth_bottom_kw = r"churidar|pyjama|pajama|dhoti|patiala salwar"
    eth_bottom = men[men["prod_name"].str.contains(eth_bottom_kw, case=False, na=False)]
    print(f"\nmen's ethnic bottoms (churidar/pyjama/dhoti), by NAME: {len(eth_bottom)}")
    print(f"  actual product_type_name values: "
          f"{eth_bottom['product_type_name'].value_counts().to_dict()}")
    nightwear_leak = eth_bottom[eth_bottom["product_type_name"] == "nightwear"]
    print(f"  of which classified as nightwear (not usable as festive bottom): "
          f"{len(nightwear_leak)}")

    section("FOOTWEAR")
    women_fw = women[women["product_type_name"] == "footwear"]
    men_fw = men[men["product_type_name"] == "footwear"]
    print(f"women's footwear: {len(women_fw)}")
    print(f"men's footwear: {len(men_fw)}")
    ethnic_fw_kw = r"jutti|mojari|kolhapuri"
    women_ethnic_fw = women[women["prod_name"].str.contains(ethnic_fw_kw, case=False, na=False)]
    men_ethnic_fw = men[men["prod_name"].str.contains(ethnic_fw_kw, case=False, na=False)]
    print(f"women's ethnic footwear (jutti/mojari/kolhapuri): {len(women_ethnic_fw)}")
    print(f"men's ethnic footwear (jutti/mojari/kolhapuri): {len(men_ethnic_fw)}")

    section("JEWELLERY / BRIDAL ACCESSORIES")
    jewel_kw = r"necklace|earring|bangle|bracelet|maang tikka|jewell?ery|nose pin|anklet"
    jewellery = df[df["prod_name"].str.contains(jewel_kw, case=False, na=False)]
    print(f"jewellery items (any gender, by name): {len(jewellery)}")
    jewel_type = df[
        df["product_type_name"].str.contains("jewell?ery", case=False, na=False, regex=True)
    ]
    print(f"jewellery by product_type_name facet: {len(jewel_type)}")

    section("COLOUR DEPTH — men's ethnic occasionwear")
    for kw, label in [("sherwani", "sherwani"), ("bandhgala", "bandhgala")]:
        rows = men[men["prod_name"].str.contains(kw, case=False, na=False)]
        print(f"{label}: n={len(rows)}  "
              f"colours_available={sorted(rows['colour_group_name'].dropna().unique().tolist())}")

    kurta_men = men[men["product_type_name"] == "kurta"]
    print(f"\nmen's kurta (for contrast, healthy category): n={len(kurta_men)}  "
          f"distinct colours={kurta_men['colour_group_name'].nunique()}")

    section("STORE CONCENTRATION — thin occasionwear categories")
    for label, rows in [("sherwani", sherwani), ("blazer(men)", blazer),
                        ("women's footwear", women_fw)]:
        print(f"{label}: stores = {rows['store'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
