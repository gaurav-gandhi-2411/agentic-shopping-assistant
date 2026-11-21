"""classify_item: product_type 'footwear' is authoritative over a garment word in the NAME.

Regression (more-options D1): "Men's Sherwani Jutti", product_type footwear, classified as
men_formalwear (name scan for generic facets), so a pair of juttis was both ineligible for the
footwear slot and eligible as a garment. 40 real public rows were affected.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.agents.outfit.slots import classify_item


@pytest.mark.parametrize(
    "name",
    [
        "RoyalGold Silk Men's Sherwani Jutti",
        "Cream Jute Gold Glitz Men's Wedding Sherwani Mojaris",
        "Black Silk Men's Sherwani Juttis",
    ],
)
def test_footwear_typed_sherwani_juttis_are_footwear(name: str) -> None:
    assert classify_item("footwear", name) == "footwear"


def test_the_generic_outerwear_name_rule_is_not_weakened() -> None:
    # 2026-07-30: a Nehru Jacket typed 'outerwear' must keep resolving by NAME to men_formalwear
    assert classify_item("outerwear", "Cream Golden Floral Nehru Jacket") == "men_formalwear"


def test_other_footwear_outcomes_are_unchanged() -> None:
    assert classify_item("footwear", "Women Gold Heels") == "footwear"
    # footwear+bag combos stay accessory (deliberately untouched)
    assert classify_item("footwear", "Red Pixie Jutti & Pixie Potli (Combo)") == "accessory"
    # a garment is still a garment
    assert classify_item("kurta", "Men's Sherwani Style Kurta") != "footwear"


_INDEX_DIR = Path(os.environ.get("OUTFIT_SNAPSHOT_INDEX_DIR", "data/processed/unified"))


def test_real_rows_blast_radius_is_exactly_the_garment_typed_footwear_rows() -> None:
    if not (_INDEX_DIR / "catalogue.parquet").exists():
        pytest.skip(f"no built unified index at {_INDEX_DIR}")
    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        pytest.skip("public view only")
    from src.retrieval.snapshot import public_catalogue

    df = public_catalogue(pd.read_parquet(_INDEX_DIR / "catalogue.parquet"))
    ft = df[df["product_type_name"].fillna("").astype(str).str.lower() == "footwear"]
    classes = [
        classify_item(a, b)
        for a, b in zip(ft["product_type_name"].astype(str), ft["prod_name"].astype(str), strict=True)
    ]
    counts = pd.Series(classes).value_counts().to_dict()
    # Property (any index build): no garment-class outcome survives for a footwear-typed row, and
    # the override only ever moves rows to footwear (accessory combos stay accessory). The exact
    # counts of ONE build (3855+39+1 footwear, 90 accessory) are reported in the commit message,
    # not asserted: CI downloads a different build.
    assert counts.get("men_formalwear", 0) == 0 and counts.get("outerwear", 0) == 0
    assert set(counts) <= {"footwear", "accessory"} and counts.get("footwear", 0) > 1000
