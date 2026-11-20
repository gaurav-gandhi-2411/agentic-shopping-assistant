"""has_gender_text_conflict: a name that OPENS with a gender word conflicts with the other gender.

Regression: price pushdown exposed cheap rows such as "Men Aqua Silk Blend Embroidered Dupatta"
(catalogue gender "women") in women's looks, because only "Men's"/"Mens"/"For Men" matched.
Real-row coverage skips without a built unified index (OUTFIT_SNAPSHOT_INDEX_DIR).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.catalogue.cleaning import has_gender_text_conflict


@pytest.mark.parametrize(
    "name",
    [
        "Men Aqua Silk Blend Embroidered Dupatta",
        "Men Pista Green Embroidered Dupatta",
        "men aqua dupatta",  # case-insensitive
        "  Men Gold Kurta",  # leading whitespace
        "Men's Yellow - Dupatta",  # pre-existing pattern still works
        "Mens Cotton Kurta",
        "Men’s Silk Stole",  # curly apostrophe
        "Boys Cotton Kurta",
        "Boy Printed Shirt",
    ],
)
def test_leading_men_word_conflicts_with_women_look(name: str) -> None:
    assert has_gender_text_conflict(name, "women") is True


@pytest.mark.parametrize(
    "name",
    ["Women Pink Cotton Kurta", "Womens Silk Saree", "Women’s Heels", "Ladies Cotton Kurti", "Girls Frock"],
)
def test_leading_women_word_conflicts_with_men_look(name: str) -> None:
    assert has_gender_text_conflict(name, "men") is True


@pytest.mark.parametrize(
    "name",
    [
        "Mendhi Design Silk Dupatta",  # 'Men' only as a prefix of another word
        "Menswear Style Cotton Shirt",
        "Pink Silk Dupatta",
        "Aqua Embroidered Dupatta For Festive Wear",
        "Silk Dupatta Men Collection Free Size",  # gender word NOT leading and no men's/for-men form
        "Women Silk Dupatta",  # the women's own look is never flagged by a leading Women
    ],
)
def test_no_false_conflict_for_women_look(name: str) -> None:
    assert has_gender_text_conflict(name, "women") is False


@pytest.mark.parametrize(
    "name",
    [
        "Men & Women Running Sneakers",
        "Men and Women Cotton Cap",
        "Women Men Sports Watch",
        "Unisex Cotton Cap",
        "Men/Women Sunglasses",
    ],
)
def test_dual_marketed_names_never_conflict(name: str) -> None:
    assert has_gender_text_conflict(name, "women") is False
    assert has_gender_text_conflict(name, "men") is False


def test_unisex_and_empty_never_conflict() -> None:
    assert has_gender_text_conflict("Men Kurta", "unisex") is False
    assert has_gender_text_conflict(None, "women") is False
    assert has_gender_text_conflict("", "men") is False


def test_conflict_is_enforced_in_the_slot_gate_for_every_slot() -> None:
    """The slot gate (_score_candidates) rejects such an item whatever slot it is offered for."""
    from src.agents.outfit.composer import _score_candidates

    item = {
        "article_id": "x1",
        "prod_name": "Men Aqua Silk Blend Embroidered Dupatta",
        "display_name": "Men Aqua Silk Blend Embroidered Dupatta",
        "colour": "aqua",
        "product_type": "dupatta",
        "gender": "women",  # the catalogue label is wrong, the name is the trustworthy signal
        "price_inr": 729.0,
        "score": 0.9,
    }
    for slot in ("accessory", "top", "bottom", "outerwear", "footwear"):
        out = _score_candidates(
            [item],
            query="dupatta",
            slot_name=slot,
            occasion_slug="sangeet",
            gender="women",
            anchor_colour="",
            seen_ids=set(),
            seen_prod_colour=set(),
            budget_remaining=None,
            pairing_stats=None,
            anchor_class="",
            seen_stores=None,
            neutral_fallback_ids=set(),
        )
        assert out == [], slot


_INDEX_DIR = Path(os.environ.get("OUTFIT_SNAPSHOT_INDEX_DIR", "data/processed/unified"))


def test_real_rows_with_leading_gender_word_all_conflict() -> None:
    if not (_INDEX_DIR / "catalogue.parquet").exists():
        pytest.skip(f"no built unified index at {_INDEX_DIR}")
    from src.retrieval.snapshot import public_catalogue

    df = public_catalogue(pd.read_parquet(_INDEX_DIR / "catalogue.parquet"))
    name = df["prod_name"].fillna("").astype(str)
    gender = df["gender"].fillna("").astype(str).str.lower()
    lead_men = name.str.contains(r"^\W*(?:men|mens|men['’]s|boys?)\b", case=False, regex=True)
    rows = df[(gender == "women") & lead_men]
    assert len(rows) > 0, "expected women-labelled rows opening with a men's word"
    # every such row that is not explicitly dual-marketed is rejected for a women's look
    from src.catalogue.cleaning import _DUAL_GENDER_MARKETING_RE

    for n in rows["prod_name"].astype(str):
        assert has_gender_text_conflict(n, "women") or _DUAL_GENDER_MARKETING_RE.search(n), n
