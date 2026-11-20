"""The lazy, ascending-price footwear floor equals the original full-scan derivation.

The original implementation (kept below as the reference) built item dicts for EVERY
footwear-class row up front, sorted them by price, and took the first gate survivor. The current
one only prefilters vectorised and builds items lazily, cheapest first. Same floors required.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pytest

from src.agents.outfit import composer
from src.agents.outfit.composer import _score_candidates, footwear_price_floor
from src.agents.outfit.occasions import OCCASIONS
from src.agents.outfit.slots import FOOTWEAR_KEYWORDS, classify_item


def _reference_rows(catalogue_df: pd.DataFrame) -> dict[str, list[dict]]:
    pt = catalogue_df["product_type_name"].fillna("").astype(str)
    nm = catalogue_df["prod_name"].fillna("").astype(str)
    text = (pt + " " + nm).str.lower()
    pattern = "|".join(re.escape(k) for k in sorted(FOOTWEAR_KEYWORDS))
    hit = text.str.contains(pattern, regex=True).to_numpy()
    out: dict[str, list[dict]] = {"men": [], "women": []}
    sub = catalogue_df[hit]
    for aid, ptv, nmv, row in zip(sub.index, pt[hit], nm[hit], sub.itertuples(index=False)):
        gender = str(getattr(row, "gender", "") or "").lower()
        price = getattr(row, "price_inr", None)
        if gender not in out or not isinstance(price, (int, float)) or not price > 0:
            continue
        if classify_item(ptv, nmv) != "footwear":
            continue
        facets = getattr(row, "facets", None)
        facets = facets if isinstance(facets, dict) else {}
        out[gender].append(
            {
                "article_id": str(aid),
                "prod_name": nmv,
                "display_name": getattr(row, "display_name", nmv),
                "colour": facets.get("colour_group_name", ""),
                "product_type": ptv,
                "detail_desc": getattr(row, "detail_desc", None),
                "gender": gender,
                "price_inr": float(price),
                "store": getattr(row, "store", None),
            }
        )
    for rows in out.values():
        rows.sort(key=lambda r: r["price_inr"])
    return out


def _reference_floor(catalogue_df: pd.DataFrame, gender: str, occasion: str) -> float | None:
    for item in _reference_rows(catalogue_df).get(gender, []):
        if _score_candidates(
            [item],
            query="",
            slot_name="footwear",
            occasion_slug=occasion,
            gender=gender,
            anchor_colour="",
            seen_ids=set(),
            seen_prod_colour=set(),
            budget_remaining=None,
            pairing_stats=None,
            anchor_class="",
            seen_stores=None,
            neutral_fallback_ids=set(),
        ):
            return float(item["price_inr"])
    return None


class _Cat:
    def __init__(self, df: pd.DataFrame) -> None:
        self.catalogue_df = df


def _synthetic() -> pd.DataFrame:
    rows = [
        ("1", "Women Gold Heels", "footwear", "women", 899.0),
        ("2", "Women Embellished Juttis", "footwear", "women", 649.0),
        ("3", "Women Rugged Hiking Boots", "footwear", "women", 199.0),
        ("4", "Men Brown Juttis", "footwear", "men", 349.0),
        ("5", "Women Silk Kurta", "kurta", "women", 99.0),
        ("6", "Women Bad Price Heels", "footwear", "women", 0.0),
        ("7", "Mens Leather Loafers", "footwear", "men", 1299.0),
        ("8", "Women Tied Flats", "footwear", "women", 649.0),  # price tie: catalogue order kept
        ("9", "Unknown Gender Sandals", "footwear", "unknown", 99.0),
    ]
    df = pd.DataFrame(
        rows, columns=["article_id", "prod_name", "product_type_name", "gender", "price_inr"]
    )
    df["display_name"] = df["prod_name"]
    df["detail_desc"] = ""
    df["store"] = "s"
    df["facets"] = [{} for _ in range(len(df))]
    return df.set_index("article_id")


@pytest.mark.parametrize("gender", ["women", "men"])
def test_synthetic_floors_match_reference_for_every_occasion(gender: str) -> None:
    df = _synthetic()
    r = _Cat(df)
    for occ in OCCASIONS:
        assert footwear_price_floor(r, gender, occ) == _reference_floor(df, gender, occ), occ  # type: ignore[arg-type]


def test_only_a_prefix_of_rows_is_classified_when_the_cheapest_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _synthetic()
    seen: list[str] = []
    real = composer.classify_item
    monkeypatch.setattr(composer, "classify_item", lambda p, n: (seen.append(n), real(p, n))[1])
    assert footwear_price_floor(_Cat(df), "women", "sangeet") == 649.0  # type: ignore[arg-type]
    # cheapest-first stops at the first survivor: the Rs 99 kurta never reaches the
    # classifier (prefilter), and the dearer heels/loafers are never touched
    assert "Women Gold Heels" not in seen and "Mens Leather Loafers" not in seen


_INDEX_DIR = Path(os.environ.get("OUTFIT_SNAPSHOT_INDEX_DIR", "data/processed/unified"))


def test_real_index_floors_identical_to_original_for_every_occasion() -> None:
    if not (_INDEX_DIR / "catalogue.parquet").exists():
        pytest.skip(f"no built unified index at {_INDEX_DIR}")
    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        pytest.skip("public view only")
    from src.retrieval.snapshot import public_catalogue

    df = public_catalogue(pd.read_parquet(_INDEX_DIR / "catalogue.parquet")).set_index("article_id")
    r = _Cat(df)
    rows = _reference_rows(df)  # one reference scan, reused across occasions
    checked = 0
    for gender in ("women", "men"):
        for occ in OCCASIONS:
            want = None
            for item in rows[gender]:
                if _score_candidates(
                    [item], query="", slot_name="footwear", occasion_slug=occ, gender=gender,
                    anchor_colour="", seen_ids=set(), seen_prod_colour=set(), budget_remaining=None,
                    pairing_stats=None, anchor_class="", seen_stores=None, neutral_fallback_ids=set(),
                ):
                    want = float(item["price_inr"])
                    break
            assert footwear_price_floor(r, gender, occ) == want, (gender, occ)  # type: ignore[arg-type]
            checked += 1
    assert checked == 2 * len(OCCASIONS)
