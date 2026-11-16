"""Registry ``allowed_categories`` and its build-time enforcement (R1)."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import src.config.store_registry as reg
from src.catalogue.hygiene import CATEGORIES, restrict_to_allowed_categories


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["1", "2", "3", "4", "5"],
            "store": ["shopx", "shopx", "shopx", "other", "SHOPX"],
            "prod_name": ["Kurta", "Earrings", "Mystery", "Kurta", "Kurta B"],
            "product_type_name": ["kurta", "earrings", "x", "kurta", "kurta"],
            "price_inr": [1.0] * 5,
            "category": ["garment", "jewellery", "unknown", "garment", "garment"],
        }
    )


def test_registry_default_is_unrestricted_and_values_are_valid_categories() -> None:
    for slug, rec in reg.STORE_REGISTRY.items():
        if rec.allowed_categories is not None:
            assert rec.allowed_categories <= set(CATEGORIES), slug
    assert reg.STORE_REGISTRY["libas"].allowed_categories is None


def test_no_restriction_is_a_noop_that_returns_the_same_rows() -> None:
    df = _frame()
    kept, dropped = restrict_to_allowed_categories(df, {})
    assert kept is df
    assert dropped.empty


def test_restricted_store_drops_other_categories_keeps_unknown_and_other_stores() -> None:
    kept, dropped = restrict_to_allowed_categories(_frame(), {"shopx": frozenset({"jewellery"})})
    # shopx garment rows (1, and 5 via case-insensitive store match) are dropped; the jewellery
    # row, the 'unknown' row (rule gap fails open) and the other store's garment stay.
    assert kept["article_id"].tolist() == ["2", "3", "4"]
    assert dropped["article_id"].tolist() == ["1", "5"]
    assert set(dropped["reason"]) == {"category_not_allowed_for_store"}


def test_missing_category_or_store_column_is_a_noop() -> None:
    df = _frame().drop(columns=["category"])
    kept, dropped = restrict_to_allowed_categories(df, {"shopx": frozenset({"jewellery"})})
    assert len(kept) == len(df) and dropped.empty


def test_default_reads_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = dataclasses.replace(
        reg.STORE_REGISTRY["libas"], allowed_categories=frozenset({"garment"})
    )
    monkeypatch.setitem(reg.STORE_REGISTRY, "libas", rec)
    assert reg.allowed_categories_by_store() == {"libas": frozenset({"garment"})}
    df = pd.DataFrame(
        {"article_id": ["1", "2"], "store": ["libas", "libas"], "category": ["garment", "footwear"]}
    )
    kept, dropped = restrict_to_allowed_categories(df)
    assert kept["article_id"].tolist() == ["1"]
    assert dropped["article_id"].tolist() == ["2"]
