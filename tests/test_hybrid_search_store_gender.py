"""Unit tests for HybridRetriever inactive-store exclusion and strict gender filtering.

Fully self-contained (no index, no network) — follows the mock dense/sparse pattern
established in tests/test_store_diversity_rerank.py.
"""
from __future__ import annotations

import unittest.mock as mock

import pandas as pd

from src.config.stores import get_inactive_stores
from src.retrieval.hybrid_search import HybridRetriever

_DEFAULT_CONFIG: dict = {
    "retrieval": {
        "final_k": 10,
        "top_k": 10,
        "rrf_k": 60,
        "store_diversity": 0.0,
    }
}


def _make_catalogue(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal catalogue_df from a list of row dicts.

    Each row dict may specify: article_id, store, gender, product_type_name,
    prod_name. Missing fields fall back to safe defaults so tests only need to
    set the columns they care about.

    prod_name defaults to a value derived from article_id (not a fixed literal)
    so that HybridRetriever's pre-rerank (prod_name, colour) dedup — added in
    Phase A (2026-07-06) — does not collapse distinct synthetic rows that happen
    to share the default placeholder name/colour; tests that want to exercise
    dedup should set prod_name/facets explicitly instead.
    """
    full_rows = []
    for r in rows:
        article_id = r.get("article_id", "unknown")
        defaults = {
            "display_name": f"Item {article_id}",
            "prod_name": f"Item {article_id}",
            "detail_desc": "desc",
            "image_url": None,
            "price_inr": 500.0,
            "pdp_handle": None,
            "pdp_live": True,
            "product_type_name": "shirt",
            "facets": {
                "colour_group_name": "Black",
                "product_type_name": "shirt",
                "department_name": "Menswear",
            },
        }
        merged = {**defaults, **r}
        full_rows.append(merged)
    return pd.DataFrame(full_rows)


def _make_retriever(cat_df: pd.DataFrame, config: dict | None = None) -> HybridRetriever:
    """Build a HybridRetriever whose dense+sparse mocks return every row in order."""
    article_ids = cat_df["article_id"].tolist()
    dense_mock = mock.MagicMock()
    dense_mock.search.return_value = [(aid, 1.0 / (i + 1)) for i, aid in enumerate(article_ids)]
    sparse_mock = mock.MagicMock()
    sparse_mock.search.return_value = [(aid, 1.0 / (i + 1)) for i, aid in enumerate(article_ids)]
    return HybridRetriever(dense_mock, sparse_mock, cat_df, config or _DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Inactive-store exclusion
# ---------------------------------------------------------------------------


class TestInactiveStoreExclusion:
    def test_berrylush_excluded_from_results(self) -> None:
        """berrylush is flagged inactive in STORE_CONFIG; it must never surface."""
        assert "berrylush" in get_inactive_stores()  # sanity check on fixture assumption
        cat_df = _make_catalogue(
            [
                {"article_id": "a1", "store": "berrylush", "gender": "women"},
                {"article_id": "a2", "store": "myntra", "gender": "women"},
                {"article_id": "a3", "store": "berrylush", "gender": "women"},
            ]
        )
        retriever = _make_retriever(cat_df)
        results = retriever.search("shirt", top_k=10)
        stores = {r["store"] for r in results}
        assert "berrylush" not in stores
        assert stores == {"myntra"}

    def test_hm_excluded_from_results(self) -> None:
        """hm is flagged inactive (dormant) in STORE_CONFIG; it must never surface."""
        assert "hm" in get_inactive_stores()
        cat_df = _make_catalogue(
            [
                {"article_id": "a1", "store": "hm", "gender": "men"},
                {"article_id": "a2", "store": "snitch", "gender": "men"},
            ]
        )
        retriever = _make_retriever(cat_df)
        results = retriever.search("shirt", top_k=10)
        stores = {r["store"] for r in results}
        assert stores == {"snitch"}

    def test_inactive_store_excluded_even_without_store_filter(self) -> None:
        """Exclusion applies globally, not just when a store filter is explicitly set."""
        cat_df = _make_catalogue(
            [
                {"article_id": "a1", "store": "berrylush", "gender": "women"},
                {"article_id": "a2", "store": "libas", "gender": "women"},
            ]
        )
        retriever = _make_retriever(cat_df)
        results = retriever.search("shirt", top_k=10, filters=None)
        assert all(r["store"] != "berrylush" for r in results)

    def test_active_stores_unaffected(self) -> None:
        """Active stores must pass through unfiltered by the inactive-store guard."""
        cat_df = _make_catalogue(
            [
                {"article_id": "a1", "store": "myntra", "gender": "women"},
                {"article_id": "a2", "store": "snitch", "gender": "men"},
                {"article_id": "a3", "store": "libas", "gender": "women"},
            ]
        )
        retriever = _make_retriever(cat_df)
        results = retriever.search("shirt", top_k=10)
        assert {r["store"] for r in results} == {"myntra", "snitch", "libas"}


# ---------------------------------------------------------------------------
# Strict gender filter
# ---------------------------------------------------------------------------


class TestStrictGenderFilter:
    def test_explicit_men_filter_excludes_unknown_gender(self) -> None:
        """Explicit gender='men' filter must exclude unknown-gender rows (bug fix)."""
        cat_df = _make_catalogue(
            [
                {"article_id": "a1", "store": "globalrepublic", "gender": "men"},
                {"article_id": "a2", "store": "globalrepublic", "gender": "unknown"},
                {"article_id": "a3", "store": "globalrepublic", "gender": "women"},
            ]
        )
        retriever = _make_retriever(cat_df)
        results = retriever.search("shirt", top_k=10, filters={"gender": "men"})
        ids = {r["article_id"] for r in results}
        assert ids == {"a1"}

    def test_explicit_women_filter_excludes_unknown_gender(self) -> None:
        cat_df = _make_catalogue(
            [
                {"article_id": "a1", "store": "globalrepublic", "gender": "men"},
                {"article_id": "a2", "store": "globalrepublic", "gender": "unknown"},
                {"article_id": "a3", "store": "globalrepublic", "gender": "women"},
            ]
        )
        retriever = _make_retriever(cat_df)
        results = retriever.search("shirt", top_k=10, filters={"gender": "women"})
        ids = {r["article_id"] for r in results}
        assert ids == {"a3"}

    def test_explicit_gender_filter_excludes_null_gender(self) -> None:
        """Rows with a missing/null gender column value must also be excluded."""
        cat_df = _make_catalogue(
            [
                {"article_id": "a1", "store": "myntra", "gender": "men"},
                {"article_id": "a2", "store": "myntra", "gender": None},
            ]
        )
        retriever = _make_retriever(cat_df)
        results = retriever.search("shirt", top_k=10, filters={"gender": "men"})
        ids = {r["article_id"] for r in results}
        assert ids == {"a1"}

    def test_no_gender_filter_keeps_unknown_rows(self) -> None:
        """With no explicit gender filter, unknown-gender rows must still pass through."""
        cat_df = _make_catalogue(
            [
                {"article_id": "a1", "store": "globalrepublic", "gender": "unknown"},
                {"article_id": "a2", "store": "globalrepublic", "gender": "men"},
            ]
        )
        retriever = _make_retriever(cat_df)
        results = retriever.search("shirt", top_k=10)
        ids = {r["article_id"] for r in results}
        assert ids == {"a1", "a2"}

    def test_index_group_name_translation_still_excludes_unknown(self) -> None:
        """index_group_name='menswear' → gender='men' translation must also apply strictly."""
        cat_df = _make_catalogue(
            [
                {"article_id": "a1", "store": "myntra", "gender": "men"},
                {"article_id": "a2", "store": "myntra", "gender": "unknown"},
            ]
        )
        retriever = _make_retriever(cat_df)
        results = retriever.search("shirt", top_k=10, filters={"index_group_name": "Menswear"})
        ids = {r["article_id"] for r in results}
        assert ids == {"a1"}
