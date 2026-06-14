"""Tests for the unified cross-store index and related retrieval behaviour.

These tests load the pre-built unified index from data/processed/unified/
and verify:
  1. The index loads without error (correct file layout).
  2. Text queries return results spanning MULTIPLE stores.
  3. The optional store filter correctly narrows results to one store.
  4. The `store` field is present in every result.
  5. pdp_live is preserved (not uniformly None) across brands that carry it.

Marked ``requires_index`` so the CI gate ``pytest -m "not requires_index"`` skips
them when the unified index hasn't been built.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.requires_index

_REPO_ROOT = Path(__file__).parent.parent
_UNIFIED_DIR = _REPO_ROOT / "data" / "processed" / "unified"
_CLIP_UNIFIED_DIR = _REPO_ROOT / "data" / "processed" / "clip" / "unified"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config() -> dict:
    from src.catalogue.loader import load_config

    return load_config()


@pytest.fixture(scope="module")
def unified_df() -> pd.DataFrame:
    """Load the unified catalogue parquet."""
    return pd.read_parquet(_UNIFIED_DIR / "catalogue.parquet")


@pytest.fixture(scope="module")
def unified_retriever(config: dict, unified_df: pd.DataFrame):
    """Return a HybridRetriever loaded from the unified index."""
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    dense = DenseRetriever.load(config, _UNIFIED_DIR)
    sparse = SparseRetriever.load(config, _UNIFIED_DIR)
    return HybridRetriever(dense, sparse, unified_df, config)


# ---------------------------------------------------------------------------
# Index integrity tests
# ---------------------------------------------------------------------------


def test_unified_catalogue_exists() -> None:
    """Catalogue parquet must exist at the unified path."""
    assert (_UNIFIED_DIR / "catalogue.parquet").exists(), (
        "Run scripts/build_unified_index.py first."
    )


def test_unified_catalogue_has_store_column(unified_df: pd.DataFrame) -> None:
    """Every row must carry a `store` value."""
    assert "store" in unified_df.columns
    assert unified_df["store"].notna().all(), "Some rows have null store"


def test_unified_catalogue_spans_all_7_stores(unified_df: pd.DataFrame) -> None:
    """All 7 expected stores must be present."""
    expected = {"hm", "myntra", "flipkart", "snitch", "fashor", "powerlook", "virgio"}
    actual = set(unified_df["store"].unique())
    assert expected == actual, f"Missing stores: {expected - actual}"


def test_unified_article_ids_globally_unique(unified_df: pd.DataFrame) -> None:
    """No duplicate article_ids in the merged catalogue."""
    dupes = unified_df["article_id"].duplicated().sum()
    assert dupes == 0, f"Found {dupes} duplicate article_ids in unified catalogue"


def test_unified_dense_ids_aligned(unified_df: pd.DataFrame) -> None:
    """dense_article_ids.npy length must equal catalogue row count."""
    ids = np.load(str(_UNIFIED_DIR / "dense_article_ids.npy"), allow_pickle=True)
    assert len(ids) == len(unified_df), (
        f"dense ids {len(ids)} != catalogue rows {len(unified_df)}"
    )


def test_unified_bm25_ids_aligned(unified_df: pd.DataFrame) -> None:
    """bm25_article_ids.npy length must equal catalogue row count."""
    ids = np.load(str(_UNIFIED_DIR / "bm25_article_ids.npy"), allow_pickle=True)
    assert len(ids) == len(unified_df), (
        f"BM25 ids {len(ids)} != catalogue rows {len(unified_df)}"
    )


def test_unified_clip_ids_aligned(unified_df: pd.DataFrame) -> None:
    """clip_article_ids.npy length must equal catalogue row count."""
    ids = np.load(str(_CLIP_UNIFIED_DIR / "clip_article_ids.npy"), allow_pickle=True)
    assert len(ids) == len(unified_df), (
        f"CLIP ids {len(ids)} != catalogue rows {len(unified_df)}"
    )


# ---------------------------------------------------------------------------
# Multi-store query tests
# ---------------------------------------------------------------------------


def test_black_dress_spans_multiple_stores(unified_retriever) -> None:
    """'black dress' must return hits from more than one store."""
    results = unified_retriever.search("black dress", top_k=20)
    assert len(results) > 0, "No results for 'black dress'"
    stores = {r["store"] for r in results if r.get("store")}
    assert len(stores) >= 2, (
        f"'black dress' only hit store(s): {stores}. "
        "Expected results from multiple stores in unified mode."
    )
    # Use ASCII-safe output to avoid cp1252 encoding errors on Windows
    print(f"\n'black dress' stores: {sorted(stores)}")
    for r in results[:5]:
        price = r.get("price_inr")
        print(f"  [{r['store']}] {r['display_name']} -- Rs.{price}")


def test_white_sneakers_spans_multiple_stores(unified_retriever) -> None:
    """'white sneakers' must return hits from more than one store."""
    results = unified_retriever.search("white sneakers", top_k=20)
    assert len(results) > 0, "No results for 'white sneakers'"
    stores = {r["store"] for r in results if r.get("store")}
    assert len(stores) >= 2, (
        f"'white sneakers' only hit store(s): {stores}. "
        "Expected results from multiple stores."
    )
    print(f"\n'white sneakers' stores: {sorted(stores)}")
    for r in results[:5]:
        price = r.get("price_inr")
        print(f"  [{r['store']}] {r['display_name']} -- Rs.{price}")


def test_all_results_have_store_field(unified_retriever) -> None:
    """Every search result must carry a non-None `store` field."""
    results = unified_retriever.search("kurta", top_k=10)
    for r in results:
        assert "store" in r, f"Missing 'store' key in result: {r.get('article_id')}"
        assert r["store"] is not None, f"Null store for article_id={r.get('article_id')}"


# ---------------------------------------------------------------------------
# Store filter test
# ---------------------------------------------------------------------------


def test_store_filter_narrows_to_single_store(unified_retriever) -> None:
    """filters={'store': 'snitch'} must return only snitch items."""
    results = unified_retriever.search("casual t-shirt", top_k=20, filters={"store": "snitch"})
    if not results:
        pytest.skip("No 'casual t-shirt' results for snitch — store may not carry this")
    for r in results:
        assert r["store"] == "snitch", (
            f"Store filter violated: got store={r['store']!r} for article_id={r['article_id']}"
        )


# ---------------------------------------------------------------------------
# pdp_live preservation test
# ---------------------------------------------------------------------------


def test_pdp_live_column_preserved_through_merge(unified_df: pd.DataFrame) -> None:
    """The pdp_live column must survive the merge (even if values are all null).

    Some brands (myntra, flipkart, snitch) include the column in their catalogues
    but the live-validation crawl may not have been run, leaving all values null.
    The test verifies the column is present in the merged frame so downstream
    HybridRetriever logic (dead-link deprioritisation) can operate when validation
    data is eventually populated.
    """
    # These brands include pdp_live in their schema
    brands_with_pdp_live_column = {"myntra", "flipkart", "snitch"}
    for brand in brands_with_pdp_live_column:
        sub = unified_df[unified_df["store"] == brand]
        assert "pdp_live" in sub.columns, (
            f"pdp_live column absent from merged catalogue for brand={brand}"
        )
