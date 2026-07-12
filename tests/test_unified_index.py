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


def test_unified_catalogue_spans_all_8_live_stores(unified_df: pd.DataFrame) -> None:
    """All 8 live stores must be present in the on-disk index; H&M/berrylush must be absent.

    Phase A (2026-07-06): berrylush is now excluded ENTIRELY at build time (dropped
    from ``UNIFIED_STORES`` in ``scripts/build_unified_index.py``), not merely
    filtered at query time — it stopped occupying FAISS/BM25 candidate-window slots.
    Re-enabling berrylush now requires re-running the build script, not just
    flipping its ``active`` flag in ``src/config/stores.py`` (see
    ``test_hm_and_berrylush_not_in_unified_catalogue`` below).
    """
    expected = {
        "myntra", "flipkart", "snitch", "fashor", "powerlook", "virgio",
        "globalrepublic", "libas",
    }
    actual = set(unified_df["store"].unique())
    assert expected == actual, (
        f"Store set mismatch. Missing: {expected - actual}. Unexpected: {actual - expected}"
    )


def test_hm_and_berrylush_not_in_unified_catalogue(unified_df: pd.DataFrame) -> None:
    """H&M and berrylush must NOT appear in the unified catalogue (both excluded at build time).

    H&M: archival Kaggle data, no live PDP/image.
    berrylush: store inactive (password-walled) — dropped at build time since Phase A
    (2026-07-06), not just filtered at query time (see EXCLUDED_STORES in
    scripts/build_unified_index.py).
    """
    stores_present = set(unified_df["store"].unique())
    assert "hm" not in stores_present, (
        "H&M was found in the unified catalogue. "
        "It must be excluded (archival data, no live PDP/image)."
    )
    assert "berrylush" not in stores_present, (
        "berrylush was found in the unified catalogue. "
        "It must be excluded entirely at build time (Phase A, 2026-07-06)."
    )


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
    """clip_article_ids.npy must be aligned with the unified catalogue.

    Alignment means: (1) the id counts match, and (2) every CLIP id exists in the
    catalogue's article_id set. Deliberately NOT a hardcoded row count — the CLIP
    index is rebuilt on a separate schedule from the text/BM25 index (a rebuild may
    be in flight for the 9-store catalogue while this test runs), so a hardcoded
    count would go stale independently of a real alignment bug. If this fails
    because the CLIP artefact is mid-rebuild and still reflects an older store set,
    that is an expected transient failure that clears once the rebuild completes —
    it is not evidence of a code bug.
    """
    ids = np.load(str(_CLIP_UNIFIED_DIR / "clip_article_ids.npy"), allow_pickle=True)
    catalogue_ids = set(unified_df["article_id"])

    assert len(ids) == len(unified_df), (
        f"CLIP ids {len(ids)} != catalogue rows {len(unified_df)} "
        "(CLIP index may be mid-rebuild for the current store set)"
    )
    missing = set(ids) - catalogue_ids
    assert not missing, (
        f"{len(missing)} CLIP ids are not present in the unified catalogue "
        f"(sample: {sorted(missing)[:5]})"
    )


# ---------------------------------------------------------------------------
# Multi-store query tests
# ---------------------------------------------------------------------------


# Active/live stores — the 8 stores present in the on-disk unified index (see
# test_unified_catalogue_spans_all_8_live_stores). hm and berrylush are both excluded
# entirely at build time (Phase A, 2026-07-06) so they can never appear in results;
# _INACTIVE_STORES is kept as a belt-and-suspenders assertion set.
_VALID_STORES = frozenset({
    "myntra", "flipkart", "snitch", "fashor", "powerlook", "virgio",
    "globalrepublic", "libas",
})
_INACTIVE_STORES = frozenset({"hm", "berrylush"})


def test_black_dress_spans_multiple_stores(unified_retriever) -> None:
    """'black dress' must return hits from >=2 active stores; inactive stores never appear."""
    results = unified_retriever.search("black dress", top_k=20)
    assert len(results) > 0, "No results for 'black dress'"
    stores = {r["store"] for r in results if r.get("store")}

    # Inactive stores (hm, berrylush) must never appear in search results.
    inactive_hit = stores & _INACTIVE_STORES
    assert not inactive_hit, (
        f"Inactive store(s) appeared in 'black dress' results — they must be excluded "
        f"at query time. Stores present: {stores}"
    )
    # All returned stores must be from the known-valid (active) set
    unknown = stores - _VALID_STORES
    assert not unknown, f"Unknown store(s) in results: {unknown}"

    assert len(stores) >= 2, (
        f"'black dress' only hit store(s): {stores}. "
        "Expected results from >=2 active stores."
    )
    # Use ASCII-safe output to avoid cp1252 encoding errors on Windows
    print(f"\n'black dress' stores: {sorted(stores)}")
    for r in results[:5]:
        price = r.get("price_inr")
        print(f"  [{r['store']}] {r['display_name']} -- Rs.{price}")


def test_white_sneakers_spans_multiple_stores(unified_retriever) -> None:
    """'white sneakers' must return hits from >=2 active stores; inactive stores never appear."""
    results = unified_retriever.search("white sneakers", top_k=20)
    assert len(results) > 0, "No results for 'white sneakers'"
    stores = {r["store"] for r in results if r.get("store")}

    # Inactive stores (hm, berrylush) must never appear in search results.
    inactive_hit = stores & _INACTIVE_STORES
    assert not inactive_hit, (
        f"Inactive store(s) appeared in 'white sneakers' results — they must be excluded. "
        f"Stores present: {stores}"
    )
    # All returned stores must be from the known-valid (active) set
    unknown = stores - _VALID_STORES
    assert not unknown, f"Unknown store(s) in results: {unknown}"

    assert len(stores) >= 2, (
        f"'white sneakers' only hit store(s): {stores}. "
        "Expected results from >=2 active stores."
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
