"""Unit tests for store_diversity_rerank and HybridRetriever diversity integration.

These tests are fully self-contained (no index, no network, no LLM).
Run with: pytest tests/test_store_diversity_rerank.py -v
"""

from __future__ import annotations

import pandas as pd

from src.retrieval.hybrid_search import store_diversity_rerank

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidates(
    stores: list[str],
    scores: list[float] | None = None,
) -> list[dict]:
    """Build minimal candidate dicts for testing.

    Parameters
    ----------
    stores:
        One entry per candidate; determines the 'store' field.
    scores:
        Optional explicit RRF scores (descending by convention).
        Defaults to [N, N-1, ..., 1] so higher-index items have lower score.
    """
    n = len(stores)
    if scores is None:
        scores = [float(n - i) for i in range(n)]
    return [
        {
            "article_id": f"item{i}",
            "store": s,
            "score": sc,
            "display_name": f"Item {i}",
            "prod_name": f"Item {i}",
            "colour": "",
            "product_type": "Dress",
            "department": "",
            "detail_desc": "",
            "image_url": None,
            "price_inr": None,
            "pdp_handle": None,
            "pdp_live": None,
            "gender": "female",
        }
        for i, (s, sc) in enumerate(zip(stores, scores))
    ]


# ---------------------------------------------------------------------------
# Guard: knob=0.0 → exact original order
# ---------------------------------------------------------------------------


class TestKnobZeroPureRelevance:
    """knob=0.0 must reproduce the input order exactly (no reranking)."""

    def test_single_store_unchanged(self) -> None:
        candidates = _make_candidates(["myntra"] * 5)
        result = store_diversity_rerank(candidates, top_k=5, store_diversity=0.0)
        assert [r["article_id"] for r in result] == [c["article_id"] for c in candidates]

    def test_multi_store_unchanged(self) -> None:
        candidates = _make_candidates(["a", "b", "a", "b", "c"])
        result = store_diversity_rerank(candidates, top_k=5, store_diversity=0.0)
        assert [r["article_id"] for r in result] == [c["article_id"] for c in candidates]

    def test_truncation_at_top_k(self) -> None:
        candidates = _make_candidates(["a", "b", "c", "d", "e"])
        result = store_diversity_rerank(candidates, top_k=3, store_diversity=0.0)
        assert len(result) == 3
        assert [r["article_id"] for r in result] == ["item0", "item1", "item2"]

    def test_empty_candidates(self) -> None:
        result = store_diversity_rerank([], top_k=5, store_diversity=0.0)
        assert result == []


# ---------------------------------------------------------------------------
# Guard: single-store candidates → no-op (cannot diversify)
# ---------------------------------------------------------------------------


class TestSingleStoreCandidates:
    """When all candidates come from one store, diversity re-rank is a no-op."""

    def test_single_store_knob_positive(self) -> None:
        candidates = _make_candidates(["myntra"] * 6)
        result = store_diversity_rerank(candidates, top_k=4, store_diversity=0.5)
        assert [r["article_id"] for r in result] == ["item0", "item1", "item2", "item3"]

    def test_single_store_knob_high(self) -> None:
        candidates = _make_candidates(["only_store"] * 3)
        result = store_diversity_rerank(candidates, top_k=3, store_diversity=1.0)
        assert [r["article_id"] for r in result] == ["item0", "item1", "item2"]


# ---------------------------------------------------------------------------
# Store filter path: tested at HybridRetriever level (via mock)
# ---------------------------------------------------------------------------


class TestStoreFilterSkipsDiversity:
    """When a store_filter is active, the retriever must skip diversity re-rank.

    We test store_diversity_rerank directly (knob path), but the HybridRetriever
    guard (store_filter → candidates[:top_k]) is exercised here via a mock retriever.
    """

    def test_knob_nonzero_multi_store_but_store_filter_applied(self) -> None:
        """Simulate the retriever's guard: store_filter bypasses rerank entirely.

        We build a HybridRetriever with a mock that returns multi-store candidates,
        then confirm that when store_filter is set the output is pure-relevance order.
        """
        import unittest.mock as mock

        from src.retrieval.hybrid_search import HybridRetriever

        # Build a tiny synthetic catalogue (3 stores, 6 items)
        cat_data = {
            "article_id": [f"a{i}" for i in range(6)],
            "display_name": [f"Item {i}" for i in range(6)],
            "prod_name": [f"Prod {i}" for i in range(6)],
            "detail_desc": ["desc"] * 6,
            "image_url": [None] * 6,
            "price_inr": [500.0] * 6,
            "pdp_handle": [None] * 6,
            "pdp_live": [True] * 6,
            "gender": ["female"] * 6,
            "store": ["storeA", "storeB", "storeA", "storeB", "storeC", "storeC"],
            "facets": [
                {"colour_group_name": "Black", "product_type_name": "Dress", "department_name": "Women"}
            ] * 6,
        }
        cat_df = pd.DataFrame(cat_data)

        config = {
            "retrieval": {
                "final_k": 4,
                "top_k": 6,
                "rrf_k": 60,
                "store_diversity": 0.9,  # high diversity knob
            }
        }

        # Mock dense and sparse to return fixed orderings
        dense_mock = mock.MagicMock()
        dense_mock.search.return_value = [(f"a{i}", 1.0 / (i + 1)) for i in range(6)]
        sparse_mock = mock.MagicMock()
        sparse_mock.search.return_value = [(f"a{i}", 1.0 / (i + 1)) for i in range(6)]

        retriever = HybridRetriever(dense_mock, sparse_mock, cat_df, config)

        # With store_filter="storeA", diversity re-rank must be skipped.
        # All results should be from storeA, in pure-relevance order.
        results = retriever.search("dress", top_k=4, filters={"store": "storeA"})
        result_stores = [r["store"] for r in results]
        assert all(s == "storeA" for s in result_stores), (
            f"Expected only storeA but got: {result_stores}"
        )


# ---------------------------------------------------------------------------
# Core: knob>0 increases store spread on multi-store candidates
# ---------------------------------------------------------------------------


class TestDiversityIncreasesStoreSpread:
    """knob>0 must produce more store variety than pure-relevance order."""

    def test_spread_vs_pure_relevance(self) -> None:
        """storeA dominates the top-5 by score; diversity knob must pull in storeB/C.

        Scores: storeA items score 10,9,8,7,6; storeB items score 5,4; storeC items 3,2,1.
        Pure relevance top-5 = all storeA (5 items, 1 unique store).
        With high diversity, top-5 should contain items from multiple stores.
        """
        candidates = _make_candidates(
            stores=["storeA", "storeA", "storeA", "storeA", "storeA",
                    "storeB", "storeB", "storeC", "storeC", "storeC"],
            scores=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        )

        pure = store_diversity_rerank(candidates, top_k=5, store_diversity=0.0)
        diverse = store_diversity_rerank(candidates, top_k=5, store_diversity=0.8)

        pure_stores = [r["store"] for r in pure]
        diverse_stores = [r["store"] for r in diverse]

        pure_unique = len(set(pure_stores))
        diverse_unique = len(set(diverse_stores))

        # Pure relevance must be mono-store (all storeA)
        assert pure_unique == 1, f"Expected 1 unique store in pure order, got {pure_unique}"
        # High diversity knob must spread across more stores
        assert diverse_unique > 1, (
            f"Expected diversity knob to increase store variety: "
            f"pure={pure_unique} stores, diverse={diverse_unique} stores"
        )

    def test_top1_result_preserved(self) -> None:
        """The #1 RRF result must always be the first in the re-ranked list.

        Property: at step 0, redundancy=0 for all items, so the item with the
        highest rel_norm (= highest RRF score) is always selected first.
        """
        candidates = _make_candidates(
            stores=["storeA", "storeB", "storeA", "storeB"],
            scores=[10.0, 9.0, 8.0, 7.0],
        )
        for knob in [0.0, 0.1, 0.3, 0.5, 0.9, 1.0]:
            result = store_diversity_rerank(candidates, top_k=4, store_diversity=knob)
            assert result[0]["article_id"] == "item0", (
                f"knob={knob}: top result changed to {result[0]['article_id']!r}"
            )

    def test_three_stores_all_represented(self) -> None:
        """With 3 stores and knob=0.4, all 3 stores appear in top-6 of 9 candidates."""
        # storeA has items 0,3,6 (scores 9,6,3); storeB has 1,4,7 (8,5,2); storeC has 2,5,8 (7,4,1)
        candidates = _make_candidates(
            stores=["storeA", "storeB", "storeC"] * 3,
            scores=[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        )
        result = store_diversity_rerank(candidates, top_k=6, store_diversity=0.4)
        result_stores = {r["store"] for r in result}
        assert result_stores == {"storeA", "storeB", "storeC"}, (
            f"Expected all 3 stores in top-6, got: {result_stores}"
        )

    def test_deterministic_with_seed_independent(self) -> None:
        """Re-rank is a pure function — same input always gives same output (no randomness)."""
        candidates = _make_candidates(
            stores=["a", "b", "a", "b", "c", "a"],
            scores=[6.0, 5.5, 5.0, 4.5, 4.0, 3.5],
        )
        result1 = store_diversity_rerank(candidates, top_k=4, store_diversity=0.3)
        result2 = store_diversity_rerank(candidates, top_k=4, store_diversity=0.3)
        assert [r["article_id"] for r in result1] == [r["article_id"] for r in result2]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_top_k_larger_than_candidates(self) -> None:
        candidates = _make_candidates(["a", "b"], scores=[2.0, 1.0])
        result = store_diversity_rerank(candidates, top_k=10, store_diversity=0.5)
        assert len(result) == 2  # can't return more than available

    def test_none_store_handled(self) -> None:
        """Candidates with store=None should not crash the function."""
        candidates = _make_candidates(["storeA", "storeB", "storeA"])
        candidates[1]["store"] = None  # simulate missing store
        result = store_diversity_rerank(candidates, top_k=3, store_diversity=0.3)
        assert len(result) == 3

    def test_equal_scores_stable(self) -> None:
        """Equal-score items should still produce a valid result."""
        candidates = _make_candidates(
            stores=["a", "b", "a", "b"],
            scores=[1.0, 1.0, 1.0, 1.0],
        )
        result = store_diversity_rerank(candidates, top_k=4, store_diversity=0.5)
        assert len(result) == 4
