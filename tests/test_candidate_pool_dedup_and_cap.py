"""Unit tests for hybrid_search's pre-rerank candidate dedup and per-store cap.

Phase A index-quality task (2026-07-06). Fully self-contained (no index, no
network, no LLM) — follows the mock/fixture pattern in
tests/test_store_diversity_rerank.py.
"""

from __future__ import annotations

from src.retrieval.hybrid_search import apply_per_store_cap, dedup_candidates_keep_cheapest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item(
    article_id: str,
    prod_name: str,
    colour: str = "",
    store: str = "storeA",
    price_inr: float | None = None,
) -> dict:
    return {
        "article_id": article_id,
        "prod_name": prod_name,
        "display_name": prod_name,
        "colour": colour,
        "store": store,
        "price_inr": price_inr,
    }


# ---------------------------------------------------------------------------
# dedup_candidates_keep_cheapest
# ---------------------------------------------------------------------------


class TestDedupCandidatesKeepCheapest:
    def test_no_duplicates_returns_unchanged(self) -> None:
        candidates = [
            _item("a1", "Black Dress", "black", price_inr=999),
            _item("a2", "Blue Jeans", "blue", price_inr=1499),
        ]
        result = dedup_candidates_keep_cheapest(candidates)
        assert [c["article_id"] for c in result] == ["a1", "a2"]

    def test_duplicate_by_name_and_colour_keeps_cheapest(self) -> None:
        candidates = [
            _item("a1", "Gyda Slim Trousers", "black", price_inr=1999),
            _item("a2", "Gyda Slim Trousers", "black", price_inr=1299),
        ]
        result = dedup_candidates_keep_cheapest(candidates)
        assert len(result) == 1
        assert result[0]["article_id"] == "a2"

    def test_group_positioned_at_first_occurrence(self) -> None:
        """The surviving item stays at the group's first-seen rank, even if a
        cheaper duplicate appears later in the pool."""
        candidates = [
            _item("a1", "Gyda Slim Trousers", "black", price_inr=1999),
            _item("b1", "Other Item", "red", price_inr=500),
            _item("a2", "Gyda Slim Trousers", "black", price_inr=1299),
        ]
        result = dedup_candidates_keep_cheapest(candidates)
        assert [c["article_id"] for c in result] == ["a2", "b1"]

    def test_different_colours_not_deduped(self) -> None:
        candidates = [
            _item("a1", "Gyda Slim Trousers", "black"),
            _item("a2", "Gyda Slim Trousers", "blue"),
        ]
        result = dedup_candidates_keep_cheapest(candidates)
        assert len(result) == 2

    def test_none_price_never_displaces_priced_item(self) -> None:
        candidates = [
            _item("a1", "Gyda Slim Trousers", "black", price_inr=999),
            _item("a2", "Gyda Slim Trousers", "black", price_inr=None),
        ]
        result = dedup_candidates_keep_cheapest(candidates)
        assert len(result) == 1
        assert result[0]["article_id"] == "a1"

    def test_empty_input(self) -> None:
        assert dedup_candidates_keep_cheapest([]) == []


# ---------------------------------------------------------------------------
# apply_per_store_cap
# ---------------------------------------------------------------------------


class TestApplyPerStoreCap:
    def test_cap_zero_or_none_is_noop(self) -> None:
        selected = [_item(f"a{i}", f"Item {i}", store="storeA") for i in range(5)]
        assert apply_per_store_cap(selected, selected, 0, 5) == selected
        assert apply_per_store_cap(selected, selected, None, 5) == selected  # type: ignore[arg-type]

    def test_enforces_cap_per_store(self) -> None:
        selected = [
            _item("a1", "Item 1", store="storeA"),
            _item("a2", "Item 2", store="storeA"),
            _item("a3", "Item 3", store="storeA"),
            _item("b1", "Item 4", store="storeB"),
        ]
        result = apply_per_store_cap(selected, selected, cap=2, top_k=4)
        stores = [r["store"] for r in result]
        assert stores.count("storeA") == 2
        assert stores.count("storeB") == 1
        assert len(result) == 3  # third storeA item dropped, no backfill available

    def test_backfills_from_full_pool_to_reach_top_k(self) -> None:
        selected = [
            _item("a1", "Item 1", store="storeA"),
            _item("a2", "Item 2", store="storeA"),
            _item("a3", "Item 3", store="storeA"),
        ]
        full_pool = selected + [
            _item("b1", "Item 4", store="storeB"),
            _item("c1", "Item 5", store="storeC"),
        ]
        result = apply_per_store_cap(selected, full_pool, cap=2, top_k=3)
        assert len(result) == 3
        stores = [r["store"] for r in result]
        assert stores.count("storeA") == 2
        # Backfilled from storeB (first non-selected item in full_pool under cap)
        assert "b1" in [r["article_id"] for r in result]

    def test_under_cap_pool_unaffected(self) -> None:
        selected = [
            _item("a1", "Item 1", store="storeA"),
            _item("b1", "Item 2", store="storeB"),
        ]
        result = apply_per_store_cap(selected, selected, cap=4, top_k=2)
        assert result == selected

    def test_never_readds_already_kept_item(self) -> None:
        selected = [_item("a1", "Item 1", store="storeA")]
        full_pool = selected
        result = apply_per_store_cap(selected, full_pool, cap=4, top_k=5)
        assert len(result) == 1
