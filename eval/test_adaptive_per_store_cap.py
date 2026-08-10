"""Unit tests for the query-adaptive cap formula in adaptive_per_store_cap_test.py.

Pure-function tests only (no retriever/index needed) — covers the degeneracy
cases the module docstring claims the formula satisfies.
"""
from __future__ import annotations

from eval.adaptive_per_store_cap_test import compute_effective_cap


def test_single_store_cap_becomes_top_k() -> None:
    """n_stores=1 -> cap is effectively irrelevant (>= top_k)."""
    assert compute_effective_cap(n_stores=1, top_k=50, base_cap=4) == 50


def test_many_stores_falls_back_to_base_cap() -> None:
    """n_stores >= top_k/base_cap -> cap == base_cap, identical to today."""
    # top_k/base_cap = 50/4 = 12.5; at n_stores=13 the formula should floor
    # back to base_cap exactly.
    assert compute_effective_cap(n_stores=13, top_k=50, base_cap=4) == 4
    assert compute_effective_cap(n_stores=50, top_k=50, base_cap=4) == 4


def test_intermediate_n_stores_relaxes_cap_partially() -> None:
    """A moderately concentrated pool (n_stores=10) gets a cap between
    base_cap and top_k, not a hard jump to either extreme."""
    result = compute_effective_cap(n_stores=10, top_k=50, base_cap=4)
    assert 4 < result < 50
    assert result == 5  # ceil(50/10) == 5


def test_empty_pool_falls_back_to_base_cap() -> None:
    """n_stores<=0 (empty full_pool) -> base_cap, avoids division by zero."""
    assert compute_effective_cap(n_stores=0, top_k=50, base_cap=4) == 4


def test_maroon_dupatta_case_relaxes_cap() -> None:
    """The diagnosed worst case (reports/recall_gap_diagnosis_20260806.md):
    3 distinct stores (myntra/libas/vastramay) in a 55-item, 95%-myntra-
    concentrated universe -> cap should relax well above base_cap=4."""
    result = compute_effective_cap(n_stores=3, top_k=50, base_cap=4)
    assert result == 17  # ceil(50/3) == 17
    assert result > 4
