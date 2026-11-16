"""Metric and bootstrap helpers of scripts/eval_hygiene_ab.py (no index, no network)."""

from __future__ import annotations

import math

import numpy as np

from scripts import eval_hygiene_ab as ab


def _rows(labels: list[int | None]) -> list[dict]:
    return [{"id": "q", "top": [{"rank": i + 1, "label": lab} for i, lab in enumerate(labels)]}]


def test_counts_p5_and_ndcg_treat_unlabelled_as_zero_gain_but_not_as_a_miss() -> None:
    c = ab._per_query_counts(_rows([1, 0, None, 1, None]))
    assert c[0].tolist()[:3] == [2.0, 1.0, 2.0]
    assert ab._p5(c) == 2 / 3  # unlabelled items are excluded from the denominator
    dcg = 1 / math.log2(2) + 1 / math.log2(5)
    assert math.isclose(ab._ndcg5(c), dcg / ab._IDCG5)


def test_perfect_top5_has_ndcg_one() -> None:
    assert math.isclose(ab._ndcg5(ab._per_query_counts(_rows([1, 1, 1, 1, 1]))), 1.0)


def test_summarise_is_deterministic_and_paired_difference_is_exact_for_identical_arms() -> None:
    rng = np.random.default_rng(0)
    base = np.column_stack([rng.integers(0, 6, 30), rng.integers(0, 6, 30), np.zeros(30), np.zeros(30)])
    arms = {"A": base, "B": base.copy()}
    s1 = ab._summarise("p5", ab._p5, arms, 200, 42)
    s2 = ab._summarise("p5", ab._p5, arms, 200, 42)
    assert s1 == s2  # seed 42 -> identical
    assert s1["diff"]["B-A"]["point"] == 0.0
    assert s1["diff"]["B-A"]["ci95"] == [0.0, 0.0]  # same data under the same resample indices
