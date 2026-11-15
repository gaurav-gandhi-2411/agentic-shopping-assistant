"""Tests for evals.retrieval.metrics (nDCG, precision, bootstrap, paired tests)."""

from __future__ import annotations

import math

import pytest

from evals.retrieval import metrics as mt


def test_ndcg_hand_values() -> None:
    pool = [2, 2, 1, 0, 0, 0]
    ideal = (2**2 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
    assert mt.ndcg_at_k([2, 2, 1, 0, 0], pool) == pytest.approx(1.0)
    shown = [1, 0, 2, 0, 2]
    dcg = 1 / math.log2(2) + 3 / math.log2(4) + 3 / math.log2(6)
    assert mt.ndcg_at_k(shown, pool) == pytest.approx(dcg / ideal)
    assert math.isnan(mt.ndcg_at_k([0, 0, 0, 0, 0], [0, 0, 0]))  # undefined, not 0


def test_precision_thresholds_and_missing_slots() -> None:
    shown = [2, 1, 0, 2, 1]
    assert mt.precision_at_k(shown, 1) == pytest.approx(0.8)
    assert mt.precision_at_k(shown, 2) == pytest.approx(0.4)
    assert mt.precision_at_k([2, 2], 1) == pytest.approx(0.4)  # 2 shown of 5 slots


def test_bootstrap_deterministic_and_brackets_mean() -> None:
    vals = [0.2, 0.5, 0.9, 0.4, 0.6, 0.7, 0.3, 0.8] * 5
    a = mt.bootstrap_mean_ci(vals, n_boot=2000)
    b = mt.bootstrap_mean_ci(vals, n_boot=2000)
    assert a == b and a[1] <= a[0] <= a[2]
    assert a[0] == pytest.approx(sum(vals) / len(vals))
    assert mt.bootstrap_mean_ci([1.0] * 10) == (1.0, 1.0, 1.0)


def test_sign_test_exact() -> None:
    assert mt.sign_test_p([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)  # 5 of 5 positive
    assert mt.sign_test_p([1, -1, 1, -1]) == pytest.approx(1.0)
    assert mt.sign_test_p([0, 0]) == 1.0


def test_wilcoxon_symmetric_and_shifted() -> None:
    assert mt.wilcoxon_p([1, -1, 2, -2, 3, -3]) > 0.9  # perfectly symmetric
    shifted = [0.3, 0.2, 0.4, 0.1, 0.5, 0.25, 0.35, 0.15, 0.45, 0.3] * 3
    assert mt.wilcoxon_p(shifted) < 0.001
    # hand check, n=6 all positive: W+=21, mean=10.5, var=22.75 -> z=(10.5-.5)/sqrt(22.75)
    z = 10.0 / math.sqrt(22.75)
    assert mt.wilcoxon_p([1, 2, 3, 4, 5, 6]) == pytest.approx(math.erfc(z / math.sqrt(2)))


def test_paired_difference_drops_nan_and_reports_sign() -> None:
    x = [0.9, 0.8, float("nan"), 0.7, 0.9, 0.6]
    y = [0.5, 0.4, 0.5, 0.3, 0.6, 0.2]
    r = mt.paired_difference(x, y)
    assert r["n"] == 5 and r["mean_diff"] == pytest.approx(0.38)
    assert r["ci_lo"] > 0 and r["sign_p"] == pytest.approx(2 / 32)


def test_usage_summary() -> None:
    rows = [
        {"prompt_tokens": 100, "completion_tokens": 10, "latency_ms": 100},
        {"prompt_tokens": 300, "completion_tokens": 30, "latency_ms": 300},
    ]
    s = mt.usage_summary(rows)
    assert s["mean_prompt_tokens"] == 200 and s["latency_p50_ms"] == 200
