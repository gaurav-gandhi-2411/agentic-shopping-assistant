"""Unit tests for evals.retrieval.agreement (kappa, alpha, aggregation, calibration gate)."""

from __future__ import annotations

import math

import pytest

from evals.retrieval import agreement as ag


def _ref_alpha(units: list[list[int]]) -> float:
    """Independent pure-python ordinal alpha: pairwise definition over all values."""
    n_g = {g: 0.0 for g in (0, 1, 2)}
    pairs = []  # (value, value, weight) ordered within-unit pairs
    for u in units:
        m = len(u)
        for i in range(m):
            for j in range(m):
                if i != j:
                    pairs.append((u[i], u[j], 1.0 / (m - 1)))
                    n_g[u[i]] += 1.0 / (m - 1)

    def d2(c: int, k: int) -> float:
        if c == k:
            return 0.0
        lo, hi = min(c, k), max(c, k)
        return (sum(n_g[g] for g in range(lo, hi + 1)) - (n_g[lo] + n_g[hi]) / 2) ** 2

    n = sum(n_g.values())
    d_o = sum(w * d2(c, k) for c, k, w in pairs) / n
    d_e = sum(n_g[c] * n_g[k] * d2(c, k) for c in n_g for k in n_g) / (n * (n - 1))
    return 1 - d_o / d_e


def test_kappa_hand_computed() -> None:
    a, b = [0, 0, 1, 1], [0, 1, 1, 1]
    assert ag.cohen_kappa(a, b) == pytest.approx(0.5)  # po=.75, pe=.5 -> (.75-.5)/.5


def test_kappa_perfect_and_chance_and_weighted() -> None:
    x = [0, 1, 2, 0, 1, 2]
    assert ag.cohen_kappa(x, x) == pytest.approx(1.0)
    assert ag.cohen_kappa(x, x, "quadratic") == pytest.approx(1.0)
    # off-by-one disagreement costs less than off-by-two under quadratic weights
    near = ag.cohen_kappa([0, 1, 2, 2], [1, 1, 2, 2], "quadratic")
    far = ag.cohen_kappa([0, 1, 2, 2], [2, 1, 2, 2], "quadratic")
    assert near > far
    assert ag.cohen_kappa([0, 1, 2], [1, 1, 1]) == pytest.approx(0.0)  # constant judge: no skill
    assert math.isnan(ag.cohen_kappa([1, 1, 1], [1, 1, 1]))  # chance agreement is total: undefined


def test_kappa_quadratic_hand_value() -> None:
    # a=[0,1,2,2], b=[0,1,1,2]: observed weighted disagreement = (1/4)*(1/4)=1/16 (one off-by-one)
    a, b = [0, 1, 2, 2], [0, 1, 1, 2]
    exp_dis = 0.0
    n = 4
    for i, ca in enumerate((1, 1, 2)):  # a counts per label 0,1,2
        for j, cb in enumerate((1, 2, 1)):  # b counts
            exp_dis += (ca * cb / n) * ((i - j) ** 2 / 4.0)
    expected = 1 - (1 / 4 * 1 / 4) / (exp_dis / n)
    assert ag.cohen_kappa(a, b, "quadratic") == pytest.approx(expected)


def test_alpha_binary_known_value() -> None:
    units = [[0, 0], [0, 1], [1, 1], [1, 1]]
    assert ag.krippendorff_alpha_ordinal(units) == pytest.approx(0.53333, abs=1e-4)


def test_alpha_matches_reference_and_perfect() -> None:
    units = [[0, 1, 1], [2, 2, 1], [0, 0, 0], [1, 2, 2], [2, 2, 2], [0, 1, 2]]
    assert ag.krippendorff_alpha_ordinal(units) == pytest.approx(_ref_alpha(units))
    perfect = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    assert ag.krippendorff_alpha_ordinal(perfect) == pytest.approx(1.0)


def test_alpha_ignores_missing_and_ci_brackets_point() -> None:
    units = [[0, 0, None], [1, 1, 1], [2, 1, 2], [0, 1, 0], [2, 2, 2], [1, 0, 1]] * 5
    a = ag.krippendorff_alpha_ordinal(units)
    lo, hi = ag.bootstrap_alpha_ci(units, n_boot=300)
    assert lo <= a <= hi


def test_aggregate_majority_tie_and_median() -> None:
    assert ag.aggregate_item([2, 2, 1]) == {"label": 2.0, "tie": False, "n": 3}
    three_way = ag.aggregate_item([0, 1, 2])
    assert three_way["tie"] is True and three_way["label"] == 1.0
    out = ag.aggregate({"a": {"x": 0, "y": 1}, "b": {"x": 0, "y": 2}, "c": {"x": 1, "y": 0}})
    assert out["x"]["label"] == 0.0 and not out["x"]["tie"]
    assert out["y"]["tie"] and out["y"]["label"] == 1.0


def test_calibration_gate_pass_and_fail_closed() -> None:
    good = {j: {str(i): v for i, v in enumerate([0, 1, 2, 0, 1, 2, 2, 0])} for j in "abc"}
    ok, msg = ag.calibration_gate(good)
    assert ok and "passed" in msg
    bad = dict(good)
    bad["c"] = {str(i): v for i, v in enumerate([2, 0, 0, 1, 2, 1, 0, 2])}
    ok, msg = ag.calibration_gate(bad)
    assert not ok and "FAILED" in msg and "confusion" in msg
    assert not ag.calibration_gate({"a": {"1": 0}})[0]  # single judge: fail closed
    nan_case = {"a": {"1": 1, "2": 1, "3": 1}, "b": {"1": 1, "2": 1, "3": 1}}
    assert not ag.calibration_gate(nan_case)[0]  # NaN kappa fails closed


def test_pairwise_report_shape() -> None:
    jl = {"a": {"1": 0, "2": 1, "3": 2, "4": 2}, "b": {"1": 0, "2": 1, "3": 1, "4": 2}}
    rep = ag.pairwise_report(jl)
    pair = rep["pairs"]["a|b"]
    assert pair["n"] == 4 and sum(map(sum, pair["confusion"])) == 4
    assert rep["distributions"]["a"] == {"0": 1, "1": 1, "2": 2}


def test_calibration_gate_threshold_is_0_4() -> None:
    """A weakly positive kappa (0 < k < 0.4) must still stop the run."""
    a = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2]
    b = [0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 1, 2]
    weak = {"a": dict(enumerate(map(int, a))), "b": dict(enumerate(map(int, b)))}
    weak = {j: {str(k): v for k, v in lab.items()} for j, lab in weak.items()}
    assert 0 < ag.cohen_kappa(a, b) < 0.4
    ok, msg = ag.calibration_gate(weak)
    assert not ok and "FAILED" in msg
    c = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2]
    d = [0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 2, 2]  # 10/12 agreement: kappa well above 0.4
    assert ag.cohen_kappa(c, d) > 0.4
    good = {"a": {str(i): v for i, v in enumerate(c)}, "b": {str(i): v for i, v in enumerate(d)}}
    assert ag.calibration_gate(good)[0]
