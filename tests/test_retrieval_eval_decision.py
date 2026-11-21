"""Pre-registered decision rule and the label-scheme robustness recompute (pure, synthetic)."""

from __future__ import annotations

from typing import Any

from evals.retrieval import decision as dc
from evals.retrieval import robustness as rb

IDEAL = ["a1", "a2", "a3", "a4", "a5"]
BAD = ["a5", "a4", "a3", "a2", "a1"]
LABELS = {"a1": 2, "a2": 2, "a3": 1, "a4": 0, "a5": 0, "a6": 1}


def _agg(n: int = 12) -> dict[str, dict[str, Any]]:
    return {
        f"q{i}:{a}": {"label": float(v), "tie": False, "n": 3}
        for i in range(n)
        for a, v in LABELS.items()
    }


def _rows(ids: list[str], tokens: int, latency: int, n: int = 12) -> list[dict[str, Any]]:
    return [
        {
            "query_id": f"q{i}",
            "final_ids": ids,
            "prompt_tokens": tokens,
            "completion_tokens": 0,
            "latency_ms": latency,
        }
        for i in range(n)
    ]


def test_cheapest_admissible_wins_and_a_clearly_worse_cheap_arm_does_not() -> None:
    rows = {
        "A": _rows(IDEAL, 2000, 1100),
        "A2": _rows(IDEAL, 2200, 1200),
        "B": _rows(BAD, 0, 0),  # free but far below best
        "C": _rows(IDEAL, 600, 900),  # as good as best and cheapest
    }
    res = dc.apply_rule(rows, _agg())
    assert res["best"] == "A"  # tie on the point estimate resolves to the earlier arm
    assert res["variants"]["B"]["admissible"] is False and res["variants"]["B"]["ci_hi"] < 0
    assert res["chosen"] == "C"
    assert res["variants"]["C"]["tokens_per_turn"] == 600


def test_zero_cost_baseline_wins_when_within_the_margin() -> None:
    rows = {a: _rows(IDEAL, 1000, 1000) for a in ("A", "A2", "C")}
    rows["B"] = _rows(IDEAL, 0, 0)
    res = dc.apply_rule(rows, _agg())
    assert res["chosen"] == "B"  # B costs nothing and is admissible


def test_latency_breaks_a_token_tie() -> None:
    rows = {
        "A": _rows(IDEAL, 500, 1500),
        "A2": _rows(IDEAL, 500, 800),
        "B": _rows(BAD, 0, 0),
        "C": _rows(BAD, 500, 100),
    }
    assert dc.apply_rule(rows, _agg())["chosen"] == "A2"


def test_schemes_claude_only_has_no_ties_and_two_judge_split_is_a_median_tie() -> None:
    labels = {
        "gpt-oss-120b": {"q:a": 0, "q:b": 2},
        "qwen3.8-27b": {"q:a": 1, "q:b": 2},
        "claude": {"q:a": 2, "q:b": 2},
    }
    only = rb.scheme_agg(labels, rb.SCHEMES["claude_only"])
    assert only["q:a"]["label"] == 2.0 and not only["q:a"]["tie"]
    no_gpt = rb.scheme_agg(labels, rb.SCHEMES["no_gpt_oss"])
    assert no_gpt["q:a"] == {"label": 1.5, "tie": True, "n": 2}  # 1 vs 2 -> median 1.5
    assert no_gpt["q:b"]["label"] == 2.0 and not no_gpt["q:b"]["tie"]
    full = rb.scheme_agg(labels, rb.SCHEMES["consensus"])
    assert full["q:a"]["tie"] is True and full["q:a"]["label"] == 1.0  # 0/1/2 -> median 1
