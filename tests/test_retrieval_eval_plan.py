"""Token / day planning: schedule packing and the build_plan totals (pure arithmetic)."""

from __future__ import annotations

import pytest

from evals.retrieval import run


def test_schedule_packs_calls_under_the_daily_budget_without_splitting_a_call() -> None:
    days = run.schedule([("a", 5, 40.0)], budget=100)
    assert [d["calls"] for d in days] == [2, 2, 1]
    assert all(d["tokens"] <= 100 for d in days)
    assert sum(d["tokens"] for d in days) == 200


def test_schedule_uses_the_worst_case_precheck_like_the_live_runner() -> None:
    # each call costs 40 but the runner reserves prompt + completion cap (70) before calling
    days = run.schedule([("a", 3, 40.0)], budget=100, per_call_cap=70)
    assert [d["calls"] for d in days] == [1, 1, 1]  # 40 used + 70 reserved > 100


def test_schedule_keeps_task_order_and_names() -> None:
    days = run.schedule([("first", 2, 60.0), ("second", 1, 30.0)], budget=100)
    assert days[0]["work"] == {"first": 1} and days[1]["work"] == {"first": 1, "second": 1}


def _plan(reps: int, relabel: dict | None = None) -> dict:
    return run.build_plan(
        [1000] * 10, [500] * 4, [1500] * 8, [400] * 8, 150_000, 100_000, reps, relabel
    )


def test_build_plan_rerank_totals_hand_values() -> None:
    p = _plan(3)
    t = p["tokens"]
    assert t["A_all_reps"] == 8 * 3 * (1500 + 150)
    assert t["A2_all_reps"] == 8 * 3 * (1500 + 250)
    assert t["C_one_rep"] == 8 * (400 + 100)
    assert t["factors_upper_bound"] == (8 + 12) * 2 * (1500 + 250)
    assert p["n_pools_over_5_candidates"] == 8 and p["relabel_included"] is False


def test_schedule_conserves_every_planned_token_and_repeats_cost_days() -> None:
    p3, p1 = _plan(3), _plan(1)
    for p in (p1, p3):
        scheduled = sum(d["tokens"] for d in p["schedule_gpt_oss_20b"])
        planned = (
            p["tokens"]["A_all_reps"]
            + p["tokens"]["A2_all_reps"]
            + p["tokens"]["C_one_rep"]
            + p["tokens"]["factors_upper_bound"]
        )
        assert scheduled == pytest.approx(planned)
    assert p3["tokens"]["A_all_reps"] == 3 * p1["tokens"]["A_all_reps"]
    assert p3["days"]["gpt-oss-20b"] >= p1["days"]["gpt-oss-20b"] >= 2


def test_relabel_tokens_are_added_to_both_groq_judges_only() -> None:
    rel = {"expected_tokens_per_judge": 39_000, "n_calls_per_judge": 8}
    with_r, without = _plan(1, rel), _plan(1)
    for k in ("gpt-oss-120b (judge + relabel + probe)", "qwen3.8-27b (judge + relabel)"):
        assert with_r["per_model"][k] - without["per_model"][k] == 39_000
    k20 = "gpt-oss-20b (A, A2, C, factors, probe)"
    assert with_r["per_model"][k20] == without["per_model"][k20]
    assert with_r["relabel_included"] is True


def test_wall_clock_uses_the_global_8_second_spacing() -> None:
    p = _plan(1)
    w = p["wall_clock_global_1_req_per_8s"]
    assert w["judge_requests_total_both_models"] == 2 * (4 + 0 + 10)
    assert w["judge_hours_rate_limit_only"] == pytest.approx(28 * 8 / 3600)
