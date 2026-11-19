"""Distinct, counted rerank outcomes: length, parse_error, empty_selection, http_error, budget, ok.

Also the spec for the later production fix: nothing swallowed is ever called 'timeout', and an
empty `selected` with finish_reason stop is a valid 'no exact match', never a parse error."""

from __future__ import annotations

import pytest

from evals.retrieval import report as rp
from evals.retrieval import variants as vr
from evals.retrieval.groq_judge import BudgetExhaustedError, RateLimitStopError
from tests.test_retrieval_eval_variants import FakeLLM, _ids, _items

RRF5 = ["a1", "a2", "a3", "a4", "a5"]
EMPTY = '{"selected": [], "reasoning": "No red suit under 3000 available"}'


@pytest.mark.parametrize("runner", [vr.run_variant_a, vr.run_variant_a2])
def test_empty_selection_with_stop_is_a_valid_no_exact_match(runner) -> None:  # type: ignore[no-untyped-def]
    res = runner("q", _items(), FakeLLM(EMPTY, {"completion_tokens": 100}, "stop"))
    assert res["outcome"] == "empty_selection" and res["no_exact_match"] is True
    assert _ids(res) == RRF5  # retrieval order is returned
    assert res.get("length_hit", False) is False


def test_empty_list_cut_by_the_cap_is_length_not_empty_selection() -> None:
    res = vr.run_variant_a(
        "q", _items(), FakeLLM('{"selected": [], "reasoning": "No', {}, "length")
    )
    assert res["outcome"] == "length" and res["no_exact_match"] is False


def test_every_reason_is_distinct() -> None:
    cases = {
        "ok": FakeLLM('{"selected": [7, 3, 12, 1, 9]}'),
        "partial": FakeLLM('{"selected": [7, 3]}'),
        "length": FakeLLM('{"selected": [7, 3', {}, "length"),
        "parse_error": FakeLLM("Sorry, the stylist is busy right now."),
        "empty_selection": FakeLLM(EMPTY),
        "http_error": FakeLLM(RuntimeError("boom")),
        "budget": FakeLLM(BudgetExhaustedError("day budget")),
    }
    got = {k: vr.run_variant_a("q", _items(), llm)["outcome"] for k, llm in cases.items()}
    assert got == {k: k for k in cases}
    assert "timeout" not in got.values()
    assert vr.run_variant_a("q", _items(), FakeLLM(RateLimitStopError("3x429")))["outcome"] == (
        "budget"
    )


def test_selected_with_only_out_of_range_numbers_is_a_parse_error_not_no_match() -> None:
    res = vr.run_variant_a("q", _items(), FakeLLM('{"selected": [99, 100]}'))
    assert res["outcome"] == "parse_error" and res["no_exact_match"] is False


def test_variant_c_uses_the_same_taxonomy() -> None:
    assert vr.run_variant_c("q", _items(), FakeLLM("no numbers"))["outcome"] == "parse_error"
    assert vr.run_variant_c("q", _items(), FakeLLM(RuntimeError("x")))["outcome"] == "http_error"
    assert vr.run_variant_c("q", _items(), FakeLLM("7,3", {}, "length"))["outcome"] == "partial"
    assert vr.run_variant_c("q", _items(), FakeLLM("", {}, "length"))["outcome"] == "length"


def test_report_counts_each_reason_and_the_no_exact_match_flag() -> None:
    rows = [
        {"outcome": "ok"},
        {"outcome": "length", "hit_cap": True},
        {"outcome": "empty_selection", "no_exact_match": True},
        {"outcome": "http_error"},
        {"outcome": "no_llm_call"},
    ]
    r = rp.rates(rows)
    assert r["outcome_counts"] == {"ok": 1, "length": 1, "empty_selection": 1, "http_error": 1}
    assert r["no_exact_match"] == 1 and r["n_llm_calls"] == 4 and r["fallback_rate"] == 0.75
