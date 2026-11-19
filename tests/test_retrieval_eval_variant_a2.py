"""Variant A2: production reranker with the truncation fixed, and its length-hit contract."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from evals.retrieval import variants as vr
from src.agents import reranker as prod
from tests.test_retrieval_eval_variants import FakeLLM, _ids, _items

RRF5 = ["a1", "a2", "a3", "a4", "a5"]
OK = '{"selected": [7, 3, 12, 1, 9], "reasoning": "best fit"}'


def test_a2_sends_the_production_prompt_with_the_larger_cap() -> None:
    llm = FakeLLM(OK, {"completion_tokens": 230, "prompt_tokens": 1650})
    res = vr.run_variant_a2("black kurta", _items(), llm)
    assert llm.seen["system"] == prod._SYSTEM and 'Query: "black kurta"' in llm.seen["prompt"]
    assert llm.seen["max_tokens"] == vr.A2_MAX_TOKENS == 600 and llm.seen["temperature"] == 0
    assert res["max_tokens"] == 600 and res["variant"] == "A2"  # recorded cap = cap actually sent
    assert _ids(res) == ["a7", "a3", "a12", "a1", "a9"] and res["outcome"] == "ok"
    assert res["length_hit"] is False and res["hit_cap"] is False


def test_variant_a_still_uses_the_production_cap_of_150() -> None:
    llm = FakeLLM(OK)
    res = vr.run_variant_a("q", _items(), llm)
    assert llm.seen["max_tokens"] == vr.A_MAX_TOKENS == 150 and res["variant"] == "A"


def test_a2_complete_json_at_a_length_finish_is_still_not_trusted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLLM(OK, {"completion_tokens": 600}, finish="length")
    with caplog.at_level(logging.WARNING):
        res = vr.run_variant_a2("q", _items(), llm)
    assert res["length_hit"] and res["outcome"] == "length_fallback"
    assert _ids(res) == RRF5  # explicit retrieval-order fallback, not the reranker's picks
    assert "finish_reason=length" in caplog.text


def test_a2_length_hit_with_empty_content_falls_back_and_is_counted() -> None:
    res = vr.run_variant_a2("q", _items(), FakeLLM("", {"completion_tokens": 600}, "length"))
    assert res["outcome"] == "length_fallback" and _ids(res) == RRF5 and res["length_hit"]


def test_a2_completion_at_the_cap_without_a_length_reason_still_counts_as_a_hit() -> None:
    res = vr.run_variant_a2("q", _items(), FakeLLM(OK, {"completion_tokens": 600}, "stop"))
    assert res["length_hit"] and res["outcome"] == "length_fallback"


def test_assert_no_length_hits_passes_clean_runs_and_names_offenders() -> None:
    clean: list[dict[str, Any]] = [{"query_id": "q1", "length_hit": False}, {"query_id": "q2"}]
    vr.assert_no_length_hits(clean)
    dirty = [*clean, {"query_id": "q9", "rep": 2, "length_hit": True}]
    with pytest.raises(AssertionError, match=r"q9#2"):
        vr.assert_no_length_hits(dirty)


def test_a2_small_pool_makes_no_call() -> None:
    llm = FakeLLM("never")
    res = vr.run_variant_a2("q", _items(5), llm)
    assert res["outcome"] == "no_llm_call" and llm.seen == {} and res["length_hit"] is False
