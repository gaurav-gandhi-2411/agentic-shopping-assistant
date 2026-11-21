"""Tests for the A/B/C variants, including how the PRODUCTION reranker parses truncated output."""

from __future__ import annotations

from typing import Any

import pytest

from evals.retrieval import variants as vr
from src.agents import reranker as prod


def _items(n: int = 20) -> list[dict[str, Any]]:
    return [
        {
            "article_id": f"a{i}",
            "display_name": f"Item number {i} blue kurta",
            "colour": "Blue",
            "product_type": "Kurta",
            "department": "N/A",
            "price_inr": 900.0 + 400 * i,
            "detail_desc": "Some description text",
        }
        for i in range(1, n + 1)
    ]


class FakeLLM:
    def __init__(
        self, out: str | Exception, usage: dict | None = None, finish: str = "stop"
    ) -> None:
        self.out, self.usage, self.finish = out, usage or {}, finish
        self.last_meta: dict[str, Any] = {}
        self.seen: dict[str, Any] = {}

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        self.seen = {"prompt": prompt, "system": system, **kw}
        self.last_meta = {"finish_reason": self.finish, "usage": self.usage, "latency_ms": 7}
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


def _ids(res: dict[str, Any]) -> list[str]:
    return res["final_ids"]


RRF5 = ["a1", "a2", "a3", "a4", "a5"]


def test_truncated_json_falls_back_to_retrieval_order() -> None:
    """(c) src/agents/reranker.py: output cut mid-"reasoning" has no closing brace -> not
    parseable -> `_parse_selected` returns None -> rerank() returns items[:5] (RRF order), even
    though the picks were already visible in the cut-off text."""
    cut = '{"selected": [7, 3, 12, 1, 9], "reasoning": "Black kurtas that suit a hal'
    res = vr.run_variant_a("q", _items(), FakeLLM(cut, {"completion_tokens": 150}, "length"))
    assert _ids(res) == RRF5 and res["outcome"] == "length"
    assert res["hit_cap"] is True and res["max_tokens"] == 150 == vr.A_MAX_TOKENS


def test_cut_inside_selected_list_also_falls_back() -> None:
    res = vr.run_variant_a("q", _items(), FakeLLM('{"selected": [7, 3, 12', {}, "length"))
    assert _ids(res) == RRF5 and res["outcome"] == "length"


def test_empty_reasoning_exhausted_output_falls_back() -> None:
    """Reasoning models can spend the whole cap thinking: empty content -> busy message -> same."""
    for out in ("", "Sorry, the stylist is busy right now."):
        res = vr.run_variant_a("q", _items(), FakeLLM(out))
        assert _ids(res) == RRF5 and res["outcome"] == "parse_error"


def test_provider_exception_falls_back_as_http_error() -> None:
    res = vr.run_variant_a("q", _items(), FakeLLM(RuntimeError("boom")))
    assert _ids(res) == RRF5 and res["outcome"] == "http_error"


def test_complete_json_reorders_and_closed_short_list_is_topped_up() -> None:
    res = vr.run_variant_a(
        "q", _items(), FakeLLM('{"selected": [7, 3, 12, 1, 9], "reasoning": "x"}')
    )
    assert _ids(res) == ["a7", "a3", "a12", "a1", "a9"] and res["outcome"] == "ok"
    short = vr.run_variant_a("q", _items(), FakeLLM('{"selected": [7, 3]}'))
    assert _ids(short) == ["a7", "a3", "a1", "a2", "a4"] and short["outcome"] == "partial"


def test_variant_a_uses_production_prompt_and_cap() -> None:
    llm = FakeLLM('{"selected": [1,2,3,4,5]}')
    vr.run_variant_a("black kurta", _items(), llm)
    assert llm.seen["system"] == prod._SYSTEM and llm.seen["max_tokens"] == 150
    assert llm.seen["temperature"] == 0 and 'Query: "black kurta"' in llm.seen["prompt"]


def test_variant_a_no_llm_call_when_pool_small() -> None:
    llm = FakeLLM("never")
    res = vr.run_variant_a("q", _items(5), llm)
    assert res["outcome"] == "no_llm_call" and llm.seen == {}


def test_variant_b_is_rrf_order() -> None:
    assert vr.run_variant_b(_items())["final_ids"] == RRF5


def test_compact_prompt_uses_numbers_short_titles_and_stable_prefix() -> None:
    items = _items()
    items[0]["display_name"] = " ".join(f"w{i}" for i in range(30))
    s1, u1 = vr.build_compact_prompt("q1", items)
    s2, _ = vr.build_compact_prompt("q2", _items())
    assert s1 == s2  # byte-stable instruction prefix
    assert "a1" not in u1 and "1. w0 w1" in u1 and "w12" not in u1  # numbers, 12-word cap
    assert "<=₹1500" in u1 and u1.count("\n") >= 21
    assert vr.C_MAX_TOKENS >= 48


def test_parse_compact_robust() -> None:
    assert vr.parse_compact("4, 1, 9, 2", 20) == [4, 1, 9, 2]
    assert vr.parse_compact("Top picks: 3 > 3 > 25 > 7", 20) == [3, 7]  # dup + out of range dropped
    assert vr.parse_compact("no numbers", 20) == []
    assert vr.parse_compact("", 20) == []


def test_variant_c_fallbacks() -> None:
    ok = vr.run_variant_c("q", _items(), FakeLLM("7,3,12,1,9,2,4,5"))
    assert _ids(ok) == ["a7", "a3", "a12", "a1", "a9"] and ok["outcome"] == "ok"
    bad = vr.run_variant_c("q", _items(), FakeLLM("I cannot do that"))
    assert _ids(bad) == RRF5 and bad["outcome"] == "parse_error"
    part = vr.run_variant_c("q", _items(), FakeLLM("7,3"))
    assert _ids(part) == ["a7", "a3", "a1", "a2", "a4"] and part["outcome"] == "partial"
    boom = vr.run_variant_c("q", _items(), FakeLLM(ValueError("x")))
    assert _ids(boom) == RRF5 and boom["outcome"] == "http_error"


def test_variant_c_records_cap_hit_and_params() -> None:
    llm = FakeLLM("7,3", {"prompt_tokens": 500, "completion_tokens": vr.C_MAX_TOKENS}, "length")
    res = vr.run_variant_c("q", _items(), llm)
    assert (
        res["hit_cap"]
        and llm.seen["max_tokens"] == vr.C_MAX_TOKENS
        and llm.seen["temperature"] == 0
    )
    assert res["prompt_tokens"] == 500


@pytest.mark.parametrize(
    "price,band", [(500, "<=₹500"), (1200, "<=₹1500"), (99999, ">₹10000"), (None, "price n/a")]
)
def test_price_band(price: float | None, band: str) -> None:
    assert vr.price_band(price) == band


def test_variant_c_skips_llm_when_pool_not_larger_than_top_k() -> None:
    llm = FakeLLM("never")
    res = vr.run_variant_c("q", _items(5), llm)
    assert res["outcome"] == "no_llm_call" and llm.seen == {} and _ids(res) == RRF5
