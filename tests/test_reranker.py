"""Unit tests for src/agents/reranker.py (stage-2 variant C promoted to production).

One class per concern: prompt parity with the eval's variant C, reply parsing, ID validation +
retrieval-order fill, one test per logged outcome, and the request parameters. No network: the
LLM is a fake exposing `complete()` (GroqClient's contract) or plain `generate()`.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from src.agents import reranker as rr
from src.agents.reranker import (
    RERANK_MAX_TOKENS,
    RERANK_SYSTEM,
    ReplyFormatError,
    build_prompt,
    parse_reply,
    rerank,
    validate_ids,
)
from src.llm.client import LLMCallError, LLMCompletion


def _item(i: int, **extra: Any) -> dict:
    base: dict = {
        "article_id": f"a{i}",
        "colour": "Black",
        "display_name": f"Item {i} dress",
        "price_inr": 999.0,
        "product_type": "Dress",
    }
    base.update(extra)
    return base


POOL = [_item(i) for i in range(1, 21)]  # 20 candidates, like the eval pools
RETRIEVAL_TOP5 = ["a1", "a2", "a3", "a4", "a5"]


def _done(text: str, finish: str | None = "stop", **kw: Any) -> LLMCompletion:
    return LLMCompletion(
        text, finish, kw.get("prompt", 650), kw.get("completion", 120), kw.get("reasoning", 100),
        kw.get("latency", 900), "openai/gpt-oss-20b",
    )


class FakeComplete:
    """Stands in for GroqClient: records what `complete()` was asked for."""

    def __init__(self, result: LLMCompletion | Exception) -> None:
        self.result = result
        self.calls: list[tuple[list[dict], dict]] = []

    def complete(self, messages: list[dict], **kw: Any) -> LLMCompletion:
        self.calls.append((messages, kw))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeGenerate:
    """A client with only generate() (Ollama / test stubs): text only, no finish_reason."""

    def __init__(self, text: str | Exception) -> None:
        self.text, self.kw = text, {}

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        self.kw = kw
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


def _ids(items: list[dict]) -> list[str]:
    return [it["article_id"] for it in items]


def _last_log(caplog: pytest.LogCaptureFixture) -> dict:
    recs = [json.loads(r.getMessage()) for r in caplog.records if r.name == "src.agents.reranker"]
    assert len(recs) == 1, recs
    return recs[0]


# --- parity with eval variant C ------------------------------------------------------------

EVAL_C_SYSTEM = (  # copied from evals/retrieval/variants.py::C_SYSTEM (feat/v4-retrieval-eval)
    "You rank fashion products for a shopper's query. Choose the 8 best matches, best first.\n"
    "Rules: a stated gender, colour, category and budget are hard requirements; the occasion "
    "must fit; never pick sleepwear or swimwear for a formal occasion. Each line is: number. "
    "title | price band | colour.\n"
    "Reply with ONLY the product numbers, comma-separated, best first, for example: 4,1,9,2,7,3,"
    "12,5. No words, no JSON."
)


class TestPromptParityWithEvalC:
    def test_system_prompt_is_byte_identical_to_eval_c(self) -> None:
        assert RERANK_SYSTEM == EVAL_C_SYSTEM

    def test_user_prompt_golden(self) -> None:
        items = [
            _item(1, display_name="Black Satin Midi Dress for the evening out tonight ok yes more extra",
                  price_inr=1499.0, colour="Black"),
            _item(2, display_name="Red Kurta", price_inr=None, colour=None),
            _item(3, display_name="Silk Saree", price_inr=12000, colour="Gold"),
        ]
        _, user = build_prompt("black dress under 2000", items)
        assert user == (
            'Query: "black dress under 2000"\nProducts:\n'
            "1. Black Satin Midi Dress for the evening out tonight ok yes more | <=₹1500 | Black\n"
            "2. Red Kurta | price n/a | n/a\n"
            "3. Silk Saree | >₹10000 | Gold\n"
            "Best 8, numbers only:"
        )

    def test_matches_eval_module_when_present(self) -> None:
        variants = pytest.importorskip("evals.retrieval.variants")  # only on feat/v4-retrieval-eval
        assert variants.build_compact_prompt("grey gown <3000", POOL) == build_prompt(
            "grey gown <3000", POOL
        )
        assert variants.C_MAX_TOKENS == RERANK_MAX_TOKENS


# --- parsing and ID validation --------------------------------------------------------------

class TestParseReply:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("3,1,2", [3, 1, 2]),
            (" 3, 1 ,2. ", [3, 1, 2]),
            ("3 1 2", [3, 1, 2]),
            ("[3, 1, 2]", [3, 1, 2]),
            ("", []),
            ("   \n", []),
            ("[]", []),
        ],
    )
    def test_valid_shapes(self, text: str, expected: list[int]) -> None:
        assert parse_reply(text) == expected

    @pytest.mark.parametrize(
        "text",
        ['{"selected": [1,2]}', "Top picks: 3,1,2", "3,one,2", "[1, 2", '["1","2"]', "[1.5]",
         "[true]", "-3,2"],
    )
    def test_malformed_raises(self, text: str) -> None:
        with pytest.raises(ReplyFormatError):
            parse_reply(text)


class TestValidateIds:
    def test_drops_unknown_and_duplicates_keeps_order(self) -> None:
        assert validate_ids([5, 0, 3, 5, 21, 1, -2], 20) == ([5, 3, 1], 3, 1)

    def test_boundaries(self) -> None:
        assert validate_ids([1, 20], 20) == ([1, 20], 0, 0)
        assert validate_ids([21], 20) == ([], 1, 0)


# --- outcomes -------------------------------------------------------------------------------

class TestOutcomes:
    def test_ok_uses_model_order_and_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        out = rerank("q", POOL, FakeComplete(_done("7,3,9,1,12,5,2,8")))
        assert _ids(out) == ["a7", "a3", "a9", "a1", "a12"]
        log = _last_log(caplog)
        assert log["outcome"] == "ok"
        assert (log["finish_reason"], log["prompt_tokens"], log["completion_tokens"]) == (
            "stop", 650, 120)
        assert (log["reasoning_tokens"], log["latency_ms"]) == (100, 900)
        assert (log["n_candidates"], log["n_selected"]) == (20, 8)

    def test_partial_fills_from_retrieval_order(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        out = rerank("q", POOL, FakeComplete(_done("14,1")))
        assert _ids(out) == ["a14", "a1", "a2", "a3", "a4"]  # fill skips the already-picked a1
        log = _last_log(caplog)
        assert (log["outcome"], log["n_selected"]) == ("partial", 2)

    def test_unknown_ids_dropped_valid_kept_in_order(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        out = rerank("q", POOL, FakeComplete(_done("99,4,0,4,6,2,3,8")))
        assert _ids(out) == ["a4", "a6", "a2", "a3", "a8"]
        log = _last_log(caplog)
        assert (log["outcome"], log["n_unknown"], log["n_duplicate"]) == ("ok", 2, 1)

    def test_empty_selection_with_stop_is_a_valid_no_exact_match(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        out = rerank("q", POOL, FakeComplete(_done("", finish="stop", completion=62)))
        assert _ids(out) == RETRIEVAL_TOP5
        log = _last_log(caplog)
        assert (log["outcome"], log["no_exact_match"]) == ("empty_selection", True)
        assert caplog.records[0].levelno == logging.INFO  # a valid answer, not a warning

    def test_empty_json_list_with_stop_is_empty_selection(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        rerank("q", POOL, FakeComplete(_done("[]")))
        assert _last_log(caplog)["outcome"] == "empty_selection"

    @pytest.mark.parametrize("text", ["Sorry, none match.", '{"selected": [1,2]}', "3,x,5"])
    def test_parse_error(self, text: str, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        out = rerank("q", POOL, FakeComplete(_done(text)))
        assert _ids(out) == RETRIEVAL_TOP5
        assert _last_log(caplog)["outcome"] == "parse_error"
        assert caplog.records[0].levelno == logging.WARNING

    def test_all_ids_unknown_is_parse_error(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        out = rerank("q", POOL, FakeComplete(_done("40,41,0")))
        assert _ids(out) == RETRIEVAL_TOP5
        log = _last_log(caplog)
        assert (log["outcome"], log["n_unknown"]) == ("parse_error", 3)

    def test_length_is_never_trusted_even_when_the_text_parses(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        out = rerank("q", POOL, FakeComplete(_done("7,3,9", finish="length", completion=400)))
        assert _ids(out) == RETRIEVAL_TOP5  # a cut reply could end mid-number
        assert _last_log(caplog)["outcome"] == "length"

    def test_length_with_empty_text(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        rerank("q", POOL, FakeComplete(_done("", finish="length", completion=400)))
        assert _last_log(caplog)["outcome"] == "length"  # not mistaken for empty_selection

    @pytest.mark.parametrize("kind", ["timeout", "server", "rate_limit", "fatal", "malformed"])
    def test_http_error(self, kind: str, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        out = rerank("q", POOL, FakeComplete(LLMCallError(kind, 503)))
        assert _ids(out) == RETRIEVAL_TOP5
        log = _last_log(caplog)
        assert (log["outcome"], log["error_kind"]) == ("http_error", kind)
        assert log["finish_reason"] is None and log["latency_ms"] >= 0

    def test_budget_is_the_daily_quota(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        out = rerank("q", POOL, FakeComplete(LLMCallError("daily_quota", 429)))
        assert _ids(out) == RETRIEVAL_TOP5
        assert _last_log(caplog)["outcome"] == "budget"

    def test_unexpected_exception_from_complete_is_not_swallowed(self) -> None:
        with pytest.raises(RuntimeError):
            rerank("q", POOL, FakeComplete(RuntimeError("bug")))

    def test_pool_not_larger_than_top_k_skips_the_llm(self) -> None:
        llm = FakeComplete(_done("1,2,3"))
        assert rerank("q", POOL[:5], llm) == POOL[:5]
        assert llm.calls == []

    def test_generate_only_client_ok(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        llm = FakeGenerate("3,1,2,5,4")
        assert _ids(rerank("q", POOL, llm)) == ["a3", "a1", "a2", "a5", "a4"]
        assert llm.kw == {"max_tokens": RERANK_MAX_TOKENS, "temperature": 0}
        assert _last_log(caplog)["outcome"] == "ok"

    def test_generate_only_client_raising_is_a_logged_http_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        items = [_item(i, colour=None) for i in range(8)]
        out = rerank("date night look", items, FakeGenerate(RuntimeError("boom")))
        assert len(out) == 5
        assert _last_log(caplog)["outcome"] == "http_error"

    def test_generate_only_client_empty_reply_is_parse_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # without a finish_reason an empty reply cannot be told from a failure: not trusted
        caplog.set_level(logging.INFO)
        rerank("q", POOL, FakeGenerate(""))
        assert _last_log(caplog)["outcome"] == "parse_error"

    def test_every_logged_outcome_is_declared(self) -> None:
        assert {"ok", "partial", "parse_error", "length", "empty_selection", "http_error",
                "budget"} <= set(rr.OUTCOMES)


class TestRequestParameters:
    def test_complete_receives_low_effort_temperature_zero_and_the_cap(self) -> None:
        llm = FakeComplete(_done("1,2,3,4,5"))
        rerank("grey gown", POOL, llm)
        messages, kw = llm.calls[0]
        assert kw == {"temperature": 0, "max_tokens": 400, "reasoning_effort": "low"}
        assert [m["role"] for m in messages] == ["system", "user"]
        assert (messages[0]["content"], messages[1]["content"]) == build_prompt("grey gown", POOL)

    def test_none_valued_fields_render_blank_not_none(self) -> None:
        _, user = build_prompt("q", [_item(1, colour=None, display_name=None, price_inr=None)])
        assert "None" not in user
