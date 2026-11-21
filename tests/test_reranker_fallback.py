"""Reranker fallback path: one attempt on gpt-oss-120b after a 429/5xx/timeout, then retrieval
order; total turn budget 3 s; the path taken is logged. No network."""
from __future__ import annotations

import logging

import pytest

from src.agents import reranker as rr
from src.agents.reranker import rerank
from src.llm.client import LLMCallError
from tests.test_reranker import (
    POOL,
    RETRIEVAL_TOP5,
    FakeComplete,
    FakeGenerate,
    _done,
    _ids,
    _last_log,
)


class TestFallbackPath:
    @pytest.mark.parametrize("kind", ["rate_limit", "daily_quota", "server", "timeout"])
    def test_primary_failure_then_one_fallback_attempt(
        self, kind: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        llm = FakeComplete([LLMCallError(kind, 429), _done("7,3,9,1,12")])
        out = rerank("q", POOL, llm)
        assert _ids(out) == ["a7", "a3", "a9", "a1", "a12"]
        assert len(llm.calls) == 2
        first, second = llm.calls[0][1], llm.calls[1][1]
        assert "model" not in first and second["model"] == "openai/gpt-oss-120b"
        assert llm.calls[0][0] == llm.calls[1][0]  # same prompt
        assert second["timeout_s"] <= 3.0 and second["retry"] is False
        log = _last_log(caplog)
        assert (log["path"], log["outcome"]) == ("fallback_120b", "ok")

    def test_both_fail_ends_in_retrieval_order_after_exactly_two_attempts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        llm = FakeComplete([LLMCallError("server", 503), LLMCallError("timeout", None)])
        out = rerank("q", POOL, llm)
        assert _ids(out) == RETRIEVAL_TOP5 and len(llm.calls) == 2
        log = _last_log(caplog)
        assert (log["path"], log["outcome"], log["error_kind"]) == (
            "retrieval_order", "http_error", "timeout")

    @pytest.mark.parametrize("kind", ["fatal", "malformed"])
    def test_non_transient_errors_do_not_fall_back(
        self, kind: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        llm = FakeComplete([LLMCallError(kind, 400), _done("1,2,3,4,5")])
        rerank("q", POOL, llm)
        assert len(llm.calls) == 1 and _last_log(caplog)["path"] == "retrieval_order"

    def test_a_bad_primary_reply_is_not_retried_on_the_fallback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        llm = FakeComplete([_done("no idea"), _done("1,2,3,4,5")])
        rerank("q", POOL, llm)
        log = _last_log(caplog)
        assert (len(llm.calls), log["path"], log["outcome"]) == (1, "primary", "parse_error")

    def test_no_fallback_when_the_budget_is_spent(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        times = iter([0.0, 2.9, 2.9, 2.9, 2.9, 2.9])  # the primary "took" 2.9 s of the 3 s
        monkeypatch.setattr(rr.time, "monotonic", lambda: next(times))
        llm = FakeComplete([LLMCallError("timeout", None), _done("1,2,3,4,5")])
        rerank("q", POOL, llm)
        assert len(llm.calls) == 1 and _last_log(caplog)["path"] == "retrieval_order"

    def test_generate_only_client_has_no_fallback_and_logs_primary(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        rerank("q", POOL, FakeGenerate("1,2,3,4,5"))
        assert _last_log(caplog)["path"] == "primary"

    def test_budget_constants(self) -> None:
        assert rr.RERANK_BUDGET_S == 3.0 and rr.PRIMARY_TIMEOUT_S < rr.RERANK_BUDGET_S
