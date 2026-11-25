"""Cache metadata, empty replies, GLOBAL pacing, worst-case budget and pinned limits (mock only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.retrieval import common, run
from evals.retrieval import groq_judge as gj

SPEC = gj.ModelSpec("j", "openai/gpt-oss-120b", {}, 100, 10_000)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _runner(tmp_path: Path, transport: Any, clock: Clock | None = None) -> gj.GroqRunner:
    clock = clock or Clock()
    return gj.GroqRunner(
        transport,
        gj.Ledger(tmp_path / "l.jsonl"),
        gj.ResultCache(tmp_path / "c.jsonl"),
        clock=clock.now,
        sleep=clock.sleep,
    )


def _ok() -> gj.HttpResult:
    return _body("ok", "stop", 50)


def _body(text: str, finish: str, comp: int) -> gj.HttpResult:
    return gj.HttpResult(
        200,
        {},
        {
            "choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 600, "completion_tokens": comp, "total_tokens": 600 + comp},
        },
    )


def test_cache_resume_keeps_finish_reason_usage_and_latency(tmp_path: Path) -> None:
    clock = Clock()

    def transport(payload: dict) -> gj.HttpResult:
        clock.t += 1.25  # the call "takes" 1250 ms
        return _body("cut off", "length", 150)

    first = _runner(tmp_path, transport, clock).call(SPEC, "q1", "s", "u", "h1")
    assert first and first["latency_ms"] == 1250 and first["finish_reason"] == "length"
    resumed = _runner(tmp_path, lambda p: pytest.fail("no call on resume")).call(
        SPEC, "q1", "s", "u", "h1"
    )
    assert resumed is not None and resumed["cached"] is True
    assert resumed["finish_reason"] == "length"
    assert resumed["usage"]["completion_tokens"] == 150 and resumed["usage"]["total_tokens"] == 750
    assert resumed["latency_ms"] == 1250


def test_empty_content_reply_is_cached_with_meta(tmp_path: Path) -> None:
    calls: list[dict] = []

    def transport(payload: dict) -> gj.HttpResult:
        calls.append(payload)
        return _body("", "length", 150)  # reasoning ate the whole cap

    first = _runner(tmp_path, transport).call(SPEC, "q1", "s", "u", "h1")
    assert first is not None and first["text"] == ""
    again = _runner(tmp_path, transport).call(SPEC, "q1", "s", "u", "h1")
    assert again is not None and len(calls) == 1 and again["cached"] and again["text"] == ""
    assert again["finish_reason"] == "length" and again["usage"]["completion_tokens"] == 150


def test_old_text_only_cache_records_still_load(tmp_path: Path) -> None:
    key = gj.ResultCache.key("j", "q1", "h1")
    (tmp_path / "c.jsonl").write_text(json.dumps({"key": key, "text": "old"}) + "\n")
    hit = _runner(tmp_path, lambda p: pytest.fail("cached")).call(SPEC, "q1", "s", "u", "h1")
    assert hit is not None and hit["text"] == "old"
    assert hit["finish_reason"] is None and hit["usage"] == {}


def test_pacing_is_global_across_models(tmp_path: Path) -> None:
    clock = Clock()
    other = gj.ModelSpec("k", "qwen/qwen3.8-27b", {}, 100, 10_000)
    r = _runner(tmp_path, lambda p: _ok(), clock)
    r.call(SPEC, "q1", "s", "u", "h1")
    r.call(other, "q1", "s", "u", "h2")  # a DIFFERENT model still waits the full interval
    r.call(SPEC, "q2", "s", "u", "h3")
    assert clock.sleeps == [pytest.approx(8.0), pytest.approx(8.0)]


def test_budget_precheck_uses_the_full_completion_cap(tmp_path: Path) -> None:
    # 650 used + ~2 prompt tokens + max_tokens 1500 > 2000, although a typical call is ~50 tokens
    big = gj.ModelSpec("j", "openai/gpt-oss-120b", {}, 1500, 2_000)
    r = _runner(tmp_path, lambda p: _ok())
    assert r.call(big, "q1", "s", "u", "h1") is not None  # 0 + 1502 <= 2000
    with pytest.raises(gj.BudgetExhaustedError):
        r.call(big, "q2", "s", "u", "h2")  # 650 + 1502 > 2000


def test_architect_limits_are_pinned(tmp_path: Path) -> None:
    assert common.MIN_REQUEST_INTERVAL_S == 8.0
    assert common.JUDGE_DAILY_TOKEN_BUDGET == 150_000
    assert common.RERANKER_DAILY_TOKEN_BUDGET == 100_000
    assert gj.MAX_CONSECUTIVE_429 == 3
    assert all(s.daily_budget == 150_000 for s in run.judge_specs().values())
    assert run.reranker_spec().daily_budget == 100_000
    assert gj.ModelSpec("x", "m").daily_budget == 150_000
    runner = gj.GroqRunner(lambda p: _ok(), gj.Ledger(tmp_path / "l.jsonl"))
    assert runner.min_interval_s == 8.0
