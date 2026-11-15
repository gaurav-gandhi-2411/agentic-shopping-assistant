"""Tests for the throttled/budgeted Groq runner, the live guard and the caching probe (mock only)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from evals.retrieval import caching_probe as cp
from evals.retrieval import groq_judge as gj
from evals.retrieval.mock import mock_transport

SPEC = gj.ModelSpec("j", "openai/gpt-oss-120b", {}, 100, 10_000, 100)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _runner(
    tmp_path: Path,
    transport: Any,
    clock: Clock | None = None,
    cache: bool = True,
    interval: float = 8.0,
) -> gj.GroqRunner:
    clock = clock or Clock()
    return gj.GroqRunner(
        transport,
        gj.Ledger(tmp_path / "l.jsonl"),
        gj.ResultCache(tmp_path / "c.jsonl") if cache else None,
        min_interval_s=interval,
        clock=clock.now,
        sleep=clock.sleep,
    )


def _ok(headers: dict[str, str] | None = None) -> gj.HttpResult:
    body = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 600, "completion_tokens": 50, "total_tokens": 650},
    }
    return gj.HttpResult(200, headers or {}, body)


def test_live_guard_requires_flag_and_env() -> None:
    with pytest.raises(gj.LiveNotAllowedError):
        gj.assert_live_allowed(False, {"STAGE2_LIVE_OK": "1"})
    with pytest.raises(gj.LiveNotAllowedError):
        gj.assert_live_allowed(True, {})
    with pytest.raises(gj.LiveNotAllowedError):
        gj.assert_live_allowed(True, {"STAGE2_LIVE_OK": "0"})
    gj.assert_live_allowed(True, {"STAGE2_LIVE_OK": "1"})  # does not raise


def test_cli_refuses_live_without_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from evals.retrieval import run

    monkeypatch.delenv("STAGE2_LIVE_OK", raising=False)
    rc = run.main(["--stage", "judge", "--live", "--out", str(tmp_path)])
    assert rc == 3  # STOPPED: LiveNotAllowedError, no transport was ever built


def test_cache_prevents_duplicate_calls(tmp_path: Path) -> None:
    calls = []

    def transport(payload: dict) -> gj.HttpResult:
        calls.append(payload)
        return _ok()

    r = _runner(tmp_path, transport)
    a = r.call(SPEC, "q1", "sys", "user", "h1")
    b = r.call(SPEC, "q1", "sys", "user", "h1")
    assert a and b and b["cached"] and len(calls) == 1
    r2 = _runner(tmp_path, transport)  # a fresh process resumes from the JSONL cache
    assert r2.call(SPEC, "q1", "sys", "user", "h1")["cached"] and len(calls) == 1
    r.call(SPEC, "q1", "sys", "user", "h2")  # different prompt hash -> new call
    assert len(calls) == 2


def test_throttle_enforces_min_interval(tmp_path: Path) -> None:
    clock = Clock()
    r = _runner(tmp_path, lambda p: _ok(), clock)
    r.call(SPEC, "q1", "s", "u", "h1")
    r.call(SPEC, "q2", "s", "u", "h2")
    assert clock.sleeps and clock.sleeps[0] == pytest.approx(8.0)


def test_daily_budget_hard_stop(tmp_path: Path) -> None:
    small = gj.ModelSpec("j", "openai/gpt-oss-120b", {}, 100, 1_400, 100)
    r = _runner(tmp_path, lambda p: _ok())
    r.call(small, "q1", "s", "u", "h1")  # ledger now holds 650 tokens
    r.call(small, "q2", "s", "u", "h2")  # 650 + ~102 est fits (1300 total)
    with pytest.raises(gj.BudgetExhaustedError):
        r.call(small, "q3", "s", "u", "h3")  # 1300 + estimate > 1400


def test_budget_is_per_day_and_per_model(tmp_path: Path) -> None:
    day = {"d": dt.datetime(2026, 11, 20, 10, tzinfo=dt.timezone.utc)}
    ledger = gj.Ledger(tmp_path / "l.jsonl", now=lambda: day["d"])
    ledger.append({"model": "m1", "total_tokens": 900})
    ledger.append({"model": "m2", "total_tokens": 50})
    assert ledger.tokens_today("m1") == 900 and ledger.tokens_today("m2") == 50
    day["d"] += dt.timedelta(days=1)
    assert ledger.tokens_today("m1") == 0


def test_three_consecutive_429_stops_and_two_recover(tmp_path: Path) -> None:
    seq = iter([gj.HttpResult(429, {"retry-after": "3"}, {})] * 3)
    clock = Clock()
    r = _runner(tmp_path, lambda p: next(seq), clock)
    with pytest.raises(gj.RateLimitStopError):
        r.call(SPEC, "q1", "s", "u", "h1")
    assert r.consecutive_429 == 3 and any(s >= 3 for s in clock.sleeps)  # Retry-After honoured
    seq2 = iter([gj.HttpResult(429, {}, {}), gj.HttpResult(429, {}, {}), _ok()])
    r2 = _runner(tmp_path / "b", lambda p: next(seq2))
    assert r2.call(SPEC, "q1", "s", "u", "h1")["text"] == "ok" and r2.consecutive_429 == 0


def test_ledger_records_usage_fields(tmp_path: Path) -> None:
    import json

    body = {
        "choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 4},
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    }
    r = _runner(
        tmp_path, lambda p: gj.HttpResult(200, {"x-ratelimit-remaining-tokens": "123"}, body)
    )
    r.call(SPEC, "q1", "s", "u", "h1")
    rec = json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])
    assert (rec["prompt_tokens"], rec["reasoning_tokens"], rec["cached_tokens"]) == (10, 4, 3)
    assert rec["remaining_tokens"] == "123" and rec["finish_reason"] == "length"


def test_parse_reset_seconds() -> None:
    assert gj.parse_reset_seconds("2m59.56s") == pytest.approx(179.56)
    assert gj.parse_reset_seconds("250ms") == pytest.approx(0.25)
    assert gj.parse_reset_seconds(None) == 0.0


def test_header_slowdown_when_bucket_low(tmp_path: Path) -> None:
    clock = Clock()
    hdr = {"x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": "5s"}
    r = _runner(tmp_path, lambda p: _ok(hdr), clock)
    r.call(SPEC, "q1", "s", "u", "h1")
    assert 5.0 in clock.sleeps


def _probe_transport(not_counted: bool) -> Any:
    state = {"remaining": 8000.0, "n": 0}

    def post(payload: dict) -> gj.HttpResult:
        state["n"] += 1
        cached = 0 if state["n"] % 3 == 1 else 1024
        total = 1500
        state["remaining"] -= total - (cached if not_counted else 0)
        body = {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1400,
                "completion_tokens": 100,
                "total_tokens": total,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
        return gj.HttpResult(
            200,
            {
                "x-ratelimit-remaining-tokens": str(state["remaining"]),
                "x-ratelimit-limit-tokens": "8000",
            },
            body,
        )

    return post


@pytest.mark.parametrize("not_counted", [True, False])
def test_caching_probe_mock_verdicts_and_cap(tmp_path: Path, not_counted: bool) -> None:
    clock = Clock()
    r = _runner(tmp_path, _probe_transport(not_counted), clock, cache=False, interval=0.0)
    out = cp.probe(r, "sys", "user")
    assert out["calls"] == 6 <= cp.MAX_LIVE_CALLS
    for m in out["models"].values():
        assert m["verdict"] == "VERIFIED"
        assert ("NOT counted" in m["reason"]) == not_counted


def test_caching_probe_no_cached_tokens_is_not_verified(tmp_path: Path) -> None:
    r = _runner(tmp_path, mock_transport(lambda s, u, m: "{}"), cache=False, interval=0.0)
    out = cp.probe(r, "sys", "user")
    assert all(m["verdict"] == "NOT VERIFIED" for m in out["models"].values())


def test_probe_cap_is_hard(tmp_path: Path) -> None:
    r = _runner(tmp_path, lambda p: _ok(), cache=False, interval=0.0)
    with pytest.raises(RuntimeError, match="cap"):
        cp.probe(r, "s", "u", models={"a": "m", "b": "m", "c": "m"})


def test_probe_refuses_without_live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STAGE2_LIVE_OK", "1")
    with pytest.raises(gj.LiveNotAllowedError):
        cp.main(["--out", str(tmp_path)])
