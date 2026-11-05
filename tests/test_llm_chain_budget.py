"""Hard total time budget for the Groq retry/fallback chain (default 20 s).

Everything runs on a fake clock and a fake SDK whose create() honours the `timeout`
it is given (advances the clock by min(latency, timeout) and raises a timeout error
when the latency is longer), so the tests are instant and need no network.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.llm.client import GroqClient
from src.llm.resilience import LLM_BUSY_MESSAGE

MSGS = [{"role": "user", "content": "hi"}]
EPS = 0.5  # allowed overhead on top of the budget (simulated seconds)


class APITimeoutError(Exception):
    """Same class name as groq.APITimeoutError (classification is by name)."""


class FakeRateLimit(Exception):
    def __init__(self, retry_after: str) -> None:
        super().__init__("Error code: 429 - tokens per minute (TPM)")
        self.status_code = 429
        self.response = SimpleNamespace(headers={"retry-after": retry_after})


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _completion(text: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def _cfg(**llm: Any) -> dict:
    cfg = {
        "provider": "groq", "temperature": 0.2, "max_tokens": 400, "timeout_seconds": 60,
        "groq_model": "m1", "groq_fallback_models": ["m2", "m3"], "groq_timeout_seconds": 20,
    }
    cfg.update(llm)
    return {"llm": cfg}


class Env:
    def __init__(self, clock: FakeClock, calls: list[dict[str, Any]]) -> None:
        self.clock, self.calls = clock, calls

    @property
    def models(self) -> list[str]:
        return [c["model"] for c in self.calls]


def _install(
    monkeypatch: pytest.MonkeyPatch, latency: dict[str, Any], *,
    stream_first_chunk_delay: float | None = None, default_timeout: float = 20.0,
) -> Env:
    """latency[model]: seconds the call takes, an Exception raised immediately, or a list
    of those consumed one per call."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr("random.uniform", lambda a, b: 0.0)  # backoff = exactly 1.0 s
    clock = FakeClock()
    monkeypatch.setattr(
        "src.llm.client.time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
    )
    calls: list[dict[str, Any]] = []

    def create(**kw: Any) -> Any:
        calls.append(kw)
        item = latency[kw["model"]]
        if isinstance(item, list):
            item = item.pop(0)
        if isinstance(item, Exception):
            raise item
        timeout = kw.get("timeout", default_timeout)
        if item > timeout:  # the SDK gives up after `timeout`
            clock.t += timeout
            raise APITimeoutError("Request timed out.")
        clock.t += item
        if kw.get("stream"):
            def gen() -> Any:
                if stream_first_chunk_delay is not None:
                    clock.t += stream_first_chunk_delay  # keep-alive stall, no content
                    yield SimpleNamespace(choices=[], usage=None)
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="tok"),
                                             finish_reason=None)], usage=None)
            return gen()
        return _completion("fast")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("groq.Groq", lambda api_key, timeout, max_retries: fake)
    return Env(clock, calls)


def _events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict]:
    out = []
    for r in caplog.records:
        try:
            d = json.loads(r.getMessage())
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("event") == name:
            out.append(d)
    return out


def test_slow_model_yields_busy_within_budget_and_later_models_not_started(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    env = _install(monkeypatch, {"m1": 100.0, "m2": 100.0, "m3": 100.0})
    client = GroqClient(_cfg(groq_chain_budget_seconds=5))
    with caplog.at_level(logging.WARNING):
        out = client.chat(MSGS)
    assert out == LLM_BUSY_MESSAGE
    assert env.clock.t <= 5 + EPS, f"wall time {env.clock.t}s blew the 5 s budget"
    assert env.models == ["m1"], "no model may start after the deadline"
    ev = _events(caplog, "llm_chain_budget_exceeded")
    assert len(ev) == 1
    assert ev[0]["budget_s"] == 5.0 and ev[0]["models_tried"] == ["m1"]
    assert ev[0]["elapsed_s"] <= 5 + EPS and ev[0]["turn_id"]


def test_slow_primary_fast_fallback_inside_budget_is_served_by_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # m1 times out at 2 s, backs off 1 s, times out again at 5 s; m2 answers at 6 s < 8 s.
    env = _install(monkeypatch, {"m1": 100.0, "m2": 1.0, "m3": 100.0})
    client = GroqClient(_cfg(groq_chain_budget_seconds=8, groq_timeout_seconds=2))
    assert client.chat(MSGS) == "fast"
    assert env.models == ["m1", "m1", "m2"]
    assert env.clock.t <= 8


def test_attempt_timeout_and_backoff_sleep_are_clamped_to_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 429 with Retry-After 4 s (under the 8 s cap): sleeps 4 s of a 5 s budget, then the
    # retry may only use the 1 s left, and the first attempt is capped at the 5 s budget.
    env = _install(monkeypatch, {"m1": [FakeRateLimit("4"), 100.0], "m2": 100.0})
    assert GroqClient(_cfg(groq_chain_budget_seconds=5)).chat(MSGS) == LLM_BUSY_MESSAGE
    assert env.clock.sleeps == [4.0]
    assert [c["timeout"] for c in env.calls] == [5.0, 1.0]
    assert env.clock.t <= 5 + EPS


def test_retry_after_longer_than_remaining_budget_is_not_slept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _install(monkeypatch, {"m1": [FakeRateLimit("6")], "m2": 1.0, "m3": 1.0})
    # A 6 s wait would cross the 5 s deadline: skip the retry, serve from the next model.
    assert GroqClient(_cfg(groq_chain_budget_seconds=5)).chat(MSGS) == "fast"
    assert env.clock.sleeps == [] and env.models == ["m1", "m2"]


def test_sleep_within_never_sleeps_past_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _install(monkeypatch, {"m1": 1.0})
    GroqClient._sleep_within(10.0, deadline=env.clock.t + 3.0)
    GroqClient._sleep_within(10.0, deadline=env.clock.t - 1.0)  # already past: no sleep
    assert env.clock.sleeps == [3.0, 0.0]


def test_budget_is_read_from_config_default_20(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"m1": 1.0})
    assert GroqClient(_cfg())._chain_budget_s == 20.0
    assert GroqClient(_cfg(groq_chain_budget_seconds=5))._chain_budget_s == 5.0


def test_budget_value_from_config_is_honoured_by_the_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _install(monkeypatch, {"m1": 100.0, "m2": 100.0, "m3": 100.0})
    GroqClient(_cfg(groq_chain_budget_seconds=12)).chat(MSGS)
    assert 12 - EPS <= env.clock.t <= 12 + EPS and env.models == ["m1"]


def test_stream_time_to_first_chunk_ends_in_exactly_one_busy_message(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    env = _install(monkeypatch, {"m1": 100.0, "m2": 100.0, "m3": 100.0})
    client = GroqClient(_cfg(groq_chain_budget_seconds=5))
    with caplog.at_level(logging.WARNING):
        out = list(client.chat_stream(MSGS))
    assert out == [LLM_BUSY_MESSAGE]
    assert env.clock.t <= 5 + EPS and env.models == ["m1"]
    assert len(_events(caplog, "llm_chain_budget_exceeded")) == 1


def test_stream_stalled_before_first_token_past_deadline_is_busy_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Connection opens fast but only a content-less chunk arrives, after the deadline.
    env = _install(monkeypatch, {"m1": 0.5, "m2": 0.5, "m3": 0.5}, stream_first_chunk_delay=6.0)
    out = list(GroqClient(_cfg(groq_chain_budget_seconds=5)).chat_stream(MSGS))
    assert out == [LLM_BUSY_MESSAGE]
    assert env.models == ["m1"]


def test_stream_already_yielding_is_not_cut_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _install(monkeypatch, {"m1": 4.0})
    client = GroqClient(_cfg(groq_chain_budget_seconds=5))
    stream = client.chat_stream(MSGS)
    assert next(stream) == "tok"
    env.clock.t += 30  # long after the deadline, mid-reply
    assert list(stream) == []  # ends normally: no busy message, no error sentinel
