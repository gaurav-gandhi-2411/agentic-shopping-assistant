"""W4a: Groq model chain + bounded retry + explicit busy message.

Every SDK call is mocked (no network, zero live chat completions). Covers, for
chat() and chat_stream(): primary ok, retry-once on 429/5xx/timeout (with
Retry-After), fallback model, non-retryable errors, empty content, falling
and the final busy message (no non-Groq tier exists).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import STREAM_ERROR_SENTINEL, GroqClient, classify_error, retry_wait_seconds
from src.llm.resilience import LLM_BUSY_MESSAGE

MSGS = [{"role": "user", "content": "hi"}]


class FakeAPIError(Exception):
    """Stand-in for groq.APIStatusError: .status_code and .response.headers."""

    def __init__(self, status: int, body: str = "", headers: dict | None = None) -> None:
        super().__init__(f"Error code: {status} - {body}")
        self.status_code = status
        self.response = SimpleNamespace(headers=headers or {})


class APITimeoutError(Exception):
    """Same class name as groq.APITimeoutError (classification is by name)."""


def _completion(content: str | None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _chunks(*texts: str) -> list:
    out = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))], usage=None)
           for t in texts]
    out.append(SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2)))
    return out


def _config(**llm: Any) -> dict:
    cfg = {
        "provider": "groq", "temperature": 0.2, "max_tokens": 400, "timeout_seconds": 60,
        "groq_model": "m1", "groq_fallback_models": ["m2"],
        "openrouter_model": "or-model",
    }
    cfg.update(llm)
    return {"llm": cfg}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """No external tiers by default; time.sleep recorded instead of slept."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    return slept


def _sdk(monkeypatch: pytest.MonkeyPatch, script: dict[str, list]) -> list[str]:
    """Patch groq.Groq; `script[model]` is popped per call (Exception => raised)."""
    calls: list[str] = []

    def create(**kw: Any) -> Any:
        calls.append(kw["model"])
        item = script[kw["model"]].pop(0)
        if isinstance(item, Exception):
            raise item
        return iter(item) if kw.get("stream") else item

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("groq.Groq", lambda api_key, timeout, max_retries: fake)
    return calls


def _openrouter(monkeypatch: pytest.MonkeyPatch, **kw: Any) -> MagicMock:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    create = MagicMock(**kw)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("openai.OpenAI", lambda api_key, base_url, timeout: fake)
    return create


# ----------------------------------------------------------------- chat()

def test_primary_ok(monkeypatch):
    calls = _sdk(monkeypatch, {"m1": [_completion("hello")]})
    assert GroqClient(_config()).chat(MSGS) == "hello"
    assert calls == ["m1"]


def test_429_retries_once_with_backoff_then_ok(monkeypatch, _env):
    calls = _sdk(monkeypatch, {"m1": [FakeAPIError(429, "tokens per minute (TPM)"), _completion("ok")]})
    assert GroqClient(_config()).chat(MSGS) == "ok"
    assert calls == ["m1", "m1"]
    assert len(_env) == 1 and 1.0 <= _env[0] <= 2.0  # backoff + jitter


def test_retry_after_header_respected(monkeypatch, _env):
    err = FakeAPIError(429, "tokens per minute", headers={"retry-after": "5"})
    _sdk(monkeypatch, {"m1": [err, _completion("ok")]})
    assert GroqClient(_config()).chat(MSGS) == "ok"
    assert _env == [5.0]


def test_retry_after_beyond_cap_skips_retry_and_uses_fallback_model(monkeypatch, _env):
    err = FakeAPIError(429, "tokens per minute", headers={"retry-after": "120"})
    calls = _sdk(monkeypatch, {"m1": [err], "m2": [_completion("from m2")]})
    assert GroqClient(_config()).chat(MSGS) == "from m2"
    assert calls == ["m1", "m2"] and _env == []


def test_429_twice_falls_back_to_second_groq_model(monkeypatch):
    e = lambda: FakeAPIError(429, "tokens per minute")  # noqa: E731
    calls = _sdk(monkeypatch, {"m1": [e(), e()], "m2": [_completion("from m2")]})
    assert GroqClient(_config()).chat(MSGS) == "from m2"
    assert calls == ["m1", "m1", "m2"]


def test_5xx_retried_then_ok(monkeypatch):
    calls = _sdk(monkeypatch, {"m1": [FakeAPIError(503, "overloaded"), _completion("ok")]})
    assert GroqClient(_config()).chat(MSGS) == "ok"
    assert calls == ["m1", "m1"]


def test_timeout_retried_then_falls_back(monkeypatch):
    calls = _sdk(monkeypatch, {"m1": [APITimeoutError("t"), APITimeoutError("t")],
                               "m2": [_completion("from m2")]})
    assert GroqClient(_config()).chat(MSGS) == "from m2"
    assert calls == ["m1", "m1", "m2"]


def test_404_model_not_found_is_not_retried(monkeypatch, _env):
    """The 2026-08-16 incident shape: a dead model must move on immediately."""
    calls = _sdk(monkeypatch, {"m1": [FakeAPIError(404, "model_not_found")],
                               "m2": [_completion("from m2")]})
    assert GroqClient(_config()).chat(MSGS) == "from m2"
    assert calls == ["m1", "m2"] and _env == []


def test_daily_quota_429_is_not_retried_on_same_model(monkeypatch, _env):
    calls = _sdk(monkeypatch, {"m1": [FakeAPIError(429, "on tokens per day (TPD): Limit 200000")],
                               "m2": [_completion("from m2")]})
    assert GroqClient(_config()).chat(MSGS) == "from m2"
    assert calls == ["m1", "m2"] and _env == []


def test_empty_content_is_a_failure_and_moves_to_next_model(monkeypatch):
    calls = _sdk(monkeypatch, {"m1": [_completion(None)], "m2": [_completion("from m2")]})
    assert GroqClient(_config()).chat(MSGS) == "from m2"
    assert calls == ["m1", "m2"]


def test_everything_fails_returns_busy_message(monkeypatch):
    _sdk(monkeypatch, {"m1": [FakeAPIError(500, "x")] * 2, "m2": [FakeAPIError(500, "x")] * 2})
    result = GroqClient(_config()).chat(MSGS)
    assert result == LLM_BUSY_MESSAGE and result.strip()


def test_everything_empty_returns_busy_message(monkeypatch):
    _sdk(monkeypatch, {"m1": [_completion("  ")], "m2": [_completion(None)]})
    assert GroqClient(_config()).chat(MSGS) == LLM_BUSY_MESSAGE


def test_zero_budget_never_calls_groq_and_returns_busy(monkeypatch):
    calls = _sdk(monkeypatch, {"m1": [], "m2": []})
    assert GroqClient(_config(groq_chain_budget_seconds=0)).chat(MSGS) == LLM_BUSY_MESSAGE
    assert calls == []


# ------------------------------------------------------------- chat_stream()

def test_stream_primary_ok(monkeypatch):
    calls = _sdk(monkeypatch, {"m1": [_chunks("hel", "lo")]})
    assert "".join(GroqClient(_config()).chat_stream(MSGS)) == "hello"
    assert calls == ["m1"]


def test_stream_429_retries_once_then_ok(monkeypatch, _env):
    calls = _sdk(monkeypatch, {"m1": [FakeAPIError(429, "tokens per minute"), _chunks("ok")]})
    assert "".join(GroqClient(_config()).chat_stream(MSGS)) == "ok"
    assert calls == ["m1", "m1"] and len(_env) == 1


def test_stream_falls_back_to_second_groq_model(monkeypatch):
    calls = _sdk(monkeypatch, {"m1": [FakeAPIError(404, "x")], "m2": [_chunks("from m2")]})
    assert "".join(GroqClient(_config()).chat_stream(MSGS)) == "from m2"
    assert calls == ["m1", "m2"]


def test_stream_empty_stream_moves_to_next_model(monkeypatch):
    calls = _sdk(monkeypatch, {"m1": [_chunks()], "m2": [_chunks("from m2")]})
    assert "".join(GroqClient(_config()).chat_stream(MSGS)) == "from m2"
    assert calls == ["m1", "m2"]


def test_stream_everything_fails_yields_busy_message_not_sentinel_or_empty(monkeypatch):
    _sdk(monkeypatch, {"m1": [FakeAPIError(500, "x")] * 2, "m2": [FakeAPIError(500, "x")] * 2})
    tokens = list(GroqClient(_config()).chat_stream(MSGS))
    assert tokens == [LLM_BUSY_MESSAGE]


def test_stream_everything_empty_yields_busy_message(monkeypatch):
    _sdk(monkeypatch, {"m1": [_chunks()], "m2": [_chunks()]})
    assert list(GroqClient(_config()).chat_stream(MSGS)) == [LLM_BUSY_MESSAGE]


def test_stream_failure_after_first_token_still_signals_error(monkeypatch):
    def broken():
        yield _chunks("par")[0]
        raise APITimeoutError("dropped")

    calls = _sdk(monkeypatch, {"m1": [], "m2": []})
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: broken())))
    monkeypatch.setattr("groq.Groq", lambda api_key, timeout, max_retries: fake)
    tokens = list(GroqClient(_config()).chat_stream(MSGS))
    assert tokens == ["par", STREAM_ERROR_SENTINEL] and calls == []


# ------------------------------------------------------------- classification

@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (FakeAPIError(429, "tokens per minute (TPM)"), "rate_limit"),
        (FakeAPIError(429, "tokens per day (TPD)"), "daily_quota"),
        (FakeAPIError(503, ""), "server"),
        (APITimeoutError("t"), "timeout"),
        (FakeAPIError(404, "model_not_found"), "fatal"),
        (Exception("Error code: 429 - ... tokens per minute ... try again in 7.5s"), "rate_limit"),
        (ValueError("boom"), "fatal"),
    ],
)
def test_classify_error(exc, kind):
    assert classify_error(exc).kind == kind


def test_retry_wait_is_none_for_non_retryable_and_over_cap():
    assert retry_wait_seconds(classify_error(FakeAPIError(404)), 8) is None
    assert retry_wait_seconds(classify_error(FakeAPIError(429, "x", {"retry-after": "30"})), 8) is None
    assert 1.0 <= retry_wait_seconds(classify_error(FakeAPIError(503)), 8) <= 2.0


# ------------------------------------------- B4: malformed response payloads

def test_empty_choices_is_a_failed_model_not_an_exception(monkeypatch):
    bad = SimpleNamespace(choices=[], usage=None)
    calls = _sdk(monkeypatch, {"m1": [bad], "m2": [_completion("from m2")]})
    assert GroqClient(_config()).chat(MSGS) == "from m2"
    assert calls == ["m1", "m2"]


def test_empty_choices_everywhere_returns_busy_message(monkeypatch):
    bad = SimpleNamespace(choices=[], usage=None)
    _sdk(monkeypatch, {"m1": [bad], "m2": [bad]})
    assert GroqClient(_config()).chat(MSGS) == LLM_BUSY_MESSAGE


# ------------- no OpenRouter tier: every Groq failure kind ends in the busy message

_ALL_FAIL = {
    "429-rate-limit": lambda: FakeAPIError(429, "tokens per minute", {"retry-after": "3"}),
    "429-daily": lambda: FakeAPIError(429, "on tokens per day (TPD)"),
    "5xx": lambda: FakeAPIError(503, "overloaded"),
    "timeout": lambda: APITimeoutError("t"),
    "404": lambda: FakeAPIError(404, "model_not_found"),
}


def _forbid_openrouter(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """OPENROUTER_API_KEY is set, but constructing any OpenRouter client fails the test."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    boom = MagicMock(side_effect=AssertionError("OpenRouter must not be consulted"))
    monkeypatch.setattr("openai.OpenAI", boom)
    monkeypatch.setattr("src.llm.openrouter_client.OpenRouterClient", boom)
    return boom


@pytest.mark.parametrize("kind", sorted(_ALL_FAIL))
def test_every_groq_model_failing_chat_returns_busy_never_empty(monkeypatch, kind):
    mk = _ALL_FAIL[kind]
    calls = _sdk(monkeypatch, {"m1": [mk(), mk()], "m2": [mk(), mk()]})
    boom = _forbid_openrouter(monkeypatch)
    result = GroqClient(_config()).chat(MSGS)
    assert result == LLM_BUSY_MESSAGE and result.strip()
    assert set(calls) == {"m1", "m2"}  # the whole Groq chain was tried first
    boom.assert_not_called()


@pytest.mark.parametrize("kind", sorted(_ALL_FAIL))
def test_every_groq_model_failing_stream_yields_busy_message(monkeypatch, kind):
    mk = _ALL_FAIL[kind]
    _sdk(monkeypatch, {"m1": [mk(), mk()], "m2": [mk(), mk()]})
    boom = _forbid_openrouter(monkeypatch)
    assert list(GroqClient(_config()).chat_stream(MSGS)) == [LLM_BUSY_MESSAGE]
    boom.assert_not_called()


def test_rate_limited_chain_ends_fast_with_bounded_sleeps(monkeypatch, _env):
    rl = _ALL_FAIL["429-rate-limit"]
    _sdk(monkeypatch, {"m1": [rl(), rl()], "m2": [rl(), rl()]})
    t0 = time.monotonic()
    assert GroqClient(_config()).chat(MSGS) == LLM_BUSY_MESSAGE
    assert time.monotonic() - t0 < 5
    assert sum(_env) < 15 and max(_env) < 10  # only the Groq 1-retry waits


def test_chain_collapse_to_single_model_is_visible(monkeypatch):
    """Config with one model has a 1-long chain; the shipped one must have >= 2."""
    assert GroqClient(_config())._chain == ["m1", "m2"]
    assert GroqClient(_config(groq_fallback_models=[]))._chain == ["m1"]
