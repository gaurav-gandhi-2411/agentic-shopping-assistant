"""GroqClient.complete(): primary-model-only call returning finish_reason, token usage and typed
errors (used by the reranker). Fake SDK, fake clock: no network, no real sleeps."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.llm.client import GroqClient, LLMCallError

MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


class FakeStatusError(Exception):
    def __init__(self, status: int, text: str) -> None:
        super().__init__(f"Error code: {status} - {text}")
        self.status_code = status


def _resp(text: str | None, finish: str = "stop", reasoning: int | None = 7) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(
            prompt_tokens=600, completion_tokens=90,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )


def _client(monkeypatch: pytest.MonkeyPatch, results: list[Any], **llm: Any) -> tuple[
    GroqClient, list[dict[str, Any]], list[float]
]:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr("random.uniform", lambda a, b: 0.0)
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm.client.time.sleep", lambda s: sleeps.append(s))
    calls: list[dict[str, Any]] = []

    def create(**kw: Any) -> Any:
        calls.append(kw)
        item = results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("groq.Groq", lambda api_key, timeout, max_retries: fake)
    cfg = {"provider": "groq", "temperature": 0.2, "max_tokens": 400, "timeout_seconds": 60,
           "groq_model": "openai/gpt-oss-20b", "groq_fallback_models": ["m2"],
           "groq_reasoning_effort": "medium", **llm}
    return GroqClient({"llm": cfg}), calls, sleeps


def test_returns_text_finish_reason_and_all_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls, _ = _client(monkeypatch, [_resp("3,1,2")])
    out = client.complete(MSGS, temperature=0, max_tokens=400, reasoning_effort="low")
    assert (out.text, out.finish_reason) == ("3,1,2", "stop")
    assert (out.prompt_tokens, out.completion_tokens, out.reasoning_tokens) == (600, 90, 7)
    assert out.model == "openai/gpt-oss-20b"
    kw = calls[0]
    # the request actually sent: primary model, temperature 0, the cap, explicit low effort
    assert (kw["model"], kw["temperature"], kw["max_tokens"], kw["reasoning_effort"]) == (
        "openai/gpt-oss-20b", 0, 400, "low")
    assert len(calls) == 1  # never walks the fallback chain


def test_empty_content_is_returned_not_retried_or_chained(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls, _ = _client(monkeypatch, [_resp(None, finish="stop")])
    out = client.complete(MSGS, temperature=0, max_tokens=400)
    assert (out.text, out.finish_reason) == ("", "stop")
    assert len(calls) == 1
    assert calls[0]["reasoning_effort"] == "medium"  # falls back to the configured effort


def test_length_finish_reason_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch, [_resp("", finish="length")])
    assert client.complete(MSGS, temperature=0, max_tokens=400).finish_reason == "length"


def test_effort_not_sent_to_non_gpt_oss_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls, _ = _client(monkeypatch, [_resp("1")], groq_model="qwen/x")
    client.complete(MSGS, temperature=0, max_tokens=10, reasoning_effort="low")
    assert "reasoning_effort" not in calls[0]


def test_daily_quota_429_is_typed_and_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    err = FakeStatusError(429, "Rate limit reached ... tokens per day (TPD): Limit 200000")
    client, calls, _ = _client(monkeypatch, [err])
    with pytest.raises(LLMCallError) as ei:
        client.complete(MSGS, temperature=0, max_tokens=400)
    assert (ei.value.kind, ei.value.status) == ("daily_quota", 429)
    assert len(calls) == 1


def test_rate_limit_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls, sleeps = _client(
        monkeypatch, [FakeStatusError(429, "tokens per minute (TPM)"), _resp("1,2")]
    )
    assert client.complete(MSGS, temperature=0, max_tokens=400).text == "1,2"
    assert len(calls) == 2 and sleeps == [1.0]


def test_rate_limit_twice_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    e = FakeStatusError(429, "tokens per minute (TPM)")
    client, calls, _ = _client(monkeypatch, [e, e])
    with pytest.raises(LLMCallError) as ei:
        client.complete(MSGS, temperature=0, max_tokens=400)
    assert ei.value.kind == "rate_limit" and len(calls) == 2


def test_retry_false_makes_a_single_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls, _ = _client(monkeypatch, [FakeStatusError(503, "overloaded"), _resp("1")])
    with pytest.raises(LLMCallError) as ei:
        client.complete(MSGS, temperature=0, max_tokens=400, retry=False)
    assert (ei.value.kind, len(calls)) == ("server", 1)


def test_fatal_error_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch, [FakeStatusError(404, "model_not_found")])
    with pytest.raises(LLMCallError) as ei:
        client.complete(MSGS, temperature=0, max_tokens=400)
    assert (ei.value.kind, ei.value.status) == ("fatal", 404)


def test_malformed_payload_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _client(monkeypatch, [SimpleNamespace(choices=[], usage=None)])
    with pytest.raises(LLMCallError) as ei:
        client.complete(MSGS, temperature=0, max_tokens=400)
    assert ei.value.kind == "malformed"
