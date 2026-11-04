"""
GroqClient -> OpenRouter automatic fallback chain on TPD (daily quota)
exhaustion.

All Groq/OpenRouter SDK calls are mocked — no live API calls, no network.
Covers the 2026-07-29/30 incident: Groq's TPD retry-and-wait loop caused
multi-minute hangs that the WS turn-deadline had to cut off. The fix falls
over to OpenRouter immediately instead of waiting on the same exhausted
provider. (A Gemini tier existed 2026-07-31..W7 and was removed.)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import GroqClient
from src.llm.resilience import LLM_BUSY_MESSAGE

TPD_ERROR = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`llama-3.1-8b-instant` ... on tokens per day (TPD): Limit 500000, "
    "Used 499539, Requested 1378. Please try again in 2m38s.', "
    "'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)
TPM_ERROR = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`llama-3.1-8b-instant` ... on tokens per minute (TPM): Limit 6000.', "
    "'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)


def _config(with_openrouter_key: bool = True) -> dict:
    return {
        "llm": {
            "provider": "groq",
            "model": "llama3.1:8b",
            "temperature": 0.2,
            "max_tokens": 400,
            "timeout_seconds": 60,
            "groq_model": "llama-3.1-8b-instant",
            "openrouter_model": "openai/gpt-oss-20b:free",
        }
    }


def _fake_groq_completion(content: str, prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _fake_groq_stream_chunk(content: str | None, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))] if content is not None else [],
        usage=usage,
    )


@pytest.fixture
def groq_env(monkeypatch):
    """Full chain configured: Groq and OpenRouter both have keys."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")


@pytest.fixture
def groq_env_no_openrouter(monkeypatch):
    """Nothing configured beyond Groq itself — exercises the legacy
    wait-and-retry path (zero fallback tiers available)."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def _patch_groq_sdk(monkeypatch, create_side_effect):
    mock_create = MagicMock(side_effect=create_side_effect)
    fake_sdk_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
    )
    monkeypatch.setattr(
        "groq.Groq", lambda api_key, timeout, max_retries: fake_sdk_client
    )
    return mock_create


def _patch_openrouter_sdk(monkeypatch, create_return=None, create_side_effect=None):
    mock_create = MagicMock(return_value=create_return, side_effect=create_side_effect)
    fake_openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
    )
    monkeypatch.setattr(
        "openai.OpenAI", lambda api_key, base_url, timeout: fake_openai_client
    )
    return mock_create


def test_chat_falls_back_to_openrouter_on_tpd_exhaustion(monkeypatch, groq_env):
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))
    _patch_openrouter_sdk(
        monkeypatch, create_return=_fake_groq_completion("fallback response")
    )

    client = GroqClient(_config())
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "fallback response"


def test_chat_stream_falls_back_to_openrouter_on_tpd_exhaustion(monkeypatch, groq_env):
    def groq_raises(*a, **k):
        raise Exception(TPD_ERROR)

    _patch_groq_sdk(monkeypatch, create_side_effect=groq_raises)

    fallback_chunks = [
        _fake_groq_stream_chunk("hel"),
        _fake_groq_stream_chunk("lo"),
        _fake_groq_stream_chunk(None, usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2)),
    ]
    _patch_openrouter_sdk(monkeypatch, create_return=iter(fallback_chunks))

    client = GroqClient(_config())
    chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert "".join(chunks) == "hello"


def test_chat_no_fallback_configured_returns_busy_message_not_a_wait(
    monkeypatch, groq_env_no_openrouter
):
    """Replaces the legacy 10 x 60 s TPD wait loop, which could blow the 90 s WS
    turn deadline: with no fallback tier configured the caller now gets the
    explicit busy message immediately."""
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    client = GroqClient(_config())
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == LLM_BUSY_MESSAGE
    assert not slept


def test_chat_openrouter_empty_content_is_a_failed_tier(monkeypatch, groq_env):
    """OpenRouter's free reasoning model (gpt-oss-20b:free) can burn its whole
    max_tokens budget on hidden reasoning and return content: None — that must
    count as a failed tier (busy message, chain exhausted), not as a successful
    blank response."""
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))
    mock_openrouter = _patch_openrouter_sdk(
        monkeypatch, create_return=_fake_groq_completion(None)
    )

    client = GroqClient(_config())
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == LLM_BUSY_MESSAGE
    assert mock_openrouter.call_count == 1


def test_fallback_tiers_are_openrouter_only(monkeypatch, groq_env):
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))
    _patch_openrouter_sdk(monkeypatch, create_return=_fake_groq_completion("x"))

    client = GroqClient(_config())

    assert [name for name, _ in client._iter_fallback_tiers()] == ["openrouter"]


def test_chat_all_tiers_fail_returns_busy_message(monkeypatch, groq_env):
    """OpenRouter configured but failing: the chain ends in
    the busy message (previously fell into the legacy wait-and-retry loop)."""
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))
    mock_openrouter = _patch_openrouter_sdk(
        monkeypatch, create_side_effect=Exception("401 User not found.")
    )
    monkeypatch.setattr("time.sleep", lambda s: None)

    client = GroqClient(_config())
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == LLM_BUSY_MESSAGE
    # Bounded fallback tier: one attempt, no internal retries.
    assert mock_openrouter.call_count == 1


def test_get_fallback_caches_unavailable_result(monkeypatch, groq_env_no_openrouter):
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))

    client = GroqClient(_config())
    assert client._get_openrouter_fallback() is None
    assert client._openrouter_unavailable is True
    # Second call must not re-attempt construction (would raise again if it did,
    # but more importantly this asserts the sticky-cache behaviour directly).
    assert client._get_openrouter_fallback() is None
