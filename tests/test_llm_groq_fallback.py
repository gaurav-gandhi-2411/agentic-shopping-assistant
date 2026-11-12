"""
GroqClient on TPD (daily quota) exhaustion: the chain is Groq models, then the
explicit busy message. There is NO non-Groq fallback tier (the OpenRouter tier was
removed: `:free` not enforced, key validity unproven, account charge-ability
unknown; see DEPLOY.md "OpenRouter removed").

All Groq SDK calls are mocked — no live API calls, no network. Covers the
2026-07-29/30 incident: Groq's TPD retry-and-wait loop caused multi-minute hangs
that the WS turn-deadline had to cut off; the fix is to never wait on an exhausted
provider and to answer with a bounded busy message instead.
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
MSGS = [{"role": "user", "content": "hi"}]


def _config() -> dict:
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


@pytest.fixture
def groq_env_with_openrouter_key(monkeypatch):
    """OPENROUTER_API_KEY is present in the env: it must still never be used."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr("time.sleep", lambda s: None)


def _patch_groq_sdk(monkeypatch, create_side_effect):
    mock_create = MagicMock(side_effect=create_side_effect)
    fake_sdk_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
    )
    monkeypatch.setattr("groq.Groq", lambda api_key, timeout, max_retries: fake_sdk_client)
    return mock_create


def _forbid_openrouter(monkeypatch) -> MagicMock:
    """Any construction of the OpenAI SDK client or OpenRouterClient fails the test."""
    boom = MagicMock(side_effect=AssertionError("OpenRouter must not be consulted"))
    monkeypatch.setattr("openai.OpenAI", boom)
    monkeypatch.setattr("src.llm.openrouter_client.OpenRouterClient", boom)
    return boom


def test_chat_returns_busy_message_on_tpd_exhaustion_without_openrouter(
    monkeypatch, groq_env_with_openrouter_key
):
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))
    boom = _forbid_openrouter(monkeypatch)

    result = GroqClient(_config()).chat(MSGS)

    assert result == LLM_BUSY_MESSAGE
    boom.assert_not_called()


def test_chat_stream_yields_busy_message_on_tpd_exhaustion_without_openrouter(
    monkeypatch, groq_env_with_openrouter_key
):
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))
    boom = _forbid_openrouter(monkeypatch)

    chunks = list(GroqClient(_config()).chat_stream(MSGS))

    assert "".join(chunks) == LLM_BUSY_MESSAGE
    boom.assert_not_called()


def test_busy_message_is_immediate_not_a_wait(monkeypatch, groq_env_with_openrouter_key):
    """Replaces the legacy 10 x 60 s TPD wait loop: a daily-quota 429 gets no sleep."""
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    assert GroqClient(_config()).chat(MSGS) == LLM_BUSY_MESSAGE
    assert not slept


def test_client_has_no_openrouter_fallback_machinery() -> None:
    for name in ("_iter_fallback_tiers", "_get_openrouter_fallback", "_fallback_chat",
                 "_openrouter_client", "_fallback_tier_s"):
        assert not hasattr(GroqClient, name), name
