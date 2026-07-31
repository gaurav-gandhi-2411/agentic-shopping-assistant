"""
GroqClient -> OpenRouter -> Gemini automatic fallback chain on TPD (daily
quota) exhaustion.

All Groq/OpenRouter/Gemini SDK calls are mocked — no live API calls, no
network. Covers the 2026-07-29/30 incident: Groq's TPD retry-and-wait loop
caused multi-minute hangs that the WS turn-deadline had to cut off. The fix
falls over to OpenRouter immediately instead of waiting on the same exhausted
provider. Gemini was added 2026-07-31 as a second fallback tier, and as a
safety net for OpenRouter's free reasoning model (gpt-oss-20b:free)
returning empty content when max_tokens is too tight for it to finish its
hidden reasoning.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import GroqClient

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
            "gemini_model": "gemini-2.5-flash",
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
    """Full chain configured: Groq, OpenRouter, and Gemini all have keys."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


@pytest.fixture
def groq_env_no_openrouter(monkeypatch):
    """Nothing configured beyond Groq itself — exercises the legacy
    wait-and-retry path (zero fallback tiers available)."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def groq_env_no_gemini(monkeypatch):
    """OpenRouter configured, Gemini not — used to exercise "chain
    configured but every tier fails" falling through to legacy retry."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def _patch_groq_sdk(monkeypatch, create_side_effect):
    mock_create = MagicMock(side_effect=create_side_effect)
    fake_sdk_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
    )
    monkeypatch.setattr(
        "groq.Groq", lambda api_key, timeout: fake_sdk_client
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


def _fake_gemini_response(text, prompt_tokens=10, candidates_tokens=5, cached_tokens=0):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidates_tokens,
            cached_content_token_count=cached_tokens,
        ),
    )


def _fake_gemini_stream_chunk(text, prompt_tokens=None, candidates_tokens=None):
    usage = None
    if prompt_tokens is not None:
        usage = SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidates_tokens,
            cached_content_token_count=0,
        )
    return SimpleNamespace(text=text, usage_metadata=usage)


def _patch_gemini_sdk(
    monkeypatch,
    generate_content_return=None,
    generate_content_side_effect=None,
    stream_return=None,
):
    # No history in these tests' single-user-message fixtures, so
    # GeminiClient.chat()/chat_stream() always take the no-history branch
    # (models.generate_content / models.generate_content_stream) — see
    # src/llm/gemini_client.py::_split_messages.
    mock_generate = MagicMock(
        return_value=generate_content_return, side_effect=generate_content_side_effect
    )
    mock_stream = MagicMock(return_value=stream_return)
    fake_gemini_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=mock_generate, generate_content_stream=mock_stream)
    )
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: fake_gemini_client)
    return mock_generate, mock_stream


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


def test_chat_tpm_error_does_not_fall_back(monkeypatch, groq_env):
    calls = {"groq": 0, "openrouter": 0}

    def groq_side_effect(*a, **k):
        calls["groq"] += 1
        raise Exception(TPM_ERROR)

    _patch_groq_sdk(monkeypatch, create_side_effect=groq_side_effect)

    def openrouter_side_effect(*a, **k):
        calls["openrouter"] += 1
        raise AssertionError("fallback should not be attempted for non-TPD errors")

    _patch_openrouter_sdk(monkeypatch, create_side_effect=openrouter_side_effect)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = GroqClient(_config())
    with pytest.raises(Exception, match="tokens per minute"):
        client.chat([{"role": "user", "content": "hi"}])

    assert calls["openrouter"] == 0
    assert calls["groq"] == 3  # initial + 2 TPM backoff retries, then raise


def test_chat_fallback_unavailable_preserves_legacy_wait_retry(
    monkeypatch, groq_env_no_openrouter
):
    call_count = {"n": 0}

    def groq_side_effect(*a, **k):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise Exception(TPD_ERROR)
        return _fake_groq_completion("recovered")

    _patch_groq_sdk(monkeypatch, create_side_effect=groq_side_effect)
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    client = GroqClient(_config())
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "recovered"
    assert slept, "expected legacy TPD wait when no fallback is configured"


def test_chat_falls_through_to_gemini_when_openrouter_fails(monkeypatch, groq_env):
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))
    mock_openrouter = _patch_openrouter_sdk(
        monkeypatch, create_side_effect=Exception("401 User not found.")
    )
    mock_gemini, _ = _patch_gemini_sdk(
        monkeypatch, generate_content_return=_fake_gemini_response("gemini response")
    )
    monkeypatch.setattr("time.sleep", lambda _s: None)

    client = GroqClient(_config())
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "gemini response"
    # Prove the chain was actually walked, not just that some response came
    # back from an unverified tier. OpenRouterClient.chat() itself retries
    # non-429 errors twice (1s, 3s backoff) before giving up — 3 calls total.
    assert mock_openrouter.call_count == 3
    assert mock_gemini.call_count == 1


def test_chat_falls_through_to_gemini_when_openrouter_returns_empty(monkeypatch, groq_env):
    """OpenRouter's free reasoning model (gpt-oss-20b:free) can burn its whole
    max_tokens budget on hidden reasoning and return content: None — this
    must advance to the next tier, not be treated as a successful (blank)
    response."""
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))
    mock_openrouter = _patch_openrouter_sdk(
        monkeypatch, create_return=_fake_groq_completion(None)
    )
    mock_gemini, _ = _patch_gemini_sdk(
        monkeypatch, generate_content_return=_fake_gemini_response("gemini response")
    )

    client = GroqClient(_config())
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "gemini response"
    assert mock_openrouter.call_count == 1
    assert mock_gemini.call_count == 1


def test_chat_stream_falls_through_to_gemini_when_openrouter_fails(monkeypatch, groq_env):
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))
    mock_openrouter = _patch_openrouter_sdk(
        monkeypatch, create_side_effect=Exception("401 User not found.")
    )
    gemini_chunks = [
        _fake_gemini_stream_chunk("gem"),
        _fake_gemini_stream_chunk("ini", prompt_tokens=9, candidates_tokens=3),
    ]
    mock_generate, mock_stream = _patch_gemini_sdk(
        monkeypatch, stream_return=iter(gemini_chunks)
    )

    client = GroqClient(_config())
    chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert "".join(chunks) == "gemini"
    assert mock_openrouter.call_count == 1
    assert mock_stream.call_count == 1


def test_chat_all_tiers_fail_falls_through_to_legacy_retry(monkeypatch, groq_env_no_gemini):
    """Chain configured (OpenRouter) but every tier fails, and Gemini isn't
    configured at all — must fall through to the SAME legacy wait-and-retry
    loop as the "nothing configured" case (test
    test_chat_fallback_unavailable_preserves_legacy_wait_retry), not raise
    immediately. See the "deliberately not distinguishing" comment in
    GroqClient.chat()."""
    calls = {"groq": 0}

    def groq_side_effect(*a, **k):
        calls["groq"] += 1
        if calls["groq"] < 2:
            raise Exception(TPD_ERROR)
        return _fake_groq_completion("recovered")

    _patch_groq_sdk(monkeypatch, create_side_effect=groq_side_effect)
    mock_openrouter = _patch_openrouter_sdk(
        monkeypatch, create_side_effect=Exception("401 User not found.")
    )
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    client = GroqClient(_config())
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "recovered"
    assert slept, "expected legacy TPD wait when the fallback chain is exhausted"
    # 3 = 1 initial + 2 internal OpenRouterClient backoff retries (see the
    # matching comment in test_chat_falls_through_to_gemini_when_openrouter_fails).
    assert mock_openrouter.call_count == 3


def test_get_fallback_caches_unavailable_result(monkeypatch, groq_env_no_openrouter):
    _patch_groq_sdk(monkeypatch, create_side_effect=Exception(TPD_ERROR))

    client = GroqClient(_config())
    assert client._get_openrouter_fallback() is None
    assert client._openrouter_unavailable is True
    # Second call must not re-attempt construction (would raise again if it did,
    # but more importantly this asserts the sticky-cache behaviour directly).
    assert client._get_openrouter_fallback() is None
    # Gemini is likewise unconfigured under this fixture.
    assert client._get_gemini_fallback() is None
    assert client._gemini_unavailable is True
