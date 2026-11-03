"""W4b: startup /models check, chain probe, and /readyz real-LLM-call caching.

All provider access is mocked: GET /models and chat completions never touch the
network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import api.deps as deps
from api.main import app
from api.routes import health
from src.llm.client import GroqClient
from src.llm.resilience import LLMStartupError, ProbeCache


class FakeAPIError(Exception):
    def __init__(self, status: int, body: str = "") -> None:
        super().__init__(f"Error code: {status} - {body}")
        self.status_code = status
        self.response = SimpleNamespace(headers={})


class APITimeoutError(Exception):
    """Same class name as groq.APITimeoutError (classification is by name)."""


def _completion(content: str | None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
    )


def _config(**llm: Any) -> dict:
    cfg = {
        "provider": "groq", "temperature": 0.2, "max_tokens": 400, "timeout_seconds": 60,
        "groq_model": "m1", "groq_fallback_models": ["m2", "m3"],
        "openrouter_model": "or", "gemini_model": "gem",
    }
    cfg.update(llm)
    return {"llm": cfg}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("time.sleep", lambda _s: None)


def _sdk(monkeypatch: pytest.MonkeyPatch, *, models: Any = None, create: Any = None) -> SimpleNamespace:
    """Patch groq.Groq. `models` is a list of ids or an Exception for GET /models."""
    opts: dict[str, Any] = {}

    def list_models() -> Any:
        if isinstance(models, Exception):
            raise models
        return SimpleNamespace(data=[SimpleNamespace(id=m) for m in models])

    def with_options(**kw: Any) -> Any:
        opts.update(kw)
        return SimpleNamespace(models=SimpleNamespace(list=list_models))

    fake = SimpleNamespace(
        with_options=with_options,
        chat=SimpleNamespace(completions=SimpleNamespace(create=create or MagicMock())),
        opts=opts,
    )
    monkeypatch.setattr("groq.Groq", lambda api_key, timeout, max_retries: fake)
    return fake


# ------------------------------------------------------------ startup check

def test_startup_primary_listed_keeps_listed_fallbacks_and_drops_unlisted(monkeypatch, caplog):
    fake = _sdk(monkeypatch, models=["m1", "m3", "other"])
    client = GroqClient(_config())
    with caplog.at_level("WARNING"):
        assert client.verify_models() == "verified"
    assert client._chain == ["m1", "m3"]  # m2 unlisted -> dropped
    assert "llm_fallback_model_unlisted" in caplog.text and '"m2"' in caplog.text
    assert fake.opts == {"timeout": 5.0, "max_retries": 0}


def test_startup_primary_missing_raises(monkeypatch):
    _sdk(monkeypatch, models=["m2", "m3"])
    with pytest.raises(LLMStartupError, match="m1"):
        GroqClient(_config()).verify_models()


@pytest.mark.parametrize("err", [APITimeoutError("t"), FakeAPIError(503, "x"),
                                 FakeAPIError(429, "rate"), ConnectionError("dns")])
def test_startup_models_endpoint_unreachable_continues_unverified(monkeypatch, caplog, err):
    """Documented fail-open: a Groq outage at deploy time must not block deploys."""
    _sdk(monkeypatch, models=err)
    client = GroqClient(_config())
    with caplog.at_level("ERROR"):
        assert client.verify_models() == "unverified"
    assert client._chain == ["m1", "m2", "m3"]  # chain untouched
    assert "llm_startup_check_unverified" in caplog.text


@pytest.mark.parametrize("status", [401, 403])
def test_startup_rejected_key_fails_closed(monkeypatch, status):
    _sdk(monkeypatch, models=FakeAPIError(status, "invalid_api_key"))
    with pytest.raises(LLMStartupError, match=str(status)):
        GroqClient(_config()).verify_models()


# -------------------------------------------------------------------- probe

def test_probe_primary_ok_uses_max_tokens_1(monkeypatch):
    create = MagicMock(return_value=_completion("p"))
    _sdk(monkeypatch, create=create)
    res = GroqClient(_config()).probe()
    assert res["tier"] == "groq" and res["model"] == "m1"
    assert create.call_args.kwargs["max_tokens"] == 1 and create.call_count == 1


def test_probe_accepts_empty_content_from_reasoning_model(monkeypatch):
    _sdk(monkeypatch, create=MagicMock(return_value=_completion(None)))
    assert GroqClient(_config()).probe()["model"] == "m1"


def test_probe_reports_fallback_model_when_primary_fails(monkeypatch):
    create = MagicMock(side_effect=[FakeAPIError(404, "gone"), _completion("p")])
    _sdk(monkeypatch, create=create)
    res = GroqClient(_config()).probe()
    assert res["model"] == "m2" and create.call_count == 2  # no retry sleeps in a probe


def test_probe_reports_openrouter_tier_when_all_groq_models_fail(monkeypatch):
    _sdk(monkeypatch, create=MagicMock(side_effect=FakeAPIError(500, "x")))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    or_create = MagicMock(return_value=_completion("p"))
    fake_or = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=or_create)))
    monkeypatch.setattr("openai.OpenAI", lambda api_key, base_url, timeout: fake_or)
    assert GroqClient(_config()).probe()["tier"] == "openrouter"


def test_probe_raises_when_whole_chain_fails(monkeypatch):
    _sdk(monkeypatch, create=MagicMock(side_effect=FakeAPIError(500, "x")))
    with pytest.raises(RuntimeError, match="all LLM tiers failed"):
        GroqClient(_config()).probe()


# --------------------------------------------------------------- ProbeCache

def test_probe_cache_second_call_within_ttl_makes_no_call():
    probe = MagicMock(return_value={"tier": "groq", "model": "m1", "latency_ms": 5})
    cache = ProbeCache(ttl_s=60)
    assert cache.get(probe) == cache.get(probe)
    assert probe.call_count == 1


def test_probe_cache_expires_after_ttl(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("src.llm.resilience.time.monotonic", lambda: now[0])
    probe = MagicMock(return_value={"tier": "groq", "model": "m1", "latency_ms": 5})
    cache = ProbeCache(ttl_s=60)
    cache.get(probe)
    now[0] += 61
    cache.get(probe)
    assert probe.call_count == 2


def test_probe_cache_caches_failures_too():
    probe = MagicMock(side_effect=RuntimeError("boom"))
    cache = ProbeCache(ttl_s=60)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="boom"):
            cache.get(probe)
    assert probe.call_count == 1


def test_probe_cache_times_out_instead_of_hanging():
    import threading

    release = threading.Event()
    cache = ProbeCache(ttl_s=60, timeout_s=0.05)
    with pytest.raises(RuntimeError, match="timed out"):
        cache.get(lambda: (release.wait(5), {"tier": "groq"})[1])
    release.set()


# ------------------------------------------------------------------ /readyz

class _ProbeLLM:
    model = "m1"

    def __init__(self, probe: Any) -> None:
        self.probe = probe


@pytest.fixture
def ready_client(monkeypatch):
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
    monkeypatch.setattr(deps, "get_retriever",
                        lambda: SimpleNamespace(dense=SimpleNamespace(index=None)))
    monkeypatch.setattr(deps, "get_catalogue_df", lambda: [])
    monkeypatch.setattr(deps, "get_db_engine", lambda: None)
    monkeypatch.setattr(health, "_llm_probe_cache", ProbeCache(ttl_s=60))
    return TestClient(app)


def test_readyz_makes_one_real_llm_call_then_serves_from_cache(monkeypatch, ready_client):
    probe = MagicMock(return_value={"tier": "groq", "model": "m1", "latency_ms": 42})
    monkeypatch.setattr(deps, "get_llm", lambda: _ProbeLLM(probe))
    first = ready_client.get("/readyz")
    second = ready_client.get("/readyz")
    assert first.status_code == second.status_code == 200
    assert first.json()["checks"]["llm"] == "ok (groq:m1, 42 ms)"
    assert probe.call_count == 1  # second hit within 60 s makes no LLM call


def test_readyz_flags_fallback_model(monkeypatch, ready_client):
    probe = MagicMock(return_value={"tier": "groq", "model": "m2", "latency_ms": 9})
    monkeypatch.setattr(deps, "get_llm", lambda: _ProbeLLM(probe))
    body = ready_client.get("/readyz").json()
    assert "FALLBACK" in body["checks"]["llm"] and "m2" in body["checks"]["llm"]


def test_readyz_degraded_503_when_whole_chain_fails(monkeypatch, ready_client):
    probe = MagicMock(side_effect=RuntimeError("all LLM tiers failed"))
    monkeypatch.setattr(deps, "get_llm", lambda: _ProbeLLM(probe))
    resp = ready_client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not ready"
    assert "all LLM tiers failed" in resp.json()["checks"]["llm"]
    ready_client.get("/readyz")
    assert probe.call_count == 1  # failure is cached as well


def test_readyz_without_probe_support_stays_ok(monkeypatch, ready_client):
    monkeypatch.setattr(deps, "get_llm", lambda: SimpleNamespace())  # e.g. OllamaClient
    resp = ready_client.get("/readyz")
    assert resp.status_code == 200 and resp.json()["checks"]["llm"] == "ok"
