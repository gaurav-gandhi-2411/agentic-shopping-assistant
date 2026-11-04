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
        "openrouter_model": "or",
    }
    cfg.update(llm)
    return {"llm": cfg}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
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


# ------------------------------------------------- B3: lifespan wiring (api/main.py)

@pytest.fixture
def light_startup(monkeypatch, tmp_path):
    """Run the real lifespan with the heavy index loading stubbed out."""
    import pandas as pd

    import src.retrieval.dense_search as dense
    import src.retrieval.hybrid_search as hybrid
    import src.retrieval.index_store as store
    import src.retrieval.sparse_search as sparse

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("INDEX_STORE_URI", raising=False)
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.setattr(store, "ensure_index_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pd.DataFrame())
    stub = SimpleNamespace(load=lambda *a, **k: SimpleNamespace(index=None))
    monkeypatch.setattr(dense, "DenseRetriever", stub)
    monkeypatch.setattr(sparse, "SparseRetriever", stub)
    monkeypatch.setattr(hybrid, "HybridRetriever", lambda *a, **k: SimpleNamespace(dense=stub.load()))


def test_lifespan_aborts_startup_when_primary_model_not_listed(monkeypatch, light_startup):
    # Real config.yaml primary is openai/gpt-oss-20b; Groq no longer lists it.
    _sdk(monkeypatch, models=["openai/gpt-oss-120b"])
    with pytest.raises(LLMStartupError, match="gpt-oss-20b"), TestClient(app):
        pass


def test_lifespan_calls_verify_models_before_serving(monkeypatch, light_startup):
    """Primary listed: startup passes the model check and proceeds past it (a later
    unrelated init step is made to stop the run so the test stays light)."""
    _sdk(monkeypatch, models=["openai/gpt-oss-20b"])
    seen: list[str] = []
    orig = GroqClient.verify_models

    def spy(self: GroqClient, *a: Any, **k: Any) -> str:
        seen.append(orig(self, *a, **k))
        return seen[-1]

    monkeypatch.setattr(GroqClient, "verify_models", spy)
    monkeypatch.setattr("api.main._build_session_store",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop-after-llm-check")))
    with pytest.raises(RuntimeError, match="stop-after-llm-check"), TestClient(app):
        pass
    assert seen == ["verified"]


def test_probe_cache_keeps_result_of_probe_that_finishes_after_timeout():
    import threading

    release = threading.Event()
    done = threading.Event()
    calls = []

    def slow() -> dict:
        calls.append(1)
        release.wait(5)
        return {"tier": "groq", "model": "m1", "latency_ms": 1}

    cache = ProbeCache(ttl_s=60, timeout_s=0.05)
    with pytest.raises(RuntimeError, match="timed out"):
        cache.get(slow)
    cache._inflight.add_done_callback(lambda _f: done.set())
    release.set()
    assert done.wait(5)
    assert cache.get(slow)["model"] == "m1" and len(calls) == 1  # served from cache


def test_late_probe_result_does_not_overwrite_a_newer_cached_result():
    from concurrent.futures import Future

    cache = ProbeCache(ttl_s=60)
    cache.get(lambda: {"tier": "groq", "model": "newer", "latency_ms": 1})
    stale = Future()  # an OLDER probe finishing after the newer one was cached
    stale.set_result((0.0, {"tier": "groq", "model": "older", "latency_ms": 1}, None))
    cache._cache_late_result(stale)
    assert cache._cached[1]["model"] == "newer"
    assert cache.get(lambda: pytest.fail("must be served from cache"))["model"] == "newer"


def test_late_callback_attached_once_and_stamped_with_completion_time():
    import threading
    import time as _t

    release = threading.Event()
    cache = ProbeCache(ttl_s=60, timeout_s=0.05)

    def slow() -> dict:
        release.wait(5)
        return {"tier": "groq", "model": "m1", "latency_ms": 1}

    for _ in range(3):  # three timeouts on the same in-flight probe
        with pytest.raises(RuntimeError, match="timed out"):
            cache.get(slow)
    fut = cache._inflight
    assert len(fut._done_callbacks) == 1
    done = threading.Event()
    fut.add_done_callback(lambda _f: done.set())
    t_release = _t.monotonic()
    release.set()
    assert done.wait(5)
    assert cache._cached[0] >= t_release and cache._cached[0] <= _t.monotonic()


def test_late_callback_on_already_completed_future_does_not_deadlock():
    """The probe may finish between the timeout and callback registration, which
    runs the callback inline; registering under the (non-reentrant) lock deadlocked."""
    import threading
    from concurrent.futures import Future
    from concurrent.futures import TimeoutError as FutTimeout

    class _FinishesMidGet(Future):
        """Looks in-flight to the first done() check, times out on the timed
        result() wait, yet is complete by the time the callback is registered."""

        _first = True

        def done(self) -> bool:
            if self._first:
                self._first = False
                return False
            return super().done()

        def result(self, timeout=None):
            if timeout is not None:
                raise FutTimeout
            return super().result()

    cache = ProbeCache(ttl_s=60, timeout_s=0.05)
    fut = _FinishesMidGet()
    fut.set_result((1.0, {"tier": "groq", "model": "m1", "latency_ms": 1}, None))
    cache._inflight = fut
    outcome: list = []

    def call() -> None:
        try:
            cache.get(lambda: pytest.fail("must not start a new probe"))
        except RuntimeError as exc:
            outcome.append(str(exc))

    t = threading.Thread(target=call, daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "deadlock: callback re-acquired the held lock"
    assert "timed out" in outcome[0]
    assert cache._cached[1]["model"] == "m1"  # inline callback cached the finished probe
