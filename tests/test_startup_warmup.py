"""Startup warm-up: runs, never blocks readiness on failure, is bounded, needs no LLM/network."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pandas as pd
import pytest

from api import main as api_main
from api import warmup
from src.utils import timing_log


class _Retriever:
    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, int | None, dict | None]] = []
        self.fail_on = fail_on

    def search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> list:
        self.calls.append((query, top_k, filters))
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("boom in search")
        return []


@pytest.fixture(autouse=True)
def _no_outfit_graph(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Default: don't build the real graph (tests that want it override this)."""
    turns: list[int] = []
    monkeypatch.setattr(warmup, "_outfit_turn", lambda *a, **k: turns.append(1))
    return turns


def test_warmup_runs_searches_for_both_genders_and_one_outfit_turn(
    _no_outfit_graph: list[int],
) -> None:
    r = _Retriever()
    res = warmup.run_warmup(r, pd.DataFrame(), {}, timeout_s=5)
    assert res.ok and not res.timed_out and res.error is None
    assert [c[2] for c in r.calls] == [None, {"gender": "women"}, {"gender": "men"}]
    assert _no_outfit_graph == [1]
    assert set(res.steps_ms) == {"search_plain", "search_women", "search_men", "outfit_turn"}


def test_extra_steps_run_after_the_turn_and_a_failing_one_is_contained() -> None:
    order: list[str] = []

    def clip() -> None:
        order.append("clip")

    res = warmup.run_warmup(
        _Retriever(), pd.DataFrame(), {}, timeout_s=5, extra_steps={"clip": clip}
    )
    assert res.ok and order == ["clip"] and "clip" in res.steps_ms

    def broken() -> None:
        raise OSError("disk gone")

    res2 = warmup.run_warmup(
        _Retriever(), pd.DataFrame(), {}, timeout_s=5, extra_steps={"clip": broken}
    )
    assert not res2.ok and "disk gone" in (res2.error or "")


def test_failure_never_raises_and_is_logged_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    r = _Retriever(fail_on="kurta")
    with caplog.at_level(logging.WARNING, logger=warmup.logger.name):
        res = warmup.run_warmup(r, pd.DataFrame(), {}, timeout_s=5)  # must not raise
    assert not res.ok and not res.timed_out
    assert res.error is not None and "boom in search" in res.error
    failed = [m for m in caplog.records if "warm-up failed" in m.getMessage()]
    assert len(failed) == 1


def test_failure_in_the_outfit_turn_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise ValueError("graph exploded")

    monkeypatch.setattr(warmup, "_outfit_turn", boom)
    res = warmup.run_warmup(_Retriever(), pd.DataFrame(), {}, timeout_s=5)
    assert not res.ok and "graph exploded" in (res.error or "")


def test_bounded_by_the_timeout_and_startup_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    monkeypatch.setattr(warmup, "_run_steps", lambda *a, **k: release.wait(10))
    t0 = time.perf_counter()
    res = warmup.run_warmup(_Retriever(), pd.DataFrame(), {}, timeout_s=0.3)
    elapsed = time.perf_counter() - t0
    release.set()  # let the daemon thread finish
    assert res.timed_out and not res.ok
    assert elapsed < 2.0  # returned at the ceiling, not after the 10 s the work wanted


def test_env_switches() -> None:
    assert warmup.warmup_enabled({}) is True
    for off in ("0", "false", "OFF", "no"):
        assert warmup.warmup_enabled({"STARTUP_WARMUP": off}) is False
    assert warmup.warmup_enabled({"STARTUP_WARMUP": "1"}) is True
    assert warmup.warmup_timeout_s({}) == warmup.DEFAULT_TIMEOUT_S == 30.0
    assert warmup.warmup_timeout_s({"STARTUP_WARMUP_TIMEOUT_S": "12.5"}) == 12.5
    assert warmup.warmup_timeout_s({"STARTUP_WARMUP_TIMEOUT_S": "junk"}) == 30.0
    assert warmup.warmup_timeout_s({"STARTUP_WARMUP_TIMEOUT_S": "-3"}) == 30.0


# --- the lifespan wrapper: readiness is never blocked ---------------------------------------------


def test_lifespan_wrapper_swallows_even_an_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*a: Any, **k: Any) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(warmup, "run_warmup", explode)
    monkeypatch.delenv("STARTUP_WARMUP", raising=False)
    assert api_main._run_startup_warmup(_Retriever(), pd.DataFrame(), {}) is None  # no raise


def test_lifespan_wrapper_skips_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(warmup, "run_warmup", lambda *a, **k: called.append(1))
    monkeypatch.setenv("STARTUP_WARMUP", "0")
    assert api_main._run_startup_warmup(_Retriever(), pd.DataFrame(), {}) is None
    assert called == []


# --- no LLM, no network, no key ------------------------------------------------------------------------


def test_warmup_llm_always_raises_so_template_paths_are_used() -> None:
    llm = warmup.WarmupLLM()
    with pytest.raises(RuntimeError):
        llm.generate("x")
    with pytest.raises(RuntimeError):
        llm.generate_stream("x")


def test_real_graph_turn_with_the_stub_llm_completes_without_keys_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real _outfit_turn (compiled graph + stub LLM) with composition stubbed: template
    rationales are used and the turn finishes with no provider key in the environment."""
    from src.agents import graph as graph_mod
    from src.agents.outfit import composer

    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    # undo the autouse stub: exercise the real function
    monkeypatch.undo()
    seed = {"article_id": "s", "display_name": "Seed", "prod_name": "Seed Lehenga",
            "colour": "red", "price_inr": 600.0, "store": "snitch", "product_type": "lehenga"}

    def probe(**kw: Any) -> dict:
        return {"look_id": "l", "seed_item": seed, "complements": [], "outfit_rationale": "r",
                "empty_slots": [], "occasion": "sangeet", "gender": "women",
                "budget_total_inr": 600.0}

    seen: dict[str, Any] = {}
    real_generate = graph_mod.generate_rationales

    def spy(*a: Any, **k: Any) -> Any:
        seen["llm"] = a[1] if len(a) > 1 else k.get("llm")
        return real_generate(*a, **k)

    monkeypatch.setattr(graph_mod, "compose_outfit_tool", probe)
    monkeypatch.setattr(graph_mod, "generate_rationales", spy)
    monkeypatch.setattr(composer, "compose_biased_look", lambda **kw: None)
    catalogue = pd.DataFrame(
        {"article_id": ["a"], "colour_group_name": ["Red"], "product_type_name": ["Kurta"]}
    )
    cfg = {"agent": {"max_iterations": 3},
           "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
           "retrieval": {"final_k": 10, "top_k": 50, "rrf_k": 60}}
    warmup._outfit_turn(_Retriever(), catalogue, cfg)  # must not raise
    assert isinstance(seen["llm"], warmup.WarmupLLM)  # the stub, never a real provider client


def test_timing_log_first_call_logs_once(caplog: pytest.LogCaptureFixture) -> None:
    timing_log.reset_for_tests()
    with caplog.at_level(logging.INFO, logger="timing"):
        for _ in range(3):
            with timing_log.first_call("unit_stage"):
                pass
        with timing_log.stage("unit_startup"):
            pass
    msgs = [r.getMessage() for r in caplog.records]
    assert sum(m.startswith("first_call unit_stage ms=") for m in msgs) == 1
    assert any(m.startswith("startup_stage unit_startup ms=") for m in msgs)
    rec = next(r for r in caplog.records if r.getMessage().startswith("first_call"))
    assert rec.__dict__["event"] == "first_call" and rec.__dict__["stage"] == "unit_stage"
