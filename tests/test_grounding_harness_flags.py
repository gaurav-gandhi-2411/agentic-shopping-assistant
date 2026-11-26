"""W10 harness additions for the V5 live eval: --offset/--limit slicing, LLM-call pacing, and the
stub agent mirroring respond_node's grounded rendering. Offline: mock LLM, no network."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.grounding import run as harness


def test_offset_limit_slice_and_zero_violations_on_the_mock_ungrounded_stub(tmp_path: Path) -> None:
    rc = harness.main(["--llm", "mock", "--agent", "stub", "--mock-mode", "ungrounded",
                       "--offset", "10", "--limit", "6", "--out-dir", str(tmp_path)])
    assert rc == 0
    report = json.loads(next(tmp_path.glob("grounding_eval_*.json")).read_text(encoding="utf-8"))
    assert report["n_prompts"] == 6 and report["n_runs"] == 12
    assert report["total_violations"] == 0 and report["measurement_valid"] is True


def test_offset_limit_defaults_run_every_prompt(tmp_path: Path) -> None:
    harness.main(["--llm", "mock", "--agent", "stub", "--mock-mode", "grounded",
                  "--path", "rest", "--out-dir", str(tmp_path)])
    report = json.loads(next(tmp_path.glob("grounding_eval_*.json")).read_text(encoding="utf-8"))
    assert report["n_prompts"] == 150


class _Inner:
    def generate(self, *a: Any, **k: Any) -> str:
        return "ok"


def test_counting_llm_paces_calls_and_enforces_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    slept: list[float] = []

    def fake_sleep(s: float) -> None:
        slept.append(s)
        clock[0] += s

    monkeypatch.setattr(harness.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(harness.time, "sleep", fake_sleep)
    llm = harness.CountingLLM(_Inner(), cap=3, min_interval_s=8.0)
    for _ in range(3):  # the first call never sleeps (no previous call), the next two wait 8 s
        llm.generate("x")
    assert slept == [pytest.approx(8.0), pytest.approx(8.0)] and llm.n_calls == 3
    with pytest.raises(RuntimeError, match="cap 3"):
        llm.generate("x")


def test_counting_llm_default_has_no_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness.time, "sleep", lambda s: pytest.fail("must not sleep"))
    llm = harness.CountingLLM(_Inner(), cap=5)
    llm.generate("x")
    llm.generate("x")
    assert llm.n_calls == 2
