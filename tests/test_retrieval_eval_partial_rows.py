"""variants_partial.json is merged, never replaced by a run that stops before its first row."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.retrieval import run
from evals.retrieval.groq_judge import BudgetExhaustedError


def test_a_stopped_run_does_not_wipe_stored_partial_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the first day-2 launch stopped on the spent day budget before any row and
    overwrote variants_partial.json (A and B rows) with empty arms."""
    stored = {"A": [{"query_id": "q01", "rep": 1}], "A2": [], "B": [{"query_id": "q01"}], "C": []}
    (tmp_path / "variants_partial.json").write_text(json.dumps(stored))

    def stop(*a: object, **k: object) -> None:
        raise BudgetExhaustedError("day budget spent")

    monkeypatch.setattr(run, "_run_arm", stop)
    rc = run.main(
        ["--stage", "variants", "--arms", "A2,C", "--mock", "--limit", "3", "--out", str(tmp_path)]
    )
    assert rc == 3  # STOPPED
    after = json.loads((tmp_path / "variants_partial.json").read_text())
    assert after["A"] == stored["A"] and after["B"] == stored["B"]


def test_write_partial_replaces_only_arms_run_now_with_rows(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    run.write_partial(p, {"A2": [{"query_id": "q1"}]}, ["A2"])
    run.write_partial(p, {"A2": [], "C": [{"query_id": "q1"}]}, ["A2", "C"])
    got = json.loads(p.read_text())
    assert got["A2"] == [{"query_id": "q1"}] and got["C"] == [{"query_id": "q1"}]
