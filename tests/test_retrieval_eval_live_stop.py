"""Day-budget / 429 stops must abort the variants stage, not become fake 'timeout' rows; --arms."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.retrieval import run
from evals.retrieval.common import CANDIDATE_DIR, load_candidates, load_cases
from evals.retrieval.groq_judge import (
    BudgetExhaustedError,
    GroqRunner,
    Ledger,
    ResultCache,
)
from evals.retrieval.mock import mock_transport


def _llm(tmp_path: Path, budget: int) -> run.GroqLLM:
    runner = GroqRunner(
        mock_transport(),
        Ledger(tmp_path / "ledger.jsonl"),
        ResultCache(tmp_path / "cache.jsonl"),
        min_interval_s=0.0,
        sleep=lambda s: None,
    )
    return run.GroqLLM(runner, run.reranker_spec(budget), "A")


def test_budget_stop_aborts_instead_of_recording_a_timeout_row(tmp_path: Path) -> None:
    case = next(c for c in load_cases() if c["id"] == "q02")
    pool = load_candidates("q02", CANDIDATE_DIR)["items"]
    llm = _llm(tmp_path, budget=10)  # far below one call's worst case
    with pytest.raises(BudgetExhaustedError):
        run._run_arm("A", case, pool, llm, 1)
    assert llm.stop_exc is None  # consumed by the raise: the next call starts clean


def test_arms_subset_writes_partial_file_only(tmp_path: Path) -> None:
    base = ["--mock", "--limit", "3", "--out", str(tmp_path), "--reps", "1"]
    assert run.main(["--stage", "variants", "--arms", "A", *base]) == 0
    assert not (tmp_path / "variants.json").exists()
    rows = json.loads((tmp_path / "variants_partial.json").read_text())
    assert rows["A"] and not rows["A2"] and not rows["B"] and not rows["C"]
    assert all(r["rep"] == 1 for r in rows["A"])


def test_bad_arms_value_is_rejected(tmp_path: Path) -> None:
    rc = run.main(["--stage", "variants", "--arms", "A,Z", "--mock", "--out", str(tmp_path)])
    assert rc == 3  # STOPPED: ValueError reported as a clean refusal
