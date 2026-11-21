"""Offline results (no LLM): B, F1/F2 for B, re-scored A rep 1. Mock judges/variants, tmp dirs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.retrieval import offline, run
from evals.retrieval.common import CANDIDATE_DIR, MASTER_CANDIDATE_DIR


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("offline")
    base = ["--mock", "--limit", "30", "--out", str(out)]
    assert run.main(["--stage", "judge", *base]) == 0
    assert run.main(["--stage", "variants", "--arms", "A", "--reps", "1", *base]) == 0
    return out


def test_reclassify_turns_empty_selection_into_no_exact_match() -> None:
    stop_empty = {
        "outcome": "parse_error",
        "raw_output": '{"selected": [], "reasoning": "none"}',
        "finish_reason": "stop",
        "parsed": None,
    }
    r = offline.reclassify(stop_empty)
    assert r["outcome"] == "empty_selection" and r["no_exact_match"] is True
    cut = {**stop_empty, "raw_output": '{"selected": [', "finish_reason": "length"}
    assert offline.reclassify(cut)["outcome"] == "length"
    boom = {"outcome": "timeout", "raw_output": None, "finish_reason": None}
    assert offline.reclassify(boom)["outcome"] == "http_error"
    skipped = {"outcome": "no_llm_call"}
    assert offline.reclassify(skipped)["outcome"] == "no_llm_call"


def test_compute_populations_exclusions_and_factors(out_dir: Path) -> None:
    res = offline.compute(out_dir, CANDIDATE_DIR, MASTER_CANDIDATE_DIR)
    prim = res["populations"]["primary_pool_gt_5"]
    sens = res["populations"]["sensitivity_all_with_candidates"]
    assert 0 < prim["n_queries"] <= sens["n_queries"] <= 30
    # the pool <= 5 queries are exactly the ones the primary population drops
    assert sens["n_queries"] - prim["n_queries"] == len(res["excluded_pool_le_5"])
    assert res["labels"].startswith("LLM-consensus")
    for pop in (prim, sens):
        assert {"A", "B"} == set(pop["metrics"]) and "A-B" in pop["paired"]
        assert pop["paired"]["A-B"]["ndcg5"]["n"] == pop["n_queries"]
    f = res["factors_B_only"]
    assert f["tables"]["F1"]["n_queries"] == len(f["f1_queries"])  # flagged ethnic-wear queries
    assert f["tables"]["F2"]["n_queries"] == len(f["f2_queries"])
    assert f["tables"]["F1"]["pool_loss"]["n_removed"] >= 0


def test_main_writes_results_and_keeps_stored_a_rows(out_dir: Path, tmp_path: Path) -> None:
    before = json.loads((out_dir / "variants_partial.json").read_text())["A"]
    rc = offline.main(["--out", str(out_dir), "--results-dir", str(tmp_path)])
    assert rc == 0
    after = json.loads((out_dir / "variants_partial.json").read_text())
    assert after["A"] == before and after["B"]  # A untouched, B added
    assert not (out_dir / "variants.json").exists()
    md = (tmp_path / "offline_results.md").read_text()
    assert "primary_pool_gt_5" in md and "F1 treated" in md and "not human ground truth" in md
