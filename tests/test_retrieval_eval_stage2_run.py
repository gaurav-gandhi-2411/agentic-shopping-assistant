"""Mock end-to-end: A/A2/B/C with repeats, stability, factors F1/F2 and the report (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.retrieval import run
from evals.retrieval.common import load_cases
from evals.retrieval.factors import is_gender_unspecified, is_set_demotion_query

LIMIT = "30"  # q01..q30 on the real frozen candidates: 7 F1 queries, 6 F2 queries


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("stage2")
    base = ["--mock", "--limit", LIMIT, "--out", str(out)]
    assert run.main(["--stage", "judge", *base]) == 0
    assert run.main(["--stage", "variants", *base]) == 0
    assert run.main(["--stage", "factors", *base]) == 0
    assert run.main(["--stage", "report", *base]) == 0
    return out


def test_variants_rows_and_reps(out_dir: Path) -> None:
    v = json.loads((out_dir / "variants.json").read_text())
    assert set(v) == {"A", "A2", "B", "C"}
    n = len({r["query_id"] for r in v["B"]})
    assert len(v["B"]) == n and len(v["C"]) == n  # B and C: one draw per query
    for arm in ("A", "A2"):
        assert len(v[arm]) == 3 * n and {r["rep"] for r in v[arm]} == {1, 2, 3}
    assert all("cards" in r for rows in v.values() for r in rows)
    assert all(len(r["cards"]) <= 5 for rows in v.values() for r in rows)


def test_a2_never_hits_length_while_a_does_and_both_are_reported(out_dir: Path) -> None:
    rep = json.loads((out_dir / "report.json").read_text())
    assert rep["rates"]["A"]["length_hits"] > 0 and rep["rates"]["A"]["length_rate"] > 0
    assert rep["rates"]["A2"]["length_hits"] == 0
    assert rep["a2_length_check"].startswith("A2 length check passed")
    assert rep["rates"]["A"]["fallback_rate"] > 0  # truncation -> parse_error -> RRF fallback


def test_report_has_all_arms_pairs_stability_and_top2(out_dir: Path) -> None:
    rep = json.loads((out_dir / "report.json").read_text())
    assert set(rep["metrics"]) == {"A", "A2", "B", "C"}
    assert {"A2-A", "A2-B", "A-B", "C-B", "C-A", "C-A2"} <= set(rep["paired"])
    assert len(rep["top2_by_ndcg5_point_estimate"]) == 2
    st = rep["stability"]["A"]
    assert st["n_queries_repeated"] > 0 and "card_count_differs_rate" in st
    assert set(rep["rep_spread"]["A2"]) == {"rep1", "rep2", "rep3"}
    assert rep["usage"]["B"]["n"] == 0 and rep["usage"]["B"]["n_no_llm_call"] > 0
    assert "metrics_post_gates" in rep and "A" in rep["metrics_post_gates"]


def test_factors_cover_exactly_the_flagged_queries(out_dir: Path) -> None:
    f = json.loads((out_dir / "factors.json").read_text())
    cases = load_cases(int(LIMIT))
    n1 = sum(1 for c in cases if is_set_demotion_query(c))
    n2 = sum(1 for c in cases if is_gender_unspecified(c))
    assert (n1, n2) == (7, 6)
    assert len(f["top2"]) == 2
    for v in f["top2"]:
        assert len(f["F1"][v]["base"]) == len(f["F1"][v]["treated"]) <= n1
        assert len(f["F2"][v]["base"]) == len(f["F2"][v]["treated"]) <= n2
        loss = f["F1"][v]["pool_loss"]
        assert loss["n_removed"] == loss["n_before"] - loss["n_after"] >= 0
    rep = json.loads((out_dir / "report.json").read_text())
    for fac in ("F1", "F2"):
        for v in f["top2"]:
            e = rep["factors"][fac][v]
            assert "paired_treated_minus_base" in e and e["n_queries"] > 0
    assert "sets_in_shown_top5" in rep["factors"]["F1"][f["top2"][0]]


def test_a_second_run_makes_no_new_calls(out_dir: Path) -> None:
    n = len((out_dir / "ledger.jsonl").read_text().splitlines())
    base = ["--mock", "--limit", LIMIT, "--out", str(out_dir)]
    assert run.main(["--stage", "judge", *base]) == 0
    assert len((out_dir / "ledger.jsonl").read_text().splitlines()) == n
