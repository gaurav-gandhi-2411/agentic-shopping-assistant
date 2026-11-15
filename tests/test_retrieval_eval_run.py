"""End-to-end mock tests of the stage driver (judge gate, variants, report) and the fixtures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from evals.retrieval import candidates as cd
from evals.retrieval import make_queries as mq
from evals.retrieval import run
from evals.retrieval.common import load_cases
from evals.retrieval.mock import _default_responder, mock_transport


def _write_candidates(cand_dir: Path, n_cases: int = 3) -> None:
    cand_dir.mkdir(parents=True, exist_ok=True)
    for case in load_cases(n_cases):
        c = case["constraints"]
        items = []
        for i in range(1, 21):
            wrong = i % 3 == 0
            gender = (
                "Women's"
                if wrong and c["gender"] == "men"
                else ("Men's" if c["gender"] == "men" else "Women's")
            )
            items.append(
                {
                    "rank": i,
                    "article_id": f"{case['id']}x{i}",
                    "store": "kisah",
                    "display_name": f"{gender} {c['colour'] or 'Red'} {c['category']} {i}",
                    "price_inr": 800.0 + 150 * i,
                    "product_type": c["category"],
                    "colour": c["colour"] or "Red",
                    "department": "N/A",
                    "detail_desc": "Nice wear",
                    "gender": "women",
                    "rrf_score": 0.01,
                }
            )
        (cand_dir / f"{case['id']}.json").write_text(
            json.dumps({"query_id": case["id"], "text": case["text"], "items": items})
        )


def test_fixtures_are_the_frozen_generator_output() -> None:
    gen = {c["id"]: c for c in mq.build_cases()}
    disk = {c["id"]: c for c in load_cases()}
    assert gen == disk and len(disk) == 60


def test_stratification_properties() -> None:
    cases = load_cases()
    assert Counter(c["gender"] for c in cases) == {"women": 24, "men": 24, "either": 12}
    assert all(v == 12 for v in Counter(c["language_style"] for c in cases).values())
    assert all(v == 12 for v in Counter(str(c["budget_inr"]) for c in cases).values())
    assert all(v == 5 for v in Counter(c["occasion"] for c in cases).values())
    hard = [c for c in cases if c["hard"]]
    assert len(hard) == 12 and all(c["constraints"]["colour"] and c["budget_inr"] for c in hard)
    assert len({c["text"] for c in cases}) == 60  # no duplicate queries
    assert all(
        set(c)
        >= {
            "id",
            "text",
            "language_style",
            "occasion",
            "gender",
            "budget_inr",
            "constraints",
            "notes",
        }
        for c in cases
    )


def test_generator_is_deterministic() -> None:
    assert mq.build_cases() == mq.build_cases()


def test_public_store_list() -> None:
    from evals.retrieval.common import load_public_stores

    stores = load_public_stores()
    assert len(stores) == 38
    assert not {"myntra", "flipkart", "bhasinbrothers", "priyaasi"} & set(stores)


def test_judge_variants_report_mock_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cand, out = tmp_path / "cand", tmp_path / "out"
    _write_candidates(cand)
    base = ["--mock", "--limit", "3", "--cand", str(cand), "--out", str(out)]
    assert run.main(["--stage", "judge", *base]) == 0
    assert (out / "labels_gpt-oss-120b.json").exists() and (
        out / "labels_qwen3.8-27b.json"
    ).exists()
    assert len(json.loads((out / "labels_gpt-oss-120b.json").read_text())) == 60
    assert run.main(["--stage", "variants", *base]) == 0
    assert run.main(["--stage", "report", *base]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert set(rep["metrics"]) == {"A", "B", "C"} and rep["n_queries"] == 3
    assert "not human ground truth" in capsys.readouterr().out
    # rerun is fully cached: no new ledger rows for judging
    n_rows = len((out / "ledger.jsonl").read_text().splitlines())
    assert run.main(["--stage", "judge", *base]) == 0
    assert len((out / "ledger.jsonl").read_text().splitlines()) == n_rows


def test_calibration_failure_refuses_main_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cand, out = tmp_path / "cand", tmp_path / "out"
    _write_candidates(cand)

    def bad_responder(system: str, user: str, model: str) -> str:
        txt = _default_responder(system, user, model)
        if "qwen" in model:  # this judge inverts every label
            d = json.loads(txt)
            d["labels"] = {k: 2 - v for k, v in d["labels"].items()}
            return json.dumps(d)
        return txt

    monkeypatch.setattr(run, "mock_transport", lambda *a, **k: mock_transport(bad_responder))
    rc = run.main(
        ["--stage", "judge", "--mock", "--limit", "3", "--cand", str(cand), "--out", str(out)]
    )
    assert rc == 2 and "REFUSING" in capsys.readouterr().out
    assert not (out / "labels_gpt-oss-120b.json").exists()  # no main run happened


def _live_path_with_mock_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the non-mock (live) code path, but with a fake transport and no real sleeping."""
    import functools

    monkeypatch.setenv("STAGE2_LIVE_OK", "1")
    monkeypatch.setattr(run, "httpx_transport", lambda *a, **k: mock_transport())
    monkeypatch.setattr(
        run,
        "GroqRunner",
        functools.partial(run.GroqRunner, min_interval_s=0.0, sleep=lambda s: None),
    )


def test_kappa_gate_stops_before_main_run_and_prints_confusion_and_rubric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cand, out = tmp_path / "cand", tmp_path / "out"
    _write_candidates(cand)
    (out / "claude_packets").mkdir(parents=True)
    inverted = {k: 2 - v for k, v in run._cal_gold().items()}  # a Claude disagreeing with everyone
    (out / "claude_packets" / "labels_cal_claude.json").write_text(json.dumps(inverted))
    _live_path_with_mock_transport(monkeypatch)
    rc = run.main(
        ["--stage", "judge", "--live", "--limit", "3", "--cand", str(cand), "--out", str(out)]
    )
    text = capsys.readouterr().out
    assert rc == 2 and "CALIBRATION GATE FAILED" in text and "confusion" in text
    assert "Label every number" in text  # the rubric is printed
    assert not (out / "labels_gpt-oss-120b.json").exists()  # stopped BEFORE the main run


def test_gate_passes_with_agreeing_claude_and_runs_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cand, out = tmp_path / "cand", tmp_path / "out"
    _write_candidates(cand)
    (out / "claude_packets").mkdir(parents=True)
    (out / "claude_packets" / "labels_cal_claude.json").write_text(json.dumps(run._cal_gold()))
    _live_path_with_mock_transport(monkeypatch)
    assert (
        run.main(
            ["--stage", "judge", "--live", "--limit", "3", "--cand", str(cand), "--out", str(out)]
        )
        == 0
    )
    assert (out / "labels_gpt-oss-120b.json").exists()


def test_partial_gate_requires_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without Claude's calibration labels the gate cannot run on 3 judges: fail closed."""
    cand, out = tmp_path / "cand", tmp_path / "out"
    _write_candidates(cand)
    _live_path_with_mock_transport(monkeypatch)
    args = ["--stage", "judge", "--live", "--limit", "3", "--cand", str(cand), "--out", str(out)]
    assert run.main(args) == 2
    assert not (out / "labels_gpt-oss-120b.json").exists()
    assert run.main([*args, "--allow-partial-gate"]) == 0


def test_candidate_record_helpers() -> None:
    it = {
        "article_id": 5,
        "store": "kisah",
        "display_name": "T",
        "price_inr": float("nan"),
        "product_type": "Kurta",
        "colour": "Red",
        "detail_desc": "x" * 500,
        "score": 0.03,
        "gender": "women",
    }
    rec = cd.item_record(1, it)
    assert rec["article_id"] == "5" and rec["price_inr"] is None and len(rec["detail_desc"]) == 300
    case = {"id": "q01", "text": "t"}
    out = cd.case_record(case, [dict(it, article_id=i) for i in range(25)], "graph_rerank", 0)
    assert out["n_candidates"] == 20 and out["pool_size_before_cap"] == 25
    summ = cd.summarize([out])
    assert summ["under_20"] == [] and summ["store_distribution"] == {"kisah": 20}


def test_restrict_to_stores_patches_and_restores() -> None:
    pytest.importorskip("faiss")
    import src.retrieval.hybrid_search as hs

    before = hs.get_inactive_stores
    with cd.restrict_to_stores(["kisah"], {"kisah", "myntra", "flipkart"}):
        assert {"myntra", "flipkart"} <= hs.get_inactive_stores()
        assert "kisah" not in hs.get_inactive_stores()
    assert hs.get_inactive_stores is before


def test_export_then_import_claude_via_cli(tmp_path: Path) -> None:
    cand, out = tmp_path / "cand", tmp_path / "out"
    _write_candidates(cand)
    common = ["--limit", "3", "--cand", str(cand), "--out", str(out)]
    assert run.main(["--stage", "judge", "--export-packets", *common]) == 0
    pdir = out / "claude_packets"
    manifest = json.loads((pdir / "manifest.json").read_text())
    assert any(k.startswith("cal_") for k in manifest) and any(
        k.startswith("main_") for k in manifest
    )
    for pid, meta in manifest.items():
        res = {
            q: {"labels": {str(i + 1): 1 for i in range(len(order))}}
            for q, order in meta["orders"].items()
        }
        (pdir / f"{pid}.result.json").write_text(json.dumps(res))
    assert run.main(["--stage", "judge", "--import-claude", *common]) == 0
    assert len(json.loads((out / "labels_claude.json").read_text())) == 60
    assert len(json.loads((pdir / "labels_cal_claude.json").read_text())) == 20
