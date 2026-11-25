"""Tests for the judge prompt builder, parser, calibration set and Claude packets (mock only)."""

from __future__ import annotations

import json
from pathlib import Path

from evals.retrieval import judge as jd
from evals.retrieval import packets as pk
from evals.retrieval.mock import mock_judge_text

CASE = {
    "id": "q99",
    "text": "men's black kurta set under ₹3000",
    "constraints": {"gender": "men", "colour": "black", "category": "kurta set", "max_price": 3000},
}


def _cands(n: int = 20) -> list[dict]:
    return [
        {
            "article_id": f"a{i}",
            "display_name": f"Men's Black Kurta Set {i}",
            "store": "SECRETSTORE",
            "price_inr": 1000.0 + i,
            "product_type": "Kurta set",
            "colour": "Black",
            "detail_desc": "Festive wear. " * 20,
            "rank": i + 1,
            "rrf_score": 0.9,
        }
        for i in range(n)
    ]


def test_prompt_is_blind_and_complete() -> None:
    p = jd.build_judge_prompt(CASE, _cands(), 42)
    assert (
        "SECRETSTORE" not in p.user and "rrf" not in p.user.lower() and "rank" not in p.user.lower()
    )
    assert len(p.order) == 20 and sorted(p.order) == sorted(f"a{i}" for i in range(20))
    assert "20. " in p.user and "Label all 20" in p.user
    assert all(len(line) < 400 for line in p.user.splitlines())  # description snippet is capped


def test_shuffle_deterministic_per_query_and_seed() -> None:
    a = jd.build_judge_prompt(CASE, _cands(), 42)
    b = jd.build_judge_prompt(CASE, _cands(), 42)
    c = jd.build_judge_prompt(CASE, _cands(), 43)
    d = jd.build_judge_prompt({**CASE, "id": "q98"}, _cands(), 42)
    assert a.order == b.order and a.prompt_hash == b.prompt_hash
    assert a.order != c.order and a.order != d.order
    assert a.order != [f"a{i}" for i in range(20)]  # actually shuffled


def test_parse_labels_valid_and_invalid() -> None:
    ok = json.dumps({"labels": {str(i): i % 3 for i in range(1, 21)}, "reason": "x"})
    assert jd.parse_labels("prefix " + ok + " suffix", 20) is not None
    assert jd.parse_labels(ok[:-30], 20) is None  # truncated JSON
    missing = json.dumps({"labels": {str(i): 1 for i in range(1, 20)}})
    assert jd.parse_labels(missing, 20) is None
    bad = json.dumps({"labels": {str(i): 3 for i in range(1, 21)}})
    assert jd.parse_labels(bad, 20) is None
    assert jd.parse_labels("not json", 20) is None


def test_labels_map_back_through_permutation() -> None:
    p = jd.build_judge_prompt(CASE, _cands(), 42)
    labels = {i: 2 if p.order[i - 1] == "a3" else 0 for i in range(1, 21)}
    mapped = jd.labels_to_ids(labels, p.order)
    assert mapped["a3"] == 2 and sum(mapped.values()) == 2


def test_calibration_set_shape_and_gold_distribution() -> None:
    cals = jd.calibration_cases()
    gold = {k: v for c in cals for k, v in c["gold"].items()}
    assert len(gold) == 20 and len(cals) == 4
    assert set(gold.values()) == {0, 1, 2}
    # by-construction checks from the brief
    assert gold["c1b"] == 0  # women's saree for a men's wedding kurta query
    assert gold["c1a"] == 2  # exact category, gender, colour, in budget


def test_score_calibration_pass_fail_and_fail_closed() -> None:
    gold = {k: v for c in jd.calibration_cases() for k, v in c["gold"].items()}
    assert jd.score_calibration(gold, gold)["passed"]
    two_off = dict(gold)
    two_off["c1a"], two_off["c1b"] = 0, 2  # two off-by-two errors
    assert not jd.score_calibration(two_off, gold)["passed"]
    one_wrong = dict(gold)
    one_wrong["c1c"] = 2  # off by one, 19/20 exact -> still passes
    assert jd.score_calibration(one_wrong, gold)["passed"]
    three_wrong = dict(gold)
    for k in ("c1c", "c2b", "c2e"):
        three_wrong[k] = 2
    assert not jd.score_calibration(three_wrong, gold)["passed"]  # 17/20
    assert not jd.score_calibration({}, gold)["passed"]  # missing predictions fail


def test_mock_judge_passes_calibration() -> None:
    gold = {k: v for c in jd.calibration_cases() for k, v in c["gold"].items()}
    pred: dict[str, int] = {}
    for cal in jd.calibration_cases():
        p = jd.build_judge_prompt(cal["case"], cal["candidates"], 42)
        parsed = jd.parse_labels(mock_judge_text(p.user), 5)
        assert parsed is not None
        pred.update(jd.labels_to_ids(parsed, p.order))
    assert jd.score_calibration(pred, gold)["passed"]


def test_packets_export_import_roundtrip(tmp_path: Path) -> None:
    prompts = [
        jd.build_judge_prompt({**CASE, "id": f"q{i:02d}"}, _cands(), 42) for i in range(1, 14)
    ]
    paths = pk.export_packets(prompts, tmp_path, "main")
    assert len(paths) == 3  # 6 + 6 + 1 queries
    assert all(p.read_text("utf-8").count("=== QUERY") <= 6 for p in paths)
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    # write fake Claude results: label = 2 for article a0 else 0 (labels given in shown numbering)
    for pid, meta in manifest.items():
        res = {
            q: {"labels": {str(i + 1): (2 if a == "a0" else 0) for i, a in enumerate(order)}}
            for q, order in meta["orders"].items()
        }
        (tmp_path / f"{pid}.result.json").write_text(json.dumps(res), encoding="utf-8")
    labels, problems = pk.import_results(tmp_path)
    assert problems == []
    assert len(labels["claude"]) == 13 * 20
    assert labels["claude"]["q05:a0"] == 2 and labels["claude"]["q05:a1"] == 0


def test_packet_import_reports_missing_and_invalid(tmp_path: Path) -> None:
    prompts = [jd.build_judge_prompt({**CASE, "id": f"q{i:02d}"}, _cands(), 42) for i in (1, 2)]
    pk.export_packets(prompts, tmp_path, "main")
    labels, problems = pk.import_results(tmp_path)
    assert labels["claude"] == {} and "result file missing" in problems[0]
    (tmp_path / "main_01.result.json").write_text(json.dumps({"q01": {"labels": {"1": 2}}}))
    _, problems = pk.import_results(tmp_path)
    assert any("q01" in p for p in problems) and any("q02" in p for p in problems)
