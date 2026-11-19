"""Consensus relabelling: selection, blindness, majority, agreement stats, gate, proposal (mock)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest
import yaml

from evals.retrieval import relabel as rl
from evals.retrieval import relabel_p5 as p5

QUERIES, OLDER, BATCH = rl.load_gold()


def _article(aid: str, query: str) -> dict:
    """Synthetic catalogue row: half the items echo the query (relevant for the mock rule)."""
    echo = int(hashlib.md5(aid.encode()).hexdigest(), 16) % 2 == 0
    title = f"{query} piece {aid}" if echo else f"Unrelated sandals {aid} VASTRAMAY"
    return {
        "article_id": aid,
        "display_name": title,
        "colour_group_name": "Red",
        "price_inr": 900.0,
        "product_type_name": "kurta",
        "detail_desc": f"Soft fabric. Sold by Libas. {aid}",
    }


def _items(n_hygiene: int = 0) -> list[rl.Item]:
    drift = rl.sample_drift(OLDER, QUERIES)
    first_q = sorted(QUERIES)[0]
    hyg = [(first_q, f"hyg{i}") for i in range(n_hygiene)]
    keys = [*BATCH, *drift, *hyg]
    articles = {k[1]: _article(k[1], QUERIES[k[0]]["query"]) for k in keys}
    return rl.build_items(QUERIES, OLDER, BATCH, drift, hyg, articles)


# ---------------------------------------------------------------- selection


def test_label_file_splits_into_older_labels_and_the_127_item_batch() -> None:
    assert len(BATCH) == 127  # the ddb4dfd batch (commit message and file header agree)
    assert not set(BATCH) & set(OLDER)
    assert len(OLDER) > 300
    with pytest.raises(ValueError, match="batch marker"):
        rl.split_label_file("labels: []\n")


def test_drift_sample_is_60_seeded_stratified_and_from_older_labels() -> None:
    a = rl.sample_drift(OLDER, QUERIES)
    b = rl.sample_drift(OLDER, QUERIES)
    assert a == b and len(a) == 60 == len(set(a)) and set(a) <= set(OLDER)
    assert a != rl.sample_drift(OLDER, QUERIES, seed=43)
    sampled = {c: 0 for c in {q["category"] for q in QUERIES.values()}}
    pool = dict(sampled)
    for k in a:
        sampled[QUERIES[k[0]]["category"]] += 1
    for k in OLDER:
        pool[QUERIES[k[0]]["category"]] += 1
    for c, n in sampled.items():  # proportional allocation: within one item of 60 * share
        assert abs(n - 60 * pool[c] / len(OLDER)) <= 1.0, c
    with pytest.raises(ValueError):
        rl.sample_drift(OLDER, QUERIES, n=len(OLDER) + 1)


def test_build_items_reports_missing_catalogue_rows_instead_of_skipping() -> None:
    drift = rl.sample_drift(OLDER, QUERIES)
    with pytest.raises(KeyError, match="not in the catalogue"):
        rl.build_items(QUERIES, OLDER, BATCH, drift, [], {})


# ---------------------------------------------------------------- prompts and blindness


def test_every_item_is_in_exactly_one_chunk_and_queries_are_not_split_within_a_chunk() -> None:
    items = _items(3)
    chunks = rl.chunk_items(items)
    keys = [i.key for c in chunks for i in c]
    assert sorted(keys) == sorted(i.key for i in items) and len(keys) == len(set(keys))
    assert sum(1 for c in chunks for _ in c) == len(items)


def test_calibration_chunk_comes_first_and_has_both_classes() -> None:
    items = _items()
    chunks = rl.chunk_items(items)
    cal = chunks[0]
    assert {i.group for i in cal} == {"drift"} and len(cal) >= rl.CHUNK_ITEMS
    assert sum(1 for i in cal if i.stored is False) >= 1 and any(i.stored for i in cal)
    prompts = rl.build_prompts(items)
    assert prompts[0].calibration and not any(p.calibration for p in prompts[1:])


def test_prompt_numbering_matches_order_and_is_blind() -> None:
    items = _items()
    by_key = {i.key: i for i in items}
    for p in rl.build_prompts(items):
        printed = {int(m.group(1)): m.group(0) for m in re.finditer(r"^(\d+)\. .*$", p.user, re.M)}
        assert len(printed) == len(p.order)
        for n, key in enumerate(p.order, 1):
            assert str(by_key[key].article_id) in printed[n]  # line n IS order[n-1]
        low = p.user.lower()
        assert "vastramay" not in low and "libas" not in low  # store names scrubbed
        assert "stored" not in low and "ddb4dfd" not in low and "drift" not in low
        assert not re.search(r"gold_\d+|occ_\w+_\d+", p.user)  # no query ids
        assert "relevant" not in p.user  # no hint of a stored label


def test_prompts_are_deterministic_and_seed_dependent() -> None:
    items = _items()
    a, b = rl.build_prompts(items), rl.build_prompts(items)
    assert [p.prompt_hash for p in a] == [p.prompt_hash for p in b]
    assert [p.order for p in a] != [p.order for p in rl.build_prompts(items, seed=7)]


def test_parse_relabel_valid_invalid_and_reason_filtering() -> None:
    ok = json.dumps(
        {"labels": {"1": 1, "2": 0, "3": 0}, "reasons": {"2": "budget", "3": "nope", "1": "budget"}}
    )
    assert rl.parse_relabel("junk " + ok, 3) == ({1: 1, 2: 0, 3: 0}, {2: "budget"})
    assert rl.parse_relabel(json.dumps({"labels": {"1": 1, "2": 2}}), 2) is None  # not binary
    assert rl.parse_relabel(json.dumps({"labels": {"1": 1}}), 2) is None  # missing item
    assert rl.parse_relabel("not json", 1) is None
    assert set(rl.MISS_REASONS) >= {"set-not-single", "budget", "type-confusion"}


def test_packets_round_trip_to_the_same_keys(tmp_path: Path) -> None:
    prompts = rl.build_prompts(_items())
    paths = rl.export_packets(prompts, tmp_path)
    assert len(paths) == len(prompts)
    for p in prompts:
        res = {
            p.chunk_id: {
                "labels": {str(i): i % 2 for i in range(1, len(p.order) + 1)},
                "reasons": {},
            }
        }
        for i in range(1, len(p.order) + 1):
            if i % 2 == 0:
                res[p.chunk_id]["reasons"][str(i)] = "type-confusion"
        (tmp_path / f"{p.chunk_id}.result.json").write_text(json.dumps(res))
    labels, reasons, problems = rl.import_packets(tmp_path)
    assert problems == [] and len(labels) == sum(len(p.order) for p in prompts)
    first = prompts[0]
    assert labels[first.order[0]] == 1 and labels[first.order[1]] == 0
    assert reasons[first.order[1]] == "type-confusion"
    (tmp_path / f"{prompts[1].chunk_id}.result.json").unlink()
    assert any("missing" in p for p in rl.import_packets(tmp_path)[2])


# ---------------------------------------------------------------- consensus and statistics


def test_majority_three_judges_and_two_judge_tie_fails_closed() -> None:
    assert rl.majority({"a": 1, "b": 1, "c": 0})["label"] == 1
    assert rl.majority({"a": 0, "b": 1, "c": 0})["label"] == 0
    tie = rl.majority({"a": 1, "b": 0})
    assert tie["label"] is None and tie["tie"] is True


def test_binary_agreement_hand_values() -> None:
    r = rl.binary_agreement([1, 1, 0, 0], [1, 0, 0, 0])
    assert r["percent"] == 75.0 and r["confusion"] == [[2, 0], [1, 1]]
    assert r["kappa"] == pytest.approx(0.5)  # po .75, pe .5
    assert math.isnan(rl.binary_agreement([1, 1], [1, 1])["kappa"])  # single label: undefined
    assert rl.binary_agreement([0, 1, 0, 1], [0, 1, 0, 1])["kappa"] == pytest.approx(1.0)


def test_agreement_vs_stored_groups_and_tie_exclusion() -> None:
    items = _items()
    ddb = [i for i in items if i.group == "ddb4dfd"][:4]
    cons = {
        ddb[0].key: rl.majority(
            {"a": int(ddb[0].stored or 0), "b": int(ddb[0].stored or 0), "c": 0}
        ),
        ddb[1].key: rl.majority({"a": 1, "b": 0}),  # tie
    }
    out = rl.agreement_vs_stored(cons, items)
    assert out["ddb4dfd"]["n_tied_excluded"] == 1 and out["ddb4dfd"]["n"] == 1
    assert out["all"]["n"] == 1 and out["drift"]["n"] == 0


def _labels(flip: bool = False, const: bool = False) -> dict[str, int]:
    base = {f"k{i}": i % 2 for i in range(30)}
    if const:
        return {k: 1 for k in base}
    return {k: 1 - v if flip else v for k, v in base.items()}


def test_gate_passes_for_agreeing_judges() -> None:
    ok, msg = rl.kappa_gate({"a": _labels(), "b": _labels(), "c": _labels()})
    assert ok and "passed" in msg


def test_gate_fails_below_0_4_and_prints_the_confusion_table() -> None:
    ok, msg = rl.kappa_gate({"a": _labels(), "b": _labels(), "c": _labels(flip=True)})
    assert not ok and "FAILED" in msg and "confusion a|c" in msg and "[[0, 15], [15, 0]]" in msg


def test_gate_fails_closed_on_undefined_kappa_and_on_a_single_judge() -> None:
    assert not rl.kappa_gate({"a": _labels(), "b": _labels(const=True)})[0]
    assert not rl.kappa_gate({"a": _labels()})[0]


def test_pair_report_alpha_is_one_for_identical_judges() -> None:
    rep = rl.pair_report({"a": _labels(), "b": _labels(), "c": _labels()})
    assert rep["alpha"] == pytest.approx(1.0) and set(rep["pairs"]) == {"a|b", "a|c", "b|c"}


def test_proposal_lists_replacements_new_labels_and_unresolved_without_editing_anything() -> None:
    items = _items(2)
    stored = next(i for i in items if i.group == "ddb4dfd" and i.stored is True)
    stored_no = next(i for i in items if i.group == "ddb4dfd" and i.stored is False)
    hyg = [i for i in items if i.group == "hygiene"]
    cons = {
        stored.key: rl.majority({"a": 0, "b": 0, "c": 1}),  # flips True -> False
        stored_no.key: rl.majority({"a": 0, "b": 0, "c": 0}),  # agrees: not listed
        hyg[0].key: rl.majority({"a": 1, "b": 1, "c": 0}),  # new label
        hyg[1].key: rl.majority({"a": 1, "b": 0}),  # unresolved tie
    }
    reasons = {"a": {stored.key: "budget"}, "b": {stored.key: "budget"}, "c": {}}
    prop = rl.build_proposal(cons, items, reasons)
    assert (prop["n_replacements"], prop["n_new_labels"], prop["n_unresolved_ties"]) == (1, 1, 1)
    rep = prop["replacements"][0]
    assert rep["stored"] is True and rep["proposed"] is False and rep["reason"] == "budget"
    assert prop["new_labels"][0]["proposed"] is True and "PROPOSAL ONLY" in prop["note"]


def test_judge_count_warning() -> None:
    assert rl.judge_count_warning({"a": {"k": 1}, "b": {"k": 1}, "c": {"k": 1}}) == []
    (w,) = rl.judge_count_warning({"a": {"k": 1}, "b": {"k": 1}, "c": {}})
    assert "ONLY 2 of 3" in w


def test_plan_counts_calls_items_and_days() -> None:
    p = rl.plan(rl.build_prompts(_items(7)))
    assert p["items"] == 127 + 60 + 7 and p["n_calls_per_judge"] == len(p["prompt_tokens_per_call"])
    assert p["expected_tokens_per_judge"] < p["worst_case_tokens_per_judge"]
    assert p["days_expected"] >= 1 and p["daily_budget"] == 150_000


# ---------------------------------------------------------------- strict P@5 before / after


def test_strict_p5_hand_values_and_unlabelled_pairs() -> None:
    top5 = {"queries": {"g1": ["a", "b", "c"], "g2": ["d", "e"]}}
    labels = {("g1", "a"): True, ("g1", "b"): False, ("g2", "d"): True, ("g2", "e"): True}
    cats = {"g1": "style", "g2": "literal"}
    res = p5.strict_p5(top5, labels, cats)
    assert (res["n_relevant"], res["n_scored"], res["n_unlabelled"]) == (3, 4, 1)
    assert res["p_at_5"] == pytest.approx(0.75) and res["by_category"]["style"]["p_at_5"] == 0.5
    assert p5.unlabelled_pairs(top5, labels) == [("g1", "c")]


def test_before_after_applies_replacements_and_new_labels_only() -> None:
    top5 = {"queries": {"g1": ["a", "b", "c"]}}
    labels = {("g1", "a"): True, ("g1", "b"): False}
    proposal = {
        "replacements": [{"query_id": "g1", "article_id": "b", "proposed": True}],
        "new_labels": [{"query_id": "g1", "article_id": "c", "proposed": False}],
        "unresolved": [{"query_id": "g1", "article_id": "zzz"}],
    }
    res = p5.before_after(top5, labels, proposal, {"g1": "style"})
    assert res["before"]["p_at_5"] == 0.5 and res["after"]["p_at_5"] == pytest.approx(2 / 3)
    assert res["after"]["n_unlabelled"] == 0 and res["before"]["n_unlabelled"] == 1
    assert labels == {("g1", "a"): True, ("g1", "b"): False}  # input untouched


# ---------------------------------------------------------------- mock end to end


def _prepared(tmp_path: Path) -> Path:
    out = tmp_path / "relabel"
    rl.save_items(_items(2), out)
    return out


def test_mock_pipeline_writes_labels_consensus_and_proposal_without_touching_the_label_file(
    tmp_path: Path,
) -> None:
    labels_before = rl.LABELS_PATH.read_bytes()
    out = _prepared(tmp_path)
    assert rl.main(["--stage", "judge", "--mock", "--out", str(out)]) == 0
    for j in rl.JUDGES:
        assert len(json.loads((out / f"labels_{j}.json").read_text())["labels"]) == 127 + 60 + 2
    assert rl.main(["--stage", "consensus", "--out", str(out)]) == 0
    cons = json.loads((out / "consensus.json").read_text())
    assert cons["report"]["warnings"] == [] and cons["report"]["n_ties"] == 0
    assert cons["report"]["vs_stored"]["all"]["n"] == 127 + 60
    prop = yaml.safe_load((out / "proposal.yaml").read_text())
    assert prop["n_new_labels"] == 2 and "PROPOSAL ONLY" in prop["note"]
    assert rl.LABELS_PATH.read_bytes() == labels_before  # strict_gold_labels.yaml untouched
    n = len((out / "ledger.jsonl").read_text().splitlines())
    assert rl.main(["--stage", "judge", "--mock", "--out", str(out)]) == 0
    assert len((out / "ledger.jsonl").read_text().splitlines()) == n  # fully cached rerun


def test_mock_gate_stops_the_run_and_prints_the_confusion_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _prepared(tmp_path)
    monkeypatch.setitem(rl.MOCK_NOISE, rl.CLAUDE, 1.0)  # a Claude that disagrees with everyone
    assert rl.main(["--stage", "judge", "--mock", "--out", str(out)]) == 2
    text = capsys.readouterr().out
    assert "RELABEL CALIBRATION GATE FAILED" in text and "confusion" in text
    assert not (out / "labels_gpt-oss-120b.json").exists()  # stopped BEFORE the main run


def test_live_is_refused_without_the_flag_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _prepared(tmp_path)
    monkeypatch.delenv("STAGE2_LIVE_OK", raising=False)
    assert rl.main(["--stage", "judge", "--live", "--out", str(out)]) == 3  # no transport built
    assert rl.main(["--stage", "judge", "--out", str(out)]) == 3  # neither --mock nor --live


def test_load_articles_reads_only_requested_ids(tmp_path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "article_id": ["1", "2", "3"],
            "prod_name": ["a", "b", "c"],
            "display_name": ["A", "B", "C"],
            "product_type_name": ["kurta"] * 3,
            "colour_group_name": ["Red"] * 3,
            "price_inr": [1.0, 2.0, 3.0],
            "detail_desc": ["x"] * 3,
            "extra": [0, 0, 0],
        }
    )
    path = tmp_path / "catalogue.parquet"
    df.to_parquet(path)
    got = rl.load_articles(path, ["2", "3", "99"])
    assert set(got) == {"2", "3"} and got["2"]["display_name"] == "B"
