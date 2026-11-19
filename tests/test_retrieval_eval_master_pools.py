"""Pools regenerated on master (main comparison) vs the frozen pools (F1 'sets not demoted' arm).

Mock only, every root is a tmp_path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from evals.retrieval import candidates as cd
from evals.retrieval import run
from evals.retrieval.common import (
    CANDIDATE_DIR,
    MASTER_CANDIDATE_DIR,
    load_candidates,
    resolve_pools_dir,
)
from evals.retrieval.factors import demote_sets


def _item(qid: str, i: int) -> dict:
    return {"article_id": f"{qid}x{i}", "rank": i}


def _write(d: Path, qid: str, ids: list[int]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{qid}.json").write_text(
        json.dumps({"query_id": qid, "items": [_item(qid, i) for i in ids]}), encoding="utf-8"
    )


def test_resolve_pools_dir_rules(tmp_path: Path) -> None:
    other = tmp_path / "scratch"
    assert resolve_pools_dir(other, None) == other  # any non-default --cand is its own pool
    assert resolve_pools_dir(CANDIDATE_DIR, "x/y") == Path("x/y")  # explicit wins
    expected = MASTER_CANDIDATE_DIR if MASTER_CANDIDATE_DIR.exists() else CANDIDATE_DIR
    assert resolve_pools_dir(CANDIDATE_DIR, None) == expected


def test_compare_pools_lists_differing_and_new_items(tmp_path: Path) -> None:
    frozen, new = tmp_path / "frozen", tmp_path / "new"
    _write(frozen, "q01", [1, 2, 3])
    _write(new, "q01", [1, 2, 3])  # identical
    _write(frozen, "q02", [1, 2, 3])
    _write(new, "q02", [2, 3])  # a set removed: subset, no new item
    _write(frozen, "q03", [1, 2])
    _write(new, "q03", [1, 9])  # a NEW item: must be reported
    _write(new, "q04", [1])  # no frozen counterpart: ignored
    out = cd.compare_pools(frozen, new)
    assert out["n_identical"] == 1
    assert out["differing"] == ["q02", "q03"]
    assert out["new_items_vs_frozen"] == {"q03": ["q03x9"]}


def test_summary_carries_provenance_and_comparison() -> None:
    rec = {"query_id": "q01", "n_candidates": 2, "route": "graph_rerank", "items": []}
    s = cd.summary_with_provenance([rec], {"code_commit": "abc"}, {"differing": ["q01"]})
    assert s["provenance"]["code_commit"] == "abc"
    assert s["compared_to_frozen"]["differing"] == ["q01"]
    assert s["n_cases"] == 1


def _setup(tmp_path: Path, extra_item: bool = False) -> tuple[Path, Path, list[str]]:
    """frozen = real q01/q02 pools; master = same but q02's single-garment sets demoted."""
    frozen, master = tmp_path / "frozen", tmp_path / "master"
    frozen.mkdir()
    master.mkdir()
    for q in ("q01", "q02"):
        shutil.copy(CANDIDATE_DIR / f"{q}.json", frozen / f"{q}.json")
        shutil.copy(CANDIDATE_DIR / f"{q}.json", master / f"{q}.json")
    rec = load_candidates("q02", frozen)
    reordered = demote_sets(rec["items"])
    assert [i["article_id"] for i in reordered] != [i["article_id"] for i in rec["items"]]
    rec["items"] = reordered
    if extra_item:
        rec["items"] = [{**rec["items"][0], "article_id": "brand-new"}, *rec["items"][1:]]
    (master / "q02.json").write_text(json.dumps(rec), encoding="utf-8")
    return frozen, master, ["q01", "q02"]


def _run_all(tmp_path: Path, frozen: Path, master: Path) -> tuple[Path, int]:
    out = tmp_path / "out"
    base = ["--mock", "--limit", "2", "--cand", str(frozen), "--pools", str(master)]
    base += ["--out", str(out)]
    assert run.main(["--stage", "judge", *base]) == 0
    assert run.main(["--stage", "variants", *base]) == 0
    return out, run.main(["--stage", "factors", *base])


def test_f1_base_arm_runs_on_frozen_pool_and_treated_is_the_main_run(tmp_path: Path) -> None:
    frozen, master, _ = _setup(tmp_path)
    out, rc = _run_all(tmp_path, frozen, master)
    assert rc == 0
    v = json.loads((out / "variants.json").read_text())
    # main comparison ran on the master pool: B's top-5 is the master order
    master_q02 = [i["article_id"] for i in load_candidates("q02", master)["items"]]
    b_q02 = next(r for r in v["B"] if r["query_id"] == "q02")
    assert b_q02["final_ids"] == master_q02[:5]
    f = json.loads((out / "factors.json").read_text())
    for arm in f["top2"]:
        base = {r["query_id"]: r for r in f["F1"][arm]["base"]}
        treated = {r["query_id"]: r for r in f["F1"][arm]["treated"]}
        assert treated["q02"]["reused"] is True  # the main run is the demoted arm
        assert base["q02"]["reused"] is False  # re-run on the frozen pool
        assert base["q01"]["reused"] is True and treated["q01"]["reused"] is True  # same pool
        frozen_q02 = [i["article_id"] for i in load_candidates("q02", frozen)["items"]]
        if arm == "B":
            assert base["q02"]["final_ids"] == frozen_q02[:5]


def test_unlabelled_master_item_fails_loudly(tmp_path: Path) -> None:
    frozen, master, _ = _setup(tmp_path, extra_item=True)
    _, rc = _run_all(tmp_path, frozen, master)
    assert rc == 3  # STOPPED: a master pool item the judges never saw
