#!/usr/bin/env python
"""Before/after measurement for the V3(b2)(b) catalogue hygiene + category guard (no LLM, no network).

Arms, all in the PUBLIC view (default env: only public stores reach the retriever):
  A  baseline     base index copy, no ``category`` column  (guard is a no-op, so == master)
  D  removal only control: the base dense/BM25 indices untouched, the rows hygiene removes simply
                  absent from the catalogue frame (the retriever masks them out of both fetches);
                  isolates "these rows are gone" from "BM25 IDF and doc lengths shifted"
  B  hygiene      hygiene index copy (scripts/make_hygiene_index_copy.py), category_guard OFF
  C  hygiene+guard the same copy, category_guard ON (the shipped default)
Paired differences reported: D-A (effect of removing the rows), B-D (effect of rebuilding BM25
over the survivors), C-B (effect of the guard), C-A (total).

Metrics (queries are the resampling unit; seed 42; 10,000 paired bootstrap resamples):
  strict gold  scripts/eval_strict.py's --mode pipeline retrieval (``_retrieve_pipeline``, imported,
               not reimplemented) scored against eval/fixtures/strict_gold_labels.yaml:
               P@5 = relevant / (relevant + miss) over LABELLED top-5 items (eval_strict's
               definition), counts of labelled / unlabelled items, and nDCG@5 with binary labels,
               unlabelled counted as 0 and an ideal of five relevant items (a pessimistic bound
               that penalises an arm for unlabelled items; the unlabelled counts are reported next
               to it). Items that are unlabelled in B/C but were NOT in A's top-5 are listed.
  r1           scripts/eval_model.py's R1 stage (property-based relevance, no labels): P@5, P@10,
               nDCG@10 over the queries that carry a ``relevance`` block.

Usage:
    python scripts/eval_hygiene_ab.py --base-dir data/processed/unified \
        --hyg-dir data/processed/unified_hyg --out-dir reports/wave1b/v3b2b
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_QUERIES = _ROOT / "eval" / "fixtures" / "strict_gold_queries.yaml"
_LABELS = _ROOT / "eval" / "fixtures" / "strict_gold_labels.yaml"
_R1_FIXTURES = _ROOT / "eval" / "fixtures" / "model_eval_queries.yaml"
_IDCG5 = sum(1.0 / math.log2(i + 1) for i in range(1, 6))


def _load_labels() -> dict[tuple[str, str], bool]:
    raw = yaml.safe_load(_LABELS.read_text(encoding="utf-8"))["labels"]
    return {
        (e["query_id"], str(it["article_id"])): bool(it["relevant"])
        for e in raw
        for it in e["items"]
    }


def strict_rows(retriever: Any, queries: list[dict], labels: dict) -> list[dict[str, Any]]:
    """One row per query: top-5 ids and label outcome, via eval_strict's own pipeline mirror."""
    from eval_strict import _retrieve_pipeline

    rows = []
    for q in queries:
        items = _retrieve_pipeline(retriever, q["query"], q["gender"], occasion_gate=True)[:5]
        top = []
        for rank, it in enumerate(items, start=1):
            aid = str(it["article_id"])
            lab = labels.get((q["id"], aid))
            top.append(
                {"rank": rank, "article_id": aid, "store": it.get("store"),
                 "name": it.get("prod_name") or it.get("display_name"),
                 "label": None if lab is None else int(lab)}
            )
        rows.append({"id": q["id"], "query": q["query"], "category": q.get("category"), "top": top})
    return rows


def _per_query_counts(rows: list[dict]) -> np.ndarray:
    """(n, 4) array: relevant, miss, unlabelled, dcg_with_unlabelled_as_0."""
    out = np.zeros((len(rows), 4))
    for i, r in enumerate(rows):
        for t in r["top"]:
            if t["label"] is None:
                out[i, 2] += 1
            elif t["label"] == 1:
                out[i, 0] += 1
                out[i, 3] += 1.0 / math.log2(t["rank"] + 1)
            else:
                out[i, 1] += 1
    return out


def _p5(c: np.ndarray) -> float:
    d = c[:, 0].sum() + c[:, 1].sum()
    return float(c[:, 0].sum() / d) if d else float("nan")


def _ndcg5(c: np.ndarray) -> float:
    return float(c[:, 3].mean() / _IDCG5)


def _boot(stat, arrays: list[np.ndarray], n_boot: int, seed: int):
    """Paired bootstrap over queries. Returns dict of per-arm CI and pairwise-difference CIs."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = np.empty((n_boot, len(arrays)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, a in enumerate(arrays):
            vals[b, k] = stat(a[idx])
    return vals


def _ci(x: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def _summarise(name: str, stat, arrays: dict[str, np.ndarray], n_boot: int, seed: int) -> dict:
    keys = list(arrays)
    vals = _boot(stat, [arrays[k] for k in keys], n_boot, seed)
    out: dict[str, Any] = {"arms": {}, "diff": {}}
    for j, k in enumerate(keys):
        out["arms"][k] = {"point": stat(arrays[k]), "ci95": _ci(vals[:, j])}
    for a, b in (("D", "A"), ("B", "D"), ("B", "A"), ("C", "B"), ("C", "A")):
        if a not in keys or b not in keys:
            continue
        ja, jb = keys.index(a), keys.index(b)
        d = vals[:, ja] - vals[:, jb]
        out["diff"][f"{a}-{b}"] = {
            "point": stat(arrays[a]) - stat(arrays[b]),
            "ci95": _ci(d),
            "p_boot_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
        }
    return out


def r1_per_query(retriever: Any, queries: list[dict]) -> dict[str, dict[str, float]]:
    from eval_model import run_r1_stage

    res = run_r1_stage(queries, retriever)
    return {r["id"]: r for r in res["per_query"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-dir", required=True, type=Path)
    ap.add_argument("--hyg-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from eval_model import _build_components, load_fixture_queries

    queries = yaml.safe_load(_QUERIES.read_text(encoding="utf-8"))["queries"]
    labels = _load_labels()
    r1_queries = load_fixture_queries(_R1_FIXTURES)

    base = _build_components(need_agent=False, data_dir=args.base_dir)["retriever"]
    hyg = _build_components(need_agent=False, data_dir=args.hyg_dir)["retriever"]
    assert "category" not in base.catalogue_df.columns, "base index must predate hygiene"
    assert "category" in hyg.catalogue_df.columns, "hygiene index lacks the category column"

    # Control D: base indices, but a catalogue frame without the rows hygiene removed (see module
    # docstring). HybridRetriever builds its fetch masks from the frame, so removed rows cannot
    # occupy any fetch slot, while BM25 IDF / doc lengths stay those of the base index.
    from src.retrieval.hybrid_search import HybridRetriever

    kept_ids = set(hyg.catalogue_df.index.astype(str))
    base_frame = base.catalogue_df.reset_index()
    control = HybridRetriever(
        base.dense,
        base.sparse,
        base_frame[base_frame["article_id"].astype(str).isin(kept_ids)],
        copy.deepcopy(base.config),
    )
    assert len(control.catalogue_df) == len(hyg.catalogue_df), "control frame != hygiene frame"

    arms = {"A": (base, None), "D": (control, None), "B": (hyg, False), "C": (hyg, True)}
    strict: dict[str, list[dict]] = {}
    r1: dict[str, dict] = {}
    for arm, (retr, guard) in arms.items():
        if guard is not None:
            retr.config["retrieval"]["category_guard"] = guard
        print(f"arm {arm}: guard={guard} strict + r1 ...", flush=True)
        strict[arm] = strict_rows(retr, queries, labels)
        r1[arm] = r1_per_query(retr, r1_queries)

    counts = {a: _per_query_counts(strict[a]) for a in arms}
    report: dict[str, Any] = {
        "seed": args.seed, "n_boot": args.boot, "n_strict_queries": len(queries),
        "index_rows": {"base": len(base.catalogue_df), "hyg": len(hyg.catalogue_df)},
        "strict": {
            "p_at_5": _summarise("p5", _p5, counts, args.boot, args.seed),
            "ndcg_at_5_unlabelled_as_0": _summarise("ndcg", _ndcg5, counts, args.boot, args.seed),
            "labelled_items": {a: int(counts[a][:, 0].sum() + counts[a][:, 1].sum()) for a in arms},
            "relevant_items": {a: int(counts[a][:, 0].sum()) for a in arms},
            "unlabelled_items": {a: int(counts[a][:, 2].sum()) for a in arms},
            "queries_with_fewer_than_5_items": {
                a: sum(1 for r in strict[a] if len(r["top"]) < 5) for a in arms
            },
        },
    }
    a_unl = {(r["id"], t["article_id"]) for r in strict["A"] for t in r["top"] if t["label"] is None}
    report["strict"]["new_unlabelled_vs_A"] = {
        a: [
            {"query_id": r["id"], "article_id": t["article_id"], "store": t["store"], "name": t["name"]}
            for r in strict[a] for t in r["top"]
            if t["label"] is None and (r["id"], t["article_id"]) not in a_unl
        ]
        for a in ("D", "B", "C")
    }
    # r1: only queries scored in every arm
    common = sorted(set.intersection(*(set(r1[a]) for a in arms)))
    report["r1"] = {"n_queries": len(common)}
    for metric in ("precision_at_5", "precision_at_10", "ndcg_at_10"):
        arrs = {a: np.array([[r1[a][q][metric]] for q in common], dtype=float) for a in arms}
        report["r1"][metric] = _summarise(metric, lambda x: float(np.mean(x)), arrs, args.boot, args.seed)
    changed = [q for q in common if r1["A"][q]["precision_at_5"] != r1["C"][q]["precision_at_5"]]
    report["r1"]["queries_whose_p5_changed_A_to_C"] = changed
    # top-5 membership change (any arm pair), strict
    report["strict"]["queries_whose_top5_changed"] = {
        f"{x}-{y}": [
            r["id"] for r, s in zip(strict[x], strict[y], strict=True)
            if [t["article_id"] for t in r["top"]] != [t["article_id"] for t in s["top"]]
        ]
        for x, y in (("A", "D"), ("D", "B"), ("B", "C"), ("A", "C"))
    }

    (args.out_dir / "ab_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "ab_strict_rows.json").write_text(json.dumps(strict, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "strict"}, indent=2)[:3000])
    print(json.dumps({k: v for k, v in report["strict"].items() if k != "new_unlabelled_vs_A"}, indent=2))
    print("new unlabelled", {a: len(v) for a, v in report["strict"]["new_unlabelled_vs_A"].items()})


if __name__ == "__main__":
    main()
