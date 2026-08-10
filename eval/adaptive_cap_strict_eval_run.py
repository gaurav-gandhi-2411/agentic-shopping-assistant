#!/usr/bin/env python
"""Decisive strict-eval confirmation for the adaptive_per_store_cap candidate.

`eval/adaptive_per_store_cap_test.py` measured the query-adaptive per_store_cap
against cheap proxy metrics only (recall_subset_queries.yaml / literal P@5).
Per eval/README.md's own standing rule ("Proxy metrics may gate exploration;
only the hand-labeled strict eval gates shipping" — precedent:
reports/pushdown_fix_20260806.md, where the SAME literal-P@5 proxy read a cap
change as flat while the real strict eval showed a genuine -0.8pp/-2.4pp
occasion regression) a proxy win is not sufficient evidence to ship. This
script runs the REAL strict eval (scripts/eval_strict.py --mode pipeline,
hand-labeled eval/fixtures/strict_gold_labels.yaml) under the adaptive-cap
monkeypatch, following the exact same two established patterns rather than
reinventing either:
  - the adaptive-cap monkeypatch itself: reuses
    eval.adaptive_per_store_cap_test's `_adaptive_apply_per_store_cap` /
    `compute_effective_cap` verbatim (same patch target, same formula).
  - the unlabeled-pair discovery: reuses scripts/_dump_unlabeled.py's exact
    pattern (retrieve top-5 via eval_strict._retrieve_pipeline, mode
    pipeline, diff against strict_gold_labels.yaml's existing keys) but under
    the adaptive-cap patch instead of the shipped fixed cap.

Two-step workflow (label the gap between steps by hand, per eval/README.md —
the checker must never grade itself):

    python -m eval.adaptive_cap_strict_eval_run --step dump
        -> writes reports/unlabeled_after_adaptive_cap.yaml, one entry per
           (query_id, article_id) pair the adaptive cap surfaces in the top-5
           that has no existing hand label. Hand-label each into
           eval/fixtures/strict_gold_labels.yaml (same file, same shape —
           see that file's own header for the reason taxonomy), then re-run
           this step until it reports 0 unlabeled.

    python -m eval.adaptive_cap_strict_eval_run --step compare
        -> once strict_gold_labels.yaml is fully labeled for both configs,
           runs scripts/eval_strict.py --mode pipeline TWICE for real: once
           unmodified (today's shipped fixed cap=4), once with the
           adaptive-cap monkeypatch applied. Writes both --json-out summaries
           (overall + per-category precision) under reports/ for the report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent
_SCRIPTS_DIR = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import src.retrieval.hybrid_search as hybrid_search_module  # noqa: E402
from eval.adaptive_per_store_cap_test import (  # noqa: E402
    _adaptive_apply_per_store_cap,
    compute_effective_cap,  # noqa: F401  (re-exported for callers/report scripts)
)

_QUERIES_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_queries.yaml"
_LABELS_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_labels.yaml"
_UNLABELED_OUT = _ROOT / "reports" / "unlabeled_after_adaptive_cap.yaml"
_FIXED_JSON_OUT = _ROOT / "reports" / "_adaptive_cap_compare_fixed.json"
_ADAPTIVE_JSON_OUT = _ROOT / "reports" / "_adaptive_cap_compare_adaptive.json"

_REAL_APPLY_PER_STORE_CAP = hybrid_search_module.apply_per_store_cap


def _dump_unlabeled() -> None:
    """Find every (query_id, article_id) pair the adaptive cap surfaces in the
    top-5 that has no existing hand label, and write it for manual review.

    Mirrors scripts/_dump_unlabeled.py's logic exactly (same retriever
    assembly, same _retrieve_pipeline call, same diff-against-labels
    approach), with the adaptive-cap monkeypatch applied around retrieval and
    always restored afterward, even on error.
    """
    from eval_model import _build_components
    from eval_strict import _retrieve_pipeline

    queries = yaml.safe_load(_QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    labels_raw = yaml.safe_load(_LABELS_PATH.read_text(encoding="utf-8"))["labels"]
    labels = {
        (entry["query_id"], str(item["article_id"]))
        for entry in labels_raw
        for item in entry["items"]
    }

    comps = _build_components(need_agent=False, data_dir=_ROOT / "data" / "processed" / "unified")
    retriever = comps["retriever"]

    hybrid_search_module.apply_per_store_cap = _adaptive_apply_per_store_cap
    try:
        unlabeled: list[dict] = []
        for q in queries:
            items = _retrieve_pipeline(retriever, q["query"], q["gender"], occasion_gate=True)
            for rank, it in enumerate(items[:5], start=1):
                aid = str(it.get("article_id"))
                key = (q["id"], aid)
                if key not in labels:
                    unlabeled.append({
                        "query_id": q["id"],
                        "query": q["query"],
                        "category": q.get("category"),
                        "rank": rank,
                        "article_id": aid,
                        "prod_name": it.get("prod_name") or it.get("display_name"),
                        "colour": it.get("colour"),
                        "product_type": it.get("product_type"),
                        "price_inr": it.get("price_inr"),
                        "gender": it.get("gender"),
                        "detail_desc": it.get("detail_desc"),
                    })
    finally:
        hybrid_search_module.apply_per_store_cap = _REAL_APPLY_PER_STORE_CAP

    _UNLABELED_OUT.write_text(
        yaml.safe_dump(unlabeled, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"{len(unlabeled)} unlabeled (query_id, article_id) pairs written to {_UNLABELED_OUT}")


def _compare() -> None:
    """Run scripts/eval_strict.py --mode pipeline for real, twice, against the
    now-fully-labeled strict_gold_labels.yaml: once with the shipped fixed
    cap=4 (unmodified src/), once with the adaptive-cap monkeypatch applied.
    Each run writes a --json-out summary (overall + per-category precision)
    consumed by this task's report.
    """
    import eval_strict

    old_argv = sys.argv
    print("\n########## FIXED cap=4 (shipped, unmodified) ##########")
    try:
        sys.argv = ["eval_strict.py", "--mode", "pipeline", "--json-out", str(_FIXED_JSON_OUT)]
        eval_strict.main()
    finally:
        sys.argv = old_argv

    print("\n########## ADAPTIVE cap (query-aware, monkeypatched) ##########")
    hybrid_search_module.apply_per_store_cap = _adaptive_apply_per_store_cap
    try:
        sys.argv = ["eval_strict.py", "--mode", "pipeline", "--json-out", str(_ADAPTIVE_JSON_OUT)]
        eval_strict.main()
    finally:
        hybrid_search_module.apply_per_store_cap = _REAL_APPLY_PER_STORE_CAP
        sys.argv = old_argv

    print(f"\nfixed summary:    {_FIXED_JSON_OUT}")
    print(f"adaptive summary: {_ADAPTIVE_JSON_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", choices=("dump", "compare"), default="dump")
    args = parser.parse_args()
    if args.step == "dump":
        _dump_unlabeled()
    else:
        _compare()


if __name__ == "__main__":
    main()
