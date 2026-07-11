#!/usr/bin/env python
"""Strict gold-relevance eval — human-audited precision@5, no self-grading.

Re-runs retrieval for eval/fixtures/strict_gold_queries.yaml (mirroring
eval_model.py's R1 stage call exactly), scores the top-5 against the
HAND labels in strict_gold_labels.yaml, and prints strict precision@5 plus a
miss taxonomy split into CODE-FIXABLE vs DATA-CEILING/DATA-QUALITY causes.

An item retrieval returns that has NO label is counted separately as
`unlabeled` and NEVER scored — the checker must never grade itself. A run with
unlabeled items means retrieval changed since labeling: re-audit those items,
extend the label file, re-run.

Usage:
    python scripts/eval_strict.py
    python scripts/eval_strict.py --data-dir data/processed/unified
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

_SCRIPTS_DIR = Path(__file__).parent
_ROOT = _SCRIPTS_DIR.parent
for p in (str(_ROOT), str(_SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

_QUERIES_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_queries.yaml"
_LABELS_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_labels.yaml"

# Which miss reasons a code change can remove vs what only data/inventory can.
CODE_FIXABLE_REASONS = frozenset({
    "type-confusion", "set-not-single", "kids-leak", "budget",
    "occasion-register", "attribute-contradiction",
})
DATA_REASONS = frozenset({"data-ceiling", "data-mislabel"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=str(_ROOT / "data" / "processed" / "unified"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    from eval_model import _build_components  # heavy import deferred past --help

    queries = yaml.safe_load(_QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    labels_raw = yaml.safe_load(_LABELS_PATH.read_text(encoding="utf-8"))["labels"]
    labels: dict[tuple[str, str], dict] = {
        (entry["query_id"], str(item["article_id"])): item
        for entry in labels_raw
        for item in entry["items"]
    }

    comps = _build_components(need_agent=False, data_dir=Path(args.data_dir))
    retriever = comps["retriever"]

    n_scored = n_relevant = n_unlabeled = 0
    reasons: Counter[str] = Counter()
    per_query: list[tuple[str, str, int, int, int]] = []

    for q in queries:
        items = retriever.search(q["query"], top_k=50, filters={"gender": q["gender"]})
        top = items[: args.top_k]
        rel = miss = unl = 0
        for it in top:
            key = (q["id"], str(it.get("article_id")))
            label = labels.get(key)
            if label is None:
                unl += 1
                continue
            if label["relevant"]:
                rel += 1
            else:
                miss += 1
                reasons[label.get("reason", "unspecified")] += 1
        n_scored += rel + miss
        n_relevant += rel
        n_unlabeled += unl
        per_query.append((q["id"], q["query"], rel, miss, unl))

    print(f"\nSTRICT GOLD EVAL — hand-audited relevance (rubric: {_QUERIES_PATH.name})")
    print(f"queries={len(queries)}  scored_items={n_scored}  unlabeled_items={n_unlabeled}")
    if n_unlabeled:
        print("  ** UNLABELED ITEMS PRESENT — retrieval changed since labeling. **")
        print("  ** Strict P@5 below covers labeled items only; re-audit before comparing. **")
    p5 = n_relevant / n_scored if n_scored else 0.0
    print(f"\nstrict precision@{args.top_k}: {p5:.3f}  ({n_relevant}/{n_scored})")

    code_misses = sum(c for r, c in reasons.items() if r in CODE_FIXABLE_REASONS)
    data_misses = sum(c for r, c in reasons.items() if r in DATA_REASONS)
    print("\nmiss taxonomy:")
    for reason, count in reasons.most_common():
        bucket = ("CODE-FIXABLE" if reason in CODE_FIXABLE_REASONS
                  else "DATA" if reason in DATA_REASONS else "?")
        print(f"  {count:3d}  {reason:24s} [{bucket}]")
    if n_scored:
        ceiling_p5 = (n_relevant + code_misses) / n_scored
        print(f"\nif every CODE-FIXABLE miss were fixed: {ceiling_p5:.3f} "
              f"(remaining gap = {data_misses} data-capped items)")

    print("\nweakest queries:")
    for qid, text, rel, miss, unl in sorted(per_query, key=lambda t: t[2]):
        if miss + unl == 0:
            continue
        print(f"  {qid}  {rel}/{rel + miss + unl}  {text!r}")


if __name__ == "__main__":
    main()
