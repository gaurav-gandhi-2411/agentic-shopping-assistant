#!/usr/bin/env python
"""One-off: dump every CURRENT top-5 miss (not the full historical label
file, which accumulates labels across old retrieval snapshots) with full
detail -- category, reason, note, query, item -- for the compose-logic
miss-taxonomy breakdown. Mirrors scripts/eval_strict.py's own main() loop
exactly (same _retrieve_pipeline call) so counts match its printed summary.
Not a permanent script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

_SCRIPTS_DIR = Path(__file__).parent
_ROOT = _SCRIPTS_DIR.parent
for p in (str(_ROOT), str(_SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_model import _build_components  # noqa: E402
from eval_strict import _retrieve_pipeline  # noqa: E402

_QUERIES_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_queries.yaml"
_LABELS_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_labels.yaml"


def main() -> None:
    queries = yaml.safe_load(_QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    labels_raw = yaml.safe_load(_LABELS_PATH.read_text(encoding="utf-8"))["labels"]
    labels = {
        (entry["query_id"], str(item["article_id"])): item
        for entry in labels_raw
        for item in entry["items"]
    }

    comps = _build_components(need_agent=False, data_dir=_ROOT / "data" / "processed" / "unified")
    retriever = comps["retriever"]

    misses = []
    n_scored = n_relevant = n_unlabeled = 0
    for q in queries:
        items = _retrieve_pipeline(retriever, q["query"], q["gender"], occasion_gate=True)
        for rank, it in enumerate(items[:5], start=1):
            aid = str(it.get("article_id"))
            label = labels.get((q["id"], aid))
            if label is None:
                n_unlabeled += 1
                continue
            n_scored += 1
            if label["relevant"]:
                n_relevant += 1
                continue
            misses.append({
                "query_id": q["id"],
                "category": q.get("category"),
                "query": q["query"],
                "rank": rank,
                "article_id": aid,
                "prod_name": it.get("prod_name") or it.get("display_name"),
                "colour": it.get("colour"),
                "product_type": it.get("product_type"),
                "price_inr": it.get("price_inr"),
                "reason": label.get("reason", "unspecified"),
                "note": label.get("note", ""),
            })

    print(f"n_scored={n_scored} n_relevant={n_relevant} n_miss={len(misses)} n_unlabeled={n_unlabeled}")
    out_path = _ROOT / "reports" / "strict_misses_full_20260807.yaml"
    out_path.write_text(yaml.safe_dump(misses, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Full miss detail written to {out_path}")


if __name__ == "__main__":
    main()
