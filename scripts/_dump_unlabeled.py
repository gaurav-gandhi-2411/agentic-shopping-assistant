#!/usr/bin/env python
"""One-off: dump (query_id, article_id) pairs eval_strict.py's pipeline mode
retrieves that have no hand label yet, for manual review. Not a permanent
script — supports the wave-measurement retrieval fix's regression check.
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
        (entry["query_id"], str(item["article_id"]))
        for entry in labels_raw
        for item in entry["items"]
    }

    comps = _build_components(need_agent=False, data_dir=_ROOT / "data" / "processed" / "unified")
    retriever = comps["retriever"]

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
                })

    out_path = _ROOT / "reports" / "unlabeled_after_pushdown_fix.yaml"
    out_path.write_text(yaml.safe_dump(unlabeled, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"{len(unlabeled)} unlabeled (query_id, article_id) pairs written to {out_path}")


if __name__ == "__main__":
    main()
