#!/usr/bin/env python
"""One-off: dump the strict-gold unlabeled (query_id, article_id) pairs with item
detail so they can be hand-audited into eval/fixtures/strict_gold_labels.yaml.

Throwaway utility for the 2026-07-13 kids-leak-fix re-audit — not wired into CI.
Mirrors eval_strict.py's --mode pipeline retrieval + labels-lookup exactly so the
(query_id, article_id) keys line up with what eval_gate.py actually scores.

Usage:
    python scripts/_dump_unlabeled_strict.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent
_SCRIPTS_DIR = Path(__file__).parent
for p in (str(_ROOT), str(_SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_model import _build_components  # noqa: E402
from eval_strict import _retrieve_pipeline  # noqa: E402

_QUERIES_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_queries.yaml"
_LABELS_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_labels.yaml"


def main() -> None:
    """Print (query_id, article_id, query, title, product_type, gender, colour, price)
    for every top-5 retrieved item that has no entry in strict_gold_labels.yaml."""
    queries = yaml.safe_load(_QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    labels_raw = yaml.safe_load(_LABELS_PATH.read_text(encoding="utf-8"))["labels"]
    labels: dict[tuple[str, str], dict] = {
        (entry["query_id"], str(item["article_id"])): item
        for entry in labels_raw
        for item in entry["items"]
    }

    comps = _build_components(need_agent=False, data_dir=_ROOT / "data" / "processed" / "unified")
    retriever = comps["retriever"]

    n = 0
    for q in queries:
        items = _retrieve_pipeline(retriever, q["query"], q["gender"], occasion_gate=True)
        top = items[:5]
        for rank, it in enumerate(top, start=1):
            key = (q["id"], str(it.get("article_id")))
            if key in labels:
                continue
            n += 1
            title = it.get("prod_name") or it.get("display_name") or ""
            print(f"\n--- unlabeled #{n} ---")
            print(f"query_id:     {q['id']}")
            print(f"query:        {q['query']!r}")
            print(f"gender_arg:   {q['gender']}")
            print(f"rank:         {rank}")
            print(f"article_id:   {it.get('article_id')}")
            print(f"title:        {title!r}")
            print(f"product_type: {it.get('product_type')!r}")
            print(f"gender:       {it.get('gender')!r}")
            print(f"colour:       {it.get('colour')!r}")
            print(f"price_inr:    {it.get('price_inr')!r}")
            print(f"detail_desc:  {(it.get('detail_desc') or '')[:200]!r}")

    print(f"\ntotal unlabeled: {n}")


if __name__ == "__main__":
    main()
