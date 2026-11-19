#!/usr/bin/env python
"""Before/after measurement for a store ``allowed_categories`` restriction (no LLM, no network).

Two index copies made by ``scripts/make_hygiene_index_copy.py`` (same procedure, same source):
  --before-dir  hygiene applied, no restriction
  --after-dir   the same plus ``--allowed-categories STORE=CAT,...``
Public view (default env), seed 42, 10,000 paired bootstrap resamples over queries. Reuses the
metrics of ``scripts/eval_hygiene_ab.py`` (strict gold P@5 over labelled items, nDCG@5 with
unlabelled as 0, R1 precision) and reports, in ``allowed_categories_ab.json``: both arms' point
values with 95% intervals, the paired difference (after - before) with its interval, labelled /
unlabelled counts, and every (query_id, article_id) that is unlabelled in a top-5 of either arm.
Arm names in the json follow eval_hygiene_ab's pairing: "A" = before, "B" = after (diff "B-A").

Usage:
    python scripts/eval_allowed_categories_ab.py --before-dir <hyg> --after-dir <hyg_restricted> \
        --out-dir reports/wave1b/r1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(_SCRIPTS_DIR.parent), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_hygiene_ab as ab  # noqa: E402


def unlabelled_items(rows: list[dict]) -> list[dict[str, Any]]:
    """Every top-5 item without a gold label, as ``(query_id, article_id)`` records."""
    return [
        {"query_id": r["id"], "article_id": t["article_id"], "store": t["store"], "name": t["name"]}
        for r in rows
        for t in r["top"]
        if t["label"] is None
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--before-dir", required=True, type=Path)
    ap.add_argument("--after-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from eval_model import _build_components, load_fixture_queries

    queries = yaml.safe_load(ab._QUERIES.read_text(encoding="utf-8"))["queries"]
    labels = ab._load_labels()
    r1_queries = load_fixture_queries(ab._R1_FIXTURES)
    arms = {
        "A": _build_components(need_agent=False, data_dir=args.before_dir)["retriever"],
        "B": _build_components(need_agent=False, data_dir=args.after_dir)["retriever"],
    }
    strict = {a: ab.strict_rows(r, queries, labels) for a, r in arms.items()}
    r1 = {a: ab.r1_per_query(r, r1_queries) for a, r in arms.items()}
    counts = {a: ab._per_query_counts(strict[a]) for a in arms}
    common = sorted(set(r1["A"]) & set(r1["B"]))
    report: dict[str, Any] = {
        "seed": args.seed,
        "n_boot": args.boot,
        "index_rows": {a: len(r.catalogue_df) for a, r in arms.items()},
        "strict_p_at_5": ab._summarise("p5", ab._p5, counts, args.boot, args.seed),
        "strict_ndcg_at_5_unlabelled_as_0": ab._summarise(
            "ndcg", ab._ndcg5, counts, args.boot, args.seed
        ),
        "labelled_items": {a: int(counts[a][:, :2].sum()) for a in arms},
        "relevant_items": {a: int(counts[a][:, 0].sum()) for a in arms},
        "unlabelled_items_count": {a: int(counts[a][:, 2].sum()) for a in arms},
        "unlabelled_items": {a: unlabelled_items(strict[a]) for a in arms},
        "queries_with_fewer_than_5_items": {
            a: sum(1 for r in strict[a] if len(r["top"]) < 5) for a in arms
        },
        "top5_changed_queries": [
            r["id"]
            for r, s in zip(strict["A"], strict["B"], strict=True)
            if [t["article_id"] for t in r["top"]] != [t["article_id"] for t in s["top"]]
        ],
        "r1_n_queries": len(common),
    }
    for metric in ("precision_at_5", "precision_at_10", "ndcg_at_10"):
        arrs = {a: np.array([[r1[a][q][metric]] for q in common], dtype=float) for a in arms}
        report[f"r1_{metric}"] = ab._summarise(
            metric, lambda x: float(np.mean(x)), arrs, args.boot, args.seed
        )
    (args.out_dir / "allowed_categories_ab.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.out_dir / "allowed_categories_ab_rows.json").write_text(
        json.dumps(strict, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "unlabelled_items"}, indent=2))


if __name__ == "__main__":
    main()
