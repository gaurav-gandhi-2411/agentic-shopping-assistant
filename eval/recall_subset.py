#!/usr/bin/env python
"""Recall@k on the bounded-universe literal-query subset.

Runs eval/fixtures/recall_subset_queries.yaml's 28 tight-filter queries
against the live HybridRetriever and reports recall@5/10/20/50. Every query's
candidate universe (catalogue_universe_ids over product_type_contains +
gender_in + colour_in + price_max) was verified <=100 items when the fixture
was built — see that file's header for why this is the ONLY query class this
repo can currently compute real recall for, and what it does NOT measure
(literal filter-satisfaction, not general shopper-relevant recall).

Usage:
    python -m eval.recall_subset
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS_DIR = _ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_model import catalogue_universe_ids, recall_at_k  # noqa: E402
from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_FIXTURE_PATH = _ROOT / "eval" / "fixtures" / "recall_subset_queries.yaml"
_DATA_DIR = _ROOT / "data" / "processed" / "unified"
_KS = (5, 10, 20, 50)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    config = load_config()
    catalogue_df = __import__("pandas").read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)

    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]

    # Two passes, deliberately NOT conflated:
    #   "gender_only" — mirrors run_r1_stage's existing methodology exactly (only
    #     the gender filter is pushed into HybridRetriever.search; colour/price are
    #     left for the dense+sparse ranker to surface from free text alone). This is
    #     the number directly comparable to the main 227-query fixture's Stage 2.
    #   "structured" — additionally pushes colour_group_name + price_max as hard
    #     facet filters, matching what src/agents/graph.py's router DOES push into
    #     search() when it parses those out of a real user query (see graph.py
    #     ~line 1161-1224). product_type_name is deliberately NOT pushed here: this
    #     fixture's universe used substring matching (e.g. "gown" items whose real
    #     product_type_name is "dress") to bound the universe, which is broader than
    #     the exact-match facet vocabulary the production LLM router extracts — a
    #     type filter here would not be an apples-to-apples comparison, so it's left
    #     to the ranker on both passes.
    per_query: list[dict[str, Any]] = []
    for q in queries:
        last_turn = q["turns"][-1]
        gender = (q.get("expected_intent") or {}).get("gender")
        must = q["relevance"]["must"]

        gender_only_filters = {"gender": gender} if gender else None
        items_gender_only = retriever.search(last_turn, top_k=max(_KS), filters=gender_only_filters)
        retrieved_gender_only = [it["article_id"] for it in items_gender_only]

        structured_filters = dict(gender_only_filters or {})
        colour_in = must.get("colour_in") or []
        if colour_in:
            structured_filters["colour_group_name"] = colour_in[0]
        if must.get("price_max") is not None:
            structured_filters["price_max"] = must["price_max"]
        items_structured = retriever.search(last_turn, top_k=max(_KS), filters=structured_filters)
        retrieved_structured = [it["article_id"] for it in items_structured]

        universe_ids = catalogue_universe_ids(retriever.catalogue_df, must)
        universe_size = len(universe_ids)

        if universe_size == 0:
            print(f"  [SKIP] {q['id']}: universe size 0 — must-filter matched no catalogue items "
                  f"(fixture drifted from current catalogue snapshot)")
            continue
        if universe_size > 100:
            print(f"  [SKIP] {q['id']}: universe size {universe_size} > 100 — no longer bounded "
                  f"(fixture drifted from current catalogue snapshot)")
            continue

        rec = {
            "id": q["id"],
            "query": q["query"],
            "universe_size": universe_size,
            **{f"recall_at_{k}": recall_at_k(retrieved_gender_only, universe_ids, k) for k in _KS},
            **{f"structured_recall_at_{k}": recall_at_k(retrieved_structured, universe_ids, k) for k in _KS},
        }
        per_query.append(rec)
        print(
            f"  {q['id']:<14} universe={universe_size:>3}  gender-only "
            + "  ".join(f"R@{k}={rec[f'recall_at_{k}']:.2f}" for k in _KS)
            + "  |  structured "
            + "  ".join(f"R@{k}={rec[f'structured_recall_at_{k}']:.2f}" for k in _KS)
        )

    print(f"\n{'=' * 70}")
    print(f"Scored {len(per_query)}/{len(queries)} queries (rest skipped — see SKIP lines above)")
    print("gender-only filter (mirrors existing Stage 2 R1 methodology):")
    for k in _KS:
        vals = [r[f"recall_at_{k}"] for r in per_query]
        print(f"  mean recall@{k}: {_mean(vals):.3f}  (n={len(vals)})")
    print("structured filter (gender + colour_group_name + price_max, matches graph.py's router):")
    for k in _KS:
        vals = [r[f"structured_recall_at_{k}"] for r in per_query]
        print(f"  mean structured_recall@{k}: {_mean(vals):.3f}  (n={len(vals)})")

    ts = __import__("datetime").datetime.now(__import__("datetime").UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = _ROOT / "reports" / f"recall_subset_{ts}.json"
    import json

    report_path.write_text(
        json.dumps(
            {
                "scope_caveat": (
                    "Literal tight-filter subset only (type+gender+colour+price). "
                    "Measures 'does retrieval find items matching a hard filter', "
                    "NOT general shopper-relevant recall. Do not cite as system recall. "
                    "'recall_at_k' = gender-only filter pushed (matches existing Stage 2 R1 "
                    "methodology, comparable to the main 227-query fixture). "
                    "'structured_recall_at_k' = gender+colour+price pushed as hard facet "
                    "filters (matches what src/agents/graph.py's router pushes into search() "
                    "for a real user query with those fields parsed out); product_type_name is "
                    "NOT pushed on either pass — see recall_subset.py's inline comment for why "
                    "that filter isn't an apples-to-apples match to this fixture's universe "
                    "definition."
                ),
                "n_scored": len(per_query),
                "n_total": len(queries),
                "mean_recall": {f"recall_at_{k}": _mean([r[f"recall_at_{k}"] for r in per_query]) for k in _KS},
                "mean_structured_recall": {
                    f"structured_recall_at_{k}": _mean([r[f"structured_recall_at_{k}"] for r in per_query])
                    for k in _KS
                },
                "per_query": per_query,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nReport written: {report_path}")


if __name__ == "__main__":
    main()
