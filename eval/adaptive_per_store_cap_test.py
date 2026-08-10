#!/usr/bin/env python
"""Query-adaptive per_store_cap: design + measurement, NOT a src/ change.

Prior work (reports/part3_pipeline_bottleneck_20260807.md, recommendation #2)
flagged that a single GLOBAL per_store_cap value mechanically truncates
recall on queries whose true qualifying universe is itself store-concentrated
(nothing genuine to diversify against — see
reports/recall_gap_diagnosis_20260806.md's maroon-dupatta case: 55-item
universe, 95% in one store, cap=4 truncates 15->6 post-dedup items regardless
of ranking quality). Raising the cap GLOBALLY (4->8) fixed that but cost
strict P@5 on the broad/occasion population (reports/pushdown_fix_20260806.md
-style regression noted in eval/README.md).

This script tests a QUERY-ADAPTIVE cap instead:

    effective_cap = base_cap                                   if n_stores <= 0
    effective_cap = max(base_cap, ceil(top_k / n_stores))       otherwise

where n_stores = number of distinct stores in `full_pool` (the wide,
pre-diversity-rerank, filter-passing candidate pool apply_per_store_cap
already receives — see src/retrieval/hybrid_search.py::apply_per_store_cap's
own docstring: "full_pool ... already contains every filter-passing
candidate, so {item.get('store') for item in full_pool} gives you the true
count of distinct stores actually holding qualifying inventory for that
query").

Rationale / degeneracy check (both fall out of the same max() with no extra
branch needed):
  - n_stores == 1: ceil(top_k/1) == top_k, so effective_cap == max(base_cap,
    top_k) == top_k (top_k >= base_cap always in this codebase) -> the cap
    can no longer truncate a single-store universe below top_k. (In practice
    apply_per_store_cap's OWN guard already no-ops whenever the full_pool has
    <2 distinct stores, so this case never even reaches a live cap check --
    the formula still degenerates correctly, it's just redundant with an
    existing safety net.)
  - n_stores large (>= top_k/base_cap): ceil(top_k/n_stores) <= base_cap, so
    max(...) picks base_cap -- IDENTICAL to today's fixed-cap behaviour. No
    separate "only when small" gate is needed; the max() does it structurally.

Implementation note (why no src/ edit was needed): apply_per_store_cap is
called by HybridRetriever.search() as a bare module-level name lookup
(`results = apply_per_store_cap(...)`), resolved from
src.retrieval.hybrid_search's own module globals at call time. Monkeypatching
that module attribute to a thin wrapper -- which computes effective_cap from
the full_pool it's handed, then calls through to the REAL, unmodified
apply_per_store_cap with that computed int -- exercises the exact production
capping/backfill logic end to end, with only the cap VALUE swapped per call.
No production file is edited; the patch lives entirely in this script's
process and is restored at the end of main().

Usage:
    python -m eval.adaptive_per_store_cap_test
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS_DIR = _ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_model import catalogue_universe_ids, precision_at_k, recall_at_k  # noqa: E402

import src.retrieval.hybrid_search as hybrid_search_module  # noqa: E402
from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed" / "unified"
_NICHE_FIXTURE = _ROOT / "eval" / "fixtures" / "recall_subset_queries.yaml"
_BROAD_FIXTURE = _ROOT / "eval" / "fixtures" / "model_eval_queries.yaml"
_KS = (5, 10, 20, 50)
_BASE_CAP = 4  # today's shipped fixed cap (config.yaml retrieval.per_store_cap)
_TOP_K = 50  # matches both per_store_cap_sweep.py and per_store_cap_sweep_broad.py

_REAL_APPLY_PER_STORE_CAP = hybrid_search_module.apply_per_store_cap

# Populated by _adaptive_apply_per_store_cap on every call, read by the caller
# immediately after retriever.search() returns (single-threaded, synchronous —
# safe to use as a call-scoped out-param without a lock).
_last_call_info: dict[str, int] = {}


def compute_effective_cap(n_stores: int, top_k: int, base_cap: int) -> int:
    """Query-adaptive per-store cap.

    See module docstring for full rationale and degeneracy proof. Returns
    `base_cap` unchanged when `n_stores <= 0` (empty pool — cap is moot;
    apply_per_store_cap's own <2-distinct-stores guard will no-op anyway for
    n_stores in {0, 1}).
    """
    if n_stores <= 0:
        return base_cap
    return max(base_cap, math.ceil(top_k / n_stores))


def _adaptive_apply_per_store_cap(
    selected: list[dict], full_pool: list[dict], cap: int, top_k: int
) -> list[dict]:
    """Drop-in replacement for hybrid_search.apply_per_store_cap.

    Ignores the `cap` argument HybridRetriever.search() passes (the fixed
    config value) and recomputes a query-adaptive cap from `full_pool`
    instead, then delegates to the REAL apply_per_store_cap with that value —
    the capping/backfill logic itself is never reimplemented.
    """
    n_stores = len({item.get("store") for item in full_pool})
    effective_cap = compute_effective_cap(n_stores, top_k, _BASE_CAP)
    _last_call_info["n_stores"] = n_stores
    _last_call_info["effective_cap"] = effective_cap
    return _REAL_APPLY_PER_STORE_CAP(selected, full_pool, effective_cap, top_k)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _diversity(items: list[dict], k: int) -> tuple[int, float]:
    stores = [it.get("store") for it in items[:k] if it.get("store")]
    if not stores:
        return 0, 0.0
    counts = Counter(stores)
    return len(counts), max(counts.values()) / len(stores)


def _select_broad_queries(retriever: HybridRetriever) -> list[dict]:
    """Identical selection logic to per_store_cap_sweep_broad.py, duplicated
    here (not imported — that module executes its sweep at import time via
    no __main__ guard issue, but duplicating this ~20-line selector keeps
    this script self-contained and matches the template's own approach of
    each sweep script being independently runnable)."""
    data = yaml.safe_load(_BROAD_FIXTURE.read_text(encoding="utf-8"))
    seen_text: set[str] = set()
    out: list[dict] = []
    for q in data["queries"]:
        if q.get("category") == "refinement":
            continue
        relevance = q.get("relevance")
        if not relevance:
            continue
        text = q["turns"][-1]
        if text.lower() in seen_text:
            continue
        must = relevance.get("must") or {}
        universe = catalogue_universe_ids(retriever.catalogue_df, must)
        if len(universe) <= 500:
            continue
        seen_text.add(text.lower())
        out.append(q)
        if len(out) >= 20:
            break
    return out


def _run_niche(retriever: HybridRetriever, adaptive: bool) -> dict[str, Any]:
    with open(_NICHE_FIXTURE, encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]

    recalls = {k: [] for k in _KS}
    literal_p5 = []
    maxshare50 = []
    per_query: list[dict[str, Any]] = []

    for q in queries:
        query_text = q["turns"][-1]
        gender = (q.get("expected_intent") or {}).get("gender")
        must = q["relevance"]["must"]
        universe_ids = catalogue_universe_ids(retriever.catalogue_df, must)
        if not universe_ids or len(universe_ids) > 100:
            continue

        structured_filters = {"gender": gender} if gender else {}
        colour_in = must.get("colour_in") or []
        if colour_in:
            structured_filters["colour_group_name"] = colour_in[0]
        if must.get("price_max") is not None:
            structured_filters["price_max"] = must["price_max"]

        _last_call_info.clear()
        items = retriever.search(query_text, top_k=_TOP_K, filters=structured_filters)

        retrieved_ids = [it["article_id"] for it in items]
        q_recall = {k: recall_at_k(retrieved_ids, universe_ids, k) for k in _KS}
        for k in _KS:
            recalls[k].append(q_recall[k])

        top5 = retrieved_ids[:5]
        p5 = sum(1 for a in top5 if a in universe_ids) / len(top5) if top5 else 0.0
        literal_p5.append(p5)

        _, share50 = _diversity(items, 50)
        maxshare50.append(share50)

        per_query.append(
            {
                "id": q["id"],
                "query": query_text,
                "universe_size": len(universe_ids),
                "n_stores_full_pool": _last_call_info.get("n_stores") if adaptive else None,
                "effective_cap": _last_call_info.get("effective_cap") if adaptive else _BASE_CAP,
                "recall_50": q_recall[50],
                "literal_p5": p5,
            }
        )

    return {
        "recall": {k: _mean(recalls[k]) for k in _KS},
        "literal_p5": _mean(literal_p5),
        "maxshare50": _mean(maxshare50),
        "per_query": per_query,
    }


def _run_broad(retriever: HybridRetriever, queries: list[dict]) -> dict[str, Any]:
    p5s, p10s, maxshare50 = [], [], []
    for q in queries:
        expected = q.get("expected_intent") or {}
        gender = expected.get("gender")
        filters = {"gender": gender} if gender else None

        items = retriever.search(q["turns"][-1], top_k=_TOP_K, filters=filters)

        relevance = q["relevance"]
        p5s.append(precision_at_k(items, relevance, 5))
        p10s.append(precision_at_k(items, relevance, 10))
        _, share50 = _diversity(items, 50)
        maxshare50.append(share50)

    return {"p5": _mean(p5s), "p10": _mean(p10s), "maxshare50": _mean(maxshare50)}


def main() -> None:
    config = load_config()
    catalogue_df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)
    # Nonzero so HybridRetriever.search()'s `if per_store_cap:` gate fires;
    # the actual value is irrelevant once the adaptive wrapper is patched in
    # (it recomputes and ignores this), and equals _BASE_CAP for the fixed run.
    retriever.config["retrieval"]["per_store_cap"] = _BASE_CAP

    broad_queries = _select_broad_queries(retriever)

    results: dict[str, Any] = {}

    print("=== Fixed cap=4 (baseline, shipped config) ===")
    results["fixed_niche"] = _run_niche(retriever, adaptive=False)
    results["fixed_broad"] = _run_broad(retriever, broad_queries)
    print(
        f"niche: recall@50={results['fixed_niche']['recall'][50]:.3f} "
        f"literalP@5={results['fixed_niche']['literal_p5']:.3f} "
        f"maxshare50={results['fixed_niche']['maxshare50']:.3f}"
    )
    print(
        f"broad: P@5={results['fixed_broad']['p5']:.3f} P@10={results['fixed_broad']['p10']:.3f} "
        f"maxshare50={results['fixed_broad']['maxshare50']:.3f}"
    )

    print("\n=== Adaptive cap (query-aware) ===")
    hybrid_search_module.apply_per_store_cap = _adaptive_apply_per_store_cap
    try:
        results["adaptive_niche"] = _run_niche(retriever, adaptive=True)
        results["adaptive_broad"] = _run_broad(retriever, broad_queries)
    finally:
        hybrid_search_module.apply_per_store_cap = _REAL_APPLY_PER_STORE_CAP

    print(
        f"niche: recall@50={results['adaptive_niche']['recall'][50]:.3f} "
        f"literalP@5={results['adaptive_niche']['literal_p5']:.3f} "
        f"maxshare50={results['adaptive_niche']['maxshare50']:.3f}"
    )
    print(
        f"broad: P@5={results['adaptive_broad']['p5']:.3f} P@10={results['adaptive_broad']['p10']:.3f} "
        f"maxshare50={results['adaptive_broad']['maxshare50']:.3f}"
    )

    maroon = next(
        (pq for pq in results["adaptive_niche"]["per_query"] if "dupatta" in pq["query"]), None
    )
    maroon_fixed = next(
        (pq for pq in results["fixed_niche"]["per_query"] if "dupatta" in pq["query"]), None
    )
    print("\n=== maroon dupatta (recall_009, diagnosed worst case) ===")
    if maroon and maroon_fixed:
        print(
            f"fixed cap=4:    recall@50={maroon_fixed['recall_50']:.3f}\n"
            f"adaptive cap={maroon['effective_cap']} (n_stores={maroon['n_stores_full_pool']}): "
            f"recall@50={maroon['recall_50']:.3f}"
        )

    import json
    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _ROOT / "reports" / f"adaptive_per_store_cap_{ts}.json"
    out_path.write_text(
        json.dumps({"broad_queries": [q["turns"][-1] for q in broad_queries], **results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nRaw results written: {out_path}")


if __name__ == "__main__":
    main()
