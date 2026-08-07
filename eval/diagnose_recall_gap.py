#!/usr/bin/env python
"""Stage-by-stage recall-gap diagnostic for the worst recall_subset queries.

Report-only (no fixes). Mirrors HybridRetriever.search()'s internal stages
exactly (dense fetch -> sparse fetch -> RRF fusion -> gender/facet/price
filter -> dedup -> store-diversity rerank/cap -> final top_k) so we can see,
for each stage, how many of the query's TRUE universe items (from
catalogue_universe_ids, ground truth) are still present. Uses the real
production module's own helper functions (dedup_candidates_keep_cheapest,
store_diversity_rerank, apply_per_store_cap, _facet_value_matches,
get_inactive_stores) rather than reimplementing the logic, so this cannot
silently drift from what search() actually does.

Usage:
    python -m eval.diagnose_recall_gap
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS_DIR = _ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_model import catalogue_universe_ids  # noqa: E402

from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import (  # noqa: E402
    HybridRetriever,  # noqa: E402
    _facet_value_matches,
    apply_per_store_cap,
    dedup_candidates_keep_cheapest,
    get_inactive_stores,
    store_diversity_rerank,
)
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed" / "unified"

# The 5 worst recall_subset queries (from reports/recall_subset_20260806T130728Z.json).
_TARGET_IDS = {"recall_019", "recall_028", "recall_024", "recall_020", "recall_009"}


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else f"{n}/0"


def trace_query(retriever: HybridRetriever, q: dict) -> dict:
    query_text = q["turns"][-1]
    gender = (q.get("expected_intent") or {}).get("gender")
    must = q["relevance"]["must"]

    universe_ids = catalogue_universe_ids(retriever.catalogue_df, must)
    U = len(universe_ids)

    fetch_k = retriever.config["retrieval"]["top_k"]  # 50 -> fetch window = 100
    rrf_k = retriever.config["retrieval"]["rrf_k"]

    # --- Stage 1: raw dense / sparse candidate generation (fetch_k*2 each) ---
    dense_hits = retriever.dense.search(query_text, top_k=fetch_k * 2)
    sparse_hits = retriever.sparse.search(query_text, top_k=fetch_k * 2, allowed_ids=None)
    dense_ids = {a for a, _ in dense_hits}
    sparse_ids = {a for a, _ in sparse_hits}
    in_dense = len(dense_ids & universe_ids)
    in_sparse = len(sparse_ids & universe_ids)

    # --- Stage 2: RRF fusion pool (union of the two raw windows, BEFORE any filter) ---
    rrf_scores: dict[str, float] = {}
    for rank, (aid, _) in enumerate(dense_hits, start=1):
        rrf_scores[aid] = rrf_scores.get(aid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (aid, _) in enumerate(sparse_hits, start=1):
        rrf_scores[aid] = rrf_scores.get(aid, 0.0) + 1.0 / (rrf_k + rank)
    fusion_ids = set(rrf_scores.keys())
    in_fusion = len(fusion_ids & universe_ids)

    # --- Stage 3: gender + colour + price filter applied to the fusion pool ---
    colour_val = (must.get("colour_in") or [None])[0]
    price_max = must.get("price_max")
    inactive_stores = get_inactive_stores()
    filtered: list[dict] = []
    for article_id, score in rrf_scores.items():
        if article_id not in retriever.catalogue_df.index:
            continue
        row = retriever.catalogue_df.loc[article_id]
        facets = row["facets"] if isinstance(row["facets"], dict) else {}
        store_raw = str(row["store"]).lower() if "store" in row.index and row["store"] is not None else ""
        if store_raw in inactive_stores:
            continue
        item_gender = str(row["gender"]).lower() if "gender" in row.index and row["gender"] is not None else "unknown"
        if gender and (item_gender not in ("men", "women") or item_gender != gender.lower()):
            continue
        if colour_val and not _facet_value_matches(facets, "colour_group_name", colour_val):
            continue
        if price_max is not None:
            price = row.get("price_inr")
            if price is None or pd.isna(price) or float(price) > float(price_max):
                continue
        filtered.append({
            "article_id": article_id,
            "prod_name": row.get("prod_name", ""),
            "store": store_raw,
            "colour": facets.get("colour_group_name", ""),
            "price_inr": row.get("price_inr"),
            "score": score,
        })
    filtered.sort(key=lambda x: x["score"], reverse=True)
    filtered_ids = {c["article_id"] for c in filtered}
    in_filtered = len(filtered_ids & universe_ids)

    # --- Stage 4: dedup ---
    deduped = dedup_candidates_keep_cheapest(filtered)
    deduped_ids = {c["article_id"] for c in deduped}
    in_deduped = len(deduped_ids & universe_ids)

    # --- Stage 5: store-diversity rerank + per-store cap, truncated to top_k=50 ---
    store_diversity = retriever.config["retrieval"].get("store_diversity", 0.0)
    top_k_final = 50
    reranked = store_diversity_rerank(deduped, top_k_final, store_diversity)
    per_store_cap = retriever.config["retrieval"].get("per_store_cap", 0)
    if per_store_cap:
        reranked = apply_per_store_cap(reranked, deduped, per_store_cap, top_k_final)
    final_ids = {c["article_id"] for c in reranked}
    in_final = len(final_ids & universe_ids)

    return {
        "id": q["id"],
        "query": q["query"],
        "universe_size": U,
        "stage_dense_window": (in_dense, fetch_k * 2),
        "stage_sparse_window": (in_sparse, fetch_k * 2),
        "stage_fusion_pool": (in_fusion, len(fusion_ids)),
        "stage_post_filter": (in_filtered, len(filtered_ids)),
        "stage_post_dedup": (in_deduped, len(deduped_ids)),
        "stage_final_top50": (in_final, len(final_ids)),
    }


def main() -> None:
    config = load_config()
    catalogue_df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)

    with open(_ROOT / "eval" / "fixtures" / "recall_subset_queries.yaml", encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]
    targets = [q for q in queries if q["id"] in _TARGET_IDS]

    print(f"{'query':<40} {'universe':>8} {'dense win':>12} {'sparse win':>12} "
          f"{'fusion pool':>14} {'post-filter':>14} {'post-dedup':>13} {'final top50':>13}")
    rows = []
    for q in targets:
        r = trace_query(retriever, q)
        rows.append(r)
        print(
            f"{r['id']:<40} {r['universe_size']:>8} "
            f"{_pct(*r['stage_dense_window']):>12} {_pct(*r['stage_sparse_window']):>12} "
            f"{_pct(r['stage_fusion_pool'][0], r['universe_size']):>14} "
            f"{_pct(r['stage_post_filter'][0], r['universe_size']):>14} "
            f"{_pct(r['stage_post_dedup'][0], r['universe_size']):>13} "
            f"{_pct(r['stage_final_top50'][0], r['universe_size']):>13}"
        )

    import json
    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _ROOT / "reports" / f"recall_gap_diagnosis_{ts}.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
