#!/usr/bin/env python
"""per_store_cap sweep: recall, literal-P@5, store-diversity, latency at
cap in {4 (current), 8, 16, off}, on the 28-query bounded-universe subset.

Calls the REAL HybridRetriever.search() code path unmodified for each cap
value (via config["retrieval"]["per_store_cap"]) — no reimplementation, so
this can't silently drift from what apply_per_store_cap/store_diversity_rerank
actually do. store_diversity (the MMR knob, separate from the hard cap) is
held fixed at its current config value throughout — this sweep isolates the
cap only.

Metrics per cap value (means over the 28 queries):
  - recall@5/10/20/50 (structured filter: gender+colour+price, matches the
    pushdown fix)
  - literal P@5: fraction of the top-5 returned items that satisfy the
    query's exact filter (catalogue_universe_ids membership) — hand-label-free
    by construction for this literal-query subset (see recall_subset_queries.yaml
    header for why "satisfies filter" == "relevant" only holds for this class).
  - store diversity: mean distinct-store-count in top-10/top-50, and mean
    max-single-store-share in top-50 (concentration — the thing the cap exists
    to bound)
  - latency: mean wall-clock ms per full retriever.search() call

Usage:
    python -m eval.per_store_cap_sweep
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
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

_DATA_DIR = _ROOT / "data" / "processed" / "unified"
_CAP_VALUES = (4, 8, 16, 0)  # 0 == "off" (apply_per_store_cap no-ops on cap<=0)
_KS = (5, 10, 20, 50)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _diversity(items: list[dict], k: int) -> tuple[int, float]:
    stores = [it.get("store") for it in items[:k] if it.get("store")]
    if not stores:
        return 0, 0.0
    counts = Counter(stores)
    n_distinct = len(counts)
    max_share = max(counts.values()) / len(stores)
    return n_distinct, max_share


def main() -> None:
    config = load_config()
    catalogue_df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)

    with open(_ROOT / "eval" / "fixtures" / "recall_subset_queries.yaml", encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]

    print(f"{'cap':>5} {'recall@5':>9} {'recall@10':>10} {'recall@20':>10} {'recall@50':>10} "
          f"{'literalP@5':>11} {'distinct@10':>12} {'distinct@50':>12} {'maxshare@50':>12} {'ms/call':>9}")

    results = []
    for cap in _CAP_VALUES:
        retriever.config["retrieval"]["per_store_cap"] = cap

        recalls = {k: [] for k in _KS}
        literal_p5 = []
        distinct10 = []
        distinct50 = []
        maxshare50 = []
        latencies = []

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

            t0 = time.perf_counter()
            items = retriever.search(query_text, top_k=max(_KS), filters=structured_filters)
            latencies.append(time.perf_counter() - t0)

            retrieved_ids = [it["article_id"] for it in items]
            for k in _KS:
                recalls[k].append(recall_at_k(retrieved_ids, universe_ids, k))

            top5 = retrieved_ids[:5]
            literal_p5.append(
                sum(1 for a in top5 if a in universe_ids) / len(top5) if top5 else 0.0
            )

            d10, _ = _diversity(items, 10)
            d50, share50 = _diversity(items, 50)
            distinct10.append(d10)
            distinct50.append(d50)
            maxshare50.append(share50)

        cap_label = "off" if cap == 0 else str(cap)
        row = {
            "cap": cap_label,
            "recall": {k: _mean(recalls[k]) for k in _KS},
            "literal_p5": _mean(literal_p5),
            "distinct10": _mean(distinct10),
            "distinct50": _mean(distinct50),
            "maxshare50": _mean(maxshare50),
            "ms_per_call": _mean(latencies) * 1000,
        }
        results.append(row)
        print(
            f"{cap_label:>5} {row['recall'][5]:>9.3f} {row['recall'][10]:>10.3f} "
            f"{row['recall'][20]:>10.3f} {row['recall'][50]:>10.3f} {row['literal_p5']:>11.3f} "
            f"{row['distinct10']:>12.2f} {row['distinct50']:>12.2f} {row['maxshare50']:>12.2f} "
            f"{row['ms_per_call']:>9.1f}"
        )

    import json
    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _ROOT / "reports" / f"per_store_cap_sweep_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
