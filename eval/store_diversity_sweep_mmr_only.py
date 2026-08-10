#!/usr/bin/env python
"""store_diversity (MMR-only) sweep: recall, literal-P@5, store-diversity,
latency at store_diversity in {0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0}, with
per_store_cap forced OFF (0) throughout, on the 28-query bounded-universe
subset.

Purpose: test whether pure MMR re-ranking (store_diversity_rerank alone) can
match or beat the current two-mechanism combo (store_diversity=0.20 held
fixed + hard per_store_cap=4) on the same (recall@50, maxshare50, literal
P@5) axes — see eval/per_store_cap_sweep.py for the cap-only sweep this
mirrors, and reports/per_store_cap_sweep_20260806.md for the baseline numbers
this is compared against.

Calls the REAL HybridRetriever.search() code path unmodified for each
store_diversity value (via config["retrieval"]["store_diversity"]) — no
reimplementation, so this can't silently drift from what store_diversity_rerank
actually does. per_store_cap is forced to 0 (off) for every row in this sweep
so the only re-ranking mechanism in play is MMR itself.

Metrics per store_diversity value (means over the 28 queries):
  - recall@5/10/20/50 (structured filter: gender+colour+price, matches the
    pushdown fix)
  - literal P@5: fraction of the top-5 returned items that satisfy the
    query's exact filter (catalogue_universe_ids membership) — hand-label-free
    by construction for this literal-query subset (see recall_subset_queries.yaml
    header for why "satisfies filter" == "relevant" only holds for this class).
    This is a PROXY ONLY — see eval/README.md's standing rule: this exact proxy
    previously read a real -0.8pp strict-P@5 regression (per_store_cap 4->8) as
    "flat" (~0.721->0.714). A promising proxy result here does NOT by itself
    prove shippability; it only justifies running the real hand-labeled strict
    eval next.
  - store diversity: mean distinct-store-count in top-10/top-50, and mean
    max-single-store-share in top-50 (concentration)
  - latency: mean wall-clock ms per full retriever.search() call

Usage:
    python -m eval.store_diversity_sweep_mmr_only
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
_DIVERSITY_VALUES = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0)
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

    # per_store_cap forced OFF for every row in this sweep — isolates MMR alone.
    retriever.config["retrieval"]["per_store_cap"] = 0

    with open(_ROOT / "eval" / "fixtures" / "recall_subset_queries.yaml", encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]

    print(f"{'div':>6} {'recall@5':>9} {'recall@10':>10} {'recall@20':>10} {'recall@50':>10} "
          f"{'literalP@5':>11} {'distinct@10':>12} {'distinct@50':>12} {'maxshare@50':>12} {'ms/call':>9}")

    results = []
    for store_diversity in _DIVERSITY_VALUES:
        retriever.config["retrieval"]["store_diversity"] = store_diversity

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

        div_label = f"{store_diversity:.2f}"
        row = {
            "store_diversity": div_label,
            "recall": {k: _mean(recalls[k]) for k in _KS},
            "literal_p5": _mean(literal_p5),
            "distinct10": _mean(distinct10),
            "distinct50": _mean(distinct50),
            "maxshare50": _mean(maxshare50),
            "ms_per_call": _mean(latencies) * 1000,
        }
        results.append(row)
        print(
            f"{div_label:>6} {row['recall'][5]:>9.3f} {row['recall'][10]:>10.3f} "
            f"{row['recall'][20]:>10.3f} {row['recall'][50]:>10.3f} {row['literal_p5']:>11.3f} "
            f"{row['distinct10']:>12.2f} {row['distinct50']:>12.2f} {row['maxshare50']:>12.2f} "
            f"{row['ms_per_call']:>9.1f}"
        )

    import json
    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _ROOT / "reports" / f"store_diversity_mmr_only_sweep_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
