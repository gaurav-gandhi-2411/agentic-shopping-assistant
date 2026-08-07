#!/usr/bin/env python
"""per_store_cap sweep on BROAD (large-universe) queries — the population the
cap was actually designed for (config.yaml's comment: "prevent one store
dominating the candidate pool handed to the LLM reranker"), as a check on
whether a recommendation drawn from eval/per_store_cap_sweep.py's niche
28-query subset (deliberately tight-filter, often single-store-dominated by
construction) generalizes. Recall isn't computable here (universe > 500 for
every query, no bounded ground truth) — reports property-floor P@5
(precision_at_k, same metric as eval_model.py's Stage 2 R1) and the same
store-diversity metrics instead.

Usage:
    python -m eval.per_store_cap_sweep_broad
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

from eval_model import catalogue_universe_ids, precision_at_k  # noqa: E402

from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed" / "unified"
_CAP_VALUES = (4, 8, 16, 0)
_FIXTURE_PATH = _ROOT / "eval" / "fixtures" / "model_eval_queries.yaml"


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _diversity(items: list[dict], k: int) -> tuple[int, float]:
    stores = [it.get("store") for it in items[:k] if it.get("store")]
    if not stores:
        return 0, 0.0
    counts = Counter(stores)
    return len(counts), max(counts.values()) / len(stores)


def _select_broad_queries(retriever: HybridRetriever) -> list[dict]:
    data = yaml.safe_load(_FIXTURE_PATH.read_text(encoding="utf-8"))
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
            continue  # skip literal duplicates (t-shirt/tee/tshirt style)
        must = relevance.get("must") or {}
        universe = catalogue_universe_ids(retriever.catalogue_df, must)
        if len(universe) <= 500:
            continue  # this is the broad (not niche-bounded) population
        seen_text.add(text.lower())
        out.append(q)
        if len(out) >= 20:
            break
    return out


def main() -> None:
    config = load_config()
    catalogue_df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)

    queries = _select_broad_queries(retriever)
    print(f"Sampled {len(queries)} broad (universe > 500) queries: "
          f"{[q['turns'][-1] for q in queries]}\n")

    print(f"{'cap':>5} {'P@5':>7} {'P@10':>7} {'distinct@10':>12} {'distinct@50':>12} "
          f"{'maxshare@50':>12} {'ms/call':>9}")

    results = []
    for cap in _CAP_VALUES:
        retriever.config["retrieval"]["per_store_cap"] = cap
        p5s, p10s, distinct10, distinct50, maxshare50, latencies = [], [], [], [], [], []

        for q in queries:
            expected = q.get("expected_intent") or {}
            gender = expected.get("gender")
            filters = {"gender": gender} if gender else None

            t0 = time.perf_counter()
            items = retriever.search(q["turns"][-1], top_k=50, filters=filters)
            latencies.append(time.perf_counter() - t0)

            relevance = q["relevance"]
            p5s.append(precision_at_k(items, relevance, 5))
            p10s.append(precision_at_k(items, relevance, 10))
            d10, _ = _diversity(items, 10)
            d50, share50 = _diversity(items, 50)
            distinct10.append(d10)
            distinct50.append(d50)
            maxshare50.append(share50)

        cap_label = "off" if cap == 0 else str(cap)
        row = {
            "cap": cap_label, "p5": _mean(p5s), "p10": _mean(p10s),
            "distinct10": _mean(distinct10), "distinct50": _mean(distinct50),
            "maxshare50": _mean(maxshare50), "ms_per_call": _mean(latencies) * 1000,
        }
        results.append(row)
        print(
            f"{cap_label:>5} {row['p5']:>7.3f} {row['p10']:>7.3f} {row['distinct10']:>12.2f} "
            f"{row['distinct50']:>12.2f} {row['maxshare50']:>12.2f} {row['ms_per_call']:>9.1f}"
        )

    import json
    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _ROOT / "reports" / f"per_store_cap_sweep_broad_{ts}.json"
    out_path.write_text(json.dumps({"queries": [q["turns"][-1] for q in queries], "results": results}, indent=2), encoding="utf-8")
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
