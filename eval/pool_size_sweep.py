#!/usr/bin/env python
"""Pool-size sweep: does simply widening the fetch window fix recall, without
pushing colour/price filters into retrieval (current architecture, gender-only
pushed — matches recall_subset.py's "gender-only" pass)?

Sweeps the raw dense/sparse fetch window (currently fixed at
config["retrieval"]["top_k"] * 2 = 100) across 100/200/500/1000 for all 28
bounded-universe queries, reporting mean recall@50 and per-query latency at
each size. Run BEFORE eval/diagnose_pushdown_mechanism.py's filter-pushdown
fix is applied, as the comparison point for "is 100 too small" (mechanism a)
vs "filters applied post-fetch" (mechanism b).

Usage:
    python -m eval.pool_size_sweep
"""
from __future__ import annotations

import sys
import time
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
from src.retrieval.hybrid_search import (  # noqa: E402
    _RELEVANCE_FLOOR,
    HybridRetriever,
)
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed" / "unified"
_WINDOW_SIZES = (100, 200, 500, 1000)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    config = load_config()
    catalogue_df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)
    indexed_df = retriever.catalogue_df

    with open(_ROOT / "eval" / "fixtures" / "recall_subset_queries.yaml", encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]

    print(f"{'window':>8} {'mean recall@50':>16} {'mean dense ms/call':>20} {'mean sparse ms/call':>21}")
    for window in _WINDOW_SIZES:
        recalls = []
        dense_times = []
        sparse_times = []
        for q in queries:
            query_text = q["turns"][-1]
            gender = (q.get("expected_intent") or {}).get("gender")
            must = q["relevance"]["must"]
            universe_ids = catalogue_universe_ids(indexed_df, must)
            if not universe_ids or len(universe_ids) > 100:
                continue  # same skip rule as recall_subset.py

            gender_allowed = (
                catalogue_universe_ids(indexed_df, {"gender_in": [gender]}) if gender else None
            )

            t0 = time.perf_counter()
            dense_hits = dense.search(query_text, top_k=window)
            dense_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            sparse_hits = sparse.search(
                query_text, top_k=window,
                allowed_ids=None,  # current architecture: BM25 type-filter only, no gender push here
            )
            sparse_times.append(time.perf_counter() - t0)

            # RRF-fuse and rank-sort BEFORE truncating — a plain set union (as an
            # earlier version of this script did) loses rank order entirely, which
            # made recall look like it WORSENED as the window widened (an artifact
            # of Python set iteration order, not a real retrieval effect). Mirror
            # HybridRetriever.search's actual fusion exactly.
            rrf_k = config["retrieval"]["rrf_k"]
            rrf_scores: dict[str, float] = {}
            for rank, (aid, _) in enumerate(dense_hits, start=1):
                rrf_scores[aid] = rrf_scores.get(aid, 0.0) + 1.0 / (rrf_k + rank)
            for rank, (aid, _) in enumerate(sparse_hits, start=1):
                rrf_scores[aid] = rrf_scores.get(aid, 0.0) + 1.0 / (rrf_k + rank)
            ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            retrieved_ids = []
            for aid, score in ranked:
                if score < _RELEVANCE_FLOOR:
                    continue
                if gender_allowed is not None and aid not in gender_allowed:
                    continue
                retrieved_ids.append(aid)
            recalls.append(recall_at_k(retrieved_ids, universe_ids, 50))

        print(
            f"{window:>8} {_mean(recalls):>16.3f} "
            f"{_mean(dense_times)*1000:>20.2f} {_mean(sparse_times)*1000:>21.2f}"
            f"   (n={len(recalls)} queries)"
        )


if __name__ == "__main__":
    main()
