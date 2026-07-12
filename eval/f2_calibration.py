"""
F2 calibration: score distribution across calibration queries.

Loads the live unified index, runs each query through hybrid_search without
top-k truncation, reports RRF score distribution (min/p10/p25/p50/p75/p90/max)
and — for 'black dress' — how many of the top-20 items are actual dresses vs
garbage (shorts/jackets/etc.).

Usage:
    python -m eval.f2_calibration
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.catalogue.loader import load_config
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.sparse_search import SparseRetriever

# ---------------------------------------------------------------------------
# Calibration queries
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "name": "black dress for women",
        "query": "black dress for women",
        "expect_good": "dress",
        "expect_bad": ["shorts", "jacket", "sweater", "top"],
    },
    {
        "name": "blue jeans for men",
        "query": "blue jeans for men",
        "expect_good": "jeans",
        "expect_bad": ["dress", "top", "blouse"],
    },
    {
        "name": "formal blazer",
        "query": "formal blazer",
        "expect_good": "blazer",
        "expect_bad": ["dress", "shorts", "jeans"],
    },
    {
        "name": "kurta for holi",
        "query": "kurta for holi",
        "expect_good": "kurta",
        "expect_bad": ["dress", "shorts", "blouse"],
    },
    {
        "name": "laptop bag (OOC — should score near 0)",
        "query": "laptop bag",
        "expect_good": "bag",  # bags may score OK; the key is no garments
        "expect_bad": ["dress", "top", "shorts"],
    },
]

SAVE_DIR = Path("data/processed/unified")
TOP_N_REPORT = 20  # number of top candidates to inspect for type breakdown


def _pct(arr: list[float], p: float) -> float:
    return float(np.percentile(arr, p)) if arr else 0.0


def main() -> None:
    print("Loading config and index...")
    config = load_config()

    catalogue_df = pd.read_parquet(SAVE_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, SAVE_DIR)
    sparse = SparseRetriever.load(config, SAVE_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)

    print(f"Catalogue loaded: {len(catalogue_df):,} items\n")
    print("=" * 70)

    for q_spec in QUERIES:
        query = q_spec["query"]
        print(f"\nQUERY: {query!r}")
        print("-" * 60)

        # --- Run with a large fetch_k to see the full score distribution ---
        original_fetch_k = config["retrieval"]["top_k"]
        config["retrieval"]["top_k"] = 200  # expand window for calibration
        all_results = retriever.search(query, top_k=200, filters=None)
        config["retrieval"]["top_k"] = original_fetch_k  # restore

        if not all_results:
            print("  No results returned.")
            continue

        scores = [r["score"] for r in all_results]
        top_n = all_results[:TOP_N_REPORT]

        # Score distribution
        print(
            f"  Candidates returned: {len(all_results)}"
            f"  |  min={min(scores):.4f}  max={max(scores):.4f}"
        )
        print(
            f"  Percentiles  p10={_pct(scores, 10):.4f}  p25={_pct(scores, 25):.4f}"
            f"  p50={_pct(scores, 50):.4f}  p75={_pct(scores, 75):.4f}"
            f"  p90={_pct(scores, 90):.4f}"
        )

        # Type breakdown in top-N
        top_types: dict[str, int] = {}
        for r in top_n:
            ptype = (r.get("product_type") or "unknown").lower().strip()
            top_types[ptype] = top_types.get(ptype, 0) + 1
        sorted_types = sorted(top_types.items(), key=lambda x: -x[1])
        print(f"  Top-{TOP_N_REPORT} product_type breakdown:")
        for ptype, cnt in sorted_types:
            marker = "[Y]" if q_spec["expect_good"] in ptype else "[X]"
            print(f"    {marker}  {ptype!r:<30} {cnt:>3} items")

        # Score of the score-20 threshold item
        if len(scores) >= 20:
            print(f"  Score at rank-20 cutoff: {scores[19]:.4f}")

        # Gap analysis: score of top good-type item vs top bad-type item
        good_scores = [
            r["score"] for r in top_n
            if q_spec["expect_good"] in (r.get("product_type") or "").lower()
        ]
        bad_scores = [
            r["score"] for r in top_n
            if any(
                b in (r.get("product_type") or "").lower()
                for b in q_spec["expect_bad"]
            )
        ]
        if good_scores and bad_scores:
            gap = min(good_scores) - max(bad_scores)
            print(
                f"  Gap: best-good={max(good_scores):.4f}  worst-good={min(good_scores):.4f}"
                f"  |  best-bad={max(bad_scores):.4f}  worst-bad={min(bad_scores):.4f}"
            )
            print(f"  -> min(good) - max(bad) = {gap:+.4f}  {'CLEAN GAP [Y]' if gap > 0 else 'OVERLAP [X]'}")
        elif not good_scores:
            print("  [!] No good-type items in top-20 -- type filter may be needed")
        else:
            print("  [Y] No bad-type items in top-20")

        # Show top-5 items by score
        print("  Top-5 results:")
        for i, r in enumerate(all_results[:5], 1):
            print(
                f"    {i}. [{r['score']:.4f}] {r.get('product_type', 'N/A')!r:<20}"
                f" | {r.get('display_name') or r.get('prod_name', '')[:50]}"
            )

    print("\n" + "=" * 70)
    print("F2 CALIBRATION COMPLETE")
    print()
    print("Next step: propose relevance-floor threshold based on where the")
    print("clean gap falls across all queries (the valley between good and bad).")


if __name__ == "__main__":
    main()
