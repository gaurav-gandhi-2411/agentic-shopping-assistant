#!/usr/bin/env python
"""Does per_store_cap have to trade recall against precision because one
store's ITEMS are genuinely better matches, or because its TITLES score
higher for reasons unrelated to relevance (keyword-stuffing / title-length
artifact)? Tests the score-normalization hypothesis: within a set of items
that are ALL equally-qualifying (same gender/colour/price/type — i.e. all in
the query's true universe, so relevance is already controlled for), does one
store's raw dense/BM25/RRF score distribution sit systematically higher than
another's? If yes, per-store score normalization could recover recall
without the diversity cost. If no (scores are comparable, the store just has
more genuinely-matching inventory), normalization would not help and the
real lever stays per_store_cap/dedup.

Usage:
    python -m eval.diagnose_store_score_bias
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS_DIR = _ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_model import catalogue_universe_ids  # noqa: E402
from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed" / "unified"

# The 2 store-dominated queries already diagnosed (dupatta/myntra 95%,
# jeans/flipkart 87%) plus 2 more from the recall_subset fixture to check
# whether this is a general pattern or specific to those two.
_CASES = [
    {"id": "recall_009", "query": "maroon dupatta for women under 6000 rupees", "gender": "women",
     "must": {"product_type_contains": ["dupatta"], "gender_in": ["women"], "colour_in": ["maroon"], "price_max": 6000},
     "dominant_store": "myntra"},
    {"id": "recall_019", "query": "blue jeans for men under 1000 rupees", "gender": "men",
     "must": {"product_type_contains": ["jeans"], "gender_in": ["men"], "colour_in": ["blue"], "price_max": 1000},
     "dominant_store": "flipkart"},
    {"id": "recall_011", "query": "white sneakers for women under 7000 rupees", "gender": "women",
     "must": {"product_type_contains": ["sneaker"], "gender_in": ["women"], "colour_in": ["white"], "price_max": 7000},
     "dominant_store": "campusshoes"},
]


def main() -> None:
    config = load_config()
    catalogue_df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)
    indexed_df = retriever.catalogue_df

    for case in _CASES:
        print(f"\n{'=' * 70}\n{case['id']}: {case['query']!r} (dominant store: {case['dominant_store']})")
        universe_ids = catalogue_universe_ids(indexed_df, case["must"])
        print(f"universe size: {len(universe_ids)}")

        # Raw dense + sparse scores for EVERY universe item (not just top-k) --
        # compute directly via cosine similarity / BM25 score against the
        # query, restricted to exactly the universe (so every item compared is
        # already equally-qualifying: same gender/colour/price/type).
        query_vec = dense.model.encode([case["query"]], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        id_to_pos = {aid: i for i, aid in enumerate(dense.article_ids)}
        universe_positions = [id_to_pos[a] for a in universe_ids if a in id_to_pos]
        # Reconstruct just the universe's embeddings (bounded size, cheap) and score directly.
        dense_scores = {}
        for pos in universe_positions:
            vec = dense.index.reconstruct(int(pos)).reshape(1, -1)
            score = float((query_vec @ vec.T)[0, 0])
            dense_scores[str(dense.article_ids[pos])] = score

        bm25_tokens = sparse._tokenize(case["query"])
        bm25_all_scores = sparse.bm25.get_scores(bm25_tokens)
        bm25_scores = {}
        for aid in universe_ids:
            if aid in sparse._id_to_pos:
                bm25_scores[aid] = float(bm25_all_scores[sparse._id_to_pos[aid]])

        rows = []
        for aid in universe_ids:
            store = str(indexed_df.loc[aid, "store"]) if aid in indexed_df.index and "store" in indexed_df.columns else None
            rows.append({
                "article_id": aid,
                "store": store,
                "is_dominant_store": store == case["dominant_store"],
                "dense_score": dense_scores.get(aid),
                "bm25_score": bm25_scores.get(aid, 0.0),
                "title_len_words": len(str(indexed_df.loc[aid, "prod_name"]).split()) if aid in indexed_df.index else None,
            })
        df = pd.DataFrame(rows).dropna(subset=["dense_score"])

        dom = df[df["is_dominant_store"]]
        other = df[~df["is_dominant_store"]]
        print(f"  n_dominant={len(dom)}  n_other={len(other)}")
        if len(dom) and len(other):
            print(f"  dense score  — dominant mean={dom['dense_score'].mean():.4f}  other mean={other['dense_score'].mean():.4f}  "
                  f"(dominant median={dom['dense_score'].median():.4f}, other median={other['dense_score'].median():.4f})")
            print(f"  bm25 score   — dominant mean={dom['bm25_score'].mean():.4f}  other mean={other['bm25_score'].mean():.4f}")
            print(f"  title length — dominant mean={dom['title_len_words'].mean():.1f} words  other mean={other['title_len_words'].mean():.1f} words")
            # Rank-based comparison: of the FULL universe sorted by dense score
            # descending, what fraction of the TOP HALF is from the dominant store,
            # vs the dominant store's overall share of the universe? If the top
            # half over-represents the dominant store relative to its overall
            # share, that store is systematically OUT-SCORING others, not just
            # out-numbering them.
            df_sorted = df.sort_values("dense_score", ascending=False)
            top_half = df_sorted.iloc[: len(df_sorted) // 2]
            overall_share = dom.shape[0] / len(df)
            top_half_share = (top_half["is_dominant_store"]).mean()
            print(f"  dominant store's share of universe: {overall_share:.2f}  |  share of TOP-HALF by dense score: {top_half_share:.2f}")
            verdict = "OUT-SCORES (ranking bias)" if top_half_share > overall_share + 0.05 else "PROPORTIONAL (no ranking bias, just more inventory)"
            print(f"  -> {verdict}")
        else:
            print("  insufficient data for comparison (all items from one store)")


if __name__ == "__main__":
    main()
