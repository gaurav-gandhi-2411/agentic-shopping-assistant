#!/usr/bin/env python
"""Mechanism diagnosis: WHY do qualifying items miss the top-100 fetch window?

Distinguishes three hypotheses for the 3 named worst queries by comparing the
SAME dense/sparse retrievers under three candidate-pool configurations:

  A. unfiltered  — current architecture: dense.search/sparse.search over the
     full 112,425-item catalogue, gender/colour/price applied only AFTER fetch
     (this repo's status quo).
  B. gender-pushed — dense restricted via FAISS IDSelector, sparse restricted
     via its existing allowed_ids mask, to ONLY the matching-gender subset,
     colour/price still applied after fetch.
  C. gender+colour+price-pushed — restricted to the full qualifying subset
     (== catalogue_universe_ids for that query) before any ranking happens.

If recall jumps sharply from A->B, gender-dilution of the pool (mechanism b:
filter-applied-too-late) is doing most of the damage. If B->C also jumps
sharply, colour/price dilution matters too. If recall stays low even in C
(where the pool literally IS the universe, just semantically re-ranked), that
would point to (c) — dense/BM25 scoring genuinely burying qualifying items —
but note C's pool size is the universe size, so recall in C should be at or
near 1.0 whenever universe_size <= top_k by construction (every universe item
is IN the search space); C mainly validates that FAISS IDSelector / BM25
allowed_ids restriction is even correct, not a new mechanism test.

Usage:
    python -m eval.diagnose_pushdown_mechanism
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import faiss
import numpy as np
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
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed" / "unified"
_TARGET_IDS = {"recall_019", "recall_028", "recall_024"}  # blue jeans/men, white shirt/men, pink saree/women
_WINDOW = 100  # matches current fetch_k*2


def _dense_search_restricted(dense: DenseRetriever, query: str, top_k: int, allowed_positions: np.ndarray):
    qv = dense.model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    sel = faiss.IDSelectorArray(allowed_positions.astype(np.int64))
    params = faiss.SearchParameters(sel=sel)
    scores, idx = dense.index.search(qv, top_k, params=params)
    return [(str(dense.article_ids[i]), float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]


def main() -> None:
    config = load_config()
    catalogue_df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)
    indexed_df = retriever.catalogue_df  # indexed by article_id, aligned with dense.article_ids positions

    # Map article_id -> position in dense.article_ids (for IDSelector construction)
    id_to_pos = {aid: i for i, aid in enumerate(dense.article_ids)}

    with open(_ROOT / "eval" / "fixtures" / "recall_subset_queries.yaml", encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]
    targets = [q for q in queries if q["id"] in _TARGET_IDS]

    print(f"{'query':<16} {'universe':>8} {'A: unfiltered':>16} {'B: gender-pushed':>18} "
          f"{'C: gender+colour+price-pushed':>30}")

    for q in targets:
        query_text = q["turns"][-1]
        gender = (q.get("expected_intent") or {}).get("gender")
        must = q["relevance"]["must"]
        colour_val = (must.get("colour_in") or [None])[0]
        price_max = must.get("price_max")

        universe_ids = catalogue_universe_ids(indexed_df, must)
        U = len(universe_ids)

        # --- A: unfiltered ---
        dense_a = dense.search(query_text, top_k=_WINDOW)
        sparse_a = sparse.search(query_text, top_k=_WINDOW, allowed_ids=None)
        pool_a = {a for a, _ in dense_a} | {a for a, _ in sparse_a}
        in_a = len(pool_a & universe_ids)

        # How much of pool A is even the right gender? (quantifies dilution)
        gender_col = indexed_df["gender"].astype(str).str.lower()
        pool_a_genders = gender_col.reindex(list(pool_a)).value_counts(dropna=False)

        # --- B: gender-pushed ---
        gender_mask = catalogue_universe_ids(indexed_df, {"gender_in": [gender]} if gender else {})
        gender_positions = np.array([id_to_pos[a] for a in gender_mask if a in id_to_pos], dtype=np.int64)
        dense_b = _dense_search_restricted(dense, query_text, _WINDOW, gender_positions)
        sparse_b = sparse.search(query_text, top_k=_WINDOW, allowed_ids=np.array(list(gender_mask)))
        pool_b = {a for a, _ in dense_b} | {a for a, _ in sparse_b}
        in_b = len(pool_b & universe_ids)

        # --- C: gender+colour+price-pushed (== universe) ---
        full_mask_ids = universe_ids
        full_positions = np.array([id_to_pos[a] for a in full_mask_ids if a in id_to_pos], dtype=np.int64)
        dense_c = _dense_search_restricted(dense, query_text, _WINDOW, full_positions) if len(full_positions) else []
        sparse_c = sparse.search(query_text, top_k=_WINDOW, allowed_ids=np.array(list(full_mask_ids))) if full_mask_ids else []
        pool_c = {a for a, _ in dense_c} | {a for a, _ in sparse_c}
        in_c = len(pool_c & universe_ids)

        print(f"{q['id']:<16} {U:>8} {f'{in_a}/{U} ({in_a/U:.0%})':>16} "
              f"{f'{in_b}/{U} ({in_b/U:.0%})':>18} {f'{in_c}/{U} ({in_c/U:.0%})':>30}")
        print(f"  pool A gender breakdown: {dict(pool_a_genders)} "
              f"(target gender={gender}, colour={colour_val}, price_max={price_max})")

    # --- Cost check: does restricted dense search via IDSelector cost more? ---
    print("\nLatency check (dense.search): unfiltered vs IDSelector-restricted, 20 reps each")
    qv_text = "blue jeans for men"
    t0 = time.perf_counter()
    for _ in range(20):
        dense.search(qv_text, top_k=_WINDOW)
    t_unfiltered = (time.perf_counter() - t0) / 20
    men_ids = catalogue_universe_ids(indexed_df, {"gender_in": ["men"]})
    men_positions = np.array([id_to_pos[a] for a in men_ids if a in id_to_pos], dtype=np.int64)
    t0 = time.perf_counter()
    for _ in range(20):
        _dense_search_restricted(dense, qv_text, _WINDOW, men_positions)
    t_restricted = (time.perf_counter() - t0) / 20
    print(f"  unfiltered: {t_unfiltered*1000:.2f}ms/call   IDSelector-restricted ({len(men_positions)} ids): {t_restricted*1000:.2f}ms/call")


if __name__ == "__main__":
    main()
