#!/usr/bin/env python
"""Generalizes eval/diagnose_store_score_bias.py's 3-query anecdote to the full
28-query eval/fixtures/recall_subset_queries.yaml fixture.

eval/diagnose_store_score_bias.py asked, for 3 hand-picked single-store-
dominated queries: within a query's TRUE UNIVERSE (every item that already
equally satisfies gender/colour/price/type -- so relevance is controlled for
by construction), does the dominant store's share of the TOP-RANKED HALF of
the universe exceed its share of the universe overall? If yes, that store's
items are being OUT-SCORED into the top ranks for reasons beyond genuine
relevance (a correctable ranking artifact); if the two shares match, the
store simply carries more genuinely-matching inventory (nothing to correct).

That script's conclusion ("not broadly supported") was explicitly flagged as
n=3, too small to trust. This script reruns the identical per-item scoring
method (raw dense cosine + raw BM25, computed EXHAUSTIVELY for every universe
item -- not via a truncated top-k fetch, so fetch-window recall loss can
never masquerade as ranking bias here) across all 28 queries in the fixture,
for every query whose universe has >=2 distinct stores (a single-store
universe has no bias to measure by construction).

Three score types are compared:
  - dense: raw cosine similarity (sentence-transformer encoder)
  - bm25: raw BM25 score (rank-bm25 library)
  - rrf_proxy: within-universe rank fusion, 1/(rrf_k+dense_rank) +
    1/(rrf_k+bm25_rank), rrf_k taken from config.yaml. NOTE: this is a proxy,
    not the literal production RRF score -- production RRF ranks are computed
    over a truncated fetch_k*2 window against the FULL catalogue, which would
    silently drop any universe item that doesn't survive that window (a
    fetch-recall effect, already covered by reports/pushdown_fix_20260806.md
    and per_store_cap_sweep.py) and make this bias measurement incomplete for
    exactly the queries where fetch coverage is worst. Restricting the rank
    fusion to the KNOWN universe (guaranteed 100% coverage, same as the dense/
    bm25 raw scores) isolates the ranking-bias question from the fetch-window
    question on purpose.

Usage:
    python -m eval.store_score_bias_full
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
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

from eval_model import catalogue_universe_ids  # noqa: E402

from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed" / "unified"
_FIXTURE_PATH = _ROOT / "eval" / "fixtures" / "recall_subset_queries.yaml"

# +/-5pp is the same threshold the original diagnose_store_score_bias.py used
# to call a delta "bias" rather than noise -- kept identical here so the two
# reports are directly comparable.
_BIAS_THRESHOLD: float = 0.05


def compute_query_bias(
    query_text: str,
    universe_ids: set[str],
    indexed_df: pd.DataFrame,
    dense: DenseRetriever,
    sparse: SparseRetriever,
    rrf_k: int,
) -> dict[str, Any] | None:
    """Compute per-store dominant-store bias for one query's true universe.

    Returns None if the universe has fewer than 2 distinct stores (nothing to
    compare) or fewer than 4 items (top-half would be 0-1 items, too noisy to
    read a share off of). Otherwise returns a dict with the dominant store,
    its overall share of the universe, and its top-half share under each of
    the three score types (dense / bm25 / rrf_proxy), plus the resulting bias
    (top_half_share - overall_share) and a verdict per score type.
    """
    query_vec = dense.model.encode(
        [query_text], normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")
    id_to_pos = {aid: i for i, aid in enumerate(dense.article_ids)}
    universe_positions = [id_to_pos[a] for a in universe_ids if a in id_to_pos]

    dense_scores: dict[str, float] = {}
    for pos in universe_positions:
        vec = dense.index.reconstruct(int(pos)).reshape(1, -1)
        dense_scores[str(dense.article_ids[pos])] = float((query_vec @ vec.T)[0, 0])

    bm25_tokens = sparse._tokenize(query_text)
    bm25_all_scores = sparse.bm25.get_scores(bm25_tokens)
    bm25_scores: dict[str, float] = {
        aid: float(bm25_all_scores[sparse._id_to_pos[aid]])
        for aid in universe_ids
        if aid in sparse._id_to_pos
    }

    rows = []
    for aid in universe_ids:
        if aid not in indexed_df.index or aid not in dense_scores:
            continue
        store = str(indexed_df.loc[aid, "store"]) if "store" in indexed_df.columns else None
        rows.append(
            {
                "article_id": aid,
                "store": store,
                "dense_score": dense_scores[aid],
                "bm25_score": bm25_scores.get(aid, 0.0),
            }
        )
    if len(rows) < 4:
        return None

    df = pd.DataFrame(rows)
    store_counts = df["store"].value_counts()
    if len(store_counts) < 2:
        return None  # single-store universe: no comparison possible

    dominant_store = store_counts.idxmax()
    overall_share = float(store_counts.max() / len(df))

    # Rank fusion for the rrf_proxy score type, within-universe (see module docstring).
    df = df.sort_values("dense_score", ascending=False).reset_index(drop=True)
    df["dense_rank"] = df.index + 1
    df = df.sort_values("bm25_score", ascending=False).reset_index(drop=True)
    df["bm25_rank"] = df.index + 1
    df["rrf_proxy_score"] = 1.0 / (rrf_k + df["dense_rank"]) + 1.0 / (rrf_k + df["bm25_rank"])

    result: dict[str, Any] = {
        "dominant_store": dominant_store,
        "universe_size": len(df),
        "n_distinct_stores": len(store_counts),
        "overall_share": overall_share,
        "by_score_type": {},
    }

    for score_col, label in (
        ("dense_score", "dense"),
        ("bm25_score", "bm25"),
        ("rrf_proxy_score", "rrf_proxy"),
    ):
        ordered = df.sort_values(score_col, ascending=False)
        top_half_n = max(1, len(ordered) // 2)
        top_half = ordered.iloc[:top_half_n]
        top_half_share = float((top_half["store"] == dominant_store).mean())
        bias = top_half_share - overall_share
        if bias > _BIAS_THRESHOLD:
            verdict = "out-scores"
        elif bias < -_BIAS_THRESHOLD:
            verdict = "under-scores"
        else:
            verdict = "proportional"
        result["by_score_type"][label] = {
            "top_half_share": top_half_share,
            "bias": bias,
            "verdict": verdict,
        }

    return result


def main() -> None:
    config = load_config()
    catalogue_df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)
    indexed_df = retriever.catalogue_df
    rrf_k = config["retrieval"]["rrf_k"]

    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]

    per_query: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for q in queries:
        query_text = q["turns"][-1]
        must = q["relevance"]["must"]
        universe_ids = catalogue_universe_ids(indexed_df, must)
        if not universe_ids or len(universe_ids) > 100:
            skipped.append({"id": q["id"], "reason": f"universe size {len(universe_ids)} out of [1,100]"})
            continue

        bias_result = compute_query_bias(query_text, universe_ids, indexed_df, dense, sparse, rrf_k)
        if bias_result is None:
            skipped.append({"id": q["id"], "reason": "single-store universe or <4 items"})
            continue

        bias_result["id"] = q["id"]
        bias_result["query"] = query_text
        per_query.append(bias_result)
        print(
            f"{q['id']:>12} store={bias_result['dominant_store']:<14} "
            f"universe={bias_result['universe_size']:>3} overall_share={bias_result['overall_share']:.2f}  "
            + "  ".join(
                f"{label}:{r['verdict']}({r['bias']:+.2f})"
                for label, r in bias_result["by_score_type"].items()
            )
        )

    print(f"\n{len(per_query)}/{len(queries)} queries have a comparable multi-store universe "
          f"({len(skipped)} skipped -- universe too large/small or single-store).")

    summary: dict[str, Any] = {"n_queries_total": len(queries), "n_comparable": len(per_query), "skipped": skipped}
    for label in ("dense", "bm25", "rrf_proxy"):
        biases = [r["by_score_type"][label]["bias"] for r in per_query]
        verdicts = [r["by_score_type"][label]["verdict"] for r in per_query]
        summary[label] = {
            "n_out_scores": verdicts.count("out-scores"),
            "n_proportional": verdicts.count("proportional"),
            "n_under_scores": verdicts.count("under-scores"),
            "mean_bias": sum(biases) / len(biases) if biases else 0.0,
            "max_bias": max(biases) if biases else 0.0,
            "min_bias": min(biases) if biases else 0.0,
        }
        print(
            f"\n[{label}] out-scores={summary[label]['n_out_scores']} "
            f"proportional={summary[label]['n_proportional']} "
            f"under-scores={summary[label]['n_under_scores']}  "
            f"mean_bias={summary[label]['mean_bias']:+.3f} max={summary[label]['max_bias']:+.3f}"
        )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _ROOT / "reports" / f"store_score_bias_full_{ts}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "per_query": per_query}, indent=2), encoding="utf-8"
    )
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
