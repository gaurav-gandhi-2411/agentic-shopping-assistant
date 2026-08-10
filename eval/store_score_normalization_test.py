#!/usr/bin/env python
"""Per-store score-normalization variant of apply_per_store_cap -- does NOT
touch src/retrieval/hybrid_search.py.

eval/store_score_bias_full.py found a real (not just anecdotal) per-store
scoring bias on a majority of the 28-query fixture's comparable queries: the
dominant store's items are, on the raw dense/BM25/RRF-proxy score, ranked
into the top half of the query's true universe MORE often than its overall
share of that universe would predict (see that script's docstring + report
for the full methodology and distribution). This module tests whether
correcting that bias -- normalizing each item's fusion ('RRF') score against
its own store's score distribution in the candidate pool, before the hard
per-store cap decides what to keep -- recovers some of the recall that
per_store_cap=4 currently costs (0.713 uncapped -> 0.438 at cap=4, see
config.yaml's per_store_cap comment and reports/pushdown_fix_20260806.md),
without materially reopening the concentration problem the cap exists to
bound.

Pipeline stages (src/retrieval/hybrid_search.py::HybridRetriever.search()):
    candidates (full RRF-ordered pool, post-dedup)
        -> store_diversity_rerank(candidates, top_k, store_diversity) -> selected
        -> apply_per_store_cap(selected, full_pool=candidates, cap, top_k)   <-- intercepted here

`search()` is one monolithic method with no stage-decomposition API, so
rather than reimplementing the (nontrivial) fetch/filter/dedup logic that
builds `candidates`, this module monkeypatches the exact module-level
function name (`apply_per_store_cap`) that `search()` calls unqualified --
Python resolves that name from `src.retrieval.hybrid_search`'s module
globals at call time, so replacing it there for the duration of a `with`
block changes what real `HybridRetriever.search()` calls, with zero
reimplementation of anything upstream. The patched function:
  1. computes each item's per-store-normalized score over `full_pool` (the
     candidate pool, exactly as specified);
  2. reorders BOTH `selected` and `full_pool` by that normalized score
     (apply_per_store_cap's cap-keep and RRF-order backfill logic is
     order-dependent, not score-value-dependent, so reordering the inputs is
     the correct injection point);
  3. delegates to the REAL, unmodified `apply_per_store_cap` (imported
     directly from src.retrieval.hybrid_search, never copied) on the
     reordered lists.

Usage:
    python -m eval.store_score_normalization_test          # comparison sweep
    pytest eval/store_score_normalization_test.py -v        # unit tests only
"""
from __future__ import annotations

import statistics
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import pandas as pd
import yaml

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS_DIR = _ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_model import catalogue_universe_ids, recall_at_k  # noqa: E402

import src.retrieval.hybrid_search as hs  # noqa: E402
from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed" / "unified"
_FIXTURE_PATH = _ROOT / "eval" / "fixtures" / "recall_subset_queries.yaml"
_KS = (5, 10, 20, 50)

NormMethod = Literal["zscore", "minmax"]

# apply_per_store_cap's own real signature is (selected, full_pool, cap, top_k) ->
# list[dict]; kept identical here so the patched function is a drop-in replacement.
_RealApplyPerStoreCap = Callable[[list[dict], list[dict], int, int], list[dict]]


def normalize_scores_per_store(
    items: list[dict], method: NormMethod = "zscore"
) -> dict[str, float]:
    """Normalize each item's 'score' against its OWN store's score distribution
    within `items`.

    Parameters
    ----------
    items:
        Candidate dicts, each with 'article_id', 'score' (float), 'store' (str | None).
        A store's distribution is computed over every item in `items` belonging
        to that store -- i.e. normalization is local to the candidate pool
        passed in, not global to the catalogue.
    method:
        'zscore' -- (score - store_mean) / store_std. A store with a single
            item (std=0, undefined) normalizes to 0.0 (no information to
            normalize against; treated as store-average, not artificially
            boosted or penalized).
        'minmax' -- (score - store_min) / (store_max - store_min), i.e. each
            store's OWN best item always lands at 1.0. A store with a single
            item (max==min) normalizes to 0.0, same rationale as zscore.

    Returns
    -------
    dict mapping article_id -> normalized_score. Does not mutate `items`.
    """
    by_store: dict[str | None, list[float]] = defaultdict(list)
    for it in items:
        by_store[it.get("store")].append(it["score"])

    stats: dict[str | None, tuple[float, float]] = {}
    for store, scores in by_store.items():
        if method == "zscore":
            mean = statistics.fmean(scores)
            std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            stats[store] = (mean, std)
        elif method == "minmax":
            stats[store] = (min(scores), max(scores))
        else:
            raise ValueError(f"unknown method: {method!r}")

    out: dict[str, float] = {}
    for it in items:
        store = it.get("store")
        a, b = stats[store]
        if method == "zscore":
            out[it["article_id"]] = (it["score"] - a) / b if b > 0 else 0.0
        else:  # minmax
            out[it["article_id"]] = (it["score"] - a) / (b - a) if (b - a) > 0 else 0.0
    return out


def make_normalizing_apply_per_store_cap(method: NormMethod) -> _RealApplyPerStoreCap:
    """Build a drop-in replacement for hybrid_search.apply_per_store_cap that
    reorders its inputs by per-store-normalized score before delegating to the
    real function.

    The real `apply_per_store_cap` is captured by reference HERE, at factory
    -build time (i.e. before `patched_apply_per_store_cap` below overwrites
    `hs.apply_per_store_cap` with the returned closure) -- so it always
    delegates to whatever the current production implementation is, and
    cannot silently drift from src/retrieval/hybrid_search.py's actual cap/
    backfill logic. Reading `hs.apply_per_store_cap` INSIDE `patched` instead
    would capture the patched function itself once the monkeypatch is live,
    causing infinite recursion -- this ordering is deliberate, not
    incidental.
    """
    real_apply_per_store_cap: _RealApplyPerStoreCap = hs.apply_per_store_cap

    def patched(
        selected: list[dict], full_pool: list[dict], cap: int, top_k: int
    ) -> list[dict]:
        norm_by_id = normalize_scores_per_store(full_pool, method)

        norm_full_pool = sorted(
            full_pool, key=lambda it: norm_by_id.get(it["article_id"], 0.0), reverse=True
        )
        # `selected` is a subset of full_pool (store_diversity_rerank's output) --
        # every id is guaranteed present in norm_by_id.
        norm_selected = sorted(
            selected, key=lambda it: norm_by_id.get(it["article_id"], 0.0), reverse=True
        )
        return real_apply_per_store_cap(norm_selected, norm_full_pool, cap, top_k)

    return patched


@contextmanager
def patched_apply_per_store_cap(method: NormMethod) -> Iterator[None]:
    """Monkeypatch src.retrieval.hybrid_search.apply_per_store_cap for the
    duration of the `with` block, restoring the original unconditionally on
    exit (even on exception) so this can never leak into other test/eval runs
    sharing the same Python process.
    """
    original = hs.apply_per_store_cap
    hs.apply_per_store_cap = make_normalizing_apply_per_store_cap(method)
    try:
        yield
    finally:
        hs.apply_per_store_cap = original


# --------------------------------------------------------------------------
# Unit tests (pure, no model/index loading required)
# --------------------------------------------------------------------------


def test_normalize_scores_per_store_zscore_equalizes_stores() -> None:
    """A store whose raw scores are uniformly higher should not automatically
    win every normalized-score comparison once normalized within its own
    distribution."""
    items = [
        {"article_id": "a1", "store": "high_store", "score": 0.90},
        {"article_id": "a2", "store": "high_store", "score": 0.80},
        {"article_id": "a3", "store": "low_store", "score": 0.30},
        {"article_id": "a4", "store": "low_store", "score": 0.10},
    ]
    norm = normalize_scores_per_store(items, method="zscore")
    # Each store's own top item normalizes to the same positive z-score
    # (symmetric 2-item distributions), despite very different raw scores.
    # round() absorbs float-precision noise (0.9999999999999989 vs ...999) --
    # the two 2-item distributions are symmetric so the z-scores are
    # mathematically identical, not just close.
    assert round(norm["a1"], 9) == round(norm["a3"], 9) > 0
    assert round(norm["a2"], 9) == round(norm["a4"], 9) < 0


def test_normalize_scores_per_store_minmax_top_item_is_one() -> None:
    """minmax normalization always puts each store's own best item at 1.0 and
    worst at 0.0, regardless of the store's raw score scale."""
    items = [
        {"article_id": "a1", "store": "s1", "score": 5.0},
        {"article_id": "a2", "store": "s1", "score": 1.0},
        {"article_id": "a3", "store": "s2", "score": 0.05},
        {"article_id": "a4", "store": "s2", "score": 0.01},
    ]
    norm = normalize_scores_per_store(items, method="minmax")
    assert norm["a1"] == 1.0
    assert norm["a2"] == 0.0
    assert norm["a3"] == 1.0
    assert norm["a4"] == 0.0


def test_normalize_scores_per_store_single_item_store_is_neutral() -> None:
    """A store with exactly one item in the pool has no distribution to
    normalize against -- must not be silently boosted or penalized (both
    methods define this as 0.0, not an error or a division by zero)."""
    items = [
        {"article_id": "solo", "store": "lonely_store", "score": 42.0},
        {"article_id": "a2", "store": "s2", "score": 1.0},
        {"article_id": "a3", "store": "s2", "score": 2.0},
    ]
    assert normalize_scores_per_store(items, method="zscore")["solo"] == 0.0
    assert normalize_scores_per_store(items, method="minmax")["solo"] == 0.0


def test_make_normalizing_apply_per_store_cap_delegates_to_real_function() -> None:
    """The patched function must produce IDENTICAL output to the real
    apply_per_store_cap when every item already has an equal (already-
    normalized) score -- i.e. normalization must not change cap/backfill
    semantics, only the ORDER fed into them."""
    selected = [
        {"article_id": f"s{i}", "store": "storeA" if i % 2 == 0 else "storeB", "score": 1.0}
        for i in range(6)
    ]
    full_pool = selected + [
        {"article_id": "extra1", "store": "storeC", "score": 1.0},
        {"article_id": "extra2", "store": "storeC", "score": 1.0},
    ]
    patched_fn = make_normalizing_apply_per_store_cap("zscore")
    # cap=2, top_k=4: real apply_per_store_cap should cap storeA/storeB at 2
    # each from `selected` (already 3 each), then not need to backfill since
    # 4 items already satisfy top_k.
    result = patched_fn(selected, full_pool, cap=2, top_k=4)
    real_result = hs.apply_per_store_cap(selected, full_pool, cap=2, top_k=4)
    assert len(result) == len(real_result) == 4
    assert Counter(it["store"] for it in result) == Counter(it["store"] for it in real_result)


def test_patched_apply_per_store_cap_restores_original_on_exit() -> None:
    """The context manager must restore the real function even after an
    exception inside the `with` block -- a leaked monkeypatch would silently
    corrupt every subsequent retriever.search() call in the same process."""
    original = hs.apply_per_store_cap
    try:
        with patched_apply_per_store_cap("zscore"):
            assert hs.apply_per_store_cap is not original
            raise RuntimeError("simulated failure inside the patched block")
    except RuntimeError:
        pass
    assert hs.apply_per_store_cap is original


# --------------------------------------------------------------------------
# Comparison sweep: baseline (cap=4, no normalization) vs normalized variants
# --------------------------------------------------------------------------


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _diversity(items: list[dict], k: int) -> tuple[int, float]:
    stores = [it.get("store") for it in items[:k] if it.get("store")]
    if not stores:
        return 0, 0.0
    counts = Counter(stores)
    return len(counts), max(counts.values()) / len(stores)


def _run_sweep_variant(
    retriever: HybridRetriever, queries: list[dict], label: str
) -> dict[str, Any]:
    """Run the real per_store_cap_sweep.py methodology (recall@k, literal P@5,
    store diversity, latency) for whatever apply_per_store_cap is CURRENTLY
    bound on the hs module -- i.e. the caller sets up (or doesn't set up) the
    monkeypatch before calling this."""
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
        literal_p5.append(sum(1 for a in top5 if a in universe_ids) / len(top5) if top5 else 0.0)

        d10, _ = _diversity(items, 10)
        d50, share50 = _diversity(items, 50)
        distinct10.append(d10)
        distinct50.append(d50)
        maxshare50.append(share50)

    row = {
        "variant": label,
        "recall": {k: _mean(recalls[k]) for k in _KS},
        "literal_p5": _mean(literal_p5),
        "distinct10": _mean(distinct10),
        "distinct50": _mean(distinct50),
        "maxshare50": _mean(maxshare50),
        "ms_per_call": _mean(latencies) * 1000,
    }
    print(
        f"{label:>22} recall@5={row['recall'][5]:.3f} recall@10={row['recall'][10]:.3f} "
        f"recall@20={row['recall'][20]:.3f} recall@50={row['recall'][50]:.3f} "
        f"literalP@5={row['literal_p5']:.3f} distinct@10={row['distinct10']:.2f} "
        f"distinct@50={row['distinct50']:.2f} maxshare@50={row['maxshare50']:.2f} "
        f"ms/call={row['ms_per_call']:.1f}"
    )
    return row


def main() -> None:
    config = load_config()
    catalogue_df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)
    # Fixed at the current production value throughout -- this sweep isolates
    # the normalization variant, not the cap value itself (already swept
    # separately in eval/per_store_cap_sweep.py).
    retriever.config["retrieval"]["per_store_cap"] = 4

    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]

    results = []
    results.append(_run_sweep_variant(retriever, queries, "baseline (cap=4)"))
    with patched_apply_per_store_cap("zscore"):
        results.append(_run_sweep_variant(retriever, queries, "normalized zscore (cap=4)"))
    with patched_apply_per_store_cap("minmax"):
        results.append(_run_sweep_variant(retriever, queries, "normalized minmax (cap=4)"))

    import json
    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _ROOT / "reports" / f"store_score_normalization_sweep_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
