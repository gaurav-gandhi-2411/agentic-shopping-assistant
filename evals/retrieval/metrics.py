"""Ranking metrics on the SHOWN top-5, bootstrap CIs and paired tests (pure numpy).

nDCG@5 uses gain 2^label - 1 and an ideal ranking drawn from ALL labelled candidates of the
query (not just the 5 shown), so variants are comparable. Queries whose whole pool is label 0
have an undefined nDCG (NaN) and drop out of nDCG means for every variant alike.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

K = 5
N_BOOT = 10_000
SEED = 42


def dcg(labels: Sequence[float], k: int = K) -> float:
    return sum((2.0**lab - 1.0) / math.log2(i + 2) for i, lab in enumerate(labels[:k]))


def ndcg_at_k(shown: Sequence[float], pool: Sequence[float], k: int = K) -> float:
    """nDCG@k of the shown labels vs the ideal ordering of `pool` (NaN if the ideal is 0)."""
    ideal = dcg(sorted(pool, reverse=True), k)
    return float("nan") if ideal == 0 else dcg(shown, k) / ideal


def precision_at_k(shown: Sequence[float], threshold: float = 1.0, k: int = K) -> float:
    """Fraction of the k shown slots with label >= threshold (missing slots count as misses)."""
    return sum(1 for lab in shown[:k] if lab >= threshold) / k


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile 95% CI of the mean over query resamples (NaNs dropped)."""
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n_boot, v.size))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_p(diffs: Sequence[float]) -> float:
    """Two-sided exact binomial sign test on non-zero differences."""
    nz = [d for d in diffs if d != 0 and not math.isnan(d)]
    n = len(nz)
    if n == 0:
        return 1.0
    k = sum(1 for d in nz if d > 0)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2.0**n
    return min(1.0, 2 * tail)


def wilcoxon_p(diffs: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p, normal approximation with tie + continuity correction.

    The normal approximation is adequate at the planned n=60; no scipy dependency."""
    d = np.asarray([x for x in diffs if x != 0 and not math.isnan(x)], dtype=float)
    n = d.size
    if n == 0:
        return 1.0
    absd = np.abs(d)
    order = np.argsort(absd, kind="mergesort")
    ranks = np.empty(n)
    sorted_abs = absd[order]
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        t = j - i + 1
        tie_term += t**3 - t
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return 1.0
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(var)
    return float(math.erfc(max(z, 0.0) / math.sqrt(2)))


def paired_difference(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """x - y over queries (NaN pairs dropped): mean, bootstrap 95% CI, sign and Wilcoxon p."""
    diffs = [a - b for a, b in zip(x, y, strict=True) if not (math.isnan(a) or math.isnan(b))]
    mean, lo, hi = bootstrap_mean_ci(diffs)
    return {
        "n": len(diffs),
        "mean_diff": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "sign_p": sign_test_p(diffs),
        "wilcoxon_p": wilcoxon_p(diffs),
    }


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else float("nan")


def usage_summary(rows: Sequence[dict]) -> dict[str, float]:
    """Mean prompt/completion tokens and latency p50/p95 over rows that made an LLM call.

    `no_llm_call` rows (pool <= top_k, reranker skipped; variant B) carry recorded zeros that
    are not measurements: including them would drag token and latency means towards 0, so they
    are excluded and only counted (`n_no_llm_call`). `n` is the number of LLM-call rows."""
    called = [r for r in rows if r.get("outcome") != "no_llm_call"]
    if not called:
        return {"n": 0, "n_no_llm_call": len(rows) - len(called)}
    pt = [r["prompt_tokens"] for r in called if r.get("prompt_tokens") is not None]
    ct = [r["completion_tokens"] for r in called if r.get("completion_tokens") is not None]
    lat = [r["latency_ms"] for r in called if r.get("latency_ms") is not None]
    return {
        "n": len(called),
        "n_no_llm_call": len(rows) - len(called),
        "mean_prompt_tokens": float(np.mean(pt)) if pt else float("nan"),
        "mean_completion_tokens": float(np.mean(ct)) if ct else float("nan"),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
    }
