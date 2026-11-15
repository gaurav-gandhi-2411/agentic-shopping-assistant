"""Inter-judge agreement and label aggregation (pure numpy; no sklearn/krippendorff dependency).

Labels are graded relevance 0/1/2. Provides Cohen's kappa (unweighted and quadratic-weighted),
Krippendorff's alpha (ordinal) with a bootstrap CI over units, pairwise confusion tables,
majority/median aggregation with tie flags, and the calibration gate (any judge pair with
kappa < 0.4 stops the run before the main judging pass).

These are LLM-consensus statistics, not agreement with human ground truth.
"""

from __future__ import annotations

import itertools
import math
import statistics
from typing import Mapping, Sequence

import numpy as np

LABELS = (0, 1, 2)
KAPPA_GATE = 0.4  # architect gate: any calibration pair below this stops the main run


def confusion(a: Sequence[int], b: Sequence[int], labels: Sequence[int] = LABELS) -> np.ndarray:
    """Confusion matrix, rows = judge a's label, columns = judge b's label."""
    idx = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for x, y in zip(a, b, strict=True):
        m[idx[x], idx[y]] += 1
    return m


def cohen_kappa(
    a: Sequence[int],
    b: Sequence[int],
    weights: str | None = None,
    labels: Sequence[int] = LABELS,
) -> float:
    """Cohen's kappa; weights=None (unweighted) or "quadratic". NaN if undefined."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("kappa needs two equal-length, non-empty label sequences")
    obs = confusion(a, b, labels).astype(float)
    n = obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    k = len(labels)
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "quadratic":
        w = ((i - j) ** 2) / float((k - 1) ** 2)
    else:
        raise ValueError(f"unknown weights {weights!r}")
    denom = float((w * exp).sum())
    if denom == 0.0:
        return float("nan")  # one judge used a single label: chance agreement is total
    return 1.0 - float((w * obs).sum()) / denom


def krippendorff_alpha_ordinal(units: Sequence[Sequence[int | None]]) -> float:
    """Krippendorff's alpha, ordinal metric. `units[u]` are the labels given to item u by each
    coder (None = missing). Units with fewer than 2 values are ignored. NaN if undefined."""
    vals = [[v for v in u if v is not None] for u in units]
    vals = [u for u in vals if len(u) >= 2]
    k = len(LABELS)
    o = np.zeros((k, k))
    for u in vals:
        m = len(u)
        for x, y in itertools.permutations(range(m), 2):
            o[u[x], u[y]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    delta = np.zeros((k, k))
    for c in range(k):
        for d in range(c + 1, k):
            g = n_c[c : d + 1].sum() - (n_c[c] + n_c[d]) / 2.0
            delta[c, d] = delta[d, c] = g * g
    d_o = float((o * delta).sum()) / n
    d_e = float((np.outer(n_c, n_c) * delta).sum()) / (n * (n - 1))
    if d_e == 0.0:
        return float("nan")
    return 1.0 - d_o / d_e


def bootstrap_alpha_ci(
    units: Sequence[Sequence[int | None]],
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% percentile CI of ordinal alpha, resampling units with replacement."""
    rng = np.random.default_rng(seed)
    arr = list(units)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(arr), len(arr))
        a = krippendorff_alpha_ordinal([arr[i] for i in pick])
        if not math.isnan(a):
            stats.append(a)
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def aggregate_item(labels: Sequence[int]) -> dict:
    """Majority label of one item's judge labels.

    `label` is the strict-majority label when one exists, else the median (the primary-metric
    resolution of ties); `tie` flags the no-strict-majority case (e.g. a 0/1/2 three-way split).
    """
    counts = {lab: list(labels).count(lab) for lab in set(labels)}
    top_lab, top_n = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    if top_n * 2 > len(labels):
        return {"label": float(top_lab), "tie": False, "n": len(labels)}
    return {"label": float(statistics.median(labels)), "tie": True, "n": len(labels)}


def aggregate(judge_labels: Mapping[str, Mapping[str, int]]) -> dict[str, dict]:
    """judge -> {item_key: label}  ->  item_key -> aggregate_item(...) over judges that labelled it."""
    keys = sorted({k for lab in judge_labels.values() for k in lab})
    return {k: aggregate_item([lab[k] for lab in judge_labels.values() if k in lab]) for k in keys}


def pairwise_report(judge_labels: Mapping[str, Mapping[str, int]]) -> dict:
    """Pairwise kappas (+confusions) on commonly labelled items, alpha over all judges, and
    per-judge label distributions."""
    judges = sorted(judge_labels)
    pairs = {}
    for ja, jb in itertools.combinations(judges, 2):
        common = sorted(set(judge_labels[ja]) & set(judge_labels[jb]))
        if not common:
            continue
        a = [judge_labels[ja][k] for k in common]
        b = [judge_labels[jb][k] for k in common]
        pairs[f"{ja}|{jb}"] = {
            "n": len(common),
            "kappa": cohen_kappa(a, b),
            "kappa_quadratic": cohen_kappa(a, b, "quadratic"),
            "confusion": confusion(a, b).tolist(),
        }
    keys = sorted({k for lab in judge_labels.values() for k in lab})
    units = [[judge_labels[j].get(k) for j in judges] for k in keys]
    lo, hi = bootstrap_alpha_ci(units) if units else (float("nan"), float("nan"))
    dist = {
        j: {str(lab): list(judge_labels[j].values()).count(lab) for lab in LABELS} for j in judges
    }
    return {
        "pairs": pairs,
        "alpha_ordinal": krippendorff_alpha_ordinal(units) if units else float("nan"),
        "alpha_ci95": [lo, hi],
        "distributions": dist,
    }


def calibration_gate(judge_labels: Mapping[str, Mapping[str, int]]) -> tuple[bool, str]:
    """(passed, message). Fails closed: a NaN kappa or a single judge counts as a failure."""
    rep = pairwise_report(judge_labels)
    if not rep["pairs"]:
        return False, "calibration gate: fewer than two judges with overlapping labels"
    bad = [
        (p, v["kappa"])
        for p, v in rep["pairs"].items()
        if math.isnan(v["kappa"]) or v["kappa"] < KAPPA_GATE
    ]
    if not bad:
        return True, "calibration gate passed: all judge-pair kappas >= %.2f" % KAPPA_GATE
    lines = [
        f"CALIBRATION GATE FAILED (kappa < {KAPPA_GATE}): "
        + ", ".join(f"{p} kappa={k:.3f}" for p, k in bad)
    ]
    for p, v in rep["pairs"].items():
        lines.append(
            f"confusion {p} (rows=first judge, cols=second; labels 0,1,2): {v['confusion']}"
        )
    return False, "\n".join(lines)
