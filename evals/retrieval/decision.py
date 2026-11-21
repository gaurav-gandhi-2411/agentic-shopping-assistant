"""Apply the pre-registered stage-2 decision rule (README, "Pre-registered decision rule").

Primary metric nDCG@5 on the consensus labels, rep 1, queries with a pool of more than 5
candidates. *best* = highest nDCG@5 point estimate. A variant is admissible when the 95% lower
bound of the paired bootstrap difference (variant - best) is >= -0.03 (10,000 resamples, seed
42). The cheapest admissible variant wins: mean tokens per turn, then median latency. P@5 and
stability are reported elsewhere and play no part here.

    python -m evals.retrieval.decision --results-dir evals/retrieval/results
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.retrieval import metrics as mt
from evals.retrieval.common import (
    CANDIDATE_DIR,
    RUN_DIR,
    load_candidates,
    resolve_pools_dir,
)
from evals.retrieval.offline import load_agg
from evals.retrieval.report import row_metrics
from evals.retrieval.variants import TOP_K

MARGIN = -0.03  # pre-registered lower-bound threshold
ARMS = ("A", "A2", "B", "C")


def tokens_per_turn(rows: Sequence[Mapping[str, Any]]) -> float:
    """Mean prompt + completion tokens over the turns of the primary population (B = 0)."""
    vals = [(r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def latency_p50(rows: Sequence[Mapping[str, Any]]) -> float:
    vals = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    return float(statistics.median(vals)) if vals else 0.0


def apply_rule(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    agg: Mapping[str, Mapping[str, Any]],
    margin: float = MARGIN,
) -> dict[str, Any]:
    """`rows[arm]` = rep-1 rows over the SAME queries (sorted by query_id). Returns the table."""
    ndcg = {a: [row_metrics(r, agg)["ndcg5"] for r in rs] for a, rs in rows.items()}
    means = {
        a: statistics.fmean([x for x in v if not math.isnan(x)]) if v else float("nan")
        for a, v in ndcg.items()
    }
    best = max(means, key=lambda a: (means[a], -ARMS.index(a)))
    table: dict[str, Any] = {}
    for a in rows:
        diffs = [
            x - y
            for x, y in zip(ndcg[a], ndcg[best], strict=True)
            if not (math.isnan(x) or math.isnan(y))
        ]
        mean, lo, hi = mt.bootstrap_mean_ci(diffs) if a != best else (0.0, 0.0, 0.0)
        table[a] = {
            "ndcg5": means[a],
            "n_ndcg": len(diffs),
            "diff_vs_best": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "admissible": a == best or lo >= margin,
            "tokens_per_turn": tokens_per_turn(rows[a]) if a != "B" else 0.0,
            "latency_p50_ms": latency_p50(rows[a]) if a != "B" else 0.0,
        }
    admissible = [a for a in rows if table[a]["admissible"]]
    chosen = min(
        admissible, key=lambda a: (table[a]["tokens_per_turn"], table[a]["latency_p50_ms"])
    )
    return {"best": best, "chosen": chosen, "margin": margin, "variants": table}


def primary_rows(
    variants: Mapping[str, Sequence[Mapping[str, Any]]], pools: Path
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    ids: set[str] | None = None
    for a in ARMS:
        have = {r["query_id"] for r in variants[a] if int(r.get("rep", 1)) == 1}
        ids = have if ids is None else ids & have
    keep = sorted(q for q in (ids or set()) if len(load_candidates(q, pools)["items"]) > TOP_K)
    rows = {
        a: sorted(
            (dict(r) for r in variants[a] if int(r.get("rep", 1)) == 1 and r["query_id"] in keep),
            key=lambda r: r["query_id"],
        )
        for a in ARMS
    }
    return rows, keep


def render_md(res: Mapping[str, Any], n_queries: int) -> str:
    L = [
        "# Stage-2 decision rule applied (pre-registered, README)",
        "",
        f"Primary population: {n_queries} queries (pool > 5 candidates), rep 1, consensus labels "
        "(LLM labels, not human ground truth). Admissible: 95% lower bound of (variant - best) "
        f">= {res['margin']}. Cheapest admissible wins (tokens per turn, then latency p50).",
        "",
        f"best by nDCG@5 point estimate: **{res['best']}**; chosen: **{res['chosen']}**",
        "",
        "| variant | nDCG@5 | n | diff vs best [95% CI] | admissible | tokens/turn | latency p50 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for a, t in res["variants"].items():
        diff = (
            "(best)"
            if a == res["best"]
            else (f"{t['diff_vs_best']:+.3f} [{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}]")
        )
        L.append(
            f"| {a} | {t['ndcg5']:.3f} | {t['n_ndcg']} | {diff} | {t['admissible']} | "
            f"{t['tokens_per_turn']:.0f} | {t['latency_p50_ms']:.0f} |"
        )
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(RUN_DIR))
    ap.add_argument("--cand", default=str(CANDIDATE_DIR))
    ap.add_argument("--pools", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out)
    variants = json.loads((out / "variants_partial.json").read_text())
    rows, keep = primary_rows(variants, resolve_pools_dir(Path(args.cand), args.pools))
    res = apply_rule(rows, load_agg(out))
    res["queries"] = keep
    md = render_md(res, len(keep))
    rdir = Path(args.results_dir) if args.results_dir else out
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "decision.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (rdir / "decision.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
