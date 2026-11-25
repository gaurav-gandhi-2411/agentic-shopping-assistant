"""Label-scheme robustness of the paired contrasts (zero LLM calls).

The primary labels are the three-judge consensus. This recomputes the paired differences under
other label schemes to see whether a conclusion depends on one judge:

    consensus     all three judges, strict majority, median for a three-way split (the default)
    claude_only   Claude's labels alone
    qwen_only     qwen3.8-27b alone
    gpt_oss_only  gpt-oss-120b alone
    no_gpt_oss    consensus of Claude + qwen only (a disagreement resolves to the median of the
                  two, i.e. the mean, such as 0.5 or 1.5, and is flagged as a tie)

Population: queries with a pool of more than 5 candidates that have a row in every arm compared.

    python -m evals.retrieval.robustness --results-dir evals/retrieval/results
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.retrieval import agreement as ag
from evals.retrieval import metrics as mt
from evals.retrieval.common import (
    CANDIDATE_DIR,
    RUN_DIR,
    load_candidates,
    resolve_pools_dir,
)
from evals.retrieval.report import _per_query
from evals.retrieval.variants import TOP_K

SCHEMES: dict[str, tuple[str, ...]] = {
    "consensus": ("gpt-oss-120b", "qwen3.8-27b", "claude"),
    "claude_only": ("claude",),
    "qwen_only": ("qwen3.8-27b",),
    "gpt_oss_only": ("gpt-oss-120b",),
    "no_gpt_oss": ("qwen3.8-27b", "claude"),
}
CONTRASTS = (("A2", "B"), ("A2", "A"), ("C", "A2"))


def scheme_agg(
    labels: Mapping[str, Mapping[str, int]], judges: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Aggregate the labels of `judges` only (see ag.aggregate for the tie rule)."""
    return ag.aggregate({j: labels[j] for j in judges})


def load_labels(out_dir: Path) -> dict[str, dict[str, int]]:
    return {j: json.loads((out_dir / f"labels_{j}.json").read_text()) for j in SCHEMES["consensus"]}


def primary_rows(
    variants: Mapping[str, Sequence[Mapping[str, Any]]], pools: Path, arms: Sequence[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Rep-1 rows of `arms` restricted to pool > 5 queries present in every arm, same order."""
    ids: set[str] | None = None
    for arm in arms:
        have = {r["query_id"] for r in variants.get(arm, []) if int(r.get("rep", 1)) == 1}
        ids = have if ids is None else ids & have
    keep = sorted(q for q in (ids or set()) if len(load_candidates(q, pools)["items"]) > TOP_K)
    out = {
        arm: sorted(
            (
                dict(r)
                for r in variants[arm]
                if int(r.get("rep", 1)) == 1 and r["query_id"] in set(keep)
            ),
            key=lambda r: r["query_id"],
        )
        for arm in arms
    }
    return out, keep


def compute(
    out_dir: Path, pools: Path, variants: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """{scheme: {contrast: {metric: paired diff}}} plus the n and queries used per contrast."""
    labels = load_labels(out_dir)
    variants = variants or json.loads((out_dir / "variants_partial.json").read_text())
    res: dict[str, Any] = {"schemes": {k: list(v) for k, v in SCHEMES.items()}, "contrasts": {}}
    for x, y in CONTRASTS:
        if not variants.get(x) or not variants.get(y):
            res["contrasts"][f"{x}-{y}"] = {
                "skipped": f"no rows for {x if not variants.get(x) else y}"
            }
            continue
        rows, keep = primary_rows(variants, pools, (x, y))
        entry: dict[str, Any] = {"n_queries": len(keep), "queries": keep}
        for name, judges in SCHEMES.items():
            agg = scheme_agg(labels, judges)
            px, py = _per_query(rows[x], agg, "final_ids"), _per_query(rows[y], agg, "final_ids")
            entry[name] = {
                m: mt.paired_difference(px[m], py[m]) for m in ("ndcg5", "p5_ge1", "p5_ge2")
            }
        res["contrasts"][f"{x}-{y}"] = entry
    return res


def render_md(res: Mapping[str, Any]) -> str:
    L = [
        "# Label-scheme robustness of the paired contrasts (no LLM calls)",
        "",
        "LLM labels, not human ground truth. Paired difference of the first minus the second arm, "
        "bootstrap 95% CI (10,000, seed 42), rep 1, queries with pool > 5 present in both arms.",
        "",
    ]
    for contrast, entry in res["contrasts"].items():
        if "skipped" in entry:
            L += [f"## {contrast}: skipped ({entry['skipped']})", ""]
            continue
        L += [
            f"## {contrast} (n = {entry['n_queries']} queries)",
            "",
            "| scheme | nDCG@5 diff [95% CI] (n) | P@5 label >= 1 diff [95% CI] (n) |",
            "|---|---|---|",
        ]
        for name in res["schemes"]:
            a, b = entry[name]["ndcg5"], entry[name]["p5_ge1"]
            L.append(
                f"| {name} | {a['mean_diff']:+.3f} [{a['ci_lo']:+.3f}, {a['ci_hi']:+.3f}] ({a['n']}) "
                f"| {b['mean_diff']:+.3f} [{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}] ({b['n']}) |"
            )
        L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(RUN_DIR))
    ap.add_argument("--cand", default=str(CANDIDATE_DIR))
    ap.add_argument("--pools", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    res = compute(Path(args.out), resolve_pools_dir(Path(args.cand), args.pools))
    md = render_md(res)
    rdir = Path(args.results_dir) if args.results_dir else Path(args.out)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "robustness.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (rdir / "robustness.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
