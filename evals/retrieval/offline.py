"""Offline stage-2 results (zero LLM calls): variant B, factors F1 / F2 for B, variant A rep 1.

Everything here re-scores rows that already exist (variant A rep 1 from `variants_partial.json`)
or that need no model (B is the pool order, so its F1 / F2 contrasts are pure reorderings). The
LLM arms of F1 / F2 need live calls and are NOT computed here.

Populations (pre-registered in README, "Pre-registered decision rule"): the primary comparison
uses the queries whose master pool has more than 5 candidates (the only ones where variants can
differ); a sensitivity row uses every query with candidates. Queries with no candidates are
excluded from both.

    python -m evals.retrieval.offline --out reports/wave1b/v4e --results-dir evals/retrieval/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from evals.retrieval import agreement as ag
from evals.retrieval import metrics as mt
from evals.retrieval.common import (
    CANDIDATE_DIR,
    RUN_DIR,
    load_candidates,
    load_cases,
    resolve_pools_dir,
)
from evals.retrieval.factors import (
    interleave_gender,
    is_gender_unspecified,
    is_set_demotion_query,
    pool_loss,
)
from evals.retrieval.report import METRICS, _per_query, _pool_labels, build_report
from evals.retrieval.variants import TOP_K, classify_reply, run_variant_b

JUDGES = ("gpt-oss-120b", "qwen3.8-27b", "claude")


def reclassify(row: Mapping[str, Any]) -> dict[str, Any]:
    """Re-label an old variant row's outcome with the current taxonomy (from its stored raw
    output and finish_reason): empty `selected` + stop becomes empty_selection, not parse_error."""
    out = dict(row)
    if row.get("outcome") == "no_llm_call":
        return out
    if row.get("raw_output") is None:
        out["outcome"] = "http_error"
    else:
        call = {"output": row["raw_output"], "meta": {"finish_reason": row.get("finish_reason")}}
        out["outcome"] = classify_reply(call, row.get("parsed") or None)
    out["no_exact_match"] = out["outcome"] == "empty_selection"
    return out


def load_agg(out_dir: Path) -> dict[str, dict[str, Any]]:
    labels = {
        j: json.loads((out_dir / f"labels_{j}.json").read_text())
        for j in JUDGES
        if (out_dir / f"labels_{j}.json").exists()
    }
    return ag.aggregate(labels)


def _row_b(qid: str, pool: list[dict[str, Any]]) -> dict[str, Any]:
    return {"query_id": qid, "rep": 1, **run_variant_b(pool)}


def _fmt(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def _paired(d: Mapping[str, float]) -> str:
    return (
        f"{d['mean_diff']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] "
        f"(n={d['n']}, sign p={d['sign_p']:.3f}, wilcoxon p={d['wilcoxon_p']:.3f})"
    )


def factor_rows(
    cases: list[dict[str, Any]],
    agg: Mapping[str, Mapping[str, Any]],
    frozen: Path,
    pools: Path,
) -> dict[str, Any]:
    """F1 and F2 for variant B (the only arm computable offline).

    F1: base = B on the frozen pool ("sets not demoted"), treated = B on the master pool (#129).
    F2: base = B on the master pool, treated = B on the gender-interleaved master pool."""
    out: dict[str, Any] = {"F1": {"B": {}}, "F2": {"B": {}}}
    base_rows, treated_rows = [], []
    loss = {"n_before": 0, "n_after": 0, "n_removed": 0, "n_removed_relevant": 0}
    f1 = [c for c in cases if is_set_demotion_query(c)]
    for c in f1:
        fz, ms = load_candidates(c["id"], frozen)["items"], load_candidates(c["id"], pools)["items"]
        lab = {a: x["label"] for a, x in _pool_labels(agg, c["id"]).items()}
        base_rows.append({**_row_b(c["id"], fz), "reused": False})
        treated_rows.append(
            {
                **_row_b(c["id"], ms),
                "reused": [i["article_id"] for i in fz] == [i["article_id"] for i in ms],
            }
        )
        for k, v in pool_loss(fz, ms, lab).items():
            loss[k] += v
    out["F1"]["B"] = {"base": base_rows, "treated": treated_rows, "pool_loss": loss}
    f1_info = [
        {
            "id": c["id"],
            "text": c["text"],
            "pool_changed": [i["article_id"] for i in load_candidates(c["id"], frozen)["items"]]
            != [i["article_id"] for i in load_candidates(c["id"], pools)["items"]],
        }
        for c in f1
    ]
    f2 = [c for c in cases if is_gender_unspecified(c)]
    b2, t2 = [], []
    for c in f2:
        pool = load_candidates(c["id"], pools)["items"]
        b2.append(_row_b(c["id"], pool))
        t2.append(_row_b(c["id"], interleave_gender(pool)))
    out["F2"]["B"] = {"base": b2, "treated": t2}
    out["_f1_queries"] = f1_info
    out["_f2_queries"] = [c["id"] for c in f2]
    return out


def factor_tables(
    factors: Mapping[str, Any], agg: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fac in ("F1", "F2"):
        arm = factors[fac]["B"]
        pb, pt = (
            _per_query(arm["base"], agg, "final_ids"),
            _per_query(arm["treated"], agg, "final_ids"),
        )
        entry: dict[str, Any] = {
            "n_queries": len(arm["base"]),
            "base": {m: mt.bootstrap_mean_ci(v) for m, v in pb.items()},
            "treated": {m: mt.bootstrap_mean_ci(v) for m, v in pt.items()},
            "paired_treated_minus_base": {m: mt.paired_difference(pt[m], pb[m]) for m in METRICS},
            "n_pool_changed_or_reordered": sum(
                1
                for b, t in zip(arm["base"], arm["treated"], strict=True)
                if b["final_ids"] != t["final_ids"]
            ),
        }
        if "pool_loss" in arm:
            entry["pool_loss"] = arm["pool_loss"]
        out[fac] = entry
    return out


def compute(out_dir: Path, frozen: Path, pools: Path) -> dict[str, Any]:
    """All offline numbers as one JSON-able dict (see module docstring)."""
    agg = load_agg(out_dir)
    cases_all = load_cases()
    a_rows = [
        reclassify(r)
        for r in json.loads((out_dir / "variants_partial.json").read_text())["A"]
        if int(r.get("rep", 1)) == 1
    ]
    have_a = {r["query_id"] for r in a_rows}
    has_pool = [c for c in cases_all if load_candidates(c["id"], pools)["items"]]
    no_cand = [c["id"] for c in cases_all if c not in has_pool]
    with_cand = [c for c in has_pool if c["id"] in have_a]  # an arm must cover every query
    primary = [c for c in with_cand if len(load_candidates(c["id"], pools)["items"]) > TOP_K]
    small = [c["id"] for c in with_cand if c not in primary]
    b_rows = [_row_b(c["id"], load_candidates(c["id"], pools)["items"]) for c in with_cand]
    result: dict[str, Any] = {
        "labels": "LLM-consensus (3 families), not human ground truth",
        "n_items_labelled": len(agg),
        "n_tied_items": sum(1 for x in agg.values() if x["tie"]),
        "excluded_no_candidates": no_cand,
        "excluded_pool_le_5": small,
        "excluded_no_variant_a_row": [c["id"] for c in has_pool if c["id"] not in have_a],
        "populations": {},
    }
    for name, cs in (
        ("primary_pool_gt_5", primary),
        ("sensitivity_all_with_candidates", with_cand),
    ):
        ids = {c["id"] for c in cs}
        variants = {
            "A": [r for r in a_rows if r["query_id"] in ids],
            "B": [r for r in b_rows if r["query_id"] in ids],
        }
        rep = build_report(cs, variants, agg)
        rep["n_queries"] = len(cs)
        result["populations"][name] = {
            k: rep[k]
            for k in (
                "n_queries",
                "metrics",
                "metrics_post_gates",
                "sensitivity_excluding_ties",
                "paired",
                "usage",
                "rates",
                "outcomes",
            )
        }
    fac = factor_rows(with_cand, agg, frozen, pools)
    result["factors_B_only"] = {
        "tables": factor_tables(fac, agg),
        "f1_queries": fac["_f1_queries"],
        "f2_queries": fac["_f2_queries"],
    }
    result["_rows"] = {"A": a_rows, "B": b_rows, "factors": {k: fac[k] for k in ("F1", "F2")}}
    return result


def render_md(res: Mapping[str, Any]) -> str:
    """Human-readable results (tables only; every number comes from `res`)."""
    L = [
        "# Stage-2 offline results (no LLM calls)",
        "",
        f"Labels: {res['labels']}; {res['n_items_labelled']} items, {res['n_tied_items']} tied "
        "(median-resolved). Metrics are bootstrap means with 95% CIs (10,000 resamples, seed 42) "
        "over queries.",
        f"Excluded, no candidates: {', '.join(res['excluded_no_candidates'])}. Excluded from the "
        f"primary comparison, pool <= 5 candidates: {', '.join(res['excluded_pool_le_5'])}.",
        "",
    ]
    for name, pop in res["populations"].items():
        L += [
            f"## {name} (n = {pop['n_queries']} queries)",
            "",
            f"n per metric: P@5 {pop['paired']['A-B']['p5_ge1']['n']}; nDCG@5 "
            f"{pop['paired']['A-B']['ndcg5']['n']} (a query whose pool holds no label >= 1 has an "
            "ideal DCG of 0, so nDCG@5 is undefined and dropped from means and pairs).",
            "",
            "| variant | nDCG@5 | P@5 (>=1) | P@5 (>=1.5) | P@5 (>=2) | nDCG@5 excl. ties |",
            "|---|---|---|---|---|---|",
        ]
        for v in ("A", "B"):
            m, s = pop["metrics"][v], pop["sensitivity_excluding_ties"][v]
            L.append(
                f"| {v} | {_fmt(m['ndcg5'])} | {_fmt(m['p5_ge1'])} | {_fmt(m['p5_ge1_5'])} | "
                f"{_fmt(m['p5_ge2'])} | {_fmt(s['ndcg5'])} |"
            )
        L += ["", "Paired A - B (same queries):", ""]
        for m in METRICS:
            L.append(f"- {m}: {_paired(pop['paired']['A-B'][m])}")
        u = pop["usage"]["A"]
        L += [
            "",
            f"A usage (live calls n={u['n']}, skipped {u['n_no_llm_call']}): mean prompt "
            f"{u.get('mean_prompt_tokens', float('nan')):.0f}, completion "
            f"{u.get('mean_completion_tokens', float('nan')):.0f}, latency p50 "
            f"{u.get('latency_p50_ms', float('nan')):.0f} ms, p95 "
            f"{u.get('latency_p95_ms', float('nan')):.0f} ms.",
            f"A outcomes: {pop['rates']['A']['outcome_counts']}; no_exact_match "
            f"{pop['rates']['A']['no_exact_match']}; hit cap {pop['rates']['A']['length_hits']}.",
            "",
        ]
    ft = res["factors_B_only"]
    L += ["## Factors, variant B only (LLM arms need live calls)", ""]
    L.append(
        "F1 sets demoted vs not, flagged ethnic-wear queries (single-garment named on an "
        "ethnic occasion): "
        + "; ".join(
            f'{q["id"]} "{q["text"]}"{" (pool changed)" if q["pool_changed"] else ""}'
            for q in res["factors_B_only"]["f1_queries"]
        )
        + "."
    )
    L.append("")
    for fac, label in (
        ("F1", "F1 treated (master, sets demoted) - base (frozen, not demoted)"),
        ("F2", "F2 treated (gender-interleaved) - base (current order)"),
    ):
        t = ft["tables"][fac]
        L += [
            f"### {label}, n = {t['n_queries']} queries ({t['n_pool_changed_or_reordered']} with a different top 5)",
            "",
            "| arm | nDCG@5 | P@5 (>=1) |",
            "|---|---|---|",
            f"| base | {_fmt(t['base']['ndcg5'])} | {_fmt(t['base']['p5_ge1'])} |",
            f"| treated | {_fmt(t['treated']['ndcg5'])} | {_fmt(t['treated']['p5_ge1'])} |",
            "",
        ]
        for m in ("ndcg5", "p5_ge1"):
            L.append(f"- paired {m}: {_paired(t['paired_treated_minus_base'][m])}")
        if "pool_loss" in t:
            L.append(f"- pool loss: {t['pool_loss']}")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(RUN_DIR))
    ap.add_argument("--cand", default=str(CANDIDATE_DIR), help="frozen pools")
    ap.add_argument("--pools", default=None, help="master-regenerated pools")
    ap.add_argument("--results-dir", default=None, help="write offline_results.md/.json here")
    args = ap.parse_args(argv)
    frozen = Path(args.cand)
    res = compute(Path(args.out), frozen, resolve_pools_dir(frozen, args.pools))
    rows = res.pop("_rows")
    out_dir = Path(args.out)
    # B rows join the partial variants file (arms A and B only; never variants.json). The stored
    # A rows stay exactly as the live run wrote them (original outcome names); the re-classified
    # outcomes only appear in the results.
    path = out_dir / "variants_partial.json"
    stored = json.loads(path.read_text())
    path.write_text(json.dumps({"A": stored["A"], "B": rows["B"]}, indent=1), encoding="utf-8")
    md = render_md(res)
    rdir = Path(args.results_dir) if args.results_dir else out_dir
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "offline_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (rdir / "offline_results.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
