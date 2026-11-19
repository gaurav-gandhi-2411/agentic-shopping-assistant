"""Report assembly: per-variant metrics with bootstrap CIs, paired differences, rates, the
card-count stability analysis (repeated runs) and the factor tables (F1 / F2).

Rows come from `variants.json`: {arm: [row, ...]}; a row has query_id, rep (1-based, default 1),
final_ids (the reranker's top 5) and cards (the ids left after the replayed post-rerank gates).
Primary comparisons use rep 1 of every arm (one draw per query, like production); the other reps
only feed the spread and stability numbers.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from evals.retrieval import metrics as mt

ARMS_ORDER = ("A", "A2", "B", "C")
PAIRS = (("A", "B"), ("A2", "B"), ("A2", "A"), ("C", "B"), ("C", "A"), ("C", "A2"))
# "timeout" and "length_fallback" are the pre-taxonomy names (rows from before 2026-11-20)
FALLBACK_OUTCOMES = (
    "parse_error",
    "length",
    "empty_selection",
    "http_error",
    "budget",
    "timeout",
    "length_fallback",
)
METRICS = ("ndcg5", "p5_ge1", "p5_ge2", "p5_ge1_5")


def _pool_labels(agg: Mapping[str, Mapping[str, Any]], qid: str) -> dict[str, Mapping[str, Any]]:
    prefix = qid + ":"
    return {k[len(prefix) :]: v for k, v in agg.items() if k.startswith(prefix)}


def row_metrics(
    row: Mapping[str, Any], agg: Mapping[str, Mapping[str, Any]], ids_key: str = "final_ids"
) -> dict[str, float]:
    """nDCG@5 / P@5 for one row. The ideal ranking comes from ALL labelled items of the query
    (the full frozen pool), never from the shown items, so arms are comparable."""
    lab = _pool_labels(agg, row["query_id"])
    ids = row.get(ids_key) or []
    shown = [lab[a]["label"] for a in ids if a in lab]
    pool = [x["label"] for x in lab.values()]
    return {
        "ndcg5": mt.ndcg_at_k(shown, pool),
        "p5_ge1": mt.precision_at_k(shown, 1.0),
        "p5_ge2": mt.precision_at_k(shown, 2.0),
        "p5_ge1_5": mt.precision_at_k(shown, 1.5),
    }


def _primary(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in rows if int(r.get("rep", 1)) == 1]


def _per_query(
    rows: Sequence[Mapping[str, Any]], agg: Mapping[str, Mapping[str, Any]], ids_key: str
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {m: [] for m in METRICS}
    for r in rows:
        for m, v in row_metrics(r, agg, ids_key).items():
            out[m].append(v)
    return out


def rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Length / parse-failure / fallback rates over the rows that made an LLM call."""
    called = [r for r in rows if r.get("outcome") != "no_llm_call"]
    n = len(called)

    def frac(k: int) -> float:
        return k / n if n else float("nan")

    return {
        "n_rows": len(rows),
        "n_llm_calls": n,
        "length_hits": sum(1 for r in called if r.get("hit_cap")),
        "length_rate": frac(sum(1 for r in called if r.get("hit_cap"))),
        "parse_error_rate": frac(sum(1 for r in called if r.get("outcome") == "parse_error")),
        "timeout_rate": frac(sum(1 for r in called if r.get("outcome") == "timeout")),
        "fallback_rate": frac(sum(1 for r in called if r.get("outcome") in FALLBACK_OUTCOMES)),
        "partial_rate": frac(sum(1 for r in called if r.get("outcome") == "partial")),
        # every reason counted on its own, never lumped into "timeout"
        "outcome_counts": {
            o: sum(1 for r in called if r.get("outcome") == o)
            for o in ("ok", "partial", "length", "parse_error", "empty_selection", "http_error")
            if any(r.get("outcome") == o for r in called)
        },
        "no_exact_match": sum(1 for r in called if r.get("no_exact_match")),
    }


def stability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How often repeated runs of the same query disagree (needs >= 2 reps of an arm).

    card count = len(cards) after the replayed post-rerank gates; "top5 set" = the reranker's
    five ids before the gates. Both are compared per query across reps."""
    by_q: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_q.setdefault(r["query_id"], []).append(r)
    multi = {q: rs for q, rs in by_q.items() if len(rs) >= 2}
    if not multi:
        return {"n_queries_repeated": 0}
    count_diff, set_diff, detail = 0, 0, {}
    for q, rs in sorted(multi.items()):
        counts = [len(r.get("cards") or []) for r in rs]
        sets = {frozenset(r.get("final_ids") or []) for r in rs}
        cd, sd = len(set(counts)) > 1, len(sets) > 1
        count_diff += cd
        set_diff += sd
        if cd or sd:
            detail[q] = {"card_counts": counts, "top5_sets_differ": sd}
    n = len(multi)
    return {
        "n_queries_repeated": n,
        "reps_per_query": sorted({len(rs) for rs in multi.values()}),
        "card_count_differs": count_diff,
        "card_count_differs_rate": count_diff / n,
        "top5_set_differs": set_diff,
        "top5_set_differs_rate": set_diff / n,
        "queries_that_differ": detail,
    }


def rep_spread(rows: Sequence[Mapping[str, Any]], agg: Mapping[str, Mapping[str, Any]]) -> dict:
    """Mean nDCG@5 of each rep separately: the run-to-run noise floor of a stochastic arm."""
    reps = sorted({int(r.get("rep", 1)) for r in rows})
    out = {}
    for k in reps:
        vals = [
            v for r in rows if int(r.get("rep", 1)) == k for v in [row_metrics(r, agg)["ndcg5"]]
        ]
        vals = [v for v in vals if not math.isnan(v)]
        out[f"rep{k}"] = sum(vals) / len(vals) if vals else float("nan")
    return out


def select_top2(metrics: Mapping[str, Mapping[str, Sequence[float]]]) -> list[str]:
    """The two arms with the highest nDCG@5 POINT estimate (ties broken by ARMS_ORDER)."""
    ranked = sorted(
        metrics,
        key=lambda a: (-metrics[a]["ndcg5"][0], ARMS_ORDER.index(a) if a in ARMS_ORDER else 9),
    )
    return ranked[:2]


def build_report(
    cases: Sequence[Mapping[str, Any]],
    variants: Mapping[str, Sequence[Mapping[str, Any]]],
    agg: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Metrics per variant (bootstrap CIs), paired differences, tie-excluded sensitivity, rates
    and stability. `variants` maps arm -> rows (see module docstring)."""
    per: dict[str, dict[str, list[float]]] = {}
    per_cards: dict[str, dict[str, list[float]]] = {}
    sens: dict[str, dict[str, list[float]]] = {}
    for v, all_rows in variants.items():
        rows = _primary(all_rows)
        per[v] = _per_query(rows, agg, "final_ids")
        per_cards[v] = _per_query([r for r in rows if "cards" in r], agg, "cards")
        sens[v] = {"ndcg5": [], "p5_ge1": []}
        for row in rows:
            lab = _pool_labels(agg, row["query_id"])
            keep = {a for a, x in lab.items() if not x["tie"]}
            s_shown = [lab[a]["label"] for a in row["final_ids"] if a in keep]
            sens[v]["ndcg5"].append(mt.ndcg_at_k(s_shown, [lab[a]["label"] for a in keep]))
            sens[v]["p5_ge1"].append(
                sum(1 for x in s_shown if x >= 1) / len(s_shown) if s_shown else float("nan")
            )

    def ci(d: Mapping[str, Sequence[float]]) -> dict[str, tuple[float, float, float]]:
        return {m: mt.bootstrap_mean_ci(vals) for m, vals in d.items()}

    summary = {v: ci(d) for v, d in per.items()}
    pairs = {}
    for a, b in PAIRS:
        if a in per and b in per:
            pairs[f"{a}-{b}"] = {m: mt.paired_difference(per[a][m], per[b][m]) for m in METRICS}
    all_called = {v: list(rows) for v, rows in variants.items()}
    return {
        "n_queries": len(cases),
        "metrics": summary,
        "metrics_post_gates": {v: ci(d) for v, d in per_cards.items() if d["ndcg5"]},
        "sensitivity_excluding_ties": {v: ci(d) for v, d in sens.items()},
        "paired": pairs,
        "usage": {v: mt.usage_summary(rows) for v, rows in all_called.items()},
        "outcomes": {
            v: {
                o: sum(1 for r in rows if r.get("outcome") == o)
                for o in {r.get("outcome") for r in rows}
            }
            for v, rows in all_called.items()
        },
        "rates": {v: rates(rows) for v, rows in all_called.items()},
        "rep_spread": {
            v: rep_spread(rows, agg) for v, rows in all_called.items() if v in ("A", "A2")
        },
        "stability": {v: stability(rows) for v, rows in all_called.items() if v in ("A", "A2")},
        "top2_by_ndcg5_point_estimate": select_top2(summary) if summary else [],
        "n_tied_items": sum(1 for x in agg.values() if x["tie"]),
        "n_items": len(agg),
    }


def build_factor_report(
    factors: Mapping[str, Any], agg: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """F1 / F2 tables. `factors` = {"top2": [...], "F1": {variant: {"base": rows, "treated":
    rows, "pool_loss": {...}}}, "F2": {...}}: base and treated rows cover the SAME queries."""
    out: dict[str, Any] = {"top2": factors.get("top2", [])}
    for fac in ("F1", "F2"):
        out[fac] = {}
        for variant, arms in (factors.get(fac) or {}).items():
            base, treated = arms["base"], arms["treated"]
            pb, pt = _per_query(base, agg, "final_ids"), _per_query(treated, agg, "final_ids")
            entry: dict[str, Any] = {
                "n_queries": len(base),
                "base": {m: mt.bootstrap_mean_ci(v) for m, v in pb.items()},
                "treated": {m: mt.bootstrap_mean_ci(v) for m, v in pt.items()},
                "paired_treated_minus_base": {
                    m: mt.paired_difference(pt[m], pb[m]) for m in METRICS
                },
                "n_reused_rows": sum(1 for r in treated if r.get("reused")),
                "usage": mt.usage_summary(treated),
                "rates": rates(treated),
            }
            if "pool_loss" in arms:
                entry["pool_loss"] = arms["pool_loss"]
            if fac == "F1":
                entry["sets_in_shown_top5"] = {
                    "base": sum(r.get("n_sets_shown", 0) for r in base),
                    "treated": sum(r.get("n_sets_shown", 0) for r in treated),
                }
            out[fac][variant] = entry
    return out
