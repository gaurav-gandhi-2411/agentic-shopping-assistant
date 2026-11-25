"""Offline eval guard for plain-search post-processing (rule: results/plain_search_gate_rule.md).

For each of the 60 stage-2 query pools (53 have a defined nDCG@5): the production
`rerank_ranked()` is run on the eval's variant C rep-1 replies (zero live calls), giving the
unprocessed top-5 ("current") and the ranked remainder; `search_postprocess.select_results` then
gives the post-processed top-5 ("new"). Both are scored with nDCG@5 on the consensus labels
(unlabelled shown item = 0 in both arms), paired bootstrap 10,000 resamples seed 42 over the
queries with a defined nDCG@5. Also reports duplicate / store-cap / budget / gender figures
before and after, on the 60 pools and on the 8 landing-chip pools (retrieval order: no model
replies exist for the chips).

    python evals/retrieval/plain_search_guard.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import rerank_gate as rg  # noqa: E402

from src.agents import search_postprocess as pp  # noqa: E402
from src.agents.intent_parser import parse_intent  # noqa: E402
from src.agents.reranker import rerank_ranked  # noqa: E402

TOP_K = 5
STAGE2 = rg.STAGE2
RUN_DIR = rg.DEFAULT_RUN_DIR
POOLS = rg.DEFAULT_POOLS
CHIPS = rg.DEFAULT_CHIPS
EVAL_CACHE = RUN_DIR / "cache.jsonl"


def ndcg_unlabelled_zero(ids: list[str], labels: dict[str, float]) -> float:
    """nDCG@5 vs the ideal ordering of all labelled candidates; an unlabelled shown item is 0."""
    shown = [labels.get(a, 0.0) for a in ids]
    return rg.ndcg_at_k(shown, list(labels.values()))


def dup_pairs(items: list[dict]) -> int:
    """Same-product pairs among the shown cards (module key, plus the legacy name+colour key)."""
    n = 0
    for i in range(len(items)):
        for j in range(i):
            if pp.same_product(items[i], items[j]) or pp.legacy_key(items[i]) == pp.legacy_key(items[j]):
                n += 1
    return n


def max_per_store(items: list[dict]) -> int:
    return max(Counter(str(i.get("store") or "") for i in items).values(), default=0)


def query_filters(text: str) -> dict[str, Any]:
    """The filters search_node would hold for this text: gender (explicit/implied) and budget."""
    intent = parse_intent(text)
    f: dict[str, Any] = {}
    if intent.gender in ("women", "men"):
        f["gender"] = intent.gender
    if intent.budget_max_inr:
        f["price_max"] = intent.budget_max_inr
    return f


def process(text: str, shown: list[dict], rest: list[dict]) -> tuple[list[dict], dict, dict]:
    f = query_filters(text)
    cap = pp.budget_cap(f, None)
    out, rep = pp.select_results(
        shown + rest, TOP_K, price_max=cap, interleave_genders=pp.query_gender(f) is None
    )
    return pp.enforce_budget(out, cap), rep, f


def stats(items: list[dict], cap: float | None) -> dict[str, Any]:
    return {
        "n": len(items),
        "dup_pairs": dup_pairs(items),
        "max_per_store": max_per_store(items),
        "over_budget": sum(1 for i in items if cap is not None and not pp.within_budget(i, cap)),
        "genders": dict(Counter(pp.gender_of(i) for i in items)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "evals" / "retrieval" / "results"),
                    help="where the short summary .md goes")
    ap.add_argument("--json-dir", default=str(REPO / "reports" / "plain_search_quality"),
                    help="where the per-query JSON goes (reports/ = generated artifacts)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    consensus = rg.load_consensus(RUN_DIR)
    replay = rg.ReplayEvalCache(EVAL_CACHE)
    rows: list[dict[str, Any]] = []
    for case in rg.load_pools(POOLS):
        qid, text, pool = str(case["id"]), str(case["text"]), case["items"]
        replay.query_id = qid
        shown, rest = rerank_ranked(text, list(pool), replay, top_k=TOP_K, allow_fallback=False)
        new, rep, f = process(text, shown, rest)
        labels = consensus.get(qid, {})
        cap = pp.budget_cap(f, None)
        cur_ids, new_ids = [i["article_id"] for i in shown], [i["article_id"] for i in new]
        rows.append({
            "query_id": qid, "text": text, "n_pool": len(pool), "gender_specified": "gender" in f,
            "price_max": cap,
            "ndcg_cur": ndcg_unlabelled_zero(cur_ids, labels) if labels else float("nan"),
            "ndcg_new": ndcg_unlabelled_zero(new_ids, labels) if labels else float("nan"),
            "unlabelled_cur": sum(1 for a in cur_ids if a not in labels),
            "unlabelled_new": sum(1 for a in new_ids if a not in labels),
            "cur": stats(shown, cap), "new": stats(new, cap), "report": rep,
            "cur_ids": cur_ids, "new_ids": new_ids, "_ranked": shown + rest,
        })

    defined = [r for r in rows if not (math.isnan(r["ndcg_cur"]) or math.isnan(r["ndcg_new"]))]
    excluded = [r["query_id"] for r in rows if r not in defined]
    diffs = [r["ndcg_new"] - r["ndcg_cur"] for r in defined]
    mean, lo, hi = rg.bootstrap_mean_ci(diffs)
    sub = [r["ndcg_new"] - r["ndcg_cur"] for r in defined if not r["gender_specified"]]
    smean, slo, shi = rg.bootstrap_mean_ci(sub)

    def agg(key: str) -> dict[str, Any]:
        rs = [r[key] for r in rows]
        short = sum(1 for r in rows if r[key]["n"] < min(TOP_K, r["n_pool"]))
        return {
            "queries_with_dup": sum(1 for s in rs if s["dup_pairs"] > 0),
            "dup_pairs": sum(s["dup_pairs"] for s in rs),
            "queries_store_cap_exceeded": sum(1 for s in rs if s["max_per_store"] > pp.MAX_PER_STORE),
            "over_budget_cards": sum(s["over_budget"] for s in rs),
            "queries_with_over_budget": sum(1 for s in rs if s["over_budget"] > 0),
            "mean_shown": sum(s["n"] for s in rs) / len(rs),
            "queries_shown_fewer_than_possible": short,
        }

    # Post-hoc diagnostic (NOT part of the pre-registered gate): which rule moves nDCG@5. Each arm
    # = de-duplication plus the named rule(s); same queries, same bootstrap.
    ablation: dict[str, dict[str, float]] = {}
    for label, (store_cap, use_budget, use_mix) in {
        "dedupe_only": (99, False, False),
        "dedupe+store_cap": (pp.MAX_PER_STORE, False, False),
        "dedupe+budget": (99, True, False),
        "dedupe+gender_mix": (99, False, True),
        "full": (pp.MAX_PER_STORE, True, True),
    }.items():
        ds = []
        for r in defined:
            labels = consensus[r["query_id"]]
            f = query_filters(r["text"])
            sel, _ = pp.select_results(
                r["_ranked"], TOP_K, price_max=pp.budget_cap(f, None) if use_budget else None,
                interleave_genders=use_mix and pp.query_gender(f) is None, max_per_store=store_cap,
            )
            ds.append(ndcg_unlabelled_zero([i["article_id"] for i in sel], labels) - r["ndcg_cur"])
        m, a, b = rg.bootstrap_mean_ci(ds)
        ablation[label] = {"diff_mean": m, "ci_lo": a, "ci_hi": b}

    chips: list[dict[str, Any]] = []
    for case in rg.load_pools(CHIPS):
        pool = case["items"]
        out, rep, f = process(str(case["text"]), pool[:TOP_K], pool[TOP_K:])
        cap = pp.budget_cap(f, None)
        chips.append({
            "id": case["id"], "text": case["text"], "gender_specified": "gender" in f, "price_max": cap,
            "cur": stats(pool[:TOP_K], cap), "new": stats(out, cap), "report": rep,
            "new_titles": [f"{i['store']} | {i.get('gender')} | {i.get('price_inr')} | {i.get('prod_name')}"
                           for i in out],
        })

    res = {
        "rule": "evals/retrieval/results/plain_search_gate_rule.md",
        "n_pools": len(rows), "n_defined": len(defined), "excluded": excluded,
        "ndcg_cur_mean": sum(r["ndcg_cur"] for r in defined) / len(defined),
        "ndcg_new_mean": sum(r["ndcg_new"] for r in defined) / len(defined),
        "diff_mean": mean, "diff_ci_lo": lo, "diff_ci_hi": hi,
        "pass": lo >= -0.03,
        "unlabelled_shown_cur": sum(r["unlabelled_cur"] for r in defined),
        "unlabelled_shown_new": sum(r["unlabelled_new"] for r in defined),
        "gender_unspecified_subset": {"n": len(sub), "diff_mean": smean, "ci_lo": slo, "ci_hi": shi},
        "ablation_post_hoc": ablation,
        "pools_current": agg("cur"), "pools_new": agg("new"),
        "n_queries_changed": sum(1 for r in rows if r["cur_ids"] != r["new_ids"]),
        "budget_queries": [r["query_id"] for r in rows if r["price_max"]],
        "chips_current_retrieval_order": {k: sum(c["cur"][k] for c in chips)
                                          for k in ("dup_pairs", "over_budget")},
        "chips_new": {k: sum(c["new"][k] for c in chips) for k in ("dup_pairs", "over_budget")},
    }
    json_dir = Path(args.json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    (json_dir / "plain_search_guard.json").write_text(
        json.dumps({"summary": res, "chips": chips,
                    "rows": [{k: v for k, v in r.items() if k != "_ranked"} for r in rows]},
                   indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plain_search_guard.md").write_text(
        "# Plain-search offline guard (generated by evals/retrieval/plain_search_guard.py)\n\n"
        f"Rule: {res['rule']}. Per-query rows: reports/plain_search_quality/plain_search_guard.json.\n\n"
        f"**{'PASS' if res['pass'] else 'FAIL'}**: mean(new - current) nDCG@5 = {mean:+.4f}, 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}] over n={len(defined)} (bar: lower bound >= -0.03).\n\n"
        "```json\n" + json.dumps(res, indent=1, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(res, indent=1, ensure_ascii=False))
    for c in chips:
        print(c["id"], c["text"], "| cur", c["cur"], "| new", c["new"], "|", c["report"]["removed"])
        for t in c["new_titles"]:
            print("    ", t)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
