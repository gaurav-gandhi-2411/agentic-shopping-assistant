"""Measure joint slot selection against the greedy path (gates 1, 2, 4, 5 of
evals/outfit/joint_selection_gate_rule.md). Retrieval-only: no LLM, no .env, public index view.

    python scripts/outfit_joint_eval.py --index-dir <dir> grid [--greedy] [--json out]
    python scripts/outfit_joint_eval.py --index-dir <dir> latency [--runs 10] [--greedy]
    python scripts/outfit_joint_eval.py --index-dir <dir> rows

``--greedy`` sets ``composer.JOINT_SELECTION_ENABLED = False`` when the flag exists (it does not on
master, which is greedy by construction), so the same script measures both sides.

Relevance (independent of the selector's own objective): for every card, take the top-20
retrieval for that card's slot query (gender filter, price_max = the cell budget; the seed uses the
occasion anchor query, complements use ``get_fill_slots`` of the chosen seed's class) and score it
``rel = retrieval score / top score`` of that list, 0 when absent. ``per_card_rel`` is the mean.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LATENCY_QUERY = "sangeet outfit under ₹3000 for women"
CASUAL_MEN_2500 = "casual outfit under ₹2500 for men"
EVAL_TOP_N = 20


def _rel_in_pool(item_id: str, pool: list[dict]) -> float:
    scored = [(float(p.get("score") or 0.0), str(p["article_id"])) for p in pool]
    if not scored:
        return 0.0
    top = max(s for s, _ in scored)
    for s, aid in scored:
        if aid == item_id:
            return s / top if top > 0 else 0.0
    return 0.0


def look_relevance(look: dict, retriever: Any, occasion: str, gender: str, budget: float) -> dict:
    """Per-card relevance of one composed look (see module docstring)."""
    from src.agents.outfit.composer import _anchor_query_for_occasion
    from src.agents.outfit.slots import (
        classify_anchor,
        get_fill_slots,
        split_accessory_query_by_family,
    )

    seed = look.get("seed_item")
    if not seed:
        return {"rels": [], "per_card_rel": None, "sum_rel": 0.0}
    filters = {"gender": gender, "price_max": float(budget)}
    anchor_pool = retriever.search(
        _anchor_query_for_occasion(occasion, gender), top_k=EVAL_TOP_N, filters=filters
    )
    rels = [_rel_in_pool(str(seed["article_id"]), anchor_pool)]
    a_class = classify_anchor(seed.get("product_type") or "", seed.get("prod_name") or "")
    specs = {s.slot_name: s for s in get_fill_slots(a_class, gender, occasion)}
    for c in look.get("complements") or []:
        spec = specs.get(c.get("_slot"))
        if spec is None:
            rels.append(0.0)
            continue
        if spec.slot_name == "accessory":
            queries = split_accessory_query_by_family(spec.search_query) or [spec.search_query]
        else:
            queries = [spec.search_query]
        rels.append(
            max(
                _rel_in_pool(
                    str(c["article_id"]), retriever.search(q, top_k=EVAL_TOP_N, filters=filters)
                )
                for q in queries
            )
        )
    return {"rels": rels, "per_card_rel": statistics.fmean(rels), "sum_rel": sum(rels)}


def run_grid(retriever: Any, df: Any) -> list[dict]:
    """The 36-cell budget grid of scripts/outfit_budget_grid.py plus per-cell relevance."""
    from scripts import outfit_budget_grid as grid
    from src.agents.outfit.composer import compose_outfit

    rows = []
    for gender in grid.GENDERS:
        for occasion in grid.OCCASIONS:
            for budget in grid.BUDGETS:
                look = compose_outfit(
                    df, retriever, occasion_slug=occasion, gender=gender, budget_inr=float(budget)
                )
                seed = look.get("seed_item")
                comps = look.get("complements") or []
                row = {
                    "gender": gender,
                    "occasion": occasion,
                    "budget": budget,
                    "cards": (1 if seed else 0) + len(comps),
                    "total": look.get("budget_total_inr"),
                    "slots": [c.get("_slot") for c in comps],
                    "empty_slots": list(look.get("empty_slots") or []),
                    "ids": ([str(seed["article_id"])] if seed else [])
                    + [str(c["article_id"]) for c in comps],
                }
                row.update(look_relevance(look, retriever, occasion, gender, float(budget)))
                rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True, type=Path)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("mode", choices=["grid", "latency", "rows"])
    args = ap.parse_args()
    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        sys.exit("INCLUDE_NON_PUBLIC_STORES is set; this eval is the public view only")

    from scripts import outfit_compose_snapshot as snap
    from src.agents.outfit import composer

    has_flag = hasattr(composer, "JOINT_SELECTION_ENABLED")
    if args.greedy and has_flag:
        composer.JOINT_SELECTION_ENABLED = False
    side = "master(no flag)" if not has_flag else ("greedy(flag off)" if args.greedy else "joint")
    retriever, df = snap.load_retriever(args.index_dir)
    snap.compose_turn("casual outfit for women", df, retriever)  # warm-up

    if args.mode == "grid":
        rows = run_grid(retriever, df)
        ok = sum(1 for r in rows if r["cards"] >= 3)
        over = [r for r in rows if r["total"] is not None and r["total"] > r["budget"]]
        print(f"[{side}] cells>=3 cards: {ok}/{len(rows)}; over-budget cells: {len(over)}")
        for r in rows:
            rel = r["per_card_rel"]
            print(
                f"  {r['gender']:5} {r['occasion']:14} {r['budget']:>5}: cards={r['cards']} "
                f"total={r['total']} rel={None if rel is None else round(rel, 3)} "
                f"slots={r['slots']} empty={r['empty_slots']}"
            )
        turn = snap.compose_turn(CASUAL_MEN_2500, df, retriever)
        base = turn["variants"][0] if turn["variants"] else {}
        n_cards = (1 if base.get("seed_item") else 0) + len(base.get("complements") or [])
        print(f"[{side}] {CASUAL_MEN_2500!r}: Base look cards={n_cards}")
        if args.json:
            args.json.write_text(
                json.dumps({"side": side, "rows": rows, "casual_men_2500_cards": n_cards}), "utf-8"
            )
    elif args.mode == "latency":
        runs = []
        for _ in range(args.runs):
            # Cold query-embedding memo every run: repeating one query ten times would otherwise
            # serve every encode from the memo, which a real turn (new text) never gets; only the
            # within-turn reuse (probe + variants share query text) should count.
            getattr(retriever.dense, "_query_vec_cache", {}).clear()
            t0 = time.perf_counter()
            snap.compose_turn(LATENCY_QUERY, df, retriever)
            runs.append((time.perf_counter() - t0) * 1000.0)
        print(
            f"[{side}] {LATENCY_QUERY!r} n={len(runs)} median_ms={statistics.median(runs):.0f} "
            f"min={min(runs):.0f} max={max(runs):.0f} runs={[round(x) for x in runs]}"
        )
    else:
        from scripts import outfit_more_options_grid as rowgrid

        rows = rowgrid.run_grid(retriever, df)
        s = rowgrid.summarise(rows)
        print(f"[{side}] cells={s['cells']} cards={s['cards']} audit_counts={s['audit_counts']}")


if __name__ == "__main__":
    main()
