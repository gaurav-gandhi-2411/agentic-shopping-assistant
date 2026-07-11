#!/usr/bin/env python
"""Strict gold-relevance eval — human-audited precision@5, no self-grading.

Re-runs retrieval for eval/fixtures/strict_gold_queries.yaml and scores the
top-5 against the HAND labels in strict_gold_labels.yaml, printing strict
precision@5 (overall + per-category) plus a miss taxonomy split into
CODE-FIXABLE vs DATA-CEILING/DATA-QUALITY causes.

TWO MODES, both label-compatible (relevance is about the ITEM, not the query
pipeline that surfaced it):
  --mode raw       (default) retriever.search(query, filters={gender}) only —
                    mirrors eval_model.py's R1 stage exactly. This is the
                    unfiltered retrieval FLOOR, not what users see.
  --mode pipeline  additionally applies the SAME garment_type facet filter and
                    occasion register gate/rerank the live search_node applies
                    (reusing intent_parser.parse_intent, coherence.
                    is_coherent_candidate, slots.fabric_score_delta directly —
                    not reimplemented) — this is what production actually
                    returns for these queries.
Report both when diagnosing whether a fix reached production; report --mode
pipeline alone when citing "real" user-facing precision.

An item retrieval returns that has NO label is counted separately as
`unlabeled` and NEVER scored — the checker must never grade itself. A run with
unlabeled items means retrieval changed since labeling: re-audit those items,
extend the label file, re-run.

Usage:
    python scripts/eval_strict.py
    python scripts/eval_strict.py --mode pipeline
    python scripts/eval_strict.py --mode pipeline --data-dir data/processed/unified
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

_SCRIPTS_DIR = Path(__file__).parent
_ROOT = _SCRIPTS_DIR.parent
for p in (str(_ROOT), str(_SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

_QUERIES_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_queries.yaml"
_LABELS_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_labels.yaml"

# Which miss reasons a code change can remove vs what only data/inventory can.
CODE_FIXABLE_REASONS = frozenset({
    "type-confusion", "set-not-single", "kids-leak", "budget",
    "occasion-register", "attribute-contradiction", "colour-family",
})
DATA_REASONS = frozenset({"data-ceiling", "data-mislabel"})

# Neutral slot name for is_coherent_candidate: only its dupatta-specific gate
# (slot_name == "accessory") is slot-dependent — every ethnic/western/office
# register gate is slot-agnostic, so "top" exercises them without ever
# tripping the accessory-only gate. See coherence.is_coherent_candidate.
_NEUTRAL_SLOT = "top"

# 2026-07-11 cross-encoder reranker A/B (Part 1b): a well-established, free,
# self-hostable BEIR/MS-MARCO cross-encoder (~22M params, fast CPU inference).
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _cross_encoder_candidate_text(item: dict) -> str:
    colour = item.get("colour") or ""
    product_type = item.get("product_type") or ""
    name = item.get("prod_name") or item.get("display_name") or ""
    return f"{name}, {colour} {product_type}".strip()


def cross_encoder_rerank(model, query: str, items: list[dict]) -> list[dict]:
    """Rerank an ALREADY-RETRIEVED candidate pool by cross-encoder relevance
    score. Reorders only — never introduces new items, so this needs NO new
    hand labels for an honest A/B against the existing strict gold labels
    (unlike an embedding-model swap, which changes the retrieved SET itself).
    """
    if not items:
        return items
    pairs = [(query, _cross_encoder_candidate_text(it)) for it in items]
    scores = model.predict(pairs)
    order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
    return [items[i] for i in order]


def _retrieve_pipeline(
    retriever, query: str, gender: str, *, occasion_gate: bool, cross_encoder=None
) -> list[dict]:
    """Mirror search_node's production filter(+gate+rerank) exactly (same
    functions, not reimplemented) so this mode reports real user-facing order.

    occasion_gate toggles ONLY the is_coherent_candidate register gate +
    fabric_score_delta rerank (the 2026-07-11 occasion-gate fix). Every other
    mechanism here — garment_type/colour_group_name filters, the single-
    garment set exclusion — is unconditional in both search_node and this
    mirror, matching production regardless of the flag. Use both values to
    isolate the occasion gate's specific contribution.

    cross_encoder: an optional loaded sentence_transformers.CrossEncoder —
    when given, reorders the FULL post-gate candidate pool by cross-encoder
    relevance score as the final step (Part 1b A/B). Reordering only, never
    introduces new items — an honest A/B against the existing hand labels
    needs no new labeling, unlike an embedding-model swap.
    """
    from src.agents.graph import _OUTFIT_INTENT_RE, _SET_INTENT_RE
    from src.agents.intent_parser import parse_intent
    from src.agents.outfit.coherence import is_coherent_candidate
    from src.agents.outfit.slots import fabric_score_delta, is_kids_item, is_multi_piece_set
    from src.agents.tools import search_catalogue

    intent = parse_intent(query)
    filters: dict = {"gender": gender}
    if intent.garment_type:
        filters["product_type_name"] = intent.garment_type
    if intent.colour:
        filters["colour_group_name"] = intent.colour

    # search_catalogue (not retriever.search directly) — it's the actual
    # production retrieval boundary and applies the colour-family widening
    # (colour_filter_values) internally, so this mirror can never silently
    # drift from what search_node does.
    items = search_catalogue(query, filters, retriever, 50)["items"]

    # Single-garment set exclusion — unconditional (not tied to occasion_gate),
    # matching search_node exactly: skipped when the query itself legitimizes
    # a multi-piece result (set/combo/co-ord/outfit/look words).
    if intent.garment_type and items and not (
        _SET_INTENT_RE.search(query) or _OUTFIT_INTENT_RE.search(query)
    ):
        set_filtered = [
            it for it in items
            if not is_multi_piece_set(
                it.get("product_type") or "", it.get("prod_name") or it.get("display_name") or ""
            )
        ]
        if set_filtered:
            items = set_filtered

    occasion_slug = intent.occasion
    if occasion_gate and occasion_slug and occasion_slug != "casual":
        gated = [
            it for it in items
            if is_coherent_candidate(it, occasion_slug, gender, _NEUTRAL_SLOT)
            and not is_kids_item(it.get("prod_name") or it.get("display_name") or "")
        ]
        if gated:  # never let the gate empty the pool (same discipline as composer)
            items = gated
        items = sorted(items, key=lambda it: fabric_score_delta(it, occasion_slug), reverse=True)

    if cross_encoder is not None:
        items = cross_encoder_rerank(cross_encoder, query, items)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=str(_ROOT / "data" / "processed" / "unified"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=("raw", "pipeline"), default="raw",
                        help="raw = unfiltered retrieval floor; "
                             "pipeline = mirrors production's filter(+gate+rerank)")
    parser.add_argument("--no-occasion-gate", action="store_true",
                        help="pipeline mode only: mirror production BEFORE the "
                             "2026-07-11 occasion-gate fix (type filter only)")
    parser.add_argument("--json-out", default=None,
                        help="Write a {precision_at_5, n_scored, n_unlabeled} summary here "
                             "(consumed by scripts/eval_gate.py)")
    parser.add_argument("--cross-encoder", action="store_true",
                        help="pipeline mode only: rerank the post-gate candidate pool with "
                             f"--cross-encoder-model (default {_CROSS_ENCODER_MODEL}), "
                             "reordering only")
    parser.add_argument("--cross-encoder-model", default=_CROSS_ENCODER_MODEL,
                        help="sentence-transformers CrossEncoder model name to use "
                             "when --cross-encoder is set (swap candidates without code changes)")
    args = parser.parse_args()

    from eval_model import _build_components  # heavy import deferred past --help

    queries = yaml.safe_load(_QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    labels_raw = yaml.safe_load(_LABELS_PATH.read_text(encoding="utf-8"))["labels"]
    labels: dict[tuple[str, str], dict] = {
        (entry["query_id"], str(item["article_id"])): item
        for entry in labels_raw
        for item in entry["items"]
    }

    comps = _build_components(need_agent=False, data_dir=Path(args.data_dir))
    retriever = comps["retriever"]

    cross_encoder = None
    if args.cross_encoder:
        from sentence_transformers import CrossEncoder
        print(f"loading cross-encoder {args.cross_encoder_model}...")
        cross_encoder = CrossEncoder(args.cross_encoder_model, device="cpu")

    n_scored = n_relevant = n_unlabeled = 0
    reasons: Counter[str] = Counter()
    per_query: list[tuple[str, str, str, int, int, int]] = []
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # [rel, miss, unl]
    mrr_scores: list[float] = []
    mrr_by_category: dict[str, list[float]] = defaultdict(list)
    mrr_excluded = 0

    for q in queries:
        if args.mode == "pipeline":
            items = _retrieve_pipeline(
                retriever, q["query"], q["gender"], occasion_gate=not args.no_occasion_gate,
                cross_encoder=cross_encoder,
            )
        else:
            items = retriever.search(q["query"], top_k=50, filters={"gender": q["gender"]})
        top = items[: args.top_k]
        rel = miss = unl = 0
        reciprocal_rank = 0.0
        first_relevant_rank: int | None = None
        any_unlabeled_in_top = False
        for rank, it in enumerate(top, start=1):
            key = (q["id"], str(it.get("article_id")))
            label = labels.get(key)
            if label is None:
                unl += 1
                any_unlabeled_in_top = True
                continue
            if label["relevant"]:
                rel += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank
            else:
                miss += 1
                reasons[label.get("reason", "unspecified")] += 1
        n_scored += rel + miss
        n_relevant += rel
        n_unlabeled += unl
        cat = q.get("category", "uncategorized")
        per_query.append((q["id"], cat, q["query"], rel, miss, unl))
        by_category[cat][0] += rel
        by_category[cat][1] += miss
        by_category[cat][2] += unl
        # MRR: skip queries with an unlabeled item ranked ABOVE the first labeled
        # relevant hit (or unlabeled anywhere, if no relevant hit was found) — an
        # unlabeled item's true relevance is unknown, so its rank position can't be
        # honestly scored either way. Queries with zero retrieved items correctly
        # score reciprocal-rank 0 (MRR is defined over "found nothing").
        unlabeled_before_first_hit = any_unlabeled_in_top and (
            first_relevant_rank is None
            or any(
                labels.get((q["id"], str(it.get("article_id")))) is None
                for it in top[: first_relevant_rank - 1]
            )
        )
        if unlabeled_before_first_hit:
            mrr_excluded += 1
            continue
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        mrr_scores.append(reciprocal_rank)
        mrr_by_category[cat].append(reciprocal_rank)

    _mode_label = args.mode
    if args.mode == "pipeline":
        _mode_label += "-no-occasion-gate" if args.no_occasion_gate else "-occasion-gated"
        if args.cross_encoder:
            _mode_label += "-cross-encoder"
    print(f"\nSTRICT GOLD EVAL [{_mode_label}] — hand-audited relevance "
          f"(rubric: {_QUERIES_PATH.name})")
    print(f"queries={len(queries)}  scored_items={n_scored}  unlabeled_items={n_unlabeled}")
    if n_unlabeled:
        print("  ** UNLABELED ITEMS PRESENT — retrieval changed since labeling. **")
        print("  ** Numbers below cover labeled items only; re-audit before comparing. **")
    p5 = n_relevant / n_scored if n_scored else 0.0
    print(f"\nstrict precision@{args.top_k} (overall): {p5:.3f}  ({n_relevant}/{n_scored})")

    print("\nper-category precision@{}:".format(args.top_k))
    for cat, (rel, miss, unl) in sorted(by_category.items()):
        cat_scored = rel + miss
        cat_p5 = rel / cat_scored if cat_scored else 0.0
        unl_note = f"  (+{unl} unlabeled)" if unl else ""
        print(f"  {cat:16s} {cat_p5:.3f}  ({rel}/{cat_scored}){unl_note}")

    mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    excl_note = f"  ({mrr_excluded} queries excluded — unlabeled item ranked above first hit)" if mrr_excluded else ""
    print(f"\nMRR@{args.top_k} (overall): {mrr:.3f}  (n={len(mrr_scores)}){excl_note}")
    print(f"per-category MRR@{args.top_k}:")
    for cat, scores in sorted(mrr_by_category.items()):
        cat_mrr = sum(scores) / len(scores) if scores else 0.0
        print(f"  {cat:16s} {cat_mrr:.3f}  (n={len(scores)})")

    code_misses = sum(c for r, c in reasons.items() if r in CODE_FIXABLE_REASONS)
    data_misses = sum(c for r, c in reasons.items() if r in DATA_REASONS)
    print("\nmiss taxonomy:")
    for reason, count in reasons.most_common():
        bucket = ("CODE-FIXABLE" if reason in CODE_FIXABLE_REASONS
                  else "DATA" if reason in DATA_REASONS else "?")
        print(f"  {count:3d}  {reason:24s} [{bucket}]")
    if n_scored:
        ceiling_p5 = (n_relevant + code_misses) / n_scored
        print(f"\nif every CODE-FIXABLE miss were fixed: {ceiling_p5:.3f} "
              f"(remaining gap = {data_misses} data-capped items)")

    print("\nweakest queries:")
    for qid, cat, text, rel, miss, unl in sorted(per_query, key=lambda t: t[3]):
        if miss + unl == 0:
            continue
        print(f"  {qid:14s} [{cat:16s}] {rel}/{rel + miss + unl}  {text!r}")

    if args.json_out:
        import json
        Path(args.json_out).write_text(json.dumps({
            "mode": _mode_label,
            "precision_at_5": p5,
            "n_relevant": n_relevant,
            "n_scored": n_scored,
            "n_unlabeled": n_unlabeled,
            "by_category": {
                cat: {"relevant": v[0], "miss": v[1], "unlabeled": v[2]}
                for cat, v in by_category.items()
            },
            "mrr_at_k": mrr,
            "mrr_n": len(mrr_scores),
            "mrr_excluded": mrr_excluded,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
