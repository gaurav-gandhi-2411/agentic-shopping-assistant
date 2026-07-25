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
  --mode pipeline  additionally applies the SAME garment_type facet filter,
                    occasion register gate/rerank, loungewear/merchandise/
                    athletic-footwear gates, formality-softener re-sort,
                    price-qualifier, and shape!=size demotion the live
                    search_node applies (reusing intent_parser.parse_intent,
                    coherence.is_coherent_candidate, graph._apply_* gates,
                    body_type.demote_size_mismatched_items directly — not
                    reimplemented) — this is what production actually
                    returns for these queries. See _retrieve_pipeline's
                    docstring for the one disclosed, deliberate gap (the
                    pre-rerank LLM-adjacent hard filter).
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
    garment set exclusion, loungewear/merchandise/athletic-footwear gates,
    the formality-softener re-sort, price-qualifier, and shape!=size
    demotion — is unconditional in both search_node and this mirror, matching
    production regardless of the flag. Use both values to isolate the
    occasion gate's specific contribution.

    2026-07-24 gate-parity audit: brought to genuine full parity with
    search_node's occasion-gated block AND its post-rerank tail (loungewear,
    merchandise, athletic-footwear, formality-softener, price-qualifier,
    shape!=size — see each call site below for provenance). One deliberate,
    disclosed gap remains: search_node also applies price_qualifier and a
    formality_softener HARD FILTER on the WIDE pre-rerank candidate pool
    (graph.py's rerank() call site) before its LLM reranker ever runs — this
    mirror has no LLM rerank step at all (by design: --mode pipeline stays
    deterministic and free), so that pre-rerank stage cannot be mirrored
    without reintroducing an LLM call into the eval. The post-rerank
    re-application of both (added here) is the closest deterministic
    equivalent and matches what a query WITHOUT the pre-rerank exclusion
    would still converge to on the top-50 pool used here.

    cross_encoder: an optional loaded sentence_transformers.CrossEncoder —
    when given, reorders the FULL post-gate candidate pool by cross-encoder
    relevance score as the final step (Part 1b A/B). Reordering only, never
    introduces new items — an honest A/B against the existing hand labels
    needs no new labeling, unlike an embedding-model swap.
    """
    from src.agents.graph import (
        _GENERIC_WEAR_ASK_RE,
        _LOUNGEWEAR_GATE_OCCASIONS,
        _OUTFIT_INTENT_RE,
        _SET_INTENT_RE,
        _apply_athletic_footwear_gate,
        _apply_loungewear_gate,
        _apply_occasion_merchandise_gate,
        _apply_price_qualifier,
    )
    from src.agents.intent_parser import parse_intent
    from src.agents.outfit.body_type import demote_size_mismatched_items
    from src.agents.outfit.coherence import is_coherent_candidate
    from src.agents.outfit.slots import (
        FORMALITY_SOFTENER_VALUES,
        classify_item,
        fabric_score_delta,
        is_attribute_contradiction,
        is_multi_piece_set,
    )
    from src.agents.tools import search_catalogue
    from src.catalogue.cleaning import is_kids_item, is_loungewear_text

    intent = parse_intent(query)
    filters: dict = {"gender": gender}
    if intent.garment_type:
        filters["product_type_name"] = intent.garment_type
    if intent.colour:
        filters["colour_group_name"] = intent.colour
    # 2026-07-25 gate-parity fix: this mirror never passed budget_max_inr
    # into the retrieval filter at all, even though search_node always does
    # (see graph.py's "if merged_intent.budget_max_inr: _plan_filters
    # ['price_max'] = ..." right before its own search_catalogue call) —
    # HybridRetriever applies price_max as a HARD exclude at the candidate-
    # building layer (src/retrieval/hybrid_search.py), not a soft rerank
    # signal, so this was a genuine, large eval-mirror gap: any "under Rs X"
    # query scored a pipeline where over-budget items were never excluded
    # from the pool at all. Confirmed via direct A/B (bandhgala under
    # Rs10000): without this filter, top-5 prices were
    # [26000, 383999, 6499, 28995, 8999]; with it, [6499, 8999, 7499, 9995,
    # 5499] -- all in budget.
    if intent.budget_max_inr:
        filters["price_max"] = intent.budget_max_inr

    # search_catalogue (not retriever.search directly) — it's the actual
    # production retrieval boundary and applies the colour-family widening
    # (colour_filter_values) internally, so this mirror can never silently
    # drift from what search_node does.
    items = search_catalogue(query, filters, retriever, 50)["items"]

    # Kids-item strip — UNCONDITIONAL (not tied to occasion_gate), matching
    # search_node exactly (2026-07-12 fix): applied immediately after retrieval,
    # before the occasion gate below, so this mirror stays honest for
    # non-occasion-keyword queries too (e.g. "red lehenga bridal").
    items = [
        it for it in items
        if not is_kids_item(it.get("prod_name") or it.get("display_name") or "")
    ]

    # Single-garment set exclusion — unconditional (not tied to occasion_gate),
    # matching search_node exactly: skipped when the query itself legitimizes
    # a multi-piece result (set/combo/co-ord/outfit/look words), or when the
    # query named a genuine second garment (garment_type_secondary) — a combo
    # listing naming both requested garments is a legitimate hit, not
    # SET-listing noise (search_node's 2026-07-24 "joggers and t-shirt" fix).
    if intent.garment_type and items and not (
        _SET_INTENT_RE.search(query)
        or _OUTFIT_INTENT_RE.search(query)
        or intent.garment_type_secondary
    ):
        set_filtered = [
            it for it in items
            if not is_multi_piece_set(
                it.get("product_type") or "", it.get("prod_name") or it.get("display_name") or ""
            )
        ]
        if set_filtered:
            items = set_filtered

    # Attribute-contradiction gate — unconditional (not tied to occasion_gate),
    # matching search_node exactly (2026-07-25 fix, added in the same commit
    # as the production gate rather than as a follow-up gap): strips
    # candidates whose own name/desc explicitly states a fit/rise/breasted/
    # silhouette/neckline word that opposes a word the query itself stated.
    if items:
        attr_filtered = [
            it for it in items
            if not is_attribute_contradiction(
                query,
                it.get("prod_name") or it.get("display_name") or "",
                it.get("detail_desc") or "",
            )
        ]
        if attr_filtered:
            items = attr_filtered

    # Accessory-exclusion gate — unconditional (not tied to occasion_gate),
    # matching search_node exactly (2026-07-25 fix, "type-confusion" strict-
    # eval bucket): a generic "outfit/look/wear" ask with no garment_type
    # resolved must never surface a standalone accessory (bag/dupatta/
    # jewellery) as an "outfit" result.
    if not intent.garment_type and _GENERIC_WEAR_ASK_RE.search(query) and items:
        acc_filtered = [
            it for it in items
            if classify_item(
                it.get("product_type") or "", it.get("prod_name") or it.get("display_name") or ""
            ) != "accessory"
        ]
        if acc_filtered:
            items = acc_filtered

    occasion_slug = intent.occasion

    # Loungewear strip for the NO-EXISTING-PROTECTION case only — matching
    # search_node exactly (2026-07-25 fix, "occasion-register" strict-eval
    # bucket): a non-occasion query ("black dress for my wife") had zero
    # sleepwear protection before this fix. Deliberately does NOT run when
    # occasion_slug is in _LOUNGEWEAR_GATE_OCCASIONS — that case is already
    # correctly handled by _apply_loungewear_gate further below (see
    # search_node's own comment for why running this EARLIER, pool-
    # underflow-protected strip in that case is actively wrong). Exempts
    # queries that themselves explicitly ask for a night dress/nightgown.
    if (
        occasion_slug not in _LOUNGEWEAR_GATE_OCCASIONS
        and items
        and not is_loungewear_text(query)
    ):
        lounge_filtered = [
            it for it in items
            if not is_loungewear_text(it.get("prod_name") or it.get("display_name") or "")
        ]
        if lounge_filtered:
            items = lounge_filtered

    # Occasion coherence gate + fabric rerank — toggled by occasion_gate (the
    # A/B flag isolating the 2026-07-11 fix's own contribution). Everything
    # below this point is unconditional in production regardless of that
    # flag, so it stays outside the toggle here too.
    if occasion_gate and occasion_slug and occasion_slug != "casual":
        gated = [
            it for it in items
            if is_coherent_candidate(it, occasion_slug, gender, _NEUTRAL_SLOT)
        ]
        if gated:  # never let the gate empty the pool (same discipline as composer)
            items = gated
        items = sorted(items, key=lambda it: fabric_score_delta(it, occasion_slug), reverse=True)

    # 2026-07-24 gate-parity fix: this mirror previously applied only the
    # occasion-merchandise gate (added 2026-07-23) and silently never called
    # search_node's loungewear, athletic-footwear, or price-qualifier gates —
    # meaning strict eval scored a DIFFERENT pipeline than production despite
    # the docstring's "mirrors production exactly" claim. Order now matches
    # search_node's occasion-gated block exactly: loungewear -> merchandise ->
    # athletic-footwear, all unconditional once an occasion is resolved (none
    # of the three are pool-underflow protected in production, so order can
    # change which items survive — see each gate's own docstring in graph.py).
    if occasion_slug and occasion_slug != "casual":
        items = _apply_loungewear_gate(items, occasion_slug)
        items = _apply_occasion_merchandise_gate(items, occasion_slug, intent.garment_type, query)
        items = _apply_athletic_footwear_gate(items, occasion_slug, intent.garment_type)

    # Formality-softener secondary re-sort — unconditional on occasion in
    # production (see _apply_formality_softener's docstring: the occasion
    # gate was never a real requirement of the underlying function).
    if items and intent.formality_softener in FORMALITY_SOFTENER_VALUES:
        items = sorted(
            items,
            key=lambda it: fabric_score_delta(
                it, occasion_slug or "", formality_override=intent.formality_softener
            ),
            reverse=True,
        )

    # Price-qualifier filter/sort — unconditional on occasion, see
    # _apply_price_qualifier docstring ("cheap lehenga" outlier exclusion).
    items = _apply_price_qualifier(items, intent.price_qualifier)

    # Shape != size demotion — a "pear shaped" query must not headline
    # explicitly Plus-Size-branded items; see demote_size_mismatched_items.
    items = demote_size_mismatched_items(items, query)

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
    parser.add_argument("--queries-path", default=str(_QUERIES_PATH),
                        help="alternate query fixture — e.g. an out-of-sample/held-out set "
                             "scored cold to check whether a fix generalizes beyond the "
                             "queries used to develop it, not just the default gold set")
    parser.add_argument("--labels-path", default=str(_LABELS_PATH),
                        help="alternate label fixture, paired with --queries-path")
    args = parser.parse_args()

    from eval_model import _build_components  # heavy import deferred past --help

    queries_path = Path(args.queries_path)
    labels_path = Path(args.labels_path)
    queries = yaml.safe_load(queries_path.read_text(encoding="utf-8"))["queries"]
    labels_raw = yaml.safe_load(labels_path.read_text(encoding="utf-8"))["labels"]
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
          f"(rubric: {queries_path.name})")
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
