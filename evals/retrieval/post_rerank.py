"""Replay of `search_node`'s post-rerank tail, to measure what the user actually SEES (card count).

After `rerank()` returns its top 5, `src/agents/graph.py::search_node` runs a chain of steps that
can drop cards without refilling them (the occasion-register gate, the attribute / theme /
accessory / loungewear / merchandise / athletic-footwear / haldi gates, the price qualifier, the
size-mismatch demotion). That is how the same query can show 5 cards on one run and 4 on the next
when the stochastic reranker picks a different item (the production blazer 4-vs-5 case).

Reproduced here with the SAME functions production calls (imported, not reimplemented), in the
same order as `search_node` (graph.py, "items_out = deduped" through
`demote_size_mismatched_items`):

  reproduced:     name+colour dedup with backfill from the pool, the beach per-type cap,
                  attribute-contradiction, theme-contradiction, accessory-exclusion, the
                  loungewear strip, the occasion coherence gate (+ loungewear, merchandise,
                  athletic-footwear and haldi-colour gates), formality-softener re-sort, the
                  price qualifier and the size-mismatch demotion.
  NOT reproduced: everything BEFORE rerank() (frozen candidates already include it), the
                  colour-chip / refinement bookkeeping (no effect on cards), the response and
                  rationale nodes, session/ownership logic, and any live-state input such as
                  `excluded_colours` (fixtures are single-turn).

The query text is parsed with the production `parse_intent`; gender for the occasion gate is
the fixture's gender ('unisex' for gender-neutral queries, as search_node derives it).
"""

from __future__ import annotations

from typing import Any

TOP_K = 5


def apply_post_rerank(
    query: str,
    shown: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    gender: str,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """The cards the user would see for `shown` (rerank output) given the candidate `pool`."""
    from src.agents.graph import (
        _BEACH_SUMMER_RE,
        _GENERIC_WEAR_ASK_RE,
        _LOUNGEWEAR_GATE_OCCASIONS,
        _apply_athletic_footwear_gate,
        _apply_haldi_colour_gate,
        _apply_loungewear_gate,
        _apply_occasion_merchandise_gate,
        _apply_price_qualifier,
    )
    from src.agents.intent_parser import parse_intent
    from src.agents.outfit.body_type import demote_size_mismatched_items
    from src.agents.outfit.coherence import is_coherent_candidate
    from src.agents.outfit.slots import (
        FORMALITY_SOFTENER_VALUES,
        NON_OUTFIT_ITEM_CLASSES,
        classify_item,
        fabric_score_delta,
        is_attribute_contradiction,
        is_theme_contradiction,
    )
    from src.catalogue.cleaning import is_loungewear_text
    from src.retrieval.hybrid_search import normalize_prod_name

    def name(it: dict[str, Any]) -> str:
        return str(it.get("prod_name") or it.get("display_name") or "")

    def key(it: dict[str, Any]) -> tuple[str, str]:
        return (
            normalize_prod_name(it.get("prod_name", it["display_name"])),
            (it.get("colour") or "").lower(),
        )

    # name+colour dedup with backfill (search_node: "items_out = deduped")
    seen: set[tuple[str, str]] = set()
    items: list[dict[str, Any]] = []
    for it in shown:
        if key(it) not in seen:
            seen.add(key(it))
            items.append(it)
    if len(items) < top_k:
        ids = {it["article_id"] for it in items}
        for it in pool:
            if len(items) >= top_k:
                break
            if it["article_id"] not in ids and key(it) not in seen:
                seen.add(key(it))
                items.append(it)

    if _BEACH_SUMMER_RE.search(query):  # at most 2 per product type, refilled from the pool
        diverse: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for it in items:
            pt = (it.get("product_type") or "").lower()
            if counts.get(pt, 0) < 2:
                diverse.append(it)
                counts[pt] = counts.get(pt, 0) + 1
        if len(diverse) < top_k:
            have = {it["article_id"] for it in diverse}
            for it in pool:
                if len(diverse) >= top_k:
                    break
                pt = (it.get("product_type") or "").lower()
                if it["article_id"] not in have and counts.get(pt, 0) < 2:
                    diverse.append(it)
                    counts[pt] = counts.get(pt, 0) + 1
                    have.add(it["article_id"])
        items = diverse[:top_k]

    intent = parse_intent(query)
    slug = intent.occasion

    def keep_unless_empty(
        cur: list[dict[str, Any]], kept: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return kept if kept else cur  # pool-underflow protection used by most gates

    if items:
        items = keep_unless_empty(
            items,
            [
                it
                for it in items
                if not is_attribute_contradiction(query, name(it), it.get("detail_desc") or "")
            ],
        )
    if items:
        items = keep_unless_empty(
            items,
            [
                it
                for it in items
                if not is_theme_contradiction(query, name(it), it.get("product_type") or "")
            ],
        )
    if not intent.garment_type and _GENERIC_WEAR_ASK_RE.search(query) and items:
        items = keep_unless_empty(
            items,
            [
                it
                for it in items
                if classify_item(it.get("product_type") or "", name(it))
                not in NON_OUTFIT_ITEM_CLASSES
            ],
        )
    if slug not in _LOUNGEWEAR_GATE_OCCASIONS and items and not is_loungewear_text(query):
        items = keep_unless_empty(items, [it for it in items if not is_loungewear_text(name(it))])
    if slug and slug != "casual" and items:
        items = keep_unless_empty(
            items, [it for it in items if is_coherent_candidate(it, slug, gender, "top")]
        )
        items = _apply_loungewear_gate(items, slug)
        items = _apply_occasion_merchandise_gate(items, slug, intent.garment_type, query)
        items = _apply_athletic_footwear_gate(items, slug, intent.garment_type)
        items = _apply_haldi_colour_gate(items, slug)
    if items and intent.formality_softener in FORMALITY_SOFTENER_VALUES:
        items = sorted(
            items,
            key=lambda it: fabric_score_delta(
                it, slug or "", formality_override=intent.formality_softener
            ),
            reverse=True,
        )
    items = _apply_price_qualifier(items, intent.price_qualifier)
    return list(demote_size_mismatched_items(items, query))
