from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import pandas as pd

from src.agents.outfit.coherence import colour_score, is_coherent_candidate
from src.agents.outfit.occasions import ETHNIC_HEAVY, ETHNIC_ONLY
from src.agents.outfit.slots import (
    classify_anchor,
    fabric_score_delta,
    get_fill_slots,
    is_ethnic_item,
)
from src.retrieval.hybrid_search import HybridRetriever, normalize_prod_name

logger = logging.getLogger(__name__)

# Transparent flywheel boost constant — documented here and in ADR.
# When >= FLYWHEEL_MIN_SIGNALS pairings observed, score × (1 + FLYWHEEL_ALPHA × positive_rate).
# Max +25% boost from conversion signal; 0 at cold-start.
FLYWHEEL_ALPHA: float = 0.25
FLYWHEEL_MIN_SIGNALS: int = 10


@dataclass
class PairingStat:
    """Conversion signal counts for one (anchor_class, fill_slot, occasion) triple.

    Injected by the flywheel phase; zero-valued at cold-start.
    """

    add_the_look: int = 0
    thumbs_up: int = 0
    thumbs_down: int = 0
    add_single_only: int = 0


def _infer_gender(item: dict, brand_gender_default: str) -> str:
    """Infer gender from item's index_group_name field, then brand default."""
    ig = (item.get("index_group_name") or item.get("department") or "").lower()
    if "menswear" in ig or "men" in ig:
        return "men"
    if "ladieswear" in ig or "women" in ig or "ladies" in ig:
        return "women"
    return brand_gender_default or "women"


def compose_outfit(
    catalogue_df: pd.DataFrame,
    retriever: HybridRetriever,
    *,
    seed_article_id: str | None = None,
    occasion_slug: str = "casual",
    gender: str = "women",
    budget_inr: float | None = None,
    pairing_stats: dict | None = None,  # {(anchor_cat, fill_cat, occasion): PairingStat}
    brand_gender_default: str = "women",
) -> dict:
    """Compose an outfit for a given occasion, optionally anchored to a seed item.

    When seed_article_id is None, finds the best anchor from the catalogue for
    the occasion using a retrieval query, then fills slots around it.

    Returns a dict with: look_id, seed_item, complements, outfit_rationale,
    empty_slots, occasion, gender, budget_total_inr.
    """
    look_id = str(uuid.uuid4())

    # ── Step 1: resolve seed item ───────────────────────────────────────────
    if seed_article_id:
        indexed = catalogue_df.set_index("article_id")
        if seed_article_id not in indexed.index:
            return _empty_result(look_id, occasion_slug, gender, "Item not found.")
        row = indexed.loc[seed_article_id]
        facets = row["facets"] if isinstance(row.get("facets"), dict) else {}
        seed_item = _row_to_item(seed_article_id, row, facets, role="seed")
    else:
        # Occasion-driven entry: retrieve an anchor appropriate for the occasion
        anchor_query = _anchor_query_for_occasion(occasion_slug, gender)
        candidates = retriever.search(anchor_query, top_k=10)
        # Filter by coherence (anchor must match occasion ethnic_lean)
        valid = [
            c for c in candidates
            if _anchor_matches_occasion(c, occasion_slug)
        ]
        if not valid:
            return _empty_result(look_id, occasion_slug, gender, "No suitable anchor found.")
        seed_item = valid[0]
        seed_item["_role"] = "seed"
        seed_article_id = seed_item["article_id"]

    # Resolve gender from seed item if not explicitly provided
    effective_gender = (
        gender if gender != "unisex"
        else _infer_gender(seed_item, brand_gender_default)
    )

    anchor_product_type = seed_item.get("product_type") or ""
    anchor_name = seed_item.get("prod_name") or seed_item.get("display_name") or ""
    anchor_class = classify_anchor(anchor_product_type, anchor_name)
    anchor_colour = (seed_item.get("colour") or "").lower()

    fill_slots = get_fill_slots(anchor_class, effective_gender, occasion_slug)

    # ── Step 2: fill each slot ──────────────────────────────────────────────
    complements: list[dict] = []
    empty_slots: list[str] = []
    seen_ids: set[str] = {seed_article_id}
    seen_prod_colour: set[tuple[str, str]] = set()
    seen_prod_colour.add(
        (normalize_prod_name(anchor_name), anchor_colour)
    )

    running_total = seed_item.get("price_inr") or 0.0

    for slot_spec in fill_slots:
        candidate = _find_best_candidate(
            query=slot_spec.search_query,
            slot_name=slot_spec.slot_name,
            occasion_slug=occasion_slug,
            gender=effective_gender,
            anchor_colour=anchor_colour,
            seen_ids=seen_ids,
            seen_prod_colour=seen_prod_colour,
            retriever=retriever,
            budget_remaining=budget_inr - running_total if budget_inr is not None else None,
            pairing_stats=pairing_stats,
            anchor_class=anchor_class,
        )
        if candidate:
            candidate["_slot"] = slot_spec.slot_name
            complements.append(candidate)
            seen_ids.add(candidate["article_id"])
            seen_prod_colour.add((
                normalize_prod_name(candidate.get("prod_name") or candidate.get("display_name") or ""),
                (candidate.get("colour") or "").lower(),
            ))
            running_total += candidate.get("price_inr") or 0.0
        elif slot_spec.required:
            empty_slots.append(slot_spec.slot_name)

    # ── Step 3: build rationale ─────────────────────────────────────────────
    comp_names = [c.get("display_name") or c.get("prod_name") for c in complements]
    seed_display = seed_item.get("display_name") or seed_item.get("prod_name") or "item"
    if comp_names:
        rationale = (
            f"**{occasion_slug.replace('_', ' ').title()} look** — "
            f"Styled **{seed_display}** with {', '.join(comp_names)}."
        )
    else:
        rationale = (
            f"Showing **{seed_display}** — no complementary items found for this occasion."
        )

    budget_total = running_total if running_total > 0 else None

    logger.debug(
        "[composer] look_id=%s occasion=%s gender=%s anchor_class=%s slots=%s empty=%s budget_total=%s",
        look_id, occasion_slug, effective_gender, anchor_class,
        [c["_slot"] for c in complements], empty_slots, budget_total,
    )

    return {
        "look_id": look_id,
        "seed_item": seed_item,
        "complements": complements,
        "outfit_rationale": rationale,
        "empty_slots": empty_slots,
        "occasion": occasion_slug,
        "gender": effective_gender,
        "budget_total_inr": budget_total,
    }


def _find_best_candidate(
    *,
    query: str,
    slot_name: str,
    occasion_slug: str,
    gender: str,
    anchor_colour: str,
    seen_ids: set[str],
    seen_prod_colour: set[tuple[str, str]],
    retriever: HybridRetriever,
    budget_remaining: float | None,
    pairing_stats: dict | None,
    anchor_class: str,
) -> dict | None:
    candidates = retriever.search(query, top_k=20)

    scored: list[tuple[float, dict]] = []
    for item in candidates:
        if item["article_id"] in seen_ids:
            continue
        item_key = (
            normalize_prod_name(item.get("prod_name") or item.get("display_name") or ""),
            (item.get("colour") or "").lower(),
        )
        if item_key in seen_prod_colour:
            continue
        if not is_coherent_candidate(item, occasion_slug, gender, slot_name):
            continue
        if budget_remaining is not None:
            price = item.get("price_inr") or 0.0
            if price > budget_remaining:
                continue

        base_score = item.get("score") or 0.5
        c_score = colour_score(item.get("colour") or "", anchor_colour, occasion_slug)
        fab_delta = fabric_score_delta(item, occasion_slug)
        fw_boost = _flywheel_boost(anchor_class, slot_name, occasion_slug, pairing_stats)

        final_score = (base_score * 0.5 + c_score * 0.3 + 0.2) * (1.0 + fw_boost) + fab_delta
        scored.append((final_score, item))

    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def _flywheel_boost(
    anchor_class: str,
    fill_slot: str,
    occasion_slug: str,
    pairing_stats: dict | None,
) -> float:
    """Return the transparent flywheel ranking boost for this (anchor_class, fill_slot, occasion).

    Cold-start (< FLYWHEEL_MIN_SIGNALS): returns 0.0.
    Warm: FLYWHEEL_ALPHA × positive_rate.
    total_signals = add_the_look + thumbs_up + thumbs_down + add_single_only.
    """
    if not pairing_stats:
        return 0.0
    stat = pairing_stats.get((anchor_class, fill_slot, occasion_slug))
    if stat is None:
        return 0.0
    total = stat.add_the_look + stat.thumbs_up + stat.thumbs_down + stat.add_single_only
    if total < FLYWHEEL_MIN_SIGNALS:
        return 0.0
    positive_rate = (stat.add_the_look + stat.thumbs_up) / total
    boost = FLYWHEEL_ALPHA * positive_rate
    logger.debug(
        "[flywheel] anchor=%s slot=%s occ=%s positive_rate=%.2f boost=%.3f signals=%d",
        anchor_class, fill_slot, occasion_slug, positive_rate, boost, total,
    )
    return boost


def _anchor_query_for_occasion(occasion_slug: str, gender: str) -> str:
    """Build a retrieval query to find an appropriate anchor for the occasion."""
    is_men = gender.lower() == "men"
    queries: dict[str, str] = {
        "sangeet": (
            "sherwani kurta embellished festive"
            if is_men
            else "sherwani kurta lehenga anarkali festive embellished"
        ),
        "haldi_mehendi":      "kurta kurti lehenga cotton floral yellow festive",
        "festive_puja": (
            "kurta festive ethnic nehru jacket"
            if is_men
            else "kurta kurti anarkali festive ethnic"
        ),
        "wedding_guest": (
            "sherwani kurta wedding formal ethnic"
            if is_men
            else "lehenga anarkali saree wedding ethnic formal"
        ),
        "traditional_ethnic": "saree lehenga traditional ethnic",
        "party_evening": (
            "shirt formal party"
            if is_men
            else "dress evening party top formal"
        ),
        "office": (
            "shirt formal office"
            if is_men
            else "top blouse formal shirt"
        ),
        "smart_casual": (
            "shirt casual"
            if is_men
            else "top casual blouse"
        ),
        "casual": (
            "shirt casual tshirt"
            if is_men
            else "top casual tshirt"
        ),
    }
    return queries.get(occasion_slug, "top casual")


def _anchor_matches_occasion(item: dict, occasion_slug: str) -> bool:
    """Check if an item retrieved as anchor is coherent with the occasion."""
    from src.agents.outfit.occasions import OCCASIONS
    occ = OCCASIONS.get(occasion_slug)
    if occ is None:
        return True
    pt = item.get("product_type") or ""
    name = item.get("prod_name") or ""
    if occ.ethnic_lean == ETHNIC_ONLY and not is_ethnic_item(pt, name):
        return False
    # Prefer ethnic anchor for ethnic_heavy occasions too
    if occ.ethnic_lean == ETHNIC_HEAVY and not is_ethnic_item(pt, name):
        return False
    return True


def _row_to_item(article_id: str, row: pd.Series, facets: dict, role: str) -> dict:
    return {
        "article_id": article_id,
        "display_name": row.get("display_name") or str(row.get("prod_name") or ""),
        "prod_name": str(row.get("prod_name") or ""),
        "colour": facets.get("colour_group_name") or str(row.get("colour_group_name") or ""),
        "product_type": facets.get("product_type_name") or str(row.get("product_type_name") or ""),
        "department": facets.get("department_name") or str(row.get("department_name") or ""),
        "index_group_name": str(row.get("index_group_name") or ""),
        "detail_desc": str(row.get("detail_desc") or ""),
        "image_url": row.get("image_url"),
        "score": 1.0,
        "price_inr": row.get("price_inr"),
        "pdp_handle": row.get("pdp_handle"),
        "_role": role,
    }


def _empty_result(look_id: str, occasion_slug: str, gender: str, reason: str) -> dict:
    return {
        "look_id": look_id,
        "seed_item": None,
        "complements": [],
        "outfit_rationale": reason,
        "empty_slots": [],
        "occasion": occasion_slug,
        "gender": gender,
        "budget_total_inr": None,
    }
