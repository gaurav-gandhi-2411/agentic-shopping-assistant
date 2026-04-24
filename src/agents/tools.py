import random

import pandas as pd
from src.retrieval.hybrid_search import HybridRetriever, normalize_prod_name


VALID_FACET_KEYS = {
    "colour_group_name",
    "product_type_name",
    "department_name",
    "index_group_name",
    "garment_group_name",
}

_NEUTRAL_COLOURS = frozenset({
    "black", "white", "grey", "dark grey", "light grey", "beige", "light beige",
})

_BOTTOM_TYPES = frozenset({"trousers", "jeans", "shorts", "skirt", "leggings"})
_DRESS_TYPES = frozenset({"dress", "jumpsuit", "playsuit", "dungarees"})
_OUTERWEAR_TYPES = frozenset({"jacket", "coat", "blazer", "cardigan", "waistcoat", "parka", "anorak"})


def search_catalogue(
    query: str,
    filters: dict | None,
    retriever: HybridRetriever,
    top_k: int,
) -> dict:
    """Runs hybrid retrieval. Returns {items: [...], query: ..., n_results: int}."""
    items = retriever.search(query, top_k=top_k, filters=filters or None)
    return {"items": items, "query": query, "n_results": len(items)}


def compare_items(article_ids: list[str], catalogue_df: pd.DataFrame) -> dict:
    """Given 2-5 article_ids, returns comparison dict with their attributes side-by-side."""
    article_ids = article_ids[:5]
    indexed = catalogue_df.set_index("article_id")
    items = []
    for aid in article_ids:
        if aid in indexed.index:
            row = indexed.loc[aid]
            facets = row["facets"] if isinstance(row["facets"], dict) else {}
            items.append({
                "article_id": aid,
                "display_name": row["display_name"],
                "colour": facets.get("colour_group_name", ""),
                "product_type": facets.get("product_type_name", ""),
                "department": facets.get("department_name", ""),
                "detail_desc": str(row["detail_desc"]),
                "image_url": row.get("image_url", ""),
                "score": 0.0,
            })
    return {"items": items, "article_ids": article_ids, "n_items": len(items)}


def apply_filter(current_filters: dict, filter_key: str, filter_value: str) -> dict:
    """Merges a new filter into current_filters. Returns updated dict.
    filter_key must be one of the valid facet keys; silently ignored otherwise."""
    if filter_key not in VALID_FACET_KEYS:
        return current_filters
    return {**current_filters, filter_key: filter_value}


def clarify(question: str) -> dict:
    """Stub tool — signals the graph to ask the user for clarification.
    The graph routes to END when this is called."""
    return {"clarification_question": question}


def suggest_outfit(
    seed_article_id: str,
    catalogue_df: pd.DataFrame,
    retriever: HybridRetriever,
) -> dict:
    """Given a seed article_id, find complementary items to build a complete outfit.

    Returns {seed_item, complements: list, outfit_rationale: str}.
    """
    indexed = catalogue_df.set_index("article_id")
    if seed_article_id not in indexed.index:
        return {"seed_item": None, "complements": [], "outfit_rationale": "Item not found."}

    row = indexed.loc[seed_article_id]
    facets = row["facets"] if isinstance(row["facets"], dict) else {}
    seed_item = {
        "article_id": seed_article_id,
        "display_name": row["display_name"],
        "colour": facets.get("colour_group_name", ""),
        "product_type": facets.get("product_type_name", ""),
        "department": facets.get("department_name", ""),
        "detail_desc": str(row["detail_desc"]),
        "image_url": row.get("image_url", ""),
        "score": 1.0,
        "_role": "seed",
    }

    colour = seed_item["colour"].lower()
    product_type = seed_item["product_type"].lower()

    seed_is_outerwear = any(t in product_type for t in _OUTERWEAR_TYPES)

    # Complement slot definitions: (search_query, human_readable_label).
    # Outerwear seeds never get another outerwear slot.
    if seed_is_outerwear:
        complement_slots = [
            ("top shirt blouse", "top"),
            ("trousers jeans skirt", "bottom"),
        ]
    elif any(t in product_type for t in _DRESS_TYPES):
        complement_slots = [
            ("jacket cardigan blazer", "jacket"),
            ("shoes sandals boots heels flat shoe", "shoes"),
            ("bag handbag", "bag"),
        ]
    elif any(t in product_type for t in _BOTTOM_TYPES):
        complement_slots = [
            ("top shirt blouse", "top"),
            ("jacket blazer coat cardigan", "jacket"),
        ]
    else:  # top/shirt/sweater/etc.
        complement_slots = [
            ("trousers jeans skirt", "bottom"),
            ("jacket blazer coat cardigan", "jacket"),
        ]

    is_neutral = colour in _NEUTRAL_COLOURS
    color_hint = "" if is_neutral else colour

    complements: list[dict] = []
    empty_slots: list[str] = []
    seen_ids: set[str] = {seed_article_id}
    seen_prod_colour: set[tuple[str, str]] = set()
    # Pre-seed with the seed item so it can't appear as a complement under any alias
    seen_prod_colour.add((normalize_prod_name(seed_item.get("prod_name", seed_item["display_name"])), colour))

    for cq, slot_label in complement_slots:
        query = f"{cq} {color_hint}".strip()
        candidates = retriever.search(query, top_k=15)

        # Collect up to 3 colour-compatible, non-duplicate candidates; pick 1 randomly
        colour_ok_pool: list[dict] = []
        for item in candidates:
            if item["article_id"] in seen_ids:
                continue
            item_type = item.get("product_type", "").lower()
            # Exclude same product type as seed
            if product_type and product_type == item_type:
                continue
            # When seed is outerwear, exclude all outerwear-category complements
            if seed_is_outerwear and any(t in item_type for t in _OUTERWEAR_TYPES):
                continue
            # Dedup: skip if same normalized prod_name + colour already chosen
            item_key = (normalize_prod_name(item.get("prod_name", item["display_name"])), item.get("colour", "").lower())
            if item_key in seen_prod_colour:
                continue
            item_colour = item.get("colour", "").lower()
            colour_ok = (
                is_neutral
                or item_colour in _NEUTRAL_COLOURS
                or item_colour == colour
            )
            if colour_ok:
                colour_ok_pool.append(item)
                if len(colour_ok_pool) >= 3:
                    break

        if colour_ok_pool:
            chosen = random.choice(colour_ok_pool)
        else:
            # Fallback within the same query (no cross-category backfill)
            fallback_pool = [
                it for it in candidates[:5]
                if it["article_id"] not in seen_ids
                and it.get("product_type", "").lower() != product_type
                and not (seed_is_outerwear and any(t in it.get("product_type", "").lower() for t in _OUTERWEAR_TYPES))
                and (normalize_prod_name(it.get("prod_name", it["display_name"])), it.get("colour", "").lower()) not in seen_prod_colour
            ]
            chosen = random.choice(fallback_pool) if fallback_pool else None

        if chosen:
            chosen["_role"] = "complement"
            complements.append(chosen)
            seen_ids.add(chosen["article_id"])
            seen_prod_colour.add((normalize_prod_name(chosen.get("prod_name", chosen["display_name"])), chosen.get("colour", "").lower()))
        else:
            empty_slots.append(slot_label)

    print(
        f"[outfit] seed={seed_article_id} type={product_type} colour={colour} "
        f"complements={[c['article_id'] for c in complements]} empty_slots={empty_slots}"
    )
    comp_names = [c["display_name"] for c in complements]
    if comp_names:
        rationale = f"Paired **{seed_item['display_name']}** with {' and '.join(comp_names)}."
    else:
        rationale = f"Showing **{seed_item['display_name']}** — no complementary items found."

    return {
        "seed_item": seed_item,
        "complements": complements,
        "outfit_rationale": rationale,
        "empty_slots": empty_slots,
    }
