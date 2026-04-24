import random

import pandas as pd
from src.retrieval.hybrid_search import HybridRetriever


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

    # Classify seed and pick what it pairs with
    if any(t in product_type for t in _DRESS_TYPES):
        complement_queries = ["jacket blazer coat", "bag accessories shoes"]
    elif any(t in product_type for t in _BOTTOM_TYPES):
        complement_queries = ["top shirt blouse", "jacket blazer coat"]
    else:
        complement_queries = ["trousers jeans skirt", "jacket blazer coat"]

    is_neutral = colour in _NEUTRAL_COLOURS
    color_hint = "" if is_neutral else colour

    complements: list[dict] = []
    seen_ids: set[str] = {seed_article_id}

    for cq in complement_queries:
        query = f"{cq} {color_hint}".strip()
        candidates = retriever.search(query, top_k=15)

        # Collect up to 3 colour-compatible candidates, then randomly pick 1 for variety
        colour_ok_pool: list[dict] = []
        for item in candidates:
            if item["article_id"] in seen_ids:
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
            # Fallback: random pick from first 3 non-seed items regardless of colour
            fallback_pool = [it for it in candidates[:5] if it["article_id"] not in seen_ids]
            chosen = random.choice(fallback_pool) if fallback_pool else None

        if chosen:
            chosen["_role"] = "complement"
            complements.append(chosen)
            seen_ids.add(chosen["article_id"])

    print(
        f"[outfit] seed={seed_article_id} type={product_type} colour={colour} "
        f"complements={[c['article_id'] for c in complements]}"
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
    }
