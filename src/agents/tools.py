import pandas as pd
from src.retrieval.hybrid_search import HybridRetriever


VALID_FACET_KEYS = {
    "colour_group_name",
    "product_type_name",
    "department_name",
    "index_group_name",
    "garment_group_name",
}


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
