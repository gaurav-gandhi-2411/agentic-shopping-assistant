from __future__ import annotations

import re

import pandas as pd

from .dense_search import DenseRetriever
from .sparse_search import SparseRetriever

_CATEGORY_SUFFIXES = frozenset(
    {
        "blouse",
        "shirt",
        "top",
        "tee",
        "t shirt",
        "tshirt",
        "dress",
        "skirt",
        "trousers",
        "trouser",
        "pants",
        "jeans",
        "jacket",
        "coat",
        "blazer",
        "sweater",
        "jumper",
        "cardigan",
        "hoodie",
        "shoe",
        "shoes",
        "bag",
        "shorts",
        "leggings",
        "tights",
        "vest",
        "bodysuit",
        "dungarees",
        "jumpsuit",
        "playsuit",
        "bikini",
    }
)


def normalize_prod_name(name: str) -> str:
    """Normalize a product name for dedup.

    Strips punctuation and trailing category-suffix words so that
    'Gyda blouse' and 'Gyda!' both reduce to 'gyda', while
    'Miami Slim' and 'Miami Slim HW' remain distinct.
    """
    if not name:
        return ""
    n = name.lower()
    # Fuse hyphenated garment terms before general punct removal so
    # "t-shirt" → "tshirt" (in suffixes) rather than "t" + "shirt" split.
    n = re.sub(r"\bt-shirt\b", "tshirt", n)
    # Replace remaining non-alphanumeric chars with spaces
    n = re.sub(r"[^\w\s]", " ", n)
    words = n.split()
    # Remove trailing category suffixes (loop handles "slim trousers" → "slim")
    while words and words[-1] in _CATEGORY_SUFFIXES:
        words.pop()
    return " ".join(words)


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        catalogue_df: pd.DataFrame,
        config: dict,
    ):
        self.dense = dense
        self.sparse = sparse
        self.catalogue_df = catalogue_df.set_index("article_id")
        self.config = config

    def search(
        self,
        query: str,
        top_k: int = None,
        filters: dict = None,
    ) -> list[dict]:
        if top_k is None:
            top_k = self.config["retrieval"]["final_k"]
        fetch_k = self.config["retrieval"]["top_k"]
        rrf_k = self.config["retrieval"]["rrf_k"]

        dense_hits = self.dense.search(query, top_k=fetch_k * 2)
        sparse_hits = self.sparse.search(query, top_k=fetch_k * 2)

        rrf_scores: dict[str, float] = {}
        for rank, (article_id, _) in enumerate(dense_hits, start=1):
            rrf_scores[article_id] = rrf_scores.get(article_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (article_id, _) in enumerate(sparse_hits, start=1):
            rrf_scores[article_id] = rrf_scores.get(article_id, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Extract optional store filter before iterating (not a facet — lives in `store` column)
        store_filter: str | None = None
        remaining_filters: dict | None = None
        if filters:
            store_filter = filters.get("store") or None
            remaining_filters = {k: v for k, v in filters.items() if k != "store"} or None

        results = []
        for article_id, score in ranked:
            if article_id not in self.catalogue_df.index:
                continue
            row = self.catalogue_df.loc[article_id]
            facets = row["facets"] if isinstance(row["facets"], dict) else {}

            # --- Store filter (cross-store unified index only; no-op on per-brand indices) ---
            if store_filter is not None:
                item_store = (
                    str(row["store"]).lower()
                    if "store" in row.index and row["store"] is not None
                    else ""
                )
                if item_store != store_filter.lower():
                    continue

            if remaining_filters:
                price_min = remaining_filters.get("price_min")
                price_max = remaining_filters.get("price_max")
                facet_filters = {
                    k: v
                    for k, v in remaining_filters.items()
                    if k not in ("price_min", "price_max")
                }

                if facet_filters and not all(
                    str(facets.get(k, "")).lower() == str(v).lower()
                    for k, v in facet_filters.items()
                ):
                    continue

                if price_min is not None or price_max is not None:
                    item_price = (
                        row.get("price_inr")
                        if hasattr(row, "get")
                        else row["price_inr"]
                        if "price_inr" in row.index
                        else None
                    )
                    if item_price is None or not isinstance(item_price, (int, float)):
                        continue  # skip items without price when price filter is active
                    if price_min is not None and float(item_price) < float(price_min):
                        continue
                    if price_max is not None and float(item_price) > float(price_max):
                        continue

            results.append(
                {
                    "article_id": article_id,
                    "prod_name": row.get("prod_name", ""),
                    "display_name": row["display_name"],
                    "colour": facets.get("colour_group_name", ""),
                    "product_type": facets.get("product_type_name", ""),
                    "department": facets.get("department_name", ""),
                    "detail_desc": row["detail_desc"],
                    "image_url": _img
                    if isinstance(_img := row.get("image_url"), str) and _img
                    else None,
                    "score": score,
                    "store": (
                        str(row["store"])
                        if "store" in row.index and row["store"] is not None
                        else None
                    ),
                    "price_inr": (
                        float(row["price_inr"])
                        if "price_inr" in row.index
                        and row["price_inr"] is not None
                        and not pd.isna(row["price_inr"])
                        else None
                    ),
                    "pdp_handle": (
                        str(row["pdp_handle"])
                        if "pdp_handle" in row.index and row["pdp_handle"] is not None
                        else None
                    ),
                    "pdp_live": (
                        bool(row["pdp_live"])
                        if "pdp_live" in row.index
                        and row["pdp_live"] is not None
                        and not pd.isna(row["pdp_live"])
                        else None
                    ),
                    "gender": (
                        str(row["gender"]).lower()
                        if "gender" in row.index and row["gender"] is not None
                        else "unknown"
                    ),
                }
            )

            if len(results) >= top_k:
                break

        # Deprioritize items with known-dead PDP links — move them to end of list
        live = [it for it in results if it.get("pdp_live") is not False]
        dead = [it for it in results if it.get("pdp_live") is False]
        return live + dead
