import re

import pandas as pd
from .dense_search import DenseRetriever
from .sparse_search import SparseRetriever


_CATEGORY_SUFFIXES = frozenset({
    "blouse", "shirt", "top", "tee", "t shirt", "tshirt",
    "dress", "skirt", "trousers", "trouser", "pants", "jeans",
    "jacket", "coat", "blazer", "sweater", "jumper",
    "cardigan", "hoodie", "shoe", "shoes", "bag", "shorts",
    "leggings", "tights", "vest", "bodysuit", "dungarees",
    "jumpsuit", "playsuit", "bikini",
})


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

        results = []
        for article_id, score in ranked:
            if article_id not in self.catalogue_df.index:
                continue
            row = self.catalogue_df.loc[article_id]
            facets = row["facets"] if isinstance(row["facets"], dict) else {}

            if filters:
                match = all(
                    str(facets.get(k, "")).lower() == str(v).lower()
                    for k, v in filters.items()
                )
                if not match:
                    continue

            results.append({
                "article_id": article_id,
                "prod_name": row.get("prod_name", ""),
                "display_name": row["display_name"],
                "colour": facets.get("colour_group_name", ""),
                "product_type": facets.get("product_type_name", ""),
                "department": facets.get("department_name", ""),
                "detail_desc": row["detail_desc"],
                "image_url": row.get("image_url", ""),
                "score": score,
            })

            if len(results) >= top_k:
                break

        return results
