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

# Store-redundancy penalty applied per additional item from the same store.
# With penalty=0.5, the 2nd item from store X costs 0.5*(1-λ) in score, the 3rd costs
# 0.25*(1-λ), etc.  Chosen so a second item from the same store is penalised but can still
# beat an irrelevant item from a different store.
_STORE_PENALTY: float = 0.5

# F2 relevance floor — items below this RRF score are excluded as noise.
# Locked at 0.0060 post-F1 rebuild (≈ p5-p10 across 5 calibration queries on the
# clean index). The primary relevance gate is the product_type_name filter; this floor
# is the backstop for queries with genuinely no catalogue matches.
_RELEVANCE_FLOOR: float = 0.0060


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


def store_diversity_rerank(
    candidates: list[dict],
    top_k: int,
    store_diversity: float,
) -> list[dict]:
    """Greedy MMR-style re-rank to spread results across stores.

    Formula
    -------
    At each greedy step, score every remaining candidate as::

        mmr_score(item) = λ * rel_norm(item) - (1-λ) * (1 - penalty ** n_store)

    where:
      - λ = 1 - store_diversity  (so λ=1.0 → pure relevance, λ=0.5 → balanced)
      - rel_norm(item) = item's RRF score / max_rrf_score  (normalised to [0,1])
      - n_store = how many items from item's store are already in selected list
      - penalty = _STORE_PENALTY = 0.5  (geometric decay per extra same-store item)
      - redundancy term = 1 - 0.5**n_store   (0 when n_store=0; →1 as n_store grows)

    Properties:
      - store_diversity=0.0 → redundancy term is zero → pure relevance order preserved
      - store_diversity=1.0 → λ=0 → pure diversity (ignores relevance)
      - The #1 RRF result is always selected first (rel_norm=1, redundancy=0 at step 0),
        preserving the most-relevant result at the top.

    Guards (skip re-rank, return pure-relevance order):
      - store_diversity == 0.0
      - fewer than 2 distinct stores in the candidate window

    Parameters
    ----------
    candidates:
        Full list of filter-passing result dicts, ordered by descending RRF score.
        Each must have 'score' (RRF float) and 'store' (str | None).
    top_k:
        Number of items to return.
    store_diversity:
        Knob in [0.0, 1.0].  0.0 = pure relevance (current behaviour). Default
        in config.yaml is 0.0; owner sets to desired value after reviewing sweep table.

    Returns
    -------
    Re-ranked list of at most top_k items.
    """
    if not candidates:
        return []

    # Guard: no-op when knob is off
    if store_diversity == 0.0:
        return candidates[:top_k]

    # Guard: no-op if fewer than 2 distinct stores in candidate window
    distinct_stores = {c["store"] for c in candidates if c["store"]}
    if len(distinct_stores) < 2:
        return candidates[:top_k]

    lam = 1.0 - store_diversity

    # Normalise RRF scores to [0,1] over the candidate window
    max_score = max(c["score"] for c in candidates)
    if max_score <= 0:
        return candidates[:top_k]

    selected: list[dict] = []
    remaining: list[dict] = list(candidates)
    store_counts: dict[str | None, int] = {}

    for _ in range(min(top_k, len(candidates))):
        best_item: dict | None = None
        best_mmr: float = float("-inf")

        for item in remaining:
            rel_norm = item["score"] / max_score
            n_store = store_counts.get(item["store"], 0)
            redundancy = 1.0 - (_STORE_PENALTY**n_store)
            mmr = lam * rel_norm - (1.0 - lam) * redundancy
            if mmr > best_mmr:
                best_mmr = mmr
                best_item = item

        if best_item is None:
            break

        selected.append(best_item)
        remaining.remove(best_item)
        s = best_item["store"]
        store_counts[s] = store_counts.get(s, 0) + 1

    return selected


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
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        if top_k is None:
            top_k = self.config["retrieval"]["final_k"]
        fetch_k = self.config["retrieval"]["top_k"]
        rrf_k = self.config["retrieval"]["rrf_k"]

        # Pre-filter BM25 by product_type_name when that facet filter is present.
        # BM25 already scores all 44k items before argsort; masking here costs nothing
        # extra and guarantees the sparse retrieval window is filled by the right type
        # rather than by unrelated items that happen to mention the type word in text.
        # Dense (FAISS) uses the wider fetch_k window to compensate for no pre-filter.
        sparse_allowed_ids: np.ndarray | None = None
        type_filter_val = (filters or {}).get("product_type_name")
        if type_filter_val is not None and "product_type_name" in self.catalogue_df.columns:
            pt_col = self.catalogue_df["product_type_name"].str.lower()
            sparse_allowed_ids = (
                self.catalogue_df.index[pt_col == type_filter_val.lower()]
                .values.astype(str)
            )

        dense_hits = self.dense.search(query, top_k=fetch_k * 2)
        sparse_hits = self.sparse.search(query, top_k=fetch_k * 2, allowed_ids=sparse_allowed_ids)

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

        # Collect ALL filter-passing candidates from the full RRF window.
        # We do NOT truncate here — diversity re-rank needs the full candidate pool.
        candidates: list[dict] = []
        for article_id, score in ranked:
            if score < _RELEVANCE_FLOOR:
                continue  # skip noise; ranked is not guaranteed sorted so use continue not break
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

            candidates.append(
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

        # --- Store-diversity re-rank (MMR nudge) ---
        # Guard: skip when a store filter is set (user explicitly narrowed to one store).
        store_diversity: float = self.config["retrieval"].get("store_diversity", 0.0)
        if store_filter is not None:
            results = candidates[:top_k]
        else:
            results = store_diversity_rerank(candidates, top_k, store_diversity)

        # Deprioritize items with known-dead PDP links — move them to end of list
        live = [it for it in results if it.get("pdp_live") is not False]
        dead = [it for it in results if it.get("pdp_live") is False]
        return live + dead
