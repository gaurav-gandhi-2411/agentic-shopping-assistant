"""Catalogue routes: item detail and FAISS similarity lookup."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

import api.deps as deps
from api.auth import get_current_user_id_or_demo
from api.schemas import ItemSummary, PriceMatch
from src.catalogue.entity_resolution import (
    find_cross_store_matches,
    get_cached_brand_index,
    get_cached_brand_stores_map,
)
from src.config.stores import build_pdp_url, get_store_display_name
from src.retrieval.hybrid_search import store_diversity_rerank

# Candidate-window multiplier for the /similar diversity re-rank.
# We fetch this many times the requested k to give the MMR algorithm a
# meaningful pool to draw from. Capped so single-item catalogues still work.
_SIMILAR_DIVERSITY_WINDOW: int = 4

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/catalogue", tags=["catalogue"])


def _build_price_matches(
    row_dict: dict[str, Any],
    catalogue_df: pd.DataFrame,
) -> list[PriceMatch] | None:
    """Run the precision-gated cross-store matcher and return PriceMatch objects.

    Uses the precomputed brand-stores map to skip the O(n) scan for single-store
    brands (the ~99.9% case), making the hot path essentially free.

    Returns None (not []) when no matches are found, which serialises to null
    in the JSON response and avoids an empty list allocation.
    """
    brand_map = get_cached_brand_stores_map(catalogue_df)
    b_index = get_cached_brand_index(catalogue_df)
    raw_matches = find_cross_store_matches(
        row_dict,
        catalogue_df,
        brand_stores_map=brand_map,
        brand_index=b_index,
    )
    if not raw_matches:
        return None
    return [
        PriceMatch(
            store=m["store"],
            store_display=m["store_display"],
            price_inr=m["price_inr"],
            pdp_url=m["pdp_url"],
            confidence=m["confidence"],
            is_snapshot_price=True,
        )
        for m in raw_matches
    ]


def _row_to_item(
    article_id: str,
    row: Any,
    score: float | None = None,
    catalogue_df: pd.DataFrame | None = None,
) -> ItemSummary:
    """Convert one catalogue row to an ItemSummary.

    Parameters
    ----------
    article_id:
        The catalogue article identifier.
    row:
        A pandas Series or dict-like object with catalogue fields.
    score:
        Optional relevance score from the retriever.
    catalogue_df:
        Full catalogue DataFrame.  When provided, the precision-gated
        cross-store price matcher runs and populates ``price_matches``.
        When None (e.g. in similarity lookup), price_matches is omitted.
    """
    facets = row.get("facets", {}) if isinstance(row.get("facets"), dict) else {}
    store = row.get("store") or None
    row_dict: dict[str, Any] = row if isinstance(row, dict) else (
        row.to_dict() if hasattr(row, "to_dict") else dict(row)
    )
    price_matches = _build_price_matches(row_dict, catalogue_df) if catalogue_df is not None else None
    return ItemSummary(
        article_id=article_id,
        prod_name=row.get("prod_name", ""),
        display_name=row.get("display_name", ""),
        colour=facets.get("colour_group_name", ""),
        product_type=facets.get("product_type_name", ""),
        department=facets.get("department_name", ""),
        image_url=row.get("image_url") or None,
        detail_desc=row.get("detail_desc") or None,
        score=score,
        price_inr=float(row.get("price_inr")) if row.get("price_inr") is not None else None,
        pdp_handle=str(row.get("pdp_handle")) if row.get("pdp_handle") is not None else None,
        store=store,
        store_display=get_store_display_name(store),
        pdp_url=build_pdp_url(store, row_dict),
        price_matches=price_matches,
    )


@router.get("/{article_id}", response_model=ItemSummary)
def get_item(
    article_id: str,
    _user_id: str = Depends(get_current_user_id_or_demo),
) -> ItemSummary:
    """Return full metadata for a single catalogue item, including cross-store price matches."""
    df = deps.get_catalogue_df()
    indexed = df.set_index("article_id") if "article_id" in df.columns else df
    if article_id not in indexed.index:
        raise HTTPException(status_code=404, detail=f"Item {article_id!r} not found")
    row = indexed.loc[article_id]
    return _row_to_item(article_id, row, catalogue_df=df)


@router.get("/{article_id}/similar", response_model=list[ItemSummary])
def get_similar(
    article_id: str,
    k: int = 5,
    _user_id: str = Depends(get_current_user_id_or_demo),
) -> list[ItemSummary]:
    """Return top-k visually-similar items for the given article, diversified across stores.

    Uses the dense retriever's stored embeddings (no re-encoding needed) to
    find nearest neighbours by cosine similarity, then applies the same
    store-diversity MMR re-rank used by the text retrieval path.

    The anchor selection path (find_anchor_from_image → outfit seed) is NOT
    touched here — that path must stay pure best-visual-match so the outfit
    seed is always the highest-fidelity CLIP match.

    Implementation notes
    --------------------
    - Fetches a wider candidate window (_SIMILAR_DIVERSITY_WINDOW * k) before
      diversifying, so the MMR algorithm has meaningful choice.
    - Guards inherited from store_diversity_rerank: single-store window or
      knob==0.0 → pure-relevance order (no-op, exact legacy behaviour).
    - Scores are inner-product cosine-similarity values (higher = more similar).
    """
    config = deps.get_config()
    df = deps.get_catalogue_df()
    indexed = df.set_index("article_id") if "article_id" in df.columns else df
    if article_id not in indexed.index:
        raise HTTPException(status_code=404, detail=f"Item {article_id!r} not found")

    retriever = deps.get_retriever()

    # Fetch a wider window to give the diversity re-rank a meaningful pool.
    # The +1 is inside search_by_id (drops the seed itself), so fetch_k here
    # is the number of candidates AFTER self-exclusion.
    fetch_k = min(k * _SIMILAR_DIVERSITY_WINDOW, retriever.dense.index.ntotal - 1)
    fetch_k = max(fetch_k, k)  # never fetch fewer than requested
    neighbours: list[tuple[str, float]] = retriever.dense.search_by_id(article_id, top_k=fetch_k)

    # Build candidate dicts with store info for the diversity re-rank.
    # Items missing from the catalogue index are silently skipped.
    candidates: list[dict[str, Any]] = []
    for aid, score in neighbours:
        if aid not in indexed.index:
            continue
        row = indexed.loc[aid]
        store = str(row["store"]) if "store" in row.index and row["store"] is not None else None
        candidates.append({"article_id": aid, "score": score, "store": store, "_row": row})

    # Apply store-diversity MMR re-rank using the same knob as text retrieval.
    # knob=0.0 is a guaranteed no-op (pure-relevance order, legacy behaviour).
    store_diversity: float = float(config.get("retrieval", {}).get("store_diversity", 0.0))
    diversified = store_diversity_rerank(candidates, top_k=k, store_diversity=store_diversity)

    items: list[ItemSummary] = [
        _row_to_item(c["article_id"], c["_row"], score=c["score"]) for c in diversified
    ]

    logger.info(
        "similar lookup",
        extra={
            "article_id": article_id,
            "k": k,
            "fetch_k": fetch_k,
            "n_candidates": len(candidates),
            "n_results": len(items),
            "store_diversity": store_diversity,
        },
    )
    return items
