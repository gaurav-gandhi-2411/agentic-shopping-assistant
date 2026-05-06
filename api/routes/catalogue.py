"""Catalogue routes: item detail and FAISS similarity lookup."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

import api.deps as deps
from api.schemas import ItemSummary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/catalogue", tags=["catalogue"])


def _row_to_item(article_id: str, row: Any, score: float | None = None) -> ItemSummary:
    facets = row.get("facets", {}) if isinstance(row.get("facets"), dict) else {}
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
    )


@router.get("/{article_id}", response_model=ItemSummary)
def get_item(article_id: str) -> ItemSummary:
    """Return full metadata for a single catalogue item."""
    df = deps.get_catalogue_df()
    indexed = df.set_index("article_id") if "article_id" in df.columns else df
    if article_id not in indexed.index:
        raise HTTPException(status_code=404, detail=f"Item {article_id!r} not found")
    row = indexed.loc[article_id]
    return _row_to_item(article_id, row)


@router.get("/{article_id}/similar", response_model=list[ItemSummary])
def get_similar(article_id: str, k: int = 5) -> list[ItemSummary]:
    """Return top-k FAISS-similar items for the given article.

    Uses the dense retriever's stored embeddings (no re-encoding needed) to
    find nearest neighbours by cosine similarity.  Scores are inner-product
    values (higher = more similar) and are included in each returned item so
    the frontend can display a confidence indicator without extra calls.
    """
    df = deps.get_catalogue_df()
    indexed = df.set_index("article_id") if "article_id" in df.columns else df
    if article_id not in indexed.index:
        raise HTTPException(status_code=404, detail=f"Item {article_id!r} not found")

    retriever = deps.get_retriever()
    neighbours: list[tuple[str, float]] = retriever.dense.search_by_id(article_id, top_k=k)

    items: list[ItemSummary] = []
    for aid, score in neighbours:
        if aid not in indexed.index:
            continue
        items.append(_row_to_item(aid, indexed.loc[aid], score=score))

    logger.info(
        "similar lookup",
        extra={"article_id": article_id, "k": k, "n_results": len(items)},
    )
    return items
