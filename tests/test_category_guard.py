"""Category guard in HybridRetriever.search (V3(b2)(b)): garment queries never return
accessory / jewellery / non_fashion rows. Self-contained (mock dense/sparse, no index)."""

from __future__ import annotations

import unittest.mock as mock

import pandas as pd
import pytest

from src.agents.intent_parser import _GARMENT_RULES
from src.catalogue.hygiene import is_garment_type_name
from src.retrieval.hybrid_search import HybridRetriever

_PUBLIC_STORE = "libas"


def _catalogue(with_category: bool = True) -> pd.DataFrame:
    rows = [
        # id, type, category
        ("g1", "lehenga", "garment"),
        ("g2", "lehenga", "garment"),
        ("a1", "lehenga", "accessory"),  # type says lehenga, title/category says accessory
        ("j1", "lehenga", "jewellery"),
        ("n1", "lehenga", "non_fashion"),
        ("u1", "lehenga", "unknown"),  # unknown is never excluded (fail open)
        ("e1", "earrings", "jewellery"),
    ]
    df = pd.DataFrame(
        {
            "article_id": [r[0] for r in rows],
            "store": _PUBLIC_STORE,
            "gender": "women",
            "product_type_name": [r[1] for r in rows],
            "prod_name": [f"Item {r[0]}" for r in rows],
            "display_name": [f"Item {r[0]}" for r in rows],
            "detail_desc": "desc",
            "image_url": None,
            "price_inr": 500.0,
            "pdp_handle": None,
            "pdp_live": True,
            "facets": [{"colour_group_name": "Red", "product_type_name": r[1]} for r in rows],
        }
    )
    if with_category:
        df["category"] = [r[2] for r in rows]
    return df


def _retriever(df: pd.DataFrame, **retrieval: object) -> tuple[HybridRetriever, mock.MagicMock]:
    ids = df["article_id"].tolist()
    dense = mock.MagicMock()
    dense.search.return_value = [(a, 1.0 / (i + 1)) for i, a in enumerate(ids)]
    dense.article_ids = df["article_id"].values
    sparse = mock.MagicMock()
    sparse.search.return_value = [(a, 1.0 / (i + 1)) for i, a in enumerate(ids)]
    config = {
        "retrieval": {"final_k": 10, "top_k": 10, "rrf_k": 60, "store_diversity": 0.0, **retrieval}
    }
    return HybridRetriever(dense, sparse, df, config), sparse


def _ids(results: list[dict]) -> set[str]:
    return {r["article_id"] for r in results}


def test_garment_query_excludes_accessory_jewellery_non_fashion() -> None:
    retr, sparse = _retriever(_catalogue())
    got = _ids(retr.search("lehenga", top_k=10, filters={"product_type_name": "lehenga"}))
    assert got == {"g1", "g2", "u1"}
    # pushed into the BM25 fetch too, not only post-filtered
    allowed = set(sparse.search.call_args.kwargs["allowed_ids"])
    assert allowed == {"g1", "g2", "u1"}


def test_accessory_query_is_not_guarded() -> None:
    retr, _ = _retriever(_catalogue())
    got = _ids(retr.search("earrings", top_k=10, filters={"product_type_name": "earrings"}))
    assert got == {"e1"}


def test_query_without_garment_type_is_not_guarded() -> None:
    retr, _ = _retriever(_catalogue())
    got = _ids(retr.search("something red", top_k=10))
    assert got == {"g1", "g2", "a1", "j1", "n1", "u1", "e1"}


def test_guard_can_be_switched_off_by_config() -> None:
    retr, _ = _retriever(_catalogue(), category_guard=False)
    got = _ids(retr.search("lehenga", top_k=10, filters={"product_type_name": "lehenga"}))
    assert got == {"g1", "g2", "a1", "j1", "n1", "u1"}


def test_index_without_category_column_is_unchanged() -> None:
    retr, _ = _retriever(_catalogue(with_category=False))
    got = _ids(retr.search("lehenga", top_k=10, filters={"product_type_name": "lehenga"}))
    assert got == {"g1", "g2", "a1", "j1", "n1", "u1"}


@pytest.mark.parametrize(
    "name",
    sorted({t for _, t in _GARMENT_RULES}),
)
def test_every_intent_garment_type_is_classified_correctly(name: str) -> None:
    """The guard keys on is_garment_type_name(intent.garment_type); a wrong answer here would
    either shield an accessory query (false negative) or hide jewellery from a jewellery query."""
    non_garment = {"bag", "dupatta", "earrings", "footwear", "jewellery", "jhumka", "necklace"}
    assert is_garment_type_name(name) == (name not in non_garment)
