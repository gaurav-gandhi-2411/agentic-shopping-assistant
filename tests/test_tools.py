"""Tests for the four agent tools — pure functions, no LLM required."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalogue.loader import load_config
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.sparse_search import SparseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.agents.tools import (
    search_catalogue,
    compare_items,
    apply_filter,
    clarify,
    VALID_FACET_KEYS,
)

SAVE_DIR = Path("data/processed")


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def catalogue_df():
    return pd.read_parquet(SAVE_DIR / "catalogue.parquet")


@pytest.fixture(scope="module")
def retriever(config, catalogue_df):
    dense = DenseRetriever.load(config, SAVE_DIR)
    sparse = SparseRetriever.load(config, SAVE_DIR)
    return HybridRetriever(dense, sparse, catalogue_df, config)


# ---------------------------------------------------------------------------
# search_catalogue
# ---------------------------------------------------------------------------

def test_search_tool_returns_items(retriever, config):
    top_k = config["retrieval"]["final_k"]
    result = search_catalogue("black jacket", None, retriever, top_k)
    assert "items" in result
    assert "query" in result
    assert "n_results" in result
    assert result["n_results"] == len(result["items"])
    assert result["n_results"] > 0
    required = {"article_id", "display_name", "colour", "product_type", "department", "detail_desc", "score"}
    for item in result["items"]:
        assert required.issubset(item.keys())


def test_search_tool_with_filter(retriever, config):
    top_k = config["retrieval"]["final_k"]
    result = search_catalogue("dress", {"colour_group_name": "Black"}, retriever, top_k)
    assert result["n_results"] > 0
    for item in result["items"]:
        assert item["colour"].lower() == "black"


# ---------------------------------------------------------------------------
# compare_items
# ---------------------------------------------------------------------------

def test_compare_tool_handles_2_items(catalogue_df):
    ids = catalogue_df["article_id"].iloc[:2].tolist()
    result = compare_items(ids, catalogue_df)
    assert result["n_items"] == 2
    assert len(result["items"]) == 2
    for item in result["items"]:
        assert "display_name" in item
        assert "colour" in item


def test_compare_tool_handles_1_item_gracefully(catalogue_df):
    ids = catalogue_df["article_id"].iloc[:1].tolist()
    result = compare_items(ids, catalogue_df)
    # Should still return whatever it found without raising
    assert result["n_items"] == 1


def test_compare_tool_truncates_to_5(catalogue_df):
    ids = catalogue_df["article_id"].iloc[:6].tolist()
    result = compare_items(ids, catalogue_df)
    assert result["n_items"] <= 5


def test_compare_tool_handles_unknown_id(catalogue_df):
    ids = ["DOES_NOT_EXIST_001", catalogue_df["article_id"].iloc[0]]
    result = compare_items(ids, catalogue_df)
    # Should skip the unknown id and return the one valid item
    assert result["n_items"] == 1


# ---------------------------------------------------------------------------
# apply_filter
# ---------------------------------------------------------------------------

def test_apply_filter_merges_correctly():
    existing = {"colour_group_name": "Black"}
    updated = apply_filter(existing, "product_type_name", "Dress")
    assert updated == {"colour_group_name": "Black", "product_type_name": "Dress"}


def test_apply_filter_overwrites_same_key():
    existing = {"colour_group_name": "Black"}
    updated = apply_filter(existing, "colour_group_name", "Blue")
    assert updated == {"colour_group_name": "Blue"}


def test_apply_filter_ignores_invalid_key():
    existing = {"colour_group_name": "Black"}
    updated = apply_filter(existing, "price_range", "cheap")
    assert updated == existing  # unchanged


def test_apply_filter_all_valid_keys():
    for key in VALID_FACET_KEYS:
        result = apply_filter({}, key, "test_value")
        assert result == {key: "test_value"}


# ---------------------------------------------------------------------------
# clarify
# ---------------------------------------------------------------------------

def test_clarify_returns_question():
    result = clarify("What is your budget?")
    assert result == {"clarification_question": "What is your budget?"}
