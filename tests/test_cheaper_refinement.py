"""RED 5c — "cheaper options" after a search must actually return cheaper items.

Root cause (pre-fix): the deterministic search-plan builder in graph.py only applied
a price_max filter when IntentParser extracted an EXPLICIT numeric budget ("under
₹1000"). "cheaper options" has no such number, so the refinement re-ran the search
completely unconstrained on price — embeddings have no price awareness, so the new
result set could easily have a HIGHER max price than the original turn (the live
regression: 2309 -> 3149).

Fix: when the query matches a cheaper/budget-refinement phrase and no explicit
budget was already given, cap price_max at 70% of the previous turn's max shown
price, while the garment/gender/colour context still carries forward normally via
IntentParser's session-context merge.

Uses the real unified index (requires_index).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from src.agents.graph import build_graph
from src.memory.conversation import ConversationMemory
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.sparse_search import SparseRetriever

UNIFIED_DIR = Path("data/processed/unified")

_MINIMAL_CONFIG: dict = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
    "retrieval": {
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dense_dim": 384,
        "rrf_k": 60,
        "top_k": 50,
        "final_k": 10,
        "store_diversity": 0.2,
    },
}


class _MockLLM:
    """IntentParser handles routing deterministically for plain product queries, so
    these canned responses only need to satisfy respond_node's token generation.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def _next(self) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        return self._next()

    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
        yield self._next()

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        yield self._next()


@pytest.fixture(scope="module")
def _unified_index() -> tuple[HybridRetriever, pd.DataFrame]:
    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _MINIMAL_CONFIG)
    return retriever, catalogue_df


def _blank_state(query: str, memory) -> dict:
    return {
        "messages": [{"role": "user", "content": query}],
        "user_query": query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": [],
        "filters": {},
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
        "out_of_catalogue": False,
        "excluded_colours": None,
        "anchor_article_id": None,
        "outfit_rationale": None,
        "outfit_variants": None,
        "_memory": memory,
    }


def _next_turn_state(prior_result: dict, query: str, memory) -> dict:
    return {
        "messages": prior_result.get("messages", []) + [{"role": "user", "content": query}],
        "user_query": query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": prior_result.get("retrieved_items", []),
        "filters": prior_result.get("filters", {}),
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
        "out_of_catalogue": False,
        "excluded_colours": prior_result.get("excluded_colours"),
        "anchor_article_id": prior_result.get("anchor_article_id"),
        "outfit_rationale": None,
        "outfit_variants": None,
        "_memory": memory,
    }


@pytest.mark.requires_index
def test_cheaper_options_caps_price_below_prior_turn(
    _unified_index: tuple[HybridRetriever, pd.DataFrame],
) -> None:
    """Turn 1 "black dress for women" then turn 2 "cheaper options": every turn-2
    item's price must be <= 70% of turn-1's max price, and garment/gender context
    (dress, women) must be preserved.
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM(["Here you go."] * 10)
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    turn1_state = _blank_state("black dress for women", memory)
    turn1_result = agent.invoke(turn1_state)
    turn1_items = turn1_result.get("retrieved_items", [])
    turn1_prices = [it["price_inr"] for it in turn1_items if it.get("price_inr")]
    assert turn1_prices, "precondition: turn 1 must return priced items"
    turn1_max = max(turn1_prices)

    turn2_state = _next_turn_state(turn1_result, "cheaper options", memory)
    turn2_result = agent.invoke(turn2_state)
    turn2_items = turn2_result.get("retrieved_items", [])

    assert turn2_items, "expected turn 2 to return items"
    cap = turn1_max * 0.7
    turn2_prices = [it.get("price_inr") for it in turn2_items]
    assert all(p is not None and p <= cap for p in turn2_prices), (
        f"expected all turn-2 prices <= cap={cap:.0f} (70% of turn-1 max={turn1_max:.0f}), "
        f"got prices={turn2_prices}"
    )
    turn2_types = {it.get("product_type", "").lower() for it in turn2_items}
    assert turn2_types == {"dress"}, f"expected garment context (dress) preserved, got {turn2_types}"
