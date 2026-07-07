"""RED 2c — first-turn occasion-driven outfit requests and look-refinement follow-ups
must deterministically produce a look_id, without depending on the LLM router to
free-parse occasion/gender out of raw text.

Turn 1: "put together a casual look for women" (no prior items) must compose a look.
Turn 2: "Make this look more formal" (chip text) on the SAME conversation must
recompose a NEW look (>=2 items + look_id), reconstructing the anchor from the
seed item still present in session retrieved_items and the occasion from turn 1's
message text (session persistence only carries retrieved_items/filters/messages
across turns — see api/routes/chat.py::_persist_result).

Uses the real unified index (requires_index).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from src.agents.graph import (
    _OCCASION_LOOK_RE,
    _OUTFIT_INTENT_RE,
    _OUTFIT_OCCASION_RE,
    build_graph,
)
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
    """Deliberately returns a WRONG (non-outfit) router decision + non-JSON rationale
    fallback text — proves the routing itself is fully deterministic, never relying
    on the LLM to correctly classify the occasion request or the refinement follow-up.
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
    """Mirrors api/routes/chat.py::_build_invoke_state's session-persistence
    semantics: only messages/retrieved_items/filters survive across turns.
    """
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
def test_first_turn_casual_look_for_women_composes(
    _unified_index: tuple[HybridRetriever, pd.DataFrame],
) -> None:
    """"put together a casual look for women" on turn 1 (no prior items) must
    compose a look with a non-null look_id and >=1 complement.
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "casual top"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    state = _blank_state("put together a casual look for women", memory)
    result = agent.invoke(state)

    assert result.get("look_id"), f"expected look_id, tool_calls={result.get('tool_calls')}"
    assert result.get("look_gender") == "women"
    items = result.get("retrieved_items", [])
    assert len(items) >= 2, f"expected seed + >=1 complement, got {items}"


@pytest.mark.requires_index
def test_make_this_look_more_formal_recomposes_new_look(
    _unified_index: tuple[HybridRetriever, pd.DataFrame],
) -> None:
    """Turn 2 "Make this look more formal" on the SAME conversation must produce a
    NEW look (>=2 items + look_id), reconstructing occasion from turn-1 history and
    the anchor from the seed item still present in session retrieved_items.
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "casual top"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    turn1_state = _blank_state("put together a casual look for women", memory)
    turn1_result = agent.invoke(turn1_state)
    assert turn1_result.get("look_id"), "precondition: turn 1 must compose a look"

    turn2_state = _next_turn_state(turn1_result, "Make this look more formal", memory)
    turn2_result = agent.invoke(turn2_state)

    assert turn2_result.get("look_id"), (
        f"expected a new look_id on the refinement turn, "
        f"tool_calls={turn2_result.get('tool_calls')}"
    )
    turn2_items = turn2_result.get("retrieved_items", [])
    assert len(turn2_items) >= 2, f"expected >=2 items on the refinement turn, got {turn2_items}"


# ---------------------------------------------------------------------------
# Phase B task 3: "<occasion> look" phrasing must route to outfit composition.
# ---------------------------------------------------------------------------


def _routes_to_outfit(query: str) -> bool:
    """Mirrors graph.py router_node's RED 2c first-turn gate condition:
    `_OUTFIT_OCCASION_RE.search(raw_q) and (_OUTFIT_INTENT_RE.search(raw_q) or
    _OCCASION_LOOK_RE.search(raw_q))`.
    """
    return bool(
        _OUTFIT_OCCASION_RE.search(query)
        and (_OUTFIT_INTENT_RE.search(query) or _OCCASION_LOOK_RE.search(query))
    )


class TestOccasionLookRoutingRegex:
    """Root cause (pre-fix): "office look for women" carries no
    _OUTFIT_INTENT_RE action verb ("outfit", "style this/me/it", "complete
    the look", ...) — bare "look" alone was never recognised as an outfit
    action, so this phrasing fell through to a plain search instead of
    outfit composition. "casual outfit for men"/"an office outfit" already
    routed correctly pre-fix since "outfit" itself IS in _OUTFIT_INTENT_RE —
    included here as regression guards, not as newly-fixed cases.
    """

    def test_office_look_for_women_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("office look for women") is True

    def test_wedding_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("wedding look") is True

    def test_casual_outfit_for_men_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("casual outfit for men") is True

    def test_an_office_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("an office outfit") is True

    def test_look_for_black_dresses_is_search_not_outfit(self) -> None:
        """Negative: no occasion word directly precedes "look" here (in fact
        no occasion word at all) — must remain a plain product search."""
        assert _routes_to_outfit("look for black dresses") is False

    def test_looking_for_shirts_is_search_not_outfit(self) -> None:
        assert _routes_to_outfit("looking for shirts") is False


@pytest.mark.requires_index
def test_office_look_for_women_direct_phrasing_composes_board(
    _unified_index: tuple[HybridRetriever, pd.DataFrame],
) -> None:
    """Phase B task 3 — integration proof: the direct phrasing "office look
    for women" (no prior turn, no explicit anchor) must compose an outfit
    board on its own, not fall through to a plain search."""
    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "office top"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    state = _blank_state("office look for women", memory)
    result = agent.invoke(state)

    assert result.get("look_id"), f"expected look_id, tool_calls={result.get('tool_calls')}"
    assert result.get("look_gender") == "women"
