"""Tests for the history-aware, stylist-quality customer-facing response fix.

Covers:
- Successful search turns no longer use the canned ONE_SENTENCE_PROMPT — the
  richer RESPOND_PROMPT (2-3 sentences, grounded, no fabricated attributes) is
  used for BOTH product-search and conversational turns.
- Recent conversation history (via graph._format_messages) is fed into the
  customer-facing prompt so follow-up turns can reference earlier turns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest

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


class _CapturingLLM:
    """Records every prompt passed to generate(); returns a fixed canned reply."""

    def __init__(self, reply: str = "Great pick! Here's why it works for you.") -> None:
        self.prompts: list[str] = []
        self._reply = reply

    def generate(self, prompt: str, system: str = None, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        return self._reply

    def generate_stream(self, prompt: str, system: str = None, **kwargs: Any) -> Iterator[str]:
        self.prompts.append(prompt)
        yield self._reply

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        return self._reply

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        yield self._reply


@pytest.fixture(scope="module")
def _unified_index() -> tuple:
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _MINIMAL_CONFIG)
    return retriever, catalogue_df


def _blank_state(query: str, memory: Any) -> dict:
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


def _next_turn_state(prior_result: dict, query: str, memory: Any) -> dict:
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
def test_successful_search_reply_uses_stylist_prompt_not_one_sentence_cap(
    _unified_index: tuple,
) -> None:
    """A successful product search must be answered with the stylist-quality
    RESPOND_PROMPT (2-3 sentences), never the retired one-sentence-cap prompt.
    """
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever, catalogue_df = _unified_index
    llm = _CapturingLLM()
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

    state = _blank_state("red dress", memory)
    result = agent.invoke(state)

    assert result.get("retrieved_items"), "precondition: search must return items"
    assert llm.prompts, "expected respond_node to call llm.generate at least once"
    respond_prompt = llm.prompts[-1]
    assert "exactly ONE sentence" not in respond_prompt, (
        "the retired one-sentence-cap prompt must no longer be used for "
        "successful searches"
    )
    assert "2-3 sentences" in respond_prompt


@pytest.mark.requires_index
def test_followup_turn_prompt_includes_prior_conversation_history(
    _unified_index: tuple,
) -> None:
    """A follow-up turn's customer-facing prompt must include recent conversation
    history (prior user message + assistant reply) so the LLM can reference
    earlier turns naturally.
    """
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever, catalogue_df = _unified_index
    llm = _CapturingLLM(reply="Great pick! Here's why it works for you.")
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

    turn1_state = _blank_state("red dress", memory)
    turn1_result = agent.invoke(turn1_state)
    assert turn1_result.get("retrieved_items"), "precondition: turn 1 must return items"

    turn2_state = _next_turn_state(turn1_result, "what about in blue", memory)
    turn2_result = agent.invoke(turn2_state)
    assert turn2_result.get("retrieved_items"), "precondition: turn 2 must return items"

    turn2_prompt = llm.prompts[-1]
    assert "red dress" in turn2_prompt, (
        "turn 2's customer-facing prompt must include turn 1's user message in "
        "its conversation history section"
    )
    assert "Great pick" in turn2_prompt, (
        "turn 2's customer-facing prompt must include turn 1's assistant reply in "
        "its conversation history section"
    )
