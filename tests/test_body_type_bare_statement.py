"""Wave 7 hang fix — a bare body-type STATEMENT ("I have an inverted triangle
silhouette") sent as the FIRST and ONLY message of a fresh session must
complete the turn with a real, non-empty reply — never fall through to a
meaningless product search of unrelated items.

Root cause (see src/agents/graph.py's router_node comment tagged "Wave 7
hang fix"): the deterministic router correctly classified the message as
conversational (action="respond"), but route_decision's LLM-hallucination
guard force-converted ANY first-call "respond" with no retrieved_items into
"search" regardless of why "respond" was chosen — sending the pure shape
statement through search_node with zero filters, retrieving semantically
unrelated items, then asking the LLM to describe them as if relevant. This
is exactly the message shape the photo body-shape confirm button sends
(frontend/lib/poseShape.ts's bodyShapeMessage()) with no occasion attached.

Uses the real unified index (requires_index) — same harness pattern as
tests/test_occasion_outfit_and_refinement.py.
"""
from __future__ import annotations

import json
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
    """Never actually called for this repro — the fix short-circuits before
    any LLM round-trip — but the graph constructor requires an LLMClient.
    """

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        return "{}"

    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
        yield "{}"

    def chat(self, messages: list[dict], **kwargs) -> str:
        return "{}"

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        yield "{}"


@pytest.fixture(scope="module")
def _unified_index() -> tuple[HybridRetriever, pd.DataFrame]:
    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _MINIMAL_CONFIG)
    return retriever, catalogue_df


def _blank_state(query: str, memory) -> dict:
    """Mirrors api/routes/chat.py::_build_invoke_state for a brand new session
    (no prior messages, no retrieved_items, no filters).
    """
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


@pytest.mark.requires_index
@pytest.mark.parametrize(
    "query",
    [
        "I have a pear silhouette",
        "I have an apple silhouette",
        "I have an hourglass silhouette",
        "I have a rectangle silhouette",
        "I have an inverted triangle silhouette",
    ],
)
def test_bare_body_type_statement_completes_with_ack_not_search(
    _unified_index: tuple[HybridRetriever, pd.DataFrame], query: str
) -> None:
    """The exact live repro: fresh session, single message, no occasion/garment.

    Must never route to "search" (which is how the hang manifested downstream
    — a real LLM streaming call over an irrelevant-items prompt) and must
    return a non-empty reply within the graph invocation itself (no pending
    external round-trip required to produce SOME response text).
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM()
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    state = _blank_state(query, memory)
    result = agent.invoke(state)

    plan = json.loads(result.get("current_plan") or "{}")
    assert plan.get("action") == "pending_answer", (
        f"expected a deterministic pending_answer (clarify) plan, got {plan!r} "
        f"— tool_calls={result.get('tool_calls')}"
    )
    assert plan.get("text"), "expected non-empty acknowledgement text"
    assert not result.get("retrieved_items"), (
        "a bare body-type statement must never trigger a product search"
    )


@pytest.mark.requires_index
def test_body_type_stated_with_occasion_still_composes_outfit(
    _unified_index: tuple[HybridRetriever, pd.DataFrame],
) -> None:
    """Regression guard: body type + occasion in the SAME message (the
    existing P3 flow) must still compose an outfit — the new bare-statement
    short-circuit must never fire when a genuine product/occasion signal is
    present in the same turn.
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM()
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    query = "I'm pear-shaped, sangeet look under 8000"
    state = _blank_state(query, memory)
    result = agent.invoke(state)

    assert result.get("look_id"), (
        f"expected look_id for body-type + occasion turn, "
        f"tool_calls={result.get('tool_calls')}"
    )
