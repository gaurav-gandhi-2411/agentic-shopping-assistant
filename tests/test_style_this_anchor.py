"""RED 2b/3/B3c — "Style this <item>" must deterministically compose a look around the
referenced session item instead of running a plain search.

Root cause (pre-fix): graph.py's outfit-intent gate required _OUTFIT_INTENT_RE to match
AND `not intent.garment_type`. Item names embed garment nouns (e.g. "...Shirt"), so
IntentParser always extracts a garment_type from them, and the veto fires — the query
falls through to a plain product search, never reaching the LLM router or outfit_node.

Fix: detect an explicit anchor reference ("style this ...", "what goes with the/this ...")
FIRST, resolve the referenced item from session retrieved_items via case/whitespace-
insensitive substring matching, and build the outfit plan deterministically — bypassing
both the garment_type veto AND the LLM router entirely for this fully-determined case.

Uses the real unified index (requires_index) so seed_article_id resolves to an actual
catalogue row inside compose_outfit.
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

# Real unified-catalogue article — article_id="7624165523678".  Used verbatim so
# compose_outfit's seed resolution (catalogue_df.set_index("article_id").loc[...])
# succeeds against the real index, exactly as it would in production.
_ANCHOR_ARTICLE_ID = "7624165523678"
_ANCHOR_PROD_NAME = "Men White Semi- Formal Shirt"


class _MockLLM:
    """Cycles through canned responses. Deliberately returns a WRONG (non-outfit)
    router decision so the test only passes if the anchor path is fully deterministic
    and never delegates to the LLM router for this query.
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


def _make_session_shirt_items() -> list[dict]:
    """5 session shirt items, one of which is the exact anchor the frontend would
    reference via "Style this {prod_name}".
    """
    filler_names = [
        ("F1", "Men Navy Blue Semi Formal Shirt", "men"),
        ("F2", "Men Sky Blue Semi Formal Shirt", "men"),
        ("F3", "Men Checked Casual Shirt", "men"),
        ("F4", "Men Striped Formal Shirt", "men"),
    ]
    items = [
        {
            "article_id": aid,
            "prod_name": name,
            "display_name": name,
            "colour": "Blue",
            "product_type": "shirt",
            "department": "Men",
            "detail_desc": "",
            "image_url": "https://example.com/img.jpg",
            "score": 0.8,
            "store": "globalrepublic",
            "price_inr": 999.0,
            "pdp_handle": "some-shirt",
            "gender": "men",
        }
        for aid, name, _ in filler_names
    ]
    items.append(
        {
            "article_id": _ANCHOR_ARTICLE_ID,
            "prod_name": _ANCHOR_PROD_NAME,
            "display_name": f"{_ANCHOR_PROD_NAME} (White shirt)",
            "colour": "White",
            "product_type": "shirt",
            "department": "Men",
            "detail_desc": "",
            "image_url": "https://example.com/anchor.jpg",
            "score": 0.9,
            "store": "globalrepublic",
            "price_inr": 1399.0,
            "pdp_handle": "men-white-semi-formal-shirt",
            "gender": "men",
        }
    )
    return items


def _blank_state_with_session(query: str, session_items: list[dict], memory) -> dict:
    return {
        "messages": [{"role": "user", "content": query}],
        "user_query": query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": session_items,
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
def test_style_this_named_item_composes_look(
    _unified_index: tuple[HybridRetriever, pd.DataFrame],
) -> None:
    """"Style this Men White Semi- Formal Shirt" must compose a look anchored on the
    exact session item, not fall through to a plain search.
    """
    retriever, catalogue_df = _unified_index
    # Wrong-on-purpose canned LLM response: if the router ever delegates to the LLM
    # for this query, the test fails because "search" != "outfit".
    llm = _MockLLM([json.dumps({"action": "search", "query": "shirt"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    session_items = _make_session_shirt_items()
    state = _blank_state_with_session(
        f"Style this {_ANCHOR_PROD_NAME}", session_items, memory
    )

    result = agent.invoke(state)

    assert result.get("look_id"), (
        f"expected a non-null look_id, got tool_calls={result.get('tool_calls')}"
    )
    complements = [
        it for it in result.get("retrieved_items", [])
        if it.get("_role") == "complement"
    ]
    assert len(complements) >= 2, (
        f"expected >=2 complements with slot_role complement, got {complements}"
    )
    assert result.get("look_gender") == "men", (
        f"expected look_gender='men' from the matched anchor item, "
        f"got {result.get('look_gender')}"
    )


@pytest.mark.requires_index
def test_style_this_recovers_office_occasion_from_history(
    _unified_index: tuple[HybridRetriever, pd.DataFrame],
) -> None:
    """Phase B task 1 — occasion propagation. Root cause (pre-fix): the
    style-anchor branch (graph.py router_node) used `state.get("occasion")`,
    which is NEVER persisted across turns (see
    _reconstruct_occasion_from_history's docstring — only retrieved_items/
    filters/messages survive in the session dict) — so every "Style this"
    click silently defaulted to occasion="casual", regardless of what the
    prior search turn asked for. Live-proven: "black top for office for
    women" -> Style this -> outfit composed with occasion="casual", dropping
    the office formality gate and letting a denim mini skirt into the bottom
    slot.

    Fix: reconstruct occasion from conversation history the same way the
    look-refinement and partner-look branches already do. This test asserts
    the router's resolved plan carries occasion="office" — the composed
    look's bottom-slot correctness for office is covered separately by the
    formality-gate tests in test_phase_b_gender_slot_coherence.py.
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "shirt"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    session_items = _make_session_shirt_items()
    query = f"Style this {_ANCHOR_PROD_NAME}"
    state = _blank_state_with_session(query, session_items, memory)
    # Simulate the prior turn's occasion-bearing user message still present in
    # session history, as api/routes/chat.py::_persist_result accumulates it.
    state["messages"] = [
        {"role": "user", "content": "black top for office for women"},
        {"role": "assistant", "content": "Here are some office tops."},
        {"role": "user", "content": query},
    ]

    result = agent.invoke(state)

    router_decisions = [
        tc["router_decision"] for tc in result.get("tool_calls", []) if "router_decision" in tc
    ]
    assert router_decisions, f"expected a router_decision tool_call, got {result.get('tool_calls')}"
    assert router_decisions[0]["occasion"] == "office", (
        f"expected occasion='office' recovered from history, got {router_decisions[0]}"
    )


@pytest.mark.requires_index
def test_style_this_no_session_match_falls_back(
    _unified_index: tuple[HybridRetriever, pd.DataFrame],
) -> None:
    """When no session item matches the referenced name, the deterministic anchor
    path must NOT fire — falls back to existing behaviour (garment_type veto still
    applies since there's no anchor to resolve).
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "shirt"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    session_items = _make_session_shirt_items()
    state = _blank_state_with_session(
        "Style this Nonexistent Product Name", session_items, memory
    )

    result = agent.invoke(state)

    # No anchor found -> falls through to deterministic search (garment_type="shirt"
    # extracted from "Style this ... Product Name" is None here since no garment noun
    # is present) -- what matters is that it must NOT crash and must NOT silently
    # invent a look_id for a nonexistent anchor.
    assert result.get("look_id") is None
