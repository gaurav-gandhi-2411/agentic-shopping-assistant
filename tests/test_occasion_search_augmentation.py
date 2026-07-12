"""RED 5b/D — "something for a wedding" must return wedding-appropriate items
(sarees/lehengas/kurtas/anarkalis), not generic accessories/footwear that merely
mention "wedding" in their product description.

Investigation note (see final report): no in-process crash reproduced for this
query on the current codebase/unified-index state despite extensive attempts
(direct graph.invoke, full WS TestClient round-trip) — _OUTFIT_INTENT_RE never
matches "something for a wedding" (no outfit-building verb), so router_backend
.decide()/outfit_node are never reached; the query is always deterministic search.
This test targets the concrete, verifiable half of RED D's ask: search QUALITY.

Root cause: IntentParser already extracts occasion="wedding_guest" deterministically,
but router_node never consumed that field anywhere, and the LLM's own SEASONAL/
OCCASION QUERY REWRITING guidance (ROUTER_PROMPT) never applies on this fully
deterministic path. search_node now appends occasion-appropriate garment terms to
the retrieval query when no explicit garment type is present, mirroring that LLM
guidance without depending on the LLM at all.

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

_WEDDING_APPROPRIATE_TYPES = {
    "dress", "saree", "lehenga", "anarkali", "kurta", "kurti", "sherwani",
    "sharara", "coord",
}


class _MockLLM:
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


@pytest.mark.requires_index
def test_something_for_a_wedding_returns_ethnic_wear(
    _unified_index: tuple[HybridRetriever, pd.DataFrame],
) -> None:
    """"something for a wedding" (no explicit garment) must not crash and must
    return items dominated by wedding-appropriate ethnic wear.
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM(["Here you go."] * 5)
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    state = _blank_state("something for a wedding", memory)
    result = agent.invoke(state)  # must not raise

    items = result.get("retrieved_items", [])
    assert items, "expected items to be returned for a wedding query"
    types = [it.get("product_type", "").lower() for it in items]
    appropriate = sum(1 for t in types if t in _WEDDING_APPROPRIATE_TYPES)
    assert appropriate >= len(items) * 0.6, (
        f"expected majority wedding-appropriate ethnic wear, got product_types={types}"
    )


@pytest.mark.requires_index
def test_lehenga_for_sangeet_keeps_lehenga_not_sherwani(
    _unified_index: tuple[HybridRetriever, pd.DataFrame],
) -> None:
    """2026-07-12 regression: the occasion-term injection above appends
    "lehenga sherwani kurta embellished festive" to the RETRIEVAL query for any
    sangeet query with no garment-type filter yet set, purely to broaden dense/
    BM25 recall. That augmented string was then reused by the facet
    auto-extraction step, so "sherwani" (men's garment, longer string, sorts
    first) could silently win the length-sorted facet match over the user's own
    literal "lehenga" -- hard-filtering a "lehenga for sangeet" query to
    sherwanis. Facet extraction must read the user's actual words, never a
    retrieval-widening string, regardless of which happens to be longer.
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM(["Here you go."] * 5)
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    state = _blank_state("lehenga for sangeet", memory)
    result = agent.invoke(state)

    items = result.get("retrieved_items", [])
    assert items, "expected items to be returned for 'lehenga for sangeet'"
    types = {it.get("product_type", "").lower() for it in items}
    assert "sherwani" not in types, (
        f"'lehenga for sangeet' must never hard-filter to sherwani, got product_types={types}"
    )


@pytest.mark.requires_index
def test_haldi_outfit_not_hard_filtered_to_single_injected_type(
    _unified_index: tuple[HybridRetriever, pd.DataFrame],
) -> None:
    """Companion regression to the sangeet case above: a garment-type-less
    occasion query ("haldi outfit for women") must NOT be hard-filtered down
    to a single garment type accidentally picked up from the occasion-term
    injection's own word list (previously always resolved to "lehenga", the
    longest word in the haldi/mehendi occasion-term list) -- RED 5b/D's intent
    was to broaden retrieval across occasion-appropriate types, not narrow it
    to one.
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM(["Here you go."] * 5)
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    state = _blank_state("haldi outfit for women", memory)
    result = agent.invoke(state)

    items = result.get("retrieved_items", [])
    assert items, "expected items to be returned for 'haldi outfit for women'"
    types = {it.get("product_type", "").lower() for it in items}
    assert len(types) > 1, (
        f"'haldi outfit for women' must not be hard-filtered to a single garment "
        f"type, got product_types={types}"
    )
