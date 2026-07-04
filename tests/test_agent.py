"""
Agent graph tests.
Some require real Ollama; others use a mock LLM.
"""
import json
import os
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.graph import _parse_router_response, build_graph
from src.catalogue.loader import load_config
from src.llm.client import get_llm_client
from src.memory.conversation import ConversationMemory
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.sparse_search import SparseRetriever

SAVE_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Mock LLM helper
# ---------------------------------------------------------------------------

class MockLLM:
    """Cycles through a list of responses; repeats the last one when exhausted."""

    def __init__(self, responses: list[str]):
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


def _blank_state(query: str, **overrides) -> dict:
    state = {
        "messages": [{"role": "user", "content": query}],
        "user_query": query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": [],
        "filters": {},
        "final_answer": None,
        "iteration": 0,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Router JSON parsing (no LLM / graph needed)
# ---------------------------------------------------------------------------

def test_router_parses_valid_json():
    valid = '{"action": "search", "query": "black jacket", "filters": {}}'
    parsed = _parse_router_response(valid, "fallback")
    assert parsed["action"] == "search"
    assert parsed["query"] == "black jacket"


def test_router_parses_json_embedded_in_prose():
    prose = 'Sure! Here is my decision: {"action": "filter", "key": "colour_group_name", "value": "Blue"} done.'
    parsed = _parse_router_response(prose, "fallback")
    assert parsed["action"] == "filter"
    assert parsed["key"] == "colour_group_name"


def test_router_fallback_on_bad_json():
    garbage = "I think we should look for some nice jackets!"
    parsed = _parse_router_response(garbage, "black jacket")
    assert parsed["action"] == "search"
    assert parsed["query"] == "black jacket"


def test_router_fallback_on_missing_action():
    no_action = '{"query": "jacket"}'
    parsed = _parse_router_response(no_action, "default query")
    assert parsed["action"] == "search"
    assert parsed["query"] == "default query"


def test_default_router_is_llm_only():
    from src.agents.router import LLMRouterBackend, get_router_backend
    cfg = load_config()
    assert cfg.get("router", {}).get("provider") == "llm", (
        "config.yaml router.provider must be 'llm' for production"
    )
    backend = get_router_backend(cfg, llm=None)
    assert isinstance(backend, LLMRouterBackend), (
        f"Expected LLMRouterBackend but got {type(backend).__name__}"
    )


# ---------------------------------------------------------------------------
# Max iterations cap (mock LLM, always returns search)
# ---------------------------------------------------------------------------

@pytest.mark.requires_index
def test_agent_max_iterations_cap(config, retriever, catalogue_df):
    # LLM always asks to search — agent must terminate with a final answer.
    # The respond guard (search+results → respond) fires before the iteration cap,
    # so iteration will be 1 even though max_iterations=2.
    always_search = MockLLM(
        [json.dumps({"action": "search", "query": "jacket"})] * 20
        + ["A great jacket for you!"]
    )
    test_config = {**config, "agent": {**config["agent"], "max_iterations": 2}}
    memory = ConversationMemory(always_search, test_config)
    agent = build_graph(retriever, catalogue_df, always_search, test_config)

    result = agent.invoke(_blank_state("show me jackets", _memory=memory))

    assert result["final_answer"] is not None, "Expected final_answer to be set after cap"
    assert result["iteration"] >= 1, "Expected at least one iteration before termination"


# ---------------------------------------------------------------------------
# Agent-loop router fast-path (mock LLM, no external dependencies)
# ---------------------------------------------------------------------------

class _CallCountingLLM(MockLLM):
    """MockLLM that also tracks how many times generate() is called."""

    def __init__(self, responses: list[str]):
        super().__init__(responses)
        self.call_count = 0

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        self.call_count += 1
        return super().generate(prompt, system, **kwargs)


def _make_counting_memory(llm, config):
    return ConversationMemory(llm, config)


@pytest.mark.requires_index
def test_fast_path_search_skips_alr(config, retriever, catalogue_df):
    """IntentParser routes product queries deterministically; the LLM router is skipped."""
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    # F3 IntentParser: product queries are routed without calling the LLM router.
    # 2 LLM calls remain: (1) search_node's reranker (src/agents/reranker.rerank, in
    # place since "Phase 7e: LLM reranker") fires whenever the fetched candidate pool
    # exceeds top_k, which is the normal case for a broad "blue dresses" query against
    # the ~20k-row catalogue; (2) the respond node. Neither call goes through the LLM
    # *router* — that is still fully bypassed by IntentParser, which is what this test
    # actually verifies via the total staying flat instead of growing with an extra
    # router call.
    llm = _CallCountingLLM([
        "Here are some blue dresses for you.",
    ])
    memory = _make_counting_memory(llm, config)
    agent = build_graph(retriever, catalogue_df, llm, config)

    result = agent.invoke(_blank_state("show me blue dresses", _memory=memory))

    assert result["final_answer"] is not None
    # 2 LLM calls: reranker + respond. Router is deterministic (IntentParser), no LLM.
    assert llm.call_count == 2, (
        f"Expected 2 LLM calls (reranker + respond) but got {llm.call_count}; "
        "IntentParser should route product queries without calling the LLM router"
    )


@pytest.mark.requires_index
def test_fast_path_compare_skips_alr(config, retriever, catalogue_df):
    """Compare path: route_decision forces compare; ALR fast-path skips LLM after compare."""
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    # F3: IntentParser routes "compare these two" as non-product → respond, but
    # route_decision's compare guard overrides to compare when retrieved_items exist.
    # After compare runs, the ALR fast-path (last_tool==compare → respond) skips the LLM router.
    # Total LLM calls: 1 (respond node only — router_node fast-path covers post-compare).
    llm = _CallCountingLLM([
        "Here is the comparison result.",
    ])
    # Seed items in state so compare_node has something to work with
    seed_items = [
        {"article_id": "A001", "display_name": "Jacket A", "colour": "Black",
         "product_type": "Jacket", "department": "Ladieswear", "description": ""},
        {"article_id": "A002", "display_name": "Jacket B", "colour": "Black",
         "product_type": "Jacket", "department": "Ladieswear", "description": ""},
    ]
    memory = _make_counting_memory(llm, config)
    agent = build_graph(retriever, catalogue_df, llm, config)

    result = agent.invoke(_blank_state(
        "compare these two",
        retrieved_items=seed_items,
        _memory=memory,
    ))

    assert result["final_answer"] is not None
    # Only 1 LLM call: respond node. Router is deterministic; fast-path covers post-compare.
    assert llm.call_count == 1, (
        f"Expected 1 LLM call (respond only) but got {llm.call_count}; "
        "ALR fast-path should have skipped the LLM router after compare"
    )


@pytest.mark.requires_index
def test_fast_path_disabled_restores_alr(config, retriever, catalogue_df):
    """With AGENT_LOOP_FAST_PATH=false, IntentParser still routes deterministically."""
    os.environ["AGENT_LOOP_FAST_PATH"] = "false"
    try:
        # F3 IntentParser runs after both fast-path checks, so product queries are still
        # handled without calling the LLM router even when AGENT_LOOP_FAST_PATH=false.
        # 2 LLM calls remain: search_node's reranker (fires whenever the fetched
        # candidate pool exceeds top_k — see test_fast_path_search_skips_alr) + respond.
        llm = _CallCountingLLM([
            "Here are some blue dresses for you.",
        ])
        memory = _make_counting_memory(llm, config)
        agent = build_graph(retriever, catalogue_df, llm, config)

        result = agent.invoke(_blank_state("show me blue dresses", _memory=memory))

        assert result["final_answer"] is not None
        # 2 LLM calls: reranker + respond. IntentParser is deterministic regardless of
        # the fast-path flag, so no router call is added here either.
        assert llm.call_count == 2, (
            f"Expected 2 LLM calls (reranker + respond) but got {llm.call_count}; "
            "IntentParser should route product queries without an LLM router call "
            "even with fast-path disabled"
        )
    finally:
        os.environ["AGENT_LOOP_FAST_PATH"] = "true"


@pytest.mark.requires_index
def test_intent_parser_bypasses_filter_node(config, retriever, catalogue_df):
    """IntentParser routes 'dresses in blue' directly to search, bypassing filter node."""
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    # F3: IntentParser detects garment_type=dress + colour=Blue → routes straight to search.
    # The filter→search dance no longer fires for queries with both type and colour signals.
    # 2 LLM calls remain: search_node's reranker (fires whenever the fetched candidate
    # pool exceeds top_k — see test_fast_path_search_skips_alr) + respond. No filter-node
    # or router-node LLM call is made either way.
    llm = _CallCountingLLM([
        "Here are some blue dresses for you.",
    ])
    memory = _make_counting_memory(llm, config)
    agent = build_graph(retriever, catalogue_df, llm, config)

    result = agent.invoke(_blank_state("show me dresses in blue", _memory=memory))

    assert result["final_answer"] is not None
    # 2 LLM calls: reranker + respond. IntentParser routes directly to search, no
    # filter-node or router-node LLM call.
    assert llm.call_count == 2, (
        f"Expected 2 LLM calls (reranker + respond) but got {llm.call_count}; "
        "IntentParser should bypass the filter node and LLM router for queries with "
        "type+colour signals"
    )
    # Blue colour filter should be present in the search filters
    tool_calls = result.get("tool_calls", [])
    router_decisions = [tc["router_decision"] for tc in tool_calls if "router_decision" in tc]
    assert router_decisions, "Expected at least one router_decision tool call"
    first_plan = router_decisions[0]
    assert first_plan.get("action") == "search", (
        f"Expected action=search but got {first_plan.get('action')}"
    )


# ---------------------------------------------------------------------------
# End-to-end with real Ollama
# ---------------------------------------------------------------------------

@pytest.mark.requires_ollama
def test_agent_end_to_end_search(config, retriever, catalogue_df):
    llm = get_llm_client(config)
    memory = ConversationMemory(llm, config)
    agent = build_graph(retriever, catalogue_df, llm, config)

    result = agent.invoke(_blank_state("show me black jackets", _memory=memory))

    assert result["final_answer"], "Expected non-empty final_answer"
    assert len(result["final_answer"]) > 20, "Answer seems too short"
    # The agent should have retrieved some items and used them
    assert len(result["retrieved_items"]) > 0


@pytest.mark.requires_ollama
def test_agent_filter_query(config, retriever, catalogue_df):
    """Blue filter query should result in blue items."""
    llm = get_llm_client(config)
    memory = ConversationMemory(llm, config)
    agent = build_graph(retriever, catalogue_df, llm, config)

    result = agent.invoke(_blank_state("show me blue dresses", _memory=memory))

    assert result["final_answer"]
    # Retrieved items should be predominantly blue
    colours = [r["colour"].lower() for r in result["retrieved_items"]]
    blue_count = sum(1 for c in colours if "blue" in c)
    assert blue_count >= 1, f"Expected blue items, got colours: {colours}"
