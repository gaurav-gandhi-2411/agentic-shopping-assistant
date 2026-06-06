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


def test_fast_path_search_skips_alr(config, retriever, catalogue_df):
    """After search returns items, the ALR LLM call is skipped (fast-path fires)."""
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    # Responses: (1) router decides search, (2) respond node answer.
    # If ALR were called it would consume response[1] and push respond to response[2].
    llm = _CallCountingLLM([
        json.dumps({"action": "search", "query": "blue dress"}),
        "Here are some blue dresses for you.",
    ])
    memory = _make_counting_memory(llm, config)
    agent = build_graph(retriever, catalogue_df, llm, config)

    result = agent.invoke(_blank_state("show me blue dresses", _memory=memory))

    assert result["final_answer"] is not None
    # Only 2 LLM calls: initial router + respond node. ALR must NOT have fired.
    assert llm.call_count == 2, (
        f"Expected 2 LLM calls (router + respond) but got {llm.call_count}; "
        "ALR fast-path may not have fired"
    )


def test_fast_path_compare_skips_alr(config, retriever, catalogue_df):
    """After compare node runs, the ALR LLM call is skipped (fast-path fires)."""
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    # Responses: (1) initial router → search, (2) ALR after search (skipped by fast-path),
    # (3) router → compare, (4) ALR after compare (skipped), (5) respond.
    # With fast-path: (1) router→search, (2) respond after search, then the compare branch
    # never fires because respond ends the turn.
    # Use a query that has items then ask to compare via a second invoke.
    llm = _CallCountingLLM([
        json.dumps({"action": "compare", "article_ids": []}),
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
    # Only 2 LLM calls: initial router + respond node. ALR after compare must be skipped.
    assert llm.call_count == 2, (
        f"Expected 2 LLM calls (router + respond) but got {llm.call_count}; "
        "ALR fast-path may not have fired after compare"
    )


def test_fast_path_disabled_restores_alr(config, retriever, catalogue_df):
    """With AGENT_LOOP_FAST_PATH=false the ALR LLM call is NOT skipped."""
    os.environ["AGENT_LOOP_FAST_PATH"] = "false"
    try:
        # Responses: (1) router→search, (2) ALR→respond (the extra call), (3) respond answer.
        llm = _CallCountingLLM([
            json.dumps({"action": "search", "query": "blue dress"}),
            json.dumps({"action": "respond"}),
            "Here are some blue dresses for you.",
        ])
        memory = _make_counting_memory(llm, config)
        agent = build_graph(retriever, catalogue_df, llm, config)

        result = agent.invoke(_blank_state("show me blue dresses", _memory=memory))

        assert result["final_answer"] is not None
        # 3 LLM calls: router + ALR + respond when fast-path is disabled.
        assert llm.call_count == 3, (
            f"Expected 3 LLM calls (router + ALR + respond) but got {llm.call_count}"
        )
    finally:
        os.environ["AGENT_LOOP_FAST_PATH"] = "true"


def test_filter_then_search_hits_llm_router(config, retriever, catalogue_df):
    """filter→search is non-trivial: the ALR must call the LLM to generate the search query."""
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    # Responses: (1) router→filter, (2) ALR after filter must hit LLM to get search query,
    # (3) ALR after search (fast-path skips), (4) respond answer.
    llm = _CallCountingLLM([
        json.dumps({"action": "filter", "key": "colour_group_name", "value": "Blue"}),
        json.dumps({"action": "search", "query": "blue dress"}),
        "Here are some blue dresses for you.",
    ])
    memory = _make_counting_memory(llm, config)
    agent = build_graph(retriever, catalogue_df, llm, config)

    result = agent.invoke(_blank_state("show me dresses in blue", _memory=memory))

    assert result["final_answer"] is not None
    # 3 LLM calls: router → filter, ALR → search query gen, respond node.
    # Fast-path skips the ALR after search; filter ALR must NOT be skipped.
    assert llm.call_count == 3, (
        f"Expected 3 LLM calls (router→filter, ALR→search, respond) but got {llm.call_count}; "
        "fast-path may have incorrectly skipped the filter→search ALR"
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
