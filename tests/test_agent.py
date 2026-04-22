"""
Agent graph tests.
Some require real Ollama; others use a mock LLM.
"""
import sys
import json
from pathlib import Path
from typing import Iterator
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalogue.loader import load_config
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.sparse_search import SparseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.llm.client import get_llm_client
from src.memory.conversation import ConversationMemory
from src.agents.graph import build_graph, _parse_router_response

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


# ---------------------------------------------------------------------------
# Max iterations cap (mock LLM, always returns search)
# ---------------------------------------------------------------------------

def test_agent_max_iterations_cap(config, retriever, catalogue_df):
    # LLM always asks to search — the graph must still terminate via the cap.
    always_search = MockLLM(
        [json.dumps({"action": "search", "query": "jacket"})] * 20
        + ["A great jacket for you!"]
    )
    test_config = {**config, "agent": {**config["agent"], "max_iterations": 2}}
    memory = ConversationMemory(always_search, test_config)
    agent = build_graph(retriever, catalogue_df, always_search, memory, test_config)

    result = agent.invoke(_blank_state("show me jackets"))

    assert result["final_answer"] is not None, "Expected final_answer to be set after cap"
    assert result["iteration"] >= test_config["agent"]["max_iterations"]


# ---------------------------------------------------------------------------
# End-to-end with real Ollama
# ---------------------------------------------------------------------------

@pytest.mark.requires_ollama
def test_agent_end_to_end_search(config, retriever, catalogue_df):
    llm = get_llm_client(config)
    memory = ConversationMemory(llm, config)
    agent = build_graph(retriever, catalogue_df, llm, memory, config)

    result = agent.invoke(_blank_state("show me black jackets"))

    assert result["final_answer"], "Expected non-empty final_answer"
    assert len(result["final_answer"]) > 20, "Answer seems too short"
    # The agent should have retrieved some items and used them
    assert len(result["retrieved_items"]) > 0


@pytest.mark.requires_ollama
def test_agent_filter_query(config, retriever, catalogue_df):
    """Blue filter query should result in blue items."""
    llm = get_llm_client(config)
    memory = ConversationMemory(llm, config)
    agent = build_graph(retriever, catalogue_df, llm, memory, config)

    result = agent.invoke(_blank_state("show me blue dresses"))

    assert result["final_answer"]
    # Retrieved items should be predominantly blue
    colours = [r["colour"].lower() for r in result["retrieved_items"]]
    blue_count = sum(1 for c in colours if "blue" in c)
    assert blue_count >= 1, f"Expected blue items, got colours: {colours}"
