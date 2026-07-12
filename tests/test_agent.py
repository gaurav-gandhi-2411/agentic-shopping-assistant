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


# ---------------------------------------------------------------------------
# Multi-turn colour-only refinement must inherit from the MOST RECENT search
# turn, not from any earlier turn in a longer, mixed-gender conversation.
# Mirrors api/routes/chat.py's _build_invoke_state / _persist_result session
# plumbing exactly so this reproduces the live single-instance bug in-process.
#
# Uses the unified cross-store index (data/processed/unified) rather than the
# module-scoped `retriever`/`catalogue_df` fixtures above: those point at the
# legacy H&M-only sample (data/processed/catalogue.parquet), which predates
# the "gender" column entirely (see tests/test_hybrid_search_store_gender.py).
# Without a real gender column every gender-filtered search on that fixture
# returns 0 items (item_gender defaults to "unknown"), which cascades into the
# unrelated full-filter-drop fallback and masks the bug under test.
# ---------------------------------------------------------------------------

UNIFIED_SAVE_DIR = Path("data/processed/unified")


@pytest.fixture(scope="module")
def unified_catalogue_df():
    return pd.read_parquet(UNIFIED_SAVE_DIR / "catalogue.parquet")


@pytest.fixture(scope="module")
def unified_retriever(config, unified_catalogue_df):
    dense = DenseRetriever.load(config, UNIFIED_SAVE_DIR)
    sparse = SparseRetriever.load(config, UNIFIED_SAVE_DIR)
    return HybridRetriever(dense, sparse, unified_catalogue_df, config)


def _run_turn(agent, session: dict, query: str) -> dict:
    """Invoke the graph for one turn and persist the result back into session.

    Mirrors api.routes.chat._build_invoke_state / _persist_result so multi-turn
    tests exercise the exact same session-state shape the live API uses.
    """
    invoke_state = {
        "messages": session["messages"] + [{"role": "user", "content": query}],
        "user_query": query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": session["retrieved_items"],
        "filters": session["filters"],
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
        "out_of_catalogue": False,
        "excluded_colours": session.get("excluded_colours"),
        "anchor_article_id": session.get("anchor_article_id"),
        "outfit_rationale": None,
        "outfit_variants": None,
        "_memory": session["_memory"],
    }
    result = agent.invoke(invoke_state)
    session["messages"] = result.get("messages", session["messages"])
    session["retrieved_items"] = result.get("retrieved_items", session["retrieved_items"])
    session["filters"] = result.get("filters", session["filters"])
    if result.get("excluded_colours") is not None:
        session["excluded_colours"] = result["excluded_colours"]
    return result


@pytest.mark.requires_index
def test_colour_refinement_inherits_most_recent_turn_gender_and_garment(
    config, unified_retriever, unified_catalogue_df, monkeypatch
):
    """4-turn mixed-gender conversation: colour-only refinement must inherit from
    the MOST RECENT search turn (men/shirt), not from an earlier turn (women/dress).

    Reproduces the live bug: turn 4 ("in blue now") fell back to a generic
    women's search, losing both gender=men and garment_type=shirt carried from
    turn 3, whenever the fully-filtered search came up empty and search_node's
    progressive fallback had to drop the "gender" key.

    With ample real inventory (data/processed/unified) the fully-filtered
    combo (shirt + men + blue) never actually comes up empty, so the buggy
    fallback branch is never exercised end-to-end in practice here — the
    corrupted state is real (see turn-by-turn filters asserted below) but
    silently harmless as long as the "gender" key survives untouched. To
    reliably exercise the fallback branch itself (the mechanism through which
    the corrupted state becomes visibly wrong on live, sparser queries), this
    test forces every gender-filtered search to report zero results via
    monkeypatch, so search_node must fall through to the
    {product_type_name, index_group_name}-only candidate — the exact
    candidate that silently re-derives gender from a stale index_group_name.
    """
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    import src.agents.graph as graph_module
    from src.agents.tools import search_catalogue as real_search_catalogue

    def _fake_search_catalogue(query, filters, retriever, top_k):
        # Force a "zero results" outcome for any search where a gender constraint is
        # explicitly present (via the "gender" key) so search_node must fall through
        # its progressive-fallback ladder to the {product_type_name, index_group_name}
        # candidate — the one path that reconstructs gender from index_group_name
        # alone, with no "gender" key. That is the exact seam this test targets;
        # forcing it here removes any dependence on which items happen to fall
        # inside the real FAISS/BM25 top-k window for a given query.
        if filters and filters.get("gender"):
            return {"items": [], "query": query, "n_results": 0}
        return real_search_catalogue(query, filters, retriever, top_k)

    monkeypatch.setattr(graph_module, "search_catalogue", _fake_search_catalogue)

    llm = MockLLM(["ok"] * 20)
    session = {
        "messages": [],
        "retrieved_items": [],
        "filters": {},
        "excluded_colours": None,
        "_memory": ConversationMemory(llm, config),
    }
    agent = build_graph(unified_retriever, unified_catalogue_df, llm, config)

    def _genders(items: list[dict]) -> list[str]:
        return [it.get("gender", "").lower() for it in items]

    _run_turn(agent, session, "saree")
    _run_turn(agent, session, "black dress for women")
    turn3 = _run_turn(agent, session, "white shirt men")

    # Sanity: turn 3 must actually land on men's shirts before we test the refinement.
    turn3_types = [it.get("product_type", "").lower() for it in turn3["retrieved_items"]]
    turn3_genders = _genders(turn3["retrieved_items"])
    assert turn3_types and all("shirt" in t for t in turn3_types), (
        f"Precondition failed: turn 3 should be all shirts, got {turn3_types}"
    )
    assert turn3_genders and all(g == "men" for g in turn3_genders), (
        f"Precondition failed: turn 3 should be all men's, got {turn3_genders}"
    )

    turn4 = _run_turn(agent, session, "in blue now")

    filters = turn4.get("filters", {})
    assert filters.get("product_type_name", "").lower() == "shirt", (
        f"Expected garment_type carried forward as 'shirt', got filters={filters}"
    )
    assert filters.get("index_group_name") == "menswear", (
        f"Expected gender carried forward as 'men' (index_group_name=menswear), "
        f"got filters={filters}"
    )

    turn4_types = [it.get("product_type", "").lower() for it in turn4["retrieved_items"]]
    turn4_genders = _genders(turn4["retrieved_items"])
    assert turn4_types and all("shirt" in t for t in turn4_types), (
        f"Expected turn 4 items to stay shirts, got {turn4_types}"
    )
    assert turn4_genders and all(g == "men" for g in turn4_genders), (
        f"Expected turn 4 items to stay men's, got {turn4_genders}"
    )


# ---------------------------------------------------------------------------
# S3a fix: a genuine garment-type PIVOT must NOT inherit a stale colour filter
# from an unrelated earlier turn.
#
# Live-proven bug (phase-b browser retest, 2026-07-07): "black dress for
# women" -> "Style this <dress>" -> "white shirt for men" -> "Style this
# <shirt>" -> "style a kurta for sangeet for women" silently narrowed the
# sangeet-kurta search to only WHITE kurtas (colour_group_name="White" carried
# over from the unrelated men's-shirt turn), which returned far fewer results
# than the un-narrowed search and, on a sparser catalogue slice, can return 0.
# ---------------------------------------------------------------------------


@pytest.mark.requires_index
def test_garment_pivot_drops_stale_colour_filter(
    config, unified_retriever, unified_catalogue_df
):
    """A turn that pivots to a NEW garment type + gender must not inherit an
    unrelated earlier turn's colour filter, even though the intervening
    "Style this" turn never touched `filters` (outfit turns leave it as-is).
    """
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    llm = MockLLM(["ok"] * 20)
    session = {
        "messages": [],
        "retrieved_items": [],
        "filters": {},
        "excluded_colours": None,
        "_memory": ConversationMemory(llm, config),
    }
    agent = build_graph(unified_retriever, unified_catalogue_df, llm, config)

    turn1 = _run_turn(agent, session, "black dress for women")
    assert turn1["retrieved_items"], "precondition: turn 1 must return dresses"
    first_dress = turn1["retrieved_items"][0]
    _run_turn(agent, session, f"Style this {first_dress.get('prod_name')}")

    turn3 = _run_turn(agent, session, "white shirt for men")
    assert turn3["retrieved_items"], "precondition: turn 3 must return men's shirts"
    first_shirt = turn3["retrieved_items"][0]
    _run_turn(agent, session, f"Style this {first_shirt.get('prod_name')}")

    turn5 = _run_turn(agent, session, "style a kurta for sangeet for women")

    filters = turn5.get("filters", {})
    assert "colour_group_name" not in filters, (
        f"Expected the stale 'White' colour filter from turn 3 to be dropped "
        f"on this garment pivot, got filters={filters}"
    )
    assert filters.get("product_type_name", "").lower() == "kurta"
    assert filters.get("gender") == "women"

    turn5_items = turn5.get("retrieved_items", [])
    colours = {(it.get("colour") or "").lower() for it in turn5_items}
    assert len(colours) > 1, (
        f"Expected kurtas of more than one colour once the stale 'white' filter "
        f"is dropped, got colours={colours}"
    )
