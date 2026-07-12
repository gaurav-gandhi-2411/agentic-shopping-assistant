"""Standalone in-process proof: budget (price_max) persists across turns.

Runs a 3-turn conversation through the actual compiled agent graph (no network
hop — same in-process invoke/session-dict plumbing as
api/routes/chat.py::_build_invoke_state / _persist_result, mirrored here the
same way tests/test_agent.py::_run_turn does for its multi-turn tests):

  turn 1: "kurta under 3000"        -> sets budget_max_inr=3000
  turn 2: "show blue ones"          -> colour swap, no budget mentioned
  turn 3: "show me more like this"  -> refinement, no budget mentioned
           ("more like this" alone is not classified as a product query by
           IntentParser's is_product_query heuristic — a separate, pre-existing
           gap unrelated to this fix — so "show me more like this" is used here
           to actually exercise a fresh retrieval on turn 3.)

Asserts price_max stays ~3000 in the retrieval filters on turns 2 and 3.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["AGENT_LOOP_FAST_PATH"] = "true"

import pandas as pd

from src.agents.graph import build_graph
from src.catalogue.loader import load_config
from src.memory.conversation import ConversationMemory
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.sparse_search import SparseRetriever

SAVE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "unified"


class MockLLM:
    """Cycles through canned responses; the deterministic IntentParser router
    path used here never actually calls the LLM, but build_graph requires one.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def _next(self) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str | None = None, **kwargs: object) -> str:
        return self._next()

    def generate_stream(
        self, prompt: str, system: str | None = None, **kwargs: object
    ) -> Iterator[str]:
        yield self._next()

    def chat(self, messages: list[dict], **kwargs: object) -> str:
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs: object) -> Iterator[str]:
        yield self._next()


def _run_turn(agent: object, session: dict, query: str) -> dict:
    """Invoke the graph for one turn and persist the result back into session.

    Mirrors api/routes/chat.py's _build_invoke_state / _persist_result exactly.
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


def main() -> int:
    config = load_config()
    catalogue_df = pd.read_parquet(SAVE_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, SAVE_DIR)
    sparse = SparseRetriever.load(config, SAVE_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)
    llm = MockLLM(["ok"] * 20)
    session = {
        "messages": [],
        "retrieved_items": [],
        "filters": {},
        "excluded_colours": None,
        "_memory": ConversationMemory(llm, config),
    }
    agent = build_graph(retriever, catalogue_df, llm, config)

    turns = ["kurta under 3000", "show blue ones", "show me more like this"]
    results = []
    for q in turns:
        result = _run_turn(agent, session, q)
        filters = result.get("filters", {})
        print(f"turn query={q!r}")
        print(f"  filters={filters}")
        print(f"  price_max in filters = {filters.get('price_max')}")
        results.append(filters)

    ok = True
    for i in (1, 2):
        price_max = results[i].get("price_max")
        turn_ok = price_max is not None and 2900 <= price_max <= 3100
        ok &= turn_ok
        print(
            f"[assert] turn {i + 1} price_max={price_max} "
            f"-> {'PASS' if turn_ok else 'FAIL'}"
        )

    print("\nVERDICT:", "ALL PASS" if ok else "SOME FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
