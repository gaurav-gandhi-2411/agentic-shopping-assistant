"""Standalone in-process proof: "more sarees" pool-underflow fallback.

Runs a repeated-refinement conversation through the actual compiled agent
graph (no network hop — same in-process invoke/session-dict plumbing as
api/routes/chat.py::_build_invoke_state / _persist_result, mirrored here the
same way scripts/budget_persistence_smoke.py does):

  turn 1: "sarees"       -> initial search, 5 items shown
  turn 2: "more sarees"  -> refinement, prior_ids has 5 excluded
  turn 3: "more sarees"  -> prior_ids has 10 excluded
  turn 4: "more sarees"  -> prior_ids has 15 excluded
  turn 5: "more sarees"  -> prior_ids has 20 excluded — this is exactly the
           default fetch_k=20 retrieval window, so on the pre-fix code path
           the fresh-exclusion filter empties out (deterministic embedding
           search returns the SAME top-20 saree window every turn) and the
           `len(fresh) >= 2` guard fails, silently falling back to
           RE-SHOWING the same 20 already-seen items instead of a fresh page.
  turn 6: "more sarees"  -> same failure mode recurs one turn later without
           the fix (window still exhausted).

For each turn this prints how many of the 5 returned article_ids were NOT
in the previous turn's article_ids (i.e. genuinely fresh) and whether the
search_meta thin_category flag fired.

Run this TWICE:
  1. `git stash push -- src/agents/graph.py` (reverts to the pre-fix code)
     then run this script -> turns 5-6 show 0 fresh items (bug reproduced).
  2. `git stash pop` (restores the fix) then run this script again ->
     turns 5-6 show 5 fresh items (or an honest thin_category flag if the
     widened pool still can't clear 5), never a silent full repeat.
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

    turns = ["sarees"] + ["more sarees"] * 5
    prior_ids: set[str] = set()
    all_seen: set[str] = set()
    ok = True
    for i, q in enumerate(turns, start=1):
        result = _run_turn(agent, session, q)
        items = result.get("retrieved_items", [])
        ids = [it["article_id"] for it in items]
        fresh_vs_prior_turn = [aid for aid in ids if aid not in prior_ids]
        fresh_vs_all_seen = [aid for aid in ids if aid not in all_seen]
        thin_category = any(
            tc.get("search", {}).get("thin_category")
            for tc in result.get("tool_calls", [])
        )
        print(f"turn {i}: query={q!r}")
        print(f"  returned {len(ids)} items: {ids}")
        print(
            f"  fresh vs immediately-prior turn: {len(fresh_vs_prior_turn)}/5   "
            f"fresh vs ALL previously seen: {len(fresh_vs_all_seen)}/5   "
            f"thin_category={thin_category}"
        )
        # From turn 2 onward this is a refinement turn — expect either a full
        # fresh page, or an honestly-flagged thin_category underflow, but
        # NEVER a silent full repeat of items already shown last turn.
        if i >= 2:
            silently_repeated = len(fresh_vs_prior_turn) == 0 and not thin_category
            ok &= not silently_repeated
            print(
                f"  [assert] turn {i}: "
                f"{'FAIL (silent full repeat, no thin_category flag)' if silently_repeated else 'PASS'}"
            )
        prior_ids = set(ids)
        all_seen |= prior_ids

    print("\nVERDICT:", "ALL PASS" if ok else "SOME FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
