"""
End-to-end smoke test — runs 4 multi-turn queries through the full agent.
Expects data/processed/ indices and Ollama with llama3.1:8b to be available.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.catalogue.loader import load_config
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.sparse_search import SparseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.llm.client import get_llm_client
from src.memory.conversation import ConversationMemory
from src.agents.graph import build_graph


def main():
    config = load_config()
    save_dir = Path("data/processed")

    print("Loading components...")
    df = pd.read_parquet(save_dir / "catalogue.parquet")
    dense = DenseRetriever.load(config, save_dir)
    sparse = SparseRetriever.load(config, save_dir)
    retriever = HybridRetriever(dense, sparse, df, config)
    llm = get_llm_client(config)
    memory = ConversationMemory(llm, config)
    agent = build_graph(retriever, df, llm, memory, config)
    print("Agent ready.\n")

    # (query, must_not_be_action) — set must_not_be to a string to assert that
    # action never appears in tool_calls for that turn.
    test_queries = [
        ("show me some black jackets",              None),
        ("something for summer, light and breathable", None),
        ("something more casual",                   "clarify"),  # follow-up refinement — must search
        ("compare the first two you showed me",     None),       # tests retrieved_items carry-over
        ("anything in blue instead?",               None),       # tests filter then re-search
    ]

    # State persisted across turns
    messages: list[dict] = []
    filters: dict = {}
    retrieved_items: list[dict] = []

    for query, must_not_be in test_queries:
        print("=" * 65)
        print(f"User: {query}")

        result = agent.invoke({
            "messages": messages + [{"role": "user", "content": query}],
            "user_query": query,
            "current_plan": None,
            "tool_calls": [],
            "retrieved_items": retrieved_items,
            "filters": filters,
            "final_answer": None,
            "iteration": 0,
        })

        answer = result.get("final_answer", "")
        tools_used = [list(t.keys())[0] for t in result.get("tool_calls", [])]

        print(f"Assistant: {answer}")
        print(f"[tools: {tools_used}]")

        if must_not_be:
            assert must_not_be not in tools_used, (
                f"FAIL: router used '{must_not_be}' on query '{query}'. "
                f"Tools: {tools_used}"
            )
            print(f"[PASS: '{must_not_be}' not triggered]")
        print()

        # Carry state forward for next turn
        messages = result.get("messages", messages)
        filters = result.get("filters", filters)
        retrieved_items = result.get("retrieved_items", retrieved_items)

    # -----------------------------------------------------------------------
    # Refinement dedup scenario: follow-up must return DIFFERENT items
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("SCENARIO: refinement dedup (summer dresses -> something more casual)")

    messages2: list[dict] = []
    filters2: dict = {}
    retrieved2: list[dict] = []

    for turn_query in ("show me summer dresses", "something more casual"):
        print(f"\nUser: {turn_query}")
        r = agent.invoke({
            "messages": messages2 + [{"role": "user", "content": turn_query}],
            "user_query": turn_query,
            "current_plan": None, "tool_calls": [],
            "retrieved_items": retrieved2, "filters": filters2,
            "final_answer": None, "iteration": 0,
        })
        tools_used2 = [list(t.keys())[0] for t in r.get("tool_calls", [])]
        new_items = r.get("retrieved_items", [])
        print(f"[tools: {tools_used2}]")
        print(f"Items: {[it['display_name'] for it in new_items]}")

        if turn_query == "something more casual":
            prev_ids = {it["article_id"] for it in retrieved2}
            new_ids = {it["article_id"] for it in new_items}
            overlap = len(prev_ids & new_ids)
            assert overlap < len(prev_ids), (
                f"FAIL: refinement returned identical items (overlap={overlap}/{len(prev_ids)})"
            )
            assert "clarify" not in tools_used2, (
                f"FAIL: router over-clarified on follow-up. Tools: {tools_used2}"
            )
            print(f"[PASS: overlap={overlap}/{len(prev_ids)}, no clarify]")

        messages2 = r.get("messages", messages2)
        filters2 = r.get("filters", filters2)
        retrieved2 = new_items

    # -----------------------------------------------------------------------
    # Outfit bundling scenario: show dress -> build outfit around it
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("SCENARIO: outfit bundling (black dress -> build an outfit around it)")

    messages3: list[dict] = []
    filters3: dict = {}
    retrieved3: list[dict] = []

    for turn_query in ("show me a black dress", "build an outfit around it"):
        print(f"\nUser: {turn_query}")
        r = agent.invoke({
            "messages": messages3 + [{"role": "user", "content": turn_query}],
            "user_query": turn_query,
            "current_plan": None, "tool_calls": [],
            "retrieved_items": retrieved3, "filters": filters3,
            "final_answer": None, "iteration": 0,
        })
        tools_used3 = [list(t.keys())[0] for t in r.get("tool_calls", [])]
        items3 = r.get("retrieved_items", [])
        print(f"[tools: {tools_used3}]")
        print(f"Items: {[it['display_name'] for it in items3]}")

        if turn_query == "build an outfit around it":
            assert "outfit" in tools_used3, (
                f"FAIL: outfit tool not called. Tools: {tools_used3}"
            )
            assert len(items3) >= 2, (
                f"FAIL: outfit should return seed + at least 1 complement (got {len(items3)})"
            )
            missing_key = [it["display_name"] for it in items3 if "image_url" not in it]
            assert not missing_key, f"FAIL: items missing image_url key: {missing_key}"
            print(f"[PASS: outfit called, {len(items3)} items, all have image_url key]")

        messages3 = r.get("messages", messages3)
        filters3 = r.get("filters", filters3)
        retrieved3 = items3


if __name__ == "__main__":
    main()
