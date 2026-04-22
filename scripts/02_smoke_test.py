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

    test_queries = [
        "show me some black jackets",
        "something for summer, light and breathable",
        "compare the first two you showed me",   # tests memory / retrieved_items carry-over
        "anything in blue instead?",             # tests filter then re-search
    ]

    # State persisted across turns
    messages: list[dict] = []
    filters: dict = {}
    retrieved_items: list[dict] = []

    for query in test_queries:
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
        print()

        # Carry state forward for next turn
        messages = result.get("messages", messages)
        filters = result.get("filters", filters)
        retrieved_items = result.get("retrieved_items", retrieved_items)


if __name__ == "__main__":
    main()
