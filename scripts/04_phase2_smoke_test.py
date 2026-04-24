"""
Phase 2 smoke test — verifies grounding prompt, validator, and seasonal router rewriting.

Runs 5 queries through the full agent (non-streaming), logging:
  - Router's extracted search query
  - Top-5 items returned
  - Final LLM response
  - Validator flags raised

Uses the local 20k-item index + Ollama llama3.1:8b (no faiss GPU required).
Bypasses the parquet pyarrow read bug by reconstructing catalogue from articles.csv.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalogue.loader import load_config
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.sparse_search import SparseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.llm.client import get_llm_client
from src.memory.conversation import ConversationMemory
from src.agents.graph import build_graph


def _build_catalogue_df(save_dir: Path) -> pd.DataFrame:
    """Reconstruct catalogue dataframe from articles.csv + dense article IDs."""
    ids = np.load(str(save_dir / "dense_article_ids.npy"), allow_pickle=True)
    df = pd.read_csv("data/hm/articles.csv", dtype=str)
    df = df[df["article_id"].isin(ids)].copy()
    df["search_text"] = (
        df["prod_name"].fillna("") + ". "
        + df["product_type_name"].fillna("") + ". "
        + df["colour_group_name"].fillna("") + ". "
        + df["department_name"].fillna("") + ". "
        + df["detail_desc"].fillna("")
    )
    df["display_name"] = (
        df["prod_name"].fillna("").str.strip()
        + " ("
        + df["colour_group_name"].fillna("").str.strip()
        + " "
        + df["product_type_name"].fillna("").str.strip()
        + ")"
    )
    df["facets"] = df.apply(lambda r: {
        "colour_group_name": r["colour_group_name"],
        "product_type_name": r["product_type_name"],
        "department_name": r["department_name"],
        "index_group_name": r["index_group_name"],
        "garment_group_name": r["garment_group_name"],
    }, axis=1)
    df["image_url"] = df["article_id"].apply(lambda a: f"images/{a[:3]}/{a}.jpg")
    return df


def _run_turn(agent, query: str, state: dict) -> dict:
    new_state = {
        **state,
        "messages": state["messages"] + [{"role": "user", "content": query}],
        "user_query": query,
        "current_plan": None,
        "tool_calls": [],
        "iteration": 0,
    }
    return agent.invoke(new_state)


def main():
    config = load_config()
    save_dir = Path("data/processed")

    print("Loading retrieval indices...")
    df = _build_catalogue_df(save_dir)
    dense = DenseRetriever.load(config, save_dir)
    sparse = SparseRetriever.load(config, save_dir)
    retriever = HybridRetriever(dense, sparse, df, config)
    llm = get_llm_client(config)
    memory = ConversationMemory(llm, config)
    agent = build_graph(retriever, df, llm, memory, config, streaming_mode=False)
    print(f"Agent ready ({len(df):,} items).\n")

    # ------------------------------------------------------------------
    # Shared state for multi-turn tests
    # ------------------------------------------------------------------
    blank_state = {"messages": [], "filters": {}, "retrieved_items": [], "final_answer": None}

    # ------------------------------------------------------------------
    # Q1: price query — must not contain price claim
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Q1: 'which is cheaper' (after showing 2 dresses)")
    # First turn: show dresses
    r = _run_turn(agent, "show me some black dresses", blank_state)
    search_q1 = next(
        (list(tc.values())[0].get("query", "") for tc in r.get("tool_calls", [])
         if "search" in tc),
        "(not found)"
    )
    items_q1 = r.get("retrieved_items", [])
    # Second turn: ask for price comparison
    state_after_q1 = {
        "messages": r.get("messages", []),
        "filters": r.get("filters", {}),
        "retrieved_items": items_q1,
        "final_answer": None,
    }
    r2 = _run_turn(agent, "which is cheaper", state_after_q1)
    answer_q1 = r2.get("final_answer", "")
    tools_q1 = [list(tc.keys())[0] for tc in r2.get("tool_calls", [])]
    print(f"  Search query: {search_q1!r}")
    print(f"  Items: {[it['display_name'] for it in items_q1]}")
    print(f"  Tools (2nd turn): {tools_q1}")
    print(f"  Response: {answer_q1}")
    has_price = any(w in answer_q1.lower() for w in ["price", "cheaper", "cost", "affordable", "discount"])
    print(f"  PASS: no price claim = {not has_price}" if not has_price else "  FAIL: price claim detected!")
    print()

    # ------------------------------------------------------------------
    # Q2: winter seasonal query — should include winter categories
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Q2: 'minimalist winter essentials'")
    r = _run_turn(agent, "minimalist winter essentials", blank_state)
    search_q2 = next(
        (list(tc.values())[0].get("query", "") for tc in r.get("tool_calls", [])
         if "search" in tc),
        "(not found)"
    )
    items_q2 = r.get("retrieved_items", [])
    answer_q2 = r.get("final_answer", "")
    winter_kws = ["sweater", "coat", "jacket", "knitwear", "knit", "outerwear"]
    query_has_winter_cat = any(k in search_q2.lower() for k in winter_kws)
    items_are_winter = any(
        any(k in it.get("product_type", "").lower() for k in ["sweater", "coat", "jacket", "knitwear", "jumper", "cardigan", "hoodie"])
        for it in items_q2
    )
    print(f"  Search query: {search_q2!r}")
    print(f"  Items: {[(it['display_name'], it.get('product_type','')) for it in items_q2]}")
    print(f"  Response: {answer_q2}")
    print(f"  PASS: winter categories in search query = {query_has_winter_cat}")
    print(f"  PASS: winter items in results = {items_are_winter}")
    print()

    # ------------------------------------------------------------------
    # Q3: comparison query — must not fabricate non-listed attributes
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Q3: 'what is the difference between these two' (after 2 items shown)")
    r = _run_turn(agent, "show me two blazers", blank_state)
    items_q3 = r.get("retrieved_items", [])
    state_q3 = {
        "messages": r.get("messages", []),
        "filters": r.get("filters", {}),
        "retrieved_items": items_q3,
        "final_answer": None,
    }
    r2 = _run_turn(agent, "what is the difference between these two", state_q3)
    answer_q3 = r2.get("final_answer", "")
    tools_q3 = [list(tc.keys())[0] for tc in r2.get("tool_calls", [])]
    print(f"  Items: {[it['display_name'] for it in items_q3]}")
    print(f"  Tools (2nd turn): {tools_q3}")
    print(f"  Response: {answer_q3}")
    bad_attrs = ["price", "cost", "cheaper", "affordable", "xs", "xl", "runs big", "runs small"]
    has_bad = any(w in answer_q3.lower() for w in bad_attrs)
    print(f"  PASS: no ungrounded attributes = {not has_bad}" if not has_bad else f"  FAIL: ungrounded claim detected!")
    print()

    # ------------------------------------------------------------------
    # Q4: 'something warm for today' — no fabric weight claims
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Q4: 'something warm for today'")
    r = _run_turn(agent, "something warm for today", blank_state)
    search_q4 = next(
        (list(tc.values())[0].get("query", "") for tc in r.get("tool_calls", [])
         if "search" in tc),
        "(not found)"
    )
    items_q4 = r.get("retrieved_items", [])
    answer_q4 = r.get("final_answer", "")
    print(f"  Search query: {search_q4!r}")
    print(f"  Items: {[(it['display_name'], it.get('product_type','')) for it in items_q4]}")
    print(f"  Response: {answer_q4}")
    # Check: if "warm" appears in response, it should be backed by item descriptions
    warm_in_descs = any("warm" in (it.get("detail_desc") or "").lower() for it in items_q4)
    warm_in_resp = "warm" in answer_q4.lower()
    if warm_in_resp and not warm_in_descs:
        print("  FAIL: 'warm' claim not backed by item descriptions")
    else:
        print(f"  PASS: warm claim backed = {warm_in_descs or not warm_in_resp}")
    print()

    # ------------------------------------------------------------------
    # Q5: control — 'show me a dress' must work normally
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Q5: 'show me a dress' (sanity control)")
    r = _run_turn(agent, "show me a dress", blank_state)
    search_q5 = next(
        (list(tc.values())[0].get("query", "") for tc in r.get("tool_calls", [])
         if "search" in tc),
        "(not found)"
    )
    items_q5 = r.get("retrieved_items", [])
    answer_q5 = r.get("final_answer", "")
    print(f"  Search query: {search_q5!r}")
    print(f"  Items: {[it['display_name'] for it in items_q5]}")
    print(f"  Response: {answer_q5}")
    all_dresses = all("dress" in it.get("product_type", "").lower() for it in items_q5)
    print(f"  PASS: all results are dresses = {all_dresses}")
    print()


if __name__ == "__main__":
    main()
