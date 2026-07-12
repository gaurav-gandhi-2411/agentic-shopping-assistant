#!/usr/bin/env python3
"""In-process telemetry measurement for the fast-path change.

Runs the same 25-turn mix used in the Wave 2 baseline directly through
agent.invoke() (no HTTP server required) with a call-counting LLM wrapper.

Outputs:
  scripts/telemetry_fastpath.json  — per-turn call counts + comparison table
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterator

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import fastparquet as fp  # noqa: E402

from src.agents.graph import build_graph  # noqa: E402
from src.catalogue.loader import load_config  # noqa: E402
from src.llm.client import get_llm_client  # noqa: E402
from src.memory.conversation import ConversationMemory  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed"
_CONFIG_PATH = str(_ROOT / "config.yaml")
_OUT_PATH = _ROOT / "scripts" / "telemetry_fastpath.json"


class CountingLLM:
    """Wraps a real LLM and counts every generate() call."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[dict] = []

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        t0 = time.perf_counter()
        result = self._inner.generate(prompt, system, **kwargs)
        self.calls.append({"ms": round((time.perf_counter() - t0) * 1000)})
        return result

    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
        for token in self._inner.generate_stream(prompt, system, **kwargs):
            yield token

    def chat(self, messages: list, **kwargs) -> str:
        return self._inner.chat(messages, **kwargs)

    def chat_stream(self, messages: list, **kwargs) -> Iterator[str]:
        for token in self._inner.chat_stream(messages, **kwargs):
            yield token

    def reset(self):
        self.calls.clear()

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _load_df():
    pf = fp.ParquetFile(str(_DATA_DIR / "catalogue.parquet"))
    df = pf.to_pandas()
    facet_keys = [
        "colour_group_name", "product_type_name", "department_name",
        "index_group_name", "garment_group_name",
    ]
    df["facets"] = [
        {k: df[f"facets.{k}"].iat[i] for k in facet_keys}
        for i in range(len(df))
    ]
    return df


def _make_state(messages, user_query, retrieved_items=None, filters=None, memory=None):
    return {
        "messages": list(messages),
        "user_query": user_query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": list(retrieved_items or []),
        "filters": dict(filters or {}),
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
        "out_of_catalogue": False,
        "excluded_colours": [],
        "_memory": memory,
    }


def run_turn(label: str, message: str, agent, llm_wrapper, config,
             prior_state: dict | None = None) -> dict:
    """Run one turn; returns result dict with call_count."""
    llm_wrapper.reset()
    memory = ConversationMemory(llm_wrapper, config)

    if prior_state:
        # Continue from prior conversation state — carry forward items + messages.
        state = _make_state(
            messages=prior_state["messages"] + [{"role": "user", "content": message}],
            user_query=message,
            retrieved_items=prior_state.get("retrieved_items", []),
            filters=prior_state.get("filters", {}),
            memory=memory,
        )
    else:
        state = _make_state(
            messages=[{"role": "user", "content": message}],
            user_query=message,
            memory=memory,
        )

    t0 = time.perf_counter()
    result = agent.invoke(state)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    n_calls = llm_wrapper.call_count
    action = _infer_action(result)
    answer_preview = (result.get("final_answer") or "")[:80].replace("\n", " ")

    print(
        f"  {label:<14}  calls={n_calls}  action={action:<12}  {elapsed_ms:>5}ms  {answer_preview!r}",
        flush=True,
    )
    return {
        "label": label,
        "message": message,
        "llm_calls": n_calls,
        "action": action,
        "elapsed_ms": elapsed_ms,
        "answer_preview": answer_preview,
        # Carry forward for multi-turn chains.
        "_state": {
            "messages": result.get("messages", []),
            "retrieved_items": result.get("retrieved_items", []),
            "filters": result.get("filters", {}),
        },
    }


def _infer_action(result: dict) -> str:
    tool_calls = result.get("tool_calls", [])
    last_action = "respond"
    for tc in reversed(tool_calls):
        key = list(tc.keys())[0]
        if key == "router_decision":
            rd = tc["router_decision"]
            if isinstance(rd, dict):
                last_action = rd.get("action", "respond")
            break
        last_action = key
        break
    return last_action


def main():
    import os
    print("Loading components...", flush=True)
    config = load_config(_CONFIG_PATH)
    config["llm"]["provider"] = "groq"
    os.environ["LLM_PROVIDER"] = "groq"

    df = _load_df()
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, df, config)

    real_llm = get_llm_client(config)
    llm = CountingLLM(real_llm)

    agent = build_graph(retriever, df, llm, config, streaming_mode=False)
    print("Ready.\n", flush=True)

    results: list[dict] = []

    def R(label, message, prior=None):
        rec = run_turn(label, message, agent, llm, config, prior)
        results.append(rec)
        time.sleep(3)  # Groq rate limiting
        return rec

    print("-- Simple searches --", flush=True)
    s1 = R("S1-search",  "show me blue dresses")
    s2 = R("S2-search",  "I want a summer top")
    s3 = R("S3-search",  "looking for a leather jacket")
    _s4 = R("S4-search",  "black trousers please")
    _s5 = R("S5-search",  "casual white trainers")
    _s6 = R("S6-search",  "evening gown for a wedding")
    s7 = R("S7-search",  "dark wash denim jeans")
    s8 = R("S8-search",  "cosy chunky knit jumper")

    print("\n-- Refinements --", flush=True)
    R("R1-refine",  "make them more formal please",           s1["_state"])
    R("R2-refine",  "something cheaper, under thirty pounds", s2["_state"])
    R("R3-refine",  "do you have something lighter for summer", s3["_state"])
    R("R4-refine",  "in a lighter wash instead",              s7["_state"])
    R("R5-refine",  "show me it in an oversized style",       s8["_state"])

    print("\n-- Outfit requests --", flush=True)
    R("O1-outfit",  "put together a complete outfit for a job interview")
    R("O2-outfit",  "full casual beach day outfit please")
    R("O3-outfit",  "what would you style with a midi skirt for a date night")
    R("O4-outfit",  "garden party outfit, smart casual please")

    print("\n-- Comparisons --", flush=True)
    R("C1-compare", "compare silk dresses versus cotton dresses for summer")
    R("C2-compare", "which is better for a petite frame, midi or maxi skirt")
    R("C3-compare", "compare leather jackets versus denim jackets for autumn")

    print("\n-- OOC --", flush=True)
    R("OOC1",       "what is the weather like today in London")
    R("OOC2",       "who are you and what can you help me with")

    print("\n-- Negations --", flush=True)
    R("N1-negate",  "show me dresses but nothing red")
    R("N2-negate",  "I want loungewear but absolutely no pyjamas")

    print("\n-- Multi-turn chain (5 turns) --", flush=True)
    mt1 = R("MT1-multi", "show me midi skirts")
    mt2 = R("MT2-multi", "I like these but need something more formal for the office", mt1["_state"])
    mt3 = R("MT3-multi", "do you have these in navy or charcoal grey",  mt2["_state"])
    mt4 = R("MT4-multi", "which of these has the best quality fabric",   mt3["_state"])
    R("MT5-multi",       "now show me blouses that would pair well with these skirts", mt4["_state"])

    # --- Summary ---
    call_counts = [r["llm_calls"] for r in results]
    total_calls = sum(call_counts)
    n_turns = len(results)

    # Load baseline for comparison
    baseline_path = _ROOT / "scripts" / "telemetry_stats.json"
    baseline_total = None
    if baseline_path.exists():
        with open(baseline_path) as fh:
            bs = json.load(fh)
        baseline_total = bs.get("total_llm_calls")

    print(f"\n{'='*64}", flush=True)
    print(f"TELEMETRY FAST-PATH RESULTS  ({n_turns} turns)", flush=True)
    print(f"{'='*64}", flush=True)
    print(f"Total LLM calls (fast-path):  {total_calls}", flush=True)
    if baseline_total:
        print(f"Total LLM calls (baseline):   {baseline_total}", flush=True)
        print(f"Calls eliminated:             {baseline_total - total_calls}  ({(baseline_total - total_calls)/baseline_total*100:.1f}%)", flush=True)
    print(f"Mean calls per turn:          {total_calls/n_turns:.2f}", flush=True)
    print(f"{'='*64}", flush=True)

    by_cat: dict[str, list[int]] = {}
    for r in results:
        cat = r["label"].split("-")[0] if "-" in r["label"] else r["label"][:3]
        by_cat.setdefault(cat, []).append(r["llm_calls"])
    print("\nBy turn type (mean calls):", flush=True)
    for cat, counts in by_cat.items():
        print(f"  {cat:<8}  mean={sum(counts)/len(counts):.2f}  values={counts}", flush=True)

    # Save
    out = {
        "run_date": time.strftime("%Y-%m-%d"),
        "n_turns": n_turns,
        "total_llm_calls": total_calls,
        "mean_calls_per_turn": round(total_calls / n_turns, 2),
        "baseline_total_calls": baseline_total,
        "calls_eliminated": (baseline_total - total_calls) if baseline_total else None,
        "results": results,
    }
    # Remove internal _state field for JSON output
    for r in out["results"]:
        r.pop("_state", None)

    with open(_OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {_OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
