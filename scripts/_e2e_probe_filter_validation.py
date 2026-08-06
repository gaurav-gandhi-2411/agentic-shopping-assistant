#!/usr/bin/env python
"""Before/after measurement of the validate-before-trust fix, through the
REAL graph (search_node's full merged-filter pipeline: LLM emission ->
_FILTER_REMAP -> [new] validate-and-drop -> retriever.search()) — not the
isolated LLM-prompt-only probe (scripts/_probe_router_stale_types.py), which
does not exercise _FILTER_REMAP or this new validation step and therefore
overstated the pre-existing production defect rate for words _FILTER_REMAP
already aliases (hoodie/jacket/coat/sweater/cardigan). Supports the PART 1
hallucination-mitigation wave. Not a permanent script.

Usage:
    python scripts/_e2e_probe_filter_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
load_dotenv(_ROOT / ".env")

from eval_harness import _invoke, _make_state  # noqa: E402
from src.agents.graph import build_graph  # noqa: E402
from src.catalogue.loader import load_config  # noqa: E402
from src.llm.client import get_llm_client  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed" / "unified"

# Same 14 stale-vocab-targeted queries as scripts/_probe_router_stale_types.py,
# run through the REAL graph this time.
_QUERIES = [
    "jacket for men",
    "warm winter coat for women",
    "sweater for women under 2000",
    "cardigan for women",
    "plain t-shirt for men",
    "vest top for women",
    "leggings for women",
    "swimsuit for women",
    "bikini for women",
    "pyjama set for men",
    "night gown for women",
    "hoodie for men under 1500",
    "bathrobe for women",
    "black hoodie for women",
    "denim jacket for women",
]

# Broader/more exotic garment words NOT in _FILTER_REMAP's table at all --
# chosen to test whether the validate-and-drop step catches cases the
# existing hand-maintained alias table does not, since the first _QUERIES set
# turned out to be fully covered by _FILTER_REMAP already (see the before/
# after comparison this script prints -- identical 0/15 empty in both states
# for that set, meaning it couldn't demonstrate this fix's marginal value).
_EXOTIC_QUERIES = [
    "romper for women",
    "camisole for women",
    "bralette for women",
    "peplum top for women",
    "culottes for women",
    "singlet for men",
    "onesie for women",
    "poncho for women",
    "shrug for women",
    "tube top for women",
    "dhoti for men",
    "capri pants for women",
    "boxer shorts for men",
    "tank top for men",
    "windcheater for men",
]


def main() -> None:
    import pandas as pd

    config = load_config()
    config["llm"]["provider"] = "groq"
    catalogue_df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)
    llm = get_llm_client(config)
    agent = build_graph(retriever, catalogue_df, llm, config, streaming_mode=False, router_backend=None)

    empty = non_empty = 0
    for q in _QUERIES + _EXOTIC_QUERIES:
        state = _make_state(messages=[], user_query=q)
        state["messages"] = [{"role": "user", "content": q}]
        result, latency = _invoke(agent, state)
        items = result.get("retrieved_items") or []
        filters = result.get("filters") or {}
        n = len(items)
        if n == 0:
            empty += 1
            verdict = "EMPTY"
        else:
            non_empty += 1
            verdict = "ok"
        top_types = {it.get("product_type") for it in items[:5]}
        print(
            f"{q:<35} n_items={n:<3} filters={filters!r:<55} "
            f"top_types={top_types} {verdict}"
        )

    _total = len(_QUERIES) + len(_EXOTIC_QUERIES)
    print(f"\n{empty}/{_total} queries returned ZERO items (real end-to-end, through the full graph)")
    print(f"{non_empty}/{_total} returned non-empty results")


if __name__ == "__main__":
    main()
