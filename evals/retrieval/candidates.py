"""Candidate generation: the CURRENT retrieval path up to the 20 reranker candidates.

The compiled LangGraph agent is driven exactly as in production (intent parsing, filters,
hybrid RRF with the gender/colour/price pushdown, kids/fabric exclusion, store diversity and
per-store cap, refinement/underflow logic, price-qualifier sort), with two seams:

* `src.agents.graph.rerank` is replaced by a recorder that captures (query, candidates) and
  returns the retrieval order, so no reranker LLM is ever called;
* the LLM handed to the graph is a stub that is never expected to answer (any call it receives,
  e.g. a router fallback, is counted per case in `llm_stub_calls`).

Public-store restriction: every store NOT in public_stores.json is added to the retriever's
"inactive store" set, which pushes the exclusion into BM25 and FAISS fetch (identical to an
index that only contains the public stores; nothing is post-filtered).

Some queries route to the outfit/composer path (a look, not a search), which never calls
rerank(). For those the `search` node is invoked directly with a plain search plan, i.e. what
the agent does for the same words as a single-item search; the case is flagged
`route: "forced_search"`.

Run (see README): python -m evals.retrieval.run --stage candidates --index-dir <copy of unified>
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from evals.retrieval.common import CANDIDATE_DIR, load_cases, load_public_stores

POOL = 20
DESC_CHARS = 300  # reranker uses the first 120; judges use the first 150


def item_record(rank: int, it: dict[str, Any]) -> dict[str, Any]:
    """One candidate as stored on disk (superset of what the reranker prompt and judges need)."""

    def clean(v: Any) -> Any:
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

    return {
        "rank": rank,
        "article_id": str(it["article_id"]),
        "store": clean(it.get("store")),
        "prod_name": it.get("prod_name"),
        "display_name": it.get("display_name"),
        "price_inr": clean(it.get("price_inr")),
        "product_type": it.get("product_type"),
        "department": it.get("department"),
        "gender": clean(it.get("gender")),
        "colour": it.get("colour"),
        "detail_desc": (it.get("detail_desc") or "")[:DESC_CHARS],
        "snippet": (it.get("detail_desc") or "")[:200],
        "rrf_score": clean(it.get("score")),
        "pdp_handle": clean(it.get("pdp_handle")),
    }


def case_record(
    case: dict[str, Any],
    pool: list[dict[str, Any]],
    route: str,
    llm_stub_calls: int,
) -> dict[str, Any]:
    """The per-case candidate file content (top-20 of the pool the reranker would receive)."""
    top = pool[:POOL]
    return {
        "query_id": case["id"],
        "text": case["text"],
        "route": route,
        "pool_size_before_cap": len(pool),
        "n_candidates": len(top),
        "llm_stub_calls": llm_stub_calls,
        "items": [item_record(i, it) for i, it in enumerate(top, 1)],
    }


class StubLLM:
    """Never meant to answer; counts calls so a case can report router-fallback LLM use."""

    model = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        self.calls += 1
        return "{}"

    def generate_stream(self, prompt: str, system: str | None = None, **kw: Any) -> Iterator[str]:
        self.calls += 1
        yield ""

    def chat(self, messages: list[dict], **kw: Any) -> str:
        self.calls += 1
        return "{}"

    def chat_stream(self, messages: list[dict], **kw: Any) -> Iterator[str]:
        self.calls += 1
        yield ""


class _Memory:
    def __init__(self, n: int) -> None:
        self.n = n

    def get_context(self, messages: list[dict], state: Any = None) -> list[dict]:
        return messages[-self.n :]


@contextmanager
def restrict_to_stores(stores: list[str] | None, all_stores: set[str]) -> Iterator[None]:
    """Treat every store outside `stores` as inactive for retrieval (None = no restriction)."""
    import src.retrieval.hybrid_search as hs

    original = hs.get_inactive_stores
    if stores is not None:
        extra = frozenset(s.lower() for s in all_stores - set(stores))
        hs.get_inactive_stores = lambda: frozenset(original() | extra)
    try:
        yield
    finally:
        hs.get_inactive_stores = original


def load_stack(index_dir: Path, config_path: Path, sparse_only: bool = False) -> tuple[Any, ...]:
    """(retriever, catalogue_df, config). `sparse_only` stubs the dense side (equality proof
    against the BM25-only reconstruction in reports/wave1b/v4/recon_prompts.json)."""
    import pandas as pd
    import yaml

    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(index_dir / "catalogue.parquet")
    sparse = SparseRetriever.load(config, index_dir)
    if sparse_only:

        class _NoDense:
            article_ids = sparse.article_ids

            def search(self, q: str, top_k: int = 20, params: Any = None) -> list:
                return []

            def search_by_id(self, *a: Any, **k: Any) -> list:
                return []

        dense: Any = _NoDense()
    else:
        from src.retrieval.dense_search import DenseRetriever

        dense = DenseRetriever.load(config, index_dir)
    return HybridRetriever(dense, sparse, df, config), df, config


def run_case(agent: Any, config: dict, case: dict[str, Any], llm: StubLLM) -> dict[str, Any]:
    """Drive the real graph for one query; fall back to the search node for outfit routes."""
    import src.agents.graph as g

    captured: list[tuple[str, list[dict]]] = []

    def capture(query: str, items: list[dict], _llm: Any, top_k: int = 5) -> list[dict]:
        captured.append((query, list(items)))
        return items[:top_k]

    original = g.rerank
    setattr(g, "rerank", capture)  # noqa: B010 - typed seam for a module-level function
    llm.calls = 0
    state = {
        "messages": [{"role": "user", "content": case["text"]}],
        "user_query": case["text"],
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": [],
        "filters": {},
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
        "out_of_catalogue": False,
        "excluded_colours": None,
        "anchor_article_id": None,
        "anchor_is_owned": False,
        "outfit_rationale": None,
        "outfit_variants": None,
        "_memory": _Memory(config["memory"]["recent_turns"]),
    }
    route = "graph_rerank"
    try:
        agent.invoke(state)
        if not captured:
            route = "forced_search"
            forced = {**state, "current_plan": json.dumps({"action": "search", "filters": {}})}
            agent.nodes["search"].invoke(forced)
    finally:
        setattr(g, "rerank", original)  # noqa: B010
    pool = captured[0][1] if captured else []
    return case_record(case, pool, route if captured else "no_candidates", llm.calls)


def generate(
    index_dir: Path,
    config_path: Path,
    out_dir: Path = CANDIDATE_DIR,
    limit: int | None = None,
    public_only: bool = True,
    sparse_only: bool = False,
) -> list[dict[str, Any]]:
    """Generate candidate files for all (or the first `limit`) cases; returns the records."""
    import src.agents.graph as g

    retriever, df, config = load_stack(index_dir, config_path, sparse_only)
    all_stores = {str(s).lower() for s in df["store"].dropna().unique()}
    stores = load_public_stores() if public_only else None
    llm = StubLLM()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with restrict_to_stores(stores, all_stores):
        agent = g.build_graph(retriever, df, llm, config, streaming_mode=True)
        for case in load_cases(limit):
            rec = run_case(agent, config, case, llm)
            (out_dir / f"{case['id']}.json").write_text(
                json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8"
            )
            records.append(rec)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-case candidate counts, route counts and the store distribution of all candidates."""
    from collections import Counter

    stores = Counter(it["store"] for r in records for it in r["items"])
    return {
        "n_cases": len(records),
        "per_case_counts": {r["query_id"]: r["n_candidates"] for r in records},
        "under_20": [r["query_id"] for r in records if r["n_candidates"] < POOL],
        "routes": dict(Counter(r["route"] for r in records)),
        "store_distribution": dict(stores.most_common()),
    }
