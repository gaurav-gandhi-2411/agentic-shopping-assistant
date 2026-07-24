"""Multi-garment "X and Y" query parsing (2026-07-24).

Live-proven bug: "sports bra and leggings" only ever returned sports bra
items — leggings never surfaced, despite the catalogue having 225+ women's
leggings rows. Root cause: intent_parser.IntentV1.garment_type is
architecturally single-valued (see _extract_garment_type's docstring), so a
"X and Y" query naming two distinct garments only ever resolves ONE of them.

Fix: intent_parser.parse_intent() additionally populates
garment_type_secondary (see IntentV1's docstring) for a genuine two-garment
conjunction, and graph.py's search_node issues a SECOND retrieval call for
that type, interleaving the two pools before rerank/truncation — mirroring
composer._find_best_candidate's per-family accessory-retrieval-then-merge
fix (commit 1717265) for the identical single-query-starves-one-type failure
mode.

Deterministic-parsing coverage lives in tests/test_intent_parser.py's
TestMultiGarmentConjunctionSplit/TestActivewearAndChuridarVocabulary. This
module covers the END-TO-END real-index behaviour (search_node's retrieval-
merge + the single-garment-set-exclusion-gate carve-out it required) —
mirrors tests/test_gym_occasion.py's / tests/test_occasion_merchandise_leak.py's
real-index harness pattern.
"""
from __future__ import annotations

import pandas as pd
import pytest


class _RealIndexMockLLM:
    """Minimal fixed-response LLM stub — mirrors tests/test_gym_occasion.py's
    identical helper. Its canned non-JSON response deterministically forces
    rerank() down the retrieval-order fallback path (`items[:top_k]`), which
    makes the interleave ORDER produced by search_node's multi-garment merge
    directly observable in `retrieved_items` — exactly the behaviour a real
    LLM reranker failure (timeout/parse_error) would also fall back to, so
    this is not a purely synthetic test condition.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def _next(self) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        return self._next()

    def generate_stream(self, prompt: str, system: str = None, **kwargs):
        yield self._next()

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self._next()


class TestMultiGarmentSearchRealIndex:
    """Reproduces the live bug end-to-end against the real unified catalogue,
    across gym/activewear AND Indian-ethnic categories — the fix is a general
    query-parsing class, not a gym special case."""

    @staticmethod
    def _run_search(query: str):
        from src.agents.graph import build_graph
        from src.memory.conversation import ConversationMemory
        from src.retrieval.dense_search import DenseRetriever
        from src.retrieval.hybrid_search import HybridRetriever
        from src.retrieval.sparse_search import SparseRetriever

        unified_dir = "data/processed/unified"
        config: dict = {
            "agent": {"max_iterations": 3},
            "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
            "retrieval": {
                "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
                "dense_dim": 384,
                "rrf_k": 60,
                "top_k": 50,
                "final_k": 5,
                "store_diversity": 0.2,
            },
        }
        dense = DenseRetriever.load(config, unified_dir)
        sparse = SparseRetriever.load(config, unified_dir)
        catalogue_df = pd.read_parquet(f"{unified_dir}/catalogue.parquet")
        retriever = HybridRetriever(dense, sparse, catalogue_df, config)

        llm = _RealIndexMockLLM(["Here you go."] * 5)
        memory = ConversationMemory(llm, config)
        agent = build_graph(retriever, catalogue_df, llm, config, streaming_mode=True)

        state = {
            "messages": [{"role": "user", "content": query}],
            "user_query": query,
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
            "outfit_rationale": None,
            "outfit_variants": None,
            "_memory": memory,
        }
        return agent.invoke(state)

    def _assert_both_types_present(
        self, query: str, primary_type: str, secondary_type: str
    ) -> None:
        result = self._run_search(query)
        items = result.get("retrieved_items", [])
        assert items, f"expected items for {query!r}"
        types = {(it.get("product_type") or "").lower() for it in items}
        assert primary_type.lower() in types, (
            f"{query!r}: expected primary type {primary_type!r} in results, got {types}"
        )
        assert secondary_type.lower() in types, (
            f"{query!r}: expected secondary type {secondary_type!r} in results "
            f"(the exact live-proven bug this fix closes — one type starving out "
            f"the other), got {types}"
        )

    @pytest.mark.requires_index
    def test_sports_bra_and_leggings_both_surface(self) -> None:
        """The exact live-reproduced query."""
        self._assert_both_types_present("sports bra and leggings", "sports_bra", "leggings")

    @pytest.mark.requires_index
    def test_joggers_and_tshirt_both_surface(self) -> None:
        self._assert_both_types_present("joggers and t-shirt", "joggers", "top")

    @pytest.mark.requires_index
    def test_kurta_and_palazzo_both_surface(self) -> None:
        self._assert_both_types_present("kurta and palazzo", "kurta", "palazzo")

    @pytest.mark.requires_index
    def test_saree_and_blouse_both_surface(self) -> None:
        self._assert_both_types_present("saree and blouse", "saree", "blouse")

    @pytest.mark.requires_index
    def test_sherwani_and_churidar_both_surface(self) -> None:
        """churidar resolves to the catalogue's existing "salwar" facet (thin,
        14-row inventory) — a near-synonym merge, not an unbacked new value
        (see intent_parser's churidar rule comment)."""
        self._assert_both_types_present("sherwani and churidar", "sherwani", "salwar")

    @pytest.mark.requires_index
    @pytest.mark.parametrize(
        "query, expected_type",
        [
            ("sports bra for women", "sports_bra"),
            ("kurta for men", "kurta"),
            ("black dress for women", "dress"),
        ],
    )
    def test_single_garment_queries_unaffected(self, query: str, expected_type: str) -> None:
        """No regression: ordinary single-garment queries must return ONLY
        that one type — the multi-garment merge must be a strict no-op."""
        result = self._run_search(query)
        items = result.get("retrieved_items", [])
        assert items, f"expected items for {query!r}"
        types = {(it.get("product_type") or "").lower() for it in items}
        assert types == {expected_type.lower()}, (
            f"{query!r}: expected only {expected_type!r}, got {types}"
        )
