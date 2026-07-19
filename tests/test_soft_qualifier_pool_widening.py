"""Regression tests for the 2026-07-19 soft-constraint (price_qualifier /
formality_softener) truncate-then-rerank fix.

Live-proven root cause (confirmed via a verifier agent trace, then re-verified
here before fixing): price_qualifier ("cheap") and formality_softener
("minimalist"/"comfortable", surfaced from phrases like "not too flashy"/
"not too heavy") were extracted correctly by IntentParser, but search_node
(src.agents.graph) only ever applied them AFTER rerank() had already truncated
the candidate pool from fetch_k (20-40) down to top_k (5) via an LLM call.
By the time the qualifier-based filter/sort ran, genuinely cheap/plain items
sitting outside that narrow post-truncation window could never be recovered —
_apply_price_qualifier/fabric_score_delta could only re-sort whatever 5 items
the reranker had already picked, which is why "not too flashy for a wedding"
was live-proven to return MORE embellished items (5/5) than the unconstrained
baseline (2/5): dense/BM25 retrieval scores the raw query text including the
negated adjective itself ("flashy"), a known embedding-negation-blindness
failure mode, so embellished items ranked UP the pool the reranker chose from.

Two changes fix this (see src.agents.graph):
1. fetch_k (the pre-rerank retrieval width) widens to >=80 whenever
   price_qualifier or formality_softener is present, so there is a
   meaningfully larger pool for qualifier filtering to work with.
2. _apply_price_qualifier / the new _apply_formality_softener run on that WIDE
   `candidates` pool BEFORE rerank() truncates to top_k, not just on the
   already-truncated items_out afterwards. _apply_formality_softener is also
   no longer gated behind occasion detection (previously nested inside
   "if _occ_slug and _occ_slug != 'casual'"), since a bare "something not too
   flashy" query with no named occasion must still demote embellished items.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest

from src.agents.graph import _apply_formality_softener, build_graph
from src.agents.outfit.slots import SANGEET_EMBELLISHMENT_KEYWORDS
from src.memory.conversation import ConversationMemory
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.sparse_search import SparseRetriever

# ── Unit tests: _apply_formality_softener ───────────────────────────────────


def _embellished_item(article_id: str) -> dict:
    kw = next(iter(SANGEET_EMBELLISHMENT_KEYWORDS))
    return {"article_id": article_id, "prod_name": f"{kw.title()} Bridal Lehenga", "detail_desc": ""}


def _plain_item(article_id: str) -> dict:
    return {"article_id": article_id, "prod_name": "Cotton Printed Kurta", "detail_desc": ""}


class TestApplyFormalitySoftener:
    def test_minimalist_filters_out_embellished_items(self) -> None:
        pool = [_embellished_item("E1"), _plain_item("P1"), _plain_item("P2")]
        out = _apply_formality_softener(pool, "minimalist")
        assert {it["article_id"] for it in out} == {"P1", "P2"}

    def test_comfortable_filters_out_embellished_items(self) -> None:
        pool = [_embellished_item("E1"), _plain_item("P1"), _plain_item("P2")]
        out = _apply_formality_softener(pool, "comfortable")
        assert {it["article_id"] for it in out} == {"P1", "P2"}

    def test_checks_display_name_and_detail_desc_too(self) -> None:
        # embellishment keyword only in detail_desc, not prod_name. Two plain
        # items so the >=2 pool-underflow floor doesn't mask the assertion.
        kw = next(iter(SANGEET_EMBELLISHMENT_KEYWORDS))
        pool = [
            {"article_id": "E1", "prod_name": "Blue Lehenga", "detail_desc": f"has {kw} work"},
            _plain_item("P1"),
            _plain_item("P2"),
        ]
        out = _apply_formality_softener(pool, "minimalist")
        assert {it["article_id"] for it in out} == {"P1", "P2"}

    def test_pool_underflow_protected_keeps_pool_if_filter_would_leave_fewer_than_two(
        self,
    ) -> None:
        pool = [_embellished_item("E1"), _embellished_item("E2")]
        out = _apply_formality_softener(pool, "minimalist")
        assert {it["article_id"] for it in out} == {"E1", "E2"}  # unfiltered, not emptied

    def test_flashy_positive_value_is_noop(self) -> None:
        # "flashy" (the user WANTS embellishment) is not in FORMALITY_SOFTENER_VALUES —
        # only the negated forms ("minimalist"/"comfortable") trigger the filter.
        pool = [_embellished_item("E1"), _plain_item("P1")]
        assert _apply_formality_softener(pool, "flashy") == pool

    def test_none_is_noop(self) -> None:
        pool = [_embellished_item("E1"), _plain_item("P1")]
        assert _apply_formality_softener(pool, None) == pool

    def test_empty_items_is_noop(self) -> None:
        assert _apply_formality_softener([], "minimalist") == []


# ── Full-pipeline integration tests (real unified index) ────────────────────
# Mirrors tests/test_batch2_trust_fixes.py's TestLiveRepros fixture shape
# exactly (established convention — each test file duplicates its own
# _unified_index fixture rather than sharing one via conftest).

UNIFIED_DIR = Path("data/processed/unified")

_MINIMAL_CONFIG: dict = {
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


class _CapturingLLM:
    """Records every prompt passed to generate(); returns a fixed canned reply.

    Mirrors tests/test_batch2_trust_fixes.py's stub exactly.
    """

    def __init__(self, reply: str = "Great pick! Here's why it works for you.") -> None:
        self.prompts: list[str] = []
        self._reply = reply

    def generate(self, prompt: str, system: str = None, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        return self._reply

    def generate_stream(self, prompt: str, system: str = None, **kwargs: Any) -> Iterator[str]:
        self.prompts.append(prompt)
        yield self._reply

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        return self._reply

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        yield self._reply


@pytest.fixture(scope="module")
def _unified_index() -> tuple[HybridRetriever, pd.DataFrame]:
    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _MINIMAL_CONFIG)
    return retriever, catalogue_df


def _blank_state(query: str, memory: Any) -> dict:
    return {
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


def _has_embellishment(item: dict) -> bool:
    text = ((item.get("prod_name") or "") + " " + (item.get("detail_desc") or "")).lower()
    return any(kw in text for kw in SANGEET_EMBELLISHMENT_KEYWORDS)


@pytest.mark.requires_index
class TestNotTooFlashyExcludesEmbellishmentEndToEnd:
    """Live-proven defect (2026-07-19): "not too flashy for a wedding" returned
    5/5 items with "Sequin"/"Embellished"/"Mirror Work" in the title, versus
    2/5 for the unconstrained baseline — the WRONG direction for a softener
    the user explicitly asked for. Re-verified locally before this fix
    (against the real unified index): "not too flashy lehenga for a wedding"
    returned embellished items in every one of only 2 surviving results.
    """

    def test_not_too_flashy_lehenga_returns_zero_embellished_items(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        result = agent.invoke(_blank_state("not too flashy lehenga for a wedding", memory))
        items = result.get("retrieved_items", [])

        assert items, "precondition: query returns results"
        assert not any(_has_embellishment(it) for it in items)

    def test_bare_not_too_flashy_with_no_occasion_still_filters(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        # Regression guard for the occasion-detection gate removed in this fix:
        # fabric_score_delta's formality_override wiring previously only ran
        # inside "if _occ_slug and _occ_slug != 'casual'" — a query naming no
        # occasion at all must still get embellishment-awareness.
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        result = agent.invoke(_blank_state("minimalist lehenga", memory))
        items = result.get("retrieved_items", [])

        assert items, "precondition: query returns results"
        assert not any(_has_embellishment(it) for it in items)


@pytest.mark.requires_index
class TestCheapLehengaWidenedPoolExcludesOutlier:
    """Live-proven defect (2026-07-19): "cheap lehenga for a wedding" — a
    query with BOTH an occasion word and a price qualifier — previously
    returned only the narrow post-rerank top_k window's own 3 items with an
    11250-12497 INR outlier included (re-verified locally against the real
    unified index before this fix), because _apply_price_qualifier's cheap-
    outlier exclusion only ever saw whatever rerank() had already truncated
    to. Widening fetch_k and pre-filtering `candidates` before rerank() fixes
    this the same way as the bare "cheap lehenga" case already covered by
    tests/test_batch2_trust_fixes.py.
    """

    def test_cheap_lehenga_for_wedding_excludes_price_outlier(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        result = agent.invoke(_blank_state("cheap lehenga for a wedding", memory))
        items = result.get("retrieved_items", [])
        prices = [it["price_inr"] for it in items if it.get("price_inr")]

        assert prices, "precondition: query returns priced items"
        assert prices == sorted(prices), "expected ascending price order"
        # None of the returned items should be a multi-x-median outlier —
        # mirrors _apply_price_qualifier's own _CHEAP_OUTLIER_FACTOR contract.
        median = sorted(prices)[len(prices) // 2]
        assert all(p <= median * 3 for p in prices)
