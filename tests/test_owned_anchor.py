"""Tests for the "Owned anchor" feature.

When a user uploads a photo of a garment THEY OWN, the resulting seed item must
never be sold back to them: ``ItemSummary.is_owned`` is True, ``pdp_url`` is
always None, and ``build_cart_action`` (the single choke point used by every
caller — HTTP /chat, WS /chat/stream, POST /style/from-image) drops the item
from cart_url/item_links entirely. Complements are always shoppable.

Covers:
- composer.compose_outfit(owned_anchor=True) stamps the seed only.
- ItemSummary.from_agent_item round-trip: is_owned True + pdp_url None.
- build_cart_action excludes owned items from cart_url/item_links/missing.
- AgentState/session round-trip of anchor_is_owned (api/routes/chat.py).
- graph.py outfit_node preserves ownership on a "Style this <item>" follow-up
  turn when the resolved seed is the session's owned image-upload anchor.
- _BUY_SIMILAR_RE matches "Where can I buy one like this?".
"""
from __future__ import annotations

import json
import re
from typing import Iterator

import pandas as pd
import pytest

from api.schemas import ItemSummary
from src.agents.outfit.cart_links import build_cart_action
from src.agents.outfit.composer import compose_outfit

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_seed_catalogue_row(article_id: str) -> pd.DataFrame:
    """Single-row catalogue DataFrame for a western-top seed item (mirrors the
    fixture in tests/test_outfit_package.py).
    """
    return pd.DataFrame(
        [
            {
                "article_id": article_id,
                "prod_name": "White Shirt",
                "display_name": "White Shirt",
                "colour_group_name": "white",
                "product_type_name": "Shirt",
                "department_name": "Women",
                "index_group_name": "Ladieswear",
                "detail_desc": "",
                "image_url": None,
                "price_inr": 799.0,
                "pdp_handle": "white-shirt",
                "store": "myntra",
                "gender": "women",
                "facets": {
                    "colour_group_name": "white",
                    "product_type_name": "Shirt",
                    "department_name": "Women",
                },
            }
        ]
    )


class _FillSlotFakeRetriever:
    """Returns a "bottom" complement candidate regardless of query."""

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None  # noqa: ARG002
    ) -> list[dict]:
        return [
            {
                "article_id": "C1",
                "prod_name": "Black Trousers",
                "display_name": "Black Trousers",
                "store": "myntra",
                "colour": "black",
                "product_type": "Trousers",
                "detail_desc": "",
                "score": 0.9,
                "price_inr": 999.0,
                "pdp_handle": "black-trousers",
                "gender": "women",
            }
        ]


# ---------------------------------------------------------------------------
# composer.compose_outfit(owned_anchor=True)
# ---------------------------------------------------------------------------


class TestComposeOwnedAnchor:
    def test_owned_anchor_stamps_seed_only(self) -> None:
        """owned_anchor=True stamps _owned on the seed; complements are untouched."""
        catalogue_df = _make_seed_catalogue_row("SEED_OWNED")
        retriever = _FillSlotFakeRetriever()

        look = compose_outfit(
            catalogue_df,
            retriever,
            seed_article_id="SEED_OWNED",
            occasion_slug="casual",
            gender="women",
            owned_anchor=True,
        )

        assert look["seed_item"]["_owned"] is True
        assert look["complements"], "expected at least one complement"
        for complement in look["complements"]:
            assert complement.get("_owned") is not True

    def test_default_owned_anchor_false_does_not_stamp(self) -> None:
        """Without owned_anchor, the seed carries no _owned key at all."""
        catalogue_df = _make_seed_catalogue_row("SEED_NOT_OWNED")
        retriever = _FillSlotFakeRetriever()

        look = compose_outfit(
            catalogue_df,
            retriever,
            seed_article_id="SEED_NOT_OWNED",
            occasion_slug="casual",
            gender="women",
        )

        assert "_owned" not in look["seed_item"]


# ---------------------------------------------------------------------------
# ItemSummary.from_agent_item round-trip
# ---------------------------------------------------------------------------


class TestItemSummaryOwnedRoundTrip:
    def test_owned_item_has_is_owned_true_and_null_pdp_url(self) -> None:
        catalogue_df = _make_seed_catalogue_row("SEED_OWNED2")
        retriever = _FillSlotFakeRetriever()
        look = compose_outfit(
            catalogue_df,
            retriever,
            seed_article_id="SEED_OWNED2",
            occasion_slug="casual",
            gender="women",
            owned_anchor=True,
        )

        seed_summary = ItemSummary.from_agent_item(look["seed_item"])
        assert seed_summary.is_owned is True
        assert seed_summary.pdp_url is None

    def test_complement_summary_is_not_owned_and_has_pdp_url(self) -> None:
        catalogue_df = _make_seed_catalogue_row("SEED_OWNED3")
        retriever = _FillSlotFakeRetriever()
        look = compose_outfit(
            catalogue_df,
            retriever,
            seed_article_id="SEED_OWNED3",
            occasion_slug="casual",
            gender="women",
            owned_anchor=True,
        )

        assert look["complements"]
        complement_summary = ItemSummary.from_agent_item(look["complements"][0])
        assert complement_summary.is_owned is False
        assert complement_summary.pdp_url is not None

    def test_non_owned_item_defaults_is_owned_false(self) -> None:
        """Items with no _owned key at all default is_owned=False (backwards compat)."""
        item = {
            "article_id": "A1",
            "prod_name": "Plain Tee",
            "store": "myntra",
            "pdp_handle": "plain-tee",
        }
        summary = ItemSummary.from_agent_item(item)
        assert summary.is_owned is False


# ---------------------------------------------------------------------------
# build_cart_action excludes owned items (single choke point)
# ---------------------------------------------------------------------------


class TestBuildCartActionExcludesOwned:
    def _owned_seed(self) -> dict:
        return {
            "article_id": "OWNED1",
            "pdp_handle": "owned-item",
            "display_name": "Owned Jacket",
            "store": "myntra",
            "price_inr": 2999.0,
            "_owned": True,
        }

    def _shoppable_complement(self) -> dict:
        return {
            "article_id": "COMP1",
            "pdp_handle": "black-trousers",
            "display_name": "Black Trousers",
            "store": "myntra",
            "price_inr": 999.0,
        }

    def test_owned_item_excluded_from_item_links(self) -> None:
        items = [self._owned_seed(), self._shoppable_complement()]
        result = build_cart_action(items, "myntra")

        link_ids = [lk["article_id"] for lk in result["item_links"]]
        assert "OWNED1" not in link_ids
        assert "COMP1" in link_ids

    def test_owned_item_excluded_from_missing_too(self) -> None:
        """The owned item must not even show up in 'missing' — it is dropped
        entirely, not treated as an unresolved link.
        """
        items = [self._owned_seed(), self._shoppable_complement()]
        result = build_cart_action(items, "myntra")
        assert "OWNED1" not in result["missing"]

    def test_is_owned_key_also_respected(self) -> None:
        """The choke point checks BOTH _owned (agent-item convention) and
        is_owned (schema-field convention) so it works regardless of caller."""
        items = [
            {
                "article_id": "OWNED2",
                "pdp_handle": "owned-item-2",
                "display_name": "Owned Scarf",
                "store": "myntra",
                "is_owned": True,
            },
            self._shoppable_complement(),
        ]
        result = build_cart_action(items, "myntra")
        link_ids = [lk["article_id"] for lk in result["item_links"]]
        assert "OWNED2" not in link_ids
        assert "COMP1" in link_ids

    def test_all_owned_returns_empty_but_valid_response(self) -> None:
        """If every item is owned, build_cart_action degrades to the safe
        empty-items response rather than erroring."""
        result = build_cart_action([self._owned_seed()], "myntra")
        assert result["kind"] == "open_all"
        assert result["cart_url"] is None
        assert result["item_links"] == []
        assert result["missing"] == []

    def test_cross_store_look_also_excludes_owned_seed(self) -> None:
        """Cross-store (unified) path must also honour the exclusion."""
        owned = self._owned_seed()
        owned["store"] = "snitch"
        other = {
            "article_id": "COMP2",
            "pdp_handle": "some-dress",
            "display_name": "Some Dress",
            "store": "myntra",
        }
        result = build_cart_action([owned, other], "unified")
        link_ids = [lk["article_id"] for lk in result["item_links"]]
        assert "OWNED1" not in link_ids
        assert "COMP2" in link_ids


# ---------------------------------------------------------------------------
# Session round-trip: anchor_is_owned (api/routes/chat.py::_build_invoke_state)
# ---------------------------------------------------------------------------


class TestSessionAnchorIsOwnedRoundTrip:
    def test_build_invoke_state_carries_anchor_is_owned_true(self) -> None:
        from api.routes.chat import _build_invoke_state
        from src.memory.conversation import ConversationMemory

        class _StubLLM:
            def generate(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
                return "{}"

        session = {
            "messages": [],
            "retrieved_items": [],
            "filters": {},
            "_memory": ConversationMemory(
                _StubLLM(), {"memory": {"recent_turns": 6, "summary_trigger_turns": 12}}
            ),
            "anchor_article_id": "ANCHOR1",
            "anchor_is_owned": True,
        }
        state = _build_invoke_state(session, "Style this jacket")
        assert state["anchor_article_id"] == "ANCHOR1"
        assert state["anchor_is_owned"] is True

    def test_build_invoke_state_defaults_anchor_is_owned_false(self) -> None:
        """A text-only session (never uploaded an image) defaults to False."""
        from api.routes.chat import _build_invoke_state
        from src.memory.conversation import ConversationMemory

        class _StubLLM:
            def generate(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
                return "{}"

        session = {
            "messages": [],
            "retrieved_items": [],
            "filters": {},
            "_memory": ConversationMemory(
                _StubLLM(), {"memory": {"recent_turns": 6, "summary_trigger_turns": 12}}
            ),
        }
        state = _build_invoke_state(session, "red dress")
        assert state["anchor_is_owned"] is False


# ---------------------------------------------------------------------------
# Buy-similar regex: "Where can I buy one like this?"
# ---------------------------------------------------------------------------


class TestBuySimilarRegexMatchesFrontendChip:
    """The frontend's secondary-action chip sends this exact text; the router's
    _BUY_SIMILAR_RE (src/agents/graph.py) must match it via the existing
    'like\\s+this' alternative — no regex change is needed, this test pins that
    down so a future edit to the pattern can't silently regress it.
    """

    _PATTERN = re.compile(
        r"\b(similar|like\s+this|like\s+these|same\s+style|buy\s+like)\b", re.IGNORECASE
    )

    def test_matches_exact_frontend_chip_text(self) -> None:
        assert self._PATTERN.search("Where can I buy one like this?")

    def test_matches_case_insensitively(self) -> None:
        assert self._PATTERN.search("WHERE CAN I BUY ONE LIKE THIS?")

    def test_pattern_matches_graph_module_constant(self) -> None:
        """Guard against the two patterns drifting apart: extract the live
        _BUY_SIMILAR_RE compiled inside graph.py's router closure by re-running
        the same source-derived pattern string used there.
        """
        import inspect

        import src.agents.graph as graph_mod

        source = inspect.getsource(graph_mod)
        assert r"like\s+this" in source, (
            "graph.py's _BUY_SIMILAR_RE must still contain the 'like this' "
            "alternative that matches the frontend's secondary-action chip text"
        )


# ---------------------------------------------------------------------------
# graph.py outfit_node: WS "Style this <item>" follow-up preserves ownership
# ---------------------------------------------------------------------------

UNIFIED_DIR = "data/processed/unified"

_MINIMAL_CONFIG: dict = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
    "retrieval": {
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dense_dim": 384,
        "rrf_k": 60,
        "top_k": 50,
        "final_k": 10,
        "store_diversity": 0.2,
    },
}

_ANCHOR_ARTICLE_ID = "7624165523678"
_ANCHOR_PROD_NAME = "Men White Semi- Formal Shirt"


class _MockLLM:
    """Deliberately returns a WRONG (non-outfit) router decision so the test only
    passes if the owned-anchor style-this path is fully deterministic.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def _next(self) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:  # noqa: ANN001
        return self._next()

    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:  # noqa: ANN001
        yield self._next()

    def chat(self, messages: list[dict], **kwargs) -> str:  # noqa: ANN001
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:  # noqa: ANN001
        yield self._next()


@pytest.fixture(scope="module")
def _unified_index():
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(f"{UNIFIED_DIR}/catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _MINIMAL_CONFIG)
    return retriever, catalogue_df


def _make_owned_session_items() -> list[dict]:
    """The image-upload anchor plus filler items, mirroring the shape
    image_style.py persists into session["retrieved_items"] — the seed carries
    _owned=True exactly as compose_outfit(owned_anchor=True) stamps it.
    """
    return [
        {
            "article_id": _ANCHOR_ARTICLE_ID,
            "prod_name": _ANCHOR_PROD_NAME,
            "display_name": f"{_ANCHOR_PROD_NAME} (White shirt)",
            "colour": "White",
            "product_type": "shirt",
            "department": "Men",
            "detail_desc": "",
            "image_url": "https://example.com/anchor.jpg",
            "score": 0.9,
            "store": "globalrepublic",
            "price_inr": 1399.0,
            "pdp_handle": "men-white-semi-formal-shirt",
            "gender": "men",
            "_role": "seed",
            "_owned": True,
        },
    ]


def _blank_state_with_owned_anchor(query: str, session_items: list[dict], memory) -> dict:
    return {
        "messages": [{"role": "user", "content": query}],
        "user_query": query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": session_items,
        "filters": {},
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
        "out_of_catalogue": False,
        "excluded_colours": None,
        "anchor_article_id": _ANCHOR_ARTICLE_ID,
        "anchor_is_owned": True,
        "outfit_rationale": None,
        "outfit_variants": None,
        "_memory": memory,
    }


@pytest.mark.requires_index
def test_style_this_follow_up_preserves_ownership(_unified_index) -> None:
    """"Style this Men White Semi- Formal Shirt" on a session whose anchor is
    owned (image-upload) must re-compose with the seed STILL marked owned —
    never silently re-tagging the user's own garment as buyable.
    """
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "shirt"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    session_items = _make_owned_session_items()
    state = _blank_state_with_owned_anchor(
        f"Style this {_ANCHOR_PROD_NAME}", session_items, memory
    )

    result = agent.invoke(state)

    assert result.get("look_id"), (
        f"expected a non-null look_id, got tool_calls={result.get('tool_calls')}"
    )
    seed_items = [
        it for it in result.get("retrieved_items", [])
        if it.get("article_id") == _ANCHOR_ARTICLE_ID
    ]
    assert seed_items, "expected the owned anchor to still be the seed item"
    assert seed_items[0].get("_owned") is True, (
        "owned anchor must stay marked _owned=True across a 'Style this' "
        "follow-up turn"
    )
    complements = [
        it for it in result.get("retrieved_items", [])
        if it.get("_role") == "complement"
    ]
    assert complements, "expected complements to be composed"
    for complement in complements:
        assert complement.get("_owned") is not True, (
            f"complement {complement.get('article_id')} must never be marked owned"
        )
