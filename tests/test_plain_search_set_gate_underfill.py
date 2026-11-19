"""Regression: the single-garment set gate must not under-fill a plain search.

Live defect (2026-11-16 trace): "sherwani for groom" returned 2 cards and "lehenga for
sangeet" 3, although the retrieval pool held 20 and 16 candidates. Root cause: the
single-garment set exclusion (is_multi_piece_set) ran AFTER rerank() had already cut the pool
to top_k=5, and its pool-underflow protection only guarded against an EMPTY list. Catalogue
listings for these garments are very often "<Garment> Set" (240 of 471 public sherwanis, 44 of
59 lehengas flagged), so 3 of the 5 reranked picks were stripped and nothing backfilled the
slots from the rest of the pool.

The gate now runs on the wide pool BEFORE rerank (same placement as the 2026-07-19
price/formality fix), and demotes instead of removing when fewer than top_k single garments
exist. Hermetic: synthetic catalogue, mocked dense/sparse, scripted LLM, no network.
"""
from __future__ import annotations

import json
import unittest.mock as mock
from collections.abc import Iterator

import pandas as pd

from src.agents.graph import _prefer_single_garment_pool, build_graph
from src.memory.conversation import ConversationMemory
from src.retrieval.hybrid_search import HybridRetriever

_CONFIG: dict = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
    "retrieval": {"final_k": 5, "top_k": 50, "rrf_k": 60, "store_diversity": 0.0},
}


def _item(aid: str, name: str, pt: str = "sherwani") -> dict:
    return {"article_id": aid, "product_type": pt, "prod_name": name, "display_name": name}


class TestPreferSingleGarmentPool:
    def test_drops_sets_when_enough_singles(self) -> None:
        items = [_item("1", "Gold Sherwani Set"), *[_item(str(i), f"Sherwani {i}") for i in range(2, 9)]]
        out = _prefer_single_garment_pool(items, 5)
        assert [i["article_id"] for i in out] == ["2", "3", "4", "5", "6", "7", "8"]

    def test_demotes_sets_behind_singles_when_underflow(self) -> None:
        items = [
            _item("1", "Gold Sherwani Set"),
            _item("2", "Plain Sherwani"),
            _item("3", "Black Sherwani Set"),
        ]
        out = _prefer_single_garment_pool(items, 5)
        # nothing is lost (underfill would be worse than a set card); singles come first
        assert [i["article_id"] for i in out] == ["2", "1", "3"]

    def test_empty_pool(self) -> None:
        assert _prefer_single_garment_pool([], 5) == []


class _ScriptedLLM:
    """Reranker picks candidates 1..5; every other call gets a canned reply."""

    def generate(self, prompt: str, system: str | None = None, **kw: object) -> str:
        if system and "reranker" in system:
            return json.dumps({"selected": [1, 2, 3, 4, 5], "reasoning": "x"})
        return "Here are some options."

    def generate_stream(self, prompt: str, system: str | None = None, **kw: object) -> Iterator[str]:
        yield self.generate(prompt, system)

    def chat(self, messages: list[dict], **kw: object) -> str:
        return "Here are some options."

    def chat_stream(self, messages: list[dict], **kw: object) -> Iterator[str]:
        yield "Here are some options."


def _catalogue(names: list[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "article_id": f"a{i:03d}",
            "prod_name": name,
            "display_name": name,
            "detail_desc": "Embroidered festive wear",
            "image_url": f"https://img.example/{i}.jpg",
            "price_inr": 5000.0 + i,
            "pdp_handle": f"h{i}",
            "pdp_live": True,
            "store": "vastramay",
            "gender": "men",
            "colour_group_name": "Gold",
            "product_type_name": "sherwani",
            "department_name": "Menswear",
            "index_group_name": "menswear",
            "facets": {
                "colour_group_name": "Gold",
                "product_type_name": "sherwani",
                "department_name": "Menswear",
            },
        })
    return pd.DataFrame(rows)


def _run(names: list[str]) -> list[dict]:
    cat = _catalogue(names)
    ids = cat["article_id"].tolist()
    dense = mock.MagicMock()
    dense.search.return_value = [(a, 1.0 / (i + 1)) for i, a in enumerate(ids)]
    dense.article_ids = ids
    sparse = mock.MagicMock()
    sparse.search.return_value = [(a, 1.0 / (i + 1)) for i, a in enumerate(ids)]
    retriever = HybridRetriever(dense, sparse, cat, _CONFIG)
    llm = _ScriptedLLM()
    agent = build_graph(retriever, cat, llm, _CONFIG, streaming_mode=False)
    query = "sherwani for groom"
    state = {
        "messages": [{"role": "user", "content": query}], "user_query": query,
        "current_plan": None, "tool_calls": [], "retrieved_items": [], "filters": {},
        "final_answer": None, "iteration": 0, "new_items_this_turn": False,
        "out_of_catalogue": False, "excluded_colours": None, "anchor_article_id": None,
        "anchor_is_owned": False, "outfit_rationale": None, "outfit_variants": None,
        "_memory": ConversationMemory(llm, _CONFIG),
    }
    return agent.invoke(state)["retrieved_items"]


def test_plain_search_fills_all_slots_when_set_listings_rank_first() -> None:
    # Retrieval order puts "Sherwani Set" listings at ranks 1, 3, 5 -> the reranker's top-5 is
    # 3 sets + 2 singles; the old post-rerank gate left 2 cards.
    names = []
    for i in range(20):
        names.append(f"Royal Gold {i} Sherwani Set" if i in (0, 2, 4, 6, 8) else f"Royal Gold {i} Sherwani")
    items = _run(names)
    assert len(items) == 5
    assert not any(i["display_name"].endswith("Sherwani Set") for i in items)


def test_plain_search_keeps_cards_when_singles_are_scarce() -> None:
    # Only 2 single garments exist: show them first, then fill with sets rather than shrink.
    names = [f"Royal Gold {i} Sherwani Set" for i in range(10)] + ["Plain Sherwani A", "Plain Sherwani B"]
    items = _run(names)
    assert len(items) == 5
    shown = [i["display_name"] for i in items]
    assert "Plain Sherwani A" in shown and "Plain Sherwani B" in shown
