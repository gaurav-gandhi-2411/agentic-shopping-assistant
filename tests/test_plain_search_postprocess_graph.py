"""search_node end to end: de-dup, <= 3 cards per store, budget, gender mix and the refine hint.

Hermetic: synthetic catalogue, mocked dense/sparse retrieval (retrieval order = catalogue order),
a scripted LLM whose reranker reply is not a number list (so the retrieval-order fallback is what
is ranked: deterministic), no network.
"""
from __future__ import annotations

import unittest.mock as mock
from collections.abc import Iterator
from typing import Any

import pandas as pd

from src.agents import search_postprocess as pp
from src.agents.graph import build_graph
from src.memory.conversation import ConversationMemory
from src.retrieval.hybrid_search import HybridRetriever

_CONFIG: dict = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
    "retrieval": {"final_k": 5, "top_k": 50, "rrf_k": 60, "store_diversity": 0.0},
}


class _LLM:
    def generate(self, prompt: str, system: str | None = None, **kw: object) -> str:
        return "Here are some options."

    def generate_stream(self, prompt: str, system: str | None = None, **kw: object) -> Iterator[str]:
        yield "Here are some options."

    def chat(self, messages: list[dict], **kw: object) -> str:
        return "Here are some options."

    def chat_stream(self, messages: list[dict], **kw: object) -> Iterator[str]:
        yield "Here are some options."


def _row(
    i: int, store: str, gender: str, title: str, price: float, colour: str = "Blue"
) -> dict[str, Any]:
    ign = "menswear" if gender == "men" else "ladieswear"
    return {
        "article_id": f"a{i:03d}", "prod_name": title, "display_name": title,
        "detail_desc": "Printed cotton kurta", "image_url": f"https://img.example/{i}.jpg",
        "price_inr": price, "pdp_handle": f"h{i}", "pdp_live": True, "store": store,
        "gender": gender, "colour_group_name": colour, "product_type_name": "kurta",
        "department_name": ign.title(), "index_group_name": ign,
        "facets": {"colour_group_name": colour, "product_type_name": "kurta",
                   "department_name": ign.title(), "index_group_name": ign},
    }


def _run(rows: list[dict], query: str) -> dict:
    cat = pd.DataFrame(rows)
    ids = cat["article_id"].tolist()
    dense = mock.MagicMock()
    dense.search.return_value = [(a, 1.0 / (i + 1)) for i, a in enumerate(ids)]
    dense.article_ids = ids
    sparse = mock.MagicMock()
    sparse.search.return_value = [(a, 1.0 / (i + 1)) for i, a in enumerate(ids)]
    retriever = HybridRetriever(dense, sparse, cat, _CONFIG)
    llm = _LLM()
    agent = build_graph(retriever, cat, llm, _CONFIG, streaming_mode=False)
    state = {
        "messages": [{"role": "user", "content": query}], "user_query": query,
        "current_plan": None, "tool_calls": [], "retrieved_items": [], "filters": {},
        "final_answer": None, "iteration": 0, "new_items_this_turn": False,
        "out_of_catalogue": False, "excluded_colours": None, "anchor_article_id": None,
        "anchor_is_owned": False, "outfit_rationale": None, "outfit_variants": None,
        "_memory": ConversationMemory(llm, _CONFIG),
    }
    return agent.invoke(state)


def _mixed_catalogue() -> list[dict]:
    rows: list[dict] = []
    # ranks 1-4: two stores, two exact-title duplicates, all men
    rows.append(_row(0, "kalpraag", "men", "Chaitanya Haldi Kurta Set", 16818.0))
    rows.append(_row(1, "kalpraag", "men", "Abhiijeet Saawant in Chaitanya Haldi Kurta Set", 16818.0))
    rows.append(_row(2, "kisah", "men", "Yellow Printed Kurta", 2999.0))
    rows.append(_row(3, "kisah", "men", "Yellow Printed Kurta", 3299.0))
    rows.append(_row(4, "kisah", "men", "Green Block Kurta", 2499.0))
    rows.append(_row(5, "jadeblue", "men", "Maroon Bandhani Kurta", 1312.0))
    rows.append(_row(6, "jadeblue", "men", "Ivory Silk Kurta", 1640.0))
    # women appear only deeper in the list
    rows.append(_row(7, "libas", "women", "Black Floral Straight Kurta", 1369.0))
    rows.append(_row(8, "libas", "women", "Pink Chikankari Kurta", 1799.0))
    rows.append(_row(9, "libas", "women", "Mustard Cotton Kurta", 1599.0))
    rows.append(_row(10, "fashor", "women", "Teal Anarkali Kurta", 2499.0))
    return rows


def _shown(res: dict) -> list[dict]:
    return res["retrieved_items"]


def test_gender_neutral_query_dedupes_caps_stores_mixes_genders_and_offers_refine() -> None:
    res = _run(_mixed_catalogue(), "kurta")
    items = _shown(res)
    titles = [i["display_name"] for i in items]
    assert len(items) == 5, titles
    stores = [i["store"] for i in items]
    assert max(stores.count(s) for s in set(stores)) <= pp.MAX_PER_STORE, stores
    assert len({t.lower() for t in titles}) == len(titles)  # no exact-title repeat
    assert not (
        "Chaitanya Haldi Kurta Set" in titles
        and "Abhiijeet Saawant in Chaitanya Haldi Kurta Set" in titles
    )
    genders = {i["gender"] for i in items}
    assert genders == {"men", "women"}, genders
    assert res["gender_refine"] == [
        {"gender": "women", "query": "kurta for women"},
        {"gender": "men", "query": "kurta for men"},
    ]


def test_gendered_query_shows_no_refine_and_keeps_one_gender() -> None:
    res = _run(_mixed_catalogue(), "men's kurta")
    assert all(i["gender"] == "men" for i in _shown(res))
    assert not res.get("gender_refine")


def test_single_gender_catalogue_shows_no_refine() -> None:
    rows = [_row(i, ("libas", "fashor", "tjori", "vastramay")[i % 4], "women", f"Saree design {i}", 1000.0 + i) for i in range(12)]
    res = _run(rows, "kurta")
    assert len(_shown(res)) == 5
    assert not res.get("gender_refine")


def test_budget_query_only_shows_items_within_budget() -> None:
    rows = [_row(i, ("libas", "fashor", "tjori", "vastramay", "kisah")[i % 5], "women", f"Kurta design {i}", 900.0 + 250 * i) for i in range(14)]
    res = _run(rows, "kurta under 2000")
    items = _shown(res)
    assert items, "expected some items under 2000"
    assert all(i["price_inr"] <= 2000 for i in items), [i["price_inr"] for i in items]


def test_narrow_pool_shows_fewer_instead_of_relaxing_the_store_cap() -> None:
    rows = [_row(i, "libas", "women", f"Kurta design {i}", 1000.0 + i) for i in range(10)]
    res = _run(rows, "women's kurta")
    stores = [i["store"] for i in _shown(res)]
    # one store, nothing else to backfill from: shown short at the cap, never relaxed
    assert len(stores) == pp.MAX_PER_STORE == 3


def test_cards_dropped_by_a_result_gate_are_refilled_from_deeper_candidates() -> None:
    # haldi gate strips black items: they used to leave the list short with nothing refilling it
    stores = ("libas", "fashor", "tjori", "vastramay", "kisah", "jadeblue", "rathore", "jompers")
    rows = [
        _row(i, stores[i], "women", f"Black Kurta {i}" if i < 3 else f"Yellow Kurta {i}",
             1000.0 + i, colour="Black" if i < 3 else "Yellow")
        for i in range(8)
    ]
    res = _run(rows, "kurta for haldi for women")
    items = _shown(res)
    assert len(items) == 5, [i["display_name"] for i in items]
    assert not any((i.get("colour") or "").lower() == "black" for i in items)


_WIDE_STORES = (
    "libas", "fashor", "tjori", "vastramay", "kisah", "jadeblue", "rathore", "jompers",
    "kalpraag", "mohanlalsons", "powerlook", "globalrepublic", "voylla", "ishhaara",
)


def _colour_rows(order: str) -> list[dict]:
    """One kurta per letter (Y = yellow, B = black), one store each, so the store cap never binds."""
    return [
        _row(i, _WIDE_STORES[i % len(_WIDE_STORES)] + ("" if i < len(_WIDE_STORES) else str(i)),
             "women", f"{'Black' if c == 'B' else 'Yellow'} Kurta {i}", 1000.0 + i,
             colour="Black" if c == "B" else "Yellow")
        for i, c in enumerate(order)
    ]


def test_gate_refill_keeps_going_past_four_passes_while_the_pool_has_more() -> None:
    # Each refill pass admits one new black card at the 5th slot, so the haldi gate drops about one
    # card per pass; a fixed 4-pass cap left 4 cards although yellow kurtas remained deeper.
    res = _run(_colour_rows("YYYY" + "BBBB" + "YYYYYY"), "kurta for haldi for women")
    items = _shown(res)
    assert len(items) == 5, [i["display_name"] for i in items]
    assert not any((i.get("colour") or "").lower() == "black" for i in items)


def test_gate_refill_terminates_on_a_long_one_drop_per_pass_pool() -> None:
    # Adversarial: 30 black cards ahead of yellow ones; the gate drops cards on every pass and each
    # pass strictly removes at least one id from the pool, so the loop must end and still fill 5.
    res = _run(_colour_rows("B" * 30 + "Y" * 6), "kurta for haldi for women")
    items = _shown(res)
    assert len(items) == 5, [i["display_name"] for i in items]


def test_gate_refill_terminates_when_every_card_is_dropped() -> None:
    # Adversarial: everything is dropped by the gate in every pass (loungewear for a wedding-tier
    # ask); the loop must end with the honest empty result, not spin or raise.
    rows = [_row(i, _WIDE_STORES[i], "women", f"Cotton Night Dress {i}", 900.0 + i) for i in range(14)]
    assert _shown(_run(rows, "kurta for sangeet for women")) == []


def _typed_row(i: int, store: str, title: str, price: float, ptype: str) -> dict:
    row = _row(i, store, "women", title, price)
    row["product_type_name"] = ptype
    row["facets"]["product_type_name"] = ptype
    return row


def test_beach_type_cap_refill_respects_store_cap_and_budget_and_dedupe() -> None:
    # The beach per-type cap (2 per product type) used to refill its shortfall from the raw wide
    # candidate pool, bypassing the store cap / budget / dedupe the selection had applied.
    rows = [_typed_row(i, "libas", f"Floral Top {i}", 900.0 + i, "top") for i in range(6)]
    rows += [_typed_row(i, "libas", f"Linen Dress {i}", 1000.0 + i, "dress") for i in range(6, 12)]
    items = _shown(_run(rows, "beach wear for women"))
    stores = [i["store"] for i in items]
    assert items and stores.count("libas") <= pp.MAX_PER_STORE, [i["display_name"] for i in items]

    over = [_typed_row(0, "libas", "Floral Top 0", 900.0, "top"),
            _typed_row(1, "fashor", "Floral Top 1", 901.0, "top"),
            _typed_row(2, "tjori", "Floral Top 2", 902.0, "top")]
    over += [_typed_row(i, _WIDE_STORES[i], f"Linen Dress {i}", 5000.0, "dress") for i in range(3, 8)]
    capped = _shown(_run(over, "beach wear for women under 1500"))
    assert capped and all(i["price_inr"] <= 1500 for i in capped), [i["price_inr"] for i in capped]


def test_gate_that_empties_the_list_is_not_refilled() -> None:
    # honest zero-results: a loungewear-only pool for a wedding-tier ask stays empty
    stores = ("libas", "fashor", "tjori", "vastramay", "kisah", "jadeblue")
    rows = [_row(i, stores[i], "women", f"Cotton Night Dress {i}", 900.0 + i) for i in range(6)]
    res = _run(rows, "kurta for sangeet for women")
    assert _shown(res) == []
