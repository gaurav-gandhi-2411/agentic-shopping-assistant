"""'More options': alternate anchors under the budget, built from the anchor stage's own candidates.

Hermetic: stub retriever / stubbed composition. The real-index check (Sangeet Rs 3000, women)
skips without a built unified index (OUTFIT_SNAPSHOT_INDEX_DIR).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from api.routes.chat import _build_more_options
from api.schemas import ChatResponse, ItemSummary
from src.agents import graph as graph_mod
from src.agents.outfit import composer
from src.agents.outfit.composer import (
    MORE_OPTIONS_MAX,
    MORE_OPTIONS_MAX_PER_STORE,
    build_more_options,
    compose_outfit,
)
from src.memory.conversation import ConversationMemory

_STORES = ["snitch", "powerlook", "fashor", "virgio"]  # Shopify slugs with a PDP template


def _a(
    aid: str,
    name: str,
    pt: str = "lehenga",
    price: float = 900.0,
    store: str = "snitch",
    gender: str = "women",
    handle: str | None = "slug",
) -> dict:
    return {
        "article_id": aid,
        "prod_name": name,
        "display_name": name,
        "colour": "Pink",
        "product_type": pt,
        "department": "",
        "detail_desc": "festive embroidered",
        "gender": gender,
        "price_inr": price,
        "store": store,
        "pdp_handle": handle if handle is None else f"{handle}-{aid}",
        "score": 0.9,
    }


def _look(seed: dict, comps: list[dict], alternates: list[dict] | None) -> dict:
    look: dict = {
        "look_id": "l1",
        "seed_item": seed,
        "complements": comps,
        "occasion": "sangeet",
        "gender": "women",
        "budget_total_inr": 1000.0,
    }
    if alternates is not None:
        look["anchor_alternates"] = alternates
    return look


def _pool(n: int = 14) -> list[dict]:
    return [
        _a(f"p{i}", f"Style {i} Silk Lehenga Set", pt=f"type{i % 5}", price=500 + 50 * i,
           store=_STORES[i % 4])
        for i in range(n)
    ]


def test_absent_without_a_budget() -> None:
    assert build_more_options([_look(_a("s", "Seed"), [], _pool())], None) is None


def test_absent_when_the_anchor_stage_did_not_run() -> None:
    # explicit "Style this" seed / owned anchor: compose_outfit never records alternates
    assert build_more_options([_look(_a("s", "Seed"), [], None)], 3000.0) is None


def test_empty_list_when_nothing_is_eligible() -> None:
    assert build_more_options([_look(_a("s", "Seed"), [], [])], 3000.0) == []


def test_count_budget_and_exclusions() -> None:
    seed = _a("s", "Seed Lehenga", price=700)
    comp = _a("c1", "Gold Jhumka", pt="jewellery", price=300)
    variant_seed = _a("p3", "Style 3 Silk Lehenga Set", pt="type3", price=650)  # also a pool item
    pool = [seed, *_pool()]
    looks = [_look(seed, [comp], pool), _look(variant_seed, [], None)]
    out = build_more_options(looks, 1000.0)
    assert out is not None and 0 < len(out) <= MORE_OPTIONS_MAX
    assert all(0 < it["price_inr"] <= 1000.0 for it in out)
    ids = {it["article_id"] for it in out}
    assert "s" not in ids and "c1" not in ids and "p3" not in ids  # seed, complements, variants
    assert len(ids) == len(out)


def test_at_most_two_per_store_and_distinct_types_preferred() -> None:
    # 10 snitch items first (relevance order), then one each from other stores
    pool = [_a(f"x{i}", f"Snitch Silk Kurta Set {i}", pt="kurta", store="snitch") for i in range(10)]
    pool += [_a("o1", "Power Silk Lehenga", pt="lehenga", store="powerlook"),
             _a("o2", "Fashor Cotton Anarkali", pt="anarkali", store="fashor"),
             _a("o3", "Virgio Silk Saree", pt="saree", store="virgio")]
    out = build_more_options([_look(_a("s", "Seed"), [], pool)], 5000.0)
    assert out is not None
    per_store: dict[str, int] = {}
    for it in out:
        per_store[it["store"]] = per_store.get(it["store"], 0) + 1
    assert max(per_store.values()) <= MORE_OPTIONS_MAX_PER_STORE == 2
    assert per_store == {"snitch": 2, "powerlook": 1, "fashor": 1, "virgio": 1}
    # distinct types first (relevance order: x0, then the other stores), then the repeat fills
    assert [it["article_id"] for it in out] == ["x0", "o1", "o2", "o3", "x1"]


def test_distinct_product_types_come_before_repeats() -> None:
    pool = [
        _a("a1", "Piece A1 Lehenga", pt="lehenga", store="snitch"),
        _a("a2", "Piece A2 Lehenga", pt="lehenga", store="powerlook"),
        _a("a3", "Piece A3 Lehenga", pt="lehenga", store="fashor"),
        _a("b1", "Piece B1 Anarkali", pt="anarkali", store="virgio"),
        _a("c1", "Piece C1 Sharara Set", pt="sharara", store="snitch"),
    ]
    out = build_more_options([_look(_a("s", "Seed"), [], pool)], 5000.0, limit=3)
    assert [it["article_id"] for it in out or []] == ["a1", "b1", "c1"]  # 3 distinct types


def test_non_fashion_and_unsellable_rows_are_dropped() -> None:
    pool = [
        _a("ok", "Pink Silk Lehenga"),
        _a("acc", "Gold Jhumka Earrings", pt="jewellery"),
        _a("shoe", "Gold Juttis", pt="footwear"),
        _a("bolt", "Unstitched Dress Material Fabric", pt="fabric_material"),
        _a("hamper", "The Shringaar Karwa Chauth Hamper", pt="Gift Hamper"),
        _a("giftkurta", "Embroidered Kurta Rakhi Gift Box", pt="Gift Hamper"),  # name looks garment
        _a("mystery", "Mystery Item 42", pt="Fashion"),  # unclassifiable: not an anchor
        _a("noprice", "Cotton Lehenga", price=0.0),
        _a("nolink", "Linkless Lehenga", handle=None),
        _a("pricey", "Bridal Lehenga", price=4000.0),
    ]
    out = build_more_options([_look(_a("s", "Seed"), [], pool)], 3000.0)
    assert [it["article_id"] for it in out or []] == ["ok"]


def test_footwear_typed_rows_never_anchor_even_when_the_name_says_sherwani() -> None:
    """D1: classify_item('footwear', "... Men's Sherwani Jutti") is men_formalwear (name token
    beats the facet), which let 'Sherwani Juttis' into the row. The structured type wins."""
    pool = [
        _a("jutti1", "RoyalGold Silk Men's Sherwani Jutti", pt="footwear", store="snitch"),
        _a("jutti2", "Black Silk Men's Sherwani Juttis", pt="footwear", store="powerlook"),
        _a("mojari", "Cream Jute Men's Wedding Sherwani Mojaris", pt="footwear", store="fashor"),
        _a("ok", "Men's Cotton Kurta Set", pt="kurta", store="virgio"),
    ]
    for it in pool:
        it["gender"] = "men"
    out = build_more_options([_look(_a("s", "Seed"), [], pool)], 3000.0)
    assert [it["article_id"] for it in out or []] == ["ok"]


def test_swim_inner_and_night_wear_never_anchor() -> None:
    """D2/D3: Bikini Bottom, Panty (globalrepublic) and standalone Pyjama/Churidar nightwear."""
    pool = [
        _a("bikini", "Women Beige Low Rise Bikini Bottom", pt="swimwear", store="snitch"),
        _a("panty", "Women Blue Printed Panty", pt="innerwear", store="powerlook"),
        _a("pyj", "VASTRAMAY Men's Blue Cotton Silk Blend Pyjama", pt="nightwear", store="fashor"),
        _a("churi", "VASTRAMAY Men's Grey Solid Churidar", pt="nightwear", store="virgio"),
        _a("nsuit", "Pink Printed Cotton Night Suit Set", pt="night suit", store="snitch"),
        _a("vest", "Men White Cotton Vest", pt="vest", store="powerlook"),
        _a("typed_generic", "Women Blue Printed Panty", pt="Fashion", store="fashor"),
        _a("ok", "Cotton Anarkali Set", pt="anarkali", store="virgio"),
    ]
    out = build_more_options([_look(_a("s", "Seed"), [], pool)], 3000.0)
    assert [it["article_id"] for it in out or []] == ["ok"]


def test_kurta_and_pyjama_set_typed_nightwear_is_kept_and_standalone_bottoms_stay() -> None:
    pool = [
        _a("kps", "Men's Yellow Lehariya Cotton Kurta & Pyjama Set", pt="nightwear",
           store="snitch"),
        _a("pal", "Cotton Palazzo", pt="palazzo", store="powerlook"),
        _a("pant", "Slim Fit Cotton Pant", pt="trousers", store="fashor"),
    ]
    out = build_more_options([_look(_a("s", "Seed"), [], pool)], 3000.0)
    assert {it["article_id"] for it in out or []} == {"kps", "pal", "pant"}


def test_colourways_of_one_product_collapse() -> None:
    pool = [
        _a("r", "Zari Silk Lehenga Set - Red", store="snitch"),
        _a("b", "Zari Silk Lehenga Set - Red", store="powerlook"),  # same normalised name
        _a("g", "Cotton Anarkali Set", pt="anarkali", store="fashor"),
    ]
    out = build_more_options([_look(_a("s", "Seed"), [], pool)], 3000.0)
    assert [it["article_id"] for it in out or []] == ["r", "g"]


def test_builder_is_cheap() -> None:
    pool = _pool(30)
    t0 = time.perf_counter()
    for _ in range(20):
        build_more_options([_look(_a("s", "Seed"), [], pool)], 5000.0)
    assert (time.perf_counter() - t0) / 20 < 0.05  # <= 50 ms per turn (typically <1 ms)


# --- the anchor stage's gates feed the row (through compose_outfit) --------------------------


class _Stub:
    def __init__(self, anchors: list[dict]) -> None:
        self.anchors = anchors
        self.catalogue_df = pd.DataFrame()

    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
        q = query.lower()
        if "footwear" in q or "juttis" in q or "dupatta" in q or "jewellery" in q:
            return []
        cap = (filters or {}).get("price_max")
        return [a for a in self.anchors if cap is None or a["price_inr"] <= cap][:top_k]


def test_anchor_stage_gates_apply_to_the_row() -> None:
    anchors = [
        _a("seed", "Pink Embroidered Lehenga Choli Set", price=600),
        _a("ok1", "Gold Sequin Lehenga", price=700, store="powerlook"),
        _a("kid", "Girls Pink Lehenga Frock Kids", price=500, store="fashor"),
        _a("man", "Men Pink Kurta Set", pt="kurta", price=500, store="virgio"),  # name conflict
        _a("mlabel", "Pink Silk Lehenga", price=500, gender="men", store="snitch"),
    ]
    look = compose_outfit(
        pd.DataFrame(), _Stub(anchors), occasion_slug="sangeet", gender="women", budget_inr=3000
    )
    out = build_more_options([look], 3000.0)
    ids = {it["article_id"] for it in out or []}
    assert look["seed_item"]["article_id"] == "seed"
    assert ids == {"ok1"}  # kid, name-conflict and wrong-gender-label rows never get there


def test_no_alternates_recorded_without_a_budget() -> None:
    look = compose_outfit(
        pd.DataFrame(), _Stub([_a("seed", "Pink Embroidered Lehenga Choli Set")]),
        occasion_slug="sangeet", gender="women",
    )
    assert "anchor_alternates" not in look
    assert build_more_options([look], None) is None


# --- payload schema ------------------------------------------------------------------------------


def test_payload_cards_carry_the_server_built_buy_link() -> None:
    out = _build_more_options({"more_options": [_a("p1", "Style 1 Lehenga")]})
    assert out is not None and len(out) == 1
    card = out[0]
    assert isinstance(card, ItemSummary)
    assert card.pdp_url and card.pdp_url.startswith("http")
    assert card.store == "snitch" and card.price_inr == 900.0 and card.slot_role is None


def test_payload_absent_vs_empty_vs_filled() -> None:
    assert _build_more_options({}) is None
    assert _build_more_options({"more_options": None}) is None
    assert _build_more_options({"more_options": []}) == []


def test_malformed_entry_is_skipped_not_fatal() -> None:
    out = _build_more_options({"more_options": [_a("p1", "Style 1 Lehenga"), "not-a-dict"]})
    assert out is not None and [c.article_id for c in out] == ["p1"]


def test_chat_response_is_backward_compatible_and_serialises() -> None:
    old = ChatResponse(conversation_id="c", response="hi")  # an old-shape response
    assert old.more_options is None and old.more_options_budget_inr is None
    new = ChatResponse(
        conversation_id="c",
        response="hi",
        more_options=_build_more_options({"more_options": [_a("p1", "Style 1 Lehenga")]}),
        more_options_budget_inr=3000.0,
    )
    dumped = new.model_dump()
    assert dumped["more_options_budget_inr"] == 3000.0
    assert dumped["more_options"][0]["article_id"] == "p1"
    assert set(dumped["more_options"][0]) >= {
        "article_id", "prod_name", "price_inr", "store", "store_display", "pdp_url", "image_url",
    }


# --- graph: outfit node ---------------------------------------------------------------------------


_CATALOGUE = pd.DataFrame(
    {"article_id": ["a"], "colour_group_name": ["Red"], "product_type_name": ["Kurta"]}
)
_CFG: dict = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
    "retrieval": {"final_k": 10, "top_k": 50, "rrf_k": 60},
}


class _NoLLM:
    def generate(self, *a: Any, **k: Any) -> str:
        raise RuntimeError("no LLM in this test")

    def generate_stream(self, *a: Any, **k: Any) -> Any:
        raise RuntimeError("no LLM in this test")


def _run_node(monkeypatch: pytest.MonkeyPatch, query: str, alternates: list[dict] | None) -> dict:
    def probe(**kw: Any) -> dict:
        return _look(_a("seed", "Seed Lehenga", price=600), [], alternates)

    def biased(**kw: Any) -> dict | None:
        return None  # no extra variants: the base look alone

    monkeypatch.setattr(graph_mod, "compose_outfit_tool", probe)
    monkeypatch.setattr(composer, "compose_biased_look", biased)
    llm = _NoLLM()
    agent = graph_mod.build_graph(None, _CATALOGUE, llm, _CFG, streaming_mode=False)  # type: ignore[arg-type]
    memory = ConversationMemory(llm, _CFG)  # type: ignore[arg-type]
    state = {
        "messages": [{"role": "user", "content": query}], "user_query": query,
        "current_plan": None, "tool_calls": [], "retrieved_items": [], "filters": {},
        "final_answer": None, "iteration": 0, "new_items_this_turn": False,
        "out_of_catalogue": False, "excluded_colours": None, "anchor_article_id": None,
        "outfit_rationale": None, "outfit_variants": None, "_memory": memory,
    }
    return agent.invoke(state)


def test_outfit_node_emits_more_options_for_a_budgeted_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = [_a("seed", "Seed Lehenga", price=600), *_pool(8)]
    res = _run_node(monkeypatch, "sangeet outfit under ₹3000 for women", pool)
    assert res["more_options_budget_inr"] == 3000
    mo = res["more_options"]
    assert mo and len(mo) <= 6 and all(it["price_inr"] <= 3000 for it in mo)
    assert "seed" not in {it["article_id"] for it in mo}
    # the internal pool never leaks into persisted variant state
    assert all("anchor_alternates" not in v for v in res["outfit_variants"])


def test_outfit_node_has_no_row_without_a_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    res = _run_node(monkeypatch, "sangeet outfit for women", [])
    assert res.get("more_options") is None


_INDEX_DIR = Path(os.environ.get("OUTFIT_SNAPSHOT_INDEX_DIR", "data/processed/unified"))


def test_real_index_sangeet_3000_women_row() -> None:
    if not (_INDEX_DIR / "catalogue.parquet").exists():
        pytest.skip(f"no built unified index at {_INDEX_DIR}")
    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        pytest.skip("public view only")
    from scripts import outfit_compose_snapshot as snap

    retriever, df = snap.load_retriever(_INDEX_DIR)
    look = compose_outfit(df, retriever, occasion_slug="sangeet", gender="women", budget_inr=3000)
    t0 = time.perf_counter()
    row = build_more_options([look], 3000.0)
    assert (time.perf_counter() - t0) < 0.05
    assert row is not None and 3 <= len(row) <= 6
    assert all(r["price_inr"] <= 3000 and r["gender"] == "women" for r in row)
    per_store: dict[str, int] = {}
    for r in row:
        per_store[r["store"]] = per_store.get(r["store"], 0) + 1
    assert max(per_store.values()) <= 2
    shown = {look["seed_item"]["article_id"], *(c["article_id"] for c in look["complements"])}
    assert not shown & {r["article_id"] for r in row}


def _real_index_or_skip() -> None:
    if not (_INDEX_DIR / "catalogue.parquet").exists():
        pytest.skip(f"no built unified index at {_INDEX_DIR}")
    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        pytest.skip("public view only")


def test_real_rows_footwear_swim_inner_night_types_are_never_anchors() -> None:
    """Every REAL public row typed footwear/swimwear/innerwear/vest/briefs/... is rejected, incl.
    the 39 men's 'Sherwani Jutti' rows (kraftojodhpur) that classify as men_formalwear by name."""
    _real_index_or_skip()
    from src.agents.outfit.composer import _NON_ANCHOR_TYPES, _is_outfit_garment
    from src.retrieval.snapshot import public_catalogue

    df = public_catalogue(pd.read_parquet(_INDEX_DIR / "catalogue.parquet"))
    pt = df["product_type_name"].fillna("").astype(str)
    nm = df["prod_name"].fillna("").astype(str)
    typed = df[pt.str.strip().str.lower().isin(_NON_ANCHOR_TYPES)]
    assert len(typed) > 4000  # footwear alone is ~3,985
    sherwani_juttis = df[(pt.str.lower() == "footwear") & nm.str.contains("Sherwani", case=False)]
    assert len(sherwani_juttis) >= 30, "the real D1 rows must be present"
    allowed_exception = typed["prod_name"].astype(str).str.contains(
        r"\b(?:kurta|kurti|sherwani|bandhgala|achkan|lehenga|saree|sari|anarkali|sharara)\b",
        case=False, regex=True,
    ) & ~typed["product_type_name"].astype(str).str.lower().isin({"footwear", "socks"})
    bad = [
        (a, b)
        for a, b, ok in zip(
            typed["product_type_name"].astype(str), typed["prod_name"].astype(str),
            ~allowed_exception, strict=True,
        )
        if ok and _is_outfit_garment(a, b)
    ]
    assert bad == []
    assert not any(
        _is_outfit_garment("footwear", n) for n in sherwani_juttis["prod_name"].astype(str)
    )


def test_real_72_cell_row_grid_has_no_footwear_swim_inner_or_night_cards() -> None:
    _real_index_or_skip()
    from scripts import outfit_compose_snapshot as snap
    from scripts import outfit_more_options_grid as grid

    retriever, df = snap.load_retriever(_INDEX_DIR)
    rows = grid.run_grid(retriever, df)
    assert len(rows) == 72
    s = grid.summarise(rows)
    assert s["audit_counts"].get("footwear", 0) == 0
    assert s["audit_counts"].get("swimwear", 0) == 0
    assert s["audit_counts"].get("innerwear", 0) == 0
    # only the intentionally-kept ethnic set typed nightwear ("Kurta & Pyjama Set") may remain
    night = [leak for leak in s["leaks"] if leak[3] == "nightwear"]
    assert all("kurta" in str(leak[6]).lower() for leak in night)
    assert s["absent_row_cells"] == []
    # the empty rows are catalogue floors: no garment under 0.4 x the budget exists
    assert all(cell[2] == 800 and cell[0] == "men" for cell in s["empty_row_cells"])
