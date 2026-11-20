"""Budget-aware outfit composition: anchor cap, footwear floor, price pushdown, window retry.

Hermetic: a stub retriever records every search (query, top_k, filters). The real-index grid at
the bottom skips when no unified index is built (set OUTFIT_SNAPSHOT_INDEX_DIR).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.agents.outfit import composer
from src.agents.outfit.composer import compose_outfit, footwear_price_floor


def _item(aid: str, name: str, pt: str, price: float, *, gender: str = "women") -> dict:
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
        "store": f"store{aid}",
        "score": 0.9,
    }


class _Stub:
    """Search results by slot keyword; honours price_max like the real retriever."""

    def __init__(self, anchors: list[dict], accessories: list[dict], footwear: list[dict]) -> None:
        self.anchors, self.accessories, self.footwear = anchors, accessories, footwear
        self.calls: list[dict[str, Any]] = []
        self.catalogue_df = pd.DataFrame()

    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
        self.calls.append({"query": query, "top_k": top_k, "filters": dict(filters or {})})
        q = query.lower()
        if "footwear" in q or "juttis" in q:
            pool = self.footwear
        elif "dupatta" in q or "jewellery" in q or "clutch" in q or "accessory" in q:
            pool = self.accessories
        else:
            pool = self.anchors
        cap = (filters or {}).get("price_max")
        return [p for p in pool if cap is None or p["price_inr"] <= cap][:top_k]


def _stub() -> _Stub:
    anchors = [
        _item("a1", "Pink Embroidered Lehenga Choli Set", "lehenga", 2400),
        _item("a2", "Gold Sequin Lehenga", "lehenga", 1100),
        _item("a3", "Silk Lehenga Skirt Set", "lehenga", 700),
    ]
    acc = [_item("c1", "Gold Jhumka Earrings", "jewellery", 349)]
    shoes = [_item("f1", "Embellished Juttis", "footwear", 799)]
    return _Stub(anchors, acc, shoes)


def _anchor_calls(s: _Stub) -> list[dict]:
    return [c for c in s.calls if "lehenga" in c["query"].lower() or "sangeet" in c["query"].lower()]


def test_no_budget_means_no_price_pushdown_anywhere() -> None:
    s = _stub()
    compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women")
    assert s.calls and all("price_max" not in c["filters"] for c in s.calls)


def test_anchor_search_is_capped_at_the_fraction_of_budget() -> None:
    s = _stub()
    look = compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women", budget_inr=3000)
    assert s.calls[0]["filters"]["price_max"] == pytest.approx(3000 * composer.ANCHOR_BUDGET_FRACTION)
    assert look["seed_item"]["price_inr"] <= 3000 * composer.ANCHOR_BUDGET_FRACTION


def test_anchor_falls_back_to_cheapest_fitting_when_cap_leaves_nothing() -> None:
    s = _stub()
    # cap = 0.4 * 1500 = 600 < every anchor (cheapest 700): widened search at the full budget
    look = compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women", budget_inr=1500)
    assert look["seed_item"]["article_id"] == "a3"  # cheapest of the in-budget anchors
    assert [c["filters"].get("price_max") for c in s.calls[:2]] == [600.0, 1500]


def test_budget_total_never_exceeds_budget() -> None:
    for budget in (1000, 1500, 3000, 5000):
        look = compose_outfit(
            pd.DataFrame(), _stub(), occasion_slug="sangeet", gender="women", budget_inr=budget
        )
        total = look.get("budget_total_inr")
        assert total is None or total <= budget


def test_slot_searches_push_the_remaining_budget() -> None:
    s = _stub()
    compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women", budget_inr=3000)
    slot_calls = [c for c in s.calls[1:] if c["filters"].get("gender") == "women"]
    assert slot_calls and all("price_max" in c["filters"] for c in slot_calls)
    assert all(c["filters"]["price_max"] <= 3000 for c in slot_calls)


def test_accessory_cannot_spend_the_footwear_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _stub()
    monkeypatch.setattr(composer, "footwear_price_floor", lambda *_a, **_k: 700.0)
    compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women", budget_inr=3000)
    seed_price = s.anchors[1]["price_inr"]  # a2: the highest anchor under the 1200 cap
    acc = next(c for c in s.calls if "dupatta" in c["query"].lower())
    # remaining after the seed, minus the reserved footwear floor
    assert acc["filters"]["price_max"] == pytest.approx(3000 - seed_price - 700.0)
    fw = next(c for c in s.calls if "footwear" in c["query"].lower())
    assert fw["filters"]["price_max"] > acc["filters"]["price_max"]  # footwear may use the floor


def test_optional_slot_before_required_footwear_also_reserves_the_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agents.outfit.slots import SlotSpec

    s = _stub()
    monkeypatch.setattr(composer, "footwear_price_floor", lambda *_a, **_k: 700.0)
    monkeypatch.setattr(
        composer,
        "get_fill_slots",
        lambda *_a, **_k: [
            SlotSpec("outerwear", "jacket blazer coat cardigan", required=False),
            SlotSpec("footwear", "heels juttis ethnic footwear", required=True),
        ],
    )
    compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women", budget_inr=3000)
    outer = next(c for c in s.calls if "jacket" in c["query"].lower())
    assert outer["filters"]["price_max"] == pytest.approx(3000 - s.anchors[1]["price_inr"] - 700.0)


def test_required_non_footwear_slot_is_not_starved_by_the_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agents.outfit.slots import SlotSpec

    s = _stub()
    monkeypatch.setattr(composer, "footwear_price_floor", lambda *_a, **_k: 700.0)
    monkeypatch.setattr(
        composer,
        "get_fill_slots",
        lambda *_a, **_k: [
            SlotSpec("bottom", "trousers jacket", required=True),
            SlotSpec("footwear", "heels juttis ethnic footwear", required=True),
        ],
    )
    compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women", budget_inr=3000)
    bottom = next(c for c in s.calls if "trousers" in c["query"].lower())
    assert bottom["filters"]["price_max"] == pytest.approx(3000 - s.anchors[1]["price_inr"])


def test_no_reservation_when_footwear_cannot_fit_anyway(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _stub()
    monkeypatch.setattr(composer, "footwear_price_floor", lambda *_a, **_k: 99999.0)
    compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women", budget_inr=3000)
    acc = next(c for c in s.calls if "dupatta" in c["query"].lower())
    assert acc["filters"]["price_max"] == pytest.approx(3000 - s.anchors[1]["price_inr"])


def test_empty_pool_under_a_budget_retries_once_with_a_wider_window() -> None:
    s = _stub()

    class _Narrow(_Stub):
        def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
            res = super().search(query, top_k, filters)
            q = query.lower()
            if ("footwear" in q or "juttis" in q) and top_k < composer._BUDGET_RETRY_TOP_K:
                return []  # the cheap footwear sits below the normal window
            return res

    n = _Narrow(s.anchors, s.accessories, s.footwear)
    look = compose_outfit(pd.DataFrame(), n, occasion_slug="sangeet", gender="women", budget_inr=3000)
    assert any(c["_slot"] == "footwear" for c in look["complements"])
    assert any(c["top_k"] == composer._BUDGET_RETRY_TOP_K for c in n.calls)


def test_footwear_floor_failure_never_blocks_a_budgeted_look(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*_a: Any, **_k: Any) -> float:
        raise RuntimeError("injected floor failure")

    monkeypatch.setattr(composer, "footwear_price_floor", boom)
    monkeypatch.setattr(composer, "_FLOOR_FAILURE_LOGGED", False)
    stubs = [_stub(), _stub()]
    with caplog.at_level("WARNING", logger=composer.logger.name):
        looks = [
            compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women", budget_inr=3000)
            for s in stubs
        ]
    for look in looks:  # still a look, in budget, composed with no footwear reservation
        assert look["seed_item"] is not None
        assert (look.get("budget_total_inr") or 0) <= 3000
    acc = next(c for c in stubs[0].calls if "dupatta" in c["query"].lower())
    assert acc["filters"]["price_max"] == pytest.approx(3000 - stubs[0].anchors[1]["price_inr"])
    # logged once per process (two requests, one record), with the traceback
    msgs = [r for r in caplog.records if "footwear price floor failed" in r.getMessage()]
    assert len(msgs) == 1 and msgs[0].exc_info is not None


class _IgnoresPriceMax(_Stub):
    """A retriever whose search silently ignores price_max (a fake or a future backend)."""

    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
        f = {k: v for k, v in (filters or {}).items() if k != "price_max"}
        return super().search(query, top_k, f)


def test_all_anchors_over_budget_stay_out_even_if_retriever_ignores_price_max() -> None:
    anchors = [
        _item("o1", "Silk Lehenga Skirt Set", "lehenga", 900),
        _item("o2", "Gold Sequin Lehenga", "lehenga", 1100),
    ]
    s = _IgnoresPriceMax(anchors, [], [])
    # budget 500: both the capped and the widened search return over-budget anchors anyway; the
    # budget gate (not the pushdown) must keep them out and give the honest budget message
    look = compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women", budget_inr=500)
    assert look["seed_item"] is None
    assert "500" in look["outfit_rationale"]


def test_widened_fallback_picks_cheapest_in_budget_when_price_max_is_ignored() -> None:
    anchors = [
        _item("o1", "Silk Lehenga Skirt Set", "lehenga", 900),
        _item("o2", "Gold Sequin Lehenga", "lehenga", 1700),  # over the 1500 budget
        _item("o3", "Cotton Lehenga Set", "lehenga", 700),
    ]

    class _CapEmptyThenIgnores(_IgnoresPriceMax):
        def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
            cap = (filters or {}).get("price_max")
            if cap is not None and cap < 1500:
                return []  # the anchor-cap search finds nothing under 600
            return super().search(query, top_k, filters)  # widened search: ignores price_max

    s = _CapEmptyThenIgnores(anchors, [], [])
    # the widened search returns all three (incl. the 1700 one); only in-budget anchors may
    # be chosen, cheapest first
    look = compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women", budget_inr=1500)
    assert look["seed_item"]["article_id"] == "o3"


def test_no_retry_without_a_budget() -> None:
    s = _stub()
    s.footwear = []  # genuinely none
    compose_outfit(pd.DataFrame(), s, occasion_slug="sangeet", gender="women")
    assert all(c["top_k"] != composer._BUDGET_RETRY_TOP_K for c in s.calls)


# --- footwear floor derivation ---------------------------------------------------------------


def _catalogue() -> pd.DataFrame:
    rows = [
        ("1", "Women Gold Heels", "footwear", "women", 899.0),
        ("2", "Women Embellished Juttis", "footwear", "women", 649.0),
        ("3", "Women Rugged Hiking Boots", "footwear", "women", 199.0),  # rugged: never sangeet
        ("4", "Men Brown Juttis", "footwear", "men", 349.0),
        ("5", "Women Silk Kurta", "kurta", "women", 99.0),  # not footwear
        ("6", "Women Bad Price Heels", "footwear", "women", 0.0),  # unpriced: ignored
    ]
    df = pd.DataFrame(
        rows, columns=["article_id", "prod_name", "product_type_name", "gender", "price_inr"]
    )
    df["display_name"] = df["prod_name"]
    df["detail_desc"] = ""
    df["store"] = "s"
    df["facets"] = [{} for _ in range(len(df))]
    return df.set_index("article_id")


class _Cat:
    def __init__(self) -> None:
        self.catalogue_df = _catalogue()


def test_footwear_floor_is_the_cheapest_gate_passing_footwear_per_gender() -> None:
    r = _Cat()
    # Derived from the data, not a constant; the Rs 199 boots fail sangeet's rugged-footwear gate.
    assert footwear_price_floor(r, "women", "sangeet") == 649.0
    assert footwear_price_floor(r, "men", "sangeet") == 349.0


def test_footwear_floor_none_without_catalogue_or_footwear() -> None:
    assert footwear_price_floor(object(), "women", "sangeet") is None  # type: ignore[arg-type]
    r = _Cat()
    r.catalogue_df = r.catalogue_df[r.catalogue_df["product_type_name"] != "footwear"]
    assert footwear_price_floor(r, "women", "sangeet") is None  # type: ignore[arg-type]


def test_footwear_rows_are_scanned_once_per_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    r = _Cat()
    calls = {"n": 0}
    real = composer._footwear_floor_state

    def counting(df: pd.DataFrame) -> dict:
        calls["n"] += 1
        return real(df)

    monkeypatch.setattr(composer, "_footwear_floor_state", counting)
    for g, occ in (("women", "sangeet"), ("men", "sangeet"), ("women", "office")):
        footwear_price_floor(r, g, occ)  # type: ignore[arg-type]
    assert calls["n"] == 1


def test_floor_resolves_through_the_biased_retriever_wrapper() -> None:
    r = _Cat()
    wrapper = composer._BiasedRetriever(r, "alternate_colour", set(), "sangeet")  # type: ignore[arg-type]
    assert footwear_price_floor(wrapper, "women", "sangeet") == 649.0  # type: ignore[arg-type]
    assert hasattr(r, "_footwear_floor_state")  # cached on the underlying object


# --- real index: the 36-cell grid ----------------------------------------------------------------

_INDEX_DIR = Path(os.environ.get("OUTFIT_SNAPSHOT_INDEX_DIR", "data/processed/unified"))


@pytest.fixture(scope="module")
def _grid() -> list[dict]:
    if not (_INDEX_DIR / "catalogue.parquet").exists():
        pytest.skip(f"no built unified index at {_INDEX_DIR}")
    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        pytest.skip("grid is the public view; INCLUDE_NON_PUBLIC_STORES is set")
    from scripts import outfit_budget_grid as grid
    from scripts import outfit_compose_snapshot as snap

    retriever, df = snap.load_retriever(_INDEX_DIR)
    return grid.run_grid(retriever, df)


def test_grid_never_exceeds_the_budget(_grid: list[dict]) -> None:
    assert len(_grid) == 36
    assert [r for r in _grid if r["total"] is not None and r["total"] > r["budget"]] == []


    # 9/36 before the change; 29/36 on the recorded index build (see ANCHOR_BUDGET_FRACTION), 7 cells
    # being catalogue floors. The exact figure is build-dependent (CI downloads a different build),
    # so only a floor well above the pre-change 9 is asserted.
    assert sum(r["cards"] >= 3 for r in _grid) >= 20


def test_sangeet_3000_women_has_three_cards(_grid: list[dict]) -> None:
    cell = next(
        r
        for r in _grid
        if (r["gender"], r["occasion"], r["budget"]) == ("women", "sangeet", 3000)
    )
    assert cell["cards"] >= 3  # (accessory + footwear on the recorded build; build-dependent)
