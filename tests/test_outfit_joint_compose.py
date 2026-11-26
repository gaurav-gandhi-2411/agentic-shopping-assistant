"""Joint selection wired into compose_outfit, and the anchor-stage garment gate.

Hermetic: a stub retriever that honours price_max. The greedy path is the comparison baseline
(composer.JOINT_SELECTION_ENABLED = False), kept as the fallback.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pytest

from src.agents.outfit import composer
from src.agents.outfit.composer import _JointChoice, compose_outfit


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
    """Pools by slot keyword, in list order; honours price_max like the real retriever."""

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
        return [dict(p) for p in pool if cap is None or p["price_inr"] <= cap][:top_k]


def _trap_stub() -> _Stub:
    """Greedy takes the first (priciest) accessory and strands the required footwear."""
    return _Stub(
        anchors=[_item("a1", "Pink Embroidered Lehenga Choli Set", "lehenga", 700)],
        accessories=[
            _item("acc_big", "Heavy Gold Jhumka Earrings", "jewellery", 1600),
            _item("acc_small", "Gold Jhumka Earrings", "jewellery", 349),
        ],
        footwear=[_item("shoes", "Embellished Juttis", "footwear", 799)],
    )


def _compose(stub: _Stub, budget: float | None = 2500.0, gender: str = "women") -> dict:
    return compose_outfit(
        pd.DataFrame(), stub, occasion_slug="sangeet", gender=gender, budget_inr=budget
    )


def _cards(look: dict) -> int:
    return (1 if look.get("seed_item") else 0) + len(look.get("complements") or [])


def test_greedy_trap_baseline_strands_footwear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(composer, "JOINT_SELECTION_ENABLED", False)
    look = _compose(_trap_stub())
    assert _cards(look) == 2 and look["empty_slots"] == ["footwear"]


def test_joint_selection_fills_every_required_slot_within_budget() -> None:
    look = _compose(_trap_stub())
    assert _cards(look) == 3 and look["empty_slots"] == []
    assert [c["_slot"] for c in look["complements"]] == ["accessory", "footwear"]
    assert [c["article_id"] for c in look["complements"]] == ["acc_small", "shoes"]
    assert look["budget_total_inr"] == 700 + 349 + 799 <= 2500
    assert all(c["_role"] == "complement" for c in look["complements"])
    assert look["seed_item"]["_role"] == "seed"


def test_joint_result_is_deterministic() -> None:
    first = [c["article_id"] for c in _compose(_trap_stub())["complements"]]
    for _ in range(3):
        assert [c["article_id"] for c in _compose(_trap_stub())["complements"]] == first


def test_no_budget_never_enters_the_joint_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_k: Any) -> None:
        raise AssertionError("joint stage must not run without a budget")

    monkeypatch.setattr(composer, "_joint_selection", boom)
    assert _compose(_trap_stub(), budget=None)["seed_item"] is not None


def test_no_budget_turn_logs_no_joint_fallback_or_any_joint_activity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Parity cannot see joint running on budget None (an error there would silently fall back
    to greedy): the joint stage logs every fallback and its debug line, so none may appear."""
    with caplog.at_level(logging.DEBUG, logger=composer.logger.name):
        assert _compose(_trap_stub(), budget=None)["seed_item"] is not None
    assert [r.getMessage() for r in caplog.records if "[composer/joint]" in r.getMessage()] == []
    assert not any("falling back to greedy" in r.getMessage() for r in caplog.records)


def test_budgeted_turn_does_log_joint_activity() -> None:
    """Control for the test above: the same probe sees the joint stage when it does run."""
    import logging as _l

    records: list[str] = []

    class _H(_l.Handler):
        def emit(self, record: _l.LogRecord) -> None:
            records.append(record.getMessage())

    h = _H(level=_l.DEBUG)
    old_level = composer.logger.level
    composer.logger.addHandler(h)
    composer.logger.setLevel(_l.DEBUG)
    try:
        _compose(_trap_stub())
    finally:
        composer.logger.removeHandler(h)
        composer.logger.setLevel(old_level)
    assert any("[composer/joint]" in m for m in records)


def test_gate_cache_key_separates_anchor_classes() -> None:
    stub = _trap_stub()
    cache: dict = {}
    common: dict[str, Any] = dict(
        query="heels juttis ethnic footwear",
        slot_name="footwear",
        occasion_slug="sangeet",
        gender="women",
        anchor_colour="pink",
        seen_ids=set(),
        seen_prod_colour=set(),
        retriever=stub,
        budget_remaining=2000.0,
        pairing_stats=None,
        gate_cache=cache,
    )
    composer._ranked_candidates(anchor_class="ethnic_one_piece", **common)
    composer._ranked_candidates(anchor_class="ethnic_top", **common)
    assert len(cache) == 2
    composer._ranked_candidates(anchor_class="ethnic_top", **common)
    assert len(cache) == 2  # same class: a hit, not a new entry


def test_timeout_falls_back_to_the_greedy_look_and_logs_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(composer, "JOINT_SELECTION_ENABLED", False)
    greedy = _compose(_trap_stub())
    monkeypatch.setattr(composer, "JOINT_SELECTION_ENABLED", True)
    monkeypatch.setattr(composer, "JOINT_WALL_BUDGET_S", -1.0)  # already expired
    with caplog.at_level(logging.WARNING, logger=composer.logger.name):
        look = _compose(_trap_stub())
    assert [c["article_id"] for c in look["complements"]] == [
        c["article_id"] for c in greedy["complements"]
    ]
    assert any("falling back to greedy" in r.getMessage() for r in caplog.records)


def test_unexpected_joint_error_falls_back_to_greedy(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("injected")

    monkeypatch.setattr(composer, "select_joint", boom)
    assert _cards(_compose(_trap_stub())) == 2  # the greedy look, not a crash


def _fake_choice(stub: _Stub, *, required: int, cards: int) -> _JointChoice:
    seed = dict(stub.anchors[0])
    return _JointChoice(
        seed=seed,
        complements=[],
        empty_slots=["footwear"],
        suppressed_slots=[{"slot": "footwear", "reason": "x"}],
        total_price=seed["price_inr"],
        required_filled=required,
        cards=cards,
    )


@pytest.mark.parametrize(
    ("pt", "name"),
    [
        ("dupatta", "Festive Mirror Embroidered A-Line Pleated with Pant & Scalloped Dupatta - Green"),
        ("dupatta", "Solid Gota Patti Embroidered Straight Fit Kurta Salwar and Dupatta - Red"),
        ("outerwear", "Men Black Mirror Work Nehru Jacket with Kurta, Pyjama & Mirror Dupatta Set"),
        ("coord", "Libas Art Teal Embroidered Silk Blend 3 Piece Co-Ord Sets With Dupatta"),
        ("kurta", "Men Cotton Kurta & Pyjama Set"),
    ],
)
def test_real_sets_with_an_accessory_part_are_garments(pt: str, name: str) -> None:
    assert composer._is_outfit_garment(pt, name)


@pytest.mark.parametrize(
    ("pt", "name"),
    [
        ("dupatta", "Red Banarasi Dupatta"),  # a standalone dupatta is not a set
        ("Rakhi", "Pink Lehenga Rakhi"),
        ("Rakhi", "Designer Rakhi Set With Kurta Pyjama"),  # set + garment words, still a rakhi
        ("belt", "Leather Belt And Kurta Set"),
        ("Belt", "Olive Green & Beige Striped Men's Belt"),
        ("necklace", "Patiala Choker Necklace Set"),
        ("bangles", "Silk Thread Bangles Set With Kurta"),
        ("nightwear", "Pink Cotton Night Suit Set"),
        ("shorts", "Black Floral Print Night Suit Set"),
        ("footwear", "Men Jutti And Kurta Set"),
    ],
)
def test_set_exemption_never_lets_a_non_garment_through(pt: str, name: str) -> None:
    assert not composer._is_outfit_garment(pt, name)


def test_a_set_typed_dupatta_can_be_the_seed() -> None:
    stub = _anchor_stub(
        [
            _item("set", "Pink Embroidered Kurta Salwar and Dupatta Set", "dupatta", 500),
            _item("rakhi", "Pink Lehenga Rakhi", "Rakhi", 300),
        ]
    )
    look = compose_outfit(
        pd.DataFrame(), stub, occasion_slug="sangeet", gender="women", budget_inr=800
    )
    assert look["seed_item"]["article_id"] == "set"


def test_an_incomplete_joint_look_only_replaces_greedy_when_strictly_better(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _trap_stub()
    # greedy here: required 1 of 2 filled, 2 cards. A tie keeps greedy ...
    monkeypatch.setattr(
        composer, "_joint_selection", lambda **_k: _fake_choice(stub, required=1, cards=2)
    )
    kept = _compose(stub)
    assert [c["article_id"] for c in kept["complements"]] == ["acc_big"]
    # ... a strictly better joint look wins (its own complements are used)
    monkeypatch.setattr(
        composer, "_joint_selection", lambda **_k: _fake_choice(stub, required=2, cards=3)
    )
    won = _compose(stub)
    assert won["complements"] == [] and won["empty_slots"] == ["footwear"]


def test_every_joint_item_respects_the_hard_gates() -> None:
    stub = _trap_stub()
    stub.accessories.insert(0, _item("kid", "Girls Kids Gold Earrings", "jewellery", 100))
    stub.accessories.insert(0, _item("men", "Men Gold Chain", "jewellery", 100, gender="men"))
    look = _compose(stub)
    ids = [c["article_id"] for c in look["complements"]]
    assert "kid" not in ids and "men" not in ids


# --- C: garment gate at the anchor stage ----------------------------------------------------


def _anchor_stub(anchors: list[dict]) -> _Stub:
    return _Stub(anchors, [_item("c1", "Gold Jhumka Earrings", "jewellery", 200)], [])


@pytest.mark.parametrize("joint", [True, False])
def test_a_rakhi_or_belt_is_never_the_seed(monkeypatch: pytest.MonkeyPatch, joint: bool) -> None:
    monkeypatch.setattr(composer, "JOINT_SELECTION_ENABLED", joint)
    stub = _anchor_stub(
        [
            _item("rakhi", "Pink Floral Embellished Lehenga Rakhi", "Rakhi", 300),
            _item("belt", "Pink Leather Belt For Lehenga", "belt", 250),
            _item("real", "Pink Embroidered Lehenga Choli", "lehenga", 500),
        ]
    )
    look = compose_outfit(
        pd.DataFrame(), stub, occasion_slug="sangeet", gender="women", budget_inr=800
    )
    assert look["seed_item"]["article_id"] == "real"


@pytest.mark.parametrize("joint", [True, False])
def test_when_no_garment_fits_the_result_is_an_honest_empty_look(
    monkeypatch: pytest.MonkeyPatch, joint: bool
) -> None:
    monkeypatch.setattr(composer, "JOINT_SELECTION_ENABLED", joint)
    stub = _anchor_stub(
        [
            _item("rakhi", "Pink Lehenga Rakhi", "Rakhi", 300),
            _item("big", "Pink Embroidered Lehenga Choli", "lehenga", 5000),
        ]
    )
    look = compose_outfit(
        pd.DataFrame(), stub, occasion_slug="sangeet", gender="women", budget_inr=800
    )
    assert look["seed_item"] is None and look["complements"] == []
    assert "within" in look["outfit_rationale"] or "No women" in look["outfit_rationale"]
