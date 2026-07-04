"""Unit tests for the src.agents.outfit package — no LLM, no index required."""
from __future__ import annotations

import pytest

from src.agents.outfit.coherence import colour_score, is_coherent_candidate
from src.agents.outfit.composer import (
    FLYWHEEL_ALPHA,
    STORE_DIVERSITY_PENALTY,
    PairingStat,
    _find_best_candidate,
    _flywheel_boost,
)
from src.agents.outfit.occasions import OCCASIONS, get_occasion
from src.agents.outfit.slots import (
    classify_anchor,
    fabric_score_delta,
    get_fill_slots,
    is_ethnic_item,
    is_western_item,
)

# ── occasions ──────────────────────────────────────────────────────────────

class TestGetOccasion:
    def test_known_slug_returns_correct_occasion(self) -> None:
        occ = get_occasion("sangeet")
        assert occ.slug == "sangeet"
        assert occ.formality == 5

    def test_unknown_slug_falls_back_to_casual(self) -> None:
        occ = get_occasion("rave_party")
        assert occ.slug == "casual"

    def test_all_9_occasions_present(self) -> None:
        expected = {
            "casual", "smart_casual", "office", "haldi_mehendi",
            "party_evening", "festive_puja", "wedding_guest", "sangeet",
            "traditional_ethnic",
        }
        assert set(OCCASIONS.keys()) == expected


# ── slots / classify_anchor ────────────────────────────────────────────────

class TestClassifyAnchor:
    @pytest.mark.parametrize("pt,name,expected", [
        ("Kurta", "", "ethnic_top"),
        ("", "Anarkali Gown", "ethnic_one_piece"),
        ("Lehenga", "", "ethnic_one_piece"),
        ("Palazzo", "", "ethnic_bottom"),
        ("Sherwani", "", "men_formalwear"),
        ("Jacket", "", "outerwear"),
        ("Blazer", "", "outerwear"),
        ("Dress", "", "western_one_piece"),
        ("Trousers", "", "western_bottom"),
        ("T-shirt", "", "western_top"),
        ("Shirt", "", "western_top"),
        ("Mojari", "", "footwear"),
        ("", "heels sandals", "footwear"),
        ("Widget", "", "unknown"),
    ])
    def test_classify_anchor_cases(self, pt: str, name: str, expected: str) -> None:
        assert classify_anchor(pt, name) == expected

    def test_ethnic_one_piece_wins_over_ethnic_top(self) -> None:
        # "saree" should beat "kurti" if both appear — ethnic_one_piece listed first
        result = classify_anchor("saree kurti", "")
        assert result == "ethnic_one_piece"


class TestIsEthnicIsWestern:
    def test_kurta_is_ethnic(self) -> None:
        assert is_ethnic_item("Kurta") is True

    def test_sherwani_is_ethnic(self) -> None:
        assert is_ethnic_item("Sherwani") is True

    def test_dress_is_not_ethnic(self) -> None:
        assert is_ethnic_item("Dress") is False

    def test_shirt_is_western(self) -> None:
        assert is_western_item("Shirt") is True

    def test_kurta_is_not_western(self) -> None:
        assert is_western_item("Kurta") is False


class TestGetFillSlots:
    def test_ethnic_top_women_has_bottom_dupatta_footwear(self) -> None:
        slots = get_fill_slots("ethnic_top", "women", "festive_puja")
        names = [s.slot_name for s in slots]
        assert "bottom" in names
        assert "accessory" in names

    def test_ethnic_top_men_has_bottom_no_dupatta(self) -> None:
        slots = get_fill_slots("ethnic_top", "men", "festive_puja")
        names = [s.slot_name for s in slots]
        assert "bottom" in names
        assert "accessory" not in names

    def test_ethnic_one_piece_has_no_top_bottom(self) -> None:
        slots = get_fill_slots("ethnic_one_piece", "women", "sangeet")
        names = [s.slot_name for s in slots]
        assert "top" not in names
        assert "bottom" not in names
        assert "accessory" in names

    def test_western_top_default_slots(self) -> None:
        slots = get_fill_slots("western_top", "women", "casual")
        names = [s.slot_name for s in slots]
        assert "bottom" in names


class TestFabricScoreDelta:
    def test_sangeet_embellished_positive(self) -> None:
        item = {"prod_name": "Heavy Embroidered Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "sangeet") == pytest.approx(0.1)

    def test_sangeet_lightweight_negative(self) -> None:
        item = {"prod_name": "Cotton Floral Kurti", "detail_desc": ""}
        assert fabric_score_delta(item, "sangeet") == pytest.approx(-0.1)

    def test_haldi_lightweight_positive(self) -> None:
        item = {"prod_name": "Floral Cotton Kurti", "detail_desc": ""}
        assert fabric_score_delta(item, "haldi_mehendi") == pytest.approx(0.1)

    def test_haldi_embellished_negative(self) -> None:
        item = {"prod_name": "Heavy Zari Embroidered Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "haldi_mehendi") == pytest.approx(-0.1)

    def test_neutral_occasion_zero(self) -> None:
        item = {"prod_name": "Embroidered Floral Dress", "detail_desc": ""}
        assert fabric_score_delta(item, "party_evening") == pytest.approx(0.0)

    def test_plain_item_zero_delta(self) -> None:
        item = {"prod_name": "Plain Blue Shirt", "detail_desc": ""}
        assert fabric_score_delta(item, "sangeet") == pytest.approx(0.0)


# ── coherence ──────────────────────────────────────────────────────────────

class TestIsCoherentCandidate:
    def _make_item(self, product_type: str, prod_name: str = "", gender: str = "unknown") -> dict:
        return {"product_type": product_type, "prod_name": prod_name, "gender": gender}

    def test_dupatta_rejected_for_men(self) -> None:
        item = self._make_item("Dupatta", "silk dupatta", gender="women")
        assert is_coherent_candidate(item, "sangeet", "men", "accessory") is False

    def test_dupatta_allowed_for_women(self) -> None:
        item = self._make_item("Dupatta", "silk dupatta", gender="women")
        result = is_coherent_candidate(item, "sangeet", "women", "accessory")
        assert result is True

    def test_western_item_rejected_for_ethnic_only(self) -> None:
        item = self._make_item("Dress", "floral dress", gender="women")
        assert is_coherent_candidate(item, "sangeet", "women", "top") is False

    def test_western_formal_allowed_for_men_wedding_guest(self) -> None:
        item = self._make_item("Blazer", "formal blazer", gender="men")
        result = is_coherent_candidate(item, "wedding_guest", "men", "outerwear")
        assert result is True

    def test_western_casual_rejected_for_ethnic_heavy_occasion(self) -> None:
        item = self._make_item("T-shirt", "casual tshirt", gender="women")
        assert is_coherent_candidate(item, "festive_puja", "women", "top") is False

    def test_ethnic_item_always_passes(self) -> None:
        item = self._make_item("Kurta", "festive kurta", gender="men")
        assert is_coherent_candidate(item, "sangeet", "men", "top") is True


class TestColourScore:
    def test_haldi_yellow_scores_1(self) -> None:
        assert colour_score("yellow", "orange", "haldi_mehendi") == pytest.approx(1.0)

    def test_haldi_dark_scores_low(self) -> None:
        assert colour_score("dark grey", "yellow", "haldi_mehendi") == pytest.approx(0.2)

    def test_ethnic_same_colour_high(self) -> None:
        score = colour_score("red", "red", "sangeet")
        assert score >= 0.8

    def test_western_neutral_scores_1(self) -> None:
        assert colour_score("black", "blue", "casual") == pytest.approx(1.0)

    def test_western_mismatch_scores_low(self) -> None:
        assert colour_score("red", "blue", "casual") == pytest.approx(0.4)


# ── flywheel boost ─────────────────────────────────────────────────────────

class TestFlywheelBoost:
    def test_none_stats_returns_zero(self) -> None:
        assert _flywheel_boost("ethnic_top", "bottom", "sangeet", None) == pytest.approx(0.0)

    def test_cold_start_below_min_signals_returns_zero(self) -> None:
        stats = {("ethnic_top", "bottom", "sangeet"): PairingStat(add_the_look=5, thumbs_up=2)}
        result = _flywheel_boost("ethnic_top", "bottom", "sangeet", stats)
        assert result == pytest.approx(0.0)

    def test_warm_start_returns_positive_boost(self) -> None:
        # 8 positive out of 10 total → positive_rate = 0.8 → boost = 0.25 * 0.8 = 0.2
        stats = {
            ("ethnic_top", "bottom", "sangeet"): PairingStat(
                add_the_look=8, thumbs_up=0, thumbs_down=2, add_single_only=0
            )
        }
        result = _flywheel_boost("ethnic_top", "bottom", "sangeet", stats)
        assert result == pytest.approx(FLYWHEEL_ALPHA * 0.8)

    def test_missing_key_returns_zero(self) -> None:
        stats = {("ethnic_top", "footwear", "sangeet"): PairingStat(add_the_look=10, thumbs_up=5)}
        result = _flywheel_boost("ethnic_top", "bottom", "sangeet", stats)
        assert result == pytest.approx(0.0)


# ── store diversity preference (cross-store styling, Phase F / G4 fix) ──────

class _FakeRetriever:
    """Minimal retriever stub returning a fixed candidate list, ignoring the query."""

    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None
    ) -> list[dict]:
        return list(self._items)


def _make_candidate(
    article_id: str,
    store: str,
    score: float,
    colour: str = "black",
    price_inr: float = 500.0,
) -> dict:
    """Build a minimal candidate item dict matching the hybrid_search output shape."""
    return {
        "article_id": article_id,
        "prod_name": "Black Trousers",
        "display_name": "Black Trousers",
        "store": store,
        "colour": colour,
        "product_type": "Trousers",
        "detail_desc": "",
        "score": score,
        "price_inr": price_inr,
        "gender": "women",
    }


class TestFindBestCandidateStoreDiversity:
    """A soft store-diversity preference should break near-ties toward a new store,
    but never override a candidate that is clearly better on merit (colour/base score).
    """

    _common_kwargs = {
        "query": "trousers",
        "slot_name": "bottom",
        "occasion_slug": "casual",
        "gender": "women",
        "anchor_colour": "black",
        "seen_ids": set(),
        "seen_prod_colour": set(),
        "budget_remaining": None,
        "pairing_stats": None,
        "anchor_class": "western_top",
    }

    def test_near_equal_scores_prefer_new_store(self) -> None:
        """Seed is from store A; two near-equal complement candidates from A and B.
        B (the unrepresented store) must win.
        """
        candidate_a = _make_candidate("A1", "storea", score=0.90)
        candidate_b = _make_candidate("B1", "storeb", score=0.80)
        retriever = _FakeRetriever([candidate_a, candidate_b])

        winner = _find_best_candidate(
            **self._common_kwargs,
            retriever=retriever,
            seen_stores={"storea"},
        )

        assert winner is not None
        assert winner["article_id"] == "B1", "near-tied candidate from a new store should win"

    def test_clearly_better_same_store_still_wins(self) -> None:
        """When the same-store candidate is clearly better (not just a near-tie), it
        must still win — the diversity preference is soft, not a hard filter.
        """
        candidate_a = _make_candidate("A1", "storea", score=0.99)
        candidate_b = _make_candidate("B1", "storeb", score=0.30)
        retriever = _FakeRetriever([candidate_a, candidate_b])

        winner = _find_best_candidate(
            **self._common_kwargs,
            retriever=retriever,
            seen_stores={"storea"},
        )

        assert winner is not None
        assert winner["article_id"] == "A1", "clearly-better same-store candidate must still win"

    def test_penalty_constant_is_soft_not_zero(self) -> None:
        """Sanity-check the constant itself: it must discount, not exclude (0 < p < 1)."""
        assert 0.0 < STORE_DIVERSITY_PENALTY < 1.0

    def test_no_seen_stores_falls_back_to_plain_score_order(self) -> None:
        """When seen_stores is empty/None, ranking is unaffected — the higher raw
        score wins regardless of store.
        """
        candidate_a = _make_candidate("A1", "storea", score=0.90)
        candidate_b = _make_candidate("B1", "storeb", score=0.80)
        retriever = _FakeRetriever([candidate_a, candidate_b])

        winner = _find_best_candidate(
            **self._common_kwargs,
            retriever=retriever,
            seen_stores=set(),
        )

        assert winner is not None
        assert winner["article_id"] == "A1"
