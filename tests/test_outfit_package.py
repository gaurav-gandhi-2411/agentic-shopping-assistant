"""Unit tests for the src.agents.outfit package — no LLM, no index required."""
from __future__ import annotations

import pandas as pd
import pytest

from src.agents.outfit.coherence import colour_score, is_coherent_candidate
from src.agents.outfit.composer import (
    FLYWHEEL_ALPHA,
    STORE_DIVERSITY_PENALTY,
    PairingStat,
    _find_best_candidate,
    _flywheel_boost,
    compose_outfit,
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

    def test_haldi_mehendi_alias_resolves_to_haldi(self) -> None:
        """Legacy combined slug (pre wedding-occasion-expansion) must still
        resolve to a real Occasion rather than falling back to casual."""
        occ = get_occasion("haldi_mehendi")
        assert occ.slug == "haldi"

    def test_all_12_occasions_present(self) -> None:
        expected = {
            "casual", "smart_casual", "office", "haldi", "mehendi",
            "party_evening", "festive_puja", "wedding_guest", "engagement",
            "sangeet", "traditional_ethnic", "reception",
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

    def test_western_top_men_has_optional_footwear_and_accessory(self) -> None:
        slots = get_fill_slots("western_top", "men", "casual")
        by_name = {s.slot_name: s for s in slots}
        assert "footwear" in by_name
        assert "accessory" in by_name
        assert by_name["footwear"].required is False
        assert by_name["accessory"].required is False
        assert "men" in by_name["footwear"].search_query
        assert "men" in by_name["accessory"].search_query
        # bottom (required) still precedes footwear/accessory in greedy fill order
        names = [s.slot_name for s in slots]
        assert names.index("bottom") < names.index("footwear") < names.index("accessory")
        assert by_name["bottom"].required is True

    def test_western_top_women_has_optional_footwear_and_accessory(self) -> None:
        slots = get_fill_slots("western_top", "women", "casual")
        by_name = {s.slot_name: s for s in slots}
        assert "footwear" in by_name
        assert "accessory" in by_name
        assert by_name["footwear"].required is False
        assert by_name["accessory"].required is False
        assert "women" in by_name["footwear"].search_query
        assert "women" in by_name["accessory"].search_query

    def test_western_bottom_has_optional_footwear(self) -> None:
        slots = get_fill_slots("western_bottom", "women", "casual")
        by_name = {s.slot_name: s for s in slots}
        assert "top" in by_name
        assert by_name["top"].required is True
        assert "footwear" in by_name
        assert by_name["footwear"].required is False
        # order: top -> outerwear -> footwear
        names = [s.slot_name for s in slots]
        assert names.index("outerwear") < names.index("footwear")

    def test_western_bottom_men_footwear_query_is_gendered(self) -> None:
        slots = get_fill_slots("western_bottom", "men", "casual")
        by_name = {s.slot_name: s for s in slots}
        assert "men" in by_name["footwear"].search_query

    def test_unknown_anchor_matches_western_top_default(self) -> None:
        unknown_slots = get_fill_slots("unknown", "women", "casual")
        western_top_slots = get_fill_slots("western_top", "women", "casual")
        assert [s.slot_name for s in unknown_slots] == [s.slot_name for s in western_top_slots]
        assert [s.required for s in unknown_slots] == [s.required for s in western_top_slots]


class TestFabricScoreDelta:
    def test_sangeet_embellished_positive(self) -> None:
        item = {"prod_name": "Heavy Embroidered Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "sangeet") == pytest.approx(0.1)

    def test_sangeet_lightweight_negative(self) -> None:
        item = {"prod_name": "Cotton Floral Kurti", "detail_desc": ""}
        assert fabric_score_delta(item, "sangeet") == pytest.approx(-0.1)

    def test_haldi_lightweight_positive(self) -> None:
        item = {"prod_name": "Floral Cotton Kurti", "detail_desc": ""}
        assert fabric_score_delta(item, "haldi") == pytest.approx(0.1)

    def test_haldi_embellished_negative(self) -> None:
        item = {"prod_name": "Heavy Zari Embroidered Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "haldi") == pytest.approx(-0.1)

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
        assert colour_score("yellow", "orange", "haldi") == pytest.approx(1.0)

    def test_haldi_dark_scores_low(self) -> None:
        assert colour_score("dark grey", "yellow", "haldi") == pytest.approx(0.2)

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


# ── complement _role stamping (RED 1a/1e/B4a/B4b) ────────────────────────────

class _FillSlotFakeRetriever:
    """Returns different candidates depending on the slot query so both the
    required "bottom" slot and the optional "outerwear" slot for a western_top
    anchor get filled.
    """

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None
    ) -> list[dict]:
        if "trousers" in query:
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
                    "gender": "women",
                }
            ]
        if "jacket" in query:
            return [
                {
                    "article_id": "C2",
                    "prod_name": "Denim Jacket",
                    "display_name": "Denim Jacket",
                    "store": "myntra",
                    "colour": "blue",
                    "product_type": "Jacket",
                    "detail_desc": "",
                    "score": 0.85,
                    "price_inr": 1499.0,
                    "gender": "women",
                }
            ]
        return []


def _make_seed_catalogue_row(article_id: str) -> pd.DataFrame:
    """Single-row catalogue DataFrame for a western-top seed item."""
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


class TestComposeOutfitComplementRoleStamping:
    """Every complement in a composed look must carry _role='complement' so
    ItemSummary.from_agent_item (api/schemas.py) can populate slot_role and the
    frontend OutfitBoard renders every card, not just the seed.
    """

    def test_all_complements_get_role_complement(self) -> None:
        catalogue_df = _make_seed_catalogue_row("SEED1")
        retriever = _FillSlotFakeRetriever()
        look = compose_outfit(
            catalogue_df,
            retriever,
            seed_article_id="SEED1",
            occasion_slug="casual",
            gender="women",
        )
        assert look["complements"], "expected at least one complement to be filled"
        for complement in look["complements"]:
            assert complement.get("_role") == "complement", (
                f"complement {complement.get('article_id')} missing _role='complement'"
            )

    def test_seed_item_role_is_seed(self) -> None:
        catalogue_df = _make_seed_catalogue_row("SEED2")
        retriever = _FillSlotFakeRetriever()
        look = compose_outfit(
            catalogue_df,
            retriever,
            seed_article_id="SEED2",
            occasion_slug="casual",
            gender="women",
        )
        assert look["seed_item"]["_role"] == "seed"

    def test_item_summary_round_trip_sets_slot_role(self) -> None:
        """End-to-end: compose_outfit complement -> ItemSummary.from_agent_item
        must yield a non-null slot_role of 'complement'.
        """
        from api.schemas import ItemSummary

        catalogue_df = _make_seed_catalogue_row("SEED3")
        retriever = _FillSlotFakeRetriever()
        look = compose_outfit(
            catalogue_df,
            retriever,
            seed_article_id="SEED3",
            occasion_slug="casual",
            gender="women",
        )
        assert look["complements"]
        for complement in look["complements"]:
            summary = ItemSummary.from_agent_item(complement)
            assert summary.slot_role == "complement"
