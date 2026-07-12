"""Phase B Part 2 — cross-gender PARTNER styling.

Covers:
  1. src.agents.outfit.partner.detect_partner_intent — positive/negative intent
     detection (relationship words, "his and hers", "for him"/"for her" gated
     on a styling verb, and ambiguous-phrasing negatives).
  2. src.agents.outfit.partner.resolve_partner_gender — gender resolution.
  3. src.agents.outfit.coherence.couple_harmony_palette — couple-harmony colour map.
  4. src.agents.outfit.partner.build_coordinated_with_text — deterministic,
     grounded-by-construction board text.
  5. src.agents.outfit.partner.compose_partner_look — isolated unit test with a
     fake retriever (no real index).
  6. src.agents.grounding.validate_rationale(extra_whitelist_tokens=...) —
     partner-context tokens survive the grounding gate.
  7. api/schemas.py payload contract — ItemSummary.gender, ChatResponse
     look_role/look_title/coordinated_with, via a mocked agent (no real index).
  8. Offline real-index composition evidence (requires_index):
     (a) women's black-dress anchor → "what should my husband wear with this"
         → men's coordinated look.
     (b) men's kurta sangeet anchor → "style my wife to match" → women's
         ethnic look, no western-marker items.
     (c) "style us as a couple for a reception" with NO prior anchor — P2
         couple-from-scratch — both looks compose.
     (d) same, with a budget cap — P2 per-person budget-split assumption.
  9. src.agents.outfit.partner.compose_partner_look — P2 budget gate (in-budget
     seed preferred over an over-budget one; all-over-budget → honest empty
     result mentioning the budget).
  10. src.agents.outfit.partner.compose_couple_look — P2 from-scratch couple
      orchestration (isolated unit tests with a fake retriever).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest

from src.agents.grounding import validate_rationale
from src.agents.outfit.coherence import couple_harmony_palette
from src.agents.outfit.partner import (
    build_coordinated_with_text,
    compose_couple_look,
    compose_partner_look,
    detect_partner_intent,
    resolve_partner_gender,
)

UNIFIED_DIR = Path("data/processed/unified")


# ---------------------------------------------------------------------------
# 1. detect_partner_intent
# ---------------------------------------------------------------------------


class TestDetectPartnerIntentPositives:
    def test_husband_triggers_men(self) -> None:
        result = detect_partner_intent("what should my husband wear with this?")
        assert result.matched is True
        assert result.gender_hint == "men"

    def test_wife_triggers_women(self) -> None:
        result = detect_partner_intent("style my wife to match")
        assert result.matched is True
        assert result.gender_hint == "women"

    def test_boyfriend_triggers_men(self) -> None:
        result = detect_partner_intent("what would look good on my boyfriend?")
        assert result.matched is True
        assert result.gender_hint == "men"

    def test_girlfriend_triggers_women(self) -> None:
        result = detect_partner_intent("coordinate an outfit for my girlfriend")
        assert result.matched is True
        assert result.gender_hint == "women"

    def test_groom_triggers_men(self) -> None:
        result = detect_partner_intent("the groom needs a matching look")
        assert result.matched is True
        assert result.gender_hint == "men"

    def test_bride_triggers_women(self) -> None:
        result = detect_partner_intent("the bride needs a matching look")
        assert result.matched is True
        assert result.gender_hint == "women"

    def test_his_and_hers_triggers_opposite(self) -> None:
        result = detect_partner_intent("his and hers outfit for a wedding")
        assert result.matched is True
        assert result.gender_hint == "opposite"

    def test_couple_triggers_opposite(self) -> None:
        result = detect_partner_intent("couple outfit for a wedding")
        assert result.matched is True
        assert result.gender_hint == "opposite"

    def test_partner_word_triggers_opposite(self) -> None:
        result = detect_partner_intent("can you style my partner as well")
        assert result.matched is True
        assert result.gender_hint == "opposite"

    def test_fiance_triggers_opposite(self) -> None:
        result = detect_partner_intent("what should my fiance wear to match")
        assert result.matched is True
        assert result.gender_hint == "opposite"

    def test_for_him_with_styling_verb_triggers_men(self) -> None:
        result = detect_partner_intent("what should I style for him to match this?")
        assert result.matched is True
        assert result.gender_hint == "men"

    def test_for_her_with_styling_verb_triggers_women(self) -> None:
        result = detect_partner_intent("can you coordinate an outfit for her")
        assert result.matched is True
        assert result.gender_hint == "women"


class TestDetectPartnerIntentNegatives:
    def test_ambiguous_also_show_me_shirts_does_not_trigger(self) -> None:
        result = detect_partner_intent("also show me shirts")
        assert result.matched is False
        assert result.gender_hint is None

    def test_womens_shirts_for_me_too_does_not_trigger(self) -> None:
        result = detect_partner_intent("women's shirts for me too")
        assert result.matched is False

    def test_for_him_without_styling_verb_does_not_trigger(self) -> None:
        """A plain gendered product mention ("buy a gift for him") must not be
        mistaken for a partner-styling request — no styling/coordination verb
        present."""
        result = detect_partner_intent("buy a gift for him")
        assert result.matched is False

    def test_for_her_without_styling_verb_does_not_trigger(self) -> None:
        result = detect_partner_intent("show me shirts for her")
        assert result.matched is False

    def test_plain_gendered_search_does_not_trigger(self) -> None:
        result = detect_partner_intent("show me men's trousers")
        assert result.matched is False

    def test_bare_him_her_without_for_does_not_trigger(self) -> None:
        result = detect_partner_intent("does this suit him")
        assert result.matched is False

    def test_empty_query_does_not_trigger(self) -> None:
        result = detect_partner_intent("")
        assert result.matched is False


# ---------------------------------------------------------------------------
# 2. resolve_partner_gender
# ---------------------------------------------------------------------------


class TestResolvePartnerGender:
    def test_concrete_men_hint_wins(self) -> None:
        assert resolve_partner_gender("men", "women") == "men"

    def test_concrete_women_hint_wins(self) -> None:
        assert resolve_partner_gender("women", "men") == "women"

    def test_opposite_of_women_anchor_is_men(self) -> None:
        assert resolve_partner_gender("opposite", "women") == "men"

    def test_opposite_of_men_anchor_is_women(self) -> None:
        assert resolve_partner_gender("opposite", "men") == "women"


# ---------------------------------------------------------------------------
# 3. couple_harmony_palette
# ---------------------------------------------------------------------------


class TestCoupleHarmonyPalette:
    def test_rust_maps_to_navy_cream_olive(self) -> None:
        palette = couple_harmony_palette("rust")
        assert palette == ("navy blue", "cream", "olive")

    def test_black_maps_to_burgundy_grey_white(self) -> None:
        palette = couple_harmony_palette("black")
        assert palette == ("burgundy", "grey", "white")

    def test_palette_never_contains_the_anchor_colour_itself(self) -> None:
        for anchor_colour in ("rust", "black", "navy blue", "mustard", "teal", "beige"):
            palette = couple_harmony_palette(anchor_colour)
            assert anchor_colour not in palette, (
                f"{anchor_colour} palette must not echo the anchor colour: {palette}"
            )

    def test_unknown_colour_falls_back_to_default_neutral_palette(self) -> None:
        palette = couple_harmony_palette("some-never-seen-colour")
        assert palette == ("navy blue", "grey", "charcoal")

    def test_case_insensitive_lookup(self) -> None:
        assert couple_harmony_palette("RUST") == couple_harmony_palette("rust")


# ---------------------------------------------------------------------------
# 4. build_coordinated_with_text
# ---------------------------------------------------------------------------


class TestBuildCoordinatedWithText:
    def test_text_references_anchor_and_partner_colours(self) -> None:
        anchor_item = {"colour": "Rust", "product_type": "Dress"}
        partner_look = {
            "seed_item": {"colour": "Navy Blue"},
            "complements": [{"colour": "Cream"}],
        }
        text = build_coordinated_with_text(anchor_item, partner_look, "smart_casual")
        assert "rust" in text.lower()
        assert "dress" in text.lower()
        assert "navy blue" in text.lower()
        assert "cream" in text.lower()
        assert "smart-casual" in text.lower()

    def test_no_complement_colours_falls_back_to_generic_phrase(self) -> None:
        anchor_item = {"colour": "black", "product_type": "shirt"}
        partner_look = {"seed_item": {}, "complements": []}
        text = build_coordinated_with_text(anchor_item, partner_look, "casual")
        assert "complementary palette" in text.lower()

    def test_deduplicates_repeated_colours(self) -> None:
        anchor_item = {"colour": "black", "product_type": "dress"}
        partner_look = {
            "seed_item": {"colour": "grey"},
            "complements": [{"colour": "grey"}, {"colour": "grey"}],
        }
        text = build_coordinated_with_text(anchor_item, partner_look, "office")
        assert text.lower().count("grey") == 1


# ---------------------------------------------------------------------------
# 5. compose_partner_look — isolated unit test with a fake retriever
# ---------------------------------------------------------------------------


class _PartnerFakeRetriever:
    """Returns a men's shirt seed candidate (one colour in the harmony palette,
    one not) for the seed query, a bottom candidate for fill slots, and nothing
    for footwear/accessory/outerwear (models a thin catalogue slice — also
    exercises honest suppression, requirement 6)."""

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None  # noqa: ARG002
    ) -> list[dict]:
        q = query.lower()
        gender = (filters or {}).get("gender")
        if "shirt" in q and gender == "men":
            return [
                {
                    "article_id": "PARTNER_SEED_OFFPALETTE",
                    "prod_name": "Red Casual Shirt",
                    "display_name": "Red Casual Shirt",
                    "product_type": "shirt",
                    "colour": "red",
                    "gender": "men",
                    "score": 0.95,
                    "price_inr": 799.0,
                    "store": "myntra",
                    "detail_desc": "",
                },
                {
                    "article_id": "PARTNER_SEED_ONPALETTE",
                    "prod_name": "Navy Blue Casual Shirt",
                    "display_name": "Navy Blue Casual Shirt",
                    "product_type": "shirt",
                    "colour": "navy blue",
                    "gender": "men",
                    "score": 0.8,
                    "price_inr": 899.0,
                    "store": "myntra",
                    "detail_desc": "",
                },
            ]
        if any(w in q for w in ("trousers", "jeans", "skirt")) and gender == "men":
            return [
                {
                    "article_id": "PARTNER_BOTTOM1",
                    "prod_name": "Charcoal Trousers",
                    "display_name": "Charcoal Trousers",
                    "product_type": "trousers",
                    "colour": "charcoal",
                    "gender": "men",
                    "score": 0.9,
                    "price_inr": 1299.0,
                    "store": "myntra",
                    "detail_desc": "",
                }
            ]
        return []


def _partner_seed_catalogue_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_id": "PARTNER_SEED_ONPALETTE",
                "prod_name": "Navy Blue Casual Shirt",
                "display_name": "Navy Blue Casual Shirt",
                "colour_group_name": "navy blue",
                "product_type_name": "shirt",
                "department_name": "Men",
                "index_group_name": "Menswear",
                "detail_desc": "",
                "image_url": None,
                "price_inr": 899.0,
                "pdp_handle": "navy-shirt",
                "store": "myntra",
                "gender": "men",
                "facets": {
                    "colour_group_name": "navy blue",
                    "product_type_name": "shirt",
                    "department_name": "Men",
                },
            },
            {
                "article_id": "PARTNER_SEED_OFFPALETTE",
                "prod_name": "Red Casual Shirt",
                "display_name": "Red Casual Shirt",
                "colour_group_name": "red",
                "product_type_name": "shirt",
                "department_name": "Men",
                "index_group_name": "Menswear",
                "detail_desc": "",
                "image_url": None,
                "price_inr": 799.0,
                "pdp_handle": "red-shirt",
                "store": "myntra",
                "gender": "men",
                "facets": {
                    "colour_group_name": "red",
                    "product_type_name": "shirt",
                    "department_name": "Men",
                },
            },
        ]
    )


class TestComposePartnerLook:
    def test_prefers_harmony_palette_seed_colour(self) -> None:
        anchor_item = {"colour": "rust", "product_type": "dress", "gender": "women"}
        look = compose_partner_look(
            _partner_seed_catalogue_df(),
            _PartnerFakeRetriever(),
            anchor_item=anchor_item,
            occasion_slug="casual",
            partner_gender="men",
        )
        assert look["seed_item"] is not None
        # Navy blue IS in couple_harmony_palette("rust") — must be preferred
        # over the higher-scored off-palette red candidate.
        assert look["seed_item"]["colour"].lower() == "navy blue"

    def test_all_items_are_partner_gender(self) -> None:
        anchor_item = {"colour": "rust", "product_type": "dress", "gender": "women"}
        look = compose_partner_look(
            _partner_seed_catalogue_df(),
            _PartnerFakeRetriever(),
            anchor_item=anchor_item,
            occasion_slug="casual",
            partner_gender="men",
        )
        assert look["gender"] == "men"
        for c in look["complements"]:
            assert c["gender"] == "men"

    def test_no_candidate_returns_none_seed_with_honest_reason(self) -> None:
        class _EmptyRetriever:
            def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:  # noqa: ARG002
                return []

        anchor_item = {"colour": "rust", "product_type": "dress", "gender": "women"}
        look = compose_partner_look(
            _partner_seed_catalogue_df(),
            _EmptyRetriever(),
            anchor_item=anchor_item,
            occasion_slug="casual",
            partner_gender="men",
        )
        assert look["seed_item"] is None
        assert look["outfit_rationale"]


# ---------------------------------------------------------------------------
# 5b. compose_partner_look — P2 budget gate (mirrors
#     TestOccasionDrivenAnchorBudgetGate in tests/test_outfit_package.py)
# ---------------------------------------------------------------------------


class _PartnerBudgetFakeRetriever:
    """Returns a fixed, budget-relevant seed candidate list (rank 0 first) for
    every men's-shirt query — grey (off-palette for a rust anchor) so
    colour_rank never reorders the list, isolating the test to the budget
    gate only."""

    def __init__(self, candidates: list[dict]) -> None:
        self._candidates = candidates

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None  # noqa: ARG002
    ) -> list[dict]:
        gender = (filters or {}).get("gender")
        if "shirt" in query.lower() and gender == "men":
            return list(self._candidates)
        return []


_PARTNER_OVER_BUDGET_SHIRT: dict = {
    "article_id": "PARTNER_OVER",
    "prod_name": "Grey Casual Shirt",
    "display_name": "Grey Casual Shirt",
    "store": "myntra",
    "colour": "grey",
    "product_type": "shirt",
    "detail_desc": "",
    "score": 0.95,
    "price_inr": 4500.0,
    "gender": "men",
}

_PARTNER_WITHIN_BUDGET_SHIRT: dict = {
    "article_id": "PARTNER_OK",
    "prod_name": "Grey Cotton Shirt",
    "display_name": "Grey Cotton Shirt",
    "store": "myntra",
    "colour": "grey",
    "product_type": "shirt",
    "detail_desc": "",
    "score": 0.7,
    "price_inr": 1200.0,
    "gender": "men",
}


def _partner_budget_catalogue_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_id": "PARTNER_OK",
                "prod_name": "Grey Cotton Shirt",
                "display_name": "Grey Cotton Shirt",
                "colour_group_name": "grey",
                "product_type_name": "shirt",
                "department_name": "Men",
                "index_group_name": "Menswear",
                "detail_desc": "",
                "image_url": None,
                "price_inr": 1200.0,
                "pdp_handle": "grey-shirt-ok",
                "store": "myntra",
                "gender": "men",
                "facets": {
                    "colour_group_name": "grey",
                    "product_type_name": "shirt",
                    "department_name": "Men",
                },
            }
        ]
    )


class TestComposePartnerLookBudgetGate:
    def test_over_budget_rank0_seed_skipped_for_within_budget_rank1(self) -> None:
        """(a) rank-0 seed candidate is over budget, rank-1 is within — the
        chosen seed must be the within-budget one and the board total must
        not exceed the budget."""
        anchor_item = {"colour": "rust", "product_type": "dress", "gender": "women"}
        retriever = _PartnerBudgetFakeRetriever(
            [_PARTNER_OVER_BUDGET_SHIRT, _PARTNER_WITHIN_BUDGET_SHIRT]
        )
        look = compose_partner_look(
            _partner_budget_catalogue_df(),
            retriever,
            anchor_item=anchor_item,
            occasion_slug="casual",
            partner_gender="men",
            budget_inr=2000,
        )
        assert look["seed_item"] is not None
        assert look["seed_item"]["article_id"] == "PARTNER_OK"
        assert (look["budget_total_inr"] or 0) <= 2000

    def test_all_seed_candidates_over_budget_returns_honest_empty_result(self) -> None:
        """(b) every occasion/gender-valid seed candidate is over budget — must
        return an empty result (no seed) whose message mentions the budget,
        never a silent fall-back to an over-budget seed."""
        anchor_item = {"colour": "rust", "product_type": "dress", "gender": "women"}
        retriever = _PartnerBudgetFakeRetriever([_PARTNER_OVER_BUDGET_SHIRT])
        look = compose_partner_look(
            _partner_budget_catalogue_df(),
            retriever,
            anchor_item=anchor_item,
            occasion_slug="casual",
            partner_gender="men",
            budget_inr=2000,
        )
        assert look["seed_item"] is None
        assert "2,000" in look["outfit_rationale"]
        assert "budget" in look["outfit_rationale"].lower() or "₹" in look["outfit_rationale"]


# ---------------------------------------------------------------------------
# 10. compose_couple_look — P2 from-scratch couple orchestration
# ---------------------------------------------------------------------------


class _CoupleFakeRetriever:
    """Deterministic fake retriever for compose_couple_look unit tests.

    Returns a women's reception anchor (ethnic_one_piece "Lehenga", satisfies
    the ETHNIC_HEAVY anchor gate) for the primary occasion-driven anchor
    query, and a men's reception seed (men_formalwear "Sherwani", on the
    couple_harmony_palette("rust") palette — navy blue) for
    compose_partner_look's seed query. Everything else (complement fill
    slots) returns [] — honest suppression, not under test here.
    """

    def __init__(self, *, primary_price: float, partner_price: float) -> None:
        self._primary_price = primary_price
        self._partner_price = partner_price

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None
    ) -> list[dict]:
        q = query.lower()
        gender = (filters or {}).get("gender")
        if "lehenga" in q and gender == "women":
            return [
                {
                    "article_id": "COUPLE_PRIMARY_LEHENGA",
                    "prod_name": "Rust Embellished Lehenga",
                    "display_name": "Rust Embellished Lehenga",
                    "product_type": "Lehenga",
                    "colour": "rust",
                    "gender": "women",
                    "score": 0.9,
                    "price_inr": self._primary_price,
                    "store": "myntra",
                    "detail_desc": "",
                }
            ]
        if "sherwani" in q and gender == "men":
            return [
                {
                    "article_id": "COUPLE_PARTNER_SHERWANI",
                    "prod_name": "Navy Blue Sherwani",
                    "display_name": "Navy Blue Sherwani",
                    "product_type": "Sherwani",
                    "colour": "navy blue",
                    "gender": "men",
                    "score": 0.85,
                    "price_inr": self._partner_price,
                    "store": "myntra",
                    "detail_desc": "",
                }
            ]
        return []


def _couple_catalogue_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_id": "COUPLE_PRIMARY_LEHENGA",
                "prod_name": "Rust Embellished Lehenga",
                "display_name": "Rust Embellished Lehenga",
                "colour_group_name": "rust",
                "product_type_name": "Lehenga",
                "department_name": "Women",
                "index_group_name": "Ladieswear",
                "detail_desc": "",
                "image_url": None,
                "price_inr": 11000.0,
                "pdp_handle": "rust-lehenga",
                "store": "myntra",
                "gender": "women",
                "facets": {
                    "colour_group_name": "rust",
                    "product_type_name": "Lehenga",
                    "department_name": "Women",
                },
            },
            {
                "article_id": "COUPLE_PARTNER_SHERWANI",
                "prod_name": "Navy Blue Sherwani",
                "display_name": "Navy Blue Sherwani",
                "colour_group_name": "navy blue",
                "product_type_name": "Sherwani",
                "department_name": "Men",
                "index_group_name": "Menswear",
                "detail_desc": "",
                "image_url": None,
                "price_inr": 9000.0,
                "pdp_handle": "navy-sherwani",
                "store": "myntra",
                "gender": "men",
                "facets": {
                    "colour_group_name": "navy blue",
                    "product_type_name": "Sherwani",
                    "department_name": "Men",
                },
            },
        ]
    )


class TestComposeCoupleLook:
    def test_produces_opposite_gender_pair_with_harmony_palette_colour(self) -> None:
        retriever = _CoupleFakeRetriever(primary_price=11000.0, partner_price=9000.0)
        primary_look, partner_look = compose_couple_look(
            _couple_catalogue_df(),
            retriever,
            occasion_slug="reception",
            partner_gender="men",
        )
        assert primary_look["seed_item"] is not None
        assert partner_look["seed_item"] is not None
        assert primary_look["gender"] == "women"
        assert partner_look["gender"] == "men"
        # Navy blue IS in couple_harmony_palette("rust") — the primary look's
        # own anchor colour.
        palette = couple_harmony_palette(primary_look["seed_item"]["colour"])
        assert partner_look["seed_item"]["colour"].lower() in palette
        assert primary_look["occasion"] == "reception"
        assert partner_look["occasion"] == "reception"

    def test_budget_is_a_per_person_cap_not_a_combined_split(self) -> None:
        """Documented assumption: budget_inr is applied INDEPENDENTLY to EACH
        look, not split in half across the couple. Both looks individually
        fit under 15000, but their COMBINED total (20000) would exceed it —
        proving the cap is per-person, not a 50/50 combined split (which
        would reject at least one of these prices)."""
        retriever = _CoupleFakeRetriever(primary_price=11000.0, partner_price=9000.0)
        primary_look, partner_look = compose_couple_look(
            _couple_catalogue_df(),
            retriever,
            occasion_slug="reception",
            partner_gender="men",
            budget_inr=15000,
        )
        assert primary_look["seed_item"] is not None
        assert partner_look["seed_item"] is not None
        assert (primary_look["budget_total_inr"] or 0) <= 15000
        assert (partner_look["budget_total_inr"] or 0) <= 15000
        combined = (primary_look["budget_total_inr"] or 0) + (
            partner_look["budget_total_inr"] or 0
        )
        assert combined > 15000, "test fixture must exercise the per-person, not combined, cap"

    def test_primary_look_missing_returns_honest_empty_partner_too(self) -> None:
        """If NOTHING satisfies the primary occasion-driven anchor query, the
        partner look must be an honest empty result too — never composed
        against a look that doesn't exist."""

        class _EmptyRetriever:
            def search(
                self, query: str, top_k: int = 20, filters: dict | None = None
            ) -> list[dict]:
                return []

        primary_look, partner_look = compose_couple_look(
            pd.DataFrame(),
            _EmptyRetriever(),
            occasion_slug="reception",
            partner_gender="men",
        )
        assert primary_look["seed_item"] is None
        assert partner_look["seed_item"] is None
        assert partner_look["outfit_rationale"]


# ---------------------------------------------------------------------------
# 6. validate_rationale(extra_whitelist_tokens=...)
# ---------------------------------------------------------------------------


class TestValidateRationaleExtraWhitelistTokens:
    def test_anchor_colour_survives_with_extra_whitelist(self) -> None:
        text = "This navy blue shirt coordinates with your partner's red dress."
        look_items = [
            {"colour": "navy blue", "product_type": "shirt", "_slot": "seed"},
        ]
        cleaned, flags = validate_rationale(
            text, look_items, "casual", extra_whitelist_tokens={"red", "dress"}
        )
        assert "red" in cleaned.lower()
        assert not any(f.startswith("rationale:ungrounded") for f in flags)

    def test_without_extra_whitelist_anchor_colour_is_dropped(self) -> None:
        """Regression guard: the SAME sentence, without the extra whitelist,
        must still be FLAGGED as ungrounded — proves the extra param is what
        makes the difference, not a general loosening of the grounding gate.
        (validate_rationale's single-sentence-all-dropped fallback returns the
        UNMODIFIED text alongside the flag — see its docstring — so the
        assertion is on flags, not on the returned text content.)"""
        text = "This navy blue shirt coordinates with your partner's red dress."
        look_items = [
            {"colour": "navy blue", "product_type": "shirt", "_slot": "seed"},
        ]
        _cleaned, flags = validate_rationale(text, look_items, "casual")
        assert any(f.startswith("rationale:ungrounded_colour:red") for f in flags)
        assert "rationale:all_dropped" in flags


# ---------------------------------------------------------------------------
# 7. api/schemas.py payload contract (no real index needed)
# ---------------------------------------------------------------------------


class TestItemSummaryGenderField:
    def test_gender_populated_from_agent_item(self) -> None:
        from api.schemas import ItemSummary

        item = {
            "article_id": "A1",
            "prod_name": "Navy Shirt",
            "display_name": "Navy Shirt",
            "colour": "navy blue",
            "product_type": "shirt",
            "department": "Men",
            "gender": "men",
        }
        summary = ItemSummary.from_agent_item(item)
        assert summary.gender == "men"

    def test_missing_gender_defaults_to_unknown(self) -> None:
        from api.schemas import ItemSummary

        item = {
            "article_id": "A2",
            "prod_name": "Mystery Item",
            "display_name": "Mystery Item",
            "colour": "black",
            "product_type": "top",
            "department": "Women",
        }
        summary = ItemSummary.from_agent_item(item)
        assert summary.gender == "unknown"


class TestChatResponsePartnerFields:
    """End-to-end payload-contract check via a mocked agent (mirrors
    tests/test_api_chat.py's _MockAgent pattern) — proves ChatResponse actually
    surfaces look_role/look_title/coordinated_with without needing the real
    index or a live LLM."""

    def test_partner_board_fields_flow_through_post_chat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi.testclient import TestClient

        import api.deps as deps
        from api.main import app
        from api.session import InMemorySessionStore

        class _MockLLM:
            def generate(self, prompt: str, system: str = None, **kwargs) -> str:
                return "ok"

            def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
                yield "ok"

            def chat(self, messages: list[dict], **kwargs) -> str:
                return "ok"

            def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
                yield "ok"

        class _MockAgent:
            def __init__(self, result: dict[str, Any]) -> None:
                self._result = result

            def invoke(self, state: dict, **kwargs) -> dict[str, Any]:
                result = dict(self._result)
                result.setdefault("messages", state.get("messages", []))
                return result

        partner_result = {
            "retrieved_items": [
                {
                    "article_id": "MEN1",
                    "prod_name": "Navy Blue Shirt",
                    "display_name": "Navy Blue Shirt",
                    "colour": "navy blue",
                    "product_type": "shirt",
                    "department": "Men",
                    "image_url": "https://example.com/m1.jpg",
                    "gender": "men",
                    "_role": "seed",
                }
            ],
            "filters": {},
            "tool_calls": [{"router_decision": {"action": "outfit"}}],
            "final_answer": "**Your partner's look**",
            "iteration": 1,
            "new_items_this_turn": True,
            "out_of_catalogue": False,
            "excluded_colours": None,
            "look_id": "abc-123",
            "occasion": "casual",
            "look_gender": "men",
            "look_role": "partner",
            "look_title": "Your partner's look",
            "coordinated_with": (
                "Coordinated with the rust dress — navy blue complements it "
                "at the same casual level."
            ),
        }

        store = InMemorySessionStore()
        monkeypatch.setattr(deps, "_session_store", store)
        monkeypatch.setattr(deps, "_llm", _MockLLM())
        monkeypatch.setattr(deps, "_config", {
            "agent": {"max_iterations": 3},
            "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
        })
        monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")

        def get_factory() -> Any:
            agent = _MockAgent(partner_result)

            def factory(memory: Any, streaming: bool = False) -> _MockAgent:
                return agent

            return factory

        monkeypatch.setattr(deps, "get_agent_factory", get_factory)

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/chat", json={"message": "what should my husband wear with this?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["look_role"] == "partner"
        assert data["look_title"] == "Your partner's look"
        assert "rust dress" in data["coordinated_with"]
        assert data["items"][0]["gender"] == "men"

    def test_non_partner_response_omits_look_role(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi.testclient import TestClient

        import api.deps as deps
        from api.main import app
        from api.session import InMemorySessionStore

        class _MockLLM:
            def generate(self, prompt: str, system: str = None, **kwargs) -> str:
                return "ok"

            def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
                yield "ok"

            def chat(self, messages: list[dict], **kwargs) -> str:
                return "ok"

            def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
                yield "ok"

        class _MockAgent:
            def __init__(self, result: dict[str, Any]) -> None:
                self._result = result

            def invoke(self, state: dict, **kwargs) -> dict[str, Any]:
                result = dict(self._result)
                result.setdefault("messages", state.get("messages", []))
                return result

        plain_result = {
            "retrieved_items": [],
            "filters": {},
            "tool_calls": [{"router_decision": {"action": "search", "query": "hello"}}],
            "final_answer": "Here are some results.",
            "iteration": 1,
            "new_items_this_turn": False,
            "out_of_catalogue": False,
            "excluded_colours": None,
        }

        store = InMemorySessionStore()
        monkeypatch.setattr(deps, "_session_store", store)
        monkeypatch.setattr(deps, "_llm", _MockLLM())
        monkeypatch.setattr(deps, "_config", {
            "agent": {"max_iterations": 3},
            "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
        })
        monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")

        def get_factory() -> Any:
            agent = _MockAgent(plain_result)

            def factory(memory: Any, streaming: bool = False) -> _MockAgent:
                return agent

            return factory

        monkeypatch.setattr(deps, "get_agent_factory", get_factory)

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/chat", json={"message": "hello"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["look_role"] is None
        assert data["look_title"] is None
        assert data["coordinated_with"] is None


# ---------------------------------------------------------------------------
# 8. Offline real-index composition evidence (requires_index)
# ---------------------------------------------------------------------------

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


class _MockLLM:
    """Deliberately returns a WRONG (non-outfit) router decision so tests only
    pass if the partner-look routing is fully deterministic. Also used as the
    rationale-generation LLM — its canned (non-JSON) output always falls back
    to the deterministic template_rationale, which is fine for these tests."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def _next(self) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        return self._next()

    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
        yield self._next()

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        yield self._next()


@pytest.fixture(scope="module")
def _unified_index() -> tuple:
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _MINIMAL_CONFIG)
    return retriever, catalogue_df


def _state_with_fake_anchor(query: str, anchor_item: dict, messages: list[dict], memory) -> dict:
    """Build an AgentState with a synthetic session "seed" item already in
    retrieved_items — mirrors tests/test_style_this_anchor.py's approach of
    pre-seeding session state directly rather than running a real turn 1.
    The anchor item does NOT need to resolve against catalogue_df (it's only
    used as read-only colour/type/gender context by compose_partner_look /
    build_coordinated_with_text) — only the NEW partner-gender seed the
    composer retrieves needs to be a real catalogue row.
    """
    return {
        "messages": messages,
        "user_query": query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": [anchor_item],
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


@pytest.mark.requires_index
def test_husband_partner_look_coordinates_with_womens_black_dress(
    _unified_index: tuple,
) -> None:
    """Session anchor: a women's black dress look. "What should my husband wear
    with this?" must produce a SEPARATE men's coordinated companion look."""
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "black dress"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    anchor_item = {
        "article_id": "FAKE_ANCHOR_BLACK_DRESS",
        "prod_name": "Black Dress",
        "display_name": "Black Dress",
        "colour": "black",
        "product_type": "dress",
        "department": "Women",
        "gender": "women",
        "_role": "seed",
    }
    query = "what should my husband wear with this?"
    state = _state_with_fake_anchor(
        query, anchor_item, [{"role": "user", "content": query}], memory
    )

    result = agent.invoke(state)

    print("\n=== husband partner-look (women's black dress anchor) ===")
    print("look_role:", result.get("look_role"))
    print("look_title:", result.get("look_title"))
    print("coordinated_with:", result.get("coordinated_with"))
    print("occasion:", result.get("occasion"), "| look_gender:", result.get("look_gender"))
    for it in result.get("retrieved_items", []):
        print(" item:", it.get("_slot") or it.get("_role"), "|", it.get("prod_name"),
              "|", it.get("colour"), "|", it.get("gender"))

    assert result.get("look_role") == "partner"
    assert result.get("look_title") == "Your partner's look"
    assert result.get("look_gender") == "men"
    items = result.get("retrieved_items", [])
    assert items, "expected at least a men's seed item"
    for it in items:
        assert it.get("gender") == "men", f"non-men item leaked into partner look: {it}"

    coordinated_with = result.get("coordinated_with") or ""
    assert coordinated_with, "expected a non-empty coordinated_with board field"
    assert "black dress" in coordinated_with.lower()

    from src.agents.outfit.coherence import couple_harmony_palette
    palette = couple_harmony_palette("black")
    assert any(colour in coordinated_with.lower() for colour in palette), (
        f"expected coordinated_with to reference a harmony-map colour {palette}, "
        f"got: {coordinated_with!r}"
    )


@pytest.mark.requires_index
def test_wife_partner_look_coordinates_with_mens_sangeet_kurta(
    _unified_index: tuple,
) -> None:
    """Session anchor: a men's sangeet kurta look. "Style my wife to match"
    must produce a women's ETHNIC companion look — no western-marker items."""
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "kurta"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    anchor_item = {
        "article_id": "FAKE_ANCHOR_MENS_KURTA",
        "prod_name": "Blue Embellished Kurta",
        "display_name": "Blue Embellished Kurta",
        "colour": "blue",
        "product_type": "kurta",
        "department": "Men",
        "gender": "men",
        "_role": "seed",
    }
    # First message establishes "sangeet" occasion history (mirrors how a real
    # multi-turn conversation would have already built the men's sangeet look);
    # the second is the actual partner-styling turn under test.
    query = "style my wife to match"
    messages = [
        {"role": "user", "content": "build me a sangeet look for men"},
        {"role": "user", "content": query},
    ]
    state = _state_with_fake_anchor(query, anchor_item, messages, memory)

    result = agent.invoke(state)

    print("\n=== wife partner-look (men's sangeet kurta anchor) ===")
    print("look_role:", result.get("look_role"))
    print("occasion:", result.get("occasion"), "| look_gender:", result.get("look_gender"))
    print("coordinated_with:", result.get("coordinated_with"))
    for it in result.get("retrieved_items", []):
        print(" item:", it.get("_slot") or it.get("_role"), "|", it.get("prod_name"),
              "|", it.get("colour"), "|", it.get("gender"))

    assert result.get("look_role") == "partner"
    assert result.get("occasion") == "sangeet"
    assert result.get("look_gender") == "women"
    items = result.get("retrieved_items", [])
    assert items, "expected at least a women's seed item"
    for it in items:
        assert it.get("gender") == "women", f"non-women item leaked into partner look: {it}"
        text = ((it.get("prod_name") or "") + " " + (it.get("product_type") or "")).lower()
        assert "sneaker" not in text and "denim" not in text, f"western marker leaked: {it}"

    assert result.get("coordinated_with"), "expected a non-empty coordinated_with board field"


@pytest.mark.requires_index
def test_no_anchor_partner_request_gets_honest_prompt(_unified_index: tuple) -> None:
    """No session anchor/look yet — must respond with an honest prompt, never
    a guess (requirement 2)."""
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "shirt"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    query = "what should my husband wear with this?"
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

    result = agent.invoke(state)

    plan = json.loads(result.get("current_plan") or "{}")
    assert plan.get("action") == "pending_answer"
    assert "wearing" in plan.get("text", "").lower() or "show me" in plan.get("text", "").lower()
    assert result.get("look_role") is None


# ---------------------------------------------------------------------------
# 8(c)/(d). P2 couple-from-scratch — "style us as a couple for a reception"
# with NO prior session anchor.  An occasion IS named ("reception"), so this
# must NOT fall into the honest-refusal branch above (that branch only fires
# when NEITHER an anchor NOR a real occasion signal is present).
# ---------------------------------------------------------------------------


@pytest.mark.requires_index
def test_couple_from_scratch_no_anchor_no_budget_composes_both_looks(
    _unified_index: tuple,
) -> None:
    """No session anchor, no budget, but "reception" is a real occasion signal
    — must bootstrap a from-scratch couple pair (P2), not the honest-refusal
    prompt used by test_no_anchor_partner_request_gets_honest_prompt above."""
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "reception outfit"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    query = "style us as a couple for a reception"
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

    result = agent.invoke(state)

    print("\n=== couple-from-scratch (reception, no anchor, no budget) ===")
    print("look_role:", result.get("look_role"),
          "| partner_look_role:", result.get("partner_look_role"))
    print("occasion:", result.get("occasion"), "| look_gender:", result.get("look_gender"))
    print("partner_occasion:", result.get("partner_occasion"),
          "| partner_look_gender:", result.get("partner_look_gender"))

    assert result.get("look_role") == "couple_primary"
    assert result.get("partner_look_role") == "couple_partner"
    assert result.get("occasion") == "reception"
    assert result.get("partner_occasion") == "reception"

    primary_items = result.get("retrieved_items", [])
    partner_items = result.get("partner_retrieved_items", [])
    assert primary_items, "expected at least a primary seed item"
    assert partner_items, "expected at least a partner seed item"

    primary_gender = result.get("look_gender")
    partner_gender = result.get("partner_look_gender")
    assert primary_gender in ("men", "women")
    assert partner_gender in ("men", "women")
    assert primary_gender != partner_gender, "couple pair must be opposite genders"

    for it in primary_items:
        assert it.get("gender") == primary_gender
    for it in partner_items:
        assert it.get("gender") == partner_gender

    seed = next((it for it in primary_items if it.get("_role") == "seed"), primary_items[0])
    palette = couple_harmony_palette((seed.get("colour") or "").lower())
    partner_seed = next(
        (it for it in partner_items if it.get("_role") == "seed"), partner_items[0]
    )
    assert (partner_seed.get("colour") or "").lower() in palette or result.get(
        "partner_coordinated_with"
    ), "expected either an in-palette partner colour or a coordinated_with note"


@pytest.mark.requires_index
def test_couple_from_scratch_with_budget_respects_cap_per_person(
    _unified_index: tuple,
) -> None:
    """Same as above, WITH a stated budget — per this module's documented
    assumption (see compose_couple_look docstring), the cap applies
    INDEPENDENTLY to EACH look, not a combined couple total."""
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "reception outfit under 15000"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    query = "style us as a couple for a reception under ₹15000"
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

    result = agent.invoke(state)

    print("\n=== couple-from-scratch (reception, no anchor, budget 15000) ===")
    print("budget_total_inr:", result.get("budget_total_inr"),
          "| partner_budget_total_inr:", result.get("partner_budget_total_inr"))

    assert result.get("look_role") == "couple_primary"
    assert result.get("partner_look_role") == "couple_partner"
    # Per-person cap assumption: EACH look's own total must respect 15000 —
    # this is NOT a combined/split assertion (see compose_couple_look docstring).
    if result.get("budget_total_inr") is not None:
        assert result["budget_total_inr"] <= 15000
    if result.get("partner_budget_total_inr") is not None:
        assert result["partner_budget_total_inr"] <= 15000
