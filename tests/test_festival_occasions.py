"""Wave 8 — festival occasion model expansion.

Covers 5 new first-class occasion slugs (diwali, navratri, karva_chauth,
raksha_bandhan, eid) added alongside the existing haldi/sangeet/mehendi/
office/festive_puja model. Pure additive — festive_puja is untouched and
still owns "puja"/"festive" free text. No network/LLM calls — pure
deterministic unit tests against occasions.py/intent_parser.py/composer.py/
slots.py/coherence.py/rationale.py/graph.py, mirroring
tests/test_wedding_occasions.py's structure.
"""
from __future__ import annotations

import pytest

from src.agents.graph import _OCCASION_LOOK_RE, _OUTFIT_INTENT_RE, _OUTFIT_OCCASION_RE
from src.agents.intent_parser import parse_intent
from src.agents.outfit.coherence import colour_score, is_coherent_candidate
from src.agents.outfit.composer import _anchor_query_for_occasion
from src.agents.outfit.occasions import EITHER, ETHNIC_HEAVY, ETHNIC_ONLY, OCCASIONS, get_occasion
from src.agents.outfit.rationale import _OCCASION_REGISTER_HINTS
from src.agents.outfit.slots import (
    _FORMAL_ETHNIC_OCCASIONS,
    _occasion_register_tokens,
    fabric_score_delta,
)

# ── (a) OCCASIONS contains the 5 new slugs with expected formality/ethnic_lean ──


class TestNewOccasionsPresent:
    def test_diwali(self) -> None:
        occ = OCCASIONS["diwali"]
        assert occ.formality == 4
        assert occ.ethnic_lean == ETHNIC_HEAVY

    def test_navratri(self) -> None:
        occ = OCCASIONS["navratri"]
        assert occ.formality == 3
        assert occ.ethnic_lean == ETHNIC_ONLY

    def test_karva_chauth(self) -> None:
        occ = OCCASIONS["karva_chauth"]
        assert occ.formality == 4
        assert occ.ethnic_lean == ETHNIC_ONLY

    def test_raksha_bandhan(self) -> None:
        occ = OCCASIONS["raksha_bandhan"]
        assert occ.formality == 2
        assert occ.ethnic_lean == EITHER

    def test_eid(self) -> None:
        occ = OCCASIONS["eid"]
        assert occ.formality == 4
        assert occ.ethnic_lean == ETHNIC_HEAVY

    def test_festive_puja_untouched(self) -> None:
        """festive_puja is a sibling, not replaced by the new slugs."""
        occ = OCCASIONS["festive_puja"]
        assert occ.formality == 4
        assert occ.ethnic_lean == ETHNIC_HEAVY


def test_get_occasion_resolves_all_new_slugs() -> None:
    for slug in ("diwali", "navratri", "karva_chauth", "raksha_bandhan", "eid"):
        assert get_occasion(slug).slug == slug


# ── (b) intent parsing for new occasion free text ───────────────────────────


class TestIntentParsingNewOccasions:
    def test_diwali(self) -> None:
        assert parse_intent("outfit for diwali").occasion == "diwali"

    def test_deepavali_spelling_variant(self) -> None:
        assert parse_intent("deepavali outfit for women").occasion == "diwali"

    def test_navratri(self) -> None:
        assert parse_intent("navratri outfit for women").occasion == "navratri"

    def test_garba(self) -> None:
        assert parse_intent("garba outfit").occasion == "navratri"

    def test_dandiya(self) -> None:
        assert parse_intent("dandiya night outfit").occasion == "navratri"

    def test_chaniya_choli_phrase(self) -> None:
        assert parse_intent("chaniya choli for garba").occasion == "navratri"

    def test_karva_chauth(self) -> None:
        assert parse_intent("what to wear for karva chauth").occasion == "karva_chauth"

    def test_karwa_chauth_spelling_variant(self) -> None:
        assert parse_intent("karwa chauth outfit for women").occasion == "karva_chauth"

    def test_raksha_bandhan(self) -> None:
        assert parse_intent("raksha bandhan outfit").occasion == "raksha_bandhan"

    def test_rakhi(self) -> None:
        assert parse_intent("rakhi day outfit for sister").occasion == "raksha_bandhan"

    def test_eid(self) -> None:
        assert parse_intent("eid outfit for men").occasion == "eid"

    def test_eid_not_false_matched_inside_other_words(self) -> None:
        # "wedding" contains no "eid" substring adjacent to word boundaries,
        # but this pins the general no-false-match invariant for the new
        # bare-word "eid" entry specifically.
        assert parse_intent("wedding guest outfit").occasion != "eid"

    def test_festive_puja_keywords_unaffected(self) -> None:
        """"puja"/"festive" free text must keep resolving to festive_puja,
        not accidentally shadowed by a new slug."""
        assert parse_intent("puja outfit for women").occasion == "festive_puja"
        assert parse_intent("festive outfit for men").occasion == "festive_puja"


# ── (c) anchor query non-empty + contains signature tokens ──────────────────


class TestAnchorQueryForNewOccasions:
    def test_diwali_query(self) -> None:
        query = _anchor_query_for_occasion("diwali", "women")
        assert query
        assert "gold" in query and "embellished" in query

    def test_navratri_query(self) -> None:
        query = _anchor_query_for_occasion("navratri", "women")
        assert query
        assert "chaniya choli" in query and "dance" in query

    def test_karva_chauth_query(self) -> None:
        query = _anchor_query_for_occasion("karva_chauth", "women")
        assert query
        assert "red" in query and "traditional" in query

    def test_raksha_bandhan_query(self) -> None:
        query = _anchor_query_for_occasion("raksha_bandhan", "men")
        assert query
        assert "casual" in query

    def test_eid_query(self) -> None:
        query = _anchor_query_for_occasion("eid", "men")
        assert query
        assert "pastel" in query


# ── (d) footwear-required set ────────────────────────────────────────────────


class TestFootwearRequiredSet:
    @pytest.mark.parametrize("slug", ["diwali", "navratri", "karva_chauth", "eid"])
    def test_slug_in_formal_ethnic_occasions(self, slug: str) -> None:
        assert slug in _FORMAL_ETHNIC_OCCASIONS

    def test_raksha_bandhan_not_in_formal_ethnic_occasions(self) -> None:
        assert "raksha_bandhan" not in _FORMAL_ETHNIC_OCCASIONS


# ── (e) colour_score direction + no cross-occasion bleed ────────────────────


class TestColourScoreDirection:
    def test_diwali_favors_gold_over_black(self) -> None:
        gold_score = colour_score("gold", "red", "diwali")
        black_score = colour_score("black", "red", "diwali")
        assert gold_score > black_score

    def test_diwali_favors_red_over_pastel(self) -> None:
        red_score = colour_score("red", "gold", "diwali")
        pastel_score = colour_score("light pink", "gold", "diwali")
        assert red_score > pastel_score

    def test_navratri_favors_bright_over_muted(self) -> None:
        bright_score = colour_score("yellow", "red", "navratri")
        muted_score = colour_score("mustard", "red", "navratri")
        assert bright_score > muted_score

    def test_navratri_favors_bright_over_black(self) -> None:
        bright_score = colour_score("green", "red", "navratri")
        black_score = colour_score("black", "red", "navratri")
        assert bright_score > black_score

    def test_karva_chauth_favors_red_over_black(self) -> None:
        red_score = colour_score("red", "gold", "karva_chauth")
        black_score = colour_score("black", "gold", "karva_chauth")
        assert red_score > black_score

    def test_karva_chauth_favors_red_over_pastel(self) -> None:
        red_score = colour_score("maroon", "gold", "karva_chauth")
        pastel_score = colour_score("light blue", "gold", "karva_chauth")
        assert red_score > pastel_score

    def test_eid_favors_pastel_over_dark(self) -> None:
        pastel_score = colour_score("light blue", "white", "eid")
        dark_score = colour_score("black", "white", "eid")
        assert pastel_score > dark_score

    def test_eid_does_not_favor_diwali_gold_red_palette(self) -> None:
        """Eid must NOT default to diwali's gold/red glam palette — the two
        festivals are visually distinct and must not conflate."""
        pastel_score = colour_score("white", "cream", "eid")
        gold_score = colour_score("gold", "cream", "eid")
        red_score = colour_score("red", "cream", "eid")
        assert pastel_score > gold_score
        assert pastel_score > red_score

    def test_diwali_does_not_favor_eid_pastel_palette(self) -> None:
        """Symmetric check: diwali must not favour eid's pastel register."""
        gold_score = colour_score("gold", "maroon", "diwali")
        pastel_score = colour_score("light blue", "maroon", "diwali")
        assert gold_score > pastel_score

    def test_raksha_bandhan_falls_through_to_generic_either_branch(self) -> None:
        """No dedicated override — light-touch, generic EITHER-branch scoring."""
        neutral_score = colour_score("white", "red", "raksha_bandhan")
        assert neutral_score == 1.0  # neutral colour, matches generic western branch


# ── (f) fabric_score_delta signs ────────────────────────────────────────────


class TestFabricScoreDeltaNewSlugs:
    def test_diwali_embellished_positive(self) -> None:
        item = {"prod_name": "Heavy Embroidered Sequin Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "diwali") == pytest.approx(0.1)

    def test_diwali_lightweight_negative(self) -> None:
        item = {"prod_name": "Cotton Floral Printed Kurti", "detail_desc": ""}
        assert fabric_score_delta(item, "diwali") == pytest.approx(-0.1)

    def test_raksha_bandhan_lightweight_positive(self) -> None:
        item = {"prod_name": "Floral Cotton Kurti", "detail_desc": ""}
        assert fabric_score_delta(item, "raksha_bandhan") == pytest.approx(0.1)

    def test_raksha_bandhan_embellished_negative(self) -> None:
        item = {"prod_name": "Heavy Zari Embroidered Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "raksha_bandhan") == pytest.approx(-0.1)

    def test_navratri_no_delta(self) -> None:
        item = {"prod_name": "Heavy Embroidered Sequin Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "navratri") == 0.0

    def test_karva_chauth_no_delta(self) -> None:
        item = {"prod_name": "Heavy Embroidered Sequin Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "karva_chauth") == 0.0

    def test_eid_no_delta(self) -> None:
        item = {"prod_name": "Heavy Embroidered Sequin Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "eid") == 0.0


# ── (g) occasion register tokens ────────────────────────────────────────────


class TestOccasionRegisterTokens:
    def test_diwali_tokens(self) -> None:
        assert _occasion_register_tokens("diwali") == "festive glam gold embellished"

    def test_navratri_tokens(self) -> None:
        assert _occasion_register_tokens("navratri") == "chaniya choli bright colourful dance"

    def test_karva_chauth_tokens(self) -> None:
        assert _occasion_register_tokens("karva_chauth") == "red traditional bridal ethnic"

    def test_raksha_bandhan_tokens(self) -> None:
        assert _occasion_register_tokens("raksha_bandhan") == "casual festive light"

    def test_eid_tokens(self) -> None:
        assert _occasion_register_tokens("eid") == "pastel elegant festive"


# ── (h) rationale register hints present for all 5 new slugs ────────────────


def test_rationale_register_hints_present_for_all_new_slugs() -> None:
    for slug in ("diwali", "navratri", "karva_chauth", "raksha_bandhan", "eid"):
        assert slug in _OCCASION_REGISTER_HINTS
        assert _OCCASION_REGISTER_HINTS[slug]


# ── (i) coherence gates reject western-casual items same as haldi/sangeet ───


class TestCoherenceGatesRejectWesternForEthnicOccasions:
    @pytest.mark.parametrize("occasion_slug", ["diwali", "navratri", "karva_chauth", "eid"])
    def test_western_item_rejected(self, occasion_slug: str) -> None:
        item = {"product_type": "dress", "prod_name": "Black Bodycon Denim Mini Dress",
                "gender": "women"}
        assert not is_coherent_candidate(item, occasion_slug, "women", "top")

    @pytest.mark.parametrize("occasion_slug", ["diwali", "navratri", "karva_chauth", "eid"])
    def test_ethnic_item_passes(self, occasion_slug: str) -> None:
        item = {"product_type": "lehenga", "prod_name": "Red Silk Lehenga Choli",
                "gender": "women"}
        assert is_coherent_candidate(item, occasion_slug, "women", "top")


# ── (j) deterministic pre-LLM outfit-routing fast path covers the new words ──
# Mirrors tests/test_wedding_occasions.py's (h) block — graph.py router_node's
# RED 2c first-turn gate condition.


def _routes_to_outfit(query: str) -> bool:
    return bool(
        _OUTFIT_OCCASION_RE.search(query)
        and (_OUTFIT_INTENT_RE.search(query) or _OCCASION_LOOK_RE.search(query))
    )


class TestNewOccasionFastPathRouting:
    def test_diwali_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("diwali outfit") is True

    def test_diwali_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("diwali look for women") is True

    def test_navratri_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("navratri outfit") is True

    def test_garba_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("garba look") is True

    def test_karva_chauth_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("karva chauth outfit") is True

    def test_karwa_chauth_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("karwa chauth look") is True

    def test_raksha_bandhan_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("raksha bandhan outfit") is True

    def test_rakhi_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("rakhi look") is True

    def test_eid_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("eid outfit") is True

    def test_eid_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("eid look for men") is True
