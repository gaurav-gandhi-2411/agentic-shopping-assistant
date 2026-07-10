"""Wave 7 — Indian wedding occasion model expansion.

Covers the 4 new occasion slugs (haldi, mehendi, reception, engagement) split
out of / added alongside the original 9-slug model. No network/LLM calls —
pure deterministic unit tests against occasions.py/intent_parser.py/
composer.py/slots.py/coherence.py.
"""
from __future__ import annotations

import pytest

from src.agents.graph import _OCCASION_LOOK_RE, _OUTFIT_INTENT_RE, _OUTFIT_OCCASION_RE
from src.agents.intent_parser import parse_intent
from src.agents.outfit.coherence import colour_score
from src.agents.outfit.composer import _anchor_query_for_occasion
from src.agents.outfit.occasions import EITHER, ETHNIC_HEAVY, ETHNIC_ONLY, OCCASIONS, get_occasion
from src.agents.outfit.slots import _FORMAL_ETHNIC_OCCASIONS, fabric_score_delta

# ── (a) OCCASIONS contains the 4 new slugs with expected formality/ethnic_lean ──


class TestNewOccasionsPresent:
    def test_haldi(self) -> None:
        occ = OCCASIONS["haldi"]
        assert occ.formality == 3
        assert occ.ethnic_lean == ETHNIC_ONLY

    def test_mehendi(self) -> None:
        occ = OCCASIONS["mehendi"]
        assert occ.formality == 3
        assert occ.ethnic_lean == ETHNIC_ONLY

    def test_reception(self) -> None:
        occ = OCCASIONS["reception"]
        assert occ.formality == 5
        assert occ.ethnic_lean == ETHNIC_HEAVY

    def test_engagement(self) -> None:
        occ = OCCASIONS["engagement"]
        assert occ.formality == 4
        assert occ.ethnic_lean == EITHER


# ── (b) get_occasion("haldi_mehendi") returns haldi (legacy alias) ──────────


def test_haldi_mehendi_alias_resolves_to_haldi() -> None:
    occ = get_occasion("haldi_mehendi")
    assert occ.slug == "haldi"


# ── (c) intent parsing for new occasion free text ───────────────────────────


class TestIntentParsingNewOccasions:
    def test_haldi(self) -> None:
        assert parse_intent("outfit for haldi").occasion == "haldi"

    def test_mehendi_look(self) -> None:
        assert parse_intent("mehendi look for women").occasion == "mehendi"

    def test_mehndi_spelling_variant(self) -> None:
        assert parse_intent("what should I wear to mehndi").occasion == "mehendi"

    def test_reception_outfit(self) -> None:
        assert parse_intent("reception outfit for men").occasion == "reception"

    def test_engagement(self) -> None:
        assert parse_intent("outfit for engagement").occasion == "engagement"

    def test_roka(self) -> None:
        assert parse_intent("what to wear for roka").occasion == "engagement"

    def test_sagai(self) -> None:
        assert parse_intent("sagai ceremony outfit").occasion == "engagement"

    def test_shaadi_guest_look(self) -> None:
        assert parse_intent("shaadi guest look").occasion == "wedding_guest"

    def test_cocktail(self) -> None:
        assert parse_intent("cocktail party outfit").occasion == "reception"


# ── (d) anchor query non-empty + contains signature tokens ──────────────────


class TestAnchorQueryForNewOccasions:
    def test_haldi_query(self) -> None:
        query = _anchor_query_for_occasion("haldi", "women")
        assert query
        assert "yellow" in query and "marigold" in query

    def test_mehendi_query(self) -> None:
        query = _anchor_query_for_occasion("mehendi", "women")
        assert query
        assert "green" in query and "mint" in query

    def test_reception_query(self) -> None:
        query = _anchor_query_for_occasion("reception", "men")
        assert query
        assert "embellished" in query and "indo-western" in query

    def test_engagement_query(self) -> None:
        query = _anchor_query_for_occasion("engagement", "women")
        assert query
        assert "pastel" in query and "semi-formal" in query


# ── (e) new slugs are in the footwear-required set ──────────────────────────


class TestFootwearRequiredSet:
    @pytest.mark.parametrize("slug", ["haldi", "mehendi", "reception", "engagement"])
    def test_slug_in_formal_ethnic_occasions(self, slug: str) -> None:
        assert slug in _FORMAL_ETHNIC_OCCASIONS


# ── (f) colour_score direction checks ───────────────────────────────────────


class TestColourScoreDirection:
    def test_mehendi_favors_green_over_black(self) -> None:
        green_score = colour_score("green", "red", "mehendi")
        black_score = colour_score("black", "red", "mehendi")
        assert green_score > black_score

    def test_reception_favors_jewel_over_pale_casual(self) -> None:
        jewel_score = colour_score("wine", "red", "reception")
        pale_score = colour_score("light pink", "red", "reception")
        assert jewel_score > pale_score

    def test_reception_favors_dark_over_pale_casual(self) -> None:
        dark_score = colour_score("black", "red", "reception")
        pale_score = colour_score("light beige", "red", "reception")
        assert dark_score > pale_score


# ── (g) fabric_score_delta signs ────────────────────────────────────────────


class TestFabricScoreDeltaNewSlugs:
    def test_reception_embellished_positive(self) -> None:
        item = {"prod_name": "Heavy Embroidered Gown", "detail_desc": ""}
        assert fabric_score_delta(item, "reception") == pytest.approx(0.1)

    def test_reception_lightweight_negative(self) -> None:
        item = {"prod_name": "Cotton Floral Kurti", "detail_desc": ""}
        assert fabric_score_delta(item, "reception") == pytest.approx(-0.1)

    def test_haldi_lightweight_positive(self) -> None:
        item = {"prod_name": "Floral Cotton Kurti", "detail_desc": ""}
        assert fabric_score_delta(item, "haldi") == pytest.approx(0.1)

    def test_mehendi_lightweight_positive(self) -> None:
        item = {"prod_name": "Floral Cotton Kurti", "detail_desc": ""}
        assert fabric_score_delta(item, "mehendi") == pytest.approx(0.1)

    def test_haldi_embellished_negative(self) -> None:
        item = {"prod_name": "Heavy Zari Embroidered Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "haldi") == pytest.approx(-0.1)

    def test_mehendi_embellished_negative(self) -> None:
        item = {"prod_name": "Heavy Zari Embroidered Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "mehendi") == pytest.approx(-0.1)


# ── (h) deterministic pre-LLM outfit-routing fast path covers the new words ──
# Mirrors tests/test_occasion_outfit_and_refinement.py::_routes_to_outfit —
# graph.py router_node's RED 2c first-turn gate condition:
# `_OUTFIT_OCCASION_RE.search(raw_q) and (_OUTFIT_INTENT_RE.search(raw_q) or
# _OCCASION_LOOK_RE.search(raw_q))`.


def _routes_to_outfit(query: str) -> bool:
    return bool(
        _OUTFIT_OCCASION_RE.search(query)
        and (_OUTFIT_INTENT_RE.search(query) or _OCCASION_LOOK_RE.search(query))
    )


class TestNewOccasionFastPathRouting:
    def test_reception_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("reception look for a woman") is True

    def test_engagement_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("engagement outfit") is True

    def test_roka_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("roka outfit for men") is True

    def test_sagai_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("sagai outfit") is True

    def test_shaadi_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("shaadi look") is True

    def test_cocktail_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("cocktail look") is True
