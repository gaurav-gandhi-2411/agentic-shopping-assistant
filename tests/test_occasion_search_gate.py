"""Regression tests for the 2026-07-11 occasion register gate on the plain
search path (src.agents.graph.search_node) and the groom/bride partner-intent
over-trigger fix (src.agents.outfit.partner.detect_partner_intent).

Root cause (verified via code trace before fixing): src.retrieval.hybrid_
search.HybridRetriever.search never referenced "occasion" at all; occasion
only entered plain search as raw keyword text appended to the query, and
only when NO garment-type signal was present (_OCCASION_QUERY_TERMS). The
composer's occasion register gates (is_coherent_candidate) and
fabric/embellishment rerank (fabric_score_delta) were only ever applied
inside compose_outfit, never on plain search results.
"""
from __future__ import annotations

from src.agents.outfit.coherence import is_coherent_candidate
from src.agents.outfit.partner import detect_partner_intent
from src.agents.outfit.slots import fabric_score_delta


class TestOccasionCoherenceGateIsSlotAgnostic:
    """The search-path fix passes slot_name="top" to bypass only the
    dupatta-specific gate — every ethnic/western/office register gate must
    still fire regardless of that neutral slot name."""

    def test_western_item_rejected_for_ethnic_only_occasion(self) -> None:
        item = {"product_type": "dress", "prod_name": "Black Bodycon Denim Mini Dress",
                "gender": "women"}
        assert not is_coherent_candidate(item, "sangeet", "women", "top")

    def test_ethnic_item_passes_ethnic_only_occasion(self) -> None:
        item = {"product_type": "lehenga", "prod_name": "Red Silk Lehenga Choli",
                "gender": "women"}
        assert is_coherent_candidate(item, "sangeet", "women", "top")

    def test_ethnic_item_rejected_for_western_register_occasion(self) -> None:
        item = {"product_type": "kurta", "prod_name": "Printed Anarkali Kurta",
                "gender": "women"}
        assert not is_coherent_candidate(item, "office", "women", "top")

    def test_western_item_passes_western_register_occasion(self) -> None:
        item = {"product_type": "shirt", "prod_name": "White Formal Shirt",
                "gender": "men"}
        assert is_coherent_candidate(item, "office", "men", "top")

    def test_dupatta_gate_not_spuriously_triggered_by_neutral_slot(self) -> None:
        # Gate 1 (dupatta-for-men reject) only fires for slot_name=="accessory" —
        # passing "top" must never spuriously reject a men's item for this reason.
        item = {"product_type": "kurta", "prod_name": "Men Cotton Kurta", "gender": "men"}
        assert is_coherent_candidate(item, "sangeet", "men", "top")


class TestFabricScoreDeltaHaldiSangeetRegister:
    def test_lightweight_item_boosted_for_haldi(self) -> None:
        item = {"prod_name": "Yellow Cotton Floral Kurta", "detail_desc": ""}
        assert fabric_score_delta(item, "haldi") > 0

    def test_embellished_item_penalized_for_haldi(self) -> None:
        item = {"prod_name": "Heavy Embroidered Bridal Kurta", "detail_desc": ""}
        assert fabric_score_delta(item, "haldi") < 0

    def test_embellished_item_boosted_for_sangeet(self) -> None:
        item = {"prod_name": "Sequin Embellished Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "sangeet") > 0

    def test_no_delta_for_non_register_occasion(self) -> None:
        item = {"prod_name": "Sequin Embellished Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "office") == 0.0


class TestGroomBridePartnerIntentFix:
    """Live-proven defect (2026-07-11): 'sherwani for groom' — a plain
    single-item search — misrouted to a partner-styling clarify question and
    never searched at all, because "groom" fired the same unconditional
    relationship-noun gate as "husband"/"wife"."""

    def test_plain_groom_query_does_not_trigger_partner_intent(self) -> None:
        result = detect_partner_intent("sherwani for groom")
        assert not result.matched

    def test_plain_bride_query_does_not_trigger_partner_intent(self) -> None:
        result = detect_partner_intent("lehenga for bride")
        assert not result.matched

    def test_groom_with_styling_verb_still_triggers(self) -> None:
        # Pinned pre-existing behavior (tests/test_partner_styling.py) — must
        # not regress: "the groom needs a matching look" IS partner intent.
        result = detect_partner_intent("the groom needs a matching look")
        assert result.matched
        assert result.gender_hint == "men"

    def test_bride_with_styling_verb_still_triggers(self) -> None:
        result = detect_partner_intent("the bride needs a matching look")
        assert result.matched
        assert result.gender_hint == "women"

    def test_husband_wife_still_unconditional(self) -> None:
        # husband/wife/boyfriend/girlfriend are unchanged — only groom/bride moved.
        assert detect_partner_intent("kurta for husband").matched
        assert detect_partner_intent("saree for wife").matched
