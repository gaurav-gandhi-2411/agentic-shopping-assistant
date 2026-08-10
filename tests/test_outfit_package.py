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
    classify_item,
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

    def test_all_occasions_present(self) -> None:
        expected = {
            "casual", "smart_casual", "office", "haldi", "mehendi",
            "party_evening", "festive_puja", "wedding_guest", "engagement",
            "sangeet", "traditional_ethnic", "reception",
            # Wave 8 festival-occasion expansion — siblings of festive_puja,
            # not replacements (see occasions.py for the formality/ethnic_lean
            # rationale on each).
            "diwali", "navratri", "karva_chauth", "raksha_bandhan", "eid",
            # Wave 9 activewear/gym expansion (see occasions.py's "gym" entry).
            "gym",
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


class TestClassifyAnchorSubstringCollisionRegression2026_07_30:
    """2026-07-30 unknown-class keyword-coverage audit follow-up: bare
    "lower" and bare "cape" were originally spec'd for WESTERN_BOTTOM_
    KEYWORDS/OUTERWEAR_KEYWORDS but dropped after catalogue verification
    showed they live-match as plain substrings inside unrelated words
    ("flower", "escape"/"seascape") — classify_anchor() scans with plain
    `kw in combined`, not word-boundary matching. This matters far beyond a
    cosmetic mislabel: composer.py calls classify_anchor() directly on a
    look's own ANCHOR item to decide get_fill_slots(anchor_class, ...) — the
    entire slot composition for the look. These tests pin the CORRECT
    classification down so a future re-addition of either bare keyword
    regresses visibly here rather than silently corrupting anchor-driven
    slot composition.
    """

    def test_flower_named_top_stays_western_top_not_bottom(self) -> None:
        result = classify_anchor("top", "bebe Women Orchid Flower Essential Self Design Top")
        assert result == "western_top"

    def test_escape_named_footwear_stays_footwear_not_outerwear(self) -> None:
        result = classify_anchor("footwear", "Email Escape : Mule Heels")
        assert result == "footwear"

    def test_capes_plural_facet_still_resolves_outerwear(self) -> None:
        # "capes" (plural) is kept — both real facet values ("Ponchu &
        # Capes", "Capes & Overlays") are always plural in this catalogue,
        # so dropping the singular loses zero real coverage.
        result = classify_anchor("Capes & Overlays", "Handcrafted Mustard Pure Woolen Cape")
        assert result == "outerwear"


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


class TestClassifyAnchorBrandPrefixCollisionRegression2026_07_30:
    """2026-07-30 brand-name/ethnic-keyword collision fix: a handful of real
    catalogue BRAND names literally contain an unrelated garment keyword
    (e.g. "Jaipur Kurti" sells trousers), which used to silently override the
    item's real garment type via classify_anchor()'s combined product_type+
    name keyword scan. See slots.py's _BRAND_PREFIX_COLLISIONS docstring for
    the full catalogue-audit rationale.
    """

    def test_jaipur_kurti_trousers_is_western_not_ethnic(self) -> None:
        # Was misclassified ethnic_top via the "Kurti" brand name in
        # ETHNIC_TOP_KEYWORDS before this fix -- coherence.py's
        # is_western_item() call bypassed classify_item()'s protective
        # pt-alone early return entirely.
        assert is_western_item(
            "trousers", "Jaipur Kurti Women White Regular Fit Solid Regular Trousers"
        ) is True

    def test_salwar_studio_top_is_western_not_ethnic_bottom(self) -> None:
        assert is_western_item("top", "SALWAR STUDIO Women Orange Solid Peplum Top") is True

    def test_saree_swarg_tunic_is_ethnic_top_not_one_piece(self) -> None:
        # The brand's "Saree" (ETHNIC_ONE_PIECE_KEYWORDS, checked before
        # ETHNIC_TOP_KEYWORDS) used to force a genuine tunic to the wrong
        # ethnic sub-class, excluding it from ever filling a "top" slot
        # (SLOT_ALLOWED_CLASSES["top"] only accepts ethnic_top, not
        # ethnic_one_piece).
        result = classify_anchor("tunic", "Saree Swarg Green & Yellow Printed Tunic")
        assert result == "ethnic_top"

    def test_pepe_jeans_knitwear_is_not_western_bottom(self) -> None:
        # "jeans" (WESTERN_BOTTOM_KEYWORDS) sitting in the brand name used to
        # misclassify a Pepe Jeans sweater/cardigan as a bottom.
        result = classify_anchor("knitwear", "Pepe Jeans Men Grey Solid Crew Neck Sweater")
        assert result == "western_top"

    def test_neudis_lehenga_skirt_stays_ethnic_one_piece(self) -> None:
        """Negative control: a genuine (non-brand-collision) ethnic
        descriptor elsewhere in the name must still correctly override a
        generic/overloaded product_type facet -- the brand-prefix strip must
        never regress this."""
        result = is_ethnic_item(
            "skirt", "NEUDIS Women Maroon & Pink Floral Print Flared Maxi Lehenga Skirt"
        )
        assert result is True


class TestClassifyAnchorBrandPrefixCollisionRegression2026_08_05:
    """2026-08-05 follow-up audit: re-ran the 2026-07-30 brand-collision audit
    across the full unified catalogue and found 8 more real brand/keyword
    collisions of the same shape. See slots.py's _BRAND_PREFIX_COLLISIONS
    docstring for the full audit rationale, including why "SF Jeans by
    Pantaloons" was audited but NOT added (its rows never actually
    collide) and why "Annabelle by Pantaloons" flipped from a documented
    non-collision to a real one on 2026-08-10 (see
    test_annabelle_by_pantaloons_shrug_no_longer_western_bottom below).
    """

    def test_dressberry_top_is_western_top_not_one_piece(self) -> None:
        # "dress" (WESTERN_ONE_PIECE_KEYWORDS) in the brand name used to
        # misclassify a plain DressBerry top as western_one_piece.
        assert is_western_item(
            "top", "DressBerry Women White & Black Striped Pure Cotton Top"
        ) is True
        assert classify_anchor(
            "top", "DressBerry Women White & Black Striped Pure Cotton Top"
        ) == "western_top"

    def test_20dresses_jeans_is_western_bottom_not_one_piece(self) -> None:
        result = classify_anchor(
            "jeans", "20Dresses Women Blue Mildly Distressed Light Fade Jeans"
        )
        assert result == "western_bottom"

    def test_akkriti_by_pantaloons_top_is_western_top_not_bottom(self) -> None:
        # "pant" (WESTERN_BOTTOM_KEYWORDS) inside "Pantaloons" used to
        # misclassify a plain top as western_bottom.
        result = classify_anchor("top", "AKKRITI BY PANTALOONS Women Off-White Embroidered Boxy Top")
        assert result == "western_top"

    def test_ajile_by_pantaloons_sweatshirt_is_western_top_not_bottom(self) -> None:
        result = classify_anchor(
            "knitwear", "Ajile by Pantaloons Women Black Solid Cotton Sweatshirt"
        )
        assert result == "western_top"

    def test_honey_by_pantaloons_pullover_is_western_top_not_bottom(self) -> None:
        result = classify_anchor("Fashion", "Honey by Pantaloons Women Brown Cable Knit Pullover")
        assert result == "western_top"

    def test_rangmanch_by_pantaloons_dupatta_stays_unresolved_not_bottom(self) -> None:
        # A bare dupatta (no ethnic/western garment keyword of its own)
        # should stay "unknown" -- not get swept into western_bottom via
        # the "pant" collision in the brand name.
        result = classify_anchor(
            "dupatta", "RANGMANCH BY PANTALOONS Women Gold-Toned Woven Design Dupatta"
        )
        assert result == "unknown"

    def test_kraus_jeans_sweater_is_western_top_not_bottom(self) -> None:
        result = classify_anchor("knitwear", "Kraus Jeans Women Red Ribbed Pullover Sweater")
        assert result == "western_top"

    def test_annabelle_by_pantaloons_shrug_no_longer_western_bottom(self) -> None:
        """2026-08-10 update: this brand was a documented non-collision only
        because "shrug" (OUTERWEAR_KEYWORDS, earlier priority) always won
        before the "pant"-inside-"Pantaloons" substring was ever reached.
        Removing "shrug" from OUTERWEAR_KEYWORDS (compose-wave
        shrug-classification fix -- see slots.py's _ACCESSORY_LAYERING_
        FAMILY) exposed that collision for real, so "annabelle by
        pantaloons" was added to _BRAND_PREFIX_COLLISIONS. Without the
        brand-prefix strip this would wrongly resolve "western_bottom";
        with it, the bare shrug (no other garment keyword of its own) is
        correctly "unknown" -- classify_anchor() never returns "accessory"
        (see its own docstring), and a shrug should never anchor a look."""
        result = classify_anchor(
            "Fashion", "Annabelle by Pantaloons Women Grey Solid Open Front Winter Shrug"
        )
        assert result == "unknown"

    def test_sf_jeans_by_pantaloons_jeans_not_added_to_denylist_still_correct(self) -> None:
        result = classify_anchor("jeans", "SF JEANS by Pantaloons Women Black High-Rise Jeans")
        assert result == "western_bottom"


class TestClassifyAnchorIndowesternFirstClass2026_08_05:
    """2026-08-05: "indowestern" (product_type_name=="indowestern", 586
    catalogue rows, ~98% men's Kurta+Churidar/Dhoti/Trousers full ensembles)
    promoted from a one-off coherence.py gate special case to a first-class
    facet-equality short-circuit inside classify_anchor(). See slots.py's
    classify_anchor()/`_GENERIC_FACET_VALUES` comments for the full
    catalogue-audit rationale.
    """

    def test_bare_indowestern_set_is_ethnic_one_piece(self) -> None:
        assert classify_anchor("indowestern", "MLS INDO WESTERN 2PCS") == "ethnic_one_piece"

    def test_indowestern_with_pant_collision_is_ethnic_one_piece_not_western_bottom(self) -> None:
        # Previously misclassified western_bottom via the same "pant"
        # substring-collision mechanism as the brand-name fixes above.
        result = classify_anchor(
            "indowestern",
            "Men's Black Rayon Geometry Thigh Length Indo Western Set with Wide Leg Pant",
        )
        assert result == "ethnic_one_piece"
        assert is_western_item(
            "indowestern",
            "Men's Black Rayon Geometry Thigh Length Indo Western Set with Wide Leg Pant",
        ) is False

    def test_indowestern_with_dhoti_is_ethnic_one_piece_not_ethnic_bottom(self) -> None:
        # Previously landed on ethnic_bottom (a real ETHNIC_BOTTOM_KEYWORDS
        # "dhoti" match) instead of the full-ensemble class -- correct
        # direction (still ethnic) but SLOT_ALLOWED_CLASSES doesn't accept
        # ethnic_bottom for a "top" slot, so a 2-piece set candidate could
        # never itself fill a top+bottom anchor role consistently.
        result = classify_anchor("indowestern", "Men's Black Indowestern Set With Dhoti")
        assert result == "ethnic_one_piece"

    def test_indowestern_jewellery_mislabel_is_accessory_not_ethnic_one_piece(self) -> None:
        """Negative control: a handful of catalogue rows are jewellery
        mislabeled with product_type_name=="indowestern" (e.g. a necklace
        set) -- these must resolve "accessory" via classify_item(), not get
        swept into the new ethnic_one_piece short-circuit."""
        result = classify_item(
            "indowestern", "Multicoloured Gemstone Indo Western Necklace Set"
        )
        assert result == "accessory"

    def test_indowestern_name_substring_in_other_types_stays_unaffected(self) -> None:
        """Negative control: the short-circuit is keyed on the exact facet
        value, not a name substring -- "indo-western" also appears inside
        trousers/sherwani/kurta/nightwear rows' free-text names, where a
        substring match would wrongly reclassify unrelated items."""
        result = classify_anchor("nightwear", "Indo-Western Style Comfort Nightwear")
        assert result != "ethnic_one_piece"

    def test_is_ethnic_item_true_for_every_indowestern_row_unconditionally(self) -> None:
        assert is_ethnic_item("indowestern", "Green Indowestern | Azania") is True
        assert is_ethnic_item(
            "indowestern", "Multicoloured Gemstone Indo Western Necklace Set"
        ) is True


class TestClassifyAnchorUnknownRowApparelAudit2026_08_05:
    """2026-08-05: five product_type_name facets found genuinely apparel
    during the unknown-row audit, closed via exact facet-equality matches
    (never a name substring — each has a documented false-positive risk
    elsewhere in the catalogue if matched as a substring instead). See
    slots.py's classify_anchor() comment for the full per-facet rationale.
    """

    def test_jodhpuri_facet_is_men_formalwear(self) -> None:
        assert classify_anchor("Jodhpuri", "Men's Black Silk Blend Jodhpuri") == "men_formalwear"

    def test_jodhpuri_footwear_name_substring_unaffected(self) -> None:
        """Negative control: bare "jodhpuri" was deliberately dropped from
        MEN_FORMALWEAR_KEYWORDS as a substring because it also appears
        inside footwear rows ("Jodhpuri Mojaris") -- confirms the new
        facet-EQUALITY check (product_type_name=="footwear", not
        "jodhpuri") never reaches those rows."""
        result = classify_anchor("footwear", "Jodhpuri Mojaris")
        assert result == "footwear"

    def test_pathani_suit_is_ethnic_one_piece(self) -> None:
        assert classify_anchor("PATHANI SUIT", "MLS PATHANI SUIT 2PCS") == "ethnic_one_piece"

    def test_kids_pathani_suit_is_ethnic_one_piece(self) -> None:
        assert classify_anchor(
            "KIDS PATHANI SUIT", "MLS KIDS PATHANI SUIT 2PCS"
        ) == "ethnic_one_piece"

    def test_business_plain_suit_is_outerwear(self) -> None:
        assert classify_anchor("BUSINESS PLAIN SUIT", "MLS DOUBLE BREASTED SUIT") == "outerwear"

    def test_lower_facet_is_western_bottom(self) -> None:
        assert classify_anchor("Lower", "Grey Regular Fit Lower For Men") == "western_bottom"

    def test_lower_flower_name_substring_unaffected(self) -> None:
        """Negative control: bare "lower"/"lowers" was deliberately dropped
        from WESTERN_BOTTOM_KEYWORDS as a substring because it live-matches
        inside "flower"/"sunflower" -- confirms the new facet-EQUALITY check
        never reaches a floral-printed top typed with something other than
        "lower"."""
        result = classify_anchor("top", "Sunflower Print Boxy Top")
        assert result == "western_top"

    def test_loafer_singular_is_footwear(self) -> None:
        assert classify_anchor("Men's Loafer", "Victor") == "footwear"

    def test_sadri_is_outerwear(self) -> None:
        result = classify_anchor("sadri", "Charcoal Grey Multi-Button Sadri")
        assert result == "outerwear"

    def test_vest_pack_undergarment_stays_unknown(self) -> None:
        """Negative control: pt=="vest" (244 rows, sampled as multi-packs
        of plain undershirts, e.g. "VIP Men Vest (Pack of 11)") is
        deliberately NOT classified -- same precedent as the already-
        declined "swimwear"==briefs finding. Classifying undergarments
        would let them fill outfit slots."""
        result = classify_anchor("vest", "VIP Men Vest (Pack of 11)")
        assert result == "unknown"


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


class TestFabricScoreDeltaFormalityOverride:
    """formality_override ('minimalist'/'comfortable' — the `formality_softener`
    value a sibling intent-parser fix surfaces) fixes two live-confirmed bugs:
    (1) sangeet's base rule unconditionally boosts embellishment even for a
    "comfortable for dancing" query, with no way to suppress it; (2)
    wedding_guest has zero embellishment-awareness at all, even when the user
    explicitly asked for something low-key ("not too flashy")."""

    def test_sangeet_embellished_suppressed_when_comfortable_override(self) -> None:
        item = {"prod_name": "Heavy Embroidered Bridal Lehenga Choli", "detail_desc": ""}
        assert fabric_score_delta(
            item, "sangeet", formality_override="comfortable"
        ) == pytest.approx(-0.1)

    def test_sangeet_lightweight_favoured_when_comfortable_override(self) -> None:
        item = {"prod_name": "Cotton Floral Kurti", "detail_desc": ""}
        assert fabric_score_delta(
            item, "sangeet", formality_override="comfortable"
        ) == pytest.approx(0.1)

    def test_wedding_guest_zero_without_override(self) -> None:
        # Baseline (no signal) behaviour for wedding_guest is intentionally
        # unchanged — 0.0 regardless of embellishment.
        item = {"prod_name": "Heavy Embroidered Sequin Gown", "detail_desc": ""}
        assert fabric_score_delta(item, "wedding_guest") == pytest.approx(0.0)

    def test_wedding_guest_embellished_penalized_with_minimalist_override(self) -> None:
        item = {"prod_name": "Heavy Embroidered Sequin Gown", "detail_desc": ""}
        assert fabric_score_delta(
            item, "wedding_guest", formality_override="minimalist"
        ) == pytest.approx(-0.1)

    def test_wedding_guest_plain_favoured_with_minimalist_override(self) -> None:
        item = {"prod_name": "Cotton Floral Kurti", "detail_desc": ""}
        assert fabric_score_delta(
            item, "wedding_guest", formality_override="minimalist"
        ) == pytest.approx(0.1)

    def test_default_signature_backward_compatible(self) -> None:
        # No override arg passed at all — existing composer.py/graph.py call
        # sites (fabric_score_delta(item, occasion_slug)) must be unaffected.
        item = {"prod_name": "Heavy Embroidered Lehenga", "detail_desc": ""}
        assert fabric_score_delta(item, "sangeet") == pytest.approx(0.1)

    def test_unrecognized_override_value_falls_back_to_base_behaviour(self) -> None:
        item = {"prod_name": "Heavy Embroidered Lehenga", "detail_desc": ""}
        assert fabric_score_delta(
            item, "sangeet", formality_override="statement"
        ) == pytest.approx(0.1)


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


# ── occasion-driven anchor budget gate (live-proven bug) ─────────────────────
# "I'm pear-shaped, sangeet look under ₹8000" boarded a ₹9,900 lehenga as the
# ANCHOR — compose_outfit's occasion-driven `valid` comprehension gated
# occasion/gender/kids but never price, so `valid[0]` could pick an
# over-budget item before a single complement was even considered. Fixed in
# compose_outfit's occasion-driven branch only (see composer.py comment); the
# explicit seed_article_id path — a user's own "Style this <item>" choice —
# must never be budget-rejected (test c below).

_OVER_BUDGET_LEHENGA: dict = {
    "article_id": "ANCHOR_OVER",
    "prod_name": "Bridal Red Embellished Lehenga",
    "display_name": "Bridal Red Embellished Lehenga",
    "store": "myntra",
    "colour": "red",
    "product_type": "Lehenga",
    "detail_desc": "",
    "score": 0.95,
    "price_inr": 9900.0,
    "gender": "women",
}

_WITHIN_BUDGET_LEHENGA: dict = {
    "article_id": "ANCHOR_OK",
    "prod_name": "Maroon Embellished Lehenga",
    "display_name": "Maroon Embellished Lehenga",
    "store": "myntra",
    "colour": "maroon",
    "product_type": "Lehenga",
    "detail_desc": "",
    "score": 0.85,
    "price_inr": 6500.0,
    "gender": "women",
}


class _OccasionAnchorFakeRetriever:
    """Returns a fixed anchor candidate list (rank 0 first) for every query —
    a "Lehenga" (ethnic_one_piece) item never satisfies any complement slot's
    is_slot_type_allowed gate, so returning it unconditionally never
    accidentally fills a complement slot too; this isolates the test to
    anchor selection only.
    """

    def __init__(self, candidates: list[dict]) -> None:
        self._candidates = candidates

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None
    ) -> list[dict]:
        return list(self._candidates)


class TestOccasionDrivenAnchorBudgetGate:
    def test_over_budget_rank0_anchor_skipped_for_within_budget_rank1(self) -> None:
        """(a) rank-0 anchor is over budget, rank-1 is within — the chosen
        anchor must be the within-budget one and the board total must not
        exceed the budget."""
        retriever = _OccasionAnchorFakeRetriever(
            [_OVER_BUDGET_LEHENGA, _WITHIN_BUDGET_LEHENGA]
        )
        look = compose_outfit(
            pd.DataFrame(),
            retriever,
            seed_article_id=None,
            occasion_slug="sangeet",
            gender="women",
            budget_inr=8000,
        )
        assert look["seed_item"] is not None
        assert look["seed_item"]["article_id"] == "ANCHOR_OK"
        assert (look["budget_total_inr"] or 0) <= 8000

    def test_all_anchors_over_budget_returns_honest_empty_result(self) -> None:
        """(b) every occasion/gender-valid anchor is over budget — must return
        an empty result (no seed) whose message mentions the budget, never a
        silent fall-back to an over-budget anchor."""
        retriever = _OccasionAnchorFakeRetriever([_OVER_BUDGET_LEHENGA])
        look = compose_outfit(
            pd.DataFrame(),
            retriever,
            seed_article_id=None,
            occasion_slug="sangeet",
            gender="women",
            budget_inr=8000,
        )
        assert look["seed_item"] is None
        assert "8,000" in look["outfit_rationale"]
        assert "budget" in look["outfit_rationale"].lower() or "₹" in look["outfit_rationale"]

    def test_explicit_seed_article_id_over_budget_still_composes(self) -> None:
        """(c) an explicit seed_article_id (the user's own "Style this <item>"
        choice) must NEVER be budget-rejected — only complements are
        budget-squeezed, exactly as before this fix."""
        catalogue_df = pd.DataFrame(
            [
                {
                    "article_id": "USER_CHOSEN",
                    "prod_name": "Bridal Red Embellished Lehenga",
                    "display_name": "Bridal Red Embellished Lehenga",
                    "colour_group_name": "red",
                    "product_type_name": "Lehenga",
                    "department_name": "Women",
                    "index_group_name": "Ladieswear",
                    "detail_desc": "",
                    "image_url": None,
                    "price_inr": 9900.0,
                    "pdp_handle": "bridal-lehenga",
                    "store": "myntra",
                    "gender": "women",
                    "facets": {
                        "colour_group_name": "red",
                        "product_type_name": "Lehenga",
                        "department_name": "Women",
                    },
                }
            ]
        )
        retriever = _OccasionAnchorFakeRetriever([])  # no complement candidates; irrelevant here
        look = compose_outfit(
            catalogue_df,
            retriever,
            seed_article_id="USER_CHOSEN",
            occasion_slug="sangeet",
            gender="women",
            budget_inr=8000,
        )
        assert look["seed_item"] is not None
        assert look["seed_item"]["article_id"] == "USER_CHOSEN"
        assert look["seed_item"]["price_inr"] == 9900.0
