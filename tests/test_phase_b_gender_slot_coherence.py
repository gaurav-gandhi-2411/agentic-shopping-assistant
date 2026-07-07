"""Phase B Part 1 — single-look gender consistency + slot quality.

Covers the real-world bug: a saved women's look contained a women's rust dress
+ MEN'S cardigan + MEN'S formal shoes + a novelty "Luxury Piano Shape Handbag".

Sections:
  1. slots.classify_item / SLOT_ALLOWED_CLASSES / is_slot_type_allowed — the
     hard slot-type gate, incl. the live-proven "paperbag pants badged
     Accessory" bug and the more general co-ord-set name-vs-product_type bug.
  2. slots.accessory_query_matches — accessory family gating.
  3. slots.is_novelty_item — the literal "Luxury Piano Shape Handbag" test.
  4. slots.is_gender_neutral_accessory / is_western_marker_item.
  5. slots.resolve_look_gender — gender resolution precedence, incl. the
     image-upload owned-anchor "no gender in text" scenario.
  6. coherence.colour_score — rust + navy/cream harmony (muted-coordinating tier).
  7. composer._find_best_candidate — retrieval-time gender hard filter, the
     two-bottom-slots regression, gender-neutral-accessory fallback, honest
     suppression.
  8. composer.compose_outfit_variants — guaranteed-distinct variants.
  9. Offline real-index composition checks (requires_index) — printed evidence
     for an office look, a men's casual look, a sangeet look, and 3-variant
     pairwise-disjoint complements.

Uses data/processed/unified for the requires_index checks (per project
convention — see tests/test_occasion_outfit_and_refinement.py).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.agents.outfit.coherence import colour_score, is_coherent_candidate
from src.agents.outfit.composer import (
    _find_best_candidate,
    compose_outfit,
    compose_outfit_variants,
)
from src.agents.outfit.slots import (
    SLOT_ALLOWED_CLASSES,
    accessory_query_matches,
    classify_item,
    is_gender_neutral_accessory,
    is_kids_item,
    is_novelty_item,
    is_slot_type_allowed,
    is_western_marker_item,
    resolve_look_gender,
)

UNIFIED_DIR = Path("data/processed/unified")


# ---------------------------------------------------------------------------
# 1. classify_item / SLOT_ALLOWED_CLASSES / is_slot_type_allowed
# ---------------------------------------------------------------------------


class TestSlotAllowedClassesDisjoint:
    """Each slot's allowed classify_item() classes must be pairwise disjoint —
    this is what makes it structurally impossible for a bottom-classified item
    to also be a valid "accessory" candidate (the live-proven bug: pants badged
    "Accessory")."""

    def test_all_slot_classes_pairwise_disjoint(self) -> None:
        names = list(SLOT_ALLOWED_CLASSES.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = SLOT_ALLOWED_CLASSES[names[i]], SLOT_ALLOWED_CLASSES[names[j]]
                assert not (a & b), f"{names[i]} and {names[j]} share classes: {a & b}"

    def test_five_known_slots_present(self) -> None:
        assert set(SLOT_ALLOWED_CLASSES.keys()) == {
            "top", "bottom", "footwear", "outerwear", "accessory",
        }


class TestClassifyItemAccessoryWordBoundary:
    """Live-proven bug: "Stevie|100% Cotton Regular Paperbag Waist Pants"
    (product_type="trousers") must NEVER classify as "accessory" just because
    the freeform name contains the substring "bag" (inside "Paperbag")."""

    def test_paperbag_pants_classified_as_bottom_not_accessory(self) -> None:
        result = classify_item("trousers", "Stevie|100% Cotton Regular Paperbag Waist Pants")
        assert result == "western_bottom"

    def test_capri_not_misclassified_as_accessory_via_cap_substring(self) -> None:
        result = classify_item("shorts", "Women Capri Pants")
        assert result != "accessory"

    def test_real_bag_product_type_is_accessory(self) -> None:
        assert classify_item("bag", "Designer Sling Bag") == "accessory"

    def test_dupatta_is_accessory(self) -> None:
        assert classify_item("dupatta", "Silk Dupatta") == "accessory"

    def test_belt_watch_cap_are_accessory(self) -> None:
        assert classify_item("Belts", "Vegan Reversible Formal Belt") == "accessory"
        assert classify_item("Clothing Accessories", "Baseball Cap") == "accessory"


class TestClassifyItemTrustsProductTypeOverCoordSetName:
    """Real catalogue rows are often co-ord/bundle listings whose freeform NAME
    mentions OTHER garment parts too (e.g. a top sold as "Crop Top WITH
    PALAZZO"). classify_item must trust the authoritative product_type facet
    over incidental name-text keyword collisions — this generalises the
    "paperbag" bug beyond substring collisions to whole-word ones.
    """

    def test_top_product_type_stays_top_despite_palazzo_in_name(self) -> None:
        result = classify_item(
            "top", "Ethnic Floral Printed & Embroidered Crop Top with Palazzo - Green"
        )
        assert result == "western_top"

    def test_kurta_product_type_stays_ethnic_top_despite_trousers_in_name(self) -> None:
        result = classify_item(
            "kurta", "Khushal K Women Black Ethnic Motifs Printed Kurta with Palazzos & With Dupatta"
        )
        assert result == "ethnic_top"

    def test_generic_product_type_falls_back_to_name_scan(self) -> None:
        """When product_type itself is an uninformative catch-all ("Fashion"),
        classification still falls back to scanning the name text (unchanged
        legacy behaviour) — this is NOT a regression, just a lower-priority path."""
        assert classify_item("Fashion", "Cotton Kurta") == "ethnic_top"


class TestIsSlotTypeAllowed:
    """The hard slot-type gate: a candidate whose class isn't in the slot's
    allowed set is rejected before it can ever be scored."""

    def test_paperbag_pants_rejected_from_accessory_slot(self) -> None:
        assert is_slot_type_allowed(
            "accessory", "trousers", "Stevie|100% Cotton Regular Paperbag Waist Pants"
        ) is False

    def test_paperbag_pants_allowed_in_bottom_slot(self) -> None:
        assert is_slot_type_allowed(
            "bottom", "trousers", "Stevie|100% Cotton Regular Paperbag Waist Pants"
        ) is True

    def test_top_classified_coord_item_rejected_from_bottom_slot(self) -> None:
        """The two-bottom-slots regression: a top-typed co-ord listing whose
        name mentions "palazzo" must never fill the "bottom" slot alongside a
        real bottom."""
        assert is_slot_type_allowed(
            "bottom", "top", "Ethnic Floral Printed & Embroidered Crop Top with Palazzo - Green"
        ) is False

    def test_footwear_rejected_outside_footwear_slot(self) -> None:
        assert is_slot_type_allowed("accessory", "footwear", "White Sneakers") is False
        assert is_slot_type_allowed("footwear", "footwear", "White Sneakers") is True

    def test_unknown_slot_name_is_permissive(self) -> None:
        assert is_slot_type_allowed("not_a_real_slot", "trousers", "Jeans") is True


# ---------------------------------------------------------------------------
# 2. accessory_query_matches
# ---------------------------------------------------------------------------


class TestAccessoryQueryMatches:
    def test_dupatta_query_rejects_handbag(self) -> None:
        assert accessory_query_matches("dupatta ethnic dupatta", "bag", "Sling Bag") is False

    def test_dupatta_query_accepts_dupatta(self) -> None:
        assert accessory_query_matches("dupatta ethnic dupatta", "dupatta", "Silk Dupatta") is True

    def test_bag_query_rejects_dupatta(self) -> None:
        assert accessory_query_matches(
            "handbag sling bag earrings women accessory", "dupatta", "Silk Dupatta"
        ) is False

    def test_belt_watch_cap_query_rejects_dupatta(self) -> None:
        assert accessory_query_matches(
            "belt watch cap men accessory", "dupatta", "Silk Dupatta"
        ) is False

    def test_unrecognised_query_is_permissive(self) -> None:
        assert accessory_query_matches("some unrelated query text", "bag", "Sling Bag") is True


# ---------------------------------------------------------------------------
# 3. is_novelty_item — literal piano-handbag rejection
# ---------------------------------------------------------------------------


class TestIsNoveltyItem:
    def test_luxury_piano_shape_handbag_is_rejected(self) -> None:
        assert is_novelty_item("Luxury Piano Shape Handbag") is True

    def test_dachshund_crossbody_bag_without_literal_shape_word_is_rejected(self) -> None:
        assert is_novelty_item("Designer Dachshund Crossbody Bag") is True

    def test_costume_and_cosplay_rejected(self) -> None:
        assert is_novelty_item("Halloween Costume Dress") is True
        assert is_novelty_item("Anime Cosplay Outfit") is True

    def test_v_shape_waist_jegging_not_rejected(self) -> None:
        """No object/instrument word present — must NOT false-positive."""
        assert is_novelty_item("Black V-shape Waist Cotton Blend Bootcut Jegging") is False

    def test_novelty_town_brand_name_not_rejected(self) -> None:
        """"novelty" is not itself a denylist word — the real brand "Novelty
        Town" (an oversized-tee brand) must not be caught."""
        assert is_novelty_item("Novelty Town Lavender Oversized T-Shirt") is False

    def test_football_shoes_not_rejected(self) -> None:
        """"football" alone (no bag-family word) must not reject legitimate
        football shoes/shorts."""
        assert is_novelty_item("Football Shoes For Men (Green, Blue)") is False

    def test_football_shaped_handbag_is_rejected(self) -> None:
        assert is_novelty_item("Football Shaped Handbag With Zip & Handle") is True

    def test_flamingo_pink_shirt_not_rejected(self) -> None:
        """"flamingo" as a colour name ("Flamingo Pink") must not be caught."""
        assert is_novelty_item("Lavish Flamingo Pink 100% Linen Shirt") is False

    def test_plain_garment_not_rejected(self) -> None:
        assert is_novelty_item("Black Cotton Trousers") is False


class TestIsKidsItem:
    """S5 fix: juniors/girls/boys/kids garments mislabeled as adult inventory
    (catalogue gender column derives from index_group_name, which buckets
    juniors/kids SKUs under "Ladieswear"/"Menswear" alongside real adult
    items) — live-proven bug: a "M&H Juniors Girls Blue Straight Knee Length
    Denim Skirts" item (gender="women") filled an office look's bottom slot.
    """

    def test_juniors_girls_denim_skirt_rejected(self) -> None:
        assert is_kids_item("M&H Juniors Girls Blue Straight Knee Length Denim Skirts") is True

    def test_juniors_kids_top_rejected(self) -> None:
        assert is_kids_item("Juniors by Lifestyle Kids-Girls White Pure Cotton Print Top") is True

    def test_boys_item_rejected(self) -> None:
        assert is_kids_item("Boys Blue Cotton Shirt") is True

    def test_kids_item_rejected(self) -> None:
        assert is_kids_item("Kids Party Wear Dress") is True

    def test_plain_adult_garment_not_rejected(self) -> None:
        assert is_kids_item("Women Black Solid A-Line Dress") is False

    def test_empty_name_not_rejected(self) -> None:
        assert is_kids_item("") is False
        assert is_kids_item(None) is False


class TestFindBestCandidateKidsItemRejected:
    """Mirrors TestFindBestCandidatePianoHandbagRejected — a juniors/girls item
    must never win a slot even when it's the only candidate returned and
    passes the gender filter (the exact live scenario: the catalogue's own
    gender column already says "women")."""

    def test_juniors_denim_skirt_never_selected(self) -> None:
        juniors_skirt = {
            "article_id": "JUNIOR1",
            "prod_name": "M&H Juniors Girls Blue Straight Knee Length Denim Skirts",
            "display_name": "M&H Juniors Girls Blue Straight Knee Length Denim Skirts",
            "product_type": "skirt",
            "colour": "blue",
            "gender": "women",
            "score": 0.99,
            "price_inr": 599.0,
            "store": "myntra",
            "detail_desc": "",
        }
        retriever = _FilterRecordingRetriever([juniors_skirt])
        winner = _find_best_candidate(
            query="trousers formal tailored",
            slot_name="bottom",
            occasion_slug="office",
            gender="women",
            anchor_colour="black",
            seen_ids=set(),
            seen_prod_colour=set(),
            retriever=retriever,
            budget_remaining=None,
            pairing_stats=None,
            anchor_class="western_top",
        )
        assert winner is None


# ---------------------------------------------------------------------------
# 4. gender-neutral accessory / western-marker
# ---------------------------------------------------------------------------


class TestIsGenderNeutralAccessory:
    def test_sunglasses_are_gender_neutral(self) -> None:
        assert is_gender_neutral_accessory("Sunglasses", "Aviator Sunglasses") is True

    def test_belt_and_watch_are_gender_neutral(self) -> None:
        assert is_gender_neutral_accessory("Belts", "Leather Belt") is True
        assert is_gender_neutral_accessory("Watches", "Analog Watch") is True

    def test_dupatta_is_not_gender_neutral(self) -> None:
        assert is_gender_neutral_accessory("dupatta", "Silk Dupatta") is False

    def test_handbag_is_not_gender_neutral(self) -> None:
        assert is_gender_neutral_accessory("bag", "Sling Bag") is False


class TestIsWesternMarkerItem:
    """Extends is_western_item so footwear/outerwear/unknown-class items with
    an explicit western marker are caught by ethnic-occasion gates."""

    def test_sneakers_flagged_western(self) -> None:
        assert is_western_marker_item("footwear", "White Sneakers") is True

    def test_denim_jacket_flagged_western(self) -> None:
        assert is_western_marker_item("outerwear", "Blue Denim Jacket") is True

    def test_hoodie_bomber_tshirt_flagged_western(self) -> None:
        assert is_western_marker_item("outerwear", "Black Bomber Jacket") is True
        assert is_western_marker_item("Clothing Accessories", "Grey Hoodie") is True
        assert is_western_marker_item("top", "Plain T-Shirt") is True

    def test_juttis_not_flagged_western(self) -> None:
        assert is_western_marker_item("footwear", "Embroidered Juttis") is False

    def test_ethnic_only_occasion_rejects_sneakers_via_coherence_gate(self) -> None:
        item = {"product_type": "footwear", "prod_name": "White Sneakers", "gender": "women"}
        assert is_coherent_candidate(item, "sangeet", "women", "footwear") is False

    def test_ethnic_heavy_occasion_rejects_denim_jacket_via_coherence_gate(self) -> None:
        item = {"product_type": "outerwear", "prod_name": "Blue Denim Jacket", "gender": "women"}
        assert is_coherent_candidate(item, "festive_puja", "women", "outerwear") is False


# ---------------------------------------------------------------------------
# 5. resolve_look_gender
# ---------------------------------------------------------------------------


def _gender_catalogue(article_id: str, gender: str) -> pd.DataFrame:
    return pd.DataFrame([{"article_id": article_id, "gender": gender}])


class TestResolveLookGender:
    def test_intent_gender_wins_over_everything(self) -> None:
        df = _gender_catalogue("A1", "women")
        result = resolve_look_gender(
            intent_gender="men", session_gender="women", catalogue_df=df,
            anchor_id="A1", brand_gender_default="women",
        )
        assert result == "men"

    def test_session_gender_used_when_no_intent(self) -> None:
        df = _gender_catalogue("A1", "women")
        result = resolve_look_gender(
            intent_gender=None, session_gender="men", catalogue_df=df,
            anchor_id="A1", brand_gender_default="women",
        )
        assert result == "men"

    def test_anchor_gender_used_when_no_intent_or_session(self) -> None:
        """The image-upload owned-anchor scenario: user uploads a photo with no
        text at all — gender must come from the ANCHOR's own gender column,
        not the brand default."""
        df = _gender_catalogue("MENS_SHIRT", "men")
        result = resolve_look_gender(
            intent_gender=None, session_gender=None, catalogue_df=df,
            anchor_id="MENS_SHIRT", brand_gender_default="women",
        )
        assert result == "men"

    def test_brand_default_used_when_anchor_gender_unknown(self) -> None:
        """Conservative fallback: an anchor with gender="unknown" (or missing)
        must NOT be guessed — falls through to the brand default rather than
        inventing a per-item gender."""
        df = _gender_catalogue("UNKNOWN_ITEM", "unknown")
        result = resolve_look_gender(
            intent_gender=None, session_gender=None, catalogue_df=df,
            anchor_id="UNKNOWN_ITEM", brand_gender_default="women",
        )
        assert result == "women"

    def test_brand_default_used_when_anchor_id_is_none(self) -> None:
        df = _gender_catalogue("A1", "men")
        result = resolve_look_gender(
            intent_gender=None, session_gender=None, catalogue_df=df,
            anchor_id=None, brand_gender_default="men",
        )
        assert result == "men"

    def test_mixed_brand_default_coerces_to_women(self) -> None:
        df = _gender_catalogue("A1", "unknown")
        result = resolve_look_gender(
            intent_gender=None, session_gender=None, catalogue_df=df,
            anchor_id="A1", brand_gender_default="mixed",
        )
        assert result == "women"

    def test_never_returns_unknown(self) -> None:
        df = _gender_catalogue("A1", "unknown")
        result = resolve_look_gender(
            intent_gender=None, session_gender=None, catalogue_df=df,
            anchor_id="A1", brand_gender_default="unknown",
        )
        assert result in ("men", "women")


# ---------------------------------------------------------------------------
# 6. colour_score — muted-coordinating tier (rust + navy/cream)
# ---------------------------------------------------------------------------


class TestColourScoreMutedCoordinating:
    def test_rust_and_navy_blue_score_as_harmony(self) -> None:
        score = colour_score("navy blue", "rust", "casual")
        assert score >= 0.6, f"expected harmony, got clash-range score {score}"

    def test_rust_and_cream_score_as_harmony(self) -> None:
        score = colour_score("cream", "rust", "office")
        assert score >= 0.6

    def test_primary_hue_clash_is_unaffected(self) -> None:
        """Regression guard: red vs blue must still clash in a western context —
        the muted-coordinating tier must NOT swallow this existing behaviour."""
        assert colour_score("red", "blue", "casual") == pytest.approx(0.4)

    def test_khaki_charcoal_are_neutral(self) -> None:
        assert colour_score("khaki", "rust", "casual") == pytest.approx(1.0)
        assert colour_score("charcoal", "mustard", "office") == pytest.approx(1.0)

    def test_mustard_and_olive_score_as_harmony(self) -> None:
        assert colour_score("olive", "mustard", "casual") >= 0.6


# ---------------------------------------------------------------------------
# 7. composer._find_best_candidate — gender hard filter, slot-type gate,
#    gender-neutral fallback, honest suppression
# ---------------------------------------------------------------------------


class _FilterRecordingRetriever:
    """Records the (query, top_k, filters) of every .search() call; returns a
    caller-supplied fixed candidate list regardless of query."""

    def __init__(self, items: list[dict]) -> None:
        self._items = items
        self.calls: list[dict] = []

    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return list(self._items)


def _bottom_item(article_id: str = "BOTTOM1", gender: str = "women") -> dict:
    return {
        "article_id": article_id,
        "prod_name": "Stevie|100% Cotton Regular Paperbag Waist Pants",
        "display_name": "Stevie Paperbag Waist Pants",
        "product_type": "trousers",
        "colour": "black",
        "gender": gender,
        "score": 0.9,
        "price_inr": 999.0,
        "store": "myntra",
        "detail_desc": "",
    }


class TestFindBestCandidateGenderHardFilter:
    def test_gender_filter_is_passed_to_retriever(self) -> None:
        retriever = _FilterRecordingRetriever([_bottom_item()])
        _find_best_candidate(
            query="trousers jeans skirt",
            slot_name="bottom",
            occasion_slug="casual",
            gender="women",
            anchor_colour="black",
            seen_ids=set(),
            seen_prod_colour=set(),
            retriever=retriever,
            budget_remaining=None,
            pairing_stats=None,
            anchor_class="western_top",
        )
        assert retriever.calls[0]["filters"] == {"gender": "women"}
        assert retriever.calls[0]["top_k"] == 40


class TestFindBestCandidateTwoBottomsRegression:
    """Live-proven bug: the accessory slot's "bag handbag" query text-matched a
    pair of trousers ("Paperbag Waist Pants") via retrieval, and — with no
    slot-type gate — got badged "Accessory". Must now be impossible."""

    def test_paperbag_pants_never_wins_the_accessory_slot(self) -> None:
        retriever = _FilterRecordingRetriever([_bottom_item(article_id="ACCBUG")])
        winner = _find_best_candidate(
            query="handbag sling bag earrings women accessory",
            slot_name="accessory",
            occasion_slug="casual",
            gender="women",
            anchor_colour="black",
            seen_ids=set(),
            seen_prod_colour=set(),
            retriever=retriever,
            budget_remaining=None,
            pairing_stats=None,
            anchor_class="western_one_piece",
        )
        assert winner is None, "a bottom-classified item must never win an accessory slot"

    def test_paperbag_pants_still_wins_the_bottom_slot(self) -> None:
        retriever = _FilterRecordingRetriever([_bottom_item(article_id="OK1")])
        winner = _find_best_candidate(
            query="trousers jeans skirt",
            slot_name="bottom",
            occasion_slug="casual",
            gender="women",
            anchor_colour="black",
            seen_ids=set(),
            seen_prod_colour=set(),
            retriever=retriever,
            budget_remaining=None,
            pairing_stats=None,
            anchor_class="western_top",
        )
        assert winner is not None
        assert winner["article_id"] == "OK1"


class TestFindBestCandidatePianoHandbagRejected:
    def test_piano_handbag_never_selected(self) -> None:
        piano_bag = {
            "article_id": "PIANO1",
            "prod_name": "Luxury Piano Shape Handbag",
            "display_name": "Luxury Piano Shape Handbag",
            "product_type": "bag",
            "colour": "black",
            "gender": "women",
            "score": 0.99,
            "price_inr": 999.0,
            "store": "myntra",
            "detail_desc": "",
        }
        retriever = _FilterRecordingRetriever([piano_bag])
        winner = _find_best_candidate(
            query="handbag sling bag earrings women accessory",
            slot_name="accessory",
            occasion_slug="casual",
            gender="women",
            anchor_colour="black",
            seen_ids=set(),
            seen_prod_colour=set(),
            retriever=retriever,
            budget_remaining=None,
            pairing_stats=None,
            anchor_class="western_one_piece",
        )
        assert winner is None


class TestFindBestCandidateGenderNeutralFallback:
    def test_unknown_gender_sunglasses_fill_empty_accessory_slot(self) -> None:
        """Gendered search returns nothing (simulated by an empty first list);
        the narrow gender-neutral-accessory fallback should surface unisex
        sunglasses rather than suppressing the slot."""

        class _EmptyThenSunglassesRetriever:
            def __init__(self) -> None:
                self.calls = 0

            def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
                self.calls += 1
                if filters:
                    return []  # gendered search: nothing found
                return [
                    {
                        "article_id": "SUN1",
                        "prod_name": "Aviator Sunglasses",
                        "display_name": "Aviator Sunglasses",
                        "product_type": "Sunglasses",
                        "colour": "black",
                        "gender": "unknown",
                        "score": 0.7,
                        "price_inr": 499.0,
                        "store": "myntra",
                        "detail_desc": "",
                    }
                ]

        retriever = _EmptyThenSunglassesRetriever()
        winner = _find_best_candidate(
            query="belt watch cap men accessory",
            slot_name="accessory",
            occasion_slug="casual",
            gender="men",
            anchor_colour="black",
            seen_ids=set(),
            seen_prod_colour=set(),
            retriever=retriever,
            budget_remaining=None,
            pairing_stats=None,
            anchor_class="western_top",
        )
        assert winner is not None
        assert winner["article_id"] == "SUN1"

    def test_unknown_gender_garment_never_fills_footwear_via_fallback(self) -> None:
        """The gender-neutral fallback is accessory-only — an empty footwear
        slot must stay empty (suppressed), never filled with an unknown-gender
        shoe."""

        class _EmptyThenUnknownShoeRetriever:
            def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
                if filters:
                    return []
                return [
                    {
                        "article_id": "SHOE1",
                        "prod_name": "Unisex Sneakers",
                        "display_name": "Unisex Sneakers",
                        "product_type": "footwear",
                        "colour": "white",
                        "gender": "unknown",
                        "score": 0.9,
                        "price_inr": 999.0,
                        "store": "myntra",
                        "detail_desc": "",
                    }
                ]

        retriever = _EmptyThenUnknownShoeRetriever()
        winner = _find_best_candidate(
            query="sneakers flats heels casual shoes women",
            slot_name="footwear",
            occasion_slug="casual",
            gender="women",
            anchor_colour="black",
            seen_ids=set(),
            seen_prod_colour=set(),
            retriever=retriever,
            budget_remaining=None,
            pairing_stats=None,
            anchor_class="western_top",
        )
        assert winner is None


# ---------------------------------------------------------------------------
# 7b. compose_outfit — honest slot suppression payload
# ---------------------------------------------------------------------------


def _seed_catalogue_row(article_id: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "article_id": article_id,
        "prod_name": "Rust Dress",
        "display_name": "Rust Dress",
        "colour_group_name": "rust",
        "product_type_name": "dress",
        "department_name": "Women",
        "index_group_name": "Ladieswear",
        "detail_desc": "",
        "image_url": None,
        "price_inr": 1999.0,
        "pdp_handle": "rust-dress",
        "store": "myntra",
        "gender": "women",
        "facets": {
            "colour_group_name": "rust", "product_type_name": "dress", "department_name": "Women",
        },
    }])


class _NoFootwearNoAccessoryRetriever:
    """Returns a blazer for the outerwear query only; nothing for footwear or
    accessory queries — models the real "women's footwear ~= 1 item" gap."""

    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
        if "jacket" in query or "blazer" in query or "cardigan" in query:
            return [{
                "article_id": "OUTER1",
                "prod_name": "Black Blazer",
                "display_name": "Black Blazer",
                "product_type": "blazer",
                "colour": "black",
                "gender": "women",
                "score": 0.8,
                "price_inr": 1500.0,
                "store": "myntra",
                "detail_desc": "",
            }]
        return []


class TestSuppressedSlotsPayload:
    def test_empty_slots_recorded_as_suppressed_with_honest_reason(self) -> None:
        catalogue_df = _seed_catalogue_row("DRESS1")
        look = compose_outfit(
            catalogue_df, _NoFootwearNoAccessoryRetriever(),
            seed_article_id="DRESS1", occasion_slug="casual", gender="women",
        )
        suppressed = {s["slot"]: s["reason"] for s in look["suppressed_slots"]}
        assert "footwear" in suppressed
        assert "accessory" in suppressed
        assert suppressed["footwear"] == "No women's footwear in our partner stores yet"
        # outerwear WAS filled — must not appear in suppressed_slots
        assert "outerwear" not in suppressed
        # required-slot failure degrades to a suppression note, not an abort
        assert look["seed_item"] is not None
        assert any(c["_slot"] == "outerwear" for c in look["complements"])

    def test_empty_result_still_carries_suppressed_slots_key(self) -> None:
        catalogue_df = _seed_catalogue_row("DRESS2")
        look = compose_outfit(
            catalogue_df, _NoFootwearNoAccessoryRetriever(),
            seed_article_id="NOT_A_REAL_ID", occasion_slug="casual", gender="women",
        )
        assert look["suppressed_slots"] == []


# ---------------------------------------------------------------------------
# 8. compose_outfit_variants — guaranteed-distinct variants
# ---------------------------------------------------------------------------


class _ManyBottomsRetriever:
    """Returns 5 distinct bottom candidates (varying colour) for any bottom-ish
    query, and nothing for outerwear/footwear/accessory — isolates the
    guaranteed-distinct-variant behaviour to a single slot."""

    _COLOURS = ["black", "navy blue", "rust", "olive", "mustard"]

    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
        if "trousers" in query or "jeans" in query or "skirt" in query:
            return [
                {
                    "article_id": f"BTM{i}",
                    "prod_name": f"Trousers {i}",
                    "display_name": f"Trousers {i}",
                    "product_type": "trousers",
                    "colour": colour,
                    "gender": "women",
                    "score": 0.9 - i * 0.01,
                    "price_inr": 999.0,
                    "store": "myntra",
                    "detail_desc": "",
                }
                for i, colour in enumerate(self._COLOURS)
            ]
        return []


def _seed_top_catalogue_row(article_id: str) -> pd.DataFrame:
    """A western_top-anchor catalogue row (unlike _seed_catalogue_row's dress,
    a top anchor gets a "bottom" slot in get_fill_slots)."""
    return pd.DataFrame([{
        "article_id": article_id,
        "prod_name": "Rust Top",
        "display_name": "Rust Top",
        "colour_group_name": "rust",
        "product_type_name": "top",
        "department_name": "Women",
        "index_group_name": "Ladieswear",
        "detail_desc": "",
        "image_url": None,
        "price_inr": 799.0,
        "pdp_handle": "rust-top",
        "store": "myntra",
        "gender": "women",
        "facets": {
            "colour_group_name": "rust", "product_type_name": "top", "department_name": "Women",
        },
    }])


class TestGuaranteedDistinctVariants:
    def test_fixed_variants_hard_exclude_base_complements(self) -> None:
        catalogue_df = _seed_top_catalogue_row("SEEDTOP")
        retriever = _ManyBottomsRetriever()
        variants = compose_outfit_variants(
            catalogue_df, retriever,
            seed_article_id="SEEDTOP", occasion_slug="casual", gender="women",
        )
        assert len(variants) >= 2, "expected at least a base + one alternate variant"
        base_ids = {c["article_id"] for c in variants[0]["complements"]}
        for variant in variants[1:]:
            variant_ids = {c["article_id"] for c in variant["complements"]}
            assert not (variant_ids & base_ids), (
                f"variant {variant.get('variant_label')} must hard-exclude the base "
                f"look's complement ids; base={base_ids} variant={variant_ids}"
            )


# ---------------------------------------------------------------------------
# 9. Offline real-index composition checks (requires_index)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _unified_index() -> tuple:
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    config = {
        "retrieval": {
            "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
            "dense_dim": 384,
            "rrf_k": 60,
            "top_k": 50,
            "final_k": 10,
            "store_diversity": 0.2,
        },
    }
    dense = DenseRetriever.load(config, UNIFIED_DIR)
    sparse = SparseRetriever.load(config, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)
    return retriever, catalogue_df


def _find_article_id(catalogue_df: pd.DataFrame, gender: str, product_type: str, colour: str | None = None) -> str:
    def _pt(f: object) -> str | None:
        return f.get("product_type_name") if isinstance(f, dict) else None

    def _col(f: object) -> str | None:
        return f.get("colour_group_name") if isinstance(f, dict) else None

    pt_series = catalogue_df["facets"].apply(_pt)
    mask = (catalogue_df["gender"] == gender) & (pt_series.str.lower() == product_type.lower())
    if colour is not None:
        col_series = catalogue_df["facets"].apply(_col)
        mask &= col_series.str.lower() == colour.lower()
    matches = catalogue_df.loc[mask, "article_id"]
    assert not matches.empty, f"no fixture item found for gender={gender} pt={product_type} colour={colour}"
    return str(matches.iloc[0])


@pytest.mark.requires_index
class TestOfflineRealIndexCompositionEvidence:
    """Prints the actual composed looks as evidence per the VERIFY spec."""

    def test_office_look_for_women_black_top_anchor(self, _unified_index: tuple) -> None:
        retriever, catalogue_df = _unified_index
        anchor_id = _find_article_id(catalogue_df, "women", "top", "Black")
        look = compose_outfit(
            catalogue_df, retriever,
            seed_article_id=anchor_id, occasion_slug="office", gender="women",
        )
        print("\n=== office look for women (black top anchor) ===")
        print("seed:", look["seed_item"]["prod_name"], look["seed_item"]["gender"])
        for c in look["complements"]:
            print(" complement:", c["_slot"], "|", c["prod_name"], "|", c["product_type"], "|", c["gender"])
        print("suppressed_slots:", look["suppressed_slots"])

        assert look["seed_item"] is not None
        for c in look["complements"]:
            assert c["gender"] == "women", f"non-women complement leaked in: {c}"
            # S5 fix: live-proven bug — a "M&H Juniors Girls ... Denim Skirts"
            # item (catalogue gender="women") filled an office look's bottom
            # slot. No complement in ANY slot should carry a juniors/kids marker.
            assert not is_kids_item(c.get("prod_name") or ""), (
                f"juniors/kids item leaked into an adult office look: {c}"
            )
        bottom = next((c for c in look["complements"] if c["_slot"] == "bottom"), None)
        if bottom is not None:
            pt_name = (bottom.get("product_type") or "").lower()
            name = (bottom.get("prod_name") or "").lower()
            assert "skirt" not in pt_name and "skirt" not in name, (
                f"office bottom must be trousers-like, not a skirt: {bottom}"
            )
        footwear = next((c for c in look["complements"] if c["_slot"] == "footwear"), None)
        if footwear is None:
            suppressed = {s["slot"] for s in look["suppressed_slots"]}
            assert "footwear" in suppressed, "empty footwear slot must be recorded as suppressed"

    def test_mens_casual_look_all_complements_men(self, _unified_index: tuple) -> None:
        retriever, catalogue_df = _unified_index
        anchor_id = _find_article_id(catalogue_df, "men", "shirt")
        look = compose_outfit(
            catalogue_df, retriever,
            seed_article_id=anchor_id, occasion_slug="casual", gender="men",
        )
        print("\n=== men's casual look (shirt anchor) ===")
        print("seed:", look["seed_item"]["prod_name"], look["seed_item"]["gender"])
        for c in look["complements"]:
            print(" complement:", c["_slot"], "|", c["prod_name"], "|", c["product_type"], "|", c["gender"])

        assert look["seed_item"] is not None
        assert look["complements"], "expected at least one complement"
        for c in look["complements"]:
            assert c["gender"] == "men", f"non-men complement leaked in: {c}"

    def test_sangeet_look_no_sneaker_or_denim(self, _unified_index: tuple) -> None:
        retriever, catalogue_df = _unified_index
        anchor_id = _find_article_id(catalogue_df, "women", "kurta")
        look = compose_outfit(
            catalogue_df, retriever,
            seed_article_id=anchor_id, occasion_slug="sangeet", gender="women",
        )
        print("\n=== sangeet look (kurta anchor) ===")
        print("seed:", look["seed_item"]["prod_name"], look["seed_item"]["gender"])
        for c in look["complements"]:
            print(" complement:", c["_slot"], "|", c["prod_name"], "|", c["product_type"], "|", c["gender"])

        assert look["seed_item"] is not None
        for c in look["complements"]:
            text = ((c.get("prod_name") or "") + " " + (c.get("product_type") or "")).lower()
            assert "sneaker" not in text and "denim" not in text, f"western marker leaked: {c}"
            assert c["gender"] == "women"

    def test_three_variants_pairwise_disjoint_complements(self, _unified_index: tuple) -> None:
        """S6 fix: EVERY pair of variants (not just base-vs-alternate) must have
        disjoint complement id sets.

        Live-proven regression: for this exact anchor, "Colour story" and
        "Dressier" both independently excluded only the BASE look's complement
        ids, so when the alternate-colour candidate pool for a slot had just
        ONE non-base option, both biased variants converged on the identical
        pair of items ("Van Heusen ... Navy Blue ... Blazer" + "White Sculpted
        Cat Crossbody Bag") — a duplicate `_is_distinct_look(variant, base)`
        never caught because it only ever compared each variant to the base,
        never to each other. This test previously only asserted variant-i !=
        base (i==0), which is exactly the gap that let the live duplicate
        through; it now asserts full pairwise disjointness.
        """
        retriever, catalogue_df = _unified_index
        anchor_id = _find_article_id(catalogue_df, "women", "top", "Black")
        variants = compose_outfit_variants(
            catalogue_df, retriever,
            seed_article_id=anchor_id, occasion_slug="casual", gender="women",
        )
        print(f"\n=== variants ({len(variants)}) ===")
        id_sets = []
        for v in variants:
            ids = {c["article_id"] for c in v.get("complements", [])}
            print(v.get("variant_label"), sorted(ids))
            id_sets.append(ids)

        assert len(variants) >= 1
        for i in range(len(id_sets)):
            for j in range(i + 1, len(id_sets)):
                overlap = id_sets[i] & id_sets[j]
                assert not overlap, (
                    f"variant {i} ({variants[i].get('variant_label')}) and variant {j} "
                    f"({variants[j].get('variant_label')}) must have disjoint complement "
                    f"ids, overlap={overlap}"
                )
