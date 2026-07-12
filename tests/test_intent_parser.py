"""
Comprehensive unit tests for src/agents/intent_parser.py (F3 IntentParser).

Groups:
  1. Gender phrasings
  2. Garment synonyms
  3. Colour + garment combos
  4. Refinement-only inputs (all must be is_product_query=True)
  5. Product vs conversational routing
  6. Compound garment rules
  7. Budget extraction
  8. merge_with_context
"""

from __future__ import annotations

import pytest

from src.agents.intent_parser import merge_with_context, parse_intent

# ---------------------------------------------------------------------------
# Group 1: Gender phrasings
# ---------------------------------------------------------------------------


class TestGenderPhrasings:
    @pytest.mark.parametrize(
        "query, expected_gender",
        [
            ("black dress for my wife", "women"),
            ("show me something for my husband", "men"),
            ("for my girlfriend", "women"),
            ("for him", "men"),
            ("for her", "women"),
            ("women's jacket", "women"),
            ("mens shirt", "men"),
            ("ladies kurti", "women"),
            ("for my mom", "women"),
            ("for my dad", "men"),
            ("for my daughter", "women"),
            ("for my son", "men"),
            ("for my brother", "men"),
            ("female formal wear", "women"),
            ("he needs a blazer", "men"),
        ],
    )
    def test_gender(self, query: str, expected_gender: str) -> None:
        intent = parse_intent(query)
        assert intent.gender == expected_gender, (
            f"query={query!r}: expected gender={expected_gender!r}, got {intent.gender!r}"
        )

    def test_no_gender_signal(self) -> None:
        intent = parse_intent("show me a blue dress")
        assert intent.gender is None

    def test_women_not_confused_with_men_substring(self) -> None:
        # "women" contains "men" — must return women, not men
        intent = parse_intent("women's casual top")
        assert intent.gender == "women"


class TestExplicitGenderBeatsGarmentHeuristic:
    """2026-07-11 live-proof-caught bug: "kurta" is unisex in this catalogue
    (it stocks men's kurtas — see gold_020/gold_028 in the strict gold set),
    so an explicit gender word must always win over any garment-implied
    gender guess. Production was resolving "printed kurta for men" to
    gender=women because the old _GENDER_MAP order checked "kurta" as a
    women-marker before the explicit "men" rule."""

    @pytest.mark.parametrize(
        "query, expected_gender",
        [
            ("printed kurta for men", "men"),
            ("white straight fit kurta for men under 2000", "men"),
            ("men's kurta for a reception", "men"),
            ("kurta for men", "men"),
            ("kurta for women", "women"),
        ],
    )
    def test_explicit_gender_word_wins_over_kurta(
        self, query: str, expected_gender: str
    ) -> None:
        intent = parse_intent(query)
        assert intent.gender == expected_gender, (
            f"query={query!r}: expected gender={expected_gender!r}, got {intent.gender!r}"
        )

    def test_kurta_alone_has_no_gender_signal(self) -> None:
        # No explicit gender word and no other gender-implying garment — the
        # brand-default fallback (graph.py) handles this, not a garment guess.
        intent = parse_intent("bright yellow kurta for haldi")
        assert intent.gender is None

    def test_kurti_still_implies_women(self) -> None:
        # "kurti" (unlike "kurta") is a genuinely women-specific term.
        intent = parse_intent("kurti under 1500")
        assert intent.gender == "women"

    def test_sherwani_still_implies_men(self) -> None:
        intent = parse_intent("sherwani for groom")
        assert intent.gender == "men"


# ---------------------------------------------------------------------------
# Group 2: Garment synonyms
# ---------------------------------------------------------------------------


class TestGarmentSynonyms:
    @pytest.mark.parametrize(
        "query, expected_garment",
        [
            ("t-shirt", "top"),
            ("tee", "top"),
            ("tshirt", "top"),
            ("blouse", "blouse"),
            ("skirt", "skirt"),
            ("shorts", "shorts"),
            ("jeans", "jeans"),
            ("kurti for women", "kurti"),
            ("blazer", "blazer"),
            ("bomber jacket", "outerwear"),
            ("puffer jacket", "outerwear"),
            ("sweatshirt", "knitwear"),
            ("hoodie", "knitwear"),
            ("cardigan", "knitwear"),
            ("jumpsuit", "jumpsuit"),
            ("saree", "saree"),
            ("lehenga", "lehenga"),
            ("palazzo pants", "palazzo"),
            ("trousers", "trousers"),
            ("chinos", "trousers"),
        ],
    )
    def test_garment_synonym(self, query: str, expected_garment: str) -> None:
        intent = parse_intent(query)
        assert intent.garment_type == expected_garment, (
            f"query={query!r}: expected garment={expected_garment!r}, got {intent.garment_type!r}"
        )

    def test_no_garment_conversational(self) -> None:
        intent = parse_intent("how are you doing today")
        assert intent.garment_type is None


# ---------------------------------------------------------------------------
# Group 3: Colour + garment combos
# ---------------------------------------------------------------------------


class TestColourGarmentCombos:
    @pytest.mark.parametrize(
        "query, expected_colour, expected_garment",
        [
            ("black dress", "Black", "dress"),
            ("blue jeans", "Blue", "jeans"),
            ("red kurti", "Red", "kurti"),
            ("dark blue blazer", "Dark Blue", "blazer"),
            ("white shirt", "White", "shirt"),
            ("navy dress", "Navy Blue", "dress"),
            ("grey trousers", "Grey", "trousers"),
            ("pink blouse", "Pink", "blouse"),
            ("green top", "Green", "top"),
            ("navy blue jeans", "Navy Blue", "jeans"),
        ],
    )
    def test_colour_and_garment(
        self, query: str, expected_colour: str, expected_garment: str
    ) -> None:
        intent = parse_intent(query)
        assert intent.colour == expected_colour, (
            f"query={query!r}: expected colour={expected_colour!r}, got {intent.colour!r}"
        )
        assert intent.garment_type == expected_garment, (
            f"query={query!r}: expected garment={expected_garment!r}, got {intent.garment_type!r}"
        )

    def test_dark_blue_beats_blue(self) -> None:
        """Longer phrase must win over shorter substring."""
        intent = parse_intent("dark blue kurta")
        assert intent.colour == "Dark Blue"

    def test_navy_canonical(self) -> None:
        intent = parse_intent("navy jacket")
        assert intent.colour == "Navy Blue"


# ---------------------------------------------------------------------------
# Group 4: Refinement-only inputs — all must be is_product_query=True
# ---------------------------------------------------------------------------


class TestRefinementInputs:
    @pytest.mark.parametrize(
        "query",
        [
            "in blue",
            "in red",
            "cheaper",
            "more formal",
            "more casual",
            "different colour",
            "something similar",
            "show me more",
            "in black",
            "different color",
            "change colour",
            "like these",
        ],
    )
    def test_refinement_is_product_query(self, query: str) -> None:
        intent = parse_intent(query)
        assert intent.is_product_query is True, (
            f"query={query!r}: expected is_product_query=True, got False"
        )


# ---------------------------------------------------------------------------
# Group 5: Product vs conversational routing
# ---------------------------------------------------------------------------


class TestProductVsConversational:
    @pytest.mark.parametrize(
        "query, expected",
        [
            # Product — True
            ("can you help me buy a similar t-shirt", True),
            ("show me black dresses", True),
            ("I need a jacket", True),
            ("looking for something casual", True),
            ("find me blue jeans under 2000", True),
            ("recommend a kurti for wedding", True),
            ("what blazers do you have", True),
            ("get me a red dress", True),
            ("I want a party dress", True),
            ("shop for shoes", True),
            # Conversational — False
            ("how are you", False),
            ("what's the weather today", False),
            ("tell me a joke", False),
            ("I need some advice", False),
            ("thank you", False),
            ("that's great", False),
            ("ok", False),
        ],
    )
    def test_routing(self, query: str, expected: bool) -> None:
        intent = parse_intent(query)
        assert intent.is_product_query == expected, (
            f"query={query!r}: expected is_product_query={expected}, got {intent.is_product_query}"
        )

    def test_buy_signal_with_garment_is_product(self) -> None:
        """Explicit buy signal + garment type → product query."""
        intent = parse_intent("can you help me buy a similar t-shirt")
        assert intent.is_product_query is True
        assert intent.garment_type == "top"

    def test_pure_occasion_no_garment_is_product(self) -> None:
        """Occasion alone without garment type still triggers product path."""
        intent = parse_intent("something for the wedding")
        assert intent.is_product_query is True
        assert intent.occasion == "wedding_guest"


# ---------------------------------------------------------------------------
# Group 6: Compound garment rules
# ---------------------------------------------------------------------------


class TestCompoundGarmentRules:
    @pytest.mark.parametrize(
        "query, expected_garment",
        [
            ("shorts for under dresses", "shorts"),
            ("dress shirt", "shirt"),
            ("jacket dress", "dress"),
            ("skirt", "skirt"),
            ("co-ord set", "coord"),
            ("coord set", "coord"),
            ("co-ord", "coord"),
            ("shirt dress", "dress"),
        ],
    )
    def test_compound(self, query: str, expected_garment: str) -> None:
        intent = parse_intent(query)
        assert intent.garment_type == expected_garment, (
            f"query={query!r}: expected garment={expected_garment!r}, got {intent.garment_type!r}"
        )

    def test_shorts_not_absorbed_by_dress_in_purpose_clause(self) -> None:
        """'Shorts for under dresses' → shorts, not dress."""
        intent = parse_intent("I need shorts for under dresses")
        assert intent.garment_type == "shorts"

    def test_dress_shirt_is_shirt(self) -> None:
        intent = parse_intent("looking for a dress shirt")
        assert intent.garment_type == "shirt"


# ---------------------------------------------------------------------------
# Group 7: Budget extraction
# ---------------------------------------------------------------------------


class TestBudgetExtraction:
    @pytest.mark.parametrize(
        "query, expected_max",
        [
            ("under ₹1000", 1000),
            ("below 2000", 2000),
            ("less than 1500", 1500),
            ("up to 3000", 3000),
            ("max 500", 500),
            ("within 800", 800),
            ("upto 2500", 2500),
        ],
    )
    def test_exact_budget(self, query: str, expected_max: int) -> None:
        intent = parse_intent(query)
        assert intent.budget_max_inr == expected_max, (
            f"query={query!r}: expected budget={expected_max}, got {intent.budget_max_inr}"
        )

    def test_approx_budget_30_pct_buffer(self) -> None:
        """around / about → 30% buffer added."""
        intent = parse_intent("around 1000")
        assert intent.budget_max_inr == 1300

    def test_approx_about(self) -> None:
        intent = parse_intent("about 2000 rupees")
        assert intent.budget_max_inr == 2600

    def test_budget_with_commas(self) -> None:
        intent = parse_intent("under ₹1,500")
        assert intent.budget_max_inr == 1500

    def test_no_budget(self) -> None:
        intent = parse_intent("show me red dresses")
        assert intent.budget_max_inr is None

    def test_budget_in_longer_query(self) -> None:
        intent = parse_intent("find me blue jeans under 2000")
        assert intent.budget_max_inr == 2000
        assert intent.garment_type == "jeans"
        assert intent.colour == "Blue"


# ---------------------------------------------------------------------------
# Group 8: merge_with_context
# ---------------------------------------------------------------------------


class TestMergeWithContext:
    def test_colour_refinement_inherits_garment_and_gender(self) -> None:
        """'in blue' has no garment/gender → inherits from context."""
        intent = parse_intent("in blue")
        merged = merge_with_context(intent, {"garment_type": "dress", "gender": "women"})
        assert merged.garment_type == "dress"
        assert merged.colour == "Blue"
        assert merged.gender == "women"
        assert merged.raw_query == "in blue"  # never modified

    def test_new_garment_type_overwrites_context(self) -> None:
        """When the new intent specifies a garment, context garment is dropped."""
        intent = parse_intent("show me jeans")
        merged = merge_with_context(intent, {"garment_type": "dress", "gender": "women"})
        assert merged.garment_type == "jeans"

    def test_new_intent_gender_wins_over_context(self) -> None:
        """Explicit gender in new intent beats context gender."""
        intent = parse_intent("blue jacket for him")
        merged = merge_with_context(intent, {"gender": "women"})
        assert merged.gender == "men"

    def test_raw_query_always_from_new_intent(self) -> None:
        """raw_query must always be the current turn's query."""
        intent = parse_intent("cheaper please")
        merged = merge_with_context(intent, {"garment_type": "kurti"})
        assert merged.raw_query == "cheaper please"
        assert merged.garment_type == "kurti"  # carried from context

    def test_context_colour_carried_forward(self) -> None:
        """When new intent has no colour, context colour is inherited."""
        intent = parse_intent("something more formal")
        merged = merge_with_context(intent, {"colour": "Red", "garment_type": "dress"})
        assert merged.colour == "Red"
        assert merged.garment_type == "dress"

    def test_context_occasion_carried_forward(self) -> None:
        intent = parse_intent("in blue")
        merged = merge_with_context(intent, {"occasion": "office", "garment_type": "trousers"})
        assert merged.occasion == "office"
        assert merged.garment_type == "trousers"

    def test_new_intent_colour_overwrites_context_colour(self) -> None:
        intent = parse_intent("show me red ones")
        merged = merge_with_context(intent, {"colour": "Blue", "garment_type": "dress"})
        assert merged.colour == "Red"
        assert merged.garment_type == "dress"

    def test_empty_context_is_safe(self) -> None:
        """merge_with_context must not crash on an empty dict.

        'kurti' is an ethnic women marker, so gender='women' is expected even
        without an explicit gender word in the query.
        """
        intent = parse_intent("black kurti")
        merged = merge_with_context(intent, {})
        assert merged.garment_type == "kurti"
        assert merged.colour == "Black"
        # kurti is in the ethnic women marker list → gender='women' by design
        assert merged.gender == "women"

    def test_returns_new_intenv1_instance(self) -> None:
        """merge_with_context should return a new object, not mutate the input."""
        intent = parse_intent("in blue")
        merged = merge_with_context(intent, {"garment_type": "dress"})
        assert merged is not intent
        assert intent.garment_type is None  # original not mutated


# ---------------------------------------------------------------------------
# Group 9: Store filter extraction
# ---------------------------------------------------------------------------


class TestStoreFilter:
    def test_single_store(self) -> None:
        intent = parse_intent("show me dresses from myntra")
        assert "myntra" in intent.store_filter

    def test_multiple_stores(self) -> None:
        intent = parse_intent("compare tops on myntra and snitch")
        assert "myntra" in intent.store_filter
        assert "snitch" in intent.store_filter

    def test_no_store_returns_empty_list(self) -> None:
        intent = parse_intent("show me a red dress")
        assert intent.store_filter == []

    def test_store_mention_makes_product_query(self) -> None:
        intent = parse_intent("anything on fashor")
        assert intent.is_product_query is True


# ---------------------------------------------------------------------------
# Group 10: raw_query preservation
# ---------------------------------------------------------------------------


class TestRawQueryPreservation:
    def test_raw_query_preserved_verbatim(self) -> None:
        q = "  Show Me BLACK Dresses Under ₹2,000  "
        intent = parse_intent(q)
        assert intent.raw_query == q

    def test_raw_query_case_preserved(self) -> None:
        q = "Navy BLUE Kurti"
        intent = parse_intent(q)
        assert intent.raw_query == "Navy BLUE Kurti"
        assert intent.colour == "Navy Blue"  # canonical form used internally
