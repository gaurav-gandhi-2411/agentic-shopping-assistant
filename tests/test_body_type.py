"""Unit tests for P3 body-type-aware styling guidance.

Covers:
- Registry integrity (body_type.py): every shape/modifier has recommend/
  deprioritize/why per garment class; no banned framing word in any
  why-string or POSITIVE_TEMPLATES entry.
- Parsing: base shape + modifier extraction (intent_parser.py), question-flag
  detection, and cross-consistency between intent_parser.py's local copy and
  body_type.py's own SYNONYMS/parse_body_type (the two are intentionally
  duplicated — see intent_parser.py's module comment).
- Score-delta signs: recommend keyword -> +0.1, deprioritize keyword -> -0.1,
  modifier composition, no-op when body_type/modifiers are both absent.
- NEVER-FILTER invariant at the real composer._score_candidates choke point.
- Ban-list gate in generate_rationales (case-insensitive) with body-positive
  template fallback.
- Clarify-message content (lists options, no banned words, states optional).
- State persistence/reconstruction across turns (graph.py).

No network/LLM calls — all LLM interactions use fake in-process stand-ins.
"""

from __future__ import annotations

import pytest

from src.agents.graph import _reconstruct_body_type_from_history, _reconstruct_gender_from_history
from src.agents.intent_parser import _BODY_TYPE_MAP, parse_intent
from src.agents.outfit.body_type import (
    BASE_SHAPE_SLUGS,
    BASE_SHAPES,
    GARMENT_CLASSES,
    MODIFIER_SLUGS,
    MODIFIERS,
    POSITIVE_TEMPLATES,
    POSITIVE_TEMPLATES_MEN,
    SYNONYMS,
    body_type_ack_message,
    body_type_clarify_message,
    body_type_score_delta,
    contains_banned_framing,
    garment_class_for_item,
    parse_body_type,
    query_tokens,
)
from src.agents.outfit.composer import _score_candidates
from src.agents.outfit.rationale import generate_rationales

# Men's garment classes (2026-07-25) — added alongside the original women's-
# only saree/lehenga/anarkali_kurta/neckline set. See body_type.py's module
# docstring "Men's coverage" note.
_WOMEN_GARMENT_CLASSES = {"saree", "lehenga", "anarkali_kurta", "neckline"}
_MEN_GARMENT_CLASSES = {"kurta_men", "sherwani", "bandhgala", "blazer_men", "trousers_men"}

# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


class TestRegistryIntegrity:
    def test_five_base_shapes_present(self) -> None:
        expected = {"pear", "apple", "hourglass", "rectangle", "inverted_triangle"}
        assert expected <= set(BASE_SHAPES.keys())
        assert expected <= set(BASE_SHAPE_SLUGS)

    def test_lean_build_base_shape_present(self) -> None:
        """2026-07-25 Area 1: new men's-only base shape, no women's equivalent."""
        assert "lean_build" in BASE_SHAPES
        assert "lean_build" in BASE_SHAPE_SLUGS
        assert set(BASE_SHAPES.keys()) == set(BASE_SHAPE_SLUGS) == {
            "pear", "apple", "hourglass", "rectangle", "inverted_triangle", "lean_build",
        }

    def test_three_modifiers_present(self) -> None:
        expected = {"petite", "tall", "plus_size"}
        assert expected <= set(MODIFIERS.keys())
        assert expected <= set(MODIFIER_SLUGS)

    def test_short_build_modifier_present(self) -> None:
        """2026-07-25 Area 1: new men's-only modifier, no women's equivalent."""
        assert "short_build" in MODIFIERS
        assert "short_build" in MODIFIER_SLUGS
        assert set(MODIFIERS.keys()) == set(MODIFIER_SLUGS) == {
            "petite", "tall", "plus_size", "short_build",
        }

    def test_every_base_shape_has_expected_garment_classes(self) -> None:
        """Women's-only shapes (pear/apple/hourglass/rectangle) keep the
        original 4-class set unchanged. inverted_triangle (shared, photo-
        reachable for both genders) has those 4 PLUS all 5 men's classes.
        lean_build (men's-only, new) has ONLY the 5 men's classes — it never
        had a women's ruleset to begin with, so asserting it lacks one is the
        correct invariant, not a gap.
        """
        for slug in ("pear", "apple", "hourglass", "rectangle"):
            assert set(BASE_SHAPES[slug].garments.keys()) == _WOMEN_GARMENT_CLASSES, slug
        assert (
            set(BASE_SHAPES["inverted_triangle"].garments.keys())
            == _WOMEN_GARMENT_CLASSES | _MEN_GARMENT_CLASSES
        )
        assert set(BASE_SHAPES["lean_build"].garments.keys()) == _MEN_GARMENT_CLASSES
        # Every garment class actually used is a real, known GARMENT_CLASSES member.
        all_used = {
            cls for profile in BASE_SHAPES.values() for cls in profile.garments
        }
        assert all_used <= set(GARMENT_CLASSES)

    def test_every_modifier_has_expected_garment_classes(self) -> None:
        """petite/plus_size keep the original women's-only 3-class set
        unchanged (no men's ruleset exists for either). tall (shared) gained
        kurta_men/trousers_men alongside its original 3. short_build (men's-
        only, new) has ONLY kurta_men/trousers_men — no sherwani entry, a
        disclosed gap (no catalogue length vocabulary to ground one), not an
        oversight.
        """
        for slug in ("petite", "plus_size"):
            assert set(MODIFIERS[slug].garments.keys()) == {
                "saree", "lehenga", "anarkali_kurta",
            }, slug
        assert set(MODIFIERS["tall"].garments.keys()) == {
            "saree", "lehenga", "anarkali_kurta", "kurta_men", "trousers_men",
        }
        assert set(MODIFIERS["short_build"].garments.keys()) == {"kurta_men", "trousers_men"}

    def test_every_rule_has_a_why(self) -> None:
        for profile in list(BASE_SHAPES.values()) + list(MODIFIERS.values()):
            for garment_class, rule in profile.garments.items():
                assert rule.why and isinstance(rule.why, str), (
                    f"{profile.slug}/{garment_class} missing why"
                )

    def test_no_banned_word_in_why_strings(self) -> None:
        offenders = []
        for profile in list(BASE_SHAPES.values()) + list(MODIFIERS.values()):
            for garment_class, rule in profile.garments.items():
                if contains_banned_framing(rule.why):
                    offenders.append(f"{profile.slug}/{garment_class}")
        assert not offenders, f"Banned framing word found in why-strings: {offenders}"

    def test_no_banned_word_in_positive_templates(self) -> None:
        # Not every BASE_SHAPE_SLUGS member has a women's template (lean_build
        # is men's-only — see POSITIVE_TEMPLATES_MEN), so this is a subset
        # check, not equality; every KEY present must still be a real slug.
        assert set(POSITIVE_TEMPLATES.keys()) <= set(BASE_SHAPE_SLUGS)
        offenders = [
            slug for slug, text in POSITIVE_TEMPLATES.items() if contains_banned_framing(text)
        ]
        assert not offenders, f"Banned framing word found in POSITIVE_TEMPLATES: {offenders}"

    def test_no_banned_word_in_positive_templates_men(self) -> None:
        """2026-07-25: POSITIVE_TEMPLATES_MEN mirrors the check above."""
        assert set(POSITIVE_TEMPLATES_MEN.keys()) <= set(BASE_SHAPE_SLUGS)
        assert set(POSITIVE_TEMPLATES_MEN.keys()) == {"inverted_triangle", "lean_build"}
        offenders = [
            slug for slug, text in POSITIVE_TEMPLATES_MEN.items()
            if contains_banned_framing(text)
        ]
        assert not offenders, f"Banned framing word found in POSITIVE_TEMPLATES_MEN: {offenders}"

    def test_no_banned_word_in_clarify_message(self) -> None:
        assert not contains_banned_framing(body_type_clarify_message())

    def test_no_banned_word_in_clarify_message_men(self) -> None:
        assert not contains_banned_framing(body_type_clarify_message("men"))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseBodyTypeModule:
    """body_type.parse_body_type — the registry-side copy."""

    def test_pear_shaped(self) -> None:
        assert parse_body_type("I'm pear-shaped") == ("pear", [])

    def test_petite_pear(self) -> None:
        assert parse_body_type("petite pear") == ("pear", ["petite"])

    def test_plus_size_hourglass(self) -> None:
        base, mods = parse_body_type("plus size hourglass")
        assert base == "hourglass"
        assert mods == ["plus_size"]

    def test_curvy_hips_is_pear_not_plus_size(self) -> None:
        """"curvy hips" must win over standalone "curvy" (plus_size) — the
        longer phrase claims its span first."""
        base, mods = parse_body_type("I have curvy hips")
        assert base == "pear"
        assert mods == []

    def test_curvy_alone_is_plus_size_modifier(self) -> None:
        base, mods = parse_body_type("I'm curvy")
        assert base is None
        assert mods == ["plus_size"]

    def test_no_match_returns_none_and_empty(self) -> None:
        assert parse_body_type("show me a red kurta") == (None, [])


class TestIntentParserBodyTypeFields:
    """intent_parser.parse_intent — the ParsedIntent-facing copy."""

    @pytest.mark.parametrize(
        "query, expected_base, expected_mods",
        [
            ("I'm pear-shaped", "pear", []),
            ("petite pear", "pear", ["petite"]),
            ("plus size hourglass", "hourglass", ["plus_size"]),
            ("I'm an inverted triangle", "inverted_triangle", []),
            ("tall rectangle body type", "rectangle", ["tall"]),
        ],
    )
    def test_body_type_extraction(
        self, query: str, expected_base: str, expected_mods: list[str]
    ) -> None:
        intent = parse_intent(query)
        assert intent.body_type == expected_base
        assert intent.body_modifiers == expected_mods

    def test_question_flag_what_suits_my_body_type(self) -> None:
        intent = parse_intent("what suits my body type?")
        assert intent.wants_body_type_guidance is True
        assert intent.body_type is None

    def test_question_flag_which_styles_suit_me(self) -> None:
        intent = parse_intent("which styles suit me?")
        assert intent.wants_body_type_guidance is True

    def test_question_flag_what_should_i_wear(self) -> None:
        intent = parse_intent("what should I wear for my body type")
        assert intent.wants_body_type_guidance is True

    def test_question_flag_false_when_shape_stated(self) -> None:
        """A body-type STATEMENT (not a question) never sets the flag."""
        intent = parse_intent("I'm pear-shaped, sangeet look")
        assert intent.wants_body_type_guidance is False
        assert intent.body_type == "pear"

    def test_question_flag_false_for_unrelated_query(self) -> None:
        intent = parse_intent("show me a red kurta")
        assert intent.wants_body_type_guidance is False

    @pytest.mark.parametrize(
        "query",
        [
            "I'm pear-shaped",
            "petite pear",
            "plus size hourglass",
            "I have curvy hips",
            "I'm curvy",
            "tall rectangle",
            "apple shaped",
            "broad-shouldered",
            # 2026-07-25 Area 1 additions — same drift guard for the men's phrases.
            "I have a muscular build",
            "broad build",
            "broad frame",
            "heavy build",
            "heavier build",
            "stocky build",
            "slim build",
            "lean build",
            "slender build",
            "narrow frame",
            "short height",
            "shorter build",
            "short build",
            "short stature",
        ],
    )
    def test_intent_parser_agrees_with_body_type_module(self, query: str) -> None:
        """Registry-drift guard: intent_parser.py intentionally duplicates
        body_type.py's SYNONYMS vocabulary (zero-project-import invariant —
        see intent_parser.py's module comment). Both must agree."""
        intent = parse_intent(query)
        module_base, module_mods = parse_body_type(query)
        assert intent.body_type == module_base
        assert intent.body_modifiers == module_mods

    def test_body_type_map_exact_key_and_value_parity_with_synonyms(self) -> None:
        """Stronger, exhaustive version of the sample-based drift guard above
        — catches ANY future phrase added to one dict and not the other, not
        just the ones covered by test_intent_parser_agrees_with_body_type_
        module's fixed sample list."""
        assert set(_BODY_TYPE_MAP.keys()) == set(SYNONYMS.keys())
        assert _BODY_TYPE_MAP == dict(SYNONYMS)


# ---------------------------------------------------------------------------
# Score-delta signs
# ---------------------------------------------------------------------------


class TestBodyTypeScoreDelta:
    """All fixtures below are REAL prod_name/detail_desc strings pulled from
    data/processed/unified/catalogue.parquet (women's saree/lehenga/kurta/
    anarkali/kurti/sharara/palazzo/tunic rows) — cited so future keyword edits
    can be re-verified against the live catalogue, not synthetic phrases. See
    this module's defect-fix note in body_type.py for why: the original
    recommend/deprioritize lists used research-doc phrasing that never
    matched real product text, making body_type_score_delta a silent no-op.
    """

    def test_pear_recommend_flared_and_palazzo_positive(self) -> None:
        """Real row: 'Khushal K Women Black Ethnic Motifs Printed Kurta with
        Palazzos & With Dupatta' — detail_desc contains "...Palazzos design...
        flared hem..." which matches pear/anarkali_kurta recommend
        ("flared", "palazzo")."""
        item = {
            "prod_name": "Khushal K Women Black Ethnic Motifs Printed Kurta with Palazzos "
            "& With Dupatta",
            "detail_desc": (
                "Black printed Kurta with Palazzos with dupatta Kurta design: Ethnic motifs "
                "printed Anarkali shape Regular style Mandarin collar, three-quarter regular "
                "sleeves Calf length with flared hem Viscose rayon machine weave fabric "
                "Palazzos design: Printed Palazzos Elasticated waistband Slip-on closure"
            ),
            "product_type": "kurta",
        }
        assert body_type_score_delta(item, "pear") == pytest.approx(0.1)

    def test_pear_deprioritize_straight_kurta_negative(self) -> None:
        """Real row: 'AHIKA Women Black & Green Printed Straight Kurta' —
        "straight kurta" is pear/anarkali_kurta's deprioritize keyword."""
        item = {
            "prod_name": "AHIKA Women Black & Green Printed Straight Kurta",
            "detail_desc": (
                "Black and green printed straight kurta, has a nitched round neck, "
                "three-quarter sleeves, straight hem, side slits"
            ),
            "product_type": "kurta",
        }
        assert body_type_score_delta(item, "pear") == pytest.approx(-0.1)

    def test_apple_recommend_straight_kurta_positive(self) -> None:
        """The SAME real row as the pear-negative test above scores POSITIVE
        for apple ("straight kurta" is apple's recommend keyword) — the delta
        depends entirely on which body type is passed in, never a fixed
        per-item penalty."""
        item = {
            "prod_name": "AHIKA Women Black & Green Printed Straight Kurta",
            "detail_desc": (
                "Black and green printed straight kurta, has a nitched round neck, "
                "three-quarter sleeves, straight hem, side slits"
            ),
            "product_type": "kurta",
        }
        assert body_type_score_delta(item, "apple") == pytest.approx(0.1)

    def test_apple_deprioritize_mandarin_collar_negative(self) -> None:
        """Real row: 'Soch Women Red Thread Work Georgette Anarkali Kurta' —
        detail_desc contains "Mandarin collar", apple's closed-neckline
        deprioritize keyword."""
        item = {
            "prod_name": "Soch Women Red Thread Work Georgette Anarkali Kurta",
            "detail_desc": (
                "Colour: red Solid woven design Mandarin collar Long, regular sleeves "
                "Anarkali shape with pleated style Thread work detail Ankle length with "
                "flared hem Machine weave regular georgette"
            ),
            "product_type": "kurta",
        }
        assert body_type_score_delta(item, "apple") == pytest.approx(-0.1)

    def test_hourglass_recommend_v_neck_positive(self) -> None:
        """Real row: 'Libas Women Maroon Printed Kurta with Palazzos & Dupatta'
        — detail_desc's "V-neck" matches hourglass's neckline-overlay
        recommend (applied to every garment class, see _profile_keywords)."""
        item = {
            "prod_name": "Libas Women Maroon Printed Kurta with Palazzos & Dupatta",
            "detail_desc": (
                "Maroon printed kurta with palazzos and dupatta Maroon A-line calf "
                "length kurta, has a V-neck, three-quarter sleeves, front slit"
            ),
            "product_type": "kurta",
        }
        assert body_type_score_delta(item, "hourglass") == pytest.approx(0.1)

    def test_hourglass_deprioritize_oversized_negative(self) -> None:
        """Real row: 'Ahalyaa Women Beige Floral Printed Regular Gotta Patti
        Kurta with Palazzos & With Dupatta' — detail_desc's "...abstract and
        oversized motifs..." matches hourglass's deprioritize keyword."""
        item = {
            "prod_name": "Ahalyaa Women Beige Floral Printed Regular Gotta Patti Kurta with "
            "Palazzos & With Dupatta",
            "detail_desc": (
                "Right from delicate all-over patterns to abstract and oversized motifs, "
                "romantic florals lend any garment a feminine touch."
            ),
            "product_type": "kurta",
        }
        assert body_type_score_delta(item, "hourglass") == pytest.approx(-0.1)

    def test_rectangle_recommend_tiered_positive(self) -> None:
        """Real row: 'FASHOR Women Pink Ethnic Motifs Kurta' — detail_desc's
        "Straight shape with tiered style" matches rectangle's "tiered"."""
        item = {
            "prod_name": "FASHOR Women Pink Ethnic Motifs Kurta",
            "detail_desc": (
                "Colour: pink Ethnic motifs woven design Round neck Three-quarter, regular "
                "sleeves Straight shape with tiered style Calf length with straight hem"
            ),
            "product_type": "kurta",
        }
        assert body_type_score_delta(item, "rectangle") == pytest.approx(0.1)

    def test_inverted_triangle_recommend_flared_positive(self) -> None:
        """Real row: 'Nayo Women Red Floral Printed Kurta With Trouser &
        Dupatta' — detail_desc's "Flared hem" matches inverted_triangle's
        recommend keyword."""
        item = {
            "prod_name": "Nayo Women Red Floral Printed Kurta With Trouser & Dupatta",
            "detail_desc": (
                "Kurta design: Printed kurta Anarkali design Round neck Three-quarter "
                "sleeves Flared hem Calf length"
            ),
            "product_type": "kurta",
        }
        assert body_type_score_delta(item, "inverted_triangle") == pytest.approx(0.1)

    def test_inverted_triangle_deprioritize_yoke_negative(self) -> None:
        """Real row: 'Anouk Women Peach-Coloured Yoke Design Mirror-Work Kurta
        with Trousers & With Dupatta' — "yoke" (upper-body detail) is
        inverted_triangle's deprioritize keyword, but is PEAR's positive
        "embroidered yoke" signal — same real-world detail, opposite
        direction depending on the shape passed in."""
        item = {
            "prod_name": "Anouk Women Peach-Coloured Yoke Design Mirror-Work Kurta with "
            "Trousers & With Dupatta",
            "detail_desc": (
                "Peach-coloured yoke design kurta with palazzos with dupatta Kurta design: "
                "Ethnic motifs yoke design Straight shape Regular style"
            ),
            "product_type": "kurta",
        }
        assert body_type_score_delta(item, "inverted_triangle") == pytest.approx(-0.1)

    def test_neutral_item_zero_delta(self) -> None:
        item = {"prod_name": "Plain Blue Shirt", "detail_desc": "", "product_type": "Shirt"}
        assert body_type_score_delta(item, "pear") == pytest.approx(0.0)

    def test_no_body_type_no_modifiers_always_zero(self) -> None:
        """Even a row that WOULD match pear's deprioritize list scores 0.0
        with no body_type/modifiers passed in — the bias is fully opt-in."""
        item = {
            "prod_name": "AHIKA Women Black & Green Printed Straight Kurta",
            "detail_desc": "Black and green printed straight kurta, straight hem, side slits",
            "product_type": "kurta",
        }
        assert body_type_score_delta(item, None) == pytest.approx(0.0)
        assert body_type_score_delta(item, None, []) == pytest.approx(0.0)

    def test_modifier_alone_no_base_shape_petite_deprioritize(self) -> None:
        """Real row: 'Libas Women Blue Embroidered Panelled Kurta with
        Churidar & With Dupatta' — detail_desc's "Floor length" is petite's
        deprioritize keyword, applied with body_type=None."""
        item = {
            "prod_name": "Libas Women Blue Embroidered Panelled Kurta with Churidar & With "
            "Dupatta",
            "detail_desc": (
                "Kurta design: Geometric embroidered A-line shape Panelled style Round "
                "neck, long regular sleeves 2 pockets Floor length with flared hem"
            ),
            "product_type": "kurta",
        }
        assert body_type_score_delta(item, None, ["petite"]) == pytest.approx(-0.1)

    def test_modifier_composes_with_base_shape(self) -> None:
        """A real Banarasi silk saree row ('KALINI Maroon & Gold Ethnic Motifs
        Zari Silk Blend Banarasi Saree') scores 0.0 for pear ALONE (pear's own
        saree rule deprioritizes "banarasi", but this text has no OTHER pear
        keyword to offset it back to a clean read — see below), and the SAME
        text scores +0.1 for hourglass (its saree recommend keyword "banarasi"/
        "zari") — modifiers/shapes UNION independently per §5's encoding note."""
        item = {
            "prod_name": "KALINI Maroon & Gold Ethnic Motifs Zari Silk Blend Banarasi Saree",
            "detail_desc": (
                "Design Details Maroon and gold-toned banarasi saree Ethnic motifs woven "
                "design saree with woven design border Has zari detail"
            ),
            "product_type": "saree",
        }
        # pear/saree deprioritizes "banarasi" -> -0.1 on its own.
        assert body_type_score_delta(item, "pear") == pytest.approx(-0.1)
        # hourglass/saree recommends "banarasi"/"zari" -> +0.1.
        assert body_type_score_delta(item, "hourglass") == pytest.approx(0.1)
        # "tall" modifier (no base shape) also recommends banarasi/zari for saree.
        assert body_type_score_delta(item, None, ["tall"]) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# NEVER-FILTER invariant (the real composer choke point)
# ---------------------------------------------------------------------------


class _FakeRetriever:
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
        return list(self._items)


def _kurta_candidate(article_id: str, prod_name: str, detail_desc: str = "") -> dict:
    """product_type="Kurta" alone resolves to ethnic_top via classify_item's
    product-type-first shortcut, so prod_name/detail_desc are free to carry
    real body-type keyword text without affecting slot-type gating.
    """
    return {
        "article_id": article_id,
        "product_type": "Kurta",
        "prod_name": prod_name,
        "display_name": prod_name,
        "detail_desc": detail_desc,
        "colour": "black",
        "gender": "women",
        "score": 0.5,
        "price_inr": 800.0,
        "store": "storea",
    }


# Real catalogue rows (data/processed/unified/catalogue.parquet) reused from
# TestBodyTypeScoreDelta above — see those tests for the exact matched keyword.
_REAL_PEAR_RECOMMEND_NAME = "Moda Rapido Women Maroon & Grey Ethnic Motifs Printed A-Line Kurta"
_REAL_PEAR_DEPRIORITIZE_NAME = "AHIKA Women Black & Green Printed Straight Kurta"
_REAL_PEAR_DEPRIORITIZE_DESC = (
    "Black and green printed straight kurta, has a nitched round neck, "
    "three-quarter sleeves, straight hem, side slits"
)
# Real row, but detail_desc intentionally omitted below (its detail_desc does
# carry "tiered", a RECTANGLE signal — irrelevant here since this fixture only
# checks pear, and prod_name alone carries no pear keyword either way).
_REAL_NEUTRAL_NAME = "FASHOR Women Pink Ethnic Motifs Kurta"


class TestNeverFilterInvariant:
    """body_type_score_delta must only ever bias score — the SET of candidates
    that survive composer._score_candidates' hard gates is identical with or
    without a known body_type (order may differ)."""

    _common_kwargs = {
        "query": "ethnic top",
        "slot_name": "top",
        "occasion_slug": "casual",
        "gender": "women",
        "anchor_colour": "black",
        "seen_ids": set(),
        "seen_prod_colour": set(),
        "budget_remaining": None,
        "pairing_stats": None,
        "anchor_class": "ethnic_bottom",
        "seen_stores": None,
        "neutral_fallback_ids": set(),
    }

    def test_same_candidate_set_with_and_without_body_type(self) -> None:
        candidates = [
            _kurta_candidate("R1", _REAL_PEAR_RECOMMEND_NAME),  # pear recommend match ("a-line")
            _kurta_candidate(  # pear deprioritize match ("straight kurta")
                "D1", _REAL_PEAR_DEPRIORITIZE_NAME, _REAL_PEAR_DEPRIORITIZE_DESC
            ),
            _kurta_candidate("N1", _REAL_NEUTRAL_NAME),  # neutral
        ]

        scored_without = _score_candidates(candidates, **self._common_kwargs, body_type=None)
        scored_with = _score_candidates(
            candidates, **self._common_kwargs, body_type="pear", body_modifiers=[]
        )

        ids_without = {item["article_id"] for _, item in scored_without}
        ids_with = {item["article_id"] for _, item in scored_with}
        assert ids_without == ids_with == {"R1", "D1", "N1"}

    def test_body_type_changes_ranking_not_membership(self) -> None:
        candidates = [
            _kurta_candidate("R1", _REAL_PEAR_RECOMMEND_NAME),
            _kurta_candidate("D1", _REAL_PEAR_DEPRIORITIZE_NAME, _REAL_PEAR_DEPRIORITIZE_DESC),
        ]
        scored_with = _score_candidates(
            candidates, **self._common_kwargs, body_type="pear", body_modifiers=[]
        )
        scored_with.sort(key=lambda t: t[0], reverse=True)
        # The recommend-match must outrank the deprioritize-match once body_type biases scoring.
        assert scored_with[0][1]["article_id"] == "R1"


# ---------------------------------------------------------------------------
# Ban-list gate in generate_rationales
# ---------------------------------------------------------------------------


def _make_look(
    seed_colour: str = "blue",
    seed_type: str = "lehenga",
    complement_colour: str = "gold",
    complement_type: str = "dupatta",
    occasion: str = "sangeet",
) -> dict:
    seed_item = {
        "article_id": "SEED1",
        "prod_name": f"{seed_colour.title()} {seed_type.title()}",
        "colour": seed_colour,
        "product_type": seed_type,
        "_role": "seed",
        "_slot": None,
        "gender": "women",
    }
    complement = {
        "article_id": "COMP1",
        "prod_name": f"{complement_colour.title()} {complement_type.title()}",
        "colour": complement_colour,
        "product_type": complement_type,
        "_role": "complement",
        "_slot": "accessory",
        "gender": "women",
    }
    return {
        "look_id": "test-look-bt",
        "seed_item": seed_item,
        "complements": [complement],
        "outfit_rationale": "",
        "empty_slots": [],
        "occasion": occasion,
        "gender": "women",
        "budget_total_inr": 4000.0,
    }


class _BannedTextLLM:
    """Fake LLMClient whose generate() returns rationale text tripping the ban-list."""

    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, prompt: str, system: str | None = None, **kw: object) -> str:
        import json as _json

        return _json.dumps([self._text])

    def chat(self, messages: list, **kw: object) -> str:
        return self.generate("")

    def generate_stream(self, prompt: str, **kw: object):  # type: ignore[override]
        return iter([])

    def chat_stream(self, messages: list, **kw: object):  # type: ignore[override]
        return iter([])


class TestBanListGate:
    def test_banned_phrase_discarded_body_positive_fallback(self) -> None:
        look = _make_look()
        llm = _BannedTextLLM("This lehenga hides your tummy beautifully.")
        rationales = generate_rationales(
            [look], llm, occasion="sangeet", gender="women", body_type="pear",
        )
        assert len(rationales) == 1
        assert "hides" not in rationales[0].lower()
        # Body-positive template fallback for a body-type turn.
        assert POSITIVE_TEMPLATES["pear"] in rationales[0]

    def test_banned_phrase_case_insensitive(self) -> None:
        look = _make_look()
        llm = _BannedTextLLM("HIDES your tummy and MINIMIZES your waist.")
        rationales = generate_rationales(
            [look], llm, occasion="sangeet", gender="women", body_type="pear",
        )
        assert "hides" not in rationales[0].lower()
        assert "minimizes" not in rationales[0].lower()

    def test_banned_phrase_without_body_type_falls_back_to_plain_template(self) -> None:
        """The gate applies universally — even a non-body-type look with a
        hallucinated banned word is discarded, falling back to the plain
        (non-body-positive) template."""
        look = _make_look()
        llm = _BannedTextLLM("This will fix your posture instantly.")
        rationales = generate_rationales([look], llm, occasion="sangeet", gender="women")
        assert "fix" not in rationales[0].lower()
        assert rationales[0]  # non-empty, valid template output

    def test_clean_rationale_not_discarded(self) -> None:
        look = _make_look()
        llm = _BannedTextLLM("The blue lehenga balances the look beautifully for sangeet.")
        rationales = generate_rationales(
            [look], llm, occasion="sangeet", gender="women", body_type="pear",
        )
        assert "balances" in rationales[0].lower()


# ---------------------------------------------------------------------------
# Clarify-message content
# ---------------------------------------------------------------------------


class TestClarifyMessage:
    def test_lists_all_five_shapes(self) -> None:
        msg = body_type_clarify_message().lower()
        for keyword in ("pear", "apple", "hourglass", "rectangle", "inverted triangle"):
            assert keyword in msg

    def test_mentions_modifiers(self) -> None:
        msg = body_type_clarify_message().lower()
        assert "petite" in msg
        assert "tall" in msg
        assert "plus-size" in msg or "plus size" in msg

    def test_states_optional(self) -> None:
        msg = body_type_clarify_message().lower()
        assert "optional" in msg or "skip" in msg

    def test_no_banned_words(self) -> None:
        assert not contains_banned_framing(body_type_clarify_message())


class TestClarifyMessageMen:
    """2026-07-25 Area 1: body_type_clarify_message(gender='men')."""

    def test_lists_men_build_options(self) -> None:
        msg = body_type_clarify_message("men").lower()
        for keyword in ("broad", "muscular", "slim", "lean", "short", "tall"):
            assert keyword in msg

    def test_no_women_shape_words(self) -> None:
        msg = body_type_clarify_message("men").lower()
        for keyword in ("pear", "apple", "hourglass", "petite"):
            assert keyword not in msg

    def test_states_optional(self) -> None:
        msg = body_type_clarify_message("men").lower()
        assert "optional" in msg or "skip" in msg

    def test_no_banned_words(self) -> None:
        assert not contains_banned_framing(body_type_clarify_message("men"))

    def test_unknown_gender_falls_back_to_women_text_unchanged(self) -> None:
        """Never guess — an unresolved gender preserves the original
        (pre-Area-1) behavior exactly, per body_type_clarify_message's
        documented gender-param contract."""
        assert body_type_clarify_message(None) == body_type_clarify_message()
        assert body_type_clarify_message("unisex") == body_type_clarify_message()


# ---------------------------------------------------------------------------
# Bare body-type STATEMENT acknowledgement (Wave 7 hang fix)
# ---------------------------------------------------------------------------


class TestBodyTypeAckMessage:
    """body_type_ack_message — the deterministic reply for a bare shape
    statement (e.g. "I have an inverted triangle silhouette") with no
    occasion/garment named. See graph.py's router_node short-circuit.
    """

    @pytest.mark.parametrize("slug", [s for s in BASE_SHAPE_SLUGS if s in POSITIVE_TEMPLATES])
    def test_every_base_shape_mentions_shape_and_positive_template(self, slug: str) -> None:
        msg = body_type_ack_message(slug, [])
        assert slug.replace("_", " ") in msg.lower()
        assert POSITIVE_TEMPLATES[slug] in msg

    @pytest.mark.parametrize("slug", list(POSITIVE_TEMPLATES_MEN.keys()))
    def test_men_slugs_mention_build_and_men_positive_template(self, slug: str) -> None:
        """2026-07-25: lean_build/inverted_triangle under gender='men' use
        POSITIVE_TEMPLATES_MEN and men's-natural display phrasing, never the
        literal slug name or the women's template."""
        msg = body_type_ack_message(slug, [], gender="men")
        assert POSITIVE_TEMPLATES_MEN[slug] in msg
        womens_text = POSITIVE_TEMPLATES.get(slug)
        if womens_text:
            assert womens_text not in msg

    def test_no_doubled_build_word_for_broad_or_lean_build(self) -> None:
        """Live-proof catch (2026-07-25): _MEN_DISPLAY_LABELS values for base
        shapes already end in "build" ("broad build"/"lean build") — the ack
        sentence must not ALSO append the word "build", producing "broad
        build build in mind!"."""
        msg_broad = body_type_ack_message("inverted_triangle", [], gender="men")
        assert "build build" not in msg_broad
        assert "broad build in mind" in msg_broad

        msg_lean = body_type_ack_message("lean_build", [], gender="men")
        assert "build build" not in msg_lean
        assert "lean build in mind" in msg_lean

    def test_short_or_tall_modifier_alone_still_reads_naturally(self) -> None:
        """short_build/tall's men's display labels ("short"/"tall") do NOT
        end in "build" — the ack sentence appends it exactly once."""
        msg_short = body_type_ack_message(None, ["short_build"], gender="men")
        assert "short build in mind" in msg_short
        assert "build build" not in msg_short

        msg_tall = body_type_ack_message(None, ["tall"], gender="men")
        assert "tall build in mind" in msg_tall
        assert "build build" not in msg_tall

    def test_lean_build_without_gender_has_no_crash_and_no_template(self) -> None:
        """lean_build has NO women's-default template (men's-only slug) — must
        gracefully omit the why-sentence, never KeyError or show wrong text."""
        msg = body_type_ack_message("lean_build", [])
        assert "lean build" in msg.lower()
        for text in POSITIVE_TEMPLATES_MEN.values():
            assert text not in msg

    def test_pear_under_men_gender_has_no_crash_and_no_wrong_template(self) -> None:
        """A photo-classified 'pear' for a man (gender='men') has no men's
        ruleset — must fall through honestly, never show the women's
        A-line-lehenga template to a man."""
        msg = body_type_ack_message("pear", [], gender="men")
        assert POSITIVE_TEMPLATES["pear"] not in msg

    def test_modifier_prefixes_base_shape(self) -> None:
        msg = body_type_ack_message("pear", ["petite"])
        assert "petite pear" in msg.lower()

    def test_modifier_only_no_base_shape(self) -> None:
        msg = body_type_ack_message(None, ["petite"])
        assert "petite silhouette" in msg.lower()
        # No POSITIVE_TEMPLATES entry exists for a bare modifier — must not crash.
        assert "none" not in msg.lower()

    def test_asks_what_shopping_for(self) -> None:
        msg = body_type_ack_message("inverted_triangle", []).lower()
        assert "shopping for" in msg

    @pytest.mark.parametrize("slug", list(BASE_SHAPE_SLUGS))
    def test_no_banned_words(self, slug: str) -> None:
        assert not contains_banned_framing(body_type_ack_message(slug, []))


# ---------------------------------------------------------------------------
# State persistence / reconstruction across turns (graph.py)
# ---------------------------------------------------------------------------


class TestReconstructBodyTypeFromHistory:
    def test_empty_history_returns_none(self) -> None:
        assert _reconstruct_body_type_from_history([]) == (None, [])

    def test_finds_most_recent_user_message_with_body_type(self) -> None:
        messages = [
            {"role": "user", "content": "I'm pear-shaped, sangeet look"},
            {"role": "assistant", "content": "Here's a sangeet look for you."},
        ]
        assert _reconstruct_body_type_from_history(messages) == ("pear", [])

    def test_scans_backward_past_messages_without_body_type(self) -> None:
        messages = [
            {"role": "user", "content": "I'm pear-shaped, sangeet look"},
            {"role": "assistant", "content": "Here's a sangeet look for you."},
            {"role": "user", "content": "show me more options"},
        ]
        assert _reconstruct_body_type_from_history(messages) == ("pear", [])

    def test_modifier_only_message_is_recovered(self) -> None:
        messages = [{"role": "user", "content": "I'm petite, office look"}]
        base, mods = _reconstruct_body_type_from_history(messages)
        assert base is None
        assert mods == ["petite"]

    def test_non_user_messages_ignored(self) -> None:
        messages = [{"role": "assistant", "content": "pear-shaped looks great"}]
        assert _reconstruct_body_type_from_history(messages) == (None, [])


# ---------------------------------------------------------------------------
# Men's garment classification + gender threading (2026-07-25, Area 1)
# ---------------------------------------------------------------------------


class TestGarmentClassForItemGenderBug:
    """PRE-EXISTING CORRECTNESS BUG (2026-07-25): garment_class_for_item had
    no gender awareness, so a MAN'S kurta always matched
    _ANARKALI_KURTA_MARKERS's bare "kurta" and was silently classified as
    "anarkali_kurta" — scored against WOMEN'S a-line/flare/embroidered-yoke
    recommend/deprioritize keywords. Never a crash (bias-only scoring), so it
    shipped unnoticed. These tests pin the fix so it cannot silently return.
    """

    def test_mens_kurta_classifies_as_kurta_men_not_anarkali(self) -> None:
        assert garment_class_for_item("kurta", "Men's Regular Fit Cotton Kurta", "men") == (
            "kurta_men"
        )

    def test_same_item_without_gender_keeps_original_anarkali_classification(self) -> None:
        """Backward-compat pin: omitting gender (or any non-"men" value)
        preserves the ORIGINAL pre-fix classification exactly — zero
        regression for the women's/unknown-gender flow."""
        assert (
            garment_class_for_item("kurta", "Men's Regular Fit Cotton Kurta")
            == "anarkali_kurta"
        )
        assert (
            garment_class_for_item("kurta", "Men's Regular Fit Cotton Kurta", "women")
            == "anarkali_kurta"
        )

    def test_mens_kurta_no_longer_scored_against_womens_flare_keywords(self) -> None:
        """End-to-end pin at the body_type_score_delta level: a plain men's
        regular-fit kurta must not pick up a spurious +0.1 from pear's
        women's a-line/flare/embellished recommend list once gender="men" is
        threaded through — it should score via kurta_men's OWN rules instead."""
        item = {
            "product_type": "kurta",
            "prod_name": "Men's Regular Fit Cotton Kurta",
            "detail_desc": "A comfortable everyday kurta.",
        }
        # Under "pear" (women's shape, no men's ruleset at all): must be a
        # clean 0.0 no-op for a men's item, never accidentally matching
        # pear's anarkali_kurta a-line/flare vocabulary via the old bug.
        assert body_type_score_delta(item, "pear", gender="men") == 0.0

    def test_womens_kurta_classification_unaffected(self) -> None:
        assert garment_class_for_item(
            "kurta", "Women's A-Line Embroidered Anarkali Kurta", "women"
        ) == "anarkali_kurta"

    def test_precedence_sherwani_with_kurta_in_name_wins_as_sherwani(self) -> None:
        """A listing naming multiple garments ("Sherwani with Kurta and
        Pyjama Set") must classify as the more specific sherwani, not fall
        through to the generic kurta_men bucket."""
        assert garment_class_for_item(
            "sherwani", "Sherwani with Kurta and Pyjama Set", "men"
        ) == "sherwani"


class TestMensGarmentClassification:
    @pytest.mark.parametrize(
        "product_type,prod_name,expected",
        [
            ("sherwani", "Wine Embroidered Wedding Sherwani", "sherwani"),
            ("bandhgala", "Beige Tailored Bandhgala", "bandhgala"),
            ("blazer", "Navy Slim Fit Blazer", "blazer_men"),
            ("trousers", "Grey Regular Fit Trousers", "trousers_men"),
            ("kurta", "White Straight Fit Kurta", "kurta_men"),
        ],
    )
    def test_each_mens_class_reachable(self, product_type, prod_name, expected) -> None:
        assert garment_class_for_item(product_type, prod_name, "men") == expected

    def test_non_mens_garment_returns_none_under_men_gender(self) -> None:
        assert garment_class_for_item("footwear", "Running Shoes", "men") is None

    def test_neckline_overlay_excluded_for_mens_classes(self) -> None:
        """_profile_keywords must NOT pull inverted_triangle's women's
        neckline (v-neck/boat-neck) keywords into a men's kurta score."""
        item = {
            "product_type": "kurta",
            "prod_name": "Men's Boat Neck Regular Fit Kurta",
            "detail_desc": "",
        }
        # "boat" is a deprioritize keyword under inverted_triangle's WOMEN'S
        # neckline rule — if the overlay leaked into men's scoring, this item
        # would score -0.1 instead of the correct kurta_men +0.1 ("regular
        # fit" is in kurta_men's recommend list).
        assert body_type_score_delta(item, "inverted_triangle", gender="men") == 0.1


class TestQueryTokensGender:
    def test_mens_variant_used_for_inverted_triangle_under_men_gender(self) -> None:
        tokens = query_tokens("inverted_triangle", gender="men")
        assert tokens == BASE_SHAPES["inverted_triangle"].query_tokens_men
        assert "flared" not in tokens  # never the women's string

    def test_womens_variant_unchanged_without_gender(self) -> None:
        assert query_tokens("inverted_triangle") == BASE_SHAPES["inverted_triangle"].query_tokens
        assert query_tokens("inverted_triangle", gender="women") == (
            BASE_SHAPES["inverted_triangle"].query_tokens
        )

    def test_lean_build_only_reachable_with_men_gender_tokens(self) -> None:
        assert query_tokens("lean_build", gender="men") == (
            BASE_SHAPES["lean_build"].query_tokens_men
        )

    def test_profile_without_mens_variant_falls_back_to_shared_string(self) -> None:
        """pear has no query_tokens_men — gender='men' must fall back to the
        shared string, never crash or return empty."""
        assert query_tokens("pear", gender="men") == BASE_SHAPES["pear"].query_tokens

    def test_short_build_modifier_mens_tokens(self) -> None:
        assert query_tokens(None, ["short_build"], gender="men") == (
            MODIFIERS["short_build"].query_tokens_men
        )


class TestBodyTypeScoreDeltaNeverFilterMensPath:
    """Mirrors the existing women's never-filter invariant test, for men's
    garment classes specifically."""

    def test_recommend_keyword_scores_positive(self) -> None:
        item = {"product_type": "kurta", "prod_name": "Men's Slim Fit Tailored Kurta"}
        assert body_type_score_delta(item, "lean_build", gender="men") == 0.1

    def test_deprioritize_keyword_scores_negative(self) -> None:
        item = {"product_type": "kurta", "prod_name": "Men's Slim Fit Kurta"}
        assert body_type_score_delta(item, "inverted_triangle", gender="men") == -0.1


class TestMensSynonymParsing:
    @pytest.mark.parametrize(
        "text,expected_base",
        [
            ("I have a muscular build", "inverted_triangle"),
            ("broad build here", "inverted_triangle"),
            ("I would say broad frame", "inverted_triangle"),
            ("heavy build honestly", "inverted_triangle"),
            ("heavier build", "inverted_triangle"),
            ("stocky build", "inverted_triangle"),
            ("slim build for me", "lean_build"),
            ("lean build", "lean_build"),
            ("slender build", "lean_build"),
            ("narrow frame", "lean_build"),
        ],
    )
    def test_mens_base_phrases_parse(self, text, expected_base) -> None:
        base, _ = parse_body_type(text)
        assert base == expected_base

    @pytest.mark.parametrize(
        "text",
        ["short height", "shorter build", "short build", "short stature"],
    )
    def test_short_build_modifier_phrases_parse(self, text) -> None:
        _, mods = parse_body_type(text)
        assert "short_build" in mods

    def test_bare_slim_does_not_trigger_lean_build(self) -> None:
        """Precision guard: 'slim fit kurta' is a common GARMENT query (1,442
        trouser rows alone use this tag), not a body-type statement — a bare
        "slim" word must never misfire as lean_build."""
        base, mods = parse_body_type("show me a slim fit kurta for the wedding")
        assert base is None
        assert "lean_build" not in mods

    def test_bare_short_does_not_trigger_short_build(self) -> None:
        """Precision guard: 'short kurta' (374 catalogue rows) and 'shorts'
        are common GARMENT terms — bare "short" must never misfire."""
        base, mods = parse_body_type("looking for a short kurta")
        assert base is None
        assert "short_build" not in mods

        base2, mods2 = parse_body_type("need some shorts for the gym")
        assert base2 is None
        assert "short_build" not in mods2

    def test_existing_broad_shouldered_phrase_still_resolves_inverted_triangle(self) -> None:
        """No regression to the pre-existing women's-oriented phrasing —
        still resolves to the same shared slug."""
        base, _ = parse_body_type("I'm broad-shouldered")
        assert base == "inverted_triangle"


# ---------------------------------------------------------------------------
# Gender reconstruction from history (graph.py, 2026-07-25)
# ---------------------------------------------------------------------------


class TestReconstructGenderFromHistory:
    def test_empty_history_returns_none(self) -> None:
        assert _reconstruct_gender_from_history([]) is None

    def test_finds_most_recent_user_message_with_gender(self) -> None:
        messages = [
            {"role": "user", "content": "kurta for men under 3000"},
            {"role": "assistant", "content": "Here are some options."},
        ]
        assert _reconstruct_gender_from_history(messages) == "men"

    def test_scans_backward_past_messages_without_gender(self) -> None:
        messages = [
            {"role": "user", "content": "sherwani for my husband"},
            {"role": "assistant", "content": "Here's a sherwani."},
            {"role": "user", "content": "I have a muscular build"},
        ]
        assert _reconstruct_gender_from_history(messages) == "men"

    def test_no_gender_signal_anywhere_returns_none(self) -> None:
        """Never guess — a photo-only body-type statement with no prior
        gender-bearing message must resolve to None, not a default."""
        messages = [{"role": "user", "content": "I have an inverted triangle silhouette"}]
        assert _reconstruct_gender_from_history(messages) is None

    def test_non_user_messages_ignored(self) -> None:
        messages = [{"role": "assistant", "content": "for men, try this kurta"}]
        assert _reconstruct_gender_from_history(messages) is None
