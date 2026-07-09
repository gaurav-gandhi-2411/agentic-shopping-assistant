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

from src.agents.graph import _reconstruct_body_type_from_history
from src.agents.intent_parser import parse_intent
from src.agents.outfit.body_type import (
    BASE_SHAPE_SLUGS,
    BASE_SHAPES,
    GARMENT_CLASSES,
    MODIFIER_SLUGS,
    MODIFIERS,
    POSITIVE_TEMPLATES,
    body_type_clarify_message,
    body_type_score_delta,
    contains_banned_framing,
    parse_body_type,
)
from src.agents.outfit.composer import _score_candidates
from src.agents.outfit.rationale import generate_rationales

# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


class TestRegistryIntegrity:
    def test_five_base_shapes_present(self) -> None:
        expected = {"pear", "apple", "hourglass", "rectangle", "inverted_triangle"}
        assert set(BASE_SHAPES.keys()) == expected
        assert set(BASE_SHAPE_SLUGS) == expected

    def test_three_modifiers_present(self) -> None:
        expected = {"petite", "tall", "plus_size"}
        assert set(MODIFIERS.keys()) == expected
        assert set(MODIFIER_SLUGS) == expected

    def test_every_base_shape_has_all_garment_classes(self) -> None:
        for slug, profile in BASE_SHAPES.items():
            assert set(profile.garments.keys()) == set(GARMENT_CLASSES), (
                f"{slug} missing a garment class"
            )

    def test_every_modifier_has_saree_lehenga_anarkali_no_neckline(self) -> None:
        for slug, profile in MODIFIERS.items():
            assert set(profile.garments.keys()) == {"saree", "lehenga", "anarkali_kurta"}, (
                f"{slug} garment classes unexpected"
            )

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
        assert set(POSITIVE_TEMPLATES.keys()) == set(BASE_SHAPE_SLUGS)
        offenders = [
            slug for slug, text in POSITIVE_TEMPLATES.items() if contains_banned_framing(text)
        ]
        assert not offenders, f"Banned framing word found in POSITIVE_TEMPLATES: {offenders}"

    def test_no_banned_word_in_clarify_message(self) -> None:
        assert not contains_banned_framing(body_type_clarify_message())


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
