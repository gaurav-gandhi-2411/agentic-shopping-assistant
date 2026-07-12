"""Regression tests for the 2026-07-11 single-garment set-exclusion gate on
the plain search path (src.agents.graph._SET_INTENT_RE / search_node).

Root cause: "kurti under 1500" surfaced a "Solid Kaftan Kurta with Abstract
Patchwork Palazzo" — a 2-3 piece SET listing — when the user named ONE
garment. Reuses composer.is_multi_piece_set (never reimplemented). This was
the single largest strict-eval miss bucket (11/29, "set-not-single").
"""
from __future__ import annotations

from src.agents.graph import _OUTFIT_INTENT_RE, _SET_INTENT_RE
from src.agents.outfit.slots import is_multi_piece_set


class TestSetIntentRegex:
    def test_explicit_set_word_detected(self) -> None:
        assert _SET_INTENT_RE.search("kurta set for women")
        assert _SET_INTENT_RE.search("kurti sets under 2000")

    def test_combo_and_coord_detected(self) -> None:
        assert _SET_INTENT_RE.search("ethnic combo for men")
        assert _SET_INTENT_RE.search("co-ord set for women")

    def test_plain_single_garment_query_not_flagged(self) -> None:
        assert not _SET_INTENT_RE.search("kurti under 1500")
        assert not _SET_INTENT_RE.search("red saree for a wedding")

    def test_outfit_word_also_legitimizes_a_set(self) -> None:
        # search_node checks both regexes with OR — either legitimizes a set.
        assert _OUTFIT_INTENT_RE.search("style me a kurti under 1500")


class TestIsMultiPieceSetCatchesTheLiveEscape:
    """The catalogue's dominant multi-piece naming convention is "<garment>
    with <garment>" — very often with NO literal "set"/"sets" word at all, so
    the two pre-existing signals (product_type in _SET_PRODUCT_TYPES, or the
    literal word "set(s)" + 2 nouns) missed most real strict-eval-labeled
    misses. Every case below is a real hand-labeled miss from
    eval/fixtures/strict_gold_labels.yaml."""

    def test_kaftan_kurta_with_palazzo_is_a_set(self) -> None:
        assert is_multi_piece_set(
            "kurta", "Solid Kaftan Kurta with Abstract Patchwork Palazzo - Black"
        )

    def test_kurta_with_pant_is_a_set(self) -> None:
        assert is_multi_piece_set(
            "kurta", "Ethnic Floral Printed A-Line Flared Kurta with Pant - Peach"
        )

    def test_anarkali_with_palazzo_and_dupatta_is_a_set(self) -> None:
        assert is_multi_piece_set(
            "kurta",
            "Solid Ethnic Embroidered Anarkali Flared Kurta with Palazzo and Dupatta - Lavender",
        )

    def test_plural_palazzos_recognised(self) -> None:
        assert is_multi_piece_set(
            "kurta", "Sangria Wine-Coloured & Golden Khari Printed Kurta with Palazzos"
        )

    def test_plain_kurti_is_not_a_set(self) -> None:
        assert not is_multi_piece_set("kurti", "Multi Printed Muslin Straight Short Kurti")

    def test_single_garment_with_adjective_not_flagged(self) -> None:
        # "with" alone isn't enough — needs >=2 distinct garment nouns.
        assert not is_multi_piece_set("kurta", "Wine Embroidered Winter Kurta")
