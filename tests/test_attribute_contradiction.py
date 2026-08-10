"""Regression tests for the 2026-07-25 attribute-contradiction gate
(src.agents.outfit.slots.is_attribute_contradiction), wired into the plain
search path (src.agents.graph.search_node) and mirrored in
scripts/eval_strict.py's _retrieve_pipeline.

Root cause: plain search relies entirely on embedding similarity, which
frequently ranks a "Slim Fit" item highly for a "straight fit" query since
the two phrases sit close in embedding space despite being product-listing
OPPOSITES in this catalogue's own marketing vocabulary. This was the
"attribute-contradiction" strict-eval miss bucket.
"""
from __future__ import annotations

from src.agents.outfit.slots import is_attribute_contradiction


class TestFitTightnessCamp:
    def test_slim_query_vs_straight_item_is_contradiction(self) -> None:
        assert is_attribute_contradiction(
            "straight fit kurta for women",
            "Fabindia Women Navy Blue Striped Viscose Silk Slim Fit Kurta",
            "Straight shape with regular style",
        )

    def test_slim_and_skinny_are_compatible_not_contradiction(self) -> None:
        # Same "tight" camp -- near-synonyms, not opposites.
        assert not is_attribute_contradiction(
            "slim fit jeans for men", "Skinny Fit Jeans", "desc"
        )

    def test_straight_and_regular_are_compatible_not_contradiction(self) -> None:
        # Same "loose/not-tight" camp -- near-synonyms, not opposites.
        assert not is_attribute_contradiction(
            "straight fit kurta for women", "Regular Fit Kurta", "desc"
        )

    def test_relaxed_and_oversized_are_compatible_not_contradiction(self) -> None:
        assert not is_attribute_contradiction(
            "relaxed fit kurta for men", "Oversized Kurta", "desc"
        )

    def test_no_tracked_word_in_query_never_fires(self) -> None:
        assert not is_attribute_contradiction(
            "printed kurta for men", "Men Wine Printed Geometric Kurta", "desc"
        )


class TestSilhouetteFlareCamp:
    def test_anarkali_is_a_line_not_a_contradiction(self) -> None:
        # An anarkali kurta IS a-line by definition -- verified against real
        # catalogue desc text (article 7797797454046): "...heritage anarkali
        # style with a graceful flared silhouette...".
        assert not is_attribute_contradiction(
            "a-line kurta", "Anarkali Kurta", "desc"
        )

    def test_regular_fit_contradicts_a_line(self) -> None:
        # Real hand-label evidence: "'Regular Fit' contradicts 'a-line'" --
        # a kurta's overall cut is either flared/a-line or straight/regular,
        # not both.
        assert is_attribute_contradiction(
            "a-line kurta for women", "Regular Fit Kurta", "desc"
        )

    def test_bodycon_contradicts_fit_and_flare(self) -> None:
        assert is_attribute_contradiction(
            "bodycon dress for women", "Fit And Flare Dress", "desc"
        )

    def test_a_line_contradicts_bodycon(self) -> None:
        assert is_attribute_contradiction(
            "a-line kurta for women", "Bodycon Kurta", "desc"
        )

    def test_exact_confirmation_never_a_contradiction(self) -> None:
        assert not is_attribute_contradiction(
            "anarkali kurta for women",
            "Off White Cotton Anarkali Kurta",
            "flared silhouette",
        )


class TestFlatGroups:
    def test_rise_mismatch_is_contradiction(self) -> None:
        assert is_attribute_contradiction(
            "high waisted jeans for women", "Mid Rise Jeans", "desc"
        )

    def test_breasted_mismatch_is_contradiction(self) -> None:
        assert is_attribute_contradiction(
            "single breasted blazer for men", "Double Breasted Blazer", "desc"
        )

    def test_neckline_mismatch_is_contradiction(self) -> None:
        assert is_attribute_contradiction(
            "v-neck top for women", "Halter Neck Top", "desc"
        )

    def test_unstated_word_never_penalised(self) -> None:
        # Item states a neckline the query never asked about -- no signal
        # to contradict, must not fire.
        assert not is_attribute_contradiction(
            "cotton kurta for women", "Round Neck Cotton Kurta", "desc"
        )


class TestBodyconOwnPole:
    """2026-07-25 out-of-sample validation finding: an item with the
    structured facet line "Silhouette: Straight kurta" surfaced for a
    "bodycon kurta" query in the held-out set. Bodycon was previously folded
    into the same camp as straight-cut/straight-fit/regular-fit (as if
    compatible with them) -- too coarse, since bodycon (body-hugging
    throughout) and straight (unflared but not tight) are genuinely distinct
    kurta silhouettes. Bodycon is now its own pole, opposing BOTH the flared
    camp (a-line/anarkali/fit-and-flare) and the straight/regular camp."""

    def test_bodycon_vs_explicit_silhouette_straight_facet_line(self) -> None:
        assert is_attribute_contradiction(
            "bodycon kurta for women", "Black V-Neck Kurta", "Silhouette: Straight kurta"
        )

    def test_bodycon_vs_bare_straight_kurta_prose_not_flagged(self) -> None:
        # Deliberately narrow: bare "straight kurta" prose appears in 16% of
        # catalogue kurta rows (the default silhouette description) -- only
        # the structured "Silhouette: Straight" facet line is tracked.
        assert not is_attribute_contradiction(
            "bodycon kurta for women", "Straight Kurta", "a straight kurta for daily wear"
        )

    def test_bodycon_vs_a_line_still_contradicts(self) -> None:
        assert is_attribute_contradiction(
            "a-line kurta for women", "Bodycon Kurta", "desc"
        )

    def test_bodycon_explicit_confirmation_never_a_contradiction(self) -> None:
        assert not is_attribute_contradiction(
            "bodycon dress for women", "Black Bodycon Dress", "ruched bodycon dress"
        )


class TestGownLengthCamp:
    """2026-08-10 (occasion-register wave, Cluster A): "gown" implies a
    long/floor-length Western silhouette, previously untracked entirely — a
    query naming "gown" had no mechanism opposing an item that explicitly
    states a short/fitted silhouette word. Catalogue-audited across all 249
    catalogue rows containing "gown": mini/bodycon/slim-fit co-occur 0/249
    each, midi 3/249 (0.4%, sampled clean)."""

    def test_gown_vs_mini_is_contradiction(self) -> None:
        assert is_attribute_contradiction(
            "gown for reception", "Women Waist tied Dress", "Black ruched mini dress"
        )

    def test_gown_vs_bodycon_is_contradiction(self) -> None:
        assert is_attribute_contradiction(
            "black gown for a cocktail party", "Black Off-Shoulder Party Dress",
            "Slim fit bodycon silhouette",
        )

    def test_gown_vs_slim_fit_is_contradiction(self) -> None:
        assert is_attribute_contradiction(
            "wine coloured gown for reception", "Solid Wrap Fit & Flare Midi Dress",
            "Slim fit design",
        )

    def test_gown_vs_midi_is_contradiction(self) -> None:
        assert is_attribute_contradiction(
            "gown for reception", "Wrap Dress", "A flattering midi-length wrap dress"
        )

    def test_gown_explicit_confirmation_never_a_contradiction(self) -> None:
        assert not is_attribute_contradiction(
            "gown for reception", "Floor Length Gown", "an elegant flowing gown"
        )

    def test_no_gown_in_query_never_fires(self) -> None:
        assert not is_attribute_contradiction(
            "mini dress for a party", "Bodycon Mini Dress", "desc"
        )

    def test_gown_vs_a_line_not_a_contradiction(self) -> None:
        # Deliberately NOT opposed to the flare camp -- a gown can
        # legitimately be A-line-silhouetted; unaudited pairing, left alone.
        assert not is_attribute_contradiction(
            "gown for reception", "A-Line Evening Gown", "desc"
        )

    def test_a_line_vs_straight_still_contradicts_unaffected(self) -> None:
        # Regression pin: the a-line-vs-straight/regular pair (unrelated to
        # bodycon's new pole) must be unchanged by this fix.
        assert is_attribute_contradiction(
            "a-line kurta for women", "Regular Fit Kurta", "desc"
        )
