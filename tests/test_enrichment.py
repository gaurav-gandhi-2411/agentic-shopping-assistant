"""Unit tests for src.catalogue.enrichment — Phase C rule-based facet extraction.

Fully self-contained (no index, no network, no LLM). The Ollama-fallback path
lives in scripts/enrich_attributes.py and is exercised separately by
tests/test_enrich_attributes.py (requires_ollama-marked calls only).
"""

from __future__ import annotations

import pytest

from src.catalogue.enrichment import (
    FACET_VOCAB,
    append_enrichment_to_search_text,
    extract_fabric,
    extract_occasion,
    extract_season,
    extract_style,
    merge_enrichment,
    rules_pass,
)

# ---------------------------------------------------------------------------
# Season
# ---------------------------------------------------------------------------


class TestExtractSeason:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Lightweight Linen Summer Wear Shirt", "summer"),
            ("Cozy Wool Winter Sweater", "winter"),
            ("Waterproof Monsoon Raincoat", "monsoon"),
            ("All-season basic tee", "all season"),
            # Weaker fabric/sleeve inference (no direct season word).
            ("Sleeveless Linen Blend Top", "summer"),
            ("Fleece Quilted Jacket", "winter"),
            # Direct word wins over conflicting fabric inference.
            ("Winter wool linen sleeveless top", "winter"),
        ],
    )
    def test_season_cases(self, text: str, expected: str) -> None:
        assert extract_season(text) == expected

    def test_no_signal_returns_none(self) -> None:
        assert extract_season("Plain Round Neck Cotton T-Shirt") is None

    def test_none_and_empty_passthrough(self) -> None:
        assert extract_season(None) is None
        assert extract_season("") is None

    def test_bare_cotton_alone_is_not_a_season_signal(self) -> None:
        """Cotton alone is worn year-round — too weak a signal on its own."""
        assert extract_season("Cotton Kurta with full sleeve") is None


# ---------------------------------------------------------------------------
# Occasion
# ---------------------------------------------------------------------------


class TestExtractOccasion:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Bridal Wedding Lehenga with Sangeet vibes", "wedding"),
            ("Festive Diwali Collection Kurta", "festive"),
            ("Party wear Sequin Top", "party"),
            ("Formal Office Wear Trousers", "office"),
            ("Casual Everyday Wear Tee", "casual"),
        ],
    )
    def test_occasion_cases(self, text: str, expected: str) -> None:
        assert extract_occasion(text) == expected

    def test_most_specific_wins_wedding_over_party(self) -> None:
        assert extract_occasion("Party wear wedding lehenga") == "wedding"

    def test_no_signal_returns_none(self) -> None:
        assert extract_occasion("Round Neck Cotton T-Shirt") is None


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


class TestExtractStyle:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Bohemian Boho Maxi Dress", "boho"),
            ("Minimalist Clean Lines Top", "minimalist"),
            ("Streetwear Oversized Hoodie", "streetwear"),
            ("Indo-Western Fusion Wear Dress", "ethnic fusion"),
            ("Athleisure Track Set", "athleisure"),
            ("Timeless classic style shirt", "classic"),
        ],
    )
    def test_style_cases(self, text: str, expected: str) -> None:
        assert extract_style(text) == expected

    def test_bare_classic_collar_not_treated_as_style_signal(self) -> None:
        """'Classic collar' describes a collar shape, not the garment's overall style."""
        assert extract_style("Regular fit shirt with a classic button-down collar") is None

    def test_no_signal_returns_none(self) -> None:
        assert extract_style("Plain Round Neck Cotton T-Shirt") is None


# ---------------------------------------------------------------------------
# Fabric
# ---------------------------------------------------------------------------


class TestExtractFabric:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("70% cotton and 30% linen blend shirt", "linen"),
            ("Pure Chanderi Silk Saree", "chanderi"),
            ("Georgette Printed Saree", "georgette"),
            ("Slim Fit Denim Jeans", "denim"),
            ("Polyester Sports Tee", "polyester"),
            ("Lycra Spandex Leggings", "spandex"),
            # Newly added vocab words (2026-07-07 sample-validation gap fix).
            ("100% Acrylic Cable Knit Sweater", "acrylic"),
            ("Soft Muslin Cotton Kurta", "muslin"),
            ("Genuine Leather Biker Jacket", "leather"),
            ("Pure Cashmere Pullover", "cashmere"),
            ("Ribbed Corduroy Trousers", "corduroy"),
            ("Self-Design Tweed Blazer", "tweed"),
            ("95% Modal 5% Polyester Top", "modal"),
            ("Cotton Jersey Cropped Top", "cotton"),
            ("Single Jersey Fabric T-Shirt", "jersey"),
            # Regional-label synonyms fold into their canonical fiber. Spandex
            # is checked ahead of cotton/polyester in _FABRIC_RULES (a stretch
            # component is usually the more distinctive tag for a blend), so
            # "elastane" wins even when cotton/polyester also appear.
            ("97% Cotton, 3% Elastane Top", "spandex"),
            ("5% Elastane, 95% Polyester Top", "spandex"),
            ("56% Polyamide, 44% Nylon Legging", "nylon"),
        ],
    )
    def test_fabric_cases(self, text: str, expected: str) -> None:
        assert extract_fabric(text) == expected

    def test_elastane_synonym_maps_to_spandex_when_no_other_fiber(self) -> None:
        assert extract_fabric("5% Elastane Leggings") == "spandex"

    def test_polyamide_synonym_maps_to_nylon(self) -> None:
        assert extract_fabric("100% Polyamide Swimsuit") == "nylon"

    def test_no_signal_returns_none(self) -> None:
        assert extract_fabric("Solid Round Neck Top") is None


# ---------------------------------------------------------------------------
# rules_pass — combined extraction
# ---------------------------------------------------------------------------


class TestRulesPass:
    def test_combines_all_four_facets(self) -> None:
        result = rules_pass(
            prod_name="Linen Blend Popover Shirt",
            product_type_name="shirt",
            detail_desc="Sleeveless summer wear, breathable cotton and linen blend, casual outings.",
        )
        assert result == {
            "season": "summer",
            "occasion_tag": "casual",
            "style_tag": None,
            "fabric": "linen",
        }

    def test_honest_none_when_no_signal_anywhere(self) -> None:
        result = rules_pass(
            prod_name="Square Neck Blouson Top",
            product_type_name="top",
            detail_desc="A round neck top with tie-knot at back.",
        )
        assert result == {
            "season": None,
            "occasion_tag": None,
            "style_tag": None,
            "fabric": None,
        }

    def test_handles_none_inputs(self) -> None:
        result = rules_pass(None, None, None)
        assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# merge_enrichment — rules-priority merge with LLM vocab validation
# ---------------------------------------------------------------------------


class TestMergeEnrichment:
    def test_rules_value_always_wins_over_llm(self) -> None:
        rules = {"season": "summer", "occasion_tag": None, "style_tag": None, "fabric": None}
        llm = {"season": "winter", "occasion_tag": "office", "style_tag": None, "fabric": "cotton"}
        merged, source = merge_enrichment(rules, llm)
        assert merged["season"] == "summer"
        assert source["season"] == "rule"
        assert merged["occasion_tag"] == "office"
        assert source["occasion_tag"] == "llm"
        assert merged["fabric"] == "cotton"
        assert source["fabric"] == "llm"

    def test_llm_value_outside_vocab_discarded_to_none(self) -> None:
        """Hallucination guard: an invented/paraphrased LLM label is never kept."""
        rules = {"season": None, "occasion_tag": None, "style_tag": None, "fabric": None}
        llm = {"season": "spring", "occasion_tag": None, "style_tag": "boho-chic", "fabric": None}
        merged, source = merge_enrichment(rules, llm)
        assert merged["season"] is None
        assert source["season"] == "none"
        assert merged["style_tag"] is None
        assert source["style_tag"] == "none"

    def test_no_llm_pass_all_gaps_stay_none(self) -> None:
        rules = {"season": None, "occasion_tag": None, "style_tag": None, "fabric": None}
        merged, source = merge_enrichment(rules, None)
        assert all(v is None for v in merged.values())
        assert all(s == "none" for s in source.values())

    def test_covers_every_vocab_facet(self) -> None:
        rules = {k: None for k in FACET_VOCAB}
        merged, source = merge_enrichment(rules, None)
        assert set(merged) == set(FACET_VOCAB)
        assert set(source) == set(FACET_VOCAB)


# ---------------------------------------------------------------------------
# append_enrichment_to_search_text
# ---------------------------------------------------------------------------


class TestAppendEnrichmentToSearchText:
    def test_appends_non_null_tags(self) -> None:
        enrichment = {
            "season": "summer",
            "occasion_tag": "casual",
            "style_tag": None,
            "fabric": "linen",
        }
        result = append_enrichment_to_search_text("Base search text.", enrichment)
        assert result == "Base search text.. summer. casual. linen."

    def test_no_tags_returns_unchanged(self) -> None:
        enrichment = {"season": None, "occasion_tag": None, "style_tag": None, "fabric": None}
        assert append_enrichment_to_search_text("Base search text.", enrichment) == "Base search text."

    def test_empty_base_text_still_appends(self) -> None:
        enrichment = {"season": "winter", "occasion_tag": None, "style_tag": None, "fabric": None}
        assert append_enrichment_to_search_text("", enrichment) == "winter."
