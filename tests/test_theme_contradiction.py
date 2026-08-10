"""Regression tests for the 2026-08-10 theme-contradiction gate
(src.agents.outfit.slots.is_theme_contradiction), wired into the plain search
path (src.agents.graph.search_node) and mirrored in scripts/eval_strict.py's
_retrieve_pipeline.

Root cause: "bridal hamper" retrieved both product_type_name=="Hampers" (all
genuinely bridal) and product_type_name=="Gift Hamper" (themed Valentine/
Karwa Chauth/Diwali/Rakhi/Mother's Day -- zero bridal-themed), with no
mechanism checking a hamper's own named theme against the query's stated
theme. Scoped to exactly {"Hampers", "Gift Hamper"} -- a no-op for every
other product type.
"""
from __future__ import annotations

from src.agents.outfit.slots import is_theme_contradiction


class TestHamperThemeContradiction:
    def test_bridal_query_vs_valentine_hamper_is_contradiction(self) -> None:
        assert is_theme_contradiction(
            "bridal hamper", "Glow of Love Karwa Chauth Gift Hamper", "Gift Hamper"
        )

    def test_bridal_query_vs_bridal_item_not_a_contradiction(self) -> None:
        assert not is_theme_contradiction(
            "bridal hamper", "Mortantra X Zivame Bridal Hamper Box A", "Hampers"
        )

    def test_wedding_query_vs_bridal_item_same_group_not_a_contradiction(self) -> None:
        # "bridal" and "wedding" are near-synonyms in the same theme group.
        assert not is_theme_contradiction(
            "wedding hamper", "Mortantra X Zivame Bridal Hamper Box B", "Hampers"
        )

    def test_diwali_query_vs_rakhi_item_is_contradiction(self) -> None:
        assert is_theme_contradiction(
            "diwali gift hamper", "Elegant Bhaiya Bhabhi Rakhi Gift Hamper", "Gift Hamper"
        )

    def test_no_theme_word_in_query_never_fires(self) -> None:
        assert not is_theme_contradiction(
            "gift hamper for her", "Endless Love Valentine Hamper", "Gift Hamper"
        )

    def test_no_op_outside_hamper_product_types(self) -> None:
        # Same opposing theme words, but not a Hampers/Gift Hamper item --
        # this check must never fire outside its scoped product types.
        assert not is_theme_contradiction(
            "bridal lehenga", "Valentine Special Red Lehenga", "lehenga"
        )

    def test_case_insensitive_product_type_match(self) -> None:
        assert is_theme_contradiction(
            "bridal hamper", "Glow of Love Karwa Chauth Gift Hamper", "GIFT HAMPER"
        )

    def test_generic_themeless_hamper_never_flagged(self) -> None:
        assert not is_theme_contradiction(
            "bridal hamper", "Bloom & Glow Gift Hamper", "Gift Hamper"
        )
