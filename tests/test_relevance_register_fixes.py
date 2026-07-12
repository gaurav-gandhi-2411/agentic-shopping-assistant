"""Regression tests for the two relevance-adjacent fixes (2026-07-10 sweep).

  A) sangeet "his look" footwear slot filled with ₹759 combat boots — the
     formality gate had no footwear vocabulary at all.
  B) a "pear shaped" request surfaced an explicitly "Plus Size"-branded kurta —
     body SHAPE must never be treated as a SIZE.
"""
from __future__ import annotations

from src.agents.outfit.body_type import demote_size_mismatched_items
from src.agents.outfit.slots import is_rugged_footwear_item


class TestRuggedFootwearRegister:
    def test_combat_boots_flagged(self) -> None:
        assert is_rugged_footwear_item("Boots For Men (Black)")

    def test_athletic_and_at_home_footwear_flagged(self) -> None:
        assert is_rugged_footwear_item("Men Running Sports Shoes")
        assert is_rugged_footwear_item("Classic White Sneakers")
        assert is_rugged_footwear_item("Comfort Flip-Flops Blue")
        assert is_rugged_footwear_item("Ethnic Slippers Pack")
        # Live-proof escape (2026-07-11): white walking shoes reached a sangeet
        # board after the first vocabulary pass missed "walking".
        assert is_rugged_footwear_item("Walking Shoes For Men (White)")

    def test_festive_appropriate_footwear_passes(self) -> None:
        assert not is_rugged_footwear_item("Black Oxford Shoes")
        assert not is_rugged_footwear_item("Tan Leather Loafers For Men")
        assert not is_rugged_footwear_item("Embellished Mojaris Gold")
        assert not is_rugged_footwear_item("Women Gold-Toned Embellished Heels")

    def test_word_boundary_no_false_positive(self) -> None:
        # "bootcut" must not be flagged by the "boots?" pattern.
        assert not is_rugged_footwear_item("Women Blue Bootcut Trousers")


class TestShapeIsNotSize:
    ITEMS = [
        {"prod_name": "FAZZN Women Plus Size Blue Solid Kurta"},
        {"prod_name": "W Women Green & White Embellished Kurta"},
        {"prod_name": "Fashor Plus-Size Floral Kurta"},
    ]

    def test_pear_query_demotes_plus_size_items(self) -> None:
        out = demote_size_mismatched_items(self.ITEMS, "pear shaped, suggest kurtas for women")
        assert out[0]["prod_name"].startswith("W Women")
        assert all("Plus" in it["prod_name"] for it in out[1:])
        assert len(out) == 3  # demoted, never dropped

    def test_explicit_plus_size_query_untouched(self) -> None:
        out = demote_size_mismatched_items(self.ITEMS, "plus size kurtas for a wedding")
        assert out == self.ITEMS

    def test_curvy_treated_as_stated_size_untouched(self) -> None:
        # "curvy" maps to plus_size in SYNONYMS — user stated it, keep order.
        assert demote_size_mismatched_items(self.ITEMS, "kurtas for curvy women") == self.ITEMS

    def test_no_shape_in_query_untouched(self) -> None:
        assert demote_size_mismatched_items(self.ITEMS, "green kurta under 2000") == self.ITEMS
