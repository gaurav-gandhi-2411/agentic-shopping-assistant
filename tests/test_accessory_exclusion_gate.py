"""Regression tests for the 2026-07-25 accessory-exclusion gate
(src.agents.graph._GENERIC_WEAR_ASK_RE + the classify_item != "accessory"
filter in search_node), part of the "type-confusion" strict-eval miss bucket.

Root cause: a generic "outfit/look/wear" ask with no garment_type resolved
had no protection against a standalone accessory (bag/dupatta/necklace)
ranking into the top-5 as if it were an outfit — live-proven misses: "Women
Gotta Flower Purse" for "haldi outfit for women"/"bright haldi look for
women"/"mehendi outfit for women", "Men's Yellow - Dupatta" for "bright
haldi look for women", "Green Meena Polki Symmetry Set" (a necklace) for
"green outfit for mehendi".
"""
from __future__ import annotations

from src.agents.graph import _GENERIC_WEAR_ASK_RE
from src.agents.outfit.slots import classify_item


class TestGenericWearAskRegex:
    def test_outfit_word_detected(self) -> None:
        assert _GENERIC_WEAR_ASK_RE.search("haldi outfit for women")

    def test_bare_look_word_detected(self) -> None:
        assert _GENERIC_WEAR_ASK_RE.search("bright haldi look for women")

    def test_bare_wear_word_detected(self) -> None:
        assert _GENERIC_WEAR_ASK_RE.search("office wear for women")

    def test_specific_garment_query_not_flagged_by_regex_alone(self) -> None:
        # The regex alone doesn't gate anything — it's combined with
        # "garment_type is None" in search_node — but sanity-check it
        # doesn't match unrelated queries with no wear/outfit/look word.
        assert not _GENERIC_WEAR_ASK_RE.search("belt for men")
        assert not _GENERIC_WEAR_ASK_RE.search("red saree for a wedding")


class TestClassifyItemAccessoryDetection:
    """The actual filter predicate used by the gate — verifies the specific
    live-proven miss items are correctly classified as accessories."""

    def test_bag_is_accessory(self) -> None:
        assert classify_item("bag", "Women Gotta Flower Purse") == "accessory"

    def test_dupatta_is_accessory(self) -> None:
        assert classify_item("dupatta", "Men's Yellow - Dupatta") == "accessory"

    def test_necklace_is_accessory(self) -> None:
        assert classify_item("Necklace", "Green Meena Polki Symmetry Set") == "accessory"

    def test_kurta_is_not_accessory(self) -> None:
        assert classify_item("kurta", "Cotton Printed Kurta") != "accessory"

    def test_lehenga_is_not_accessory(self) -> None:
        assert classify_item("lehenga", "Red Embroidered Lehenga") != "accessory"
