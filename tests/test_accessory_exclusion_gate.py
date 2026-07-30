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
from src.agents.outfit.slots import NON_OUTFIT_ITEM_CLASSES, classify_item


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

    def test_pendant_is_accessory(self) -> None:
        # 2026-07-25 out-of-sample validation finding: "Pendant" (126
        # catalogue rows, all jewellery) was never in ACCESSORY_KEYWORDS at
        # all -- slipped past this gate for "office outfit for men".
        assert classify_item("Pendant", "Eclipse Black Shivling Pendant for Men") == "accessory"

    def test_lehenga_is_not_accessory(self) -> None:
        assert classify_item("lehenga", "Red Embroidered Lehenga") != "accessory"


class TestNonOutfitItemClassesCoversFootwear:
    """2026-07-30 fix: classify_item() resolves "footwear" as its own class,
    disjoint from "accessory" — the gate's original `!= "accessory"` check
    never caught a standalone shoe. Live-proven miss: "Ladies Triveni
    Kolhapuri Chappal" (article_id 8174367375495) ranked into the top-5 for
    "bright haldi look for women" (occ_adv_002 in strict_gold_labels.yaml) —
    a lone shoe is no more "an outfit" than a lone bag."""

    def test_standalone_footwear_is_excluded_by_gate(self) -> None:
        item_class = classify_item("footwear", "Ladies Triveni Kolhapuri Chappal")
        assert item_class == "footwear"
        assert item_class in NON_OUTFIT_ITEM_CLASSES

    def test_kurta_is_not_in_non_outfit_classes(self) -> None:
        assert classify_item("kurta", "Cotton Printed Kurta") not in NON_OUTFIT_ITEM_CLASSES
