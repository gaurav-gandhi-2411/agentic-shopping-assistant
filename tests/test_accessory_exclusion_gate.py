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


class TestClassifyItemJewelleryVocabularyGap2026_07_30:
    """2026-07-30 catalogue audit: pluralization mismatches + genuine bridal/
    traditional jewellery vocabulary that was never in ACCESSORY_KEYWORDS at
    all — live-proven misses "Sukkhi Traditional Peacock Gold Plated Pair
    Kada For Women" (product_type_name literally "Kada") and "Goddess
    Traditional Haram" (product_type_name generic "Fashion", name contains
    "Haram") both surfaced for a generic haldi-look query undetected."""

    def test_kada_is_accessory(self) -> None:
        assert (
            classify_item("Kada", "Sukkhi Traditional Peacock Gold Plated Pair Kada For Women")
            == "accessory"
        )

    def test_haram_name_fallback_is_accessory(self) -> None:
        # product_type is a generic, otherwise-unresolved facet value —
        # relies on the combined product_type + name fallback in
        # classify_item() to catch "haram" in the freeform name.
        assert classify_item("Fashion", "Goddess Traditional Haram") == "accessory"

    def test_bangles_plural_is_accessory(self) -> None:
        assert classify_item("Bangles", "Gold Plated Bangles Set") == "accessory"

    def test_watches_plural_is_accessory(self) -> None:
        assert classify_item("Watches", "Analog Wrist Watches for Men") == "accessory"

    def test_potli_is_accessory(self) -> None:
        assert classify_item("Potli", "Embroidered Silk Potli Bag") == "accessory"

    def test_oddiyanam_hip_belts_is_accessory(self) -> None:
        assert (
            classify_item("Oddiyanam/Hip Belts", "Bridal Nethra Diamond Like Oddiyanam")
            == "accessory"
        )

    def test_juda_typo_facet_is_accessory(self) -> None:
        assert classify_item("Judo", "Multicolor Gold Plated Mishr Juda") == "accessory"

    def test_kalangi_is_accessory(self) -> None:
        assert classify_item("Kalangi", "Kings of Rajasthan Haritansh Kalangi") == "accessory"

    def test_mangalsutra_is_accessory(self) -> None:
        assert classify_item("Mangalsutra", "Gold Plated Mangalsutra Pendant") == "accessory"

    def test_pocket_squares_plural_is_accessory(self) -> None:
        assert classify_item("Pocket Squares", "Silk Pocket Squares Set") == "accessory"


class TestClassifyItemUnknownClassKeywordCoverageAudit2026_07_30:
    """2026-07-30 unknown-class keyword-coverage audit: real catalogue
    product_type_name facet values that classify_item() resolved to
    "unknown" before this fix, verified against real per-row
    classify_item(product_type_name, prod_name) — see slots.py's own
    keyword-set comments for the full row-count audit trail per keyword.
    """

    def test_nose_pin_space_variant_is_accessory(self) -> None:
        assert classify_item("Nose Pin", "Oxidised Nose Pin") == "accessory"

    def test_maang_tikka_double_k_variant_is_accessory(self) -> None:
        assert classify_item("Maang Tikka", "Bridal Kundan Maang Tikka") == "accessory"

    def test_kamarband_is_accessory(self) -> None:
        assert (
            classify_item("Kamarband", "Sukkhi Alluring Pearl Gold Plated Kamarband For Women")
            == "accessory"
        )

    def test_wallet_is_accessory(self) -> None:
        assert classify_item("Wallet", "Men's Leather Bifold Wallet") == "accessory"

    def test_bow_tie_is_accessory(self) -> None:
        assert classify_item("Fashion", "Classic Satin Bow Tie") == "accessory"

    def test_footwear_facet_alone_is_footwear(self) -> None:
        assert classify_item("footwear", "Running Sneakers") == "footwear"

    def test_chappals_is_footwear(self) -> None:
        assert classify_item("Fashion", "1 Pair of Chappals") == "footwear"

    def test_mules_is_footwear(self) -> None:
        assert classify_item("Fashion", "Suede Leather Mules") == "footwear"

    def test_outerwear_facet_alone_is_outerwear(self) -> None:
        assert classify_item("outerwear", "Winter Layer") == "outerwear"

    def test_shrug_is_outerwear(self) -> None:
        assert classify_item("Fashion", "Winter Shrug") == "outerwear"

    def test_knitwear_is_western_top(self) -> None:
        assert classify_item("knitwear", "High Neck Textured Zipper Sweater") == "western_top"

    def test_bottomwear_is_western_bottom(self) -> None:
        assert classify_item("Bottomwear", "Men Cargos") == "western_bottom"

    def test_coord_no_hyphen_is_western_one_piece(self) -> None:
        assert classify_item("coord", "Blue Floral Printed Cotton Blend Co-Ord Set") == (
            "western_one_piece"
        )

    def test_phiran_is_ethnic_top(self) -> None:
        assert classify_item("Fashion", "Kashifa Black Pure Woollen Phiran") == "ethnic_top"

    def test_plazzo_typo_is_ethnic_bottom(self) -> None:
        assert classify_item("Fashion", "Narangi Cotton Slub Plazzos") == "ethnic_bottom"

    def test_achkan_is_men_formalwear(self) -> None:
        assert classify_item("Fashion", "Pink Brocade Embroidered Achkan") == "men_formalwear"

    def test_jodhpuri_is_men_formalwear(self) -> None:
        assert classify_item("Jodhpuri", "Blue Jodhpuri Suit") == "men_formalwear"

    def test_swimwear_stays_unknown_deliberately_not_reclassified(self) -> None:
        # Data-mislabeled facet value: samples as underwear ("Men Brief (Pack
        # of 5)"), not beachwear, in this catalogue's pipeline — must NEVER
        # be classified as a wearable top/one_piece (would let underwear
        # fill an outfit slot, worse than leaving it "unknown").
        assert classify_item("swimwear", "Men Brief (Pack of 5)") == "unknown"

    def test_vest_stays_unknown_deliberately_not_reclassified(self) -> None:
        # Same reasoning as swimwear above: samples as undershirts
        # ("SayItLoud Men Vest (Pack of 5)"), not stylish outer vests.
        assert classify_item("vest", "SayItLoud Men Vest (Pack of 5)") == "unknown"


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
