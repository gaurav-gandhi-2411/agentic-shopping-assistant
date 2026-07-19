"""Unit tests for src.catalogue.cleaning — Phase A index-quality rules.

Fully self-contained (no index, no network, no LLM).
Run with: pytest tests/test_cleaning.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.catalogue.cleaning import (
    backfill_colours,
    clean_mojibake_columns,
    drop_religious_decor_items,
    drop_true_fabric_material,
    extract_colour,
    fix_mojibake,
    is_fabric_bolt_text,
    is_loungewear_text,
    is_religious_decor_item,
    reclassify_finished_sarees,
    recompute_derived_columns,
)

# ---------------------------------------------------------------------------
# Saree reclassification
# ---------------------------------------------------------------------------


class TestReclassifyFinishedSarees:
    def test_finished_saree_with_blouse_piece_reclassified(self) -> None:
        """A saree bundled with a blouse piece is a finished, shoppable garment."""
        df = pd.DataFrame(
            {
                "product_type_name": ["fabric_material"],
                "prod_name": ["Peach Printed Georgette Saree With Unstitched Blouse Piece"],
            }
        )
        out, n = reclassify_finished_sarees(df)
        assert n == 1
        assert out["product_type_name"].tolist() == ["saree"]

    def test_saree_brand_prefix_not_reclassified(self) -> None:
        """'Saree Mall' is a brand name — the product is genuinely unstitched fabric."""
        df = pd.DataFrame(
            {
                "product_type_name": ["fabric_material"],
                "prod_name": ["Saree mall Black Unstitched Dress Material"],
            }
        )
        out, n = reclassify_finished_sarees(df)
        assert n == 0
        assert out["product_type_name"].tolist() == ["fabric_material"]

    def test_blouse_piece_without_saree_not_reclassified(self) -> None:
        """A standalone blouse-piece fabric listing (no saree word) stays fabric_material."""
        df = pd.DataFrame(
            {
                "product_type_name": ["fabric_material"],
                "prod_name": ["Blue Printed Blouse Piece Silk"],
            }
        )
        out, n = reclassify_finished_sarees(df)
        assert n == 0

    def test_non_fabric_rows_untouched(self) -> None:
        """Rows not currently tagged fabric_material are never touched."""
        df = pd.DataFrame(
            {
                "product_type_name": ["dress"],
                "prod_name": ["Saree With Blouse Piece"],
            }
        )
        out, n = reclassify_finished_sarees(df)
        assert n == 0
        assert out["product_type_name"].tolist() == ["dress"]

    def test_mixed_batch_counts_correctly(self) -> None:
        df = pd.DataFrame(
            {
                "product_type_name": [
                    "fabric_material", "fabric_material", "fabric_material", "dress",
                ],
                "prod_name": [
                    "Saree With Unstitched Blouse Piece",
                    "Saree mall Black Unstitched Dress Material",
                    "Blue Printed Blouse Piece Silk",
                    "Black Dress",
                ],
            }
        )
        out, n = reclassify_finished_sarees(df)
        assert n == 1
        assert out["product_type_name"].tolist() == [
            "saree", "fabric_material", "fabric_material", "dress",
        ]


class TestDropTrueFabricMaterial:
    def test_drops_remaining_fabric_material_rows(self) -> None:
        df = pd.DataFrame(
            {"product_type_name": ["fabric_material", "dress", "saree"]},
        )
        out, n = drop_true_fabric_material(df)
        assert n == 1
        assert len(out) == 2
        assert "fabric_material" not in out["product_type_name"].tolist()

    def test_no_fabric_rows_is_noop(self) -> None:
        df = pd.DataFrame({"product_type_name": ["dress", "saree"]})
        out, n = drop_true_fabric_material(df)
        assert n == 0
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Religious decor exclusion (BUG 2, jewellery-inventory-gap wave 2026-07-19)
# ---------------------------------------------------------------------------


class TestIsReligiousDecorItem:
    @pytest.mark.parametrize(
        "text,expected",
        [
            # Real theamethyststore rows (type="Silver Idols").
            ("Balaji Temple Statue", True),
            ("Venkateswara 3D Idol", True),
            ("Perumal 3D Idol", True),
            ("Shivling With Snake 3D Solid Idol", True),
            # Real theamethyststore rows (type="Kum Kum Box").
            ("Beautiful Kum Kum Box", True),
            ("Vishaka 3D Spl Kum Kum Box", True),
            # Real theamethyststore rows (type="Fashion", mixed with genuine
            # jewellery in the same bucket) — religious photo frames.
            ("Lakshmi Frame", True),
            ("Lord Murugan Frame With Peacock 3D SPL", True),
            # Genuine jewellery sharing the "Fashion" type bucket must NOT match.
            ("Stunnig Brooch Pin", False),
            ("Unique Kundan Saree Pin", False),
            ("Style Charm Kundan Waist Charm", False),
            # Genuine jewellery elsewhere must NOT match.
            ("Simrath Short Necklace Set", False),
            ("Precious Silver Rakhi", False),
            # Real daivik rows — "Non Idol"/"Non-Idol" is a genuine jewellery
            # style descriptor (no deity-idol motif), not a religious statue.
            ("Antique Non Idol Purple Long Necklace with Earrings", False),
            ("Non idol kemp Peacock Earcuff with Jhumkas", False),
            ("Non-Idol Gold Polish Earrings", False),
            ("AD Non Idol Jada/Hair accessory", False),
            # A genuine idol/bell decor item must still be excluded even when
            # it also mentions jewellery-adjacent deity names.
            (
                "Premium Temple Style Antique Brass Bell with Lakshmi, Hanuman & Ganesha Idols (Price for Each)",
                True,
            ),
            ("", False),
            (None, False),
        ],
    )
    def test_cases(self, text: str | None, expected: bool) -> None:
        assert is_religious_decor_item(text) is expected


class TestDropReligiousDecorItems:
    def test_drops_idol_statue_kumkum_frame_rows(self) -> None:
        df = pd.DataFrame(
            {
                "prod_name": [
                    "Balaji Temple Statue",
                    "Venkateswara 3D Idol",
                    "Beautiful Kum Kum Box",
                    "Lakshmi Frame",
                    "Simrath Short Necklace Set",
                ],
            }
        )
        out, n = drop_religious_decor_items(df)
        assert n == 4
        assert out["prod_name"].tolist() == ["Simrath Short Necklace Set"]

    def test_no_decor_rows_is_noop(self) -> None:
        df = pd.DataFrame({"prod_name": ["Simrath Short Necklace Set", "James Shirt Button Clip"]})
        out, n = drop_religious_decor_items(df)
        assert n == 0
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Shared fabric-bolt runtime exclusion predicate
# ---------------------------------------------------------------------------


class TestIsFabricBoltText:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Kanjivaram Silk Saree with Blouse Piece", False),
            ("Sangria Blue Striped Saree & Embellished Blouse Piece", False),
            ("Blue Printed Blouse Piece Silk", True),
            ("Unstitched Dress Material Floral Print Cotton Blend", True),
            ("Saree mall Black Unstitched Dress Material", True),
            ("Black Floral Maxi Dress", False),
            ("", False),
            (None, False),
        ],
    )
    def test_cases(self, text: str | None, expected: bool) -> None:
        assert is_fabric_bolt_text(text) is expected


# ---------------------------------------------------------------------------
# Loungewear-in-dress-bucket exclusion predicate
# ---------------------------------------------------------------------------


class TestIsLoungewearText:
    @pytest.mark.parametrize(
        "text,expected",
        [
            # Real catalogue rows (data/processed/unified/catalogue.parquet,
            # product_type_name="dress", brand=libas) that caused the live
            # "minimalist wedding guest dress" -> sleepwear-kaftan bug.
            ("Green Geometric Printed Cotton Kaftan Night Dress", True),
            ("Maroon Geometric Printed Cotton Kaftan Night Dress", True),
            ("Blue Printed Cotton Night Dress", True),
            ("Black Printed Cotton Night Dress", True),
            ("Yellow Printed Cotton Kaftan Night Dress", True),
            # Real catalogue rows outside the "dress" bucket that use the same
            # phrase (product_type_name="kaftan" / "All Products") — the
            # predicate is text-only and correctly flags these too.
            ("Pink Cotton Printed Kaftan Nightdress", True),
            ("Lavender Printed Cotton Nightdress", True),
            # A genuine formal/wedding dress must not be excluded.
            ("Black Floral Maxi Dress", False),
            ("Embellished Wedding Guest Gown", False),
            # Bare "kaftan" without a sleep/night signal is a legitimate
            # standalone garment style, verified against the real catalogue:
            # fashor's "Kaftan Style Midi/Maxi Dress" rows and virgio's
            # "Kaftan Maxi Dress With Lace" are described in their own
            # detail_desc as day/eveningwear ("perfect for brunches, getaways,
            # or breezy evenings"; "intimate gatherings to festive evenings"),
            # not sleepwear — must NOT match.
            ("Solid Pleated Floral Cutwork Embroidered Kaftan Style Midi Dress - Blue", False),
            ("Colorful Ombre Printed Kaftan Style Maxi Dress - Multi", False),
            ("Vilasini|Viscose Kaftan Maxi Dress With Lace", False),
            # Other legitimate "kaftan"-garment-noun rows (kurta/top/co-ord)
            # from the real catalogue, unrelated to loungewear.
            ("Libas Women Blue & White Geometric Printed Pleated Cotton Kaftan Kurti", False),
            ("Sangria Mustard Yellow Floral Print Kaftan Top", False),
            # "Night suit"/"night shorts" are a distinct, already-correctly
            # tagged nightwear category — must not false-positive on bare
            # "night" without an adjacent "dress"/"gown".
            ("Men Solid Brown Night Suit Set", False),
            ("Solid Men Black Basic Shorts, Night Shorts, Gym Shorts", False),
            ("", False),
            (None, False),
        ],
    )
    def test_cases(self, text: str | None, expected: bool) -> None:
        assert is_loungewear_text(text) is expected


# ---------------------------------------------------------------------------
# Colour extraction / backfill
# ---------------------------------------------------------------------------


class TestExtractColour:
    def test_simple_word_boundary_match(self) -> None:
        assert extract_colour("Typography Men Round Neck Black T-Shirt") == "Black"

    def test_longest_match_wins_over_shorter_substring(self) -> None:
        assert extract_colour("Solid Men Round Neck Dark Blue T-Shirt") == "Dark Blue"

    def test_trailing_parenthetical_takes_first_colour(self) -> None:
        name = "Men Solid Cotton Satin Blend Straight Kurta  (Maroon, Dark Blue, Black, Pink)"
        # 2026-07-11: maroon now maps to its own catalogue-exact "Maroon" value,
        # not the approximate neighbour "Dark Red" — see intent_parser._COLOUR_MAP.
        assert extract_colour(name) == "Maroon"

    def test_falls_back_to_detail_desc(self) -> None:
        assert extract_colour("Plain Kurta", "This black kurta is festive") == "Black"

    def test_no_colour_found_returns_none(self) -> None:
        assert extract_colour("Square Neck Blouson Top with Tie-Knot at Back") is None

    def test_extended_vocab_burgundy_and_rust(self) -> None:
        # 2026-07-11: burgundy now maps to its own catalogue-exact "Burgundy"
        # value, not the approximate neighbour "Dark Red".
        assert extract_colour("Burgundy Wrap Dress") == "Burgundy"
        assert extract_colour("Rust Corduroy Jacket") == "Rust"


class TestBackfillColours:
    def test_fills_null_colour_from_name(self) -> None:
        df = pd.DataFrame(
            {
                "colour_group_name": [None, "Existing"],
                "prod_name": ["Solid Black T-Shirt", "Ignored Name"],
                "detail_desc": ["", ""],
            }
        )
        out, n = backfill_colours(df)
        assert n == 1
        assert out["colour_group_name"].tolist() == ["Black", "Existing"]

    def test_empty_string_treated_as_null(self) -> None:
        df = pd.DataFrame(
            {
                "colour_group_name": [""],
                "prod_name": ["Red Dress"],
                "detail_desc": [""],
            }
        )
        out, n = backfill_colours(df)
        assert n == 1
        assert out["colour_group_name"].tolist() == ["Red"]

    def test_unfillable_rows_left_null(self) -> None:
        df = pd.DataFrame(
            {
                "colour_group_name": [None],
                "prod_name": ["Square Neck Blouson Top"],
                "detail_desc": [""],
            }
        )
        out, n = backfill_colours(df)
        assert n == 0
        assert out["colour_group_name"].isna().all()

    def test_no_nulls_is_noop(self) -> None:
        df = pd.DataFrame(
            {"colour_group_name": ["Black"], "prod_name": ["x"], "detail_desc": [""]}
        )
        out, n = backfill_colours(df)
        assert n == 0


# ---------------------------------------------------------------------------
# Mojibake cleanup
# ---------------------------------------------------------------------------


class TestFixMojibake:
    def test_strips_nbsp_runs(self) -> None:
        assert fix_mojibake("Sneakers For Men\xa0\xa0(Grey)") == "Sneakers For Men (Grey)"

    def test_single_replacement_char_between_letters_becomes_apostrophe(self) -> None:
        assert fix_mojibake("These men�s track pants") == "These men's track pants"

    def test_stray_replacement_chars_stripped(self) -> None:
        result = fix_mojibake("best.��Team it with any formal")
        assert "�" not in result

    def test_none_and_empty_passthrough(self) -> None:
        assert fix_mojibake(None) is None
        assert fix_mojibake("") == ""

    def test_clean_text_unchanged(self) -> None:
        assert fix_mojibake("Plain Black T-Shirt") == "Plain Black T-Shirt"


class TestCleanMojibakeColumns:
    def test_reports_changed_row_counts(self) -> None:
        df = pd.DataFrame(
            {
                "prod_name": ["Sneakers For Men\xa0\xa0(Grey)", "Plain Shirt"],
                "detail_desc": ["clean desc", "clean desc"],
            }
        )
        out, stats = clean_mojibake_columns(df)
        assert stats["prod_name"] == 1
        assert stats["detail_desc"] == 0
        assert out["prod_name"].tolist() == ["Sneakers For Men (Grey)", "Plain Shirt"]


# ---------------------------------------------------------------------------
# Derived-column recomputation
# ---------------------------------------------------------------------------


class TestRecomputeDerivedColumns:
    def test_search_text_reflects_updated_colour_and_type(self) -> None:
        df = pd.DataFrame(
            {
                "prod_name": ["Peach Saree With Blouse Piece"],
                "product_type_name": ["saree"],
                "colour_group_name": ["Peach"],
                "department_name": ["Women"],
                "detail_desc": ["A lovely saree"],
                "index_group_name": ["Ladieswear"],
                "garment_group_name": ["N/A"],
            }
        )
        out = recompute_derived_columns(df)
        assert "saree" in out["search_text"].iloc[0]
        assert "Peach" in out["search_text"].iloc[0]
        assert out["facets"].iloc[0]["colour_group_name"] == "Peach"
        assert out["facets"].iloc[0]["product_type_name"] == "saree"
        assert "Peach" in out["display_name"].iloc[0]
