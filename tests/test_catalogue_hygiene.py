"""Tests for src/catalogue/hygiene.py (Wave 1b V3(b2)(b)): deterministic category rules."""

from __future__ import annotations

import pandas as pd
import pytest

from src.catalogue.hygiene import (
    apply_hygiene,
    classify_type,
    derive_category,
    is_garment_type_name,
    title_head,
)


@pytest.mark.parametrize(
    ("ptype", "title", "category", "gtype"),
    [
        ("kurta", "Blue Embroidered Straight Kurta with Palazzo & Dupatta", "garment", "kurta"),
        ("lehenga", "Red Embellished Lehenga Choli", "garment", "lehenga"),
        ("necklace", "Kundan Choker Necklace Set", "jewellery", "necklace"),
        ("Earrings", "Gold Plated Jhumka Earrings", "jewellery", "earrings"),
        ("footwear", "Tan Leather Kolhapuri Chappal", "footwear", "footwear"),
        ("bag", "Embroidered Silk Potli", "accessory", "bag"),
        ("Rakhi", "Evil Eye Thread Rakhi For Brother", "accessory", "rakhi"),
        ("Fashion", "Kundan Jadau Chandbali Earrings", "jewellery", "earrings"),
        ("Fashion", "Bridal Chooda Set", "jewellery", "bangles"),
        ("Clothing Accessories", "Men Solid Mid-Calf/Crew (Pack of 6)", "accessory", "accessory"),
        ("Fashion", "xyzzy plugh", "unknown", None),
    ],
)
def test_category_from_type_then_title(
    ptype: str, title: str, category: str, gtype: str | None
) -> None:
    res = derive_category(ptype, title)
    assert (res.category, res.garment_type) == (category, gtype)
    assert res.quarantine_reason == ""


@pytest.mark.parametrize(
    ("ptype", "title"),
    [
        ("lehenga", "Brocade Lehenga Box"),  # the trousseau box typed as a lehenga
        ("Gift Hamper", "Bridal Jewellery Organiser"),  # packaging-only stays dropped
        ("Gift Hamper", "Wedding Trunk"),
        ("Gift Hamper", "Diwali E-Gift Card"),
        ("Gift Hamper", "Rose Perfume Gift Set Perfume"),
        ("Gift Card", "Happy Mother's Day : Gift Card"),
        ("Fragrance", "Mira| Oriental Floral Gourmand Perfume, 20 Ml"),
        ("Fashion", "Embroidered Navy Blue Bangle Box"),
        ("Covers", "Set of 2: Sky Blue Jacquard Cushion Covers"),
        ("Fashion", "Father's Day E-Gift Card"),  # "e gift" must not swallow "gift card"
        ("Fashion", "Rakhi E-Gift Card - The Perfect Digital Gift for Rakshabandhan"),  # tagline
        ("kurta", "Shirt And Kurta Organiser"),  # trailing noun of a coordinated title
    ],
)
def test_non_fashion_is_detected(ptype: str, title: str) -> None:
    assert derive_category(ptype, title).category == "non_fashion"


@pytest.mark.parametrize(
    ("ptype", "title", "category"),
    [
        # Bare "trunks" are men's swim/underwear trunks, a garment.
        ("swimwear", "Men Trunks", "garment"),
        # A garment sold inside a gift box is still the garment.
        ("kurta", "Men's Green Kurta Rakhi Gift Box for Brother | Includes Rakhi", "garment"),
        # "Box" as a modifier, not the head noun.
        ("necklace", "Reversible 2 sided Box Lakshmi and Ruby green necklace", "jewellery"),
        # "Bottle" as a colour.
        ("Earring", "Gold Plated Bottle Green Kundan Studded Dangler Earrings", "jewellery"),
        # A jewellery set sold in a box is the jewellery.
        ("Necklace Sets", "Sparkling Essentials Cluster Setting Clear CZ Box Set", "jewellery"),
        # Box pleats and box fits are garments.
        ("skirt", "Black Box Pleated Skirt", "garment"),
        ("shirt", "Brown Box Fit Shirt", "garment"),
    ],
)
def test_non_fashion_false_positives_are_avoided(ptype: str, title: str, category: str) -> None:
    res = derive_category(ptype, title)
    assert res.category == category
    assert res.quarantine_reason == ""


@pytest.mark.parametrize(
    ("ptype", "title"),
    [
        ("Gift Hamper", "Floral Bliss Valentine Hamper"),
        ("Hampers", "Mortantra X Zivame Bridal Hamper Box B"),
        ("Rakhi Hamper", "Royal Bro Rakhi Hamper for Brother"),
        ("Fashion", "Shubh Labh Hamper"),
    ],
)
def test_hampers_stay_searchable_as_accessory(ptype: str, title: str) -> None:
    res = derive_category(ptype, title)
    assert res.category == "accessory"
    assert res.garment_type == "hamper"


@pytest.mark.parametrize(
    "title",
    [
        "Forever Glow Valentine Jewellery Gift Box",
        "His Timeless Style Valentine Men�s Gift Box",
        "Key to My Heart Valentine Pendant Gift Box",
        "Sealed With Love Valentine Jewellery Gift Box",
        "Voylla Gift of Happiness Diwali Celebration Gift Box",
        "Voylla Festival Fun Diwali Celebration Box",
        "Voylla Festival of Joy Diwali Celebration Box",
        "Voylla Festive Celebration Diwali Gift Box",
    ],
)
def test_hamper_typed_gift_box_rows_are_kept_as_accessory(title: str) -> None:
    res = derive_category("Gift Hamper", title)
    assert (res.category, res.garment_type) == ("accessory", "hamper")
    assert res.quarantine_reason == ""


def test_untyped_gift_box_is_still_dropped() -> None:
    # The exemption is keyed on the merchant's hamper TYPE, not on the title alone.
    assert derive_category("Fashion", "Diwali Celebration Gift Box").category == "non_fashion"


def test_dupatta_typed_garment_set_is_garment_not_quarantined() -> None:
    # The feed normaliser typed "<Garment> Set With Dupatta" listings as dupatta.
    res = derive_category("dupatta", "Green Printed Cotton Straight Suit Set With Dupatta")
    assert res.category == "garment"
    assert res.quarantine_reason == ""
    # A real dupatta stays an accessory.
    assert derive_category("dupatta", "Maroon Embroidered Net Dupatta").category == "accessory"
    # "Kurta Dupatta Set" (dupatta named last) is a garment set.
    assert derive_category("dupatta", "ZOHAYR ROYAL BLUE KURTA DUPATTA SET").category == "garment"


@pytest.mark.parametrize(
    ("ptype", "title"),
    [
        ("Bangles", "Jodhpuri Bridal Lehenga"),  # the architect's example: Lehenga vs bangles
        ("shorts", "Designer Short Polki Party Wear Necklace"),
        ("jeans", "Denim Charm Kolhapuri Chappal for Men"),
        ("earrings", "Black Solid Slim Fit Formal Shirt"),
    ],
)
def test_title_vs_type_contradiction_is_quarantined(ptype: str, title: str) -> None:
    res = derive_category(ptype, title)
    assert res.quarantine_reason != ""
    assert "title head" in res.quarantine_reason


def test_accessory_in_garment_set_title_is_not_a_contradiction() -> None:
    assert derive_category("kurta", "Kurta Pant and Dupatta Set").quarantine_reason == ""


def test_head_noun_is_last_word_before_preposition() -> None:
    assert title_head("Maroon Kundan Choker for Lehenga") == ("necklace", "jewellery")
    assert title_head("Kurta with Jhumka") == ("kurta", "garment")
    assert title_head("Lehenga Box") == ("box", "non_fashion")
    assert title_head("Tie Dye Kurta") == ("kurta", "garment")  # "tie" before "dye" is not a tie
    assert title_head("Paperbag Waist Trousers") == ("trousers", "garment")  # not a bag
    assert title_head("") is None


def test_classify_type_generic_types_carry_no_information() -> None:
    assert classify_type("Fashion") is None
    assert classify_type("All Products") is None
    assert classify_type(None) is None
    assert classify_type("Necklace Sets") == ("necklace", "jewellery")


def test_is_garment_type_name() -> None:
    for name in ("kurta", "lehenga", "dress", "jeans", "indowestern", "sherwani"):
        assert is_garment_type_name(name), name
    for name in ("earrings", "necklace", "bag", "dupatta", "footwear", "jhumka", "bangles", None):
        assert not is_garment_type_name(name), name
    assert not is_garment_type_name("not-a-type")  # unknown is never shielded


def test_apply_hygiene_adds_columns_and_splits_rows() -> None:
    df = pd.DataFrame(
        {
            "article_id": ["1", "2", "3", "4"],
            "store": ["a", "a", "b", "b"],
            "prod_name": ["Silk Kurta", "Brocade Lehenga Box", "Gold Bangles Lehenga", "Studs"],
            "product_type_name": ["kurta", "lehenga", "Bangles", "Earrings"],
            "price_inr": [1.0, 2.0, 3.0, 4.0],
        }
    )
    kept, report = apply_hygiene(df)
    assert kept["article_id"].tolist() == ["1", "4"]
    assert kept["category"].tolist() == ["garment", "jewellery"]
    assert report["dropped"]["article_id"].tolist() == ["2"]
    assert report["quarantined"]["article_id"].tolist() == ["3"]
    assert "title head" in report["quarantined"]["reason"].iloc[0]
    # Input is not mutated.
    assert "category" not in df.columns


def test_apply_hygiene_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "article_id": [str(i) for i in range(6)],
            "store": ["s"] * 6,
            "prod_name": ["A Kurta", "B Necklace", "C Box", "D Juttis", "E Dupatta", "F Hamper"],
            "product_type_name": ["kurta", "necklace", "Fashion", "footwear", "dupatta", "Fashion"],
        }
    )
    a, ra = apply_hygiene(df)
    b, rb = apply_hygiene(df)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ra["dropped"], rb["dropped"])
