"""Unit tests for src.catalogue.normalizer.GarmentNormalizer."""

from __future__ import annotations

import pytest

from src.catalogue.normalizer import NormalizationResult, normalize_garment_type

# ---------------------------------------------------------------------------
# Mandatory trap cases (10 cases specified in task)
# ---------------------------------------------------------------------------

MANDATORY_CASES: list[tuple[str, str | None, str, str]] = [
    # (prod_name, brand, expected_garment_type, expected_confidence)
    ("Shorts For Under Dresses", None, "shorts", "high"),
    ("Dress Shirt", None, "shirt", "high"),
    ("Jacket Dress", None, "dress", "high"),
    ("Mini Skirt", None, "skirt", "high"),
    ("Co-Ord Set", None, "coord", "high"),
    ("DressBerry Women Black Shorts", "DressBerry", "shorts", "high"),
    ("20Dresses Crop Jacket", "20Dresses", "outerwear", "high"),
    ("DressBerry Sweater", "DressBerry", "knitwear", "high"),
    ("Black Floral Maxi Dress", None, "dress", "high"),
    ("Kurti For Women", None, "kurti", "high"),
]


@pytest.mark.parametrize("prod_name,brand,expected_gt,expected_conf", MANDATORY_CASES)
def test_mandatory_trap_cases(
    prod_name: str,
    brand: str | None,
    expected_gt: str,
    expected_conf: str,
) -> None:
    """All 10 mandatory trap cases must classify with the correct type and confidence."""
    result = normalize_garment_type(prod_name, brand=brand)
    assert result.garment_type == expected_gt, (
        f"'{prod_name}' (brand={brand!r}): "
        f"expected garment_type={expected_gt!r}, got {result.garment_type!r}"
    )
    assert result.type_confidence == expected_conf, (
        f"'{prod_name}': expected confidence={expected_conf!r}, got {result.type_confidence!r}"
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_normalization_result() -> None:
    """normalize_garment_type always returns a NormalizationResult instance."""
    result = normalize_garment_type("Plain T-Shirt")
    assert isinstance(result, NormalizationResult)


# ---------------------------------------------------------------------------
# Unknown / empty input
# ---------------------------------------------------------------------------


def test_empty_name_returns_unknown() -> None:
    """An empty product name with no store label should return unknown."""
    result = normalize_garment_type("", product_type_name=None, brand=None)
    assert result.garment_type is None
    assert result.type_confidence == "unknown"
    assert result.category == "unknown"


def test_no_garment_noun_returns_unknown() -> None:
    """A product name with no recognizable garment noun should return unknown."""
    result = normalize_garment_type("Summer Collection 2024")
    assert result.garment_type is None
    assert result.type_confidence == "unknown"


# ---------------------------------------------------------------------------
# Fallback to product_type_name
# ---------------------------------------------------------------------------


def test_fallback_to_product_type_name() -> None:
    """When prod_name has no garment noun, product_type_name is used with medium confidence."""
    result = normalize_garment_type(
        "Summer Collection Item", product_type_name="Dress", brand=None
    )
    assert result.garment_type == "dress"
    assert result.type_confidence == "medium"


def test_product_type_name_none_and_no_name_match() -> None:
    """Both prod_name and product_type_name missing should return unknown."""
    result = normalize_garment_type("Some Product 123", product_type_name=None)
    assert result.garment_type is None
    assert result.type_confidence == "unknown"


# ---------------------------------------------------------------------------
# Preposition barrier
# ---------------------------------------------------------------------------


def test_saree_with_blouse_barrier() -> None:
    """'with' barrier should prevent blouse from winning over saree."""
    result = normalize_garment_type("Saree With Blouse")
    assert result.garment_type == "saree"
    assert result.type_confidence == "high"


def test_kurta_for_men() -> None:
    """'for' does not discard kurta since kurta appears before the barrier."""
    result = normalize_garment_type("Kurta For Men")
    assert result.garment_type == "kurta"
    assert result.type_confidence == "high"


# ---------------------------------------------------------------------------
# Brand-prefix strip edge cases
# ---------------------------------------------------------------------------


def test_brand_strip_only_when_not_skip_value() -> None:
    """Brands listed as 'unknown'/'mixed' should not be stripped."""
    # "mixed Tops" — brand="mixed" is in skip list, so "mixed tops" should match "top"
    result = normalize_garment_type("Mixed Fabric Top", brand="mixed")
    assert result.garment_type == "top"


def test_brand_strip_with_comma_separator() -> None:
    """Brand tokens separated by comma should be stripped."""
    result = normalize_garment_type("TestBrand, Black Dress", brand="TestBrand")
    assert result.garment_type == "dress"


# ---------------------------------------------------------------------------
# Compound term table
# ---------------------------------------------------------------------------


def test_coord_set_variants() -> None:
    """All coord-set compound variants should normalize to 'coord'."""
    for name in ("Co-Ord Set", "Co Ord Set", "Coord Set", "Co-Ord"):
        result = normalize_garment_type(name)
        assert result.garment_type == "coord", f"Failed for: {name!r}"
        assert result.type_confidence == "high"


def test_dungaree_dress_compound() -> None:
    """'Dungaree Dress' hits the compound table -> 'dress'."""
    result = normalize_garment_type("Dungaree Dress")
    assert result.garment_type == "dress"
    assert result.type_confidence == "high"


def test_skirt_suit_compound() -> None:
    """'Skirt Suit' hits the compound table -> 'coord'."""
    result = normalize_garment_type("Skirt Suit")
    assert result.garment_type == "coord"
    assert result.type_confidence == "high"


# ---------------------------------------------------------------------------
# Category field
# ---------------------------------------------------------------------------


def test_footwear_category() -> None:
    """Footwear items should have category='footwear'."""
    result = normalize_garment_type("Leather Sandals")
    assert result.garment_type == "footwear"
    assert result.category == "footwear"


def test_bag_category() -> None:
    """Bag items should have category='accessories'."""
    result = normalize_garment_type("Tote Bag")
    assert result.garment_type == "bag"
    assert result.category == "accessories"


def test_outerwear_category() -> None:
    """Blazer should have category='outerwear'."""
    result = normalize_garment_type("Formal Blazer")
    assert result.garment_type == "blazer"
    assert result.category == "outerwear"


def test_apparel_category() -> None:
    """Standard apparel should have category='apparel'."""
    result = normalize_garment_type("Floral Dress")
    assert result.garment_type == "dress"
    assert result.category == "apparel"


# ---------------------------------------------------------------------------
# T-shirt specific (regression for lookbehind fix)
# ---------------------------------------------------------------------------


def test_tshirt_maps_to_top() -> None:
    """'T-Shirt' must map to 'top', not 'shirt'."""
    result = normalize_garment_type("Cotton T-Shirt")
    assert result.garment_type == "top"
    assert result.type_confidence == "high"


def test_standalone_shirt_still_works() -> None:
    """'Shirt' without 't-' prefix must still map to 'shirt'."""
    result = normalize_garment_type("Formal Shirt")
    assert result.garment_type == "shirt"
    assert result.type_confidence == "high"


# ---------------------------------------------------------------------------
# Kaftan dress — documented ambiguity
# ---------------------------------------------------------------------------


def test_kaftan_dress_rightmost_wins() -> None:
    """'Kaftan Dress' -> dress (rightmost noun per the algorithm)."""
    result = normalize_garment_type("Kaftan Dress")
    assert result.garment_type == "dress"
    assert result.type_confidence == "high"


def test_standalone_kaftan_is_kaftan() -> None:
    """'Kaftan' with no 'dress' modifier -> kaftan."""
    result = normalize_garment_type("Beautiful Kaftan")
    assert result.garment_type == "kaftan"
    assert result.type_confidence == "high"
