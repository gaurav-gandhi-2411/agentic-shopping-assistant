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


# ---------------------------------------------------------------------------
# fabric_material — fabric bolts must NOT classify as a wearable garment type
# ---------------------------------------------------------------------------

FABRIC_MATERIAL_CASES: list[tuple[str, str | None]] = [
    # (prod_name, brand)  — all must yield garment_type="fabric_material", category="raw_material"
    ("Unstitched Dress Material Floral Print Cotton Blend", None),
    ("Blue Printed Blouse Piece Silk", None),
    ("Cotton Unstitched Salwar Suit Material", None),
    ("Dress Material Set Green Geometric", None),
]


@pytest.mark.parametrize("prod_name,brand", FABRIC_MATERIAL_CASES)
def test_fabric_material_cases(prod_name: str, brand: str | None) -> None:
    """Fabric bolts and unstitched materials must classify as fabric_material, not dress/salwar."""
    result = normalize_garment_type(prod_name, brand=brand)
    assert result.garment_type == "fabric_material", (
        f"'{prod_name}': expected garment_type='fabric_material', got {result.garment_type!r}"
    )
    assert result.category == "raw_material", (
        f"'{prod_name}': expected category='raw_material', got {result.category!r}"
    )
    assert result.type_confidence == "high"


# ---------------------------------------------------------------------------
# Phase A (2026-07-06) — finished sarees sold "with blouse piece" must NOT be
# swallowed by the fabric_material compound match. See src/catalogue/cleaning.py
# for the equivalent build-time reclassification rule (same "saree + blouse
# piece" combined signal).
# ---------------------------------------------------------------------------

FINISHED_SAREE_CASES: list[str] = [
    "Peach Printed Georgette Saree With Unstitched Blouse Piece",
    "Meena Bazaar Turquoise Blue Woven Design Silk Blend Saree with Blouse Piece",
    "Sangria Blue Striped Saree & Embellished Blouse Piece",
]


@pytest.mark.parametrize("prod_name", FINISHED_SAREE_CASES)
def test_finished_saree_with_blouse_piece_classifies_as_saree(prod_name: str) -> None:
    """A saree bundled with a (possibly unstitched) blouse piece is a finished garment."""
    result = normalize_garment_type(prod_name)
    assert result.garment_type == "saree", (
        f"'{prod_name}': expected garment_type='saree', got {result.garment_type!r}"
    )
    assert result.category == "apparel"


def test_saree_brand_prefix_stays_fabric_material() -> None:
    """'Saree Mall' is a brand name — the product itself is unstitched dress material."""
    result = normalize_garment_type("Saree mall Black Unstitched Dress Material")
    assert result.garment_type == "fabric_material"
    assert result.category == "raw_material"


# ---------------------------------------------------------------------------
# Brand false-positive regression — DressBerry/20Dresses must not mis-classify
# ---------------------------------------------------------------------------

BRAND_FP_CASES: list[tuple[str, str | None, str, str]] = [
    # (prod_name, brand, expected_gt, expected_conf)
    # Brand stripped → real "Dress" in title wins correctly
    ("DressBerry Women Blue Shift Dress", "DressBerry", "dress", "high"),
    # No brand arg → rightmost-noun rule: "shorts" at far right wins over "dressBerry" prefix
    # (\bdress\b does NOT match inside "DressBerry" — no word boundary after 'dress' before 'B')
    ("DressBerry Women Casual Shorts", None, "shorts", "high"),
    # No brand arg → "skirt" (rightmost) wins over "Dresses" embedded in brand token
    ("20Dresses Women Floral Midi Skirt", None, "skirt", "high"),
    # Brand stripped → "pants" is rightmost/head noun → trousers (palazzo is the style modifier)
    ("20Dresses Women Palazzo Pants", "20Dresses", "trousers", "high"),
]


@pytest.mark.parametrize("prod_name,brand,expected_gt,expected_conf", BRAND_FP_CASES)
def test_brand_false_positive_cases(
    prod_name: str,
    brand: str | None,
    expected_gt: str,
    expected_conf: str,
) -> None:
    """Brand names containing 'dress'/'dresses' must not pollute garment classification."""
    result = normalize_garment_type(prod_name, brand=brand)
    assert result.garment_type == expected_gt, (
        f"'{prod_name}' (brand={brand!r}): "
        f"expected {expected_gt!r}, got {result.garment_type!r}"
    )
    assert result.type_confidence == expected_conf


# ---------------------------------------------------------------------------
# BUG 4 — "kurta and pyjama"/"kurta pajama" SET titles were misclassified as
# nightwear ("pyjama"/"pajama" is rightmost and wins the position scan over
# "kurta"). Confirmed live: 40 catalogue rows like "Men Kurta and Pyjama Set
# Dupion Silk" were tagged product_type_name="nightwear". Mirrored in
# src/agents/intent_parser.py's _COMPOUND_TERMS (query-parse-time fix).
# ---------------------------------------------------------------------------

KURTA_PAJAMA_CASES: list[str] = [
    "Men Kurta and Pyjama Set Jacquard",
    "Men Kurta and Pyjama Set Dupion Silk",
    "Men Kurta and Pyjama Set Pure Cotton",
    "Men Kurta Pyjama Set Cotton Blend",
    "Kurta Pajama Set",
]


@pytest.mark.parametrize("prod_name", KURTA_PAJAMA_CASES)
def test_kurta_pajama_set_resolves_to_kurta_not_nightwear(prod_name: str) -> None:
    """Ethnic kurta-pajama sets must classify as 'kurta', not 'nightwear'."""
    result = normalize_garment_type(prod_name)
    assert result.garment_type == "kurta", (
        f"'{prod_name}': expected garment_type='kurta', got {result.garment_type!r}"
    )
    assert result.type_confidence == "high"


def test_bare_pyjama_set_still_classifies_as_nightwear() -> None:
    """Without 'kurta', a bare pyjama-only set must stay nightwear — the fix
    must not blanket-map all pyjama mentions to kurta."""
    result = normalize_garment_type("Men Pyjama Set Cotton")
    assert result.garment_type == "nightwear"


# ---------------------------------------------------------------------------
# Men's-ethnic-depth wave (2026-07-19) — sherwani/bandhgala/jodhpuri-suit/
# indo-western were entirely absent from _GARMENT_RULES prior to this wave,
# so real titles from the new inventory (rathore, bhasinbrothers, mohanlalsons,
# vastramay, kisah) resolved to garment_type=None, category="unknown".
# ---------------------------------------------------------------------------

WEDDING_ETHNIC_CASES: list[tuple[str, str, str]] = [
    # (prod_name, expected_garment_type, expected_category)
    ("Men Wedding Sherwani Set", "sherwani", "apparel"),
    ("MLS Embroidered Sherwani", "sherwani", "apparel"),
    ("Indowestern Sherwani Achkan", "sherwani", "apparel"),  # rightmost noun wins
    ("Black Bandhgala", "bandhgala", "apparel"),
    ("Grey Bandgala Suit", "bandhgala", "apparel"),  # alternate spelling
    ("Jodhpuri Suit", "jodhpuri_suit", "apparel"),
    ("Men Jodhpuri Suits", "jodhpuri_suit", "apparel"),  # plural
    ("Indo Western", "indowestern", "apparel"),
    ("Wedding Indo Western Set", "indowestern", "apparel"),  # rightmost noun wins
    ("Indowestern", "indowestern", "apparel"),  # no separator, no space
]


@pytest.mark.parametrize("prod_name,expected_gt,expected_cat", WEDDING_ETHNIC_CASES)
def test_wedding_ethnic_garment_types(prod_name: str, expected_gt: str, expected_cat: str) -> None:
    """Sherwani/bandhgala/jodhpuri-suit/indo-western must resolve, not fall to unknown."""
    result = normalize_garment_type(prod_name)
    assert result.garment_type == expected_gt, (
        f"'{prod_name}': expected garment_type={expected_gt!r}, got {result.garment_type!r}"
    )
    assert result.category == expected_cat, (
        f"'{prod_name}': expected category={expected_cat!r}, got {result.category!r}"
    )
    assert result.type_confidence == "high"


# ---------------------------------------------------------------------------
# Ethnic-footwear brands wave (2026-07-19) — jutti/mojari/kolhapuri/chappal were
# entirely absent from the footwear rule, so real titles from kraftojodhpur,
# houseofvian, 5-elements, taurjuttis, and fizzygoblet mis-resolved: "sherwani"
# outranked the unrecognized "jutti" as the only known noun (39/96 kraftojodhpur
# rows), a trailing "clutch"/"handbag" noun in combo listings won the rightmost
# scan over the leading jutti/kolhapuri noun (7 houseofvian + 11 5-elements
# rows), and the fabric descriptor "denim" outranked the unrecognized "jutti"
# (1 taurjuttis + 1 fizzygoblet row, live titles "Denim Leaves Jutti" and
# "Denim Darling : Juttis").
# ---------------------------------------------------------------------------

FOOTWEAR_ETHNIC_CASES: list[tuple[str, str, str]] = [
    # (prod_name, expected_garment_type, expected_category)
    ("Amber Jute Men's Sherwani Jutti", "footwear", "footwear"),  # rightmost noun wins
    ("Firdaus Juttis & Clutch Combo", "footwear", "footwear"),  # footwear-led combo override
    ("Denim Casual Jutti", "footwear", "footwear"),  # rightmost noun wins over fabric descriptor
    ("Denim Leaves Jutti", "footwear", "footwear"),  # live taurjuttis title
    ("Denim Darling : Juttis", "footwear", "footwear"),  # live fizzygoblet title
    ("Kolhapuri Chappal", "footwear", "footwear"),
    ("Mojari", "footwear", "footwear"),
    ("Rangeela Kolhapuris & Handbag Combo", "footwear", "footwear"),  # live 5-elements title
]


@pytest.mark.parametrize("prod_name,expected_gt,expected_cat", FOOTWEAR_ETHNIC_CASES)
def test_footwear_ethnic_garment_types(prod_name: str, expected_gt: str, expected_cat: str) -> None:
    """Jutti/mojari/kolhapuri/chappal titles must resolve to footwear, not apparel/bag/jeans."""
    result = normalize_garment_type(prod_name)
    assert result.garment_type == expected_gt, (
        f"'{prod_name}': expected garment_type={expected_gt!r}, got {result.garment_type!r}"
    )
    assert result.category == expected_cat, (
        f"'{prod_name}': expected category={expected_cat!r}, got {result.category!r}"
    )
    assert result.type_confidence == "high"


# ---------------------------------------------------------------------------
# Jewellery-inventory-gap wave (2026-07-19) — BUG 1: "short"/"shirt"/"saree"
# apparel-fragment keywords collided with dedicated-jewellery titles where the
# real garment noun (necklace/jhumka/pin/clip/etc.) never existed in the rule
# set. Confirmed live: theamethyststore (1,488 "Short Necklace Set" rows ->
# "shorts", 60 "Shirt Button Clip" rows -> "shirt", 9 "Saree Pin" rows ->
# "saree"), southtemplejewellery (57 "Short Necklace Set" variant rows ->
# "shorts", including via the store's own "Short Necklaces" product_type_name
# label being re-scanned by the fallback path), daivik (56 "... Short ...
# Necklace/Haram ..." rows -> "shorts", 11 "Saree Pin" rows -> "saree").
# ---------------------------------------------------------------------------

JEWELLERY_COLLISION_CASES: list[tuple[str, str, str]] = [
    # (prod_name, expected_garment_type, expected_category)
    ("Simrath Short Necklace Set", "necklace", "accessories"),
    ("Sherinka Nagas Short Necklace", "necklace", "accessories"),
    ("Shimmering S Letter Shirt Button Clip", "jewellery", "accessories"),
    ("James Shirt Button Clip", "jewellery", "accessories"),
    ("Laksmi Lotus Saree Pin", "jewellery", "accessories"),
    ("Unique Kundan Saree Pin", "jewellery", "accessories"),
    ("Short Necklace Set", "necklace", "accessories"),
    ("Antique Necklace Set V-751", "necklace", "accessories"),
    # Jewellery noun BEFORE the fragment word — the rightmost-noun scan alone
    # would pick "short"/"top" here without the Step 3.5 precedence override.
    ("Antique Gold-Plated Temple Necklace Set - Bridal Short Design K-1835", "necklace", "accessories"),
    ("South Indian Laxmi Jhumkas - Gold-Plated Ruby Floral Top R-2733", "jhumka", "accessories"),
    # "with"-barrier real-world daivik title: "necklace" sits before "with", the
    # trailing "earrings" is barred, and "short" must still lose to "necklace".
    ("Victorian Bridal Short and Long Combo Necklace with Earrings", "necklace", "accessories"),
    ("Antique Lakshmi Coin Short and Long Necklace with Earrings", "necklace", "accessories"),
    ("Lakshmi Short JadaBillai with Green beads Mattal", "jewellery", "accessories"),
]


@pytest.mark.parametrize("prod_name,expected_gt,expected_cat", JEWELLERY_COLLISION_CASES)
def test_jewellery_apparel_fragment_collision(
    prod_name: str, expected_gt: str, expected_cat: str
) -> None:
    """Jewellery titles containing 'short'/'shirt'/'saree'/'top' must resolve to the
    real jewellery noun, not the coincidental apparel-fragment keyword."""
    result = normalize_garment_type(prod_name)
    assert result.garment_type == expected_gt, (
        f"'{prod_name}': expected garment_type={expected_gt!r}, got {result.garment_type!r}"
    )
    assert result.category == expected_cat, (
        f"'{prod_name}': expected category={expected_cat!r}, got {result.category!r}"
    )
    assert result.type_confidence == "high"


def test_short_necklace_via_store_label_fallback() -> None:
    """A store's own product_type_name label ('Short Necklaces') must not be
    re-scanned into 'shorts' by the fallback path — southtemplejewellery live
    pattern where prod_name has no garment noun at all and only the label does."""
    result = normalize_garment_type("Antique Necklace Set N-861N", product_type_name="Short Necklaces")
    assert result.garment_type == "necklace"
    assert result.category == "accessories"


def test_genuine_shorts_unaffected_by_jewellery_precedence() -> None:
    """A real shorts listing with no jewellery noun anywhere must still resolve
    to 'shorts' — the Step 3.5 override only fires when a jewellery noun is
    also present."""
    result = normalize_garment_type("DressBerry Women Black Shorts", brand="DressBerry")
    assert result.garment_type == "shorts"
    assert result.category == "apparel"


def test_genuine_shirt_unaffected_by_jewellery_precedence() -> None:
    """A real shirt listing with no jewellery noun anywhere must still resolve
    to 'shirt'."""
    result = normalize_garment_type("Formal Shirt")
    assert result.garment_type == "shirt"
    assert result.category == "apparel"


# ---------------------------------------------------------------------------
# Activewear wave (2026-07-23) — blissclub.com/silvertraq.com onboarding to
# close the gym-query catalogue gap (women's sports bras, leggings, joggers).
# "sports bra"/"track pants"/"cargo pants" needed compound-table entries
# because the bare noun they end in ("bra"/"pants") is already claimed by an
# existing generic rule (innerwear's "\bbra\b", trousers' "\bpants\b") that
# would otherwise win the rightmost-noun scan. Live titles below are drawn
# directly from the two stores' /products.json downloads.
# ---------------------------------------------------------------------------

ACTIVEWEAR_CASES: list[tuple[str, str, str]] = [
    # (prod_name, expected_garment_type, expected_category)
    ("Ultimate Printed Leggings", "leggings", "apparel"),  # live blissclub title
    ("Contour Shaper Leggings Black", "leggings", "apparel"),  # live silvertraq title
    ("Strappy Racerback Sports Bra", "sports_bra", "apparel"),  # live blissclub title
    ("High Impact Action Sports Bra Lilac", "sports_bra", "apparel"),  # live silvertraq title
    ("Keyhole Back Sports bra with Clasp Berry Kiss", "sports_bra", "apparel"),  # lowercase "bra"
    ("Ultimate Cuffed Joggers", "joggers", "apparel"),  # live blissclub title
    ("TraqEase Sweatpants Black", "joggers", "apparel"),  # synonym merge, live silvertraq title
    ("TraqLite Track Pants Black", "track_pants", "apparel"),  # live silvertraq title
    ("TraqPace Cargo Pants Lilac", "cargo_pants", "apparel"),  # live silvertraq title
    ("The Do-It All Skorts", "skort", "apparel"),  # live blissclub title
    ("TraqFlex Skort White", "skort", "apparel"),  # live silvertraq title
]


@pytest.mark.parametrize("prod_name,expected_gt,expected_cat", ACTIVEWEAR_CASES)
def test_activewear_garment_types(prod_name: str, expected_gt: str, expected_cat: str) -> None:
    """Leggings/sports_bra/joggers/track_pants/cargo_pants/skort must resolve,
    not fall through to a generic bottoms/innerwear type or unknown."""
    result = normalize_garment_type(prod_name)
    assert result.garment_type == expected_gt, (
        f"'{prod_name}': expected garment_type={expected_gt!r}, got {result.garment_type!r}"
    )
    assert result.category == expected_cat, (
        f"'{prod_name}': expected category={expected_cat!r}, got {result.category!r}"
    )
    assert result.type_confidence == "high"


def test_generic_pants_unaffected_by_activewear_rules() -> None:
    """A plain trousers listing with no activewear noun must still resolve to
    'trousers' — the new compound entries only fire on their exact phrases."""
    result = normalize_garment_type("Linen Tapered Pants Beige")
    assert result.garment_type == "trousers"
    assert result.category == "apparel"


# ---------------------------------------------------------------------------
# Activewear-footwear wave (2026-07-24) — campusshoes.com onboarding to close
# the women's genuine athletic-footwear gap (running/sneakers/walking/sports
# shoes). No rule-table change was needed: "shoe"/"shoes"/"sneaker" already
# cover every athletic-footwear title/product_type in this store's real feed.
# Live titles/types below are drawn directly from the store's /products.json
# download (data/raw/shopify/campusshoes/products.csv).
#
# Known gap (flagged, not fixed — out of scope for this ingestion): 61 rows
# whose only footwear-shaped noun is "Slides"/"Flip Flops"/"Clogs" (types
# "Men/Women Flip Flops & Slides", "Women's Clogs") fall through to
# category="unknown" because none of those three words are in the footwear
# regex. This does not affect athletic footwear (running/training/sneakers/
# walking/sports shoes all classify correctly) — see CAMPUSSHOES_CASES below.
# ---------------------------------------------------------------------------

CAMPUSSHOES_CASES: list[tuple[str, str, str, str]] = [
    # (prod_name, product_type_name, expected_garment_type, expected_category)
    ("BELL Green Women's Sneakers", "Women Sneakers", "footwear", "footwear"),
    ("RAYE Black Women's Running Shoes", "Women Running Shoes", "footwear", "footwear"),
    ("MALONE Purple Women's Walking Shoes", "Women Walking Shoes", "footwear", "footwear"),
    ("ALLEN Green Men's Walking Shoes", "Men Sports Shoes", "footwear", "footwear"),
    ("CANVA White Men's Running shoes", "Men Running Shoes", "footwear", "footwear"),
    ("NEBULA Navy Men's Running Shoes", "Men Running Shoes", "footwear", "footwear"),
]


@pytest.mark.parametrize("prod_name,product_type_name,expected_gt,expected_cat", CAMPUSSHOES_CASES)
def test_campusshoes_athletic_footwear_garment_types(
    prod_name: str, product_type_name: str, expected_gt: str, expected_cat: str
) -> None:
    """Real campusshoes.com women's/men's athletic-footwear titles must resolve
    to garment_type/category='footwear', not fall through to unknown."""
    result = normalize_garment_type(prod_name, product_type_name)
    assert result.garment_type == expected_gt, (
        f"'{prod_name}': expected garment_type={expected_gt!r}, got {result.garment_type!r}"
    )
    assert result.category == expected_cat, (
        f"'{prod_name}': expected category={expected_cat!r}, got {result.category!r}"
    )


def test_generic_bra_unaffected_by_sports_bra_rule() -> None:
    """A plain bra listing with no 'sports' modifier must still resolve to
    'innerwear' — the sports_bra compound entry only fires on 'sports bra'."""
    result = normalize_garment_type("Absolute Invisible Bra")
    assert result.garment_type == "innerwear"
    assert result.category == "apparel"
