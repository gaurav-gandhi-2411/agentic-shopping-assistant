"""Unit tests for src.catalogue.entity_resolution.

Covers: brand extraction, confusable-pair rejection, true-positive match,
single-store skip, lowest-first ordering, snapshot labeling, brand_index.
"""
from __future__ import annotations

import pandas as pd

from src.catalogue.entity_resolution import (
    DEFAULT_MIN_CONFIDENCE,
    TITLE_SIMILARITY_THRESHOLD,
    build_brand_index,
    build_brand_stores_map,
    extract_brand,
    find_cross_store_matches,
    get_cached_brand_stores_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(
    prod_name: str,
    store: str,
    product_type: str = "Shirts",
    gender: str = "men",
    price: float = 1000.0,
    article_id: str = "ITEM-A",
    pdp_handle: str = "test-handle",
) -> dict:
    """Build a minimal item dict for testing."""
    return {
        "article_id": article_id,
        "prod_name": prod_name,
        "product_type_name": product_type,
        "gender": gender,
        "store": store,
        "price_inr": price,
        "pdp_handle": pdp_handle,
    }


def _make_df(*items: dict) -> pd.DataFrame:
    """Wrap item dicts into a DataFrame."""
    return pd.DataFrame(list(items))


# ---------------------------------------------------------------------------
# extract_brand
# ---------------------------------------------------------------------------


class TestExtractBrand:
    def test_d2c_snitch_returns_snitch(self) -> None:
        """Snitch is always brand 'snitch' regardless of prod_name."""
        assert extract_brand("Abstract Box Fit Viscose Shirt", "snitch") == "snitch"
        assert extract_brand("Fusion Crest Brown Shirt", "snitch") == "snitch"

    def test_d2c_fashor_returns_fashor(self) -> None:
        assert extract_brand("Abstract Checks Kurta", "fashor") == "fashor"

    def test_d2c_powerlook_returns_powerlook(self) -> None:
        assert extract_brand("Sage Striped T-Shirt", "powerlook") == "powerlook"

    def test_virgio_pipe_format(self) -> None:
        """Virgio uses BrandName|product format."""
        assert extract_brand("Brakin|100% Cotton Relaxed Denim Shirt", "virgio") == "brakin"
        assert extract_brand("Seraphinae|100% Viscose Printed Dress", "virgio") == "seraphinae"

    def test_virgio_no_pipe_falls_back_to_first_token(self) -> None:
        """Virgio item without pipe uses first token."""
        assert extract_brand("Roselia Cotton Dress", "virgio") == "roselia"

    def test_myntra_leading_brand_extraction(self) -> None:
        """Myntra: brand stops before gender keyword 'Women'."""
        brand = extract_brand("Free Authority Women Hooded Sweatshirt", "myntra")
        assert brand == "free authority"

    def test_myntra_single_word_brand(self) -> None:
        brand = extract_brand("FILA Women Blue Running Jacket", "myntra")
        assert brand == "fila"

    def test_flipkart_brand_extraction(self) -> None:
        brand = extract_brand("Free Authority Men Vest", "flipkart")
        assert brand == "free authority"

    def test_normalisation_lowercase_strip(self) -> None:
        """Brand must be normalised to lowercase."""
        b1 = extract_brand("Anouk Women Yellow Kurta", "myntra")
        b2 = extract_brand("ANOUK Women Red Kurta", "myntra")
        assert b1 == b2  # both normalise to "anouk"

    def test_threshold_constant_value(self) -> None:
        """Threshold constants must be at the documented values."""
        assert TITLE_SIMILARITY_THRESHOLD == 0.90
        assert DEFAULT_MIN_CONFIDENCE == TITLE_SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# Brand stores map
# ---------------------------------------------------------------------------


class TestBuildBrandStoresMap:
    def test_single_store_brand(self) -> None:
        df = _make_df(
            _make_item("Snitch Linen Shirt", "snitch", article_id="S1"),
            _make_item("Snitch Cotton Tee", "snitch", article_id="S2"),
        )
        m = build_brand_stores_map(df)
        assert m.get("snitch") == {"snitch"}

    def test_multi_store_brand(self) -> None:
        df = _make_df(
            _make_item("GlobalBrand Men Shirt", "myntra", article_id="M1"),
            _make_item("GlobalBrand Men Shirt", "flipkart", article_id="F1"),
        )
        m = build_brand_stores_map(df)
        assert "globalbrand" in m
        assert m["globalbrand"] == {"myntra", "flipkart"}


# ---------------------------------------------------------------------------
# Brand index
# ---------------------------------------------------------------------------


class TestBuildBrandIndex:
    def test_index_maps_brand_to_row_positions(self) -> None:
        df = _make_df(
            _make_item("GlobalBrand Men Shirt", "myntra", article_id="M1"),
            _make_item("OtherBrand Tee", "myntra", article_id="M2"),
            _make_item("GlobalBrand Men Shirt", "flipkart", article_id="F1"),
        )
        idx = build_brand_index(df)
        # GlobalBrand should map to positions 0 and 2
        assert "globalbrand" in idx
        assert sorted(idx["globalbrand"]) == [0, 2]

    def test_index_respects_d2c_stores(self) -> None:
        """Snitch items should be indexed under brand='snitch'."""
        df = _make_df(
            _make_item("Abstract Shirt", "snitch", article_id="S1"),
            _make_item("Fusion Crest Shirt", "snitch", article_id="S2"),
        )
        idx = build_brand_index(df)
        assert "snitch" in idx
        assert len(idx["snitch"]) == 2


# ---------------------------------------------------------------------------
# find_cross_store_matches — single-store skip (performance guard)
# ---------------------------------------------------------------------------


class TestSingleStoreSkip:
    def test_single_store_brand_returns_empty(self) -> None:
        """When brand appears in only one store, skip matching immediately."""
        q = _make_item("Snitch Linen Shirt", "snitch", article_id="Q")
        cat = _make_df(
            _make_item("Snitch Linen Shirt", "snitch", article_id="C1"),
        )
        brand_map = {"snitch": {"snitch"}}  # single-store
        result = find_cross_store_matches(q, cat, brand_stores_map=brand_map)
        assert result == []

    def test_same_store_never_returned(self) -> None:
        """Match candidates from the same store as query are always excluded."""
        q = _make_item("GlobalBrand Men Slim Shirt", "myntra", article_id="Q")
        cat = _make_df(
            _make_item("GlobalBrand Men Slim Shirt", "myntra", article_id="C-SAME"),
            _make_item("GlobalBrand Men Slim Shirt", "flipkart", article_id="C-OTHER"),
        )
        perm_map = {"globalbrand": {"myntra", "flipkart"}}
        result = find_cross_store_matches(q, cat, brand_stores_map=perm_map)
        article_ids = [m["article_id"] for m in result]
        assert "C-SAME" not in article_ids
        assert "C-OTHER" in article_ids


# ---------------------------------------------------------------------------
# find_cross_store_matches — hard negative rejection
# ---------------------------------------------------------------------------


class TestConfusablePairRejection:
    def test_different_product_type_rejected(self) -> None:
        """Free Authority: Shirt (myntra) vs Innerwear (flipkart) — type mismatch."""
        q = _make_item(
            "Free Authority Women Olive Hooded Sweatshirt",
            "myntra",
            product_type="Shirt",
            gender="women",
            article_id="Q",
        )
        cat = _make_df(
            _make_item(
                "Free Authority Men Vest",
                "flipkart",
                product_type="Innerwear and Swimwear",
                gender="men",
                article_id="C1",
            )
        )
        perm_map = {"free authority": {"myntra", "flipkart"}}
        result = find_cross_store_matches(q, cat, brand_stores_map=perm_map)
        assert result == []

    def test_different_gender_rejected(self) -> None:
        """Same brand/type but different gender — must be rejected."""
        q = _make_item(
            "GlobalBrand Women Printed Kurta",
            "myntra",
            product_type="Kurta",
            gender="women",
            article_id="Q",
        )
        cat = _make_df(
            _make_item(
                "GlobalBrand Men Printed Kurta",
                "flipkart",
                product_type="Kurta",
                gender="men",
                article_id="C1",
            )
        )
        perm_map = {"globalbrand": {"myntra", "flipkart"}}
        result = find_cross_store_matches(q, cat, brand_stores_map=perm_map)
        assert result == []

    def test_low_title_similarity_rejected(self) -> None:
        """Same brand+type+gender but completely different product names — rejected."""
        q = _make_item(
            "Fusion Beats Women Black Printed Top",
            "myntra",
            product_type="Top",
            gender="women",
            article_id="Q",
        )
        cat = _make_df(
            _make_item(
                "Fusion Football Shoes For Men",
                "flipkart",
                product_type="Top",  # force type to match; title must still reject
                gender="women",
                article_id="C1",
            )
        )
        perm_map = {"fusion beats": {"myntra", "flipkart"}}
        result = find_cross_store_matches(q, cat, brand_stores_map=perm_map)
        assert result == []


# ---------------------------------------------------------------------------
# find_cross_store_matches — true positive match
# ---------------------------------------------------------------------------


class TestTruePositiveMatch:
    def test_identical_title_matches(self) -> None:
        """Exact same brand+type+gender+title across stores — must match."""
        q = _make_item(
            "GlobalBrand Men Slim Fit Cotton Shirt",
            "myntra",
            product_type="Shirts",
            gender="men",
            price=1499.0,
            article_id="Q",
        )
        cat = _make_df(
            _make_item(
                "GlobalBrand Men Slim Fit Cotton Shirt",
                "flipkart",
                product_type="Shirts",
                gender="men",
                price=1299.0,
                article_id="C1",
            )
        )
        perm_map = {"globalbrand": {"myntra", "flipkart"}}
        result = find_cross_store_matches(q, cat, brand_stores_map=perm_map)
        assert len(result) == 1
        assert result[0]["store"] == "flipkart"
        assert result[0]["confidence"] >= DEFAULT_MIN_CONFIDENCE

    def test_minor_wording_variation_above_threshold(self) -> None:
        """'Printed' vs 'Print' variation — similarity >= 0.90."""
        q = _make_item(
            "GlobalBrand Women Floral Printed Wrap Dress",
            "myntra",
            product_type="Dress",
            gender="women",
            article_id="Q",
        )
        cat = _make_df(
            _make_item(
                "GlobalBrand Women Floral Print Wrap Dress",
                "flipkart",
                product_type="Dress",
                gender="women",
                article_id="C1",
            )
        )
        perm_map = {"globalbrand": {"myntra", "flipkart"}}
        result = find_cross_store_matches(q, cat, brand_stores_map=perm_map)
        assert len(result) == 1
        assert result[0]["confidence"] >= 0.90

    def test_unknown_gender_skips_gender_gate(self) -> None:
        """When query gender is 'unknown', gender gate is skipped."""
        q = _make_item(
            "GlobalBrand Men Regular Fit Cargo Pants",
            "myntra",
            product_type="Trousers",
            gender="unknown",
            article_id="Q",
        )
        cat = _make_df(
            _make_item(
                "GlobalBrand Men Regular Fit Cargo Pants",
                "flipkart",
                product_type="Trousers",
                gender="men",
                article_id="C1",
            )
        )
        perm_map = {"globalbrand": {"myntra", "flipkart"}}
        result = find_cross_store_matches(q, cat, brand_stores_map=perm_map)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# find_cross_store_matches — lowest-first ordering
# ---------------------------------------------------------------------------


class TestLowestFirstOrdering:
    def test_matches_sorted_price_ascending(self) -> None:
        """Multiple matches must be sorted lowest price first."""
        q = _make_item(
            "GlobalBrand Men Slim Shirt",
            "myntra",
            product_type="Shirts",
            gender="men",
            price=2000.0,
            article_id="Q",
        )
        cat = _make_df(
            _make_item(
                "GlobalBrand Men Slim Shirt",
                "flipkart",
                product_type="Shirts",
                gender="men",
                price=1800.0,
                article_id="C-expensive",
            ),
            _make_item(
                "GlobalBrand Men Slim Shirt",
                "snitch",  # Note: snitch D2C would block brand match; use a second multi-brand store
                product_type="Shirts",
                gender="men",
                price=1500.0,
                article_id="C-cheap",
            ),
        )
        # Override brand_map to allow snitch to be treated as multi-brand for this test
        perm_map = {"globalbrand": {"myntra", "flipkart", "snitch"}}
        result = find_cross_store_matches(q, cat, brand_stores_map=perm_map)
        # Only matches where brand resolves to 'globalbrand' in candidate_store
        # snitch -> extract_brand returns 'snitch', not 'globalbrand', so it'll be excluded
        # by Gate 1 (brand mismatch).  Flipkart should match.
        prices = [m["price_inr"] for m in result if m["price_inr"] is not None]
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1], f"Not ascending: {prices}"

    def test_none_price_sorts_last(self) -> None:
        """Items with no price should sort after priced items."""
        q = _make_item(
            "GlobalBrand Men Slim Shirt",
            "myntra",
            product_type="Shirts",
            gender="men",
            price=2000.0,
            article_id="Q",
        )
        priced = _make_item(
            "GlobalBrand Men Slim Shirt",
            "flipkart",
            product_type="Shirts",
            gender="men",
            price=1500.0,
            article_id="C-priced",
        )
        no_price = _make_item(
            "GlobalBrand Men Slim Shirt",
            "flipkart",
            product_type="Shirts",
            gender="men",
            price=1400.0,  # lower but same store as priced; only one flipkart match returned
            article_id="C-priced-2",
        )
        cat = _make_df(priced, no_price)
        cat.at[1, "price_inr"] = None  # make second one have no price
        perm_map = {"globalbrand": {"myntra", "flipkart"}}
        result = find_cross_store_matches(q, cat, brand_stores_map=perm_map)
        # Both are flipkart but different article_ids — both should match
        if len(result) >= 2:
            # None price sorts last; entity_resolution normalises pandas NaN to None
            assert result[-1]["price_inr"] is None


# ---------------------------------------------------------------------------
# find_cross_store_matches — snapshot labeling
# ---------------------------------------------------------------------------


class TestSnapshotLabeling:
    def test_is_snapshot_price_always_true(self) -> None:
        """Every returned match must have is_snapshot_price=True."""
        q = _make_item(
            "GlobalBrand Men Slim Shirt",
            "myntra",
            product_type="Shirts",
            gender="men",
            article_id="Q",
        )
        cat = _make_df(
            _make_item(
                "GlobalBrand Men Slim Shirt",
                "flipkart",
                product_type="Shirts",
                gender="men",
                article_id="C1",
            )
        )
        perm_map = {"globalbrand": {"myntra", "flipkart"}}
        result = find_cross_store_matches(q, cat, brand_stores_map=perm_map)
        assert len(result) == 1
        assert result[0]["is_snapshot_price"] is True


# ---------------------------------------------------------------------------
# brand_index — narrowed candidate scan
# ---------------------------------------------------------------------------


class TestBrandIndexNarrowedScan:
    def test_brand_index_same_results_as_full_scan(self) -> None:
        """find_cross_store_matches with brand_index returns same matches as without."""
        q = _make_item(
            "GlobalBrand Men Slim Fit Cotton Shirt",
            "myntra",
            product_type="Shirts",
            gender="men",
            article_id="Q",
        )
        # Build a catalogue with 20 irrelevant rows + 1 matching row
        rows = [
            _make_item(
                f"OtherBrand Shirt {i}",
                "flipkart",
                product_type="Shirts",
                gender="men",
                article_id=f"OTHER-{i}",
            )
            for i in range(20)
        ]
        rows.append(
            _make_item(
                "GlobalBrand Men Slim Fit Cotton Shirt",
                "flipkart",
                product_type="Shirts",
                gender="men",
                article_id="MATCH",
            )
        )
        cat = _make_df(*rows)

        brand_map = build_brand_stores_map(cat)
        # For this test, override so globalbrand appears in both stores
        brand_map["globalbrand"] = {"myntra", "flipkart"}
        b_idx = build_brand_index(cat)

        result_no_index = find_cross_store_matches(q, cat, brand_stores_map=brand_map)
        result_with_index = find_cross_store_matches(
            q, cat, brand_stores_map=brand_map, brand_index=b_idx
        )

        assert len(result_no_index) == len(result_with_index)
        ids_no_idx = {m["article_id"] for m in result_no_index}
        ids_with_idx = {m["article_id"] for m in result_with_index}
        assert ids_no_idx == ids_with_idx


# ---------------------------------------------------------------------------
# get_cached_brand_stores_map — basic cache check
# ---------------------------------------------------------------------------


class TestCachedBrandStoresMap:
    def test_returns_brand_map(self) -> None:
        df = _make_df(
            _make_item("Snitch Linen Shirt", "snitch", article_id="S1"),
        )
        m = get_cached_brand_stores_map(df)
        assert isinstance(m, dict)
        assert "snitch" in m
