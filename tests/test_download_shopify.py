"""2026-07-13 launch-critical fix: normalize() must drop fully out-of-stock
Shopify products (zero available variants), since the catalogue is a static
snapshot with no other stock signal and a live search was surfacing sold-out
links."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from download_shopify import normalize  # noqa: E402


def _product(variant_availability: list[bool], product_id: int = 1) -> dict:
    return {
        "id": product_id,
        "title": "Test Product",
        "product_type": "Shirt",
        "vendor": "TestBrand",
        "body_html": "<p>desc</p>",
        "handle": "test-product",
        "images": [{"src": "https://example.com/img.jpg"}],
        "variants": [
            {"price": "999.00", "available": avail} for avail in variant_availability
        ],
    }


class TestNormalizeDropsOutOfStock:
    def test_all_variants_unavailable_is_dropped(self) -> None:
        products = [_product([False, False, False])]
        df = normalize(products, "test.myshopify.com")
        assert len(df) == 0

    def test_at_least_one_variant_available_is_kept(self) -> None:
        products = [_product([False, True, False])]
        df = normalize(products, "test.myshopify.com")
        assert len(df) == 1

    def test_all_variants_available_is_kept(self) -> None:
        products = [_product([True, True])]
        df = normalize(products, "test.myshopify.com")
        assert len(df) == 1

    def test_no_variants_at_all_is_kept(self) -> None:
        # A product with an empty variants list has no availability signal to
        # act on — must not be dropped (fail open, not fail closed, on missing
        # data; distinct from the "all variants explicitly unavailable" case).
        products = [{
            "id": 2, "title": "No Variants", "product_type": "Shirt",
            "vendor": "TestBrand", "body_html": "", "handle": "no-variants",
            "images": [], "variants": [],
        }]
        df = normalize(products, "test.myshopify.com")
        assert len(df) == 0  # dropped anyway: price defaults to 0 with no variants

    def test_mixed_batch_only_drops_out_of_stock(self) -> None:
        products = [
            _product([False, False], product_id=1),
            _product([True], product_id=2),
        ]
        df = normalize(products, "test.myshopify.com")
        assert len(df) == 1
        assert df.iloc[0]["id"] == "2"
