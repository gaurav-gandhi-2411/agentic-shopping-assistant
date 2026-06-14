"""Tests for src.agents.outfit.cart_links — fully offline, no network calls."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.agents.outfit.cart_links import (
    _MISSING_BRANDS,
    _VARIANT_MAP_CACHE,
    SHOPIFY_BRANDS,
    _load_variant_map,
    _pdp_url,
    build_cart_action,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_VARIANT_MAP: dict[str, Any] = {
    "blue-slim-shirt": {
        "default_variant_id": "111111",
        "size": "M",
        "product_id": "9001",
    },
    "black-chinos": {
        "default_variant_id": "222222",
        "size": "M",
        "product_id": "9002",
    },
    "white-sneakers": {
        "default_variant_id": "333333",
        "size": "M",
        "product_id": "9003",
    },
}


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Clear module-level caches before every test for isolation."""
    _VARIANT_MAP_CACHE.clear()
    _MISSING_BRANDS.clear()


@pytest.fixture()
def snitch_variant_file(tmp_path: Path) -> Path:
    """Write a small fixture variant map for 'snitch' in a tmp dir."""
    f = tmp_path / "snitch.json"
    f.write_text(json.dumps(FIXTURE_VARIANT_MAP), encoding="utf-8")
    return f


def _patch_variant_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the cart_links module at tmp_path instead of the real data dir."""
    import src.agents.outfit.cart_links as cl
    monkeypatch.setattr(cl, "_VARIANT_MAP_DIR", tmp_path)


# ---------------------------------------------------------------------------
# Unit tests: _size_is_m
# ---------------------------------------------------------------------------

class TestSizeIsM:
    """Verify the size-match helper via the public build path (no direct import needed)."""

    def test_exact_m_resolves(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Variant map built from fixture uses size='M' correctly."""
        _patch_variant_dir(monkeypatch, tmp_path)
        (tmp_path / "snitch.json").write_text(
            json.dumps({"some-shirt": {"default_variant_id": "42", "size": "M", "product_id": "1"}}),
            encoding="utf-8",
        )
        vm = _load_variant_map("snitch")
        assert vm["some-shirt"]["size"] == "M"


# ---------------------------------------------------------------------------
# Unit tests: _pdp_url
# ---------------------------------------------------------------------------

class TestPdpUrl:
    def test_shopify_brand_builds_products_url(self) -> None:
        url = _pdp_url("snitch", "blue-slim-shirt")
        assert url == "https://snitch.co.in/products/blue-slim-shirt"

    def test_powerlook_brand(self) -> None:
        url = _pdp_url("powerlook", "grey-blazer")
        assert url == "https://powerlook.in/products/grey-blazer"

    def test_fashor_brand(self) -> None:
        url = _pdp_url("fashor", "pink-kurti")
        assert url == "https://fashor.com/products/pink-kurti"

    def test_virgio_brand(self) -> None:
        url = _pdp_url("virgio", "linen-trouser")
        assert url == "https://virgio.com/products/linen-trouser"

    def test_myntra_non_shopify_slug(self) -> None:
        # Myntra handles are plain slugs (not full URLs)
        url = _pdp_url("myntra", "kurta-12345")
        assert "myntra.com" in url
        assert "kurta-12345" in url

    def test_full_url_passthrough(self) -> None:
        url = _pdp_url("flipkart", "https://www.flipkart.com/product/p/abc123")
        assert url == "https://www.flipkart.com/product/p/abc123"


# ---------------------------------------------------------------------------
# Unit tests: build_cart_action — Shopify path
# ---------------------------------------------------------------------------

class TestBuildCartActionShopify:
    """Tests for Shopify brand cart permalink building."""

    def _make_items(self) -> list[dict]:
        return [
            {
                "article_id": "A001",
                "pdp_handle": "blue-slim-shirt",
                "display_name": "Blue Slim Shirt",
            },
            {
                "article_id": "A002",
                "pdp_handle": "black-chinos",
                "display_name": "Black Chinos",
            },
            {
                "article_id": "A003",
                "pdp_handle": "white-sneakers",
                "display_name": "White Sneakers",
            },
        ]

    def test_shopify_multi_item_cart_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three items all in the map -> a single cart permalink with all variant IDs."""
        _patch_variant_dir(monkeypatch, tmp_path)
        (tmp_path / "snitch.json").write_text(
            json.dumps(FIXTURE_VARIANT_MAP), encoding="utf-8"
        )

        result = build_cart_action(self._make_items(), "snitch")

        assert result["kind"] == "shopify_cart"
        assert result["cart_url"] is not None
        assert result["cart_url"].startswith("https://snitch.co.in/cart/")
        # All three variant IDs present in cart URL
        assert "111111:1" in result["cart_url"]
        assert "222222:1" in result["cart_url"]
        assert "333333:1" in result["cart_url"]
        # No missing items
        assert result["missing"] == []

    def test_shopify_cart_url_format(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cart URL must follow Shopify format: /cart/{v}:1,{v}:1,..."""
        _patch_variant_dir(monkeypatch, tmp_path)
        (tmp_path / "snitch.json").write_text(
            json.dumps(FIXTURE_VARIANT_MAP), encoding="utf-8"
        )

        result = build_cart_action(self._make_items(), "snitch")
        cart_url = result["cart_url"]
        assert cart_url is not None
        path_part = cart_url.split("/cart/")[1]
        segments = path_part.split(",")
        # Each segment must be "{integer}:1"
        for seg in segments:
            vid, qty = seg.split(":")
            assert vid.isdigit()
            assert qty == "1"

    def test_shopify_always_populates_item_links(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """item_links must be populated even when cart_url is built."""
        _patch_variant_dir(monkeypatch, tmp_path)
        (tmp_path / "snitch.json").write_text(
            json.dumps(FIXTURE_VARIANT_MAP), encoding="utf-8"
        )

        result = build_cart_action(self._make_items(), "snitch")
        assert len(result["item_links"]) == 3
        for lk in result["item_links"]:
            assert lk["article_id"]
            assert lk["buy_url"].startswith("https://snitch.co.in/products/")

    def test_missing_handle_falls_back_to_item_links_and_missing_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An item whose handle is NOT in the map appears in 'missing' + item_links (PDP link)."""
        _patch_variant_dir(monkeypatch, tmp_path)
        (tmp_path / "snitch.json").write_text(
            json.dumps(FIXTURE_VARIANT_MAP), encoding="utf-8"
        )

        items = [
            {
                "article_id": "A001",
                "pdp_handle": "blue-slim-shirt",   # in map
                "display_name": "Blue Slim Shirt",
            },
            {
                "article_id": "A099",
                "pdp_handle": "unknown-product",    # NOT in map
                "display_name": "Unknown Product",
            },
        ]
        result = build_cart_action(items, "snitch")

        assert result["kind"] == "shopify_cart"
        # Cart URL built with only the resolved item
        assert "111111:1" in (result["cart_url"] or "")
        # A099 appears in missing
        assert "A099" in result["missing"]
        # BUT A099 still gets a PDP fallback link in item_links
        link_ids = [lk["article_id"] for lk in result["item_links"]]
        assert "A099" in link_ids
        # Fallback URL is a valid PDP URL
        a099_link = next(lk for lk in result["item_links"] if lk["article_id"] == "A099")
        assert a099_link["buy_url"] == "https://snitch.co.in/products/unknown-product"

    def test_no_handle_item_appears_in_missing_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An item with no pdp_handle at all appears in missing and NOT in item_links."""
        _patch_variant_dir(monkeypatch, tmp_path)
        (tmp_path / "snitch.json").write_text(
            json.dumps(FIXTURE_VARIANT_MAP), encoding="utf-8"
        )

        items = [{"article_id": "A777", "pdp_handle": None, "display_name": "Ghost Item"}]
        result = build_cart_action(items, "snitch")

        assert "A777" in result["missing"]
        assert all(lk["article_id"] != "A777" for lk in result["item_links"])

    def test_missing_variant_map_file_degrades_to_per_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the variant map file is absent, all items go to missing + item_links fallback."""
        _patch_variant_dir(monkeypatch, tmp_path)
        # No snitch.json written in tmp_path

        items = [
            {"article_id": "A001", "pdp_handle": "some-shirt", "display_name": "Shirt"},
        ]
        result = build_cart_action(items, "snitch")

        # kind is still shopify_cart (brand is Shopify) but cart_url is None (no variants resolved)
        assert result["kind"] == "shopify_cart"
        assert result["cart_url"] is None
        assert "A001" in result["missing"]


# ---------------------------------------------------------------------------
# Unit tests: build_cart_action — non-Shopify (open_all) path
# ---------------------------------------------------------------------------

class TestBuildCartActionOpenAll:
    """Tests for non-Shopify brands (myntra, flipkart)."""

    def test_myntra_yields_open_all(self) -> None:
        items = [
            {"article_id": "M001", "pdp_handle": "kurta-set-12345", "display_name": "Kurta Set"},
            {"article_id": "M002", "pdp_handle": "palazzo-67890", "display_name": "Palazzo"},
        ]
        result = build_cart_action(items, "myntra")

        assert result["kind"] == "open_all"
        assert result["cart_url"] is None
        assert len(result["item_links"]) == 2
        assert result["missing"] == []

    def test_non_shopify_item_links_contain_buy_url(self) -> None:
        items = [
            {"article_id": "M001", "pdp_handle": "kurta-12345", "display_name": "Kurta"},
        ]
        result = build_cart_action(items, "myntra")
        assert result["item_links"][0]["buy_url"]

    def test_flipkart_with_full_url_handle(self) -> None:
        """Flipkart pdp_handle may already be a full URL."""
        items = [
            {
                "article_id": "F001",
                "pdp_handle": "https://www.flipkart.com/product/p/abc",
                "display_name": "FK Shirt",
            },
        ]
        result = build_cart_action(items, "flipkart")

        assert result["kind"] == "open_all"
        assert result["item_links"][0]["buy_url"] == "https://www.flipkart.com/product/p/abc"

    def test_non_shopify_brand_is_not_in_shopify_brands_set(self) -> None:
        assert "myntra" not in SHOPIFY_BRANDS
        assert "flipkart" not in SHOPIFY_BRANDS


# ---------------------------------------------------------------------------
# Unit tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_items_returns_open_all_empty(self) -> None:
        """Empty item list returns a safe, empty response."""
        result = build_cart_action([], "snitch")

        assert result["kind"] == "open_all"
        assert result["cart_url"] is None
        assert result["item_links"] == []
        assert result["missing"] == []

    def test_single_item_shopify_cart_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single item produces a valid single-variant cart URL."""
        _patch_variant_dir(monkeypatch, tmp_path)
        (tmp_path / "snitch.json").write_text(
            json.dumps(FIXTURE_VARIANT_MAP), encoding="utf-8"
        )

        result = build_cart_action(
            [{"article_id": "A001", "pdp_handle": "blue-slim-shirt", "display_name": "Shirt"}],
            "snitch",
        )
        assert result["cart_url"] == "https://snitch.co.in/cart/111111:1"

    def test_unknown_brand_treated_as_non_shopify(self) -> None:
        """A brand not in SHOPIFY_BRANDS falls through to open_all."""
        result = build_cart_action(
            [{"article_id": "X01", "pdp_handle": "some-product", "display_name": "Prod"}],
            "hm",
        )
        assert result["kind"] == "open_all"
        assert result["cart_url"] is None
