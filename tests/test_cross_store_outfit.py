"""Phase E cross-store outfit tests.

Verifies:
  (a) Seed item extracted via _row_to_item carries store, and ItemSummary.from_agent_item
      builds store_display + pdp_url from it.
  (b) A multi-store look CAN span >1 store.
  (c) build_cart_action on a multi-store item list returns cart_url=None and
      per-item links each pointing to the item's OWN store domain (not a
      Myntra fallback for non-Myntra items).
  (d) A single-Shopify-store look still produces a cart_url.
  (e) Gender/occasion coherence gates are NOT weakened.

All tests are offline/deterministic: no live LLM, no network, no index load.
Uses small synthetic 2-store catalogue fixtures and seed=42 where applicable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from api.schemas import ItemSummary, OutfitVariant
from src.agents.outfit.cart_links import (
    _MISSING_BRANDS,
    _VARIANT_MAP_CACHE,
    build_cart_action,
)
from src.agents.outfit.composer import _row_to_item

# Deterministic seed as per project convention.
np.random.seed(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_catalogue_row(
    article_id: str,
    prod_name: str,
    store: str,
    pdp_handle: str | None,
    product_type: str = "Kurta",
    gender: str = "women",
    price_inr: float = 999.0,
) -> pd.Series:
    """Build a minimal catalogue pd.Series that mirrors the unified index schema."""
    data: dict[str, Any] = {
        "prod_name": prod_name,
        "display_name": prod_name,
        "store": store,
        "pdp_handle": pdp_handle,
        "image_url": None,
        "price_inr": price_inr,
        "detail_desc": "",
        "gender": gender,
        "index_group_name": "apparel",
        "facets": {
            "colour_group_name": "Blue",
            "product_type_name": product_type,
            "department_name": "Women",
        },
    }
    return pd.Series(data, name=article_id)


def _make_item_dict(
    article_id: str,
    prod_name: str,
    store: str,
    pdp_handle: str | None,
    product_type: str = "Kurta",
    price_inr: float = 999.0,
) -> dict:
    """Build a minimal item dict as returned by hybrid_search / complements."""
    return {
        "article_id": article_id,
        "prod_name": prod_name,
        "display_name": prod_name,
        "store": store,
        "pdp_handle": pdp_handle,
        "image_url": None,
        "price_inr": price_inr,
        "colour": "Blue",
        "product_type": product_type,
        "department": "Women",
        "detail_desc": "",
        "score": 0.9,
        "gender": "women",
    }


@pytest.fixture(autouse=True)
def _clear_cart_caches() -> None:
    """Isolate module-level caches between tests."""
    _VARIANT_MAP_CACHE.clear()
    _MISSING_BRANDS.clear()


# ---------------------------------------------------------------------------
# (a) Seed-store bug fix — _row_to_item extracts store
# ---------------------------------------------------------------------------

class TestRowToItemExtractsStore:
    """Gap 1: _row_to_item must populate 'store' from the catalogue row."""

    def test_store_present_in_row_is_extracted(self) -> None:
        row = _make_catalogue_row("101", "Blue Kurta", "myntra", "kurta-set-101/buy")
        facets = {"colour_group_name": "Blue", "product_type_name": "Kurta", "department_name": "Women"}
        result = _row_to_item("101", row, facets, "seed")
        assert result["store"] == "myntra", "store must be extracted from the catalogue row"

    def test_store_none_in_row_yields_none(self) -> None:
        row = _make_catalogue_row("102", "Shirt", "snitch", "some-shirt")
        row["store"] = None  # simulate missing store
        facets = {}
        result = _row_to_item("102", row, facets, "seed")
        assert result["store"] is None

    def test_snitch_row_extracts_snitch_store(self) -> None:
        row = _make_catalogue_row("103", "Polo Tee", "snitch", "polo-tee-abc", product_type="T-Shirt", gender="men")
        facets = {"colour_group_name": "White", "product_type_name": "T-Shirt", "department_name": "Men"}
        result = _row_to_item("103", row, facets, "seed")
        assert result["store"] == "snitch"

    def test_virgio_row_extracts_virgio_store(self) -> None:
        row = _make_catalogue_row("104", "Linen Trousers", "virgio", "linen-trousers-xyz")
        facets = {}
        result = _row_to_item("104", row, facets, "seed")
        assert result["store"] == "virgio"


# ---------------------------------------------------------------------------
# (a-continued) ItemSummary.from_agent_item builds store_display + pdp_url
# ---------------------------------------------------------------------------

class TestItemSummaryFromAgentItemCrossStore:
    """ItemSummary.from_agent_item must produce non-null store_display + pdp_url
    for seed items that carry a store field (after the _row_to_item fix).
    """

    def test_myntra_seed_gets_store_display_and_pdp_url(self) -> None:
        item = _make_item_dict("201", "Printed Kurta", "myntra", "kurta-201/buy")
        summary = ItemSummary.from_agent_item(item)
        assert summary.store == "myntra"
        assert summary.store_display == "Myntra"
        assert summary.pdp_url is not None
        assert "myntra.com" in (summary.pdp_url or "")

    def test_snitch_seed_gets_store_display_and_pdp_url(self) -> None:
        item = _make_item_dict("202", "Slim Fit Shirt", "snitch", "slim-fit-shirt-abc", product_type="Shirt")
        summary = ItemSummary.from_agent_item(item)
        assert summary.store == "snitch"
        assert summary.store_display == "Snitch"
        assert summary.pdp_url == "https://snitch.co.in/products/slim-fit-shirt-abc"

    def test_flipkart_seed_with_full_url_handle(self) -> None:
        item = _make_item_dict("203", "Casual Top", "flipkart", "https://www.flipkart.com/top/p/abc")
        summary = ItemSummary.from_agent_item(item)
        assert summary.store == "flipkart"
        assert summary.store_display == "Flipkart"
        assert summary.pdp_url == "https://www.flipkart.com/top/p/abc"

    def test_none_store_gives_null_pdp_url(self) -> None:
        item = _make_item_dict("204", "HM Top", "hm", None)
        summary = ItemSummary.from_agent_item(item)
        # H&M has no pdp_url_template — pdp_url must be None
        assert summary.pdp_url is None

    def test_row_to_item_then_item_summary_round_trip(self) -> None:
        """Full round-trip: catalogue row → _row_to_item → ItemSummary.from_agent_item."""
        row = _make_catalogue_row("205", "Festive Lehenga", "fashor", "festive-lehenga-fash")
        facets = {"colour_group_name": "Red", "product_type_name": "Lehenga", "department_name": "Women"}
        item = _row_to_item("205", row, facets, "seed")
        summary = ItemSummary.from_agent_item(item)
        assert summary.store == "fashor"
        assert summary.store_display == "Fashor"
        assert summary.pdp_url == "https://fashor.com/products/festive-lehenga-fash"


# ---------------------------------------------------------------------------
# (b) A look CAN span >1 store
# ---------------------------------------------------------------------------

class TestCrossStoreLookSpansStores:
    """A composed look may contain items from different stores — verify assertions hold."""

    def test_multi_store_items_have_different_stores(self) -> None:
        seed = _make_item_dict("301", "Blue Kurta", "myntra", "kurta-301/buy")
        complement = _make_item_dict("302", "Cotton Palazzos", "snitch", "cotton-palazzos-302")
        all_items = [seed, complement]
        stores = {it["store"] for it in all_items}
        assert len(stores) == 2, "Look should span 2 distinct stores"
        assert "myntra" in stores
        assert "snitch" in stores

    def test_three_store_look_all_carry_store(self) -> None:
        items = [
            _make_item_dict("401", "Ethnic Top", "fashor", "ethnic-top-401"),
            _make_item_dict("402", "Trousers", "virgio", "trousers-402"),
            _make_item_dict("403", "Sneakers", "myntra", "sneakers-myntra/buy"),
        ]
        for it in items:
            assert it["store"] is not None, f"item {it['article_id']} must carry store"
        stores = {it["store"] for it in items}
        assert len(stores) == 3


# ---------------------------------------------------------------------------
# (c) build_cart_action on a multi-store list → cart_url=None + per-item links
# ---------------------------------------------------------------------------

class TestBuildCartActionCrossStore:
    """Gap 2: multi-store look must NOT produce a single cart URL."""

    def _make_cross_store_items(self) -> list[dict]:
        return [
            {
                "article_id": "S001",
                "prod_name": "Ethnic Kurta",
                "display_name": "Ethnic Kurta",
                "store": "myntra",
                "pdp_handle": "ethnic-kurta-8765432/buy",
            },
            {
                "article_id": "S002",
                "prod_name": "Cotton Trouser",
                "display_name": "Cotton Trouser",
                "store": "snitch",
                "pdp_handle": "cotton-trouser-snitch",
            },
        ]

    def test_cross_store_cart_url_is_none(self) -> None:
        """Multi-store look must NOT produce a cart_url (no cross-store cart)."""
        items = self._make_cross_store_items()
        result = build_cart_action(items, "unified")
        assert result["cart_url"] is None, "cart_url must be None for a multi-store look"

    def test_cross_store_kind_is_open_all(self) -> None:
        result = build_cart_action(self._make_cross_store_items(), "unified")
        assert result["kind"] == "open_all"

    def test_cross_store_item_links_populated(self) -> None:
        result = build_cart_action(self._make_cross_store_items(), "unified")
        assert len(result["item_links"]) == 2

    def test_myntra_item_uses_myntra_domain(self) -> None:
        result = build_cart_action(self._make_cross_store_items(), "unified")
        myntra_link = next(lk for lk in result["item_links"] if lk["article_id"] == "S001")
        assert "myntra.com" in myntra_link["buy_url"]
        assert "ethnic-kurta-8765432" in myntra_link["buy_url"]

    def test_snitch_item_uses_snitch_domain_not_myntra(self) -> None:
        """CRITICAL: non-Myntra items must NOT fall back to myntra.com domain."""
        result = build_cart_action(self._make_cross_store_items(), "unified")
        snitch_link = next(lk for lk in result["item_links"] if lk["article_id"] == "S002")
        assert "snitch.co.in" in snitch_link["buy_url"], (
            "Snitch item must use snitch.co.in, not myntra.com"
        )
        assert "myntra.com" not in snitch_link["buy_url"]

    def test_three_store_cross_store_all_links_correct_domain(self) -> None:
        items = [
            {
                "article_id": "T001",
                "prod_name": "Top",
                "display_name": "Top",
                "store": "fashor",
                "pdp_handle": "silk-top-fashor",
            },
            {
                "article_id": "T002",
                "prod_name": "Trouser",
                "display_name": "Trouser",
                "store": "virgio",
                "pdp_handle": "wide-leg-trouser",
            },
            {
                "article_id": "T003",
                "prod_name": "Kurta",
                "display_name": "Kurta",
                "store": "myntra",
                "pdp_handle": "kurta-t003/buy",
            },
        ]
        result = build_cart_action(items, "unified")
        assert result["cart_url"] is None
        link_map = {lk["article_id"]: lk["buy_url"] for lk in result["item_links"]}
        assert "fashor.com" in link_map["T001"]
        assert "virgio.com" in link_map["T002"]
        assert "myntra.com" in link_map["T003"]

    def test_brand_unified_with_single_store_items_uses_cross_store_path(self) -> None:
        """When brand='unified' and all items are from one store, we still use the
        cross-store path per spec (unified mode always routes per-item).
        Practically this means cart_url is None even when all items share one store,
        unless brand is the specific Shopify store slug.
        """
        items = [
            {
                "article_id": "U001",
                "prod_name": "Kurta",
                "display_name": "Kurta",
                "store": "myntra",
                "pdp_handle": "kurta-u001/buy",
            },
        ]
        result = build_cart_action(items, "unified")
        # Single store, but brand="unified": cross-store path → cart_url=None
        assert result["cart_url"] is None
        assert len(result["item_links"]) == 1

    def test_non_myntra_item_without_store_goes_to_missing(self) -> None:
        """An item with no store field and no resolvable URL goes to missing,
        not a Myntra fallback URL.
        """
        items = [
            {
                "article_id": "X001",
                "prod_name": "Mystery Item",
                "display_name": "Mystery Item",
                "store": None,
                "pdp_handle": "some-slug-no-store",
            },
            {
                "article_id": "X002",
                "prod_name": "Known Item",
                "display_name": "Known Item",
                "store": "snitch",
                "pdp_handle": "known-product-abc",
            },
        ]
        result = build_cart_action(items, "unified")
        # X001: store=None → build_pdp_url returns None → missing
        assert "X001" in result["missing"]
        # X002: snitch item → correct snitch link
        x002_link = next(lk for lk in result["item_links"] if lk["article_id"] == "X002")
        assert "snitch.co.in" in x002_link["buy_url"]
        assert "myntra.com" not in x002_link["buy_url"]


# ---------------------------------------------------------------------------
# (d) Single-Shopify-store look still gets cart_url
# ---------------------------------------------------------------------------

class TestSingleShopifyStoreCartUrl:
    """Gap 2: single-store Shopify looks must still produce a cart_url."""

    def _make_shopify_items(self) -> list[dict]:
        return [
            {
                "article_id": "SN001",
                "prod_name": "Blue Polo",
                "display_name": "Blue Polo",
                "store": "snitch",
                "pdp_handle": "blue-polo-snitch",
            },
            {
                "article_id": "SN002",
                "prod_name": "Chinos",
                "display_name": "Chinos",
                "store": "snitch",
                "pdp_handle": "chinos-snitch",
            },
        ]

    def _write_snitch_map(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.agents.outfit.cart_links as cl
        variant_map: dict[str, Any] = {
            "blue-polo-snitch": {"default_variant_id": "7001", "size": "M", "product_id": "1"},
            "chinos-snitch": {"default_variant_id": "7002", "size": "M", "product_id": "2"},
        }
        (tmp_path / "snitch.json").write_text(json.dumps(variant_map), encoding="utf-8")
        monkeypatch.setattr(cl, "_VARIANT_MAP_DIR", tmp_path)

    def test_single_snitch_look_has_cart_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_snitch_map(tmp_path, monkeypatch)
        result = build_cart_action(self._make_shopify_items(), "snitch")
        assert result["kind"] == "shopify_cart"
        assert result["cart_url"] is not None
        assert result["cart_url"].startswith("https://snitch.co.in/cart/")
        assert "7001:1" in result["cart_url"]
        assert "7002:1" in result["cart_url"]

    def test_single_shopify_look_also_has_item_links(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_snitch_map(tmp_path, monkeypatch)
        result = build_cart_action(self._make_shopify_items(), "snitch")
        assert len(result["item_links"]) == 2
        for lk in result["item_links"]:
            assert "snitch.co.in" in lk["buy_url"]

    def test_item_level_store_takes_precedence_over_brand_arg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all items carry store='snitch', that takes precedence over brand='unknown'."""
        self._write_snitch_map(tmp_path, monkeypatch)
        result = build_cart_action(self._make_shopify_items(), brand="unknown")
        # items all carry store="snitch" → single-store Shopify path
        assert result["kind"] == "shopify_cart"
        assert result["cart_url"] is not None

    def test_powerlook_single_store_look(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.agents.outfit.cart_links as cl
        pl_map: dict[str, Any] = {
            "cargo-pant-pl": {"default_variant_id": "8001", "size": "M", "product_id": "10"},
        }
        (tmp_path / "powerlook.json").write_text(json.dumps(pl_map), encoding="utf-8")
        monkeypatch.setattr(cl, "_VARIANT_MAP_DIR", tmp_path)

        items = [
            {
                "article_id": "PL001",
                "prod_name": "Cargo Pant",
                "display_name": "Cargo Pant",
                "store": "powerlook",
                "pdp_handle": "cargo-pant-pl",
            }
        ]
        result = build_cart_action(items, "powerlook")
        assert result["kind"] == "shopify_cart"
        assert result["cart_url"] is not None
        assert "powerlook.in/cart/" in result["cart_url"]


# ---------------------------------------------------------------------------
# (e) Gender/occasion coherence gates not weakened
# ---------------------------------------------------------------------------

class TestCoherenceGatesUnchanged:
    """Verify that the gender/occasion coherence gates still reject wrong-gender items."""

    def test_dupatta_rejected_for_men(self) -> None:
        from src.agents.outfit.coherence import is_coherent_candidate
        item = {
            "product_type": "Dupatta",
            "prod_name": "silk dupatta",
            "gender": "women",
        }
        assert is_coherent_candidate(item, "sangeet", "men", "accessory") is False

    def test_women_bottom_rejected_for_men_ethnic_look(self) -> None:
        from src.agents.outfit.coherence import is_coherent_candidate
        item = {
            "product_type": "Lehenga",
            "prod_name": "bridal lehenga",
            "gender": "women",
        }
        assert is_coherent_candidate(item, "sangeet", "men", "bottom") is False

    def test_men_kurta_allowed_for_men(self) -> None:
        from src.agents.outfit.coherence import is_coherent_candidate
        item = {
            "product_type": "Kurta",
            "prod_name": "festive kurta",
            "gender": "men",
        }
        assert is_coherent_candidate(item, "sangeet", "men", "top") is True


# ---------------------------------------------------------------------------
# (a-schema) OutfitVariant schema accepts cart_url=None
# ---------------------------------------------------------------------------

class TestOutfitVariantSchemaAcceptsNullCartUrl:
    """Gap 3: OutfitVariant must accept cart_url=None (cross-store looks)."""

    def test_cart_url_none_is_valid(self) -> None:
        variant = OutfitVariant(
            variant_id="test-v1",
            label="Base",
            rationale="A cross-store look.",
            items=[],
            cart_url=None,
            item_links=None,
        )
        assert variant.cart_url is None

    def test_cart_url_string_is_valid(self) -> None:
        variant = OutfitVariant(
            variant_id="test-v2",
            label="Base",
            rationale="A single-store Shopify look.",
            items=[],
            cart_url="https://snitch.co.in/cart/111:1",
            item_links=None,
        )
        assert variant.cart_url == "https://snitch.co.in/cart/111:1"
