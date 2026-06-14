"""Cart-link builder for the "Add the whole look" feature.

Shopify brands (snitch, powerlook, fashor, virgio):
    Builds a single cart permalink pre-filling ALL look items via
    the Shopify cart route:
        https://{domain}/cart/{variantId}:1,{variantId}:1,...

Non-Shopify brands (myntra, flipkart):
    Returns per-item buy links and kind="open_all" so the UI can
    offer "Open all in new tabs" behaviour instead.

Cross-store looks (unified mode — items span more than one store):
    Per the spec, there is NO single cross-store cart.  Each item is
    bought via its own store's deep-link (item.pdp_url or build_pdp_url).
    build_cart_action returns kind="open_all", cart_url=None, and
    item_links where each buy_url is that item's OWN store's PDP.
    Never defaults a non-Myntra item to myntra.com.

Variant IDs are resolved from pre-built JSON maps at:
    data/processed/shopify_variants/{brand}.json

These maps are built once (offline) by scripts/build_shopify_variant_map.py
and cached in memory at module-level after the first access. No network
calls happen at request time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config.stores import build_pdp_url as _stores_build_pdp_url

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Brands whose storefronts are Shopify (support /cart/{v}:qty permalink)
SHOPIFY_BRANDS: frozenset[str] = frozenset({"snitch", "powerlook", "fashor", "virgio"})

# Where the crawled variant maps live (relative to project root)
_VARIANT_MAP_DIR = Path("data/processed/shopify_variants")

# Module-level cache: brand -> handle -> variant info
_VARIANT_MAP_CACHE: dict[str, dict[str, dict[str, str]]] = {}
_MISSING_BRANDS: set[str] = set()  # brands with no map file (logged once, then skip)

# Domain lookup mirrored from brand YAML pdp_url_template
# Kept here to avoid loading YAML at call time (pure, fast).
_BRAND_DOMAIN: dict[str, str] = {
    "snitch": "snitch.co.in",
    "powerlook": "powerlook.in",
    "fashor": "fashor.com",
    "virgio": "virgio.com",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_variant_map(brand: str) -> dict[str, dict[str, str]]:
    """Load the variant map for brand from disk (or return {} if missing).

    Results are cached indefinitely for the lifetime of the process.
    """
    if brand in _VARIANT_MAP_CACHE:
        return _VARIANT_MAP_CACHE[brand]

    if brand in _MISSING_BRANDS:
        return {}

    map_path = _VARIANT_MAP_DIR / f"{brand}.json"
    if not map_path.exists():
        if brand not in _MISSING_BRANDS:
            logger.warning(
                "Shopify variant map not found for brand=%s (path=%s). "
                "Falling back to per-item PDP links. "
                "Run scripts/build_shopify_variant_map.py to build it.",
                brand,
                map_path,
            )
            _MISSING_BRANDS.add(brand)
        return {}

    try:
        data: dict[str, Any] = json.loads(map_path.read_text(encoding="utf-8"))
        _VARIANT_MAP_CACHE[brand] = data
        logger.debug("Loaded Shopify variant map for brand=%s (%d handles)", brand, len(data))
        return data
    except Exception as exc:
        logger.warning("Failed to parse variant map for brand=%s: %s", brand, exc)
        _MISSING_BRANDS.add(brand)
        return {}


def _pdp_url(brand: str, handle: str) -> str:
    """Build a per-item PDP URL from domain and handle.

    For Shopify brands the URL is https://{domain}/products/{handle}.
    For non-Shopify brands we use the handle as-is (Myntra/Flipkart store
    the full URL in pdp_handle already, or a partial path).
    """
    domain = _BRAND_DOMAIN.get(brand)
    if domain:
        return f"https://{domain}/products/{handle}"
    # Non-shopify: handle might already be a full URL (flipkart) or a slug (myntra)
    if handle.startswith("http"):
        return handle
    # Myntra-style handle: numeric product ID used in URL
    return f"https://www.myntra.com/{handle}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cart_action(items: list[dict], brand: str) -> dict:
    """Build a cart action dict for a list of look items.

    Parameters
    ----------
    items:
        List of item dicts, each expected to have at minimum:
        - ``article_id``: str
        - ``pdp_handle``: str | None  — used for variant lookup + PDP fallback
        - ``display_name`` or ``prod_name``: str
        - ``store``: str | None  — per-item store slug (populated since Phase E)
    brand:
        Brand slug (e.g. "snitch", "myntra", "unified").  When ``brand``
        is ``"unified"`` or items span more than one store, the cross-store
        path is taken regardless of the ``brand`` argument — each item uses
        its OWN store for link building.

    Returns
    -------
    dict with shape::

        {
            "kind": "shopify_cart" | "open_all",
            "cart_url": str | None,
            "item_links": [{"article_id": str, "name": str, "buy_url": str}],
            "missing": [article_id, ...]   # items with no resolvable link
        }

    Notes
    -----
    - **Cross-store look** (items span >1 distinct store, or brand="unified"):
      ``kind="open_all"``, ``cart_url=None``, each ``buy_url`` is built from
      the item's OWN ``store`` field via ``build_pdp_url``.  A non-Myntra item
      is never defaulted to myntra.com.
    - **Single Shopify store**: ``cart_url`` is the multi-item Shopify cart
      permalink; ``item_links`` always contains per-item PDP links too.
    - **Single non-Shopify store**: ``cart_url`` is None; per-item links.
    - Items with no ``pdp_handle`` and no variant mapping appear in ``missing``.
    - Empty ``items`` list returns an empty-but-valid response with kind="open_all".
    """
    if not items:
        return {
            "kind": "open_all",
            "cart_url": None,
            "item_links": [],
            "missing": [],
        }

    # Resolve per-item stores; prefer item["store"] over the passed brand arg.
    item_stores = {
        str(it.get("store") or "").strip().lower()
        for it in items
        if it.get("store")
    }

    # Determine the effective single store slug (may be None/empty for legacy items).
    # Use item-level store when all items share one; fall back to passed brand.
    if len(item_stores) == 1:
        effective_store = next(iter(item_stores))
    elif not item_stores:
        # No per-item store info — legacy path: use brand arg as-is.
        effective_store = brand
    else:
        # Multiple stores — cross-store look.
        effective_store = None

    # Cross-store look: items span >1 store OR the caller passed brand="unified"
    # (indicating the unified cross-store index is in use).
    # Spec: no single cart across stores; buy per-item via each item's OWN deep-link.
    # When brand="unified" we always use per-item links so the Shopify cart path is
    # never accidentally triggered for a look assembled from the unified index.
    is_cross_store = effective_store is None or brand == "unified"
    if is_cross_store:
        return _build_cross_store_action(items)

    # Single-store paths (legacy per-brand behaviour preserved).
    is_shopify = effective_store in SHOPIFY_BRANDS
    if is_shopify:
        variant_map = _load_variant_map(effective_store)
        domain = _BRAND_DOMAIN.get(effective_store, "")
        return _build_shopify_action(items, effective_store, domain, variant_map)

    return _build_open_all_action(items, effective_store)


def _build_cross_store_action(items: list[dict]) -> dict:
    """Build a per-item open-all action for a cross-store look.

    Each item's buy_url is built from its OWN ``store`` field via
    ``build_pdp_url`` from src.config.stores.  A non-Myntra item is
    never defaulted to myntra.com — if no URL can be resolved the item
    appears in ``missing``.

    Returns kind="open_all", cart_url=None.
    """
    item_links: list[dict] = []
    missing: list[str] = []

    for item in items:
        article_id = str(item.get("article_id") or "")
        name = item.get("display_name") or item.get("prod_name") or article_id
        store = (item.get("store") or "").strip().lower() or None

        # Build the URL using the item's own store — never fall back to a
        # different store's domain (spec: no cross-store cart, no myntra fallback).
        buy_url = _stores_build_pdp_url(store, item)

        if buy_url:
            item_links.append({
                "article_id": article_id,
                "name": name,
                "buy_url": buy_url,
            })
        else:
            missing.append(article_id)
            logger.debug(
                "Cross-store: no PDP URL resolved for article_id=%s store=%s handle=%s",
                article_id,
                store,
                item.get("pdp_handle"),
            )

    return {
        "kind": "open_all",
        "cart_url": None,
        "item_links": item_links,
        "missing": missing,
    }


def _build_shopify_action(
    items: list[dict],
    brand: str,
    domain: str,
    variant_map: dict[str, dict[str, str]],
) -> dict:
    """Build a Shopify cart permalink + per-item links for a Shopify brand."""
    cart_segments: list[str] = []
    item_links: list[dict] = []
    missing: list[str] = []

    for item in items:
        article_id = str(item.get("article_id") or "")
        handle = (item.get("pdp_handle") or "").strip()
        name = item.get("display_name") or item.get("prod_name") or article_id

        # Per-item PDP link (always built if we have a handle)
        pdp_url = _pdp_url(brand, handle) if handle else None

        variant_info = variant_map.get(handle) if handle else None
        if variant_info:
            variant_id = variant_info["default_variant_id"]
            cart_segments.append(f"{variant_id}:1")
            item_links.append({
                "article_id": article_id,
                "name": name,
                "buy_url": pdp_url or f"https://{domain}/products/{handle}",
            })
        else:
            # No variant mapping — include as per-item fallback only
            if pdp_url:
                item_links.append({
                    "article_id": article_id,
                    "name": name,
                    "buy_url": pdp_url,
                })
            missing.append(article_id)
            if handle:
                logger.debug(
                    "No variant found for brand=%s handle=%s article_id=%s",
                    brand,
                    handle,
                    article_id,
                )

    cart_url: str | None = None
    if cart_segments and domain:
        cart_url = f"https://{domain}/cart/{','.join(cart_segments)}"

    return {
        "kind": "shopify_cart",
        "cart_url": cart_url,
        "item_links": item_links,
        "missing": missing,
    }


def _build_open_all_action(items: list[dict], brand: str) -> dict:
    """Build an open-all dict for non-Shopify brands."""
    item_links: list[dict] = []
    missing: list[str] = []

    for item in items:
        article_id = str(item.get("article_id") or "")
        handle = (item.get("pdp_handle") or "").strip()
        name = item.get("display_name") or item.get("prod_name") or article_id

        if handle:
            item_links.append({
                "article_id": article_id,
                "name": name,
                "buy_url": _pdp_url(brand, handle),
            })
        else:
            missing.append(article_id)

    return {
        "kind": "open_all",
        "cart_url": None,
        "item_links": item_links,
        "missing": missing,
    }
