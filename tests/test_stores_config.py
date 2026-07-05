"""Unit tests for src/config/stores.py — store config, build_pdp_url, display names."""
from __future__ import annotations

import pytest

from src.config.stores import (
    STORE_CONFIG,
    build_pdp_url,
    get_inactive_stores,
    get_store_display_name,
)

# ---------------------------------------------------------------------------
# get_store_display_name
# ---------------------------------------------------------------------------


def test_display_name_known_stores() -> None:
    assert get_store_display_name("hm") == "H&M"
    assert get_store_display_name("myntra") == "Myntra"
    assert get_store_display_name("flipkart") == "Flipkart"
    assert get_store_display_name("snitch") == "Snitch"
    assert get_store_display_name("fashor") == "Fashor"
    assert get_store_display_name("powerlook") == "Powerlook"
    assert get_store_display_name("virgio") == "Virgio"


def test_display_name_none_input() -> None:
    assert get_store_display_name(None) is None
    assert get_store_display_name("") is None


def test_display_name_unknown_store() -> None:
    # Unknown store returns None rather than crashing or showing empty string.
    assert get_store_display_name("unknownbrand") is None


# ---------------------------------------------------------------------------
# build_pdp_url — Myntra (relative path handle)
# ---------------------------------------------------------------------------


def test_build_pdp_url_myntra_relative_handle() -> None:
    row = {"pdp_handle": "kurtas/brand/some-kurta/17048614/buy"}
    url = build_pdp_url("myntra", row)
    assert url == "https://www.myntra.com/kurtas/brand/some-kurta/17048614/buy"


def test_build_pdp_url_myntra_none_handle() -> None:
    assert build_pdp_url("myntra", {"pdp_handle": None}) is None


# ---------------------------------------------------------------------------
# build_pdp_url — Flipkart (full URL handle)
# ---------------------------------------------------------------------------


def test_build_pdp_url_flipkart_full_url() -> None:
    full_url = "https://www.flipkart.com/some-product/p/itm123?pid=ABC&marketplace=FLIPKART"
    row = {"pdp_handle": full_url}
    url = build_pdp_url("flipkart", row)
    assert url == full_url  # returned verbatim, no template expansion


def test_build_pdp_url_flipkart_non_url_handle_returns_none() -> None:
    # Flipkart has no template; a non-URL handle (shouldn't exist in practice) returns None.
    row = {"pdp_handle": "just-a-slug"}
    assert build_pdp_url("flipkart", row) is None


# ---------------------------------------------------------------------------
# build_pdp_url — Shopify stores (slug handle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "store, expected_prefix",
    [
        ("snitch", "https://snitch.co.in/products/"),
        ("fashor", "https://fashor.com/products/"),
        ("powerlook", "https://powerlook.in/products/"),
        ("virgio", "https://virgio.com/products/"),
    ],
)
def test_build_pdp_url_shopify_stores(store: str, expected_prefix: str) -> None:
    handle = "my-product-handle-xyz"
    url = build_pdp_url(store, {"pdp_handle": handle})
    assert url == f"{expected_prefix}{handle}"


# ---------------------------------------------------------------------------
# build_pdp_url — H&M (no template, no handle)
# ---------------------------------------------------------------------------


def test_build_pdp_url_hm_no_handle() -> None:
    # H&M rows have NULL pdp_handle in the unified catalogue.
    assert build_pdp_url("hm", {"pdp_handle": None}) is None


def test_build_pdp_url_hm_with_handle_returns_none() -> None:
    # H&M has no pdp_url_template — even with a handle we return None.
    assert build_pdp_url("hm", {"pdp_handle": "some-handle"}) is None


# ---------------------------------------------------------------------------
# build_pdp_url — edge cases
# ---------------------------------------------------------------------------


def test_build_pdp_url_none_store() -> None:
    assert build_pdp_url(None, {"pdp_handle": "anything"}) is None


def test_build_pdp_url_unknown_store() -> None:
    assert build_pdp_url("unknownbrand", {"pdp_handle": "some-handle"}) is None


def test_build_pdp_url_empty_handle() -> None:
    assert build_pdp_url("snitch", {"pdp_handle": ""}) is None
    assert build_pdp_url("snitch", {"pdp_handle": "   "}) is None


# ---------------------------------------------------------------------------
# Ensure every known store in STORE_CONFIG has a display_name
# ---------------------------------------------------------------------------


def test_all_store_configs_have_display_name() -> None:
    for slug, cfg in STORE_CONFIG.items():
        assert "display_name" in cfg and cfg["display_name"], (
            f"Store '{slug}' is missing a display_name in STORE_CONFIG"
        )


# ---------------------------------------------------------------------------
# active flag + get_inactive_stores
# ---------------------------------------------------------------------------


def test_all_store_configs_have_active_flag() -> None:
    """Every STORE_CONFIG entry must declare an explicit boolean 'active' flag."""
    for slug, cfg in STORE_CONFIG.items():
        assert "active" in cfg, f"Store '{slug}' is missing the 'active' flag in STORE_CONFIG"
        assert isinstance(cfg["active"], bool), f"Store '{slug}' active flag must be a bool"


def test_berrylush_and_hm_are_inactive() -> None:
    """berrylush (password-walled since 2026-07) and hm (dormant) must be inactive."""
    assert STORE_CONFIG["berrylush"]["active"] is False
    assert STORE_CONFIG["hm"]["active"] is False


def test_other_stores_are_active() -> None:
    """All stores besides berrylush/hm must remain active."""
    for slug, cfg in STORE_CONFIG.items():
        if slug in ("berrylush", "hm"):
            continue
        assert cfg["active"] is True, f"Store '{slug}' unexpectedly inactive"


def test_get_inactive_stores_returns_frozenset() -> None:
    inactive = get_inactive_stores()
    assert isinstance(inactive, frozenset)
    assert inactive == frozenset({"berrylush", "hm"})


def test_get_inactive_stores_matches_active_flags() -> None:
    """get_inactive_stores must be derived purely from STORE_CONFIG['active'] — no drift."""
    expected = {slug for slug, cfg in STORE_CONFIG.items() if cfg.get("active") is False}
    assert get_inactive_stores() == frozenset(expected)
