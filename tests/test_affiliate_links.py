"""Tests for Phase F affiliate-tag logic in src/config/stores.py.

All tests use monkeypatch to inject ASA_AFFILIATE_* env vars and call
_get_affiliate_config.cache_clear() before each case so that the lru_cache
never bleeds state between tests.

The no-env path asserts byte-identical output to pre-Phase-F plain links,
confirming the default is unchanged.
"""
from __future__ import annotations

import urllib.parse

import pytest

from src.config.stores import _get_affiliate_config, build_pdp_url  # noqa: PLC2701

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_affiliate_cache() -> None:  # type: ignore[return]
    """Clear the lru_cache before every test so env changes are visible."""
    _get_affiliate_config.cache_clear()
    yield
    _get_affiliate_config.cache_clear()


# ---------------------------------------------------------------------------
# Default (no env) — plain links must be byte-identical to pre-Phase-F output
# ---------------------------------------------------------------------------


def test_default_no_env_myntra_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ASA_AFFILIATE_MYNTRA → plain link, identical to pre-Phase-F output."""
    monkeypatch.delenv("ASA_AFFILIATE_MYNTRA", raising=False)
    row = {"pdp_handle": "kurtas/brand/some-kurta/17048614/buy"}
    url = build_pdp_url("myntra", row)
    assert url == "https://www.myntra.com/kurtas/brand/some-kurta/17048614/buy"


def test_default_no_env_flipkart_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ASA_AFFILIATE_FLIPKART → plain URL returned verbatim."""
    monkeypatch.delenv("ASA_AFFILIATE_FLIPKART", raising=False)
    full_url = "https://www.flipkart.com/some-product/p/itm123?pid=ABC&marketplace=FLIPKART"
    url = build_pdp_url("flipkart", {"pdp_handle": full_url})
    assert url == full_url


def test_default_no_env_snitch_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASA_AFFILIATE_SNITCH", raising=False)
    url = build_pdp_url("snitch", {"pdp_handle": "my-product-handle-xyz"})
    assert url == "https://snitch.co.in/products/my-product-handle-xyz"


def test_default_no_env_fashor_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASA_AFFILIATE_FASHOR", raising=False)
    url = build_pdp_url("fashor", {"pdp_handle": "cool-dress-v2"})
    assert url == "https://fashor.com/products/cool-dress-v2"


def test_default_no_env_powerlook_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASA_AFFILIATE_POWERLOOK", raising=False)
    url = build_pdp_url("powerlook", {"pdp_handle": "slim-fit-trousers"})
    assert url == "https://powerlook.in/products/slim-fit-trousers"


def test_default_no_env_virgio_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASA_AFFILIATE_VIRGIO", raising=False)
    url = build_pdp_url("virgio", {"pdp_handle": "floral-midi-skirt"})
    assert url == "https://virgio.com/products/floral-midi-skirt"


# ---------------------------------------------------------------------------
# param mode — no-query URL (Snitch / Shopify stores)
# ---------------------------------------------------------------------------


def test_param_mode_no_query_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """param mode on a URL with no existing query string → ? separator."""
    monkeypatch.setenv("ASA_AFFILIATE_SNITCH", "param:affid=asa001")
    url = build_pdp_url("snitch", {"pdp_handle": "my-product-handle-xyz"})
    assert url == "https://snitch.co.in/products/my-product-handle-xyz?affid=asa001"


def test_param_mode_multi_param_no_query_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """param value may itself contain & — all params are appended after ?."""
    monkeypatch.setenv("ASA_AFFILIATE_FASHOR", "param:utm_source=asa&utm_medium=app")
    url = build_pdp_url("fashor", {"pdp_handle": "cool-dress-v2"})
    assert url == "https://fashor.com/products/cool-dress-v2?utm_source=asa&utm_medium=app"


def test_param_mode_myntra_no_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Myntra plain URLs have no query string → ? separator."""
    monkeypatch.setenv("ASA_AFFILIATE_MYNTRA", "param:af_channel=asa_stylist")
    row = {"pdp_handle": "kurtas/brand/some-kurta/17048614/buy"}
    url = build_pdp_url("myntra", row)
    assert url == "https://www.myntra.com/kurtas/brand/some-kurta/17048614/buy?af_channel=asa_stylist"


# ---------------------------------------------------------------------------
# param mode — Flipkart URL already has query params → & separator
# ---------------------------------------------------------------------------


def test_param_mode_flipkart_existing_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flipkart URLs already contain ?pid=… → affiliate param appended with &, not a second ?."""
    monkeypatch.setenv("ASA_AFFILIATE_FLIPKART", "param:affid=asa001")
    full_url = "https://www.flipkart.com/some-product/p/itm123?pid=ABC&marketplace=FLIPKART"
    url = build_pdp_url("flipkart", {"pdp_handle": full_url})
    assert url == f"{full_url}&affid=asa001"
    # Critically: only ONE ? in the result
    assert url.count("?") == 1


# ---------------------------------------------------------------------------
# wrap mode
# ---------------------------------------------------------------------------


def test_wrap_mode_encodes_plain_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """wrap mode URL-encodes the plain PDP URL and inserts it into {url}."""
    template = "https://linksredirect.example.com/?pub=FAKE123&source=asa&url={url}"
    monkeypatch.setenv("ASA_AFFILIATE_SNITCH", f"wrap:{template}")
    plain = "https://snitch.co.in/products/my-product-handle-xyz"
    url = build_pdp_url("snitch", {"pdp_handle": "my-product-handle-xyz"})
    encoded_plain = urllib.parse.quote(plain, safe="")
    assert url == f"https://linksredirect.example.com/?pub=FAKE123&source=asa&url={encoded_plain}"


def test_wrap_mode_url_encodes_special_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    """URL-encoding turns ? and & in the plain URL into %3F and %26."""
    template = "https://redir.example.com/?source=asa&url={url}"
    monkeypatch.setenv("ASA_AFFILIATE_FLIPKART", f"wrap:{template}")
    plain = "https://www.flipkart.com/some-product/p/itm123?pid=ABC&marketplace=FLIPKART"
    url = build_pdp_url("flipkart", {"pdp_handle": plain})
    assert "%3F" in url  # ? encoded
    assert url.startswith("https://redir.example.com/")


# ---------------------------------------------------------------------------
# Fallback / invalid config → plain link (no crash)
# ---------------------------------------------------------------------------


def test_invalid_mode_falls_back_to_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrecognised mode → plain link, no crash."""
    monkeypatch.setenv("ASA_AFFILIATE_SNITCH", "redirect:affid=asa001")
    url = build_pdp_url("snitch", {"pdp_handle": "my-product-handle-xyz"})
    assert url == "https://snitch.co.in/products/my-product-handle-xyz"


def test_wrap_missing_url_placeholder_falls_back_to_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    """wrap template without {url} → plain link, no crash."""
    monkeypatch.setenv("ASA_AFFILIATE_SNITCH", "wrap:https://redir.example.com/?pub=FAKE123")
    url = build_pdp_url("snitch", {"pdp_handle": "my-product-handle-xyz"})
    assert url == "https://snitch.co.in/products/my-product-handle-xyz"


def test_no_colon_in_env_var_falls_back_to_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var with no colon separator → plain link, no crash."""
    monkeypatch.setenv("ASA_AFFILIATE_SNITCH", "affid=asa001")
    url = build_pdp_url("snitch", {"pdp_handle": "my-product-handle-xyz"})
    assert url == "https://snitch.co.in/products/my-product-handle-xyz"


def test_empty_env_var_falls_back_to_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty env var → plain link."""
    monkeypatch.setenv("ASA_AFFILIATE_SNITCH", "")
    url = build_pdp_url("snitch", {"pdp_handle": "my-product-handle-xyz"})
    assert url == "https://snitch.co.in/products/my-product-handle-xyz"


# ---------------------------------------------------------------------------
# H&M / None store — always None regardless of any env var
# ---------------------------------------------------------------------------


def test_hm_none_regardless_of_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """H&M has no PDP template — always returns None, even with an affiliate env var set."""
    monkeypatch.setenv("ASA_AFFILIATE_HM", "param:affid=asa001")
    assert build_pdp_url("hm", {"pdp_handle": "some-handle"}) is None


def test_none_store_always_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert build_pdp_url(None, {"pdp_handle": "some-handle"}) is None


def test_none_handle_with_affiliate_env_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No affiliate tag applied when pdp_handle is absent."""
    monkeypatch.setenv("ASA_AFFILIATE_SNITCH", "param:affid=asa001")
    assert build_pdp_url("snitch", {"pdp_handle": None}) is None
