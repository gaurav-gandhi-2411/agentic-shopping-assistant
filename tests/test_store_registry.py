"""Store registry (src/config/store_registry.py): single source of truth for public/refresh flags.

Guards: the registry and STORE_CONFIG agree, every store of the live index is registered, exactly
the 38 refreshable Shopify stores are public, the opt-out switch is honoured, feed etiquette
helpers behave.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from src.config import store_registry as reg
from src.config.stores import STORE_CONFIG

_FIXTURE = Path(__file__).parent / "fixtures" / "indexed_stores.json"
_INDEXED = json.loads(_FIXTURE.read_text(encoding="utf-8"))["stores"]

# V3a inventory (reports/wave1b/v3a-catalogue-inventory.md section 1): not refreshable.
_NON_PUBLIC_INDEXED = {"myntra", "flipkart", "bhasinbrothers", "priyaasi"}


def test_registry_covers_every_store_config_entry_and_vice_versa() -> None:
    assert set(reg.STORE_REGISTRY) == set(STORE_CONFIG)


def test_every_indexed_store_is_in_the_registry() -> None:
    missing = [s["slug"] for s in _INDEXED if s["slug"] not in reg.STORE_REGISTRY]
    assert missing == []


def test_registry_agrees_with_store_config() -> None:
    for slug, rec in reg.STORE_REGISTRY.items():
        cfg = STORE_CONFIG[slug]
        assert rec.display_name == cfg["display_name"], slug
        template = cfg["pdp_url_template"]
        if rec.source_type == "shopify":
            assert template == f"https://{rec.domain}/products/{{handle}}", slug
        elif template is not None:  # myntra: relative path on its own domain
            assert template.startswith(f"https://{rec.domain}/"), slug


def test_source_type_matches_the_index_fixture() -> None:
    for s in _INDEXED:
        rec = reg.STORE_REGISTRY[s["slug"]]
        expected = "static_dump" if s["source"] == "static_dump" else "shopify"
        assert rec.source_type == expected, s["slug"]


def test_exactly_the_38_refreshable_shopify_stores_are_public() -> None:
    public = reg.public_stores()
    assert len(public) == 38
    indexed = {s["slug"] for s in _INDEXED}
    assert indexed - public == _NON_PUBLIC_INDEXED
    assert public <= indexed
    assert reg.refreshable_stores() == public
    for slug in public:
        assert reg.STORE_REGISTRY[slug].source_type == "shopify"


def test_non_public_stores_are_never_refreshed() -> None:
    for slug, rec in reg.STORE_REGISTRY.items():
        if not rec.public:
            assert rec.refresh_enabled is False, slug
            assert slug not in reg.refreshable_stores()


def test_refresh_opt_out_switch_removes_a_store_from_the_refresh_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opted_out = dataclasses.replace(reg.STORE_REGISTRY["libas"], refresh_enabled=False)
    monkeypatch.setitem(reg.STORE_REGISTRY, "libas", opted_out)
    assert "libas" not in reg.refreshable_stores()
    assert "libas" in reg.public_stores()  # opt-out stops fetching; it does not hide the store
    assert len(reg.refreshable_stores()) == 37


def test_feed_url_pattern_and_etiquette_constants() -> None:
    rec = reg.STORE_REGISTRY["snitch"]
    assert rec.feed_url_pattern == "https://snitch.co.in/products.json?limit=250&page={page}"
    assert rec.feed_url(3) == "https://snitch.co.in/products.json?limit=250&page=3"
    assert reg.STORE_REGISTRY["myntra"].feed_url(1) is None  # static dump: no feed
    assert reg.FEED_MAX_REQUESTS_PER_SECOND <= 1.0


def test_feed_user_agent_names_styleMaitri_and_needs_a_contact() -> None:
    ua = reg.feed_user_agent("ops@example.com")
    assert "StyleMaitri" in ua and "ops@example.com" in ua
    with pytest.raises(ValueError):
        reg.feed_user_agent("   ")


def test_is_public_fails_closed_for_unknown_and_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(reg.INCLUDE_NON_PUBLIC_ENV, raising=False)
    assert reg.is_public("snitch") is True
    assert reg.is_public("SNITCH") is True
    assert reg.is_public("myntra") is False
    assert reg.is_public("not-a-registered-store") is False
    assert reg.is_public(None) is False
    assert reg.get_non_public_stores() >= _NON_PUBLIC_INDEXED


def test_include_non_public_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(reg.INCLUDE_NON_PUBLIC_ENV, "1")
    assert reg.is_public("myntra") is True
    assert reg.is_public("not-a-registered-store") is True
    assert reg.get_non_public_stores() == frozenset()
    monkeypatch.setenv(reg.INCLUDE_NON_PUBLIC_ENV, "0")
    assert reg.is_public("myntra") is False
