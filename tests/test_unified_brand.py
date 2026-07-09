"""Tests for the unified (cross-store B2C) brand config — StyleMitra.

Guards against api/main.py's unified-mode detection (BRAND unset ==
unified mode) silently diverging from src.config.brand.get_brand_config's
own default, which previously fell back to "hm" and served H&M branding
in production (BRAND is unset on the production deploy).

All tests clear the get_brand_config lru_cache before/after so env changes
made via monkeypatch are actually observed, following the pattern used in
tests/test_affiliate_links.py for _get_affiliate_config.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config.brand import get_brand_config

_EXPECTED_CHIPS = [
    "Sangeet look under ₹8000",
    "Haldi outfit — bright & daytime",
    "Wedding-guest saree under ₹5000",
    "Style my partner for a reception",
    "Mehendi look in green",
]


@pytest.fixture(autouse=True)
def clear_brand_cache() -> None:  # type: ignore[return]
    """Clear the lru_cache before every test so BRAND env changes are visible."""
    get_brand_config.cache_clear()
    yield
    get_brand_config.cache_clear()


def test_brand_unset_defaults_to_unified_stylemitra(monkeypatch: pytest.MonkeyPatch) -> None:
    """With BRAND unset, get_brand_config() must load StyleMitra (unified.yaml),
    not silently fall back to H&M — this is the production deploy's actual env.
    """
    monkeypatch.delenv("BRAND", raising=False)

    cfg = get_brand_config()

    assert cfg.display_name == "StyleMitra"
    assert cfg.suggestion_chips == _EXPECTED_CHIPS
    assert cfg.gender_default == "women"
    assert cfg.currency == "INR"
    assert cfg.locale == "en-IN"
    assert cfg.sizing_system == "IN"
    assert cfg.pdp_url_template == ""


def test_brand_fashor_still_loads_fashor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit BRAND=fashor must still load the Fashor config unaffected by the
    new unified default.
    """
    monkeypatch.setenv("BRAND", "fashor")

    cfg = get_brand_config()

    assert cfg.display_name == "Fashor"
    assert cfg.gender_default == "women"


def test_api_brand_endpoint_serves_stylemitra_when_brand_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /api/brand with BRAND unset must serve StyleMitra's display name and
    suggestion chips — the end-to-end symptom of the production bug this task fixes.
    """
    monkeypatch.delenv("BRAND", raising=False)
    client = TestClient(app, raise_server_exceptions=True)

    response = client.get("/api/brand")

    assert response.status_code == 200
    body = response.json()
    assert body["display_name"] == "StyleMitra"
    assert body["suggestion_chips"] == _EXPECTED_CHIPS
