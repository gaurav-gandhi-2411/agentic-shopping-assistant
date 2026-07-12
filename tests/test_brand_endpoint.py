"""Tests for the /api/brand endpoint.

Patches get_brand_config() with a synthetic BrandConfig so tests run
without a brands/ YAML file on disk and without touching the LLM or
retrieval stack.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config.brand import BrandConfig


@pytest.fixture
def brand_client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


_MOCK_BRAND = BrandConfig(
    display_name="TestBrand",
    logo_url="https://example.com/logo.png",
    primary_colour="#FF0000",
    accent_colour="#00FF00",
    tagline="Test tagline",
    suggestion_chips=["Find me a dress", "Show basics"],
    currency="USD",
    locale="en-US",
    sizing_system="alpha",
    catalogue_path="data/processed/catalogue.parquet",
    pdp_url_template="https://example.com/products/{handle}",
)


def test_get_brand_returns_200(brand_client: TestClient) -> None:
    """GET /api/brand should return 200 with all required fields."""
    with patch("api.routes.brand.get_brand_config", return_value=_MOCK_BRAND):
        response = brand_client.get("/api/brand")

    assert response.status_code == 200
    body = response.json()
    assert body["display_name"] == "TestBrand"
    assert body["primary_colour"] == "#FF0000"
    assert body["accent_colour"] == "#00FF00"
    assert body["logo_url"] == "https://example.com/logo.png"
    assert body["tagline"] == "Test tagline"
    assert body["currency"] == "USD"
    assert body["locale"] == "en-US"
    assert body["sizing_system"] == "alpha"
    assert body["suggestion_chips"] == ["Find me a dress", "Show basics"]
    assert body["pdp_url_template"] == "https://example.com/products/{handle}"


def test_get_brand_null_logo_and_tagline(brand_client: TestClient) -> None:
    """GET /api/brand should handle null logo_url and tagline gracefully."""
    mock = _MOCK_BRAND.model_copy(update={"logo_url": None, "tagline": None})
    with patch("api.routes.brand.get_brand_config", return_value=mock):
        response = brand_client.get("/api/brand")

    assert response.status_code == 200
    body = response.json()
    assert body["logo_url"] is None
    assert body["tagline"] is None


def test_get_brand_no_auth_required(brand_client: TestClient) -> None:
    """/api/brand must be accessible without an Authorization header."""
    with patch("api.routes.brand.get_brand_config", return_value=_MOCK_BRAND):
        # Explicitly send no auth headers.
        response = brand_client.get("/api/brand", headers={})

    # 200 means the route is public (not gated by JWT middleware).
    assert response.status_code == 200
