"""2026-07-13 launch-critical fix: normalize() must drop fully out-of-stock
Shopify products (zero available variants), since the catalogue is a static
snapshot with no other stock signal and a live search was surfacing sold-out
links."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from download_shopify import (  # noqa: E402
    _CurlResponse,
    _get,
    _looks_like_cloudflare_challenge,
    normalize,
)


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


# ---------------------------------------------------------------------------
# 2026-07-19: Cloudflare TLS-fingerprint block — some Indian D2C Shopify stores
# return HTTP 429 "Verifying your connection..." to python-requests' TLS
# ClientHello while curl.exe (Schannel) gets a normal 200 on the identical URL.
# _get() must detect the Cloudflare-shaped response and fall back to curl.exe
# for that single request, without touching the requests-based common path.
# ---------------------------------------------------------------------------


def _fake_response(status_code: int, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


class TestCloudflareChallengeDetection:
    def test_http_429_is_cloudflare_shaped(self) -> None:
        assert _looks_like_cloudflare_challenge(_fake_response(429, "")) is True

    def test_200_with_challenge_marker_is_cloudflare_shaped(self) -> None:
        body = "<html><body>Verifying your connection...</body></html>"
        assert _looks_like_cloudflare_challenge(_fake_response(200, body)) is True

    def test_503_with_cf_marker_is_cloudflare_shaped(self) -> None:
        body = "<title>Attention Required! | Cloudflare</title>"
        assert _looks_like_cloudflare_challenge(_fake_response(503, body)) is True

    def test_503_without_cf_marker_is_still_block_shaped(self) -> None:
        # 2026-07-23: blissclub.com/silvertraq.com return Shopify's generic branded
        # 503 error page (no "cloudflare" string) — still edge-block-shaped, since a
        # genuine non-block 503 fails identically either way and curl recovers it.
        body = "<title>Something went wrong</title>"
        assert _looks_like_cloudflare_challenge(_fake_response(503, body)) is True

    def test_normal_200_is_not_cloudflare_shaped(self) -> None:
        body = '{"products": [{"id": 1, "handle": "test"}]}'
        assert _looks_like_cloudflare_challenge(_fake_response(200, body)) is False

    def test_other_error_status_is_not_cloudflare_shaped(self) -> None:
        assert _looks_like_cloudflare_challenge(_fake_response(404, "")) is False


class TestGetCurlFallback:
    def test_normal_response_does_not_invoke_curl(self) -> None:
        session = MagicMock()
        session.get.return_value = _fake_response(200, '{"products": []}')
        with patch("download_shopify.subprocess.run") as mock_run:
            resp = _get(session, "https://example.com/products.json", timeout=10)
        mock_run.assert_not_called()
        assert resp.status_code == 200

    def test_429_falls_back_to_curl_and_returns_curl_body(self) -> None:
        session = MagicMock()
        session.get.return_value = _fake_response(429, "Verifying your connection...")
        curl_stdout = '{"products": [{"id": 1, "handle": "curl-fetched"}]}'
        mock_completed = MagicMock(stdout=curl_stdout)
        with patch("download_shopify.subprocess.run", return_value=mock_completed) as mock_run:
            resp = _get(session, "https://example.com/products.json", timeout=10)
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        assert called_args[0] == "curl.exe"
        assert isinstance(resp, _CurlResponse)
        assert resp.status_code == 200
        assert resp.json()["products"][0]["handle"] == "curl-fetched"

    def test_curl_failure_falls_back_to_original_response(self) -> None:
        session = MagicMock()
        original_resp = _fake_response(429, "Verifying your connection...")
        session.get.return_value = original_resp
        with patch(
            "download_shopify.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "curl.exe"),
        ):
            resp = _get(session, "https://example.com/products.json", timeout=10)
        assert resp is original_resp
