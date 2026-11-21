"""Pure helpers of the live price check: URL/SSRF guard, product-JSON parsing, config, metrics."""
from __future__ import annotations

import json

import pytest

from src.pricing import metrics
from src.pricing.guards import ENV_FLAG, LiveConfig, parse_product_js, product_url


@pytest.fixture(autouse=True)
def _fresh_metrics() -> None:
    metrics.reset()


# ---------------------------------------------------------------- URL / SSRF guard


@pytest.mark.parametrize(
    "handle",
    [
        "../../etc/passwd", "a/b", "a?x=1", "a#frag", "evil.com/x", "%2e%2e/x", "a b",
        "https://evil.com/p", "//evil.com/p", "a@evil.com", "a\\b", "", None, "..", "a..b",
        "x" * 300, "café", "a\nb",
    ],
)
def test_hostile_handle_builds_no_url(handle: str | None) -> None:
    assert product_url("libas", handle) is None


def test_url_host_always_comes_from_registry() -> None:
    assert product_url("libas", "kurta-1") == "https://libas.in/products/kurta-1.js"
    assert product_url("LIBAS", "Kurta-1") == "https://libas.in/products/Kurta-1.js"
    assert product_url("voylla", "x") == "https://www.voylla.com/products/x.js"


@pytest.mark.parametrize("store", ["myntra", "flipkart", "bhasinbrothers", "priyaasi", "nope", ""])
def test_non_public_or_unknown_store_has_no_url(store: str) -> None:
    assert product_url(store, "kurta-1") is None


def test_non_public_has_no_url_even_with_include_non_public_env(monkeypatch) -> None:
    monkeypatch.setenv("INCLUDE_NON_PUBLIC_STORES", "1")
    assert product_url("bhasinbrothers", "kurta-1") is None
    assert product_url("myntra", "kurta-1") is None


# ---------------------------------------------------------------- parsing


def _js(*variants: dict, **extra) -> bytes:
    return json.dumps({"variants": list(variants), **extra}).encode()


def test_parse_reads_first_variant_price_in_rupees() -> None:
    res = parse_product_js(_js({"price": 129900, "available": True}, {"price": 99900}))
    assert res.price_inr == 1299.0 and res.available is True


def test_parse_all_variants_sold_out_is_unavailable() -> None:
    res = parse_product_js(_js({"price": 100, "available": False}, {"price": 100, "available": False}))
    assert res.available is False


def test_parse_one_variant_in_stock_is_available() -> None:
    res = parse_product_js(_js({"price": 100, "available": False}, {"price": 100, "available": True}))
    assert res.available is True


def test_parse_missing_stock_fields_fail_open() -> None:
    assert parse_product_js(_js({"price": 100})).available is True
    assert parse_product_js(b'{"available": false}').available is False


@pytest.mark.parametrize(
    "body", [b"<html>", b"[]", b'{"variants": "x"}', b'{"variants":[{"price":"abc"}]}', b""]
)
def test_parse_malformed_raises_value_error(body: bytes) -> None:
    with pytest.raises(ValueError):
        parse_product_js(body)


def test_parse_non_positive_price_is_unknown() -> None:
    assert parse_product_js(_js({"price": 0, "available": True})).price_inr is None


# ---------------------------------------------------------------- config


def test_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv(ENV_FLAG, raising=False)
    assert LiveConfig().enabled is False
    assert LiveConfig.from_config({}).enabled is False
    assert LiveConfig.from_config({"live_price_check": {"enabled": False}}).enabled is False


def test_env_overrides_config(monkeypatch) -> None:
    cfg = {"live_price_check": {"enabled": False, "budget_s": 0.7, "max_cards": 5}}
    monkeypatch.setenv(ENV_FLAG, "1")
    got = LiveConfig.from_config(cfg)
    assert got.enabled is True and got.budget_s == 0.7 and got.max_cards == 5
    monkeypatch.setenv(ENV_FLAG, "0")
    assert LiveConfig.from_config({"live_price_check": {"enabled": True}}).enabled is False


def test_spec_defaults() -> None:
    cfg = LiveConfig()
    assert (cfg.budget_s, cfg.cache_ttl_s, cfg.max_cards) == (1.5, 3600.0, 8)
    assert cfg.min_interval_s == 1.0


# ---------------------------------------------------------------- metrics


def test_prometheus_rendering_has_counters_and_histogram() -> None:
    metrics.inc(metrics.CHECKS, outcome="ok", store="libas")
    metrics.observe_duration(0.2)
    metrics.observe_duration(9.0)
    text = metrics.render_prometheus()
    assert 'asa_live_price_checks_total{outcome="ok",store="libas"} 1' in text
    assert 'asa_live_price_check_duration_seconds_bucket{le="0.25"} 1' in text
    assert 'asa_live_price_check_duration_seconds_bucket{le="+Inf"} 2' in text
    assert "asa_live_price_check_duration_seconds_count 2" in text


def test_counter_value_defaults_to_zero_and_reset_clears() -> None:
    assert metrics.counter_value(metrics.UNAVAILABLE, store="libas") == 0.0
    metrics.inc(metrics.UNAVAILABLE, store="libas")
    assert metrics.counter_value(metrics.UNAVAILABLE, store="libas") == 1.0
    metrics.reset()
    assert metrics.counter_value(metrics.UNAVAILABLE, store="libas") == 0.0
