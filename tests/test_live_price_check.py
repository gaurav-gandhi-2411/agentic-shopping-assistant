"""Tests for the render-time live price check (src/pricing/live_check.py).

Hermetic: every HTTP call goes through ``httpx.MockTransport``; no real store is contacted.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from api.schemas import ItemSummary
from src.pricing import live_check, metrics
from src.pricing.guards import USER_AGENT, LiveConfig, product_url
from src.pricing.live_check import LivePriceChecker


def _item(handle: str | None = "kurta-1", store: str = "libas", price: float = 1000.0, **kw: Any):
    return ItemSummary(
        article_id=kw.pop("article_id", f"{store}-{handle}"),
        prod_name="Kurta",
        display_name="Kurta",
        colour="Red",
        product_type="Kurta",
        department="Women",
        price_inr=price,
        pdp_handle=handle,
        store=store,
        pdp_url=f"https://{store}.example/products/{handle}",
        **kw,
    )


def _js(price_paise: int = 100000, available: list[bool] | None = None) -> bytes:
    variants = [{"price": price_paise, "available": a} for a in (available or [True])]
    return json.dumps({"price": price_paise, "variants": variants}).encode()


class Store:
    """Mock store: records every request; product responses come from ``products``."""

    def __init__(self, products: dict[str, Any] | None = None, delay: float = 0.0) -> None:
        self.requests: list[httpx.Request] = []
        self.products = products or {}
        self.delay = delay
        self.robots = "User-agent: *\nAllow: /\n"

    def product_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path != "/robots.txt"]

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=self.robots)
        handle = request.url.path.removeprefix("/products/").removesuffix(".js")
        spec = self.products.get(handle, httpx.Response(200, content=_js()))
        return spec(request) if callable(spec) else spec


def _checker(store: Store, **cfg: Any) -> LivePriceChecker:
    base = {"enabled": True, "budget_s": 1.0, "min_interval_s": 0.001}
    return LivePriceChecker(LiveConfig(**{**base, **cfg}), transport=httpx.MockTransport(store))


def _check(checker: LivePriceChecker, items: list[ItemSummary]) -> None:
    asyncio.run(checker.check(items))


@pytest.fixture(autouse=True)
def _fresh_metrics() -> None:
    metrics.reset()
    live_check.reset_checker()


# ---------------------------------------------------------------- URL / SSRF guard


@pytest.mark.parametrize(
    "handle",
    [
        "../../etc/passwd", "a/b", "a?x=1", "a#frag", "evil.com/x", "%2e%2e/x", "a b",
        "https://evil.com/p", "//evil.com/p", "a@evil.com", "a\\b", "", None, "..", "a..b",
        "x" * 300, "café", "a\nb",
    ],
)
def test_hostile_handle_builds_no_url_and_sends_no_request(handle: str | None) -> None:
    assert product_url("libas", handle) is None
    store = Store()
    item = _item(handle=handle)
    _check(_checker(store), [item])
    assert store.requests == []
    assert item.price_may_have_changed is False  # never eligible, so not flagged either


@pytest.mark.parametrize("store", ["myntra", "flipkart", "bhasinbrothers", "priyaasi", "nope", ""])
def test_non_public_or_unknown_store_is_never_fetched(store: str) -> None:
    assert product_url(store, "kurta-1") is None
    mock = Store()
    item = _item(store=store)
    _check(_checker(mock), [item])
    assert mock.requests == []
    assert item.unavailable is False and item.price_inr == 1000.0


def test_all_requests_stay_on_store_domain_https() -> None:
    mock = Store()
    _check(_checker(mock), [_item("a-1"), _item("b-2", store="snitch")])
    hosts = {(r.url.scheme, r.url.host) for r in mock.requests}
    assert hosts == {("https", "libas.in"), ("https", "snitch.co.in")}
    assert all(r.headers["User-Agent"] == USER_AGENT for r in mock.requests)


def test_redirect_is_not_followed() -> None:
    mock = Store({"moved": httpx.Response(301, headers={"Location": "https://evil.com/x"})})
    item = _item("moved")
    _check(_checker(mock), [item])
    assert {r.url.host for r in mock.requests} == {"libas.in"}
    assert item.price_may_have_changed is True and item.unavailable is False


# ---------------------------------------------------------------- results


def test_price_change_updates_price_and_keeps_indexed() -> None:
    mock = Store({"kurta-1": httpx.Response(200, content=_js(129900))})
    item = _item("kurta-1", price=1000.0)
    _check(_checker(mock), [item])
    assert item.price_inr == 1299.0
    assert item.indexed_price_inr == 1000.0
    assert item.price_may_have_changed is False and item.unavailable is False
    assert metrics.counter_value(metrics.PRICE_CHANGED, store="libas") == 1


def test_same_price_leaves_card_untouched() -> None:
    mock = Store()
    item = _item(price=1000.0)
    _check(_checker(mock), [item])
    assert item.price_inr == 1000.0 and item.indexed_price_inr is None
    assert item.price_may_have_changed is False and item.unavailable is False


def test_all_variants_sold_out_marks_unavailable_and_drops_buy_url() -> None:
    mock = Store({"gone": httpx.Response(200, content=_js(available=[False, False]))})
    item = _item("gone")
    _check(_checker(mock), [item])
    assert item.unavailable is True and item.pdp_url is None
    assert metrics.counter_value(metrics.UNAVAILABLE, store="libas") == 1


def test_one_variant_in_stock_stays_available() -> None:
    mock = Store({"some": httpx.Response(200, content=_js(available=[False, True]))})
    item = _item("some")
    _check(_checker(mock), [item])
    assert item.unavailable is False and item.pdp_url is not None


@pytest.mark.parametrize("code", [404, 410])
def test_product_gone_marks_unavailable(code: int) -> None:
    mock = Store({"gone": httpx.Response(code)})
    item = _item("gone")
    _check(_checker(mock), [item])
    assert item.unavailable is True and item.pdp_url is None


def test_malformed_json_keeps_indexed_price_and_flags() -> None:
    mock = Store({"bad": httpx.Response(200, content=b"<html>not json")})
    item = _item("bad")
    _check(_checker(mock), [item])
    assert item.price_inr == 1000.0 and item.pdp_url is not None
    assert item.price_may_have_changed is True and item.unavailable is False


@pytest.mark.parametrize("body", [b"[]", b'{"variants": "x"}', b'{"variants":[{"price":"abc"}]}'])
def test_odd_json_shapes_fail_open(body: bytes) -> None:
    mock = Store({"odd": httpx.Response(200, content=body)})
    item = _item("odd")
    _check(_checker(mock), [item])
    assert item.price_may_have_changed is True and item.unavailable is False


def test_missing_stock_fields_are_treated_as_available() -> None:
    body = json.dumps({"variants": [{"price": 100000}]}).encode()
    item = _item("nostock")
    _check(_checker(Store({"nostock": httpx.Response(200, content=body)})), [item])
    assert item.unavailable is False and item.price_may_have_changed is False


def test_owned_and_handleless_items_are_skipped() -> None:
    mock = Store()
    _check(_checker(mock), [_item(is_owned=True), _item(handle=None)])
    assert mock.requests == []


def test_only_first_max_cards_are_checked() -> None:
    mock = Store()
    items = [_item(f"h{i}") for i in range(10)]
    _check(_checker(mock, max_cards=8, burst=100), items)
    assert len(mock.product_requests()) == 8


# ---------------------------------------------------------------- budget


def test_budget_is_enforced_with_slow_store_and_card_is_flagged() -> None:
    mock = Store(delay=5.0)
    items = [_item(f"h{i}", store=s) for i, s in enumerate(["libas", "snitch", "voylla"])]
    t0 = time.monotonic()
    _check(_checker(mock, budget_s=0.3), items)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.3 + 0.25, elapsed
    assert all(i.price_may_have_changed and i.price_inr == 1000.0 for i in items)
    assert metrics.counter_value(metrics.CHECKS, outcome="timeout", store="libas") == 1


def test_partial_results_inside_budget_are_kept() -> None:
    mock =Store({"fast": httpx.Response(200, content=_js(150000))})
    slow = Store(delay=5.0)

    async def route(request: httpx.Request) -> httpx.Response:
        return await (mock if request.url.host == "libas.in" else slow)(request)

    checker = LivePriceChecker(
        LiveConfig(enabled=True, budget_s=0.4, min_interval_s=0.001),
        transport=httpx.MockTransport(route),
    )
    fast_item, slow_item = _item("fast"), _item("slow", store="snitch")
    _check(checker, [fast_item, slow_item])
    assert fast_item.price_inr == 1500.0 and fast_item.price_may_have_changed is False
    assert slow_item.price_may_have_changed is True


# ---------------------------------------------------------------- cache


def test_cache_hit_makes_no_second_request() -> None:
    mock = Store()
    checker = _checker(mock)
    _check(checker, [_item("kurta-1")])
    n = len(mock.requests)
    item = _item("kurta-1")
    _check(checker, [item])
    assert len(mock.requests) == n
    assert metrics.counter_value(metrics.CHECKS, outcome="cache_hit", store="libas") == 1


def test_cache_entry_expires_after_ttl() -> None:
    now = [0.0]
    mock = Store()
    checker = LivePriceChecker(
        LiveConfig(enabled=True, budget_s=1.0, min_interval_s=0.001, cache_ttl_s=3600),
        transport=httpx.MockTransport(mock),
        clock=lambda: now[0],
    )
    _check(checker, [_item("kurta-1")])
    n = len(mock.product_requests())
    now[0] = 3599.0
    _check(checker, [_item("kurta-1")])
    assert len(mock.product_requests()) == n
    now[0] = 3601.0
    _check(checker, [_item("kurta-1")])
    assert len(mock.product_requests()) == n + 1


def test_duplicate_cards_in_one_batch_share_one_request() -> None:
    mock = Store()
    a, b = _item("same", article_id="a"), _item("same", article_id="b")
    _check(_checker(mock), [a, b])
    assert len(mock.product_requests()) == 1


# ---------------------------------------------------------------- politeness


def test_429_is_not_retried_and_backs_off_the_store() -> None:
    mock = Store({"hot": httpx.Response(429, headers={"Retry-After": "120"})})
    checker = _checker(mock)
    item = _item("hot")
    _check(checker, [item])
    assert len(mock.product_requests()) == 1
    assert item.price_may_have_changed is True
    other = _item("other")
    _check(checker, [other])  # store is backed off: no new request at all
    assert len(mock.product_requests()) == 1
    assert other.price_may_have_changed is True


def test_circuit_breaker_opens_after_repeated_failures() -> None:
    mock = Store({f"h{i}": httpx.Response(500) for i in range(6)})
    checker = _checker(mock, breaker_failures=3)
    for i in range(3):
        _check(checker, [_item(f"h{i}")])
    assert len(mock.product_requests()) == 3
    item = _item("h3")
    _check(checker, [item])
    assert len(mock.product_requests()) == 3  # breaker open: skipped
    assert item.price_may_have_changed is True
    assert metrics.counter_value(metrics.CHECKS, outcome="circuit_open", store="libas") == 1


def test_circuit_breaker_is_per_store() -> None:
    mock = Store({f"h{i}": httpx.Response(500) for i in range(3)})
    checker = _checker(mock, breaker_failures=3)
    for i in range(3):
        _check(checker, [_item(f"h{i}")])
    ok = _item("fine", store="snitch")
    _check(checker, [ok])
    assert ok.price_may_have_changed is False


def test_success_resets_the_failure_count() -> None:
    mock = Store({"h0": httpx.Response(500), "h1": httpx.Response(500)})
    checker = _checker(mock, breaker_failures=3)
    _check(checker, [_item("h0")])
    _check(checker, [_item("h1")])
    _check(checker, [_item("ok")])  # success: resets
    _check(checker, [_item("h2")])
    n = len(mock.product_requests())
    _check(checker, [_item("h3")])
    assert len(mock.product_requests()) == n + 1  # still closed


def test_per_domain_rate_limit_skips_instead_of_waiting_past_budget() -> None:
    mock = Store()
    items = [_item(f"h{i}") for i in range(4)]
    t0 = time.monotonic()
    _check(_checker(mock, budget_s=0.4, min_interval_s=1.0, burst=2), items)
    assert time.monotonic() - t0 < 0.4 + 0.25
    # burst 2 = robots.txt + one product page; nothing further is sent inside the budget.
    assert len(mock.requests) == 2
    assert sum(i.price_may_have_changed for i in items) == 3


def test_robots_disallow_means_no_product_request() -> None:
    mock = Store()
    mock.robots = "User-agent: *\nDisallow: /products/\n"
    item = _item()
    _check(_checker(mock), [item])
    assert mock.product_requests() == []
    assert item.price_may_have_changed is True


def test_robots_is_fetched_once_per_domain() -> None:
    mock = Store()
    _check(_checker(mock, burst=100), [_item("a"), _item("b"), _item("c")])
    assert sum(r.url.path == "/robots.txt" for r in mock.requests) == 1


def test_network_error_fails_open() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    item = _item()
    checker = LivePriceChecker(
        LiveConfig(enabled=True, min_interval_s=0.001), transport=httpx.MockTransport(boom)
    )
    _check(checker, [item])
    assert item.price_may_have_changed is True and item.price_inr == 1000.0


# ---------------------------------------------------------------- flag / entry points


def test_entry_points_fail_open_when_checker_explodes(monkeypatch) -> None:
    def boom() -> None:
        raise RuntimeError("broken")

    monkeypatch.setattr(live_check, "_enabled_checker", boom)
    items = [_item()]
    assert live_check.apply_live_check_sync(items) is items
    assert asyncio.run(live_check.apply_live_check(items)) is items


def test_async_entry_point_sinks_unavailable_without_changing_count(monkeypatch) -> None:
    mock = Store({"gone": httpx.Response(404)})
    checker = _checker(mock)
    monkeypatch.setattr(live_check, "_enabled_checker", lambda: checker)
    items = [_item("gone"), _item("ok1"), _item("ok2")]
    out = asyncio.run(live_check.apply_live_check(items))
    assert [i.pdp_handle for i in out] == ["ok1", "ok2", "gone"]
    assert len(out) == 3


def test_sync_entry_point_works_from_a_worker_thread(monkeypatch) -> None:
    import concurrent.futures

    mock = Store({"gone": httpx.Response(404)})
    checker = _checker(mock)
    monkeypatch.setattr(live_check, "_enabled_checker", lambda: checker)
    items = [_item("gone"), _item("ok1")]
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        out = ex.submit(live_check.apply_live_check_sync, items).result()
    assert [i.pdp_handle for i in out] == ["ok1", "gone"]


