"""scripts/ops/refresh_feed.py: feed etiquette. Hermetic: every HTTP call is a fixture, no network."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import refresh_feed as rf  # noqa: E402

from src.config.store_registry import STORE_REGISTRY, StoreRecord  # noqa: E402

NOW = datetime(2026, 11, 16, 2, 0, tzinfo=timezone.utc)
LIBAS = STORE_REGISTRY["libas"]
ROBOTS_OK = rf.HttpResponse(200, {}, "User-agent: *\nAllow: /\n")


def _page(n: int, start: int = 0) -> rf.HttpResponse:
    products = [{"id": start + i, "title": f"p{start + i}"} for i in range(n)]
    return rf.HttpResponse(200, {}, json.dumps({"products": products}))


class FakeNet:
    """Serves scripted responses per URL substring; records every URL and header set."""

    def __init__(self, routes: dict[str, list[rf.HttpResponse]]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> rf.HttpResponse:
        self.calls.append((url, headers))
        for key, queue in self.routes.items():
            if key in url:
                return queue.pop(0) if len(queue) > 1 else queue[0]
        raise AssertionError(f"unexpected URL {url}")


def _fetcher(net: FakeNet, sleeps: list[float] | None = None) -> rf.StoreFetcher:
    clock = [0.0]
    log = sleeps if sleeps is not None else []

    def sleep(s: float) -> None:
        log.append(s)
        clock[0] += s

    throttle = rf.DomainThrottle(clock=lambda: clock[0], sleep=sleep)
    return rf.StoreFetcher(net, throttle=throttle, sleep=sleep, now=lambda: NOW)


def test_user_agent_is_the_agreed_string_and_sent_on_every_request() -> None:
    assert rf.USER_AGENT == (
        "StyleMaitriRefresh/1.0 (+https://github.com/gaurav-gandhi-2411/agentic-shopping-assistant)"
    )
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [_page(3)]})
    res = _fetcher(net).fetch_store(LIBAS)
    assert res.status == rf.OK and len(res.products) == 3
    assert {h["User-Agent"] for _, h in net.calls} == {rf.USER_AGENT}


def test_robots_is_fetched_before_any_feed_page() -> None:
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [_page(1)]})
    _fetcher(net).fetch_store(LIBAS)
    assert net.calls[0][0] == "https://libas.in/robots.txt"
    assert "products.json" in net.calls[1][0]


def test_robots_disallow_stops_the_store_without_touching_the_feed() -> None:
    robots = rf.HttpResponse(200, {}, "User-agent: *\nDisallow: /products.json\n")
    net = FakeNet({"robots.txt": [robots]})
    res = _fetcher(net).fetch_store(LIBAS)
    assert res.status == rf.ROBOTS_DISALLOWED
    assert all("products.json" not in u for u, _ in net.calls)


def test_robots_group_naming_our_token_is_honoured() -> None:
    robots = rf.HttpResponse(200, {}, "User-agent: StyleMaitriRefresh\nDisallow: /\n")
    res = _fetcher(FakeNet({"robots.txt": [robots]})).fetch_store(LIBAS)
    assert res.status == rf.ROBOTS_DISALLOWED


@pytest.mark.parametrize("status", [500, 503, 403, 429])
def test_unreadable_robots_fails_closed(status: int) -> None:
    net = FakeNet({"robots.txt": [rf.HttpResponse(status, {}, "")]})
    res = _fetcher(net).fetch_store(LIBAS)
    assert res.status == rf.ROBOTS_UNREACHABLE
    assert len(net.calls) == 1


def test_missing_robots_404_means_no_rules() -> None:
    net = FakeNet({"robots.txt": [rf.HttpResponse(404, {}, "")], "products.json": [_page(2)]})
    assert _fetcher(net).fetch_store(LIBAS).status == rf.OK


def test_crawl_delay_slows_the_domain_and_a_huge_one_stops_the_store() -> None:
    robots = rf.HttpResponse(200, {}, "User-agent: *\nCrawl-delay: 4\n")
    sleeps: list[float] = []
    net = FakeNet({"robots.txt": [robots], "products.json": [_page(rf.FEED_PAGE_SIZE), _page(1, 900)]})
    assert _fetcher(net, sleeps).fetch_store(LIBAS).status == rf.OK
    assert max(sleeps) == pytest.approx(4.0)
    huge = rf.HttpResponse(200, {}, "User-agent: *\nCrawl-delay: 600\n")
    assert _fetcher(FakeNet({"robots.txt": [huge]})).fetch_store(LIBAS).status == rf.ROBOTS_DISALLOWED


def test_at_most_one_request_per_second_per_domain() -> None:
    sleeps: list[float] = []
    pages = [_page(rf.FEED_PAGE_SIZE, i * 1000) for i in range(3)] + [_page(5, 9000)]
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": pages})
    res = _fetcher(net, sleeps).fetch_store(LIBAS)
    assert res.status == rf.OK and res.pages == 4
    assert len(net.calls) == 5  # robots + 4 pages
    assert len(sleeps) == 4 and all(s == pytest.approx(1.0) for s in sleeps)


def test_throttle_is_per_domain() -> None:
    clock = [0.0]
    slept: list[float] = []
    t = rf.DomainThrottle(clock=lambda: clock[0], sleep=slept.append)
    t.wait("a.example")
    t.wait("b.example")  # different domain: no wait
    assert slept == []
    t.wait("a.example")
    assert slept == [pytest.approx(1.0)]


def test_429_with_short_retry_after_waits_then_succeeds() -> None:
    sleeps: list[float] = []
    limited = rf.HttpResponse(429, {"retry-after": "7"}, "slow down")
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [limited, _page(2)]})
    res = _fetcher(net, sleeps).fetch_store(LIBAS)
    assert res.status == rf.OK and 7.0 in sleeps


def test_429_with_long_or_missing_retry_after_stops_the_store_and_drops_partial_rows() -> None:
    for header in ({"retry-after": "3600"}, {}):
        limited = rf.HttpResponse(429, header, "")
        net = FakeNet(
            {"robots.txt": [ROBOTS_OK], "products.json": [_page(rf.FEED_PAGE_SIZE), limited]}
        )
        res = _fetcher(net).fetch_store(LIBAS)
        assert res.status == rf.RATE_LIMITED and res.products == []


def test_second_429_after_waiting_stops_the_store() -> None:
    limited = rf.HttpResponse(429, {"retry-after": "1"}, "")
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [limited]})
    assert _fetcher(net).fetch_store(LIBAS).status == rf.RATE_LIMITED


def test_parse_retry_after_seconds_and_http_date() -> None:
    assert rf.parse_retry_after("12") == 12.0
    assert rf.parse_retry_after("Mon, 16 Nov 2026 02:01:00 GMT", now=NOW) == pytest.approx(60.0)
    assert rf.parse_retry_after("Mon, 16 Nov 2026 01:00:00 GMT", now=NOW) == 0.0
    assert rf.parse_retry_after("soon") is None and rf.parse_retry_after(None) is None


@pytest.mark.parametrize(
    "resp",
    [
        rf.HttpResponse(200, {}, "<html>Verifying your connection...</html>"),
        rf.HttpResponse(403, {}, "forbidden"),
        rf.HttpResponse(503, {}, "<html>Attention Required! | Cloudflare</html>"),
    ],
)
def test_challenge_page_stops_the_store_at_once_no_evasion(resp: rf.HttpResponse) -> None:
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [_page(rf.FEED_PAGE_SIZE), resp]})
    res = _fetcher(net).fetch_store(LIBAS)
    assert res.status == rf.CHALLENGE and res.products == []
    # the challenge page was requested once, never retried by another route
    assert sum("products.json" in u for u, _ in net.calls) == 2


def test_503_is_classified_as_challenge() -> None:
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [rf.HttpResponse(503, {}, "")]})
    assert _fetcher(net).fetch_store(LIBAS).status == rf.CHALLENGE


def test_opted_out_and_non_shopify_stores_are_never_contacted() -> None:
    opted_out = StoreRecord("libas", "Libas", "libas.in", "shopify", True, refresh_enabled=False)
    net = FakeNet({})
    assert _fetcher(net).fetch_store(opted_out).status == rf.OPTED_OUT
    assert _fetcher(net).fetch_store(STORE_REGISTRY["myntra"]).status == rf.OPTED_OUT
    assert net.calls == []


def test_non_json_and_404_feeds_are_not_shopify() -> None:
    for resp in (rf.HttpResponse(200, {}, "<html>hi</html>"), rf.HttpResponse(404, {}, "")):
        net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [resp]})
        assert _fetcher(net).fetch_store(LIBAS).status == rf.NOT_SHOPIFY


def test_empty_feed_and_repeated_page_terminate() -> None:
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [_page(0)]})
    assert _fetcher(net).fetch_store(LIBAS).status == rf.EMPTY
    same = _page(rf.FEED_PAGE_SIZE)  # a store that ignores ?page=N returns page 1 forever
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [same, same]})
    res = _fetcher(net).fetch_store(LIBAS)
    assert res.status == rf.OK and res.pages == 1 and len(res.products) == rf.FEED_PAGE_SIZE


def test_network_error_stops_the_store() -> None:
    def boom(url: str, headers: dict[str, str], timeout: float) -> rf.HttpResponse:
        raise rf.NetworkError("TimeoutError")

    res = rf.StoreFetcher(boom, throttle=rf.DomainThrottle(sleep=lambda s: None)).fetch_store(LIBAS)
    assert res.status == rf.ROBOTS_UNREACHABLE


def test_urllib_get_refuses_plain_http_without_any_network_call() -> None:
    with pytest.raises(rf.NetworkError):
        rf.urllib_get("http://libas.in/products.json", {}, 1.0)


def test_summary_has_no_product_payload() -> None:
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [_page(2)]})
    s = _fetcher(net).fetch_store(LIBAS).summary()
    assert s["products"] == 2 and "title" not in json.dumps(s)


# --- R3 part 1: pagination until exhausted (6b) and the production 429 / 5xx policy (6c) ---------


class TimedNet(FakeNet):
    """FakeNet that stamps each call with the fake clock, to prove the request rate."""

    def __init__(self, routes: dict[str, list[rf.HttpResponse]], clock: list[float]) -> None:
        super().__init__(routes)
        self.clock = clock
        self.times: list[float] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> rf.HttpResponse:
        self.times.append(self.clock[0])
        return super().__call__(url, headers, timeout)


def _timed(routes: dict[str, list[rf.HttpResponse]]) -> tuple[rf.StoreFetcher, TimedNet, list[float]]:
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        clock[0] += s

    net = TimedNet(routes, clock)
    throttle = rf.DomainThrottle(clock=lambda: clock[0], sleep=sleep)
    return rf.StoreFetcher(net, throttle=throttle, sleep=sleep, now=lambda: NOW), net, sleeps


def _assert_rate_ok(times: list[float]) -> None:
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(times, times[1:], strict=False)), times


def test_multi_page_feed_is_exhausted_and_every_product_kept() -> None:
    n = rf.FEED_PAGE_SIZE
    pages = [_page(n, 0), _page(n, 1000), _page(n, 2000), _page(37, 3000)]
    res = _fetcher(FakeNet({"robots.txt": [ROBOTS_OK], "products.json": pages})).fetch_store(LIBAS)
    assert res.status == rf.OK and res.pages == 4 and res.items_fetched == 3 * n + 37
    assert len({p["id"] for p in res.products}) == len(res.products) == 3 * n + 37


def test_exact_multiple_of_page_size_ends_on_the_empty_page() -> None:
    n = rf.FEED_PAGE_SIZE
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [_page(n, 0), _page(n, 1000), _page(0)]})
    res = _fetcher(net).fetch_store(LIBAS)
    assert res.status == rf.OK and len(res.products) == 2 * n and res.pages == 2
    assert sum("products.json" in u for u, _ in net.calls) == 3  # two full pages + the empty one


def test_short_first_page_makes_exactly_one_feed_request() -> None:
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [_page(rf.FEED_PAGE_SIZE - 1)]})
    res = _fetcher(net).fetch_store(LIBAS)
    assert res.status == rf.OK and sum("products.json" in u for u, _ in net.calls) == 1


def test_cap_hit_marks_the_store_capped_and_reports_the_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rf, "MAX_PAGES", 3)
    n = rf.FEED_PAGE_SIZE
    pages = [_page(n, i * 1000) for i in range(10)]  # always full, always new
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": pages})
    res = _fetcher(net).fetch_store(LIBAS)
    assert res.status == rf.CAPPED and res.items_fetched == 3 * n
    assert "cap" in res.detail and str(3 * n) in res.detail
    assert res.products == []  # a partial store is never used
    assert sum("products.json" in u for u, _ in net.calls) == 3  # never fetches past the cap
    assert res.summary()["items_fetched"] == 3 * n


def test_cap_constants_are_consistent() -> None:
    assert rf.MAX_PAGES * rf.FEED_PAGE_SIZE == rf.MAX_ITEMS == 100_000


def test_feed_that_ends_exactly_at_the_cap_is_not_flagged_when_followed_by_a_short_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rf, "MAX_PAGES", 3)
    n = rf.FEED_PAGE_SIZE
    pages = [_page(n, 0), _page(n, 1000), _page(5, 2000)]  # short page inside the cap
    res = _fetcher(FakeNet({"robots.txt": [ROBOTS_OK], "products.json": pages})).fetch_store(LIBAS)
    assert res.status == rf.OK and res.items_fetched == 2 * n + 5


def test_repeated_last_page_adds_nothing_and_stops() -> None:
    n = rf.FEED_PAGE_SIZE
    last = _page(n, 1000)  # a store that repeats its last page forever
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [_page(n, 0), last, last, last]})
    res = _fetcher(net).fetch_store(LIBAS)
    assert res.status == rf.OK and res.items_fetched == 2 * n and res.pages == 2
    assert sum("products.json" in u for u, _ in net.calls) == 3  # stops at the first no-new page


def test_page_with_partly_repeated_ids_keeps_only_the_new_ones() -> None:
    n = rf.FEED_PAGE_SIZE
    overlap = _page(n, n - 10)  # ids n-10 .. 2n-11: 10 repeats of page 1, n-10 new
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [_page(n, 0), overlap, _page(0)]})
    res = _fetcher(net).fetch_store(LIBAS)
    assert res.items_fetched == 2 * n - 10 and len({p["id"] for p in res.products}) == 2 * n - 10


def test_empty_feed_is_empty_not_ok() -> None:
    res = _fetcher(FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [_page(0)]})).fetch_store(LIBAS)
    assert res.status == rf.EMPTY and res.items_fetched == 0


def test_429_short_retry_after_retries_once_through_the_limiter() -> None:
    limited = rf.HttpResponse(429, {"retry-after": "0"}, "")
    fetcher, net, sleeps = _timed({"robots.txt": [ROBOTS_OK], "products.json": [limited, _page(2)]})
    res = fetcher.fetch_store(LIBAS)
    assert res.status == rf.OK and res.items_fetched == 2 and res.rate_limit_retries == 1
    _assert_rate_ok(net.times)  # Retry-After: 0 still cannot beat 1 req/s
    assert sum("products.json" in u for u, _ in net.calls) == 2


def test_429_retry_after_waits_the_header_value_and_boundary_60_is_accepted() -> None:
    limited = rf.HttpResponse(429, {"retry-after": "60"}, "")
    fetcher, net, sleeps = _timed({"robots.txt": [ROBOTS_OK], "products.json": [limited, _page(2)]})
    assert fetcher.fetch_store(LIBAS).status == rf.OK
    assert 60.0 in sleeps
    _assert_rate_ok(net.times)


def test_second_429_stops_the_store_with_the_count_so_far() -> None:
    n = rf.FEED_PAGE_SIZE
    limited = rf.HttpResponse(429, {"retry-after": "5"}, "")
    routes = {"robots.txt": [ROBOTS_OK], "products.json": [_page(n, 0), limited, limited]}
    fetcher, net, _ = _timed(routes)
    res = fetcher.fetch_store(LIBAS)
    assert res.status == rf.RATE_LIMITED and res.items_fetched == n and res.products == []
    assert "250 items fetched" in res.detail
    assert sum("products.json" in u for u, _ in net.calls) == 3  # page 1, 429, one retry, stop
    _assert_rate_ok(net.times)


def test_a_later_429_on_another_page_also_stops_the_store_after_the_one_allowed_retry() -> None:
    n = rf.FEED_PAGE_SIZE
    limited = rf.HttpResponse(429, {"retry-after": "1"}, "")
    pages = [limited, _page(n, 0), limited]  # page 1 limited then ok, page 2 limited again
    fetcher, net, _ = _timed({"robots.txt": [ROBOTS_OK], "products.json": pages})
    res = fetcher.fetch_store(LIBAS)
    assert res.status == rf.RATE_LIMITED and res.items_fetched == n
    assert sum("products.json" in u for u, _ in net.calls) == 3  # no second wait-and-retry


@pytest.mark.parametrize("headers", [{"retry-after": "61"}, {"retry-after": "3600"}, {}])
def test_429_with_long_or_missing_retry_after_stops_immediately_without_waiting(
    headers: dict[str, str],
) -> None:
    limited = rf.HttpResponse(429, headers, "")
    fetcher, net, sleeps = _timed({"robots.txt": [ROBOTS_OK], "products.json": [limited]})
    res = fetcher.fetch_store(LIBAS)
    assert res.status == rf.RATE_LIMITED and res.rate_limit_retries == 0
    assert sum("products.json" in u for u, _ in net.calls) == 1
    assert all(s <= 1.0 for s in sleeps)  # only the limiter ever slept, never a Retry-After


def test_429_retry_after_in_http_date_form() -> None:
    soon = rf.HttpResponse(429, {"retry-after": "Mon, 16 Nov 2026 02:00:30 GMT"}, "")
    fetcher, net, sleeps = _timed({"robots.txt": [ROBOTS_OK], "products.json": [soon, _page(1)]})
    assert fetcher.fetch_store(LIBAS).status == rf.OK and 30.0 in sleeps
    far = rf.HttpResponse(429, {"retry-after": "Mon, 16 Nov 2026 02:05:00 GMT"}, "")  # 300 s
    fetcher, net, _ = _timed({"robots.txt": [ROBOTS_OK], "products.json": [far]})
    assert fetcher.fetch_store(LIBAS).status == rf.RATE_LIMITED
    assert sum("products.json" in u for u, _ in net.calls) == 1


def test_5xx_is_retried_twice_with_backoff_then_the_store_stops() -> None:
    boom = rf.HttpResponse(500, {}, "oops")
    fetcher, net, sleeps = _timed({"robots.txt": [ROBOTS_OK], "products.json": [boom]})
    res = fetcher.fetch_store(LIBAS)
    assert res.status == rf.HTTP_ERROR and "500" in res.detail
    assert sum("products.json" in u for u, _ in net.calls) == 3  # 1 try + 2 retries
    assert [s for s in sleeps if s > 1.0] == [2.0, 4.0]  # backoff doubles
    _assert_rate_ok(net.times)


def test_5xx_recovers_when_a_retry_succeeds() -> None:
    boom = rf.HttpResponse(502, {}, "")
    fetcher, net, _ = _timed({"robots.txt": [ROBOTS_OK], "products.json": [boom, _page(3)]})
    res = fetcher.fetch_store(LIBAS)
    assert res.status == rf.OK and res.items_fetched == 3


def test_persistent_plain_503_is_retried_then_classified_challenge() -> None:
    fetcher, net, _ = _timed(
        {"robots.txt": [ROBOTS_OK], "products.json": [rf.HttpResponse(503, {}, "")]}
    )
    assert fetcher.fetch_store(LIBAS).status == rf.CHALLENGE
    assert sum("products.json" in u for u, _ in net.calls) == 3


@pytest.mark.parametrize("status", [400, 401, 402, 410, 418])
def test_other_4xx_is_never_retried(status: int) -> None:
    net = FakeNet({"robots.txt": [ROBOTS_OK], "products.json": [rf.HttpResponse(status, {}, "")]})
    res = _fetcher(net).fetch_store(LIBAS)
    assert res.status == rf.HTTP_ERROR
    assert sum("products.json" in u for u, _ in net.calls) == 1


def test_whole_run_never_exceeds_one_request_per_second_per_domain() -> None:
    n = rf.FEED_PAGE_SIZE
    limited = rf.HttpResponse(429, {"retry-after": "0"}, "")
    boom = rf.HttpResponse(500, {}, "")
    pages = [_page(n, 0), limited, boom, _page(n, 1000), _page(4, 2000)]
    fetcher, net, _ = _timed({"robots.txt": [ROBOTS_OK], "products.json": pages})
    assert fetcher.fetch_store(LIBAS).status == rf.OK
    assert len(net.times) == 6  # robots + 5 feed requests
    _assert_rate_ok(net.times)


def test_default_throttle_is_the_shared_process_wide_limiter() -> None:
    a, b = rf.StoreFetcher(FakeNet({})), rf.StoreFetcher(FakeNet({}))
    assert a._throttle is b._throttle is rf.GLOBAL_THROTTLE
