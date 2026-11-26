"""Polite Shopify ``/products.json`` fetcher for the daily index refresh (Wave 1b V3(c)).

Etiquette (architect decision; the numbers live in ``src/config/store_registry.py``):

* at most ``FEED_MAX_REQUESTS_PER_SECOND`` (1) request per domain, robots.txt fetches included;
* User-Agent ``StyleMaitriRefresh/1.0 (+https://github.com/gaurav-gandhi-2411/agentic-shopping-assistant)``;
* robots.txt is fetched first and honoured (``Disallow`` and ``Crawl-delay``); an unreachable
  robots.txt (network error or 5xx) stops the store, a 404 means "no rules";
* HTTP 429 honours ``Retry-After`` ONCE per store, only when it is present and <= 60 s (sleep, then
  retry the same request through the limiter); a second 429, a longer or missing header stops the
  store for the run (``rate_limited``, with the item count fetched so far);
* 5xx: at most 2 retries with exponential backoff, then the store stops; no retry on other 4xx;
* pagination runs until an empty / short page, or a page that adds no new product id, with a hard
  safety cap of ``MAX_ITEMS`` products that marks the store ``capped`` (never silent);
* a challenge / bot-management page (Cloudflare marker, 403, persistent 503) stops the store. Unlike
  ``scripts/download_shopify.py`` there is NO curl TLS-fingerprint fallback here: evading a block is
  not polite. A stopped store is reported, never retried, and the orchestrator carries its rows over
  from the previous build;
* stores with ``refresh_enabled=False`` (the per-store opt-out) are never contacted.

Pure stdlib. All network access goes through an injected ``http_get`` so tests use fixtures and make
no real request.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import NamedTuple
from urllib.parse import urlsplit

from src.config.store_registry import (
    FEED_MAX_REQUESTS_PER_SECOND,
    FEED_PAGE_SIZE,
    StoreRecord,
)

USER_AGENT = (
    "StyleMaitriRefresh/1.0 (+https://github.com/gaurav-gandhi-2411/agentic-shopping-assistant)"
)
ROBOTS_TOKEN = "StyleMaitriRefresh"  # product token matched against robots.txt groups
# Hard safety cap per store. The largest complete feed seen (voylla, 2026-11 rehearsal) has 12,172
# items; the three feeds the old 15,000 cap truncated are re-measured in
# reports/fetch/fetch_pass_2026-11-26.md. 100,000 is ~8x the largest complete feed, so a real
# store never reaches it while a runaway pager stays bounded (400 requests, ~7 min at 1 req/s).
MAX_ITEMS = 100_000
MAX_PAGES = MAX_ITEMS // FEED_PAGE_SIZE
REQUEST_TIMEOUT_S = 20.0
MAX_RETRY_AFTER_S = 60.0  # a longer ask than this stops the store instead of stalling the job
MAX_5XX_RETRIES = 2
BACKOFF_BASE_S = 2.0  # 5xx retry waits: 2 s, then 4 s
MAX_CRAWL_DELAY_S = 30.0  # a larger robots Crawl-delay than this is treated as "do not crawl"
MAX_BODY_CHARS = 50_000_000  # a hostile or broken feed cannot exhaust runner memory

# Markers seen on bot-management interstitials (same list as scripts/download_shopify.py).
CHALLENGE_MARKERS = (
    "verifying your connection",
    "checking your browser",
    "cf-browser-verification",
    "cloudflare",
    "attention required",
)

# Terminal per-store outcomes (machine-readable in the refresh summary).
OK = "ok"
OPTED_OUT = "opted_out"
ROBOTS_DISALLOWED = "robots_disallowed"
ROBOTS_UNREACHABLE = "robots_unreachable"
CHALLENGE = "challenge"
RATE_LIMITED = "rate_limited"
HTTP_ERROR = "http_error"
CAPPED = "capped"
NOT_SHOPIFY = "not_shopify"
EMPTY = "empty"
NETWORK_ERROR = "network_error"


class HttpResponse(NamedTuple):
    """Minimal response: lower-cased header names."""

    status: int
    headers: dict[str, str]
    body: str


HttpGet = Callable[[str, dict[str, str], float], HttpResponse]


class NetworkError(Exception):
    """The request could not be made (DNS, TLS, timeout). HTTP error statuses are NOT raised."""


def urllib_get(url: str, headers: dict[str, str], timeout: float) -> HttpResponse:
    """The real ``http_get``: stdlib urllib, https only, never raises on an HTTP status."""
    if not url.startswith("https://"):
        raise NetworkError("only https URLs are fetched")
    req = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310 - https checked
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read(MAX_BODY_CHARS).decode("utf-8", "replace")
            return HttpResponse(resp.status, {k.lower(): v for k, v in resp.headers.items()}, body)
    except urllib.error.HTTPError as exc:
        body = exc.read(MAX_BODY_CHARS).decode("utf-8", "replace")
        return HttpResponse(exc.code, {k.lower(): v for k, v in exc.headers.items()}, body)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NetworkError(type(exc).__name__) from exc


class DomainThrottle:
    """Enforces a minimum interval between requests to the same domain (monotonic clock)."""

    def __init__(
        self,
        max_per_second: float = FEED_MAX_REQUESTS_PER_SECOND,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be positive")
        self.min_interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._last: dict[str, float] = {}
        self._interval: dict[str, float] = {}
        self._lock = threading.Lock()  # one limiter may serve several threads

    def set_min_interval(self, domain: str, seconds: float) -> None:
        """Raise (never lower) the interval for ``domain`` (robots Crawl-delay)."""
        self._interval[domain] = max(self._interval.get(domain, 0.0), seconds)

    def wait(self, domain: str) -> None:
        """Block until the next request to ``domain`` is allowed, then record it."""
        with self._lock:
            interval = max(self.min_interval, self._interval.get(domain, 0.0))
            last = self._last.get(domain)
            if last is not None:
                remaining = interval - (self._clock() - last)
                if remaining > 0:
                    self._sleep(remaining)
            self._last[domain] = self._clock()


# Process-wide limiter: every StoreFetcher built without an explicit throttle shares it, so two
# fetchers in one process can never exceed FEED_MAX_REQUESTS_PER_SECOND against the same domain.
GLOBAL_THROTTLE = DomainThrottle()


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """``Retry-After`` as seconds: delta-seconds or an HTTP-date. None when absent or unparseable."""
    if not value:
        return None
    text = value.strip()
    if re.fullmatch(r"\d+", text):
        return float(text)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - (now or datetime.now(timezone.utc))).total_seconds())


def has_challenge_marker(resp: HttpResponse) -> bool:
    """True when the body head carries a bot-management marker."""
    head = resp.body[:2000].lower()
    return any(m in head for m in CHALLENGE_MARKERS)


def looks_like_challenge(resp: HttpResponse) -> bool:
    """A bot-management page: 403/503, or any 200/other body carrying a challenge marker."""
    return resp.status in (403, 503) or has_challenge_marker(resp)


@dataclass
class FetchResult:
    """One store's outcome. ``products`` is non-empty only for ``status == "ok"``."""

    slug: str
    status: str
    pages: int = 0
    requests: int = 0
    detail: str = ""
    fetched_at: str = ""
    items_fetched: int = 0  # distinct products fetched so far; kept when ``products`` is dropped
    rate_limit_retries: int = 0  # 429 waits already spent on this store (policy: at most 1)
    products: list[dict] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        """JSON-safe view without the product payload."""
        return {
            "slug": self.slug,
            "status": self.status,
            "pages": self.pages,
            "requests": self.requests,
            "products": len(self.products),
            "items_fetched": self.items_fetched,
            "detail": self.detail,
            "fetched_at": self.fetched_at,
        }


class StoreFetcher:
    """Fetch stores one at a time through one throttle. ``http_get`` is injectable for tests."""

    def __init__(
        self,
        http_get: HttpGet = urllib_get,
        *,
        throttle: DomainThrottle | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._get = http_get
        self._throttle = throttle or GLOBAL_THROTTLE
        self._sleep = sleep
        self._now = now
        self._headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        self.requests_by_domain: dict[str, int] = {}

    def _fetch(self, url: str) -> HttpResponse:
        domain = urlsplit(url).hostname or ""
        self._throttle.wait(domain)
        self.requests_by_domain[domain] = self.requests_by_domain.get(domain, 0) + 1
        return self._get(url, self._headers, REQUEST_TIMEOUT_S)

    def _check_robots(self, record: StoreRecord, result: FetchResult) -> bool:
        """True when ``/products.json`` may be fetched; otherwise sets ``result`` and returns False."""
        assert record.domain is not None
        domain = record.domain
        result.requests += 1
        try:
            resp = self._fetch(f"https://{domain}/robots.txt")
        except NetworkError as exc:
            result.status, result.detail = ROBOTS_UNREACHABLE, f"robots.txt: {exc}"
            return False
        if resp.status >= 500 or resp.status in (401, 403, 429):
            # Fail closed: not knowing the rules is not permission.
            result.status, result.detail = ROBOTS_UNREACHABLE, f"robots.txt HTTP {resp.status}"
            return False
        if resp.status >= 400:  # 404 etc.: no robots file means no rules
            return True
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(resp.body.splitlines())
        if not parser.can_fetch(ROBOTS_TOKEN, f"https://{domain}/products.json"):
            result.status, result.detail = ROBOTS_DISALLOWED, "robots.txt disallows /products.json"
            return False
        delay = parser.crawl_delay(ROBOTS_TOKEN)
        if delay is not None:
            if float(delay) > MAX_CRAWL_DELAY_S:
                result.status = ROBOTS_DISALLOWED
                result.detail = f"robots.txt Crawl-delay {delay}s exceeds {MAX_CRAWL_DELAY_S}s"
                return False
            self._throttle.set_min_interval(domain, float(delay))
        return True

    def fetch_store(self, record: StoreRecord) -> FetchResult:
        """Fetch every page of one store's feed, or stop at the first etiquette problem."""
        result = FetchResult(slug=record.slug, status=OK, fetched_at=self._now().isoformat())
        if not record.refresh_enabled or record.source_type != "shopify" or not record.domain:
            result.status, result.detail = OPTED_OUT, "refresh_enabled is False or no feed"
            return result
        if not self._check_robots(record, result):
            return result

        seen_ids: set[object] = set()
        for page in range(1, MAX_PAGES + 1):
            url = record.feed_url(page)
            assert url is not None
            resp = self._page(url, result)
            if resp is None:
                break  # result already holds the stop reason
            if resp.status == 404:
                result.status, result.detail = NOT_SHOPIFY, f"HTTP 404 on page {page}"
                break
            if looks_like_challenge(resp):
                result.status, result.detail = CHALLENGE, f"challenge page on page {page}"
                break
            if resp.status != 200:
                result.status, result.detail = HTTP_ERROR, f"HTTP {resp.status} on page {page}"
                break
            try:
                batch = json.loads(resp.body)["products"]
                if not isinstance(batch, list):
                    raise TypeError("products is not a list")
            except (ValueError, KeyError, TypeError):
                result.status, result.detail = NOT_SHOPIFY, f"not a products.json on page {page}"
                break
            if not batch:
                break
            fresh = []
            for p in batch:
                if not isinstance(p, dict):
                    continue
                pid = p.get("id")
                if pid is not None:
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                fresh.append(p)
            if not fresh:  # page repeats known ids (store ignores ?page=N): exhausted
                break
            result.products.extend(fresh)
            result.items_fetched = len(result.products)
            result.pages = page
            if len(batch) < FEED_PAGE_SIZE:
                break
        else:  # MAX_PAGES full, all-new pages and still not exhausted: never silent
            result.status = CAPPED
            result.detail = (
                f"safety cap hit: {MAX_PAGES} pages / {result.items_fetched} items "
                f"(MAX_ITEMS={MAX_ITEMS}); feed not exhausted"
            )
        if result.status == OK and not result.products:
            result.status, result.detail = EMPTY, "feed returned no products"
        if result.status != OK:
            result.products = []  # a partial store is never used: carry over instead
        return result

    def _page(self, url: str, result: FetchResult) -> HttpResponse | None:
        """One page with the 429 / 5xx policy; None (and ``result`` set) when the store must stop."""
        server_retries = 0
        while True:
            result.requests += 1
            try:
                resp = self._fetch(url)
            except NetworkError as exc:
                result.status, result.detail = NETWORK_ERROR, str(exc)
                return None
            if resp.status == 429:
                header = resp.headers.get("retry-after")
                wait = parse_retry_after(header, now=self._now())
                if result.rate_limit_retries >= 1 or wait is None or wait > MAX_RETRY_AFTER_S:
                    result.status = RATE_LIMITED
                    result.detail = (
                        f"HTTP 429, Retry-After={header!r}, after {result.rate_limit_retries} "
                        f"retry; {result.items_fetched} items fetched before stopping"
                    )
                    return None
                result.rate_limit_retries += 1
                self._sleep(wait)  # the retry still goes through the per-domain limiter
                continue
            if resp.status >= 500 and not has_challenge_marker(resp):
                if server_retries < MAX_5XX_RETRIES:
                    self._sleep(BACKOFF_BASE_S * 2**server_retries)
                    server_retries += 1
                    continue
                if resp.status != 503:  # a persistent 503 stays "challenge" in the caller
                    result.status = HTTP_ERROR
                    result.detail = f"HTTP {resp.status} after {MAX_5XX_RETRIES} retries"
                    return None
            return resp
