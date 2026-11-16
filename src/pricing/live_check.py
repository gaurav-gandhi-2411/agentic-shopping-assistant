"""Render-time live price/availability check for the cards of one response (V3(d)(i)).

For each card about to be returned (at most ``max_cards``), fetch the product's live price and
stock from the store's PUBLIC Shopify product JSON (``/products/<handle>.js``), in parallel, under
one total time budget. The check never blocks a response beyond the budget and never raises:

  - live price differs  -> ``price_inr`` becomes the live price, ``indexed_price_inr`` keeps the
                           catalogue value;
  - positively sold out / 404 / 410 -> ``unavailable=True`` and the Buy link is removed;
  - timeout, 429, 5xx, malformed JSON, robots disallow, open circuit breaker -> the indexed price
    stays and the card is marked ``price_may_have_changed=True`` (honest "could not confirm").

Safety rails (all covered in ``tests/test_live_price_check.py``):
  - only stores that are public, Shopify-backed and not opted out are ever contacted;
  - the URL is built by ``guards.product_url`` from the registry domain (https only) plus a
    strictly validated handle, redirects are not followed, so no request can leave the store's own
    domain (SSRF guard);
  - politeness (``state.PoliteState``): token bucket per domain (average <= 1 request/s, burst 2
    so robots.txt plus one product fit in one budget), robots.txt honoured, no retry on 429,
    per-store circuit breaker;
  - 1 h in-process cache keyed by ``(store, handle)``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.robotparser
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx

from src.pricing import metrics
from src.pricing.guards import (
    MAX_BODY_BYTES,
    ROBOTS_TOKEN,
    USER_AGENT,
    LiveConfig,
    LiveResult,
    parse_product_js,
    product_url,
)
from src.pricing.state import PoliteState

if TYPE_CHECKING:
    from api.schemas import ItemSummary

logger = logging.getLogger(__name__)

_PRICE_EPSILON_INR = 0.5  # below this a "difference" is rounding, not a price change


class _Skip(Exception):
    """The check was not performed; keep the indexed price and flag it as unconfirmed."""

    def __init__(self, outcome: str) -> None:
        super().__init__(outcome)
        self.outcome = outcome


class LivePriceChecker:
    """Holds the cache, rate limiter and breakers; one instance per process."""

    def __init__(
        self,
        config: LiveConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or LiveConfig()
        self._transport = transport
        self._clock = clock
        self._state = PoliteState(self.config, clock)

    async def _get(
        self, client: httpx.AsyncClient, domain: str, url: str, deadline: float
    ) -> httpx.Response:
        """One polite GET: wait for a rate-limit slot, then request. Never follows redirects."""
        parsed = httpx.URL(url)
        if parsed.scheme != "https" or parsed.host != domain:  # SSRF belt and braces
            raise _Skip("blocked_url")
        wait = self._state.reserve(domain, max(0.0, deadline - self._clock()))
        if wait is None:
            raise _Skip("rate_limited_local")
        if wait:
            await asyncio.sleep(wait)
        return await client.get(url)

    async def _robots_allows(
        self, client: httpx.AsyncClient, domain: str, path: str, deadline: float
    ) -> bool:
        rp = self._state.robots_get(domain)
        if rp is None:
            resp = await self._get(client, domain, f"https://{domain}/robots.txt", deadline)
            rp = urllib.robotparser.RobotFileParser()
            if resp.status_code == 200:
                rp.parse(resp.text[:MAX_BODY_BYTES].splitlines())
            elif 400 <= resp.status_code < 500:
                rp.parse([])  # no robots.txt: everything allowed
            else:
                raise _Skip("robots_unavailable")  # 5xx/redirect: do not guess, do not cache
            self._state.robots_put(domain, rp)
        return rp.can_fetch(ROBOTS_TOKEN, path)

    async def _fetch_one(
        self, client: httpx.AsyncClient, store: str, handle: str, deadline: float
    ) -> tuple[LiveResult, str]:
        """Returns (result, outcome). Raises _Skip when no live answer was obtained."""
        key = (store, handle)
        hit = self._state.cache_get(key)
        if hit is not None:
            return hit, "cache_hit"
        url = product_url(store, handle)
        if url is None:
            raise _Skip("not_eligible")
        if self._state.breaker_open(store):
            raise _Skip("circuit_open")
        domain = httpx.URL(url).host
        if not await self._robots_allows(client, domain, httpx.URL(url).path, deadline):
            raise _Skip("robots_disallow")
        resp = await self._get(client, domain, url, deadline)
        code = resp.status_code
        if code in (404, 410):
            self._state.record(store, True)
            result = LiveResult(price_inr=None, available=False)
        elif code == 429:
            try:
                retry_after = float(resp.headers.get("Retry-After", ""))
                hold = min(retry_after, self.config.max_retry_after_s)
            except ValueError:
                hold = self.config.breaker_open_s
            self._state.record(store, False, hold_s=hold)  # no retry: back off the whole store
            raise _Skip("http_429")
        elif code == 200:
            try:
                if len(resp.content) > MAX_BODY_BYTES:
                    raise ValueError("body too large")
                result = parse_product_js(resp.content)
            except ValueError:
                self._state.record(store, False)
                raise _Skip("malformed") from None
            self._state.record(store, True)
        elif 300 <= code < 400:
            raise _Skip("redirect")  # never follow: could leave the store's own domain
        else:
            self._state.record(store, False)
            raise _Skip(f"http_{code // 100}xx")
        self._state.cache_put(key, result)
        return result, "ok"

    async def check(self, items: list[ItemSummary]) -> None:
        """Annotate ``items`` in place (<= ``max_cards`` of them). Never raises."""
        cfg = self.config
        start = self._clock()
        deadline = start + cfg.budget_s
        targets: dict[tuple[str, str], list[ItemSummary]] = {}
        for it in items[: cfg.max_cards]:
            if it.is_owned or not it.store or not it.pdp_handle:
                continue
            if product_url(it.store, it.pdp_handle) is None:
                continue  # non-public / non-Shopify / hostile handle: never fetched, not flagged
            targets.setdefault((it.store.lower(), it.pdp_handle.strip()), []).append(it)
        if not targets:
            return

        outcomes: dict[tuple[str, str], tuple[LiveResult | None, str]] = {}

        async def run(client: httpx.AsyncClient, key: tuple[str, str]) -> None:
            try:
                res, outcome = await self._fetch_one(client, key[0], key[1], deadline)
                outcomes[key] = (res, outcome)
            except _Skip as skip:
                outcomes[key] = (None, skip.outcome)
            except (httpx.HTTPError, OSError) as exc:
                self._state.record(key[0], False)
                outcomes[key] = (None, "error")
                logger.debug("live_price_check network error store=%s: %r", key[0], exc)

        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=cfg.budget_s,
                follow_redirects=False,
            ) as client:
                tasks = [asyncio.ensure_future(run(client, k)) for k in targets]
                _, pending = await asyncio.wait(tasks, timeout=cfg.budget_s)
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
        except Exception as exc:  # fail open: a broken check must never break a response
            logger.warning("live_price_check failed open: %r", exc)

        self._apply(targets, outcomes, start)

    def _apply(
        self,
        targets: dict[tuple[str, str], list[ItemSummary]],
        outcomes: dict[tuple[str, str], tuple[LiveResult | None, str]],
        start: float,
    ) -> None:
        n_changed = n_unavail = 0
        for key, cards in targets.items():
            res, outcome = outcomes.get(key, (None, "timeout"))
            metrics.inc(metrics.CHECKS, outcome=outcome, store=key[0])
            for card in cards:
                if res is None:
                    card.price_may_have_changed = True
                    continue
                if not res.available:
                    card.unavailable = True
                    card.pdp_url = None  # fail closed for Buy: positively unavailable only
                    n_unavail += 1
                    metrics.inc(metrics.UNAVAILABLE, store=key[0])
                elif (
                    res.price_inr is not None
                    and card.price_inr is not None
                    and abs(res.price_inr - card.price_inr) > _PRICE_EPSILON_INR
                ):
                    card.indexed_price_inr = card.price_inr
                    card.price_inr = res.price_inr
                    n_changed += 1
                    metrics.inc(metrics.PRICE_CHANGED, store=key[0])
        elapsed = self._clock() - start
        metrics.observe_duration(elapsed)
        n_timeout = sum(1 for k in targets if outcomes.get(k, (None, "timeout"))[1] == "timeout")
        logger.info(
            "live_price_check",
            extra={
                "n_cards": sum(len(v) for v in targets.values()),
                "n_lookups": len(targets),
                "n_timeout": n_timeout,
                "n_price_changed": n_changed,
                "n_unavailable": n_unavail,
                "latency_ms": round(elapsed * 1000),
                "budget_ms": round(self.config.budget_s * 1000),
            },
        )


_checker: LivePriceChecker | None = None


def get_checker(config: dict[str, Any] | None = None) -> LivePriceChecker:
    """Process-wide checker (cache and breakers live here). ``config`` is read once."""
    global _checker
    if _checker is None:
        _checker = LivePriceChecker(LiveConfig.from_config(config))
    return _checker


def reset_checker() -> None:
    """Drop the process-wide checker (tests only)."""
    global _checker
    _checker = None


def sink_unavailable(items: list[ItemSummary]) -> list[ItemSummary]:
    """Stable partition: unavailable cards move to the end. Result count is unchanged."""
    return [i for i in items if not i.unavailable] + [i for i in items if i.unavailable]


def _enabled_checker() -> LivePriceChecker | None:
    try:
        import api.deps as deps

        cfg: dict[str, Any] | None = deps.get_config()
    except Exception:
        cfg = None
    checker = get_checker(cfg)
    return checker if checker.config.enabled else None


async def apply_live_check(items: list[ItemSummary]) -> list[ItemSummary]:
    """Flag-gated, fail-open entry point for async callers (WebSocket path)."""
    try:
        checker = _enabled_checker()
        if checker is None:
            return items
        await checker.check(items)
        return sink_unavailable(items)
    except Exception as exc:
        logger.warning("live_price_check entry failed open: %r", exc)
        return items


def apply_live_check_sync(items: list[ItemSummary]) -> list[ItemSummary]:
    """Same as :func:`apply_live_check` for sync callers (POST /chat runs in a worker thread)."""
    try:
        checker = _enabled_checker()
        if checker is None:
            return items
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no loop in this thread: the expected case, asyncio.run is safe
        else:
            raise RuntimeError("apply_live_check_sync called inside a running event loop")
        asyncio.run(checker.check(items))
        return sink_unavailable(items)
    except Exception as exc:
        logger.warning("live_price_check entry failed open: %r", exc)
        return items
