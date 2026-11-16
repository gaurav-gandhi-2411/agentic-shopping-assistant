"""Shared in-process state of the live price check: cache, circuit breakers, per-domain rate limit.

Every method takes the lock for a short, await-free critical section, so the state is safe to share
between worker threads and event loops (POST /chat runs each check in its own loop).
"""

from __future__ import annotations

import threading
import time
import urllib.robotparser
from collections.abc import Callable
from dataclasses import dataclass

from src.pricing.guards import LiveConfig, LiveResult


@dataclass
class _Breaker:
    failures: int = 0
    open_until: float = 0.0


@dataclass
class _Bucket:
    tokens: float
    updated: float


class PoliteState:
    """TTL cache keyed by ``(store, handle)``, per-store breaker, per-domain token bucket."""

    def __init__(
        self, config: LiveConfig, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.config = config
        self.clock = clock
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], tuple[float, LiveResult]] = {}
        self._breakers: dict[str, _Breaker] = {}
        self._buckets: dict[str, _Bucket] = {}
        self._robots: dict[str, tuple[float, urllib.robotparser.RobotFileParser]] = {}

    # ---- cache ----

    def cache_get(self, key: tuple[str, str]) -> LiveResult | None:
        """Live result younger than ``cache_ttl_s``, else None (and the stale entry is dropped)."""
        with self._lock:
            hit = self._cache.get(key)
            if hit and self.clock() - hit[0] < self.config.cache_ttl_s:
                return hit[1]
            self._cache.pop(key, None)
            return None

    def cache_put(self, key: tuple[str, str], value: LiveResult) -> None:
        """Store ``value``; at ``cache_max_entries`` the oldest insertion is evicted."""
        with self._lock:
            if len(self._cache) >= self.config.cache_max_entries:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (self.clock(), value)

    def robots_get(self, domain: str) -> urllib.robotparser.RobotFileParser | None:
        """Parsed robots.txt for ``domain`` if fetched within the TTL."""
        with self._lock:
            hit = self._robots.get(domain)
            if hit and self.clock() - hit[0] < self.config.cache_ttl_s:
                return hit[1]
            return None

    def robots_put(self, domain: str, parser: urllib.robotparser.RobotFileParser) -> None:
        """Remember ``domain``'s parsed robots.txt."""
        with self._lock:
            self._robots[domain] = (self.clock(), parser)

    # ---- circuit breaker ----

    def breaker_open(self, store: str) -> bool:
        """True while the store is being left alone after repeated failures or a 429."""
        with self._lock:
            b = self._breakers.get(store)
            return bool(b and b.open_until > self.clock())

    def record(self, store: str, ok: bool, *, hold_s: float | None = None) -> None:
        """Record one outcome. Success resets the streak; ``breaker_failures`` in a row open the
        breaker for ``breaker_open_s``; ``hold_s`` (a 429 Retry-After) opens it immediately."""
        with self._lock:
            b = self._breakers.setdefault(store, _Breaker())
            if ok:
                b.failures = 0
                return
            b.failures += 1
            if hold_s is not None:
                b.open_until = max(b.open_until, self.clock() + hold_s)
            elif b.failures >= self.config.breaker_failures:
                b.open_until = self.clock() + self.config.breaker_open_s

    # ---- politeness ----

    def reserve(self, domain: str, max_wait: float) -> float | None:
        """Take one request token for ``domain``.

        Token bucket: capacity ``burst``, refill one token per ``min_interval_s``, so the average
        is <= 1 request per ``min_interval_s``. Returns the seconds the caller must wait before
        sending, or None (nothing reserved) when that wait would exceed ``max_wait``.
        """
        cfg = self.config
        with self._lock:
            now = self.clock()
            bucket = self._buckets.setdefault(domain, _Bucket(float(cfg.burst), now))
            bucket.tokens = min(
                float(cfg.burst), bucket.tokens + (now - bucket.updated) / cfg.min_interval_s
            )
            bucket.updated = now
            wait = 0.0 if bucket.tokens >= 1 else (1 - bucket.tokens) * cfg.min_interval_s
            if wait > max_wait:
                return None
            bucket.tokens -= 1
            return wait
