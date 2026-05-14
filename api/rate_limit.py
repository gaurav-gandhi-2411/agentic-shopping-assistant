"""In-memory per-user sliding-window rate limiter.

Single-instance only — state lives in the process. Suitable for Fly.io
single-machine deployments; replace with a Redis-backed limiter for
multi-instance.

Config: RATE_LIMIT_PER_MINUTE env var (default 10). Read on every call so a
rolling restart isn't required to tune the limit.
"""
from __future__ import annotations

import collections
import os
import threading
import time


def _get_limit() -> int:
    return max(1, int(os.environ.get("RATE_LIMIT_PER_MINUTE", "10")))


class RateLimiter:
    """Sliding-window counter keyed on user_id.

    Thread-safe via threading.Lock — works for both sync and async routes.
    """

    def __init__(self) -> None:
        self._windows: dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def is_allowed(self, user_id: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds).

        retry_after_seconds is 0 when allowed; otherwise the seconds until the
        oldest in-window request ages out and a slot opens.
        """
        limit = _get_limit()
        now = time.monotonic()
        cutoff = now - 60.0

        with self._lock:
            if user_id not in self._windows:
                self._windows[user_id] = collections.deque()
            window = self._windows[user_id]

            # Evict timestamps older than 60 s.
            while window and window[0] < cutoff:
                window.popleft()

            if len(window) < limit:
                window.append(now)
                return True, 0

            retry_after = int(window[0] - cutoff) + 1
            return False, retry_after

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()


# Module-level singleton.
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter
