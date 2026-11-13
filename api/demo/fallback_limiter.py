"""In-process per-IP + global limiter used when the DB-backed demo limiter is unavailable.

The Postgres limiter in api/demo/guards.py used to fail OPEN (allow everything) on any DB
error or when no DB engine was configured.  This module is the fail-closed replacement:
a per-instance sliding-window per-IP limiter plus a per-instance daily request cap.

Limits are supplied as callables so they are derived from the very same env values as the
DB limiter (DEMO_PER_IP_HOUR_LIMIT / DEMO_DAILY_REQUEST_CAP) and are never looser:
  * per-IP uses a true sliding 3600 s window (the DB uses a fixed UTC hour, which can admit
    up to 2x the limit across an hour boundary);
  * the daily cap is counted per UTC day in this process only (state resets on restart and
    is not shared between instances -- the cap is therefore per instance).

Memory is bounded: at most `max_ips` tracked IPs (LRU eviction) and at most `limit`
timestamps per IP.  A flood of unique IPs evicts the least-recently-seen IPs; the global
daily cap still bounds total admitted traffic.
"""
from __future__ import annotations

import collections
import datetime
import os
import threading
import time
from collections.abc import Callable

_WINDOW_S = 3600.0
_DEFAULT_MAX_IPS = 10_000


def _get_max_ips() -> int:
    return max(1, int(os.environ.get("DEMO_FALLBACK_MAX_IPS", str(_DEFAULT_MAX_IPS))))


class FallbackLimiter:
    """Thread-safe (single lock) per-IP sliding window + global daily cap."""

    def __init__(
        self,
        per_ip_limit: Callable[[], int],
        daily_cap: Callable[[], int],
        max_ips: Callable[[], int] = _get_max_ips,
        clock: Callable[[], float] = time.time,
        window_s: float = _WINDOW_S,
    ) -> None:
        # window_s lets other per-key limiters (e.g. POST /events, per minute) reuse this
        # LRU-bounded sliding window instead of copying it; the default keeps the hourly
        # demo behaviour byte-identical.
        self._window_s = window_s
        self._per_ip_limit = per_ip_limit
        self._daily_cap = daily_cap
        self._max_ips = max_ips
        self._clock = clock
        self._lock = threading.Lock()
        # OrderedDict as LRU: most recently seen IP is moved to the end.
        self._windows: collections.OrderedDict[str, collections.deque[float]] = (
            collections.OrderedDict()
        )
        self._day: str = ""
        self._day_count: int = 0

    def check_ip(self, client_ip: str) -> tuple[bool, int]:
        """Record-and-check one request. Returns (allowed, retry_after_seconds)."""
        limit = self._per_ip_limit()
        now = self._clock()
        cutoff = now - self._window_s
        with self._lock:
            window = self._windows.get(client_ip)
            if window is None:
                window = collections.deque()
                self._windows[client_ip] = window
                max_ips = self._max_ips()
                while len(self._windows) > max_ips:
                    self._windows.popitem(last=False)
            else:
                self._windows.move_to_end(client_ip)
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= limit:
                return False, max(1, int(window[0] + self._window_s - now) + 1)
            window.append(now)
            return True, 0

    def admit_ip(self, client_ip: str, other_count: int = 0) -> tuple[bool, int]:
        """Atomically admit one request iff max(own window count, other_count) < limit.

        ``other_count`` is the DB limiter's count for the same IP (0 when the DB is unusable).
        Always counts in memory, so a DB flap can never reset or hide this window.  A denied
        request is not counted.  Returns (allowed, retry_after_seconds); retry_after is the
        in-memory window's, or 1 when only the DB count blocked (caller supplies its own).
        """
        limit = self._per_ip_limit()
        now = self._clock()
        cutoff = now - self._window_s
        with self._lock:
            window = self._windows.get(client_ip)
            if window is None:
                window = collections.deque()
                self._windows[client_ip] = window
                max_ips = self._max_ips()
                while len(self._windows) > max_ips:
                    self._windows.popitem(last=False)
            else:
                self._windows.move_to_end(client_ip)
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= limit or other_count >= limit:
                retry = int(window[0] + self._window_s - now) + 1 if len(window) >= limit else 1
                return False, max(1, retry)
            window.append(now)
            return True, 0

    def _roll_day(self) -> None:
        today = datetime.datetime.fromtimestamp(self._clock(), tz=datetime.UTC).date().isoformat()
        if today != self._day:
            self._day = today
            self._day_count = 0

    def daily_count(self) -> int:
        """Requests recorded by this instance today (UTC)."""
        with self._lock:
            self._roll_day()
            return self._day_count

    def daily_allowed(self) -> bool:
        """True while this instance's served-request count is below the daily cap."""
        with self._lock:
            self._roll_day()
            return self._day_count < self._daily_cap()

    def record_request(self) -> None:
        with self._lock:
            self._roll_day()
            self._day_count += 1

    def tracked_ips(self) -> int:
        with self._lock:
            return len(self._windows)

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()
            self._day = ""
            self._day_count = 0
