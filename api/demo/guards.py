from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import Any

from sqlalchemy import text

from api.demo.fallback_limiter import FallbackLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit / cap constants
# ---------------------------------------------------------------------------

# 35/hr sized for a genuine exploration session (search + refine + outfit +
# partner look, ~10 requests) without hitting 429 mid-session, while staying
# bounded against single-IP abuse. DEMO_PER_IP_HOUR_LIMIT env var can still
# override without a code change or Docker rebuild if it needs tuning again.
def _get_per_ip_limit() -> int:
    return max(1, int(os.environ.get("DEMO_PER_IP_HOUR_LIMIT", "35")))


# Kept at the same 20x ratio to the per-IP limit as the original 200/10
# default. DEMO_DAILY_REQUEST_CAP env var can still override without a code
# change or Docker rebuild. The $0.50/day _DAILY_COST_CAP_USD below is the
# actual cost backstop, so this cap exists to bound request volume, not spend.
def _get_daily_request_cap() -> int:
    return max(1, int(os.environ.get("DEMO_DAILY_REQUEST_CAP", "700")))

_DAILY_COST_CAP_USD: float = 0.50  # USD per brand per UTC day

# ---------------------------------------------------------------------------
# Fail-closed fallback.  When the DB limiter cannot be used (no engine, connect/query
# error) requests are limited by an in-process limiter with limits derived from the same
# env-backed getters above.  After a DB failure the DB is skipped for _DB_RETRY_S so a
# hanging/dead DB costs one failed attempt per window, not one per request.
#
# On recovery the fallback counts are NOT merged into Postgres: the per-IP DB window is one
# hour, so outage counts age out almost immediately; the daily DB count under-counts by at
# most the fallback's own daily cap (served requests are never lost, merely uncounted).
# ---------------------------------------------------------------------------

_DB_RETRY_S: float = 15.0

_fallback = FallbackLimiter(_get_per_ip_limit, _get_daily_request_cap)


class _DbHealth:
    """Tracks DB-limiter health; logs one warning per DB->memory and memory->DB transition."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.degraded = False
        self._retry_at = 0.0

    def usable(self, engine: Any, where: str) -> bool:
        if engine is None:
            self._enter(where, "no database engine configured (DATABASE_URL unset)", None)
            return False
        with self._lock:
            return not (self.degraded and time.monotonic() < self._retry_at)

    def failed(self, where: str, exc: BaseException) -> None:
        self._enter(where, "database error", exc)

    def ok(self) -> None:
        with self._lock:
            was_degraded = self.degraded
            self.degraded = False
        if was_degraded:
            logger.info(
                "demo_limiter_transition state=db reason=recovered; "
                "using database-backed limits again"
            )

    def _enter(self, where: str, reason: str, exc: BaseException | None) -> None:
        with self._lock:
            first = not self.degraded
            self.degraded = True
            self._retry_at = time.monotonic() + _DB_RETRY_S
        if first:
            logger.warning(
                "demo_limiter_transition state=memory reason=%s where=%s; failing CLOSED "
                "to in-process per-IP + global limits",
                reason,
                where,
                exc_info=(type(exc), exc, exc.__traceback__) if exc is not None else False,
            )


_health = _DbHealth()


def reset_fallback_state() -> None:
    """Clear fallback counters and health state (tests)."""
    global _health
    _fallback.reset()
    _health = _DbHealth()

# ---------------------------------------------------------------------------
# In-memory cost accumulator — survives warm instances, resets on cold start.
# This is intentionally best-effort: a process restart resets the counter, so
# the actual daily cap is enforced by the Postgres demo_daily_stats row as the
# source of truth.  The in-memory value provides a fast, lock-protected check
# on the hot path without a DB round-trip for every message.
# ---------------------------------------------------------------------------

_daily_cost_accumulated: float = 0.0
_cost_lock: threading.Lock = threading.Lock()
_current_day: str = ""  # YYYY-MM-DD UTC; used to detect day rollover
# Set when a rollover reset the accumulator; the next record_cost (which has an engine)
# reloads today's DB total so spend by other instances earlier today is not forgotten.
_reload_pending: bool = False


def _today_utc() -> str:
    """The day key (YYYY-MM-DD) in UTC. Single seam so tests can fake the clock; the day key
    must be UTC (Postgres rows and the cap are per UTC day), never the host's local date."""
    import datetime

    return datetime.datetime.now(datetime.UTC).date().isoformat()


def _roll_day_locked(today: str) -> bool:
    """Reset the accumulator on the first call of a new UTC day. Caller MUST hold _cost_lock.

    Returns True if a rollover happened. The first-ever call (empty ``_current_day``) just
    adopts the day without resetting, as before.
    """
    global _daily_cost_accumulated, _current_day, _reload_pending
    if _current_day == today:
        return False
    previous = _current_day
    _current_day = today
    if not previous:
        return False
    logger.info(
        "demo cost accumulator rolled over (stored=%s, today=%s); reset from %.6f to 0.0",
        previous,
        today,
        _daily_cost_accumulated,
    )
    _daily_cost_accumulated = 0.0
    _reload_pending = True
    return True


# ---------------------------------------------------------------------------
# Startup initialiser
# ---------------------------------------------------------------------------


def init_demo_guards(engine: Any, brand: str) -> None:
    """Load today's accumulated cost from DB into the in-memory accumulator.

    Called once at startup when DEMO_MODE=true.  Failures are non-fatal: the
    accumulator starts from 0 and the Postgres cap remains the source of truth.
    """
    global _daily_cost_accumulated, _current_day, _reload_pending

    today = _today_utc()
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT cost_usd FROM demo_daily_stats"
                    " WHERE brand = :brand AND date = :date"
                ),
                {"brand": brand, "date": today},
            ).fetchone()
            loaded = float(row[0]) if row else 0.0
    except Exception:
        logger.warning(
            "init_demo_guards: could not load accumulated cost from DB; starting from 0.0",
            exc_info=True,
        )
        loaded = 0.0

    with _cost_lock:
        _daily_cost_accumulated = loaded
        _current_day = today
        _reload_pending = False

    logger.info(
        "init_demo_guards: loaded cost_usd=%.6f for brand=%s date=%s",
        loaded,
        brand,
        today,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ip_hash(client_ip: str) -> str:
    """SHA-256 of the raw IP string, truncated to 32 hex chars for column width."""
    return hashlib.sha256(client_ip.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def check_ip_rate_limit(client_ip: str, brand: str, engine: Any) -> tuple[bool, int]:
    """Check and increment the per-IP hourly rate limit.

    Returns (allowed, retry_after_seconds).  Counts in the in-process window ALWAYS, in
    parallel with the DB, and denies when max(DB count, in-process count) >= limit, so a DB
    outage or flap can never reset or hide a window (E9).  Fails CLOSED: on a missing engine or
    any DB error the in-process count alone decides.
    """
    limit = _get_per_ip_limit()
    ip_hash = _ip_hash(client_ip)
    if not _health.usable(engine, "check_ip_rate_limit"):
        return _fallback.admit_ip(client_ip)
    decision: tuple[bool, int] | None = None
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT count FROM demo_rate_limits"
                    " WHERE ip_hash = :ih AND brand = :brand"
                    " AND window_start = date_trunc('hour', NOW() AT TIME ZONE 'UTC')"
                ),
                {"ih": ip_hash, "brand": brand},
            ).fetchone()
            current_count = int(row[0]) if row else 0

            allowed, retry = _fallback.admit_ip(client_ip, current_count)
            result = decision = (allowed, retry)
            if not allowed:
                if current_count >= limit:
                    # Seconds until the current UTC hour rolls over.
                    retry = max(retry, 3600 - int(time.time()) % 3600)
                result = decision = (False, retry)
            else:
                # Upsert: insert a fresh row or bump the counter.
                conn.execute(
                    text(
                        "INSERT INTO demo_rate_limits (ip_hash, brand, window_start, count)"
                        " VALUES ("
                        "   :ih, :brand,"
                        "   date_trunc('hour', NOW() AT TIME ZONE 'UTC'),"
                        "   1"
                        " )"
                        " ON CONFLICT (ip_hash, brand, window_start)"
                        " DO UPDATE SET count = demo_rate_limits.count + 1"
                    ),
                    {"ih": ip_hash, "brand": brand},
                )
    except Exception as exc:
        _health.failed("check_ip_rate_limit", exc)
        # If the memory window already decided before the DB error (e.g. the upsert failed),
        # keep that decision: re-admitting would double-count.
        return decision if decision is not None else _fallback.admit_ip(client_ip)

    _health.ok()
    return result


def check_daily_cap(brand: str, engine: Any) -> bool:
    """Return True if today's request_count is below the daily request cap.

    Denies when max(DB count, in-process count) >= cap, so a DB flap never resets the day's
    count.  Fails CLOSED: on a missing engine or DB error the per-instance in-process daily
    count (same cap) decides alone.
    """
    if not _health.usable(engine, "check_daily_cap"):
        return _fallback.daily_allowed()
    today = _today_utc()
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT request_count FROM demo_daily_stats"
                    " WHERE brand = :brand AND date = :date"
                ),
                {"brand": brand, "date": today},
            ).fetchone()
            count = int(row[0]) if row else 0
    except Exception as exc:
        _health.failed("check_daily_cap", exc)
        return _fallback.daily_allowed()
    _health.ok()
    return max(count, _fallback.daily_count()) < _get_daily_request_cap()


def check_daily_cost(brand: str) -> bool:
    """Return True if the in-memory accumulated cost is below _DAILY_COST_CAP_USD.

    Pure in-memory check — no DB round-trip.  If the process has been running
    across a UTC day boundary the accumulator is yesterday's: it is reset to 0
    (atomically, under the lock) at the first call of the new UTC day and today's
    total is reloaded from the DB by the next record_cost.
    """
    today = _today_utc()
    with _cost_lock:
        _roll_day_locked(today)
        return _daily_cost_accumulated < _DAILY_COST_CAP_USD


# ---------------------------------------------------------------------------
# Recorders
# ---------------------------------------------------------------------------


def record_request(brand: str, engine: Any) -> None:
    """Increment today's request_count in demo_daily_stats by 1.

    Always also counts in the in-process fallback so its daily cap is accurate if the DB
    goes away mid-day.
    """
    _fallback.record_request()
    if not _health.usable(engine, "record_request"):
        return
    today = _today_utc()
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO demo_daily_stats (brand, date, request_count, cost_usd)"
                    " VALUES (:brand, :date, :req, :cost)"
                    " ON CONFLICT (brand, date)"
                    " DO UPDATE SET"
                    "   request_count = demo_daily_stats.request_count + EXCLUDED.request_count,"
                    "   cost_usd = demo_daily_stats.cost_usd + EXCLUDED.cost_usd"
                ),
                {"brand": brand, "date": today, "req": 1, "cost": 0},
            )
    except Exception as exc:
        _health.failed("record_request", exc)


def record_cost(brand: str, usd_cost: float, engine: Any) -> None:
    """Add usd_cost to the in-memory accumulator and persist to Postgres."""
    global _daily_cost_accumulated, _reload_pending

    today = _today_utc()

    with _cost_lock:
        _roll_day_locked(today)
        _daily_cost_accumulated += usd_cost

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO demo_daily_stats (brand, date, request_count, cost_usd)"
                    " VALUES (:brand, :date, :req, :cost)"
                    " ON CONFLICT (brand, date)"
                    " DO UPDATE SET"
                    "   request_count = demo_daily_stats.request_count + EXCLUDED.request_count,"
                    "   cost_usd = demo_daily_stats.cost_usd + EXCLUDED.cost_usd"
                ),
                {"brand": brand, "date": today, "req": 0, "cost": usd_cost},
            )
    except Exception:
        logger.warning(
            "record_cost: DB error for brand=%s cost=%.6f; DB not updated",
            brand,
            usd_cost,
            exc_info=True,
        )
        return

    # After a rollover, pick up what other instances already spent today. The DB read is done
    # outside the lock; max() keeps this race-safe against concurrent record_cost calls.
    with _cost_lock:
        pending = _reload_pending and _current_day == today
        if pending:
            _reload_pending = False
    if not pending:
        return
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT cost_usd FROM demo_daily_stats"
                    " WHERE brand = :brand AND date = :date"
                ),
                {"brand": brand, "date": today},
            ).fetchone()
        db_total = float(row[0]) if row else 0.0
    except Exception:
        logger.warning("record_cost: rollover reload failed; keeping accumulator", exc_info=True)
        return
    with _cost_lock:
        if _current_day == today:
            _daily_cost_accumulated = max(_daily_cost_accumulated, db_total)
