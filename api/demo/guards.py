from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import Any

from sqlalchemy import text

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
# In-memory cost accumulator — survives warm instances, resets on cold start.
# This is intentionally best-effort: a process restart resets the counter, so
# the actual daily cap is enforced by the Postgres demo_daily_stats row as the
# source of truth.  The in-memory value provides a fast, lock-protected check
# on the hot path without a DB round-trip for every message.
# ---------------------------------------------------------------------------

_daily_cost_accumulated: float = 0.0
_cost_lock: threading.Lock = threading.Lock()
_current_day: str = ""  # YYYY-MM-DD UTC; used to detect day rollover


# ---------------------------------------------------------------------------
# Startup initialiser
# ---------------------------------------------------------------------------


def init_demo_guards(engine: Any, brand: str) -> None:
    """Load today's accumulated cost from DB into the in-memory accumulator.

    Called once at startup when DEMO_MODE=true.  Failures are non-fatal: the
    accumulator starts from 0 and the Postgres cap remains the source of truth.
    """
    global _daily_cost_accumulated, _current_day

    import datetime

    today = datetime.date.today().isoformat()
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

    Returns (allowed, retry_after_seconds).  Fails open on DB error so a
    transient DB blip does not block every anonymous visitor.
    """
    ip_hash = _ip_hash(client_ip)
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

            if current_count >= _get_per_ip_limit():
                # Seconds until the current UTC hour rolls over.
                retry_after = 3600 - int(time.time()) % 3600
                return (False, retry_after)

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
    except Exception:
        logger.warning(
            "check_ip_rate_limit: DB error for ip_hash=%s brand=%s; failing open",
            ip_hash,
            brand,
            exc_info=True,
        )
        return (True, 0)

    return (True, 0)


def check_daily_cap(brand: str, engine: Any) -> bool:
    """Return True if today's request_count is below the daily request cap.

    Fails open on DB error.
    """
    import datetime

    today = datetime.date.today().isoformat()
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
            return count < _get_daily_request_cap()
    except Exception:
        logger.warning(
            "check_daily_cap: DB error for brand=%s; failing open",
            brand,
            exc_info=True,
        )
        return True


def check_daily_cost(brand: str) -> bool:
    """Return True if the in-memory accumulated cost is below _DAILY_COST_CAP_USD.

    Pure in-memory check — no DB round-trip.  If the process has been running
    across a UTC day boundary the accumulator is stale; we log and return True
    because the Postgres daily cap is still active.
    """
    import datetime

    today = datetime.date.today().isoformat()
    with _cost_lock:
        if _current_day and _current_day != today:
            # Day rolled over since last startup; accumulator no longer valid.
            logger.info(
                "check_daily_cost: day boundary detected (stored=%s, today=%s);"
                " returning True (cost cap delegated to DB)",
                _current_day,
                today,
            )
            return True
        return _daily_cost_accumulated < _DAILY_COST_CAP_USD


# ---------------------------------------------------------------------------
# Recorders
# ---------------------------------------------------------------------------


def record_request(brand: str, engine: Any) -> None:
    """Increment today's request_count in demo_daily_stats by 1."""
    import datetime

    today = datetime.date.today().isoformat()
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
    except Exception:
        logger.warning(
            "record_request: DB error for brand=%s; request not counted",
            brand,
            exc_info=True,
        )


def record_cost(brand: str, usd_cost: float, engine: Any) -> None:
    """Add usd_cost to the in-memory accumulator and persist to Postgres."""
    import datetime

    today = datetime.date.today().isoformat()

    with _cost_lock:
        global _daily_cost_accumulated
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
