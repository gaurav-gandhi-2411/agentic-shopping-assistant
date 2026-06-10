"""Dashboard route: GET /dashboard.

Returns aggregated brand metrics from the styling_events and pairing_stats
tables created by migration 0005.  Returns zeros and empty lists when no data
exists yet — never raises a 5xx to the caller.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import text

from api.deps import get_db_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASKET_CAVEAT = (
    "Estimated lift in basket size (look vs single-item purchase, this session data). "
    "Not a controlled A/B test."
)

_MIN_PAIRING_SIGNALS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return numerator / denominator, or 0.0 when denominator is zero.

    Args:
        numerator:   The event count to divide.
        denominator: The look count to divide by.

    Returns:
        Ratio clamped to [0.0, 1.0], or 0.0 when denominator is 0.
    """
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _basket_delta(look_avg: float | None, single_avg: float | None) -> float | None:
    """Return look_avg - single_avg, or None if either value is missing.

    Args:
        look_avg:   Mean basket value for add-the-look events (INR).
        single_avg: Mean basket value for add-single events (INR).

    Returns:
        Difference in INR, or None when either input is None.
    """
    if look_avg is None or single_avg is None:
        return None
    return round(look_avg - single_avg, 2)


def _empty_dashboard() -> dict[str, Any]:
    """Return a zeroed dashboard structure when the DB is unavailable or empty."""
    return {
        "looks_shown": 0,
        "add_the_look_rate": 0.0,
        "add_single_rate": 0.0,
        "basket_size": {
            "look_avg_inr": None,
            "single_avg_inr": None,
            "delta_inr": None,
            "caveat": _BASKET_CAVEAT,
        },
        "top_pairings": [],
        "by_occasion": [],
        "by_brand": [],
    }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_dashboard(engine: Any) -> dict[str, Any]:
    """Run all dashboard queries in a single connection and return the metrics dict.

    All queries use FILTER (WHERE ...) aggregates so we touch styling_events only
    once per segmentation query.  An empty table returns zeros — no exceptions.

    Args:
        engine: A SQLAlchemy engine (sync) with an active connection pool.

    Returns:
        Populated dashboard dict.
    """
    with engine.connect() as conn:
        # ------------------------------------------------------------------
        # Overall totals
        # ------------------------------------------------------------------
        looks_shown: int = conn.execute(
            text(
                "SELECT COUNT(DISTINCT look_id) AS cnt "
                "FROM styling_events WHERE event_type = 'look_shown'"
            )
        ).scalar_one() or 0

        atl_count: int = conn.execute(
            text("SELECT COUNT(*) AS cnt FROM styling_events WHERE event_type = 'add_the_look'")
        ).scalar_one() or 0

        single_count: int = conn.execute(
            text("SELECT COUNT(*) AS cnt FROM styling_events WHERE event_type = 'add_single'")
        ).scalar_one() or 0

        basket_look_raw = conn.execute(
            text(
                "SELECT AVG(look_total_inr) AS avg_val FROM styling_events "
                "WHERE event_type = 'add_the_look' AND look_total_inr IS NOT NULL"
            )
        ).scalar_one()
        basket_look: float | None = float(basket_look_raw) if basket_look_raw is not None else None

        basket_single_raw = conn.execute(
            text(
                "SELECT AVG(look_total_inr) AS avg_val FROM styling_events "
                "WHERE event_type = 'add_single' AND look_total_inr IS NOT NULL"
            )
        ).scalar_one()
        basket_single: float | None = (
            float(basket_single_raw) if basket_single_raw is not None else None
        )

        # ------------------------------------------------------------------
        # Top pairings
        # ------------------------------------------------------------------
        pairing_rows = conn.execute(
            text(
                """
                SELECT anchor_category, fill_category, occasion, add_the_look,
                       (add_the_look + thumbs_up + thumbs_down + add_single_only) AS total_signals
                FROM pairing_stats
                WHERE (add_the_look + thumbs_up + thumbs_down + add_single_only) >= :min_signals
                ORDER BY add_the_look DESC
                LIMIT 5
                """
            ),
            {"min_signals": _MIN_PAIRING_SIGNALS},
        ).fetchall()

        top_pairings = [
            {
                "anchor_category": row.anchor_category,
                "fill_category": row.fill_category,
                "occasion": row.occasion,
                "add_the_look": row.add_the_look,
                "total_signals": row.total_signals,
            }
            for row in pairing_rows
        ]

        # ------------------------------------------------------------------
        # By occasion
        # ------------------------------------------------------------------
        occasion_rows = conn.execute(
            text(
                """
                SELECT occasion,
                       COUNT(DISTINCT look_id) FILTER (WHERE event_type = 'look_shown')
                           AS looks_shown,
                       COUNT(*) FILTER (WHERE event_type = 'add_the_look')
                           AS add_the_look,
                       COUNT(*) FILTER (WHERE event_type = 'add_single')
                           AS add_single,
                       AVG(look_total_inr) FILTER (
                           WHERE event_type = 'add_the_look' AND look_total_inr IS NOT NULL
                       ) AS basket_look,
                       AVG(look_total_inr) FILTER (
                           WHERE event_type = 'add_single' AND look_total_inr IS NOT NULL
                       ) AS basket_single
                FROM styling_events
                WHERE occasion IS NOT NULL
                GROUP BY occasion
                ORDER BY looks_shown DESC
                """
            )
        ).fetchall()

        by_occasion = [
            {
                "occasion": row.occasion,
                "looks_shown": row.looks_shown or 0,
                "add_the_look_rate": _safe_rate(
                    row.add_the_look or 0, row.looks_shown or 0
                ),
                "basket_delta_inr": _basket_delta(
                    float(row.basket_look) if row.basket_look is not None else None,
                    float(row.basket_single) if row.basket_single is not None else None,
                ),
            }
            for row in occasion_rows
        ]

        # ------------------------------------------------------------------
        # By brand
        # ------------------------------------------------------------------
        brand_rows = conn.execute(
            text(
                """
                SELECT brand,
                       COUNT(DISTINCT look_id) FILTER (WHERE event_type = 'look_shown')
                           AS looks_shown,
                       COUNT(*) FILTER (WHERE event_type = 'add_the_look')
                           AS add_the_look,
                       COUNT(*) FILTER (WHERE event_type = 'add_single')
                           AS add_single,
                       AVG(look_total_inr) FILTER (
                           WHERE event_type = 'add_the_look' AND look_total_inr IS NOT NULL
                       ) AS basket_look,
                       AVG(look_total_inr) FILTER (
                           WHERE event_type = 'add_single' AND look_total_inr IS NOT NULL
                       ) AS basket_single
                FROM styling_events
                WHERE brand IS NOT NULL
                GROUP BY brand
                ORDER BY looks_shown DESC
                """
            )
        ).fetchall()

        by_brand = [
            {
                "brand": row.brand,
                "looks_shown": row.looks_shown or 0,
                "add_the_look_rate": _safe_rate(
                    row.add_the_look or 0, row.looks_shown or 0
                ),
                "basket_delta_inr": _basket_delta(
                    float(row.basket_look) if row.basket_look is not None else None,
                    float(row.basket_single) if row.basket_single is not None else None,
                ),
            }
            for row in brand_rows
        ]

    return {
        "looks_shown": looks_shown,
        "add_the_look_rate": _safe_rate(atl_count, looks_shown),
        "add_single_rate": _safe_rate(single_count, looks_shown),
        "basket_size": {
            "look_avg_inr": round(basket_look, 2) if basket_look is not None else None,
            "single_avg_inr": round(basket_single, 2) if basket_single is not None else None,
            "delta_inr": _basket_delta(basket_look, basket_single),
            "caveat": _BASKET_CAVEAT,
        },
        "top_pairings": top_pairings,
        "by_occasion": by_occasion,
        "by_brand": by_brand,
    }


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get("")
async def get_dashboard(request: Request) -> dict[str, Any]:
    """Return brand dashboard metrics from flywheel event data.

    Reads from styling_events and pairing_stats tables.  Returns zeros and
    empty lists when no data exists yet — never raises a 5xx.

    No authentication required: all returned values are aggregate metrics,
    not personally identifiable data.
    """
    engine = get_db_engine()

    if engine is None:
        logger.warning("dashboard: no DB engine available, returning empty metrics")
        return _empty_dashboard()

    try:
        return _compute_dashboard(engine)
    except Exception:
        logger.exception("dashboard: failed to compute metrics")
        return _empty_dashboard()
