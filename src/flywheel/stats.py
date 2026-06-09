from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# total_signals = add_the_look + thumbs_up + thumbs_down + add_single_only
# positive_rate = (add_the_look + thumbs_up) / total_signals
# This lets thumbs_down AND add_single (bought anchor but skipped the look) drag the boost.
# Documented in docs/architecture/adr/0005-flywheel-ranking-blend.md.

FLYWHEEL_ALPHA: float = 0.25   # max +25% score boost from conversion signal
FLYWHEEL_MIN_SIGNALS: int = 10  # cold-start threshold: no boost below this count


@dataclass
class PairingStat:
    """Aggregated conversion signal for a (anchor_category, fill_category, occasion) triple.

    positive_rate intentionally penalises add_single (user bought anchor but skipped
    the look), not just explicit thumbs_down.  See ADR 0005.
    """

    anchor_category: str
    fill_category: str
    occasion: str
    looks_shown: int = 0
    add_the_look: int = 0
    thumbs_up: int = 0
    thumbs_down: int = 0
    add_single_only: int = 0

    @property
    def total_signals(self) -> int:
        """Sum of all signals that carry conversion information."""
        return self.add_the_look + self.thumbs_up + self.thumbs_down + self.add_single_only

    @property
    def positive_rate(self) -> float | None:
        """Fraction of signals that are positive; None when no signals recorded yet."""
        t = self.total_signals
        if t == 0:
            return None
        return (self.add_the_look + self.thumbs_up) / t

    def boost(self) -> float:
        """Transparent flywheel boost: 0.0 at cold-start; max FLYWHEEL_ALPHA at positive_rate=1.

        Returns:
            A multiplier delta in [0.0, FLYWHEEL_ALPHA].  Caller applies as:
              final_score = coherence_score * (1 + boost())
        """
        if self.total_signals < FLYWHEEL_MIN_SIGNALS:
            return 0.0
        pr = self.positive_rate or 0.0
        return FLYWHEEL_ALPHA * pr


def load_pairing_stats(
    engine: Any,
    occasion: str | None = None,
) -> dict[tuple[str, str, str], PairingStat]:
    """Load pairing stats from DB as a dict keyed by (anchor_cat, fill_cat, occasion).

    Optionally filter by occasion for faster lookups during outfit composition.
    Returns empty dict if table is empty or DB is unavailable — callers must
    handle the cold-start case (boost = 0.0 for any unknown pairing).

    Args:
        engine:  SQLAlchemy engine with an active pool.
        occasion: If provided, only rows matching this occasion are loaded.

    Returns:
        Dict mapping (anchor_category, fill_category, occasion) to PairingStat.
    """
    import sqlalchemy as sa

    query = (
        "SELECT anchor_category, fill_category, occasion, looks_shown, "
        "add_the_look, thumbs_up, thumbs_down, add_single_only "
        "FROM pairing_stats"
    )
    params: dict[str, str] = {}
    if occasion:
        query += " WHERE occasion = :occasion"
        params["occasion"] = occasion

    try:
        with engine.connect() as conn:
            rows = conn.execute(sa.text(query), params).fetchall()
        return {
            (r.anchor_category, r.fill_category, r.occasion): PairingStat(
                anchor_category=r.anchor_category,
                fill_category=r.fill_category,
                occasion=r.occasion,
                looks_shown=r.looks_shown,
                add_the_look=r.add_the_look,
                thumbs_up=r.thumbs_up,
                thumbs_down=r.thumbs_down,
                add_single_only=r.add_single_only,
            )
            for r in rows
        }
    except Exception:
        logger.exception("flywheel: failed to load pairing_stats")
        return {}


def refresh_pairing_stats(engine: Any) -> int:
    """Recompute pairing_stats from styling_events. Returns number of rows upserted.

    Called after every N new events (application-level trigger, not a DB job).
    Uses INSERT ... ON CONFLICT UPDATE to keep existing rows and add new ones.

    Note on fill_category: the current query uses 'outfit_slot' as a placeholder
    because styling_events does not yet emit slot-level data. When add_the_look
    events carry slot-level filled_slots data, this query is upgraded to unnest
    the JSONB array and derive fill_category per slot.

    Args:
        engine: SQLAlchemy engine with an active pool.

    Returns:
        Number of rows written (0 on failure, which is logged but not re-raised).
    """
    import sqlalchemy as sa

    upsert_sql = """
        INSERT INTO pairing_stats (anchor_category, fill_category, occasion,
            looks_shown, add_the_look, thumbs_up, thumbs_down, add_single_only, updated_at)
        SELECT
            anchor_category,
            -- fill_category: derive from each slot in filled_slots JSONB array.
            -- For now, use 'outfit_slot' as a placeholder until slot-level events are emitted.
            -- When add_the_look events carry slot-level data, this query is upgraded.
            'outfit_slot'                                AS fill_category,
            COALESCE(occasion, 'casual')                 AS occasion,
            COUNT(*) FILTER (WHERE event_type = 'look_shown')   AS looks_shown,
            COUNT(*) FILTER (WHERE event_type = 'add_the_look') AS add_the_look,
            COUNT(*) FILTER (WHERE event_type = 'thumbs_up')    AS thumbs_up,
            COUNT(*) FILTER (WHERE event_type = 'thumbs_down')  AS thumbs_down,
            COUNT(*) FILTER (WHERE event_type = 'add_single')   AS add_single_only,
            NOW()                                        AS updated_at
        FROM styling_events
        GROUP BY anchor_category, COALESCE(occasion, 'casual')
        ON CONFLICT (anchor_category, fill_category, occasion)
        DO UPDATE SET
            looks_shown     = EXCLUDED.looks_shown,
            add_the_look    = EXCLUDED.add_the_look,
            thumbs_up       = EXCLUDED.thumbs_up,
            thumbs_down     = EXCLUDED.thumbs_down,
            add_single_only = EXCLUDED.add_single_only,
            updated_at      = EXCLUDED.updated_at
    """
    try:
        with engine.begin() as conn:
            result = conn.execute(sa.text(upsert_sql))
            return result.rowcount
    except Exception:
        logger.exception("flywheel: failed to refresh pairing_stats")
        return 0
