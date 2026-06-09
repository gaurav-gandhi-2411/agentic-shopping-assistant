from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

VALID_EVENT_TYPES: frozenset[str] = frozenset({
    "look_shown", "item_view", "add_single", "add_the_look",
    "swap_slot", "thumbs_up", "thumbs_down",
})

# (upper_bound_exclusive, band_label) — checked in order; first match wins.
PRICE_BANDS: list[tuple[float, str]] = [
    (1000.0,  "budget_0_1k"),
    (3000.0,  "budget_1k_3k"),
    (5000.0,  "budget_3k_5k"),
    (float("inf"), "budget_5k_plus"),
]


def price_band(price_inr: float | None) -> str | None:
    """Map a price in INR to a budget band label, or None if price is None.

    Args:
        price_inr: Item or look price in Indian Rupees, or None.

    Returns:
        A string like "budget_1k_3k", or None when input is None.
    """
    if price_inr is None:
        return None
    for threshold, band in PRICE_BANDS:
        if price_inr < threshold:
            return band
    return "budget_5k_plus"


@dataclass
class StylingEvent:
    """Value object for a single look-interaction event.

    Validated on construction — raises ValueError for unknown event_type.
    All fields match the styling_events DB schema columns exactly.
    """

    event_type: str
    session_id: str
    look_id: str
    anchor_item_id: str
    anchor_category: str
    user_id: str | None = None
    filled_slots: list[dict] | None = None
    occasion: str | None = None
    price_band: str | None = None
    brand: str | None = None
    look_total_inr: int | None = None
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if self.event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"Unknown event_type: {self.event_type!r}")


def log_event(engine: Any, event: StylingEvent) -> None:
    """Insert one StylingEvent into styling_events. Append-only — never UPDATE.

    Swallows all DB exceptions so a telemetry failure never crashes the
    request path. Exceptions are logged at ERROR level for alerting.

    Args:
        engine: SQLAlchemy engine with an active pool.
        event:  A validated StylingEvent dataclass instance.
    """
    import sqlalchemy as sa

    row = {
        "event_type":      event.event_type,
        "session_id":      event.session_id,
        "user_id":         event.user_id,
        "look_id":         event.look_id,
        "anchor_item_id":  event.anchor_item_id,
        "anchor_category": event.anchor_category,
        "filled_slots":    json.dumps(event.filled_slots) if event.filled_slots is not None else None,
        "occasion":        event.occasion,
        "price_band":      event.price_band,
        "brand":           event.brand,
        "look_total_inr":  event.look_total_inr,
        "metadata":        json.dumps(event.metadata) if event.metadata is not None else None,
    }
    try:
        with engine.begin() as conn:
            conn.execute(
                sa.text(
                    """
                    INSERT INTO styling_events
                      (event_type, session_id, user_id, look_id, anchor_item_id,
                       anchor_category, filled_slots, occasion, price_band, brand,
                       look_total_inr, metadata)
                    VALUES
                      (:event_type, :session_id, :user_id, :look_id, :anchor_item_id,
                       :anchor_category, :filled_slots::jsonb, :occasion, :price_band, :brand,
                       :look_total_inr, :metadata::jsonb)
                    """
                ),
                row,
            )
    except Exception:
        # Never let telemetry failure crash the request path.
        logger.exception(
            "flywheel: failed to log event %s look_id=%s",
            event.event_type,
            event.look_id,
        )
