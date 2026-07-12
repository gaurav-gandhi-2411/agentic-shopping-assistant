from __future__ import annotations

"""Styling event ingestion endpoint.

POST /events — append-only look-interaction logging for the conversion-data
flywheel.  No PUT, PATCH, or DELETE methods exist on this resource; the
styling_events table is intentionally append-only.
"""

import logging  # noqa: E402

from fastapi import APIRouter, HTTPException, Request  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.flywheel.events import StylingEvent, log_event  # noqa: E402
from src.flywheel.events import price_band as compute_price_band  # noqa: E402

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class EventRequest(BaseModel):
    event_type: str = Field(..., description="One of: look_shown, item_view, add_single, "
                             "add_the_look, swap_slot, thumbs_up, thumbs_down")
    session_id: str
    look_id: str
    anchor_item_id: str
    anchor_category: str
    user_id: str | None = None
    filled_slots: list[dict] | None = None
    occasion: str | None = None
    brand: str | None = None
    look_total_inr: int | None = None
    metadata: dict | None = None


class EventResponse(BaseModel):
    ok: bool


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("", response_model=EventResponse)
async def post_event(req: EventRequest, request: Request) -> EventResponse:
    """Log a styling interaction event. Append-only — no updates.

    Silently succeeds (ok=True) when no DB is configured (dev mode without
    DATABASE_URL) so the frontend is never blocked by missing telemetry.

    Args:
        req:     Validated EventRequest body.
        request: Raw FastAPI request — used to access app.state.engine.

    Returns:
        EventResponse(ok=True) on success or when DB is unavailable.

    Raises:
        HTTPException 422: Unknown event_type value.
    """
    # Retrieve engine from app.state (set by _init in deps) to avoid circular
    # import with api.deps at module load time.
    engine = getattr(request.app.state, "engine", None)

    # Also fall back to the deps module-level singleton for compatibility
    # with the existing lifespan wiring (deps._db_engine, not app.state.engine).
    if engine is None:
        import api.deps as deps
        engine = deps.get_db_engine()

    if engine is None:
        # No DB configured (dev mode) — silently drop the event.
        logger.debug("flywheel: no DB engine; dropping event %s", req.event_type)
        return EventResponse(ok=True)

    band = compute_price_band(req.look_total_inr)
    try:
        event = StylingEvent(
            event_type=req.event_type,
            session_id=req.session_id,
            look_id=req.look_id,
            anchor_item_id=req.anchor_item_id,
            anchor_category=req.anchor_category,
            user_id=req.user_id,
            filled_slots=req.filled_slots,
            occasion=req.occasion,
            price_band=band,
            brand=req.brand,
            look_total_inr=req.look_total_inr,
            metadata=req.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    log_event(engine, event)
    return EventResponse(ok=True)
