"""Styling event ingestion endpoint.

POST /events — append-only look-interaction logging for the conversion-data
flywheel.  No PUT, PATCH, or DELETE methods exist on this resource; the
styling_events table is intentionally append-only.

Hardening (this endpoint carries the analytics funnel, so it is a public write path):
  * authenticated with the same dependency as the chat routes (Supabase JWT or demo token); the
    server-side identity is what gets stored, a client-supplied user_id is ignored;
  * per-identity and per-IP in-memory sliding-window rate limits, separate from the chat limiters;
  * every field is bounded (see EventRequest) and the raw body is capped at MAX_BODY_BYTES;
  * the DB write runs in the threadpool (plain ``def`` endpoint), never on the event loop;
  * errors are generic: no internal exception text and no echo of submitted values.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator
from starlette.requests import Request as StarletteRequest

from api.auth import get_current_user_id_or_demo
from api.demo.client_ip import get_client_ip
from api.demo.fallback_limiter import FallbackLimiter
from src.flywheel.events import VALID_EVENT_TYPES, StylingEvent, log_event
from src.flywheel.events import price_band as compute_price_band

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_BODY_BYTES = 16 * 1024  # raw request body, enforced before parsing
MAX_PAYLOAD_JSON_BYTES = 8 * 1024  # validated payload re-serialised
MAX_METADATA_JSON_BYTES = 2 * 1024
MAX_ID_CHARS = 128  # session_id / look_id / anchor_item_id
MAX_LABEL_CHARS = 64  # occasion / brand / anchor_category / dict keys
MAX_SLOTS = 20
MAX_DICT_KEYS = 16
MAX_SCALAR_STR_CHARS = 128
MAX_LOOK_TOTAL_INR = 10_000_000

_RATE_WINDOW_S = 60.0
_DEFAULT_PER_IDENTITY_PER_MIN = 120
_DEFAULT_PER_IP_PER_MIN = 300
_DEFAULT_MAX_KEYS = 10_000  # LRU bound on tracked identities / IPs (memory cap)


def _env_int(name: str, default: int) -> int:
    """Read a positive int from the process environment, falling back on junk."""
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _per_identity_limit() -> int:
    return _env_int("EVENTS_RATE_LIMIT_PER_IDENTITY_PER_MIN", _DEFAULT_PER_IDENTITY_PER_MIN)


def _per_ip_limit() -> int:
    return _env_int("EVENTS_RATE_LIMIT_PER_IP_PER_MIN", _DEFAULT_PER_IP_PER_MIN)


def _max_keys() -> int:
    return _env_int("EVENTS_RATE_LIMIT_MAX_KEYS", _DEFAULT_MAX_KEYS)


def _no_daily_cap() -> int:
    # FallbackLimiter also offers a daily cap; /events only uses its sliding window.
    return 2**31


# Dedicated instances: events must never consume the chat per-IP/per-user quota.
_identity_limiter = FallbackLimiter(
    _per_identity_limit, _no_daily_cap, _max_keys, window_s=_RATE_WINDOW_S
)
_ip_limiter = FallbackLimiter(_per_ip_limit, _no_daily_cap, _max_keys, window_s=_RATE_WINDOW_S)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

_FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Scalar = str | int | _FiniteFloat | bool | None


def _check_flat_dict(value: dict[str, Scalar] | None) -> None:
    """Bound a flat dict: key count, key length and string-value length."""
    if value is None:
        return
    if len(value) > MAX_DICT_KEYS:
        raise ValueError("too many keys")
    for key, item in value.items():
        if len(key) > MAX_LABEL_CHARS:
            raise ValueError("key too long")
        if isinstance(item, str) and len(item) > MAX_SCALAR_STR_CHARS:
            raise ValueError("value too long")


class EventRequest(BaseModel):
    # Unknown fields are IGNORED (not forbidden) on purpose: the live frontend sends extra
    # top-level keys on variant_selected (variant_id, variant_label) and must keep working.
    # They are not stored. The raw body cap bounds what an attacker can pad in.
    # NOTE: user_id is deliberately not a field: identity comes from the auth token.
    event_type: str = Field(..., description="One of src.flywheel.events.EVENT_TYPES")
    session_id: str = Field(..., max_length=MAX_ID_CHARS)
    look_id: str = Field(..., max_length=MAX_ID_CHARS)
    anchor_item_id: str = Field(..., max_length=MAX_ID_CHARS)
    anchor_category: str = Field(..., max_length=MAX_LABEL_CHARS)
    filled_slots: list[dict[str, Scalar]] | None = Field(default=None, max_length=MAX_SLOTS)
    occasion: str | None = Field(default=None, max_length=MAX_LABEL_CHARS)
    brand: str | None = Field(default=None, max_length=MAX_LABEL_CHARS)
    # StrictInt: lax int mode coerces JSON `true` to 1 and would store it.
    look_total_inr: StrictInt | None = Field(default=None, ge=0, le=MAX_LOOK_TOTAL_INR)
    metadata: dict[str, Scalar] | None = None

    @field_validator("event_type")
    @classmethod
    def _known_event_type(cls, value: str) -> str:
        if value not in VALID_EVENT_TYPES:
            raise ValueError("unknown event_type")
        return value

    @field_validator("filled_slots")
    @classmethod
    def _bound_slots(cls, value: list[dict[str, Scalar]] | None) -> list[dict[str, Scalar]] | None:
        for slot in value or []:
            _check_flat_dict(slot)
        return value

    @field_validator("metadata")
    @classmethod
    def _bound_metadata(cls, value: dict[str, Scalar] | None) -> dict[str, Scalar] | None:
        _check_flat_dict(value)
        if value is not None and len(json.dumps(value)) > MAX_METADATA_JSON_BYTES:
            raise ValueError("metadata too large")
        return value

    @model_validator(mode="after")
    def _bound_total_size(self) -> EventRequest:
        if len(self.model_dump_json()) > MAX_PAYLOAD_JSON_BYTES:
            raise ValueError("event payload too large")
        return self


class EventResponse(BaseModel):
    ok: bool


# ---------------------------------------------------------------------------
# Body cap + sanitised 422 (custom route so it runs BEFORE FastAPI parses the body)
# ---------------------------------------------------------------------------


def _too_large() -> JSONResponse:
    return JSONResponse(status_code=413, content={"detail": "Request body too large"})


def _sanitised_422(exc: RequestValidationError) -> JSONResponse:
    """422 with loc/msg/type only: pydantic's default also echoes the submitted ``input``."""
    errors = [
        {
            "loc": [str(part)[:MAX_LABEL_CHARS] for part in err.get("loc", ())],
            "msg": str(err.get("msg", "Invalid value"))[:200],
            "type": str(err.get("type", "value_error")),
        }
        for err in exc.errors()[:10]
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


class _BoundedEventsRoute(APIRoute):
    def get_route_handler(self) -> Callable[[StarletteRequest], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: StarletteRequest) -> Response:
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    if int(declared) > MAX_BODY_BYTES:
                        return _too_large()
                except ValueError:
                    return JSONResponse(
                        status_code=400, content={"detail": "Invalid Content-Length"}
                    )
            # Missing (chunked) or understated Content-Length: count real bytes as they
            # arrive and stop at the cap, so memory stays bounded either way.
            chunks: list[bytes] = []
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_BODY_BYTES:
                    return _too_large()
                chunks.append(chunk)
            body = b"".join(chunks)

            async def receive() -> dict[str, Any]:
                return {"type": "http.request", "body": body, "more_body": False}

            try:
                return await original(StarletteRequest(request.scope, receive))
            except RequestValidationError as exc:
                return _sanitised_422(exc)

        return handler


router = APIRouter(prefix="/events", tags=["events"], route_class=_BoundedEventsRoute)


# ---------------------------------------------------------------------------
# Dependencies: IP limit (before auth, so unauthenticated floods are throttled too),
# then auth, then the per-identity limit.
# ---------------------------------------------------------------------------


def _reject_rate_limited(retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail="Too many events",
        headers={"Retry-After": str(retry_after)},
    )


def _limit_ip(request: Request) -> None:
    allowed, retry_after = _ip_limiter.check_ip(get_client_ip(request))
    if not allowed:
        raise _reject_rate_limited(retry_after)


def _authenticated_identity(user_id: str = Depends(get_current_user_id_or_demo)) -> str:
    allowed, retry_after = _identity_limiter.check_ip(user_id)
    if not allowed:
        raise _reject_rate_limited(retry_after)
    return user_id


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("", response_model=EventResponse, dependencies=[Depends(_limit_ip)])
def post_event(
    req: EventRequest,
    request: Request,
    user_id: str = Depends(_authenticated_identity),
) -> EventResponse:
    """Log a styling interaction event. Append-only — no updates.

    A plain ``def`` so FastAPI runs it in the threadpool: the DB write is synchronous and
    must not block the event loop.

    Silently succeeds (ok=True) when no DB is configured (dev mode without
    DATABASE_URL) so the frontend is never blocked by missing telemetry.

    Args:
        req:     Validated, bounded EventRequest body.
        request: Raw FastAPI request — used to access app.state.engine.
        user_id: Authenticated identity (JWT sub or demo anon id); stored as the event's
                 user_id. Any client-supplied user_id is ignored.

    Returns:
        EventResponse(ok=True) on success or when DB is unavailable.

    Raises:
        HTTPException 401: Missing/invalid credential.
        HTTPException 429: Rate limit exceeded (Retry-After set).
        HTTPException 503: The event could not be stored (generic body; the frontend ignores
            the response, and a 5xx keeps the failure visible to monitoring).
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

    event = StylingEvent(
        event_type=req.event_type,
        session_id=req.session_id,
        look_id=req.look_id,
        anchor_item_id=req.anchor_item_id,
        anchor_category=req.anchor_category,
        user_id=user_id,
        filled_slots=req.filled_slots,
        occasion=req.occasion,
        price_band=compute_price_band(req.look_total_inr),
        brand=req.brand,
        look_total_inr=req.look_total_inr,
        metadata=req.metadata,
    )
    try:
        stored = log_event(engine, event)
    except Exception:
        logger.exception("flywheel: unexpected failure logging event %s", req.event_type)
        stored = False
    if not stored:
        raise HTTPException(status_code=503, detail="Event logging unavailable")
    return EventResponse(ok=True)
