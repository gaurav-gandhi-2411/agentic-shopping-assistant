"""Saved looks endpoints.

POST /looks  — save the current outfit board for this session; returns a share path.
GET  /looks/{look_id} — public, no auth: return the saved board for the shared-look view.

Neither endpoint requires authentication: saving is anonymous-session-scoped
(session_id from the request body) and reading is fully public.  The look UUID
is unguessable (gen_random_uuid) so it acts as an access-capability token.

If no DB engine is configured (dev mode without DATABASE_URL) POST returns 503
rather than pretending to succeed — saving must actually persist.  GET returns
404 when the id is not found.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from api.schemas import SaveLookRequest, SaveLookResponse, SharedLookResponse
from src.storage.saved_looks import get_look, get_look_memory, save_look, save_look_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/looks", tags=["looks"])


def _get_engine(request: Request):  # type: ignore[return]
    """Resolve the SQLAlchemy engine from app state or api.deps.

    Mirrors the pattern used in api/routes/events.py: try app.state.engine
    first (set by the lifespan wiring), then fall back to the api.deps
    module-level singleton for compatibility with existing startup code.

    Returns None when no DB is configured (dev / no DATABASE_URL).
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        import api.deps as deps

        engine = deps.get_db_engine()
    return engine


@router.post("", response_model=SaveLookResponse, status_code=201)
async def post_look(body: SaveLookRequest, request: Request) -> SaveLookResponse:
    """Persist an outfit board and return its UUID share slug.

    The returned ``share_path`` (/look/{id}) is what the frontend embeds in
    a share link.  The GET /looks/{look_id} endpoint renders that path.

    Args:
        body:    Validated SaveLookRequest — the full outfit board snapshot.
        request: FastAPI request — used to resolve the DB engine.

    Returns:
        SaveLookResponse with id (UUID) and share_path (/look/{id}).

    Raises:
        HTTPException 503: No database configured — saving cannot persist.
        HTTPException 500: Unexpected DB error during insert.
    """
    engine = _get_engine(request)
    if engine is None:
        # Dev fallback: use in-memory store (persists for process lifetime only)
        logger.info("looks: no DB engine; using in-memory fallback for local dev")
        try:
            new_id = save_look_memory(
                session_id=body.session_id,
                user_id=body.user_id,
                brand=body.brand,
                look_id=body.look_id,
                occasion=body.occasion,
                look_gender=body.look_gender,
                anchor_item_id=body.anchor_item_id,
                look_total_inr=body.look_total_inr,
                snapshot=body.snapshot,
            )
        except Exception as exc:
            logger.error("looks: in-memory save failed: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to save look") from exc
        logger.info("looks: memory-saved look id=%s session=%s", new_id, body.session_id)
        return SaveLookResponse(id=new_id, share_path=f"/look/{new_id}")

    try:
        new_id = save_look(
            engine,
            session_id=body.session_id,
            user_id=body.user_id,
            brand=body.brand,
            look_id=body.look_id,
            occasion=body.occasion,
            look_gender=body.look_gender,
            anchor_item_id=body.anchor_item_id,
            look_total_inr=body.look_total_inr,
            snapshot=body.snapshot,
        )
    except Exception as exc:
        logger.error("looks: insert failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save look") from exc

    logger.info("looks: saved look id=%s session=%s", new_id, body.session_id)
    return SaveLookResponse(id=new_id, share_path=f"/look/{new_id}")


@router.get("/{look_id}", response_model=SharedLookResponse)
async def get_look_by_id(look_id: str, request: Request) -> SharedLookResponse:
    """Return the saved look for the public read-only shared board.

    No authentication required — the unguessable UUID acts as a capability
    token.  Returns 404 for unknown or invalid ids.

    Args:
        look_id: UUID string — the share slug embedded in the share link.
        request: FastAPI request — used to resolve the DB engine.

    Returns:
        SharedLookResponse with board metadata and the self-contained snapshot.

    Raises:
        HTTPException 404: Look not found or id is not a valid UUID.
        HTTPException 503: No database configured.
        HTTPException 500: Unexpected DB error during select.
    """
    engine = _get_engine(request)
    if engine is None:
        # Dev fallback: check in-memory store
        record = get_look_memory(look_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Look {look_id!r} not found")
        return SharedLookResponse(
            id=record["id"],
            brand=record["brand"],
            occasion=record["occasion"],
            look_gender=record["look_gender"],
            look_total_inr=record["look_total_inr"],
            snapshot=record["snapshot"],
            created_at=record["created_at"],
        )

    try:
        record = get_look(engine, look_id)
    except Exception as exc:
        logger.error("looks: select failed look_id=%s: %s", look_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve look") from exc

    if record is None:
        raise HTTPException(status_code=404, detail=f"Look {look_id!r} not found")

    return SharedLookResponse(
        id=record["id"],
        brand=record["brand"],
        occasion=record["occasion"],
        look_gender=record["look_gender"],
        look_total_inr=record["look_total_inr"],
        snapshot=record["snapshot"],
        created_at=record["created_at"],
    )
