"""Feedback route: POST /messages/{message_id}/feedback.

Allows users to submit a thumbs-up (1) or thumbs-down (-1) rating for an
assistant message.  Uses UPSERT so re-rating the same message updates the
existing row rather than creating a duplicate.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from api.auth import get_current_user_id

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback"])

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    rating: int = Field(..., description="1 for thumbs-up, -1 for thumbs-down")
    comment: str | None = Field(default=None, max_length=2000)

    model_config = {"json_schema_extra": {"examples": [{"rating": 1, "comment": "Very helpful!"}]}}


# ---------------------------------------------------------------------------
# DB engine (lazy, module-level singleton)
# ---------------------------------------------------------------------------

_engine: Any = None


def _get_engine() -> Any:
    """Return (and lazily create) a SQLAlchemy engine from DATABASE_URL.

    Mirrors the pattern in api.main._build_session_store so both paths use
    the same driver normalization logic.  Raises HTTPException(503) when
    DATABASE_URL is not configured — feedback requires persistent storage.
    """
    global _engine
    if _engine is not None:
        return _engine

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise HTTPException(
            status_code=503,
            detail="Feedback requires a database. DATABASE_URL is not configured.",
        )

    for prefix in ("postgresql://", "postgres://"):
        if db_url.startswith(prefix):
            db_url = "postgresql+psycopg://" + db_url[len(prefix):]
            break

    _engine = create_engine(db_url, pool_pre_ping=True, pool_size=2, max_overflow=1)
    return _engine


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/messages/{message_id}/feedback", status_code=204, response_class=Response)
def post_feedback(
    message_id: str,
    body: FeedbackRequest,
    user_id: str = Depends(get_current_user_id),
) -> None:
    """Submit or update a thumbs-up/thumbs-down rating for an assistant message.

    - Verifies the message exists (404 otherwise).
    - UPSERTs into the feedback table — calling again with a different rating
      updates the existing row.
    - Returns 204 No Content on success.
    - Returns 503 when the server is running without a database.

    Args:
        message_id: UUID of the message being rated.
        body:        Rating (1 or -1) and optional free-text comment.
        user_id:     Injected by get_current_user_id (Bearer JWT).
    """
    if body.rating not in (1, -1):
        raise HTTPException(
            status_code=422,
            detail="rating must be 1 (thumbs-up) or -1 (thumbs-down)",
        )

    engine = _get_engine()

    try:
        with engine.begin() as conn:
            # Verify message exists — prevents ghost feedback rows and gives a
            # clean 404 rather than a DB constraint violation.
            row = conn.execute(
                text("SELECT id FROM messages WHERE id = CAST(:mid AS uuid)"),
                {"mid": message_id},
            ).fetchone()
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Message {message_id!r} not found",
                )

            # UPSERT: re-rating overwrites rating + comment in place.
            conn.execute(
                text(
                    """
                    INSERT INTO feedback (message_id, rating, comment)
                    VALUES (CAST(:mid AS uuid), :rating, :comment)
                    ON CONFLICT (message_id)
                    DO UPDATE SET
                        rating  = EXCLUDED.rating,
                        comment = EXCLUDED.comment
                    """
                ),
                {"mid": message_id, "rating": body.rating, "comment": body.comment},
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("feedback upsert failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    logger.info(
        "feedback recorded",
        extra={"message_id": message_id, "rating": body.rating, "user_id": user_id},
    )
