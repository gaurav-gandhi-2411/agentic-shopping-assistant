"""Saved-looks persistence layer.

Insert-only: save_look() appends a row; get_look() reads it back.  No UPDATE
or DELETE paths exist — a saved look is immutable once written.  The row id
(UUID) doubles as the public share slug so no secondary lookup table is needed.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def save_look(
    engine: Any,
    *,
    session_id: str,
    user_id: str | None,
    brand: str,
    look_id: str | None,
    occasion: str | None,
    look_gender: str | None,
    anchor_item_id: str | None,
    look_total_inr: int | None,
    snapshot: dict,
) -> str:
    """Insert one saved-look row and return its UUID as a string.

    The returned UUID is the share slug — pass it to get_look() to retrieve
    the saved data, or embed it in /look/{id} for the read-only shared board.

    Args:
        engine:         SQLAlchemy engine with an active pool.
        session_id:     Anonymous demo session identifier (from the frontend).
        user_id:        Authenticated user UUID, or None for anonymous sessions.
        brand:          Brand slug (e.g. "hm", "myntra").
        look_id:        Ephemeral look identifier from the outfit engine, or None.
        occasion:       Occasion label (e.g. "casual", "formal"), or None.
        look_gender:    Gender label ("men", "women", "unisex"), or None.
        anchor_item_id: Anchor catalogue item id, or None.
        look_total_inr: Basket total in INR, or None.
        snapshot:       Self-contained board payload (items[], rationale, etc.).

    Returns:
        The UUID string of the newly inserted row (also the share slug).

    Raises:
        Exception: Any SQLAlchemy error propagates to the caller — the API
            layer decides whether to swallow or surface it.
    """
    import sqlalchemy as sa

    with engine.begin() as conn:
        row = conn.execute(
            sa.text(
                """
                INSERT INTO saved_looks
                    (session_id, user_id, brand, look_id, occasion,
                     look_gender, anchor_item_id, look_total_inr, snapshot)
                VALUES
                    (:session_id,
                     CAST(:user_id AS uuid),
                     :brand,
                     :look_id,
                     :occasion,
                     :look_gender,
                     :anchor_item_id,
                     :look_total_inr,
                     CAST(:snapshot AS jsonb))
                RETURNING id::text
                """
            ),
            {
                "session_id": session_id,
                "user_id": user_id,
                "brand": brand,
                "look_id": look_id,
                "occasion": occasion,
                "look_gender": look_gender,
                "anchor_item_id": anchor_item_id,
                "look_total_inr": look_total_inr,
                "snapshot": json.dumps(snapshot),
            },
        )
        new_id: str = row.fetchone()[0]
    return new_id


def get_look(engine: Any, look_id: str) -> dict | None:
    """Fetch a saved look by its UUID share slug.

    Returns a dict of all columns (snapshot already parsed to a Python dict),
    or None when the id does not exist or is not a valid UUID.

    Args:
        engine:  SQLAlchemy engine with an active pool.
        look_id: UUID string — the share slug returned by save_look().

    Returns:
        A dict with keys: id, session_id, user_id, brand, look_id, occasion,
        look_gender, anchor_item_id, look_total_inr, snapshot (dict),
        created_at (ISO string); or None if not found / invalid UUID.
    """
    import uuid as _uuid

    import sqlalchemy as sa

    # Tolerate invalid UUID input gracefully — never crash.
    try:
        _uuid.UUID(look_id)
    except (ValueError, AttributeError):
        return None

    with engine.connect() as conn:
        row = conn.execute(
            sa.text(
                """
                SELECT
                    id::text,
                    session_id,
                    user_id::text,
                    brand,
                    look_id,
                    occasion,
                    look_gender,
                    anchor_item_id,
                    look_total_inr,
                    snapshot,
                    created_at::text
                FROM saved_looks
                WHERE id = CAST(:look_id AS uuid)
                """
            ),
            {"look_id": look_id},
        ).fetchone()

    if row is None:
        return None

    snapshot = row[9]
    # Some drivers return JSONB as a dict; others return a JSON string.
    if isinstance(snapshot, str):
        snapshot = json.loads(snapshot)

    return {
        "id": row[0],
        "session_id": row[1],
        "user_id": row[2],
        "brand": row[3],
        "look_id": row[4],
        "occasion": row[5],
        "look_gender": row[6],
        "anchor_item_id": row[7],
        "look_total_inr": row[8],
        "snapshot": snapshot,
        "created_at": row[10],
    }
