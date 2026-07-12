"""Saved-looks persistence layer.

Insert-only: save_look() appends a row; get_look() reads it back.  No UPDATE
or DELETE paths exist — a saved look is immutable once written.  The row id
(UUID) doubles as the public share slug so no secondary lookup table is needed.

When DATABASE_URL is not configured (local dev), the in-memory fallback
(save_look_memory / get_look_memory) is used instead.  Data persists for the
process lifetime only — sufficient for local testing and QA checks.
"""
from __future__ import annotations

import datetime as _datetime
import json
import logging
import uuid as _uuid_module
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


# ---------------------------------------------------------------------------
# In-memory fallback store — dev / local testing (no DB required)
# ---------------------------------------------------------------------------

# Dict keyed by look UUID string -> saved look record dict.
# Persists for the lifetime of the process; cleared between QA test runs.
_MEMORY_STORE: dict[str, dict] = {}


def save_look_memory(
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
    """Persist a saved look in in-process memory (dev fallback when DATABASE_URL is unset).

    Generates a fresh UUID, stores all fields in _MEMORY_STORE, and returns the
    UUID string as the share slug — same contract as save_look().

    Args:
        session_id:     Anonymous demo session identifier.
        user_id:        Authenticated user UUID string, or None for anonymous.
        brand:          Brand slug (e.g. "unified", "snitch").
        look_id:        Ephemeral outfit-engine look id, or None.
        occasion:       Occasion label, or None.
        look_gender:    Gender label, or None.
        anchor_item_id: Anchor catalogue item id, or None.
        look_total_inr: Basket total in INR, or None.
        snapshot:       Self-contained board payload.

    Returns:
        UUID string of the newly stored record (also the share slug).
    """
    new_id = str(_uuid_module.uuid4())
    _MEMORY_STORE[new_id] = {
        "id": new_id,
        "session_id": session_id,
        "user_id": user_id,
        "brand": brand,
        "look_id": look_id,
        "occasion": occasion,
        "look_gender": look_gender,
        "anchor_item_id": anchor_item_id,
        "look_total_inr": look_total_inr,
        "snapshot": snapshot,
        "created_at": _datetime.datetime.utcnow().isoformat() + "Z",
    }
    return new_id


def get_look_memory(look_id: str) -> dict | None:
    """Retrieve a memory-persisted saved look by UUID (dev fallback).

    Args:
        look_id: UUID string — the share slug returned by save_look_memory().

    Returns:
        Stored record dict, or None when the id is missing or not a valid UUID.
    """
    try:
        _uuid_module.UUID(look_id)
    except (ValueError, AttributeError):
        return None
    return _MEMORY_STORE.get(look_id)
