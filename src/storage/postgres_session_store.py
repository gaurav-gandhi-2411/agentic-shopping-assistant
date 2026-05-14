"""PostgreSQL-backed session store.

Implements the SessionStore protocol.  user_id is passed per-call (not stored
at construction time) so the same store instance can serve multiple users once
JWT auth is wired in Phase 2 prompt 2.

Schema mapping
--------------
Session dict shape used throughout the API::

    {
        "messages":          list[{"role": str, "content": str}],
        "retrieved_items":   list[dict],      # last retrieval results
        "filters":           dict,            # active filter snapshot
        "excluded_colours":  list | None,     # colour-negation preferences
        "_memory":           ConversationMemory,  # reconstructed on load
        "_db_message_count": int,             # watermark: messages already in DB
    }

Persistence strategy
--------------------
Messages
  One row per message.  Items and filters are written to the JSONB columns of
  the LAST assistant message in each set() call so that get() can restore them
  without replaying history.

excluded_colours
  Stored on the conversations row (it is conversation-level state, not per-
  message).  Restored by get() and written by set() via the conversation upsert.

Watermark + advisory lock
  _db_message_count tracks how many messages are already in the DB for this
  session dict.  set() inserts only messages[watermark:] — the messages this
  session instance has added since the last get().  pg_advisory_xact_lock
  serialises concurrent set() calls for the same conversation_id so that two
  sessions that loaded the same snapshot and each added messages do not
  interleave their inserts unpredictably.  The in-memory watermark (not the
  live DB count) is the correct slice boundary: each session knows exactly
  what it added, and the lock ensures the inserts are atomic with respect to
  other writers.
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import Engine, text


def _title_from_messages(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            return content[:100] if content else None
    return None


class PostgresSessionStore:
    def __init__(self, engine: Engine, llm: Any, config: dict) -> None:
        self._engine = engine
        self._llm = llm
        self._config = config

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def get(self, conversation_id: str, user_id: str) -> dict | None:
        from src.memory.conversation import ConversationMemory

        with self._engine.connect() as conn:
            conv_row = conn.execute(
                text(
                    "SELECT excluded_colours, summary, summary_message_count, is_public "
                    "FROM conversations "
                    "WHERE id = CAST(:cid AS uuid) AND user_id = CAST(:uid AS uuid)"
                ),
                {"cid": conversation_id, "uid": user_id},
            ).fetchone()
            if conv_row is None:
                return None

            msg_rows = conn.execute(
                text(
                    "SELECT role, content, items, filters "
                    "FROM messages "
                    "WHERE conversation_id = CAST(:cid AS uuid) "
                    "ORDER BY created_at ASC"
                ),
                {"cid": conversation_id},
            ).fetchall()

        messages = [{"role": r.role, "content": r.content} for r in msg_rows]

        retrieved_items: list = []
        filters: dict = {}
        for r in reversed(msg_rows):
            if r.role == "assistant":
                retrieved_items = r.items or []
                filters = r.filters or {}
                break

        # Restore the persisted summary count — NOT len(messages) — so the trigger
        # condition fires at the right time after a reload.
        memory = ConversationMemory(self._llm, self._config)
        memory.restore_summary(conv_row.summary, conv_row.summary_message_count)

        return {
            "messages": messages,
            "retrieved_items": retrieved_items,
            "filters": filters,
            "excluded_colours": conv_row.excluded_colours,
            "_memory": memory,
            "_db_message_count": len(messages),
            "_summary": conv_row.summary,
            "_summary_message_count": conv_row.summary_message_count,
            "_is_public": bool(conv_row.is_public),
        }

    def set(self, conversation_id: str, state: dict, user_id: str) -> None:
        messages: list[dict] = state.get("messages", [])
        watermark: int = state.get("_db_message_count", 0)
        new_messages = messages[watermark:]

        retrieved_items: list = state.get("retrieved_items", [])
        filters: dict = state.get("filters", {})
        excluded_colours = state.get("excluded_colours")
        is_public: bool = state.get("_is_public", False)
        title = _title_from_messages(messages)
        # Read summary state from top-level session keys — no access to _memory internals.
        # Both are None when a summary hasn't been computed yet; psycopg3 maps Python None
        # to SQL NULL so COALESCE in the upsert correctly preserves any existing DB value.
        summary: str | None = state.get("_summary")
        summary_message_count: int = state.get("_summary_message_count", 0)

        with self._engine.begin() as conn:
            # Serialise concurrent writers for this conversation.
            # hashtext() produces a 32-bit hash cast to bigint (fills the lower
            # 32 bits).  Collision probability is negligible at our scale — less
            # than 1-in-4B for the expected conversation count — and a collision
            # only causes unnecessary serialisation, never data loss.
            conn.execute(
                text(
                    "SELECT pg_advisory_xact_lock(CAST(hashtext(:cid) AS bigint))"
                ),
                {"cid": conversation_id},
            )

            # COALESCE / CASE strategy for nullable conversation-level state:
            # - excluded_colours and summary use COALESCE: a NULL from a turn that did not
            #   touch those fields never overwrites a stored value.  Explicit clearing
            #   requires a non-NULL value ([] for excluded_colours).
            # - (summary, summary_message_count) are updated atomically: if the incoming
            #   summary is NULL (not recomputed this turn) both are preserved; if summary
            #   is set, both take the new values.  This keeps the pair consistent at the
            #   SQL level even if the application layer ever desyncs them.
            # - excluded_colours is passed as Python None (SQL NULL) when absent, NOT as
            #   json.dumps(None) = 'null'::jsonb, so COALESCE fires correctly for JSONB.
            conn.execute(
                text(
                    """
                    INSERT INTO conversations
                        (id, user_id, title, is_public, excluded_colours, summary,
                         summary_message_count, updated_at)
                    VALUES
                        (CAST(:cid AS uuid), CAST(:uid AS uuid), :title, :is_public,
                         CAST(:excl AS jsonb), :summary, :smc, now())
                    ON CONFLICT (id) DO UPDATE SET
                        title                 = COALESCE(conversations.title, EXCLUDED.title),
                        is_public             = EXCLUDED.is_public,
                        excluded_colours      = COALESCE(EXCLUDED.excluded_colours,
                                                         conversations.excluded_colours),
                        summary               = COALESCE(EXCLUDED.summary, conversations.summary),
                        summary_message_count = CASE
                            WHEN EXCLUDED.summary IS NOT NULL
                            THEN EXCLUDED.summary_message_count
                            ELSE conversations.summary_message_count
                        END,
                        updated_at            = now()
                    """
                ),
                {
                    "cid": conversation_id,
                    "uid": user_id,
                    "title": title,
                    "is_public": is_public,
                    "excl": json.dumps(excluded_colours) if excluded_colours is not None else None,
                    "summary": summary,
                    "smc": summary_message_count,
                },
            )

            for idx, msg in enumerate(new_messages):
                global_idx = watermark + idx
                is_last_assistant = (
                    msg["role"] == "assistant" and global_idx == len(messages) - 1
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO messages
                            (conversation_id, role, content, items, filters)
                        VALUES
                            (CAST(:cid AS uuid), :role, :content,
                             CAST(:items AS jsonb), CAST(:filters AS jsonb))
                        """
                    ),
                    {
                        "cid": conversation_id,
                        "role": msg["role"],
                        "content": msg["content"],
                        "items": json.dumps(
                            retrieved_items if is_last_assistant else []
                        ),
                        "filters": json.dumps(
                            filters if is_last_assistant else {}
                        ),
                    },
                )

        state["_db_message_count"] = len(messages)

    def delete(self, conversation_id: str, user_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "DELETE FROM conversations "
                    "WHERE id = CAST(:cid AS uuid) AND user_id = CAST(:uid AS uuid)"
                ),
                {"cid": conversation_id, "uid": user_id},
            )

    def list_ids(self, user_id: str) -> list[str]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id FROM conversations "
                    "WHERE user_id = CAST(:uid AS uuid) "
                    "ORDER BY updated_at DESC"
                ),
                {"uid": user_id},
            ).fetchall()
        return [str(r.id) for r in rows]

    def list_summaries(self, user_id: str) -> list[dict]:
        """Single JOIN query replacing the list_ids + per-id get loop.

        Returns one dict per conversation with keys: conversation_id, title,
        is_public, message_count, last_message, filters.  Ordered by
        updated_at DESC (trust the DB; no Python re-sort needed).
        """
        sql = text(
            """
            WITH
            user_counts AS (
                SELECT conversation_id, COUNT(*) AS cnt
                FROM messages WHERE role = 'user'
                GROUP BY conversation_id
            ),
            first_user AS (
                SELECT DISTINCT ON (conversation_id) conversation_id, content
                FROM messages WHERE role = 'user'
                ORDER BY conversation_id, created_at ASC
            ),
            last_assistant AS (
                SELECT DISTINCT ON (conversation_id)
                    conversation_id, content, filters
                FROM messages WHERE role = 'assistant'
                ORDER BY conversation_id, created_at DESC
            )
            SELECT
                c.id::text            AS conversation_id,
                COALESCE(c.title,
                    LEFT(fu.content, 60)) AS title,
                c.is_public,
                COALESCE(uc.cnt, 0)   AS message_count,
                LEFT(la.content, 120) AS last_message,
                COALESCE(la.filters, '{}') AS filters
            FROM conversations c
            LEFT JOIN user_counts   uc ON uc.conversation_id = c.id
            LEFT JOIN first_user    fu ON fu.conversation_id = c.id
            LEFT JOIN last_assistant la ON la.conversation_id = c.id
            WHERE c.user_id = CAST(:uid AS uuid)
            ORDER BY c.updated_at DESC
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql, {"uid": user_id}).fetchall()
        return [
            {
                "conversation_id": r.conversation_id,
                "title": r.title or "New conversation",
                "is_public": bool(r.is_public),
                "message_count": int(r.message_count),
                "last_message": r.last_message,
                "filters": r.filters if isinstance(r.filters, dict) else {},
            }
            for r in rows
        ]
