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
                    "SELECT excluded_colours, summary FROM conversations "
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

        memory = ConversationMemory(self._llm, self._config)
        memory.restore_summary(conv_row.summary, len(messages))

        return {
            "messages": messages,
            "retrieved_items": retrieved_items,
            "filters": filters,
            "excluded_colours": conv_row.excluded_colours,
            "_memory": memory,
            "_db_message_count": len(messages),
        }

    def set(self, conversation_id: str, state: dict, user_id: str) -> None:
        messages: list[dict] = state.get("messages", [])
        watermark: int = state.get("_db_message_count", 0)
        new_messages = messages[watermark:]

        retrieved_items: list = state.get("retrieved_items", [])
        filters: dict = state.get("filters", {})
        excluded_colours = state.get("excluded_colours")
        title = _title_from_messages(messages)
        memory = state.get("_memory")
        summary: str | None = memory._cached_summary if memory is not None else None

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

            conn.execute(
                text(
                    """
                    INSERT INTO conversations
                        (id, user_id, title, excluded_colours, summary, updated_at)
                    VALUES
                        (CAST(:cid AS uuid), CAST(:uid AS uuid), :title,
                         CAST(:excl AS jsonb), :summary, now())
                    ON CONFLICT (id) DO UPDATE SET
                        title            = COALESCE(conversations.title, EXCLUDED.title),
                        excluded_colours = EXCLUDED.excluded_colours,
                        summary          = COALESCE(EXCLUDED.summary, conversations.summary),
                        updated_at       = now()
                    """
                ),
                {
                    "cid": conversation_id,
                    "uid": user_id,
                    "title": title,
                    "excl": json.dumps(excluded_colours),
                    "summary": summary,
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
