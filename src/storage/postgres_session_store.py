"""PostgreSQL-backed session store.

Implements the same SessionStore protocol as InMemorySessionStore.

Schema mapping
--------------
The session dict used throughout the API has this shape::

    {
        "messages":         list[{"role": str, "content": str}],
        "retrieved_items":  list[dict],   # last retrieval results
        "filters":          dict,         # active filter snapshot
        "excluded_colours": list | None,  # user colour preferences (not persisted — see note)
        "_memory":          ConversationMemory,   # reconstructed on load
        "_db_message_count": int,         # watermark: messages already in DB
    }

Messages are stored one row per turn in the messages table.  Items and
filters are serialised into the JSONB columns on the LAST assistant
message of each set() call, so that get() can restore them without
replaying the full history.

excluded_colours is NOT persisted to the DB in this phase — it will be
added to the conversations table in a later migration when the full user
preference model is designed.
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
    def __init__(self, engine: Engine, llm: Any, config: dict, user_id: str) -> None:
        self._engine = engine
        self._llm = llm
        self._config = config
        self._user_id = user_id

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def get(self, conversation_id: str) -> dict | None:
        from src.memory.conversation import ConversationMemory

        with self._engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM conversations WHERE id = CAST(:cid AS uuid)"),
                {"cid": conversation_id},
            ).fetchone()
            if exists is None:
                return None

            rows = conn.execute(
                text(
                    "SELECT role, content, items, filters "
                    "FROM messages "
                    "WHERE conversation_id = CAST(:cid AS uuid) "
                    "ORDER BY created_at ASC"
                ),
                {"cid": conversation_id},
            ).fetchall()

        messages = [{"role": r.role, "content": r.content} for r in rows]

        retrieved_items: list = []
        filters: dict = {}
        for r in reversed(rows):
            if r.role == "assistant":
                retrieved_items = r.items or []
                filters = r.filters or {}
                break

        return {
            "messages": messages,
            "retrieved_items": retrieved_items,
            "filters": filters,
            "excluded_colours": None,
            "_memory": ConversationMemory(self._llm, self._config),
            "_db_message_count": len(messages),
        }

    def set(self, conversation_id: str, state: dict) -> None:
        messages: list[dict] = state.get("messages", [])
        watermark: int = state.get("_db_message_count", 0)
        new_messages = messages[watermark:]

        retrieved_items: list = state.get("retrieved_items", [])
        filters: dict = state.get("filters", {})
        title = _title_from_messages(messages)

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO conversations (id, user_id, title, updated_at)
                    VALUES (CAST(:cid AS uuid), CAST(:uid AS uuid), :title, now())
                    ON CONFLICT (id) DO UPDATE SET
                        title    = COALESCE(conversations.title, EXCLUDED.title),
                        updated_at = now()
                    """
                ),
                {"cid": conversation_id, "uid": self._user_id, "title": title},
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

    def delete(self, conversation_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("DELETE FROM conversations WHERE id = CAST(:cid AS uuid)"),
                {"cid": conversation_id},
            )

    def list_ids(self) -> list[str]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id FROM conversations "
                    "WHERE user_id = CAST(:uid AS uuid) "
                    "ORDER BY updated_at DESC"
                ),
                {"uid": self._user_id},
            ).fetchall()
        return [str(r.id) for r in rows]
