"""Integration tests for PostgresSessionStore.

Each test generates a fresh UUID conversation_id so tests are isolated
without DB transactions.  Conversations created during tests are cleaned
up in teardown.
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.storage.postgres_session_store import PostgresSessionStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set; run with a postgres:15 container",
)


# ---------------------------------------------------------------------------
# Fixture: one store instance per test (cheap — just a wrapper around engine)
# ---------------------------------------------------------------------------

@pytest.fixture
def store(pg_engine: Engine, dev_user_id: str, mock_llm, mock_config) -> PostgresSessionStore:
    return PostgresSessionStore(pg_engine, mock_llm, mock_config, dev_user_id)


@pytest.fixture
def cid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_missing_returns_none(self, store: PostgresSessionStore):
        result = store.get(str(uuid.uuid4()))
        assert result is None

    def test_get_after_set_returns_session(self, store: PostgresSessionStore, cid: str):
        state = {
            "messages": [{"role": "user", "content": "hello"}],
            "retrieved_items": [],
            "filters": {},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)
        loaded = store.get(cid)
        assert loaded is not None
        assert loaded["messages"] == [{"role": "user", "content": "hello"}]

    def test_get_reconstructs_memory(self, store: PostgresSessionStore, cid: str):
        from src.memory.conversation import ConversationMemory

        state = {
            "messages": [{"role": "user", "content": "hi"}],
            "retrieved_items": [],
            "filters": {},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)
        loaded = store.get(cid)
        assert isinstance(loaded["_memory"], ConversationMemory)

    def test_get_sets_db_message_count_watermark(self, store: PostgresSessionStore, cid: str):
        state = {
            "messages": [
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "reply1"},
            ],
            "retrieved_items": [],
            "filters": {},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)
        loaded = store.get(cid)
        assert loaded["_db_message_count"] == 2

    def test_get_restores_items_and_filters_from_last_assistant(
        self, store: PostgresSessionStore, cid: str
    ):
        items = [{"article_id": "abc", "display_name": "Red Dress"}]
        filters = {"colour": "red"}
        state = {
            "messages": [
                {"role": "user", "content": "show red dresses"},
                {"role": "assistant", "content": "here you go"},
            ],
            "retrieved_items": items,
            "filters": filters,
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)
        loaded = store.get(cid)
        assert loaded["retrieved_items"] == items
        assert loaded["filters"] == filters


class TestSet:
    def test_set_creates_conversation_row(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str
    ):
        state = {
            "messages": [{"role": "user", "content": "first message"}],
            "retrieved_items": [],
            "filters": {},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)
        with pg_engine.connect() as conn:
            row = conn.execute(
                text("SELECT id, title FROM conversations WHERE id = CAST(:cid AS uuid)"),
                {"cid": cid},
            ).fetchone()
        assert row is not None
        assert row.title == "first message"

    def test_set_creates_message_rows(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str
    ):
        state = {
            "messages": [
                {"role": "user", "content": "query"},
                {"role": "assistant", "content": "answer"},
            ],
            "retrieved_items": [],
            "filters": {},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)
        with pg_engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT role, content FROM messages "
                    "WHERE conversation_id = CAST(:cid AS uuid) ORDER BY created_at ASC"
                ),
                {"cid": cid},
            ).fetchall()
        assert len(rows) == 2
        assert rows[0].role == "user"
        assert rows[1].role == "assistant"

    def test_set_advances_watermark(self, store: PostgresSessionStore, cid: str):
        state = {
            "messages": [{"role": "user", "content": "hi"}],
            "retrieved_items": [],
            "filters": {},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)
        assert state["_db_message_count"] == 1

    def test_set_appends_only_new_messages(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str
    ):
        state = {
            "messages": [{"role": "user", "content": "turn 1"}],
            "retrieved_items": [],
            "filters": {},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)

        state["messages"].append({"role": "assistant", "content": "reply 1"})
        state["messages"].append({"role": "user", "content": "turn 2"})
        store.set(cid, state)

        with pg_engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM messages WHERE conversation_id = CAST(:cid AS uuid)"),
                {"cid": cid},
            ).scalar()
        assert count == 3

    def test_set_stores_items_on_last_assistant_only(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str
    ):
        items = [{"article_id": "x1", "display_name": "Item A"}]
        state = {
            "messages": [
                {"role": "user", "content": "find items"},
                {"role": "assistant", "content": "found them"},
            ],
            "retrieved_items": items,
            "filters": {"colour": "blue"},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)

        with pg_engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT role, items, filters FROM messages "
                    "WHERE conversation_id = CAST(:cid AS uuid) ORDER BY created_at ASC"
                ),
                {"cid": cid},
            ).fetchall()

        user_row, asst_row = rows
        assert user_row.items == []
        assert user_row.filters == {}
        assert asst_row.items == items
        assert asst_row.filters == {"colour": "blue"}

    def test_set_idempotent_conversation_upsert(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str
    ):
        state = {
            "messages": [{"role": "user", "content": "first"}],
            "retrieved_items": [],
            "filters": {},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)
        state["messages"].append({"role": "assistant", "content": "ok"})
        store.set(cid, state)

        with pg_engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM conversations WHERE id = CAST(:cid AS uuid)"),
                {"cid": cid},
            ).scalar()
        assert count == 1


class TestDelete:
    def test_delete_removes_conversation_and_messages(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str
    ):
        state = {
            "messages": [{"role": "user", "content": "hi"}],
            "retrieved_items": [],
            "filters": {},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)
        store.delete(cid)

        assert store.get(cid) is None
        with pg_engine.connect() as conn:
            msg_count = conn.execute(
                text("SELECT COUNT(*) FROM messages WHERE conversation_id = CAST(:cid AS uuid)"),
                {"cid": cid},
            ).scalar()
        assert msg_count == 0

    def test_delete_nonexistent_is_silent(self, store: PostgresSessionStore):
        store.delete(str(uuid.uuid4()))  # must not raise


class TestListIds:
    def test_list_ids_returns_own_conversations(
        self, store: PostgresSessionStore, cid: str
    ):
        state = {
            "messages": [{"role": "user", "content": "hi"}],
            "retrieved_items": [],
            "filters": {},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)
        ids = store.list_ids()
        assert cid in ids

    def test_list_ids_excludes_other_users(
        self,
        pg_engine: Engine,
        mock_llm,
        mock_config,
        dev_user_id: str,
    ):
        other_uid = str(uuid.uuid4())
        with pg_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO auth.users (id, email) "
                    "VALUES (CAST(:uid AS uuid), 'other@test.com') "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                {"uid": other_uid},
            )
            conn.execute(
                text(
                    "INSERT INTO users (id, email) "
                    "VALUES (CAST(:uid AS uuid), 'other@test.com') "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                {"uid": other_uid},
            )

        other_store = PostgresSessionStore(pg_engine, mock_llm, mock_config, other_uid)
        other_cid = str(uuid.uuid4())
        other_store.set(
            other_cid,
            {
                "messages": [{"role": "user", "content": "other msg"}],
                "retrieved_items": [],
                "filters": {},
                "excluded_colours": None,
                "_memory": None,
            },
        )

        dev_store = PostgresSessionStore(pg_engine, mock_llm, mock_config, dev_user_id)
        dev_ids = dev_store.list_ids()
        assert other_cid not in dev_ids


class TestRoundTrip:
    def test_multi_turn_round_trip(self, store: PostgresSessionStore, cid: str):
        """Full conversation: two turns, items on turn 2, reload between turns."""
        # Turn 1
        state: dict = {
            "messages": [
                {"role": "user", "content": "show me coats"},
                {"role": "assistant", "content": "here are some coats"},
            ],
            "retrieved_items": [{"article_id": "c1", "display_name": "Wool Coat"}],
            "filters": {"product_type": "coat"},
            "excluded_colours": None,
            "_memory": None,
        }
        store.set(cid, state)
        assert state["_db_message_count"] == 2

        # Reload (simulates server restart / new request)
        reloaded = store.get(cid)
        assert reloaded is not None
        assert reloaded["_db_message_count"] == 2
        assert reloaded["messages"] == state["messages"]
        assert reloaded["retrieved_items"] == [{"article_id": "c1", "display_name": "Wool Coat"}]
        assert reloaded["filters"] == {"product_type": "coat"}

        # Turn 2 — append new messages and update items
        reloaded["messages"].append({"role": "user", "content": "in red?"})
        reloaded["messages"].append({"role": "assistant", "content": "here are red coats"})
        reloaded["retrieved_items"] = [{"article_id": "c2", "display_name": "Red Coat"}]
        reloaded["filters"] = {"product_type": "coat", "colour": "red"}
        store.set(cid, reloaded)
        assert reloaded["_db_message_count"] == 4

        # Reload again
        final = store.get(cid)
        assert final is not None
        assert len(final["messages"]) == 4
        assert final["retrieved_items"] == [{"article_id": "c2", "display_name": "Red Coat"}]
        assert final["filters"] == {"product_type": "coat", "colour": "red"}
