"""Integration tests for PostgresSessionStore.

Each test uses a fresh UUID conversation_id so tests are isolated without
requiring DB transactions or teardown.
"""
from __future__ import annotations

import os
import threading
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
# Helpers
# ---------------------------------------------------------------------------

def _empty_state(**overrides) -> dict:
    base = {
        "messages": [],
        "retrieved_items": [],
        "filters": {},
        "excluded_colours": None,
        "_memory": None,
        "_summary": None,
        "_summary_message_count": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(pg_engine: Engine, mock_llm, mock_config) -> PostgresSessionStore:
    return PostgresSessionStore(pg_engine, mock_llm, mock_config)


@pytest.fixture
def cid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# TestGet
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_missing_returns_none(self, store: PostgresSessionStore, dev_user_id: str):
        assert store.get(str(uuid.uuid4()), dev_user_id) is None

    def test_get_wrong_user_returns_none(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        """A conversation owned by user A is invisible to user B."""
        state = _empty_state(messages=[{"role": "user", "content": "hi"}])
        store.set(cid, state, dev_user_id)
        assert store.get(cid, str(uuid.uuid4())) is None

    def test_get_after_set_returns_session(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        state = _empty_state(messages=[{"role": "user", "content": "hello"}])
        store.set(cid, state, dev_user_id)
        loaded = store.get(cid, dev_user_id)
        assert loaded is not None
        assert loaded["messages"] == [{"role": "user", "content": "hello"}]

    def test_get_reconstructs_memory(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        from src.memory.conversation import ConversationMemory

        state = _empty_state(messages=[{"role": "user", "content": "hi"}])
        store.set(cid, state, dev_user_id)
        loaded = store.get(cid, dev_user_id)
        assert isinstance(loaded["_memory"], ConversationMemory)

    def test_get_sets_db_message_count_watermark(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        state = _empty_state(messages=[
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
        ])
        store.set(cid, state, dev_user_id)
        loaded = store.get(cid, dev_user_id)
        assert loaded["_db_message_count"] == 2

    def test_get_restores_items_and_filters_from_last_assistant(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        items = [{"article_id": "abc", "display_name": "Red Dress"}]
        filters = {"colour": "red"}
        state = _empty_state(
            messages=[
                {"role": "user", "content": "show red dresses"},
                {"role": "assistant", "content": "here you go"},
            ],
            retrieved_items=items,
            filters=filters,
        )
        store.set(cid, state, dev_user_id)
        loaded = store.get(cid, dev_user_id)
        assert loaded["retrieved_items"] == items
        assert loaded["filters"] == filters

    def test_get_restores_excluded_colours(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        state = _empty_state(
            messages=[{"role": "user", "content": "no red please"}],
            excluded_colours=["red", "crimson"],
        )
        store.set(cid, state, dev_user_id)
        loaded = store.get(cid, dev_user_id)
        assert loaded["excluded_colours"] == ["red", "crimson"]

    def test_get_excluded_colours_none_when_not_set(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        state = _empty_state(messages=[{"role": "user", "content": "show coats"}])
        store.set(cid, state, dev_user_id)
        loaded = store.get(cid, dev_user_id)
        assert loaded["excluded_colours"] is None


# ---------------------------------------------------------------------------
# TestSet
# ---------------------------------------------------------------------------

class TestSet:
    def test_set_creates_conversation_row(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        state = _empty_state(messages=[{"role": "user", "content": "first message"}])
        store.set(cid, state, dev_user_id)
        with pg_engine.connect() as conn:
            row = conn.execute(
                text("SELECT title FROM conversations WHERE id = CAST(:cid AS uuid)"),
                {"cid": cid},
            ).fetchone()
        assert row is not None
        assert row.title == "first message"

    def test_set_creates_message_rows(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        state = _empty_state(messages=[
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "answer"},
        ])
        store.set(cid, state, dev_user_id)
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

    def test_set_advances_watermark(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        state = _empty_state(messages=[{"role": "user", "content": "hi"}])
        store.set(cid, state, dev_user_id)
        assert state["_db_message_count"] == 1

    def test_set_appends_only_new_messages(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        state = _empty_state(messages=[{"role": "user", "content": "turn 1"}])
        store.set(cid, state, dev_user_id)

        state["messages"].append({"role": "assistant", "content": "reply 1"})
        state["messages"].append({"role": "user", "content": "turn 2"})
        store.set(cid, state, dev_user_id)

        with pg_engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM messages "
                    "WHERE conversation_id = CAST(:cid AS uuid)"
                ),
                {"cid": cid},
            ).scalar()
        assert count == 3

    def test_set_stores_items_on_last_assistant_only(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        items = [{"article_id": "x1", "display_name": "Item A"}]
        state = _empty_state(
            messages=[
                {"role": "user", "content": "find items"},
                {"role": "assistant", "content": "found them"},
            ],
            retrieved_items=items,
            filters={"colour": "blue"},
        )
        store.set(cid, state, dev_user_id)

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
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        state = _empty_state(messages=[{"role": "user", "content": "first"}])
        store.set(cid, state, dev_user_id)
        state["messages"].append({"role": "assistant", "content": "ok"})
        store.set(cid, state, dev_user_id)

        with pg_engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM conversations WHERE id = CAST(:cid AS uuid)"
                ),
                {"cid": cid},
            ).scalar()
        assert count == 1

    def test_set_persists_excluded_colours(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        state = _empty_state(
            messages=[{"role": "user", "content": "no pink"}],
            excluded_colours=["pink", "rose"],
        )
        store.set(cid, state, dev_user_id)
        with pg_engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT excluded_colours FROM conversations "
                    "WHERE id = CAST(:cid AS uuid)"
                ),
                {"cid": cid},
            ).fetchone()
        assert row.excluded_colours == ["pink", "rose"]

    def test_set_updates_excluded_colours_on_second_call(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        state = _empty_state(
            messages=[{"role": "user", "content": "no pink"}],
            excluded_colours=["pink"],
        )
        store.set(cid, state, dev_user_id)

        state["messages"].append({"role": "assistant", "content": "ok"})
        state["messages"].append({"role": "user", "content": "no blue either"})
        state["excluded_colours"] = ["pink", "blue"]
        store.set(cid, state, dev_user_id)

        with pg_engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT excluded_colours FROM conversations "
                    "WHERE id = CAST(:cid AS uuid)"
                ),
                {"cid": cid},
            ).fetchone()
        assert row.excluded_colours == ["pink", "blue"]


# ---------------------------------------------------------------------------
# TestDelete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_removes_conversation_and_messages(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        state = _empty_state(messages=[{"role": "user", "content": "hi"}])
        store.set(cid, state, dev_user_id)
        store.delete(cid, dev_user_id)

        assert store.get(cid, dev_user_id) is None
        with pg_engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM messages "
                    "WHERE conversation_id = CAST(:cid AS uuid)"
                ),
                {"cid": cid},
            ).scalar()
        assert count == 0

    def test_delete_wrong_user_does_nothing(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        """delete() with a wrong user_id must not delete the conversation."""
        state = _empty_state(messages=[{"role": "user", "content": "hi"}])
        store.set(cid, state, dev_user_id)
        store.delete(cid, str(uuid.uuid4()))  # wrong user

        # Conversation must still exist for the real owner
        assert store.get(cid, dev_user_id) is not None

    def test_delete_nonexistent_is_silent(
        self, store: PostgresSessionStore, dev_user_id: str
    ):
        store.delete(str(uuid.uuid4()), dev_user_id)


# ---------------------------------------------------------------------------
# TestListIds
# ---------------------------------------------------------------------------

class TestListIds:
    def test_list_ids_returns_own_conversations(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        state = _empty_state(messages=[{"role": "user", "content": "hi"}])
        store.set(cid, state, dev_user_id)
        assert cid in store.list_ids(dev_user_id)

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

        store = PostgresSessionStore(pg_engine, mock_llm, mock_config)
        other_cid = str(uuid.uuid4())
        store.set(
            other_cid,
            _empty_state(messages=[{"role": "user", "content": "other msg"}]),
            other_uid,
        )

        assert other_cid not in store.list_ids(dev_user_id)


# ---------------------------------------------------------------------------
# TestConcurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_set_no_duplicates_no_losses(
        self,
        store: PostgresSessionStore,
        pg_engine: Engine,
        cid: str,
        dev_user_id: str,
    ):
        """Two threads each load the same conversation, add a turn, and call
        set() concurrently.  The advisory lock serialises the writes.  The
        in-memory watermark correctly tracks what each session added so both
        threads' messages end up in the DB with no duplicates.

        Without the advisory lock there is a narrow window where both threads
        could enter their transactions and interleave inserts unpredictably;
        with it they serialise and the watermark slice is applied atomically.
        """
        # Seed 2 initial messages so both threads start with _db_count = 2.
        seed = _empty_state(messages=[
            {"role": "user", "content": "initial query"},
            {"role": "assistant", "content": "initial reply"},
        ])
        store.set(cid, seed, dev_user_id)

        errors: list[Exception] = []
        barrier = threading.Barrier(2)  # both threads start at the same moment

        def add_turn(user_msg: str, asst_msg: str) -> None:
            try:
                session = store.get(cid, dev_user_id)
                assert session is not None
                barrier.wait()  # synchronise both threads before set()
                session["messages"].append({"role": "user", "content": user_msg})
                session["messages"].append({"role": "assistant", "content": asst_msg})
                store.set(cid, session, dev_user_id)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=add_turn, args=("thread-1 query", "thread-1 reply"))
        t2 = threading.Thread(target=add_turn, args=("thread-2 query", "thread-2 reply"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"

        with pg_engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM messages "
                    "WHERE conversation_id = CAST(:cid AS uuid)"
                ),
                {"cid": cid},
            ).scalar()

        # 2 initial + 2 from thread 1 + 2 from thread 2 = 6, no duplicates.
        assert count == 6


# ---------------------------------------------------------------------------
# TestRoundTrip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_multi_turn_round_trip(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        """Full conversation: two turns, items on turn 2, reload between turns."""
        # Turn 1
        state: dict = _empty_state(
            messages=[
                {"role": "user", "content": "show me coats"},
                {"role": "assistant", "content": "here are some coats"},
            ],
            retrieved_items=[{"article_id": "c1", "display_name": "Wool Coat"}],
            filters={"product_type": "coat"},
            excluded_colours=["beige"],
        )
        store.set(cid, state, dev_user_id)
        assert state["_db_message_count"] == 2

        # Reload (simulates server restart / new request)
        reloaded = store.get(cid, dev_user_id)
        assert reloaded is not None
        assert reloaded["_db_message_count"] == 2
        assert reloaded["messages"] == state["messages"]
        assert reloaded["retrieved_items"] == [{"article_id": "c1", "display_name": "Wool Coat"}]
        assert reloaded["filters"] == {"product_type": "coat"}
        assert reloaded["excluded_colours"] == ["beige"]

        # Turn 2 — append new messages, update items and excluded_colours
        reloaded["messages"].append({"role": "user", "content": "in red?"})
        reloaded["messages"].append({"role": "assistant", "content": "here are red coats"})
        reloaded["retrieved_items"] = [{"article_id": "c2", "display_name": "Red Coat"}]
        reloaded["filters"] = {"product_type": "coat", "colour": "red"}
        reloaded["excluded_colours"] = ["beige", "brown"]
        store.set(cid, reloaded, dev_user_id)
        assert reloaded["_db_message_count"] == 4

        # Reload again
        final = store.get(cid, dev_user_id)
        assert final is not None
        assert len(final["messages"]) == 4
        assert final["retrieved_items"] == [{"article_id": "c2", "display_name": "Red Coat"}]
        assert final["filters"] == {"product_type": "coat", "colour": "red"}
        assert final["excluded_colours"] == ["beige", "brown"]


# ---------------------------------------------------------------------------
# TestMemorySummary
# ---------------------------------------------------------------------------

class TestMemorySummary:
    def test_set_persists_summary_from_state_dict(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        """_summary in the session dict (not _memory) is written to conversations.summary."""
        state = _empty_state(
            messages=[{"role": "user", "content": "show coats"}],
            _summary="three bullets about coats",
            _summary_message_count=4,
        )
        store.set(cid, state, dev_user_id)

        with pg_engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT summary, summary_message_count FROM conversations "
                    "WHERE id = CAST(:cid AS uuid)"
                ),
                {"cid": cid},
            ).fetchone()
        assert row.summary == "three bullets about coats"
        assert row.summary_message_count == 4

    def test_get_restores_summary_to_state_dict_and_memory(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str, mock_llm, mock_config
    ):
        """get() restores summary to both the session dict and the memory object."""
        from src.memory.conversation import ConversationMemory

        state = _empty_state(
            messages=[
                {"role": "user", "content": "show red dresses"},
                {"role": "assistant", "content": "here are some"},
            ],
            _summary="earlier search: red dresses under £100",
            _summary_message_count=8,
        )
        store.set(cid, state, dev_user_id)

        loaded = store.get(cid, dev_user_id)
        assert loaded is not None
        # Session dict keys
        assert loaded["_summary"] == "earlier search: red dresses under £100"
        assert loaded["_summary_message_count"] == 8
        # Memory object (restored via restore_summary — used by get_context trigger logic)
        assert isinstance(loaded["_memory"], ConversationMemory)
        assert loaded["_memory"]._cached_summary == "earlier search: red dresses under £100"
        assert loaded["_memory"]._summary_computed_at == 8  # Issue 1 fix: stored count, not len(messages)

    def test_get_summary_none_when_never_set(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        """A fresh conversation with no summary yields _summary = None in loaded state."""
        state = _empty_state(messages=[{"role": "user", "content": "hi"}])
        store.set(cid, state, dev_user_id)
        loaded = store.get(cid, dev_user_id)
        assert loaded["_summary"] is None
        assert loaded["_summary_message_count"] == 0
        assert loaded["_memory"]._cached_summary is None
        assert loaded["_memory"]._summary_computed_at == 0

    def test_summary_coalesce_preserves_existing_when_null(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        """A second set() with _summary=None does not overwrite an existing summary."""
        state = _empty_state(
            messages=[{"role": "user", "content": "turn 1"}],
            _summary="initial summary",
            _summary_message_count=5,
        )
        store.set(cid, state, dev_user_id)

        # Second set — new turn, no summary recomputed this iteration
        state["messages"].append({"role": "assistant", "content": "reply"})
        state["_summary"] = None
        state["_summary_message_count"] = 0
        store.set(cid, state, dev_user_id)

        with pg_engine.connect() as conn:
            row = conn.execute(
                text("SELECT summary FROM conversations WHERE id = CAST(:cid AS uuid)"),
                {"cid": cid},
            ).fetchone()
        assert row.summary == "initial summary"

    def test_excluded_colours_coalesce_preserves_existing_when_null(
        self, store: PostgresSessionStore, pg_engine: Engine, cid: str, dev_user_id: str
    ):
        """A second set() with excluded_colours=None does not overwrite an existing list."""
        state = _empty_state(
            messages=[{"role": "user", "content": "no red"}],
            excluded_colours=["red"],
        )
        store.set(cid, state, dev_user_id)

        # Second set — new turn, colour preference not touched → still None in state
        state["messages"].append({"role": "assistant", "content": "ok"})
        state["excluded_colours"] = None
        store.set(cid, state, dev_user_id)

        with pg_engine.connect() as conn:
            row = conn.execute(
                text("SELECT excluded_colours FROM conversations WHERE id = CAST(:cid AS uuid)"),
                {"cid": cid},
            ).fetchone()
        assert row.excluded_colours == ["red"]

    def test_summary_message_count_restored_correctly_not_len_messages(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str
    ):
        """Regression: _summary_computed_at after restore is the stored count, not len(messages).

        If restore used len(messages) instead of the stored count, a summary computed at
        turn 12 and loaded at turn 20 would have _summary_computed_at=20, delaying the
        next recompute by 8 extra turns.
        """
        # Simulate: summary computed at turn 12, now at turn 20 (8 more messages since)
        messages_at_turn_20 = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]
        state = _empty_state(
            messages=messages_at_turn_20,
            _summary="summary from turn 12",
            _summary_message_count=12,  # computed at turn 12
        )
        store.set(cid, state, dev_user_id)

        loaded = store.get(cid, dev_user_id)
        # Must be 12 (when computed), not 20 (current message count)
        assert loaded["_memory"]._summary_computed_at == 12
        assert loaded["_summary_message_count"] == 12

    def test_summary_survives_server_restart(
        self, store: PostgresSessionStore, cid: str, dev_user_id: str, mock_llm, mock_config
    ):
        """End-to-end: summary written on turn N is available on a fresh store instance."""
        state = _empty_state(
            messages=[
                {"role": "user", "content": "sustainable coats"},
                {"role": "assistant", "content": "here are some"},
            ],
            _summary="user wants sustainable coats under £200",
            _summary_message_count=6,
        )
        store.set(cid, state, dev_user_id)

        fresh_store = PostgresSessionStore(store._engine, mock_llm, mock_config)
        reloaded = fresh_store.get(cid, dev_user_id)
        assert reloaded is not None
        assert reloaded["_summary"] == "user wants sustainable coats under £200"
        assert reloaded["_summary_message_count"] == 6
        assert reloaded["_memory"]._cached_summary == "user wants sustainable coats under £200"
        assert reloaded["_memory"]._summary_computed_at == 6
