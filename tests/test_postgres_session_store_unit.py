"""Unit tests for PostgresSessionStore — no live database required.

These tests mock the SQLAlchemy engine and verify:
  1. list_summaries() maps SQL row columns to the expected dict shape.
  2. list_summaries() handles NULL title (falls back to "New conversation").
  3. list_summaries() coerces is_public to bool.
  4. list_summaries() coerces message_count to int.
  5. list_summaries() returns filters as dict (not raw JSONB string).
  6. _title_from_messages() extracts the first user message up to 100 chars.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.storage.postgres_session_store import (
    PostgresSessionStore,
    _title_from_messages,
)

# ---------------------------------------------------------------------------
# _title_from_messages unit tests
# ---------------------------------------------------------------------------

class TestTitleFromMessages:
    def test_returns_first_user_content(self) -> None:
        msgs = [
            {"role": "user", "content": "find me a red dress"},
            {"role": "assistant", "content": "here you go"},
        ]
        assert _title_from_messages(msgs) == "find me a red dress"

    def test_skips_assistant_messages(self) -> None:
        msgs = [
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "show coats"},
        ]
        assert _title_from_messages(msgs) == "show coats"

    def test_truncates_at_100_chars(self) -> None:
        long_content = "x" * 150
        msgs = [{"role": "user", "content": long_content}]
        result = _title_from_messages(msgs)
        assert result == "x" * 100

    def test_returns_none_for_empty_list(self) -> None:
        assert _title_from_messages([]) is None

    def test_returns_none_when_no_user_message(self) -> None:
        msgs = [{"role": "assistant", "content": "welcome"}]
        assert _title_from_messages(msgs) is None

    def test_returns_none_for_empty_user_content(self) -> None:
        msgs = [{"role": "user", "content": ""}]
        assert _title_from_messages(msgs) is None


# ---------------------------------------------------------------------------
# Helpers for list_summaries mocking
# ---------------------------------------------------------------------------

def _make_row(
    conversation_id: str = "abc-123",
    title: str | None = "show me coats",
    is_public: bool = False,
    message_count: int = 2,
    last_message: str | None = "Here are some coats.",
    filters: dict | None = None,
) -> MagicMock:
    """Build a mock SQLAlchemy Row with named attributes."""
    row = MagicMock()
    row.conversation_id = conversation_id
    row.title = title
    row.is_public = is_public
    row.message_count = message_count
    row.last_message = last_message
    row.filters = filters if filters is not None else {}
    return row


def _make_store_with_rows(rows: list[MagicMock]) -> PostgresSessionStore:
    """Return a PostgresSessionStore whose engine.connect() yields the given rows."""
    engine = MagicMock()
    conn_ctx = MagicMock()
    conn = MagicMock()
    result = MagicMock()

    result.fetchall.return_value = rows
    conn.execute.return_value = result
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = conn_ctx

    return PostgresSessionStore(engine=engine, llm=MagicMock(), config={})


# ---------------------------------------------------------------------------
# list_summaries unit tests
# ---------------------------------------------------------------------------

class TestListSummaries:
    def test_empty_returns_empty_list(self) -> None:
        store = _make_store_with_rows([])
        result = store.list_summaries("user-1")
        assert result == []

    def test_returns_one_dict_per_row(self) -> None:
        rows = [_make_row("id-1"), _make_row("id-2")]
        store = _make_store_with_rows(rows)
        result = store.list_summaries("user-1")
        assert len(result) == 2

    def test_dict_shape_has_required_keys(self) -> None:
        store = _make_store_with_rows([_make_row()])
        item = store.list_summaries("user-1")[0]
        assert set(item.keys()) == {
            "conversation_id", "title", "is_public",
            "message_count", "last_message", "filters",
        }

    def test_conversation_id_is_propagated(self) -> None:
        store = _make_store_with_rows([_make_row(conversation_id="conv-999")])
        assert store.list_summaries("user-1")[0]["conversation_id"] == "conv-999"

    def test_title_is_propagated(self) -> None:
        store = _make_store_with_rows([_make_row(title="My search")])
        assert store.list_summaries("user-1")[0]["title"] == "My search"

    def test_null_title_falls_back_to_default(self) -> None:
        store = _make_store_with_rows([_make_row(title=None)])
        assert store.list_summaries("user-1")[0]["title"] == "New conversation"

    def test_is_public_false_coerced_to_bool(self) -> None:
        store = _make_store_with_rows([_make_row(is_public=False)])
        result = store.list_summaries("user-1")[0]["is_public"]
        assert result is False
        assert isinstance(result, bool)

    def test_is_public_true_coerced_to_bool(self) -> None:
        store = _make_store_with_rows([_make_row(is_public=True)])
        result = store.list_summaries("user-1")[0]["is_public"]
        assert result is True
        assert isinstance(result, bool)

    def test_message_count_coerced_to_int(self) -> None:
        store = _make_store_with_rows([_make_row(message_count=5)])
        result = store.list_summaries("user-1")[0]["message_count"]
        assert result == 5
        assert isinstance(result, int)

    def test_last_message_propagated(self) -> None:
        store = _make_store_with_rows([_make_row(last_message="Here are results.")])
        assert store.list_summaries("user-1")[0]["last_message"] == "Here are results."

    def test_last_message_none_when_null(self) -> None:
        store = _make_store_with_rows([_make_row(last_message=None)])
        assert store.list_summaries("user-1")[0]["last_message"] is None

    def test_filters_dict_propagated(self) -> None:
        filt = {"colour_group_name": "Blue"}
        store = _make_store_with_rows([_make_row(filters=filt)])
        assert store.list_summaries("user-1")[0]["filters"] == filt

    def test_filters_non_dict_coerced_to_empty_dict(self) -> None:
        # DB may return a raw string for JSONB in some drivers; guard against it.
        store = _make_store_with_rows([_make_row(filters='{"colour":"blue"}')])
        result = store.list_summaries("user-1")[0]["filters"]
        assert result == {}

    def test_filters_empty_dict_when_null(self) -> None:
        store = _make_store_with_rows([_make_row(filters=None)])
        assert store.list_summaries("user-1")[0]["filters"] == {}

    def test_engine_connect_called_once(self) -> None:
        store = _make_store_with_rows([])
        store.list_summaries("user-1")
        store._engine.connect.assert_called_once()
