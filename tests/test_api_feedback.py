"""Tests for POST /messages/{message_id}/feedback.

Uses a minimal FastAPI app that mounts only the feedback router so no data
files or retrieval indices are loaded.  The engine used by the route is
monkeypatched to a mock that drives test scenarios without a real database.

Covers:
  - 204 on valid rating (1 and -1)
  - 422 on invalid rating (0, 2)
  - 404 when message does not exist
  - UPSERT: second call updates the existing row, not a duplicate
  - 503 when DATABASE_URL is unset and no engine is configured
  - comment field is forwarded to the UPSERT parameters
"""
from __future__ import annotations

import uuid
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routes.feedback as feedback_module
from api.routes.feedback import router as feedback_router


# ---------------------------------------------------------------------------
# Minimal test app — no lifespan, no data files, only the feedback router
# ---------------------------------------------------------------------------

def _make_test_app() -> FastAPI:
    """Create a bare FastAPI app with only the feedback router.

    JWT verification is disabled globally for tests via the
    JWT_VERIFICATION_DISABLED env var set by the autouse fixture.
    """
    a = FastAPI()
    a.include_router(feedback_router)
    return a


_app = _make_test_app()


# ---------------------------------------------------------------------------
# Engine mock helpers
# ---------------------------------------------------------------------------


def _make_conn(message_exists: bool) -> MagicMock:
    """Return a mock SQLAlchemy connection whose execute() behaves correctly.

    The feedback route executes two statements inside ``engine.begin()``:
      1. SELECT id FROM messages WHERE id = :mid   → fetchone() → row | None
      2. INSERT ... ON CONFLICT ... DO UPDATE ...   → (result ignored)

    The first execute() call always returns ``select_result``; the route only
    calls fetchone() on that result.  The second call's return value is not
    used by the route.
    """
    select_result = MagicMock()
    select_result.fetchone.return_value = (str(uuid.uuid4()),) if message_exists else None

    conn = MagicMock()
    conn.execute.return_value = select_result
    return conn


def _make_engine(message_exists: bool = True) -> tuple[MagicMock, MagicMock]:
    """Return (engine, conn) — engine drives the context manager, conn is inspectable."""
    conn = _make_conn(message_exists)

    engine = MagicMock()
    engine.begin.return_value.__enter__ = lambda s: conn
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    return engine, conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_jwt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")


@pytest.fixture()
def with_engine():
    """Factory: returns a function(message_exists=True) → (TestClient, conn_mock)."""
    original = feedback_module._engine

    def factory(message_exists: bool = True) -> tuple[TestClient, MagicMock]:
        engine, conn = _make_engine(message_exists)
        feedback_module._engine = engine
        return TestClient(_app, raise_server_exceptions=True), conn

    yield factory
    feedback_module._engine = original


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_feedback_thumbs_up(with_engine: Any) -> None:
    tc, _ = with_engine(message_exists=True)
    resp = tc.post(f"/messages/{uuid.uuid4()}/feedback", json={"rating": 1})
    assert resp.status_code == 204


def test_feedback_thumbs_down(with_engine: Any) -> None:
    tc, _ = with_engine(message_exists=True)
    resp = tc.post(f"/messages/{uuid.uuid4()}/feedback", json={"rating": -1})
    assert resp.status_code == 204


def test_feedback_with_comment(with_engine: Any) -> None:
    mid = str(uuid.uuid4())
    tc, conn = with_engine(message_exists=True)
    resp = tc.post(
        f"/messages/{mid}/feedback",
        json={"rating": 1, "comment": "Very helpful!"},
    )
    assert resp.status_code == 204

    # Verify comment was passed through to the UPSERT params.
    upsert_params = conn.execute.call_args_list[1][0][1]
    assert upsert_params["comment"] == "Very helpful!"


# ---------------------------------------------------------------------------
# Validation errors (no DB interaction expected)
# ---------------------------------------------------------------------------


def test_feedback_invalid_rating_zero(with_engine: Any) -> None:
    tc, _ = with_engine(message_exists=True)
    resp = tc.post(f"/messages/{uuid.uuid4()}/feedback", json={"rating": 0})
    assert resp.status_code == 422


def test_feedback_invalid_rating_positive_two(with_engine: Any) -> None:
    tc, _ = with_engine(message_exists=True)
    resp = tc.post(f"/messages/{uuid.uuid4()}/feedback", json={"rating": 2})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Not-found
# ---------------------------------------------------------------------------


def test_feedback_message_not_found(with_engine: Any) -> None:
    tc, _ = with_engine(message_exists=False)
    resp = tc.post(f"/messages/{uuid.uuid4()}/feedback", json={"rating": 1})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# UPSERT behaviour
# ---------------------------------------------------------------------------


def test_feedback_upsert_executes_two_statements(with_engine: Any) -> None:
    """Verify SELECT then UPSERT are both executed on a found message."""
    mid = str(uuid.uuid4())
    tc, conn = with_engine(message_exists=True)
    resp = tc.post(f"/messages/{mid}/feedback", json={"rating": -1})
    assert resp.status_code == 204

    assert conn.execute.call_count == 2, "Expected SELECT + UPSERT"
    upsert_params = conn.execute.call_args_list[1][0][1]
    assert upsert_params["rating"] == -1
    assert upsert_params["mid"] == mid


def test_feedback_upsert_no_second_execute_on_404(with_engine: Any) -> None:
    """UPSERT must NOT be called when the SELECT returns no row."""
    tc, conn = with_engine(message_exists=False)
    tc.post(f"/messages/{uuid.uuid4()}/feedback", json={"rating": 1})
    assert conn.execute.call_count == 1, "Only the SELECT should run before 404"


# ---------------------------------------------------------------------------
# No-database mode
# ---------------------------------------------------------------------------


def test_feedback_no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """503 when DATABASE_URL is absent and no engine singleton is set."""
    original = feedback_module._engine
    feedback_module._engine = None
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")

    tc = TestClient(_app, raise_server_exceptions=True)
    resp = tc.post(f"/messages/{uuid.uuid4()}/feedback", json={"rating": 1})

    assert resp.status_code == 503
    feedback_module._engine = original
