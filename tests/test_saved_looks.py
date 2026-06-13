"""Tests for saved-looks storage layer and API endpoints.

No live Postgres required — all DB interactions are mocked via MagicMock.
The API tests mount only the looks router in a bare FastAPI app (same pattern
as test_api_feedback.py) so no data files or retrieval indices are loaded.

Covers:
  - save_look() returns a string id on success.
  - get_look() returns the saved snapshot dict on success.
  - get_look() returns None for an unknown (valid UUID) id.
  - get_look() returns None for an invalid / non-UUID id — no crash.
  - POST /looks returns 201 + SaveLookResponse when DB is present.
  - POST /looks returns 503 when no DB engine is configured.
  - GET /looks/{look_id} returns 200 + SharedLookResponse when look exists.
  - GET /looks/{look_id} returns 404 when look is not found.
  - GET /looks/{look_id} returns 404 for an invalid UUID id.
"""
from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routes.looks as looks_module
from api.routes.looks import router as looks_router
from src.storage.saved_looks import get_look, save_look

# ---------------------------------------------------------------------------
# Snapshot fixture (self-contained board payload)
# ---------------------------------------------------------------------------

_SNAPSHOT: dict = {
    "items": [
        {
            "article_id": "art-001",
            "name": "White Kurta",
            "colour": "white",
            "type": "Kurta",
            "slot": "top",
            "role": "seed",
            "image_url": "https://example.com/img/001.jpg",
            "price_inr": 1299,
            "pdp_handle": "white-kurta-001",
            "buy_url": "https://example.com/buy/001",
        }
    ],
    "rationale": "A clean white kurta for a casual ethnic occasion.",
    "cart_url": "https://example.com/cart?ids=art-001",
    "item_links": [{"article_id": "art-001", "name": "White Kurta", "buy_url": "https://example.com/buy/001"}],
    "variant_label": "Base",
}

_LOOK_ID_STR = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Engine mock helpers (mirrors test_api_feedback.py style)
# ---------------------------------------------------------------------------


def _make_save_engine(returned_id: str = _LOOK_ID_STR) -> MagicMock:
    """Engine whose begin() context yields a conn that returns the given UUID."""
    row = MagicMock()
    row.__getitem__ = lambda self, idx: returned_id  # row[0] → returned_id

    result = MagicMock()
    result.fetchone.return_value = row

    conn = MagicMock()
    conn.execute.return_value = result

    engine = MagicMock()
    engine.begin.return_value.__enter__ = lambda s: conn
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    return engine


def _make_get_engine(record: dict | None) -> MagicMock:
    """Engine whose connect() yields a conn that returns the given row or None."""
    if record is None:
        row = None
    else:
        # Build a mock row whose indexed access mirrors the SELECT column order:
        # 0:id, 1:session_id, 2:user_id, 3:brand, 4:look_id, 5:occasion,
        # 6:look_gender, 7:anchor_item_id, 8:look_total_inr, 9:snapshot, 10:created_at
        cols = [
            record.get("id", _LOOK_ID_STR),
            record.get("session_id", "sess-1"),
            record.get("user_id"),
            record.get("brand", "hm"),
            record.get("look_id"),
            record.get("occasion"),
            record.get("look_gender"),
            record.get("anchor_item_id"),
            record.get("look_total_inr"),
            json.dumps(record.get("snapshot", _SNAPSHOT)),  # string to test the parse branch
            record.get("created_at", "2026-06-13T10:00:00+00:00"),
        ]
        row = MagicMock()
        row.__getitem__ = lambda self, idx: cols[idx]

    result = MagicMock()
    result.fetchone.return_value = row

    conn = MagicMock()
    conn.execute.return_value = result

    engine = MagicMock()
    conn_ctx = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = conn_ctx
    return engine


# ---------------------------------------------------------------------------
# Unit tests for src.storage.saved_looks
# ---------------------------------------------------------------------------


class TestSaveLook:
    def test_returns_string_id(self) -> None:
        engine = _make_save_engine(returned_id=_LOOK_ID_STR)
        result = save_look(
            engine,
            session_id="sess-1",
            user_id=None,
            brand="hm",
            look_id="look-abc",
            occasion="casual",
            look_gender="women",
            anchor_item_id="art-001",
            look_total_inr=1299,
            snapshot=_SNAPSHOT,
        )
        assert isinstance(result, str)
        assert result == _LOOK_ID_STR

    def test_engine_begin_called(self) -> None:
        engine = _make_save_engine()
        save_look(
            engine,
            session_id="sess-1",
            user_id=None,
            brand="hm",
            look_id=None,
            occasion=None,
            look_gender=None,
            anchor_item_id=None,
            look_total_inr=None,
            snapshot=_SNAPSHOT,
        )
        engine.begin.assert_called_once()

    def test_snapshot_serialised_to_json(self) -> None:
        """The snapshot dict must be JSON-serialised before it is passed to execute()."""
        # Capture the conn at mock-build time so we can inspect it after the call.
        conn = MagicMock()
        row = MagicMock()
        row.__getitem__ = lambda self, idx: _LOOK_ID_STR
        result = MagicMock()
        result.fetchone.return_value = row
        conn.execute.return_value = result

        engine = MagicMock()
        engine.begin.return_value.__enter__ = lambda s: conn
        engine.begin.return_value.__exit__ = MagicMock(return_value=False)

        save_look(
            engine,
            session_id="sess-1",
            user_id=None,
            brand="hm",
            look_id=None,
            occasion=None,
            look_gender=None,
            anchor_item_id=None,
            look_total_inr=None,
            snapshot=_SNAPSHOT,
        )
        # Retrieve the params dict passed to conn.execute()
        call_params: dict = conn.execute.call_args[0][1]
        assert call_params["snapshot"] == json.dumps(_SNAPSHOT)


class TestGetLook:
    def test_returns_dict_for_known_id(self) -> None:
        record = {
            "id": _LOOK_ID_STR,
            "session_id": "sess-1",
            "user_id": None,
            "brand": "hm",
            "look_id": "look-abc",
            "occasion": "casual",
            "look_gender": "women",
            "anchor_item_id": "art-001",
            "look_total_inr": 1299,
            "snapshot": _SNAPSHOT,
            "created_at": "2026-06-13T10:00:00+00:00",
        }
        engine = _make_get_engine(record)
        result = get_look(engine, _LOOK_ID_STR)
        assert result is not None
        assert result["id"] == _LOOK_ID_STR
        assert result["brand"] == "hm"
        assert isinstance(result["snapshot"], dict)

    def test_snapshot_parsed_from_json_string(self) -> None:
        """get_look() must parse snapshot when the driver returns a JSON string."""
        record = {"snapshot": _SNAPSHOT}
        engine = _make_get_engine(record)
        result = get_look(engine, _LOOK_ID_STR)
        assert result is not None
        assert result["snapshot"] == _SNAPSHOT

    def test_returns_none_for_missing_id(self) -> None:
        engine = _make_get_engine(None)
        result = get_look(engine, _LOOK_ID_STR)
        assert result is None

    def test_returns_none_for_invalid_uuid(self) -> None:
        """Invalid UUID must not raise — return None gracefully."""
        engine = _make_get_engine(None)
        result = get_look(engine, "not-a-uuid")
        assert result is None
        # engine.connect should never be called for an invalid UUID.
        engine.connect.assert_not_called()

    def test_returns_none_for_empty_string(self) -> None:
        engine = _make_get_engine(None)
        assert get_look(engine, "") is None

    def test_all_keys_present_in_result(self) -> None:
        record = {"snapshot": _SNAPSHOT}
        engine = _make_get_engine(record)
        result = get_look(engine, _LOOK_ID_STR)
        assert result is not None
        expected_keys = {
            "id", "session_id", "user_id", "brand", "look_id", "occasion",
            "look_gender", "anchor_item_id", "look_total_inr", "snapshot", "created_at",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Minimal test app for API-layer tests
# ---------------------------------------------------------------------------


def _make_test_app() -> FastAPI:
    """Bare FastAPI app mounting only the looks router — no lifespan, no data files."""
    a = FastAPI()
    a.include_router(looks_router)
    return a


_app = _make_test_app()


def _set_engine(engine: Any) -> None:
    """Inject engine into app.state so _get_engine() in the route resolves it."""
    _app.state.engine = engine


def _clear_engine() -> None:
    if hasattr(_app.state, "engine"):
        del _app.state.engine


@pytest.fixture(autouse=True)
def _reset_engine():
    """Ensure engine state is clean between tests."""
    _clear_engine()
    yield
    _clear_engine()


# ---------------------------------------------------------------------------
# POST /looks tests
# ---------------------------------------------------------------------------


_POST_BODY: dict = {
    "session_id": "sess-1",
    "brand": "hm",
    "look_id": "look-abc",
    "occasion": "casual",
    "look_gender": "women",
    "anchor_item_id": "art-001",
    "look_total_inr": 1299,
    "snapshot": _SNAPSHOT,
}


class TestPostLook:
    def test_returns_201_and_save_look_response(self) -> None:
        _set_engine(_make_save_engine(returned_id=_LOOK_ID_STR))
        tc = TestClient(_app, raise_server_exceptions=True)
        resp = tc.post("/looks", json=_POST_BODY)
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == _LOOK_ID_STR
        assert data["share_path"] == f"/look/{_LOOK_ID_STR}"

    def test_503_when_no_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(looks_module, "_get_engine", lambda req: None)
        tc = TestClient(_app, raise_server_exceptions=False)
        resp = tc.post("/looks", json=_POST_BODY)
        assert resp.status_code == 503
        assert "database" in resp.json()["detail"].lower()

    def test_500_on_db_error(self) -> None:
        bad_engine = MagicMock()
        bad_engine.begin.return_value.__enter__ = MagicMock(side_effect=RuntimeError("DB down"))
        bad_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
        _set_engine(bad_engine)
        tc = TestClient(_app, raise_server_exceptions=False)
        resp = tc.post("/looks", json=_POST_BODY)
        assert resp.status_code == 500

    def test_share_path_contains_look_id(self) -> None:
        custom_id = str(uuid.uuid4())
        _set_engine(_make_save_engine(returned_id=custom_id))
        tc = TestClient(_app, raise_server_exceptions=True)
        resp = tc.post("/looks", json=_POST_BODY)
        assert resp.status_code == 201
        assert resp.json()["share_path"] == f"/look/{custom_id}"


# ---------------------------------------------------------------------------
# GET /looks/{look_id} tests
# ---------------------------------------------------------------------------


class TestGetLookEndpoint:
    def _record(self, look_id: str = _LOOK_ID_STR) -> dict:
        return {
            "id": look_id,
            "session_id": "sess-1",
            "user_id": None,
            "brand": "hm",
            "look_id": "look-abc",
            "occasion": "casual",
            "look_gender": "women",
            "anchor_item_id": "art-001",
            "look_total_inr": 1299,
            "snapshot": _SNAPSHOT,
            "created_at": "2026-06-13T10:00:00+00:00",
        }

    def test_200_with_correct_payload(self) -> None:
        _set_engine(_make_get_engine(self._record()))
        tc = TestClient(_app, raise_server_exceptions=True)
        resp = tc.get(f"/looks/{_LOOK_ID_STR}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == _LOOK_ID_STR
        assert data["brand"] == "hm"
        assert data["occasion"] == "casual"
        assert isinstance(data["snapshot"], dict)

    def test_404_for_unknown_id(self) -> None:
        _set_engine(_make_get_engine(None))
        tc = TestClient(_app, raise_server_exceptions=True)
        resp = tc.get(f"/looks/{_LOOK_ID_STR}")
        assert resp.status_code == 404

    def test_404_for_invalid_uuid(self) -> None:
        # get_look() returns None for non-UUID strings; the route must surface 404.
        _set_engine(_make_get_engine(None))
        tc = TestClient(_app, raise_server_exceptions=True)
        resp = tc.get("/looks/not-a-uuid-at-all")
        assert resp.status_code == 404

    def test_503_when_no_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(looks_module, "_get_engine", lambda req: None)
        tc = TestClient(_app, raise_server_exceptions=False)
        resp = tc.get(f"/looks/{_LOOK_ID_STR}")
        assert resp.status_code == 503
