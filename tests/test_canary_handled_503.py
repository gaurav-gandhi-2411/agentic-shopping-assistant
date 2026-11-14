"""GET /canary: a run that raises is a handled 503 with a fixed body, never an unhandled 500."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api.routes.canary as canary
from api.main import app

TOKEN = "canary-503-test-token-0123456789"
SECRET = "postgresql://user:hunter2pw@db.internal/prod"  # must never reach the response


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CANARY_TOKEN", TOKEN)
    monkeypatch.setattr(canary, "_state", canary._State())

    async def boom() -> Any:
        raise RuntimeError(f"provider exploded {SECRET}")

    monkeypatch.setattr(canary, "_result", boom)
    # raise_server_exceptions=True: an unhandled error would surface as an exception here
    return TestClient(app, raise_server_exceptions=True)


def test_raising_run_is_handled_503_with_fixed_body(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR)
    r = client.get("/canary", params={"token": TOKEN})
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "fail"
    assert body["failing_assertion"] == "internal_error"
    assert body["detail"] == "canary run failed unexpectedly"
    assert body["cached"] is False
    assert "exploded" not in r.text
    assert "hunter2pw" not in r.text
    # server-side log carries the traceback (exc_info), not the response
    recs = [x for x in caplog.records if x.name == canary.logger.name and x.exc_info]
    assert recs, "exception was not logged with exc_info"


def test_raising_run_never_returns_200(client: TestClient) -> None:
    assert client.get("/canary", params={"token": TOKEN}).status_code != 200


def test_no_token_is_still_404_when_run_would_raise(client: TestClient) -> None:
    assert client.get("/canary").status_code == 404
    assert client.get("/canary", params={"token": "nope"}).status_code == 404
