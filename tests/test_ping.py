"""GET /ping: dependency-free keep-warm route, excluded from the access log. Hermetic."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture()
def client() -> TestClient:
    # No ``with``: the lifespan (index load) is not run; /ping must not need it.
    return TestClient(app)


def _access_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "api.main"]


def test_ping_200_body_without_auth(client: TestClient) -> None:
    r = client.get("/ping")  # no Authorization header at all
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ping_is_get_only(client: TestClient) -> None:
    assert client.post("/ping").status_code == 405
    assert client.delete("/ping").status_code == 405


def test_ping_emits_no_access_log_line(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="api.main"):
        assert client.get("/ping").status_code == 200
    assert _access_lines(caplog) == []


def test_other_paths_are_still_access_logged(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="api.main"):
        assert client.get("/healthz").status_code == 200
        assert client.get("/ping").status_code == 200
        assert client.get("/no-such-path").status_code == 404
    assert _access_lines(caplog) == ["GET /healthz 200", "GET /no-such-path 404"]
