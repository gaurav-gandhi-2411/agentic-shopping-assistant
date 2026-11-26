"""DISABLE_API_DOCS switch: default OFF keeps the docs routes, ON removes all three. Hermetic."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app

DOC_PATHS = ("/docs", "/redoc", "/openapi.json")


def _client(monkeypatch: pytest.MonkeyPatch, value: str | None) -> TestClient:
    if value is None:
        monkeypatch.delenv("DISABLE_API_DOCS", raising=False)
    else:
        monkeypatch.setenv("DISABLE_API_DOCS", value)
    return TestClient(create_app())  # no ``with``: lifespan (index load) not run


def test_default_keeps_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client(monkeypatch, None)
    for path in DOC_PATHS:
        assert c.get(path).status_code == 200, path


@pytest.mark.parametrize("value", ["", "false", "0"])
def test_falsy_values_keep_docs(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    c = _client(monkeypatch, value)
    assert c.get("/openapi.json").status_code == 200


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes"])
def test_switch_removes_docs(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    c = _client(monkeypatch, value)
    for path in DOC_PATHS:
        assert c.get(path).status_code == 404, path


@pytest.mark.parametrize("value", [None, "true"])
def test_real_routes_work_both_ways(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    c = _client(monkeypatch, value)
    assert c.get("/ping").status_code == 200
    assert c.get("/healthz").status_code == 200  # in-process route (Cloud Run itself 404s it)
    assert c.get("/readyz").status_code in (200, 503)  # reachable; readiness needs loaded indices
    assert c.get("/canary/status").status_code in (200, 503)  # a normal API route, handled
