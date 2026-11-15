"""/readyz and /canary mapping of the INDEX_BUILD_ID pin (revision-pinned rollout).

Hermetic: the snapshot is a stand-in around a real BuildManifest; time is patched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
import api.routes.canary as canary
from api.main import app
from api.routes import health
from src.retrieval.build_manifest import BuildManifest, StoreEntry
from src.retrieval.pinned_build import PIN_REASONS
from src.retrieval.snapshot import IndexSnapshot

NOW = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
BUILD = "20261119T000000Z"
TOKEN = "pin-readiness-token-0123456789"


def _manifest(build_id: str = BUILD, age_h: float = 3.5) -> BuildManifest:
    return BuildManifest(
        build_id=build_id,
        built_at=NOW - timedelta(hours=age_h),
        git_sha="abc123",
        stores=[StoreEntry(slug="libas", items=3), StoreEntry(slug="myntra", items=2)],
        total_items=5,
        vector_count=5,
        files=[],
    )


def _snap(
    *,
    manifest: BuildManifest | None,
    source: str,
    pinned: str | None = None,
    failure: str | None = None,
) -> IndexSnapshot:
    retriever = SimpleNamespace(dense=SimpleNamespace(index=SimpleNamespace(ntotal=5)))
    return IndexSnapshot(
        retriever=retriever,
        catalogue_df=pd.DataFrame({"article_id": list("abc")}),
        manifest=manifest,
        build_id=manifest.build_id if manifest else "legacy",
        source=source,
        pinned_build_id=pinned,
        pin_failure=failure,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(canary.INDEX_STRICT_ENV, raising=False)
    monkeypatch.setenv("CANARY_TOKEN", TOKEN)
    monkeypatch.setattr(canary, "_utcnow", lambda: NOW)
    monkeypatch.setattr(health, "_now", lambda: NOW)
    monkeypatch.setattr(deps, "_llm", SimpleNamespace())
    monkeypatch.setattr(deps, "_db_engine", None)
    monkeypatch.setattr(deps, "_reloader", None)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def pass_result(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    async def fake_result() -> tuple[str, int, dict[str, Any]]:
        calls.append(1)
        return "cached", 200, {"status": "pass"}

    monkeypatch.setattr(canary, "_result", fake_result)
    return calls


def _install(monkeypatch: pytest.MonkeyPatch, snap: IndexSnapshot) -> None:
    monkeypatch.setattr(deps, "_snapshot", snap)


# --------------------------------------------------------------------------- /readyz


def test_pinned_and_loaded_is_ok_with_the_full_index_block(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _snap(manifest=_manifest(), source="build", pinned=BUILD))
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["checks"]["index"] == f"ok ({BUILD}, age 3.5h)"
    idx = body["index"]
    assert idx["pinned_build_id"] == BUILD and idx["loaded_build_id"] == BUILD
    assert idx["source"] == "build"
    assert idx["built_at"].startswith("2026-11-20T08:30:00") and idx["age_hours"] == 3.5
    assert idx["items_built"] == 5 and idx["items_served"] == 3  # unchanged semantics


def test_nothing_pinned_is_legacy_and_ready(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _snap(manifest=None, source="flat"))
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["checks"]["index"] == "legacy (no manifest)"
    idx = r.json()["index"]
    assert idx["source"] == "legacy" and idx["pinned_build_id"] is None
    assert idx["loaded_build_id"] is None and idx["built_at"] is None
    assert idx["age_hours"] is None and idx["items_built"] is None and idx["items_served"] == 3


@pytest.mark.parametrize("code", [c for c in PIN_REASONS if c != "build_mismatch"])
def test_pinned_but_legacy_loaded_is_a_503_with_the_exact_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    _install(monkeypatch, _snap(manifest=None, source="flat", pinned=BUILD, failure=code))
    r = client.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "not ready"
    assert r.json()["checks"]["index"] == (
        f"error: pinned build {BUILD} not loaded, serving legacy ({code})"
    )
    idx = r.json()["index"]
    assert idx["source"] == "legacy-fallback" and idx["pinned_build_id"] == BUILD
    assert idx["loaded_build_id"] is None


def test_an_invalid_pin_never_echoes_the_raw_value(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _snap(manifest=None, source="flat", failure="id_invalid"))
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["checks"]["index"] == (
        "error: pinned build <invalid> not loaded, serving legacy (id_invalid)"
    )


def test_a_different_build_being_served_is_an_error_not_a_silent_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = "20261118T000000Z"
    _install(monkeypatch, _snap(manifest=_manifest(other), source="build", pinned=BUILD))
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["checks"]["index"] == (
        f"error: pinned build {BUILD} not loaded, serving build {other} (build_mismatch)"
    )


def test_the_deploy_smoke_readiness_check_would_fail_on_the_pin_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """deploy_smoke.py requires GET /readyz == 200 and status == ready; both fail here."""
    _install(monkeypatch, _snap(manifest=None, source="flat", pinned=BUILD, failure="not_found"))
    r = client.get("/readyz")
    assert r.status_code != 200 and r.json()["status"] != "ready"


def test_unpinned_manifest_bearing_index_keeps_the_existing_format(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _snap(manifest=_manifest(), source="flat"))
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["checks"]["index"].startswith(f"ok (build_id={BUILD}, age_hours=3.5")
    assert r.json()["index"]["source"] == "legacy"


# --------------------------------------------------------------------------- /canary


def test_canary_status_fails_with_index_pin_unloaded_and_runs_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pass_result: list[int]
) -> None:
    _install(monkeypatch, _snap(manifest=None, source="flat", pinned=BUILD, failure="integrity"))
    r = client.get("/canary/status")
    assert r.status_code == 503
    assert r.json() == {
        "status": "fail",
        "assertion": "index_pin_unloaded",
        "detail": "pinned index build not loaded",
    }
    assert pass_result == []  # a cached pass can never mask it


def test_canary_endpoint_fails_the_same_way(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pass_result: list[int]
) -> None:
    _install(monkeypatch, _snap(manifest=None, source="flat", pinned=BUILD, failure="integrity"))
    r = client.get("/canary", headers={"X-Canary-Token": TOKEN})
    assert r.status_code == 500
    assert "index_pin_unloaded" in r.text and pass_result == []


def test_canary_passes_when_pinned_and_loaded_and_fresh(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pass_result: list[int]
) -> None:
    _install(monkeypatch, _snap(manifest=_manifest(), source="build", pinned=BUILD))
    r = client.get("/canary/status")
    assert r.status_code == 200 and r.json() == {"status": "pass"}


def test_index_stale_is_unchanged_for_a_pinned_loaded_build(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pass_result: list[int]
) -> None:
    _install(monkeypatch, _snap(manifest=_manifest(age_h=40), source="build", pinned=BUILD))
    r = client.get("/canary/status")
    assert r.status_code == 503 and r.json()["assertion"] == "index_stale"


def test_canary_legacy_unpinned_is_unaffected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pass_result: list[int]
) -> None:
    _install(monkeypatch, _snap(manifest=None, source="flat"))
    assert client.get("/canary/status").status_code == 200
