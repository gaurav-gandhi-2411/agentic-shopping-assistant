"""Index age in /readyz and the ``index_stale`` canary assertion (Wave 1b V3(b), spec V3d).

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
from src.retrieval.snapshot import IndexSnapshot, index_age_hours

NOW = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
TOKEN = "freshness-test-token-0123456789"


def _manifest(age_h: float, build_id: str = "20261119T000000Z") -> BuildManifest:
    return BuildManifest(
        build_id=build_id,
        built_at=NOW - timedelta(hours=age_h),
        git_sha="abc123",
        stores=[StoreEntry(slug="libas", items=3), StoreEntry(slug="fashor", items=2)],
        total_items=5,
        vector_count=5,
        files=[],
    )


def _snapshot(manifest: BuildManifest | None, source: str = "build") -> IndexSnapshot:
    retriever = SimpleNamespace(dense=SimpleNamespace(index=SimpleNamespace(ntotal=5)))
    return IndexSnapshot(
        retriever=retriever,
        catalogue_df=pd.DataFrame({"article_id": list("abcde")}),
        manifest=manifest,
        build_id=manifest.build_id if manifest else "legacy",
        source=source,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(canary.INDEX_STRICT_ENV, raising=False)
    monkeypatch.setenv("CANARY_TOKEN", TOKEN)
    monkeypatch.setattr(canary, "_utcnow", lambda: NOW)
    monkeypatch.setattr(health, "_now", lambda: NOW)
    monkeypatch.setattr(deps, "_llm", SimpleNamespace())  # no .probe: readiness LLM check "ok"
    monkeypatch.setattr(deps, "_db_engine", None)
    monkeypatch.setattr(deps, "_reloader", None)


def _install(monkeypatch: pytest.MonkeyPatch, snap: IndexSnapshot | None) -> None:
    monkeypatch.setattr(deps, "_snapshot", snap)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)  # no lifespan: no index needed


@pytest.fixture()
def pass_result(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the (LLM-backed) canary run by a pass; the list counts how often it ran."""
    calls: list[int] = []

    async def fake_result() -> tuple[str, int, dict[str, Any]]:
        calls.append(1)
        return "cached", 200, {"status": "pass"}

    monkeypatch.setattr(canary, "_result", fake_result)
    return calls


# --------------------------------------------------------------------------- /canary/status


def test_fresh_index_passes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pass_result: list[int]
) -> None:
    _install(monkeypatch, _snapshot(_manifest(age_h=2.0)))
    r = client.get("/canary/status")
    assert r.status_code == 200 and r.json() == {"status": "pass"}


def test_stale_index_fails_with_fixed_safe_detail_and_skips_the_llm_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pass_result: list[int]
) -> None:
    _install(monkeypatch, _snapshot(_manifest(age_h=36.5)))
    r = client.get("/canary/status")
    assert r.status_code == 503
    assert r.json() == {
        "status": "fail",
        "assertion": "index_stale",
        "detail": "index older than 36 h",
    }
    assert pass_result == []  # a cached pass can never mask it, and no query is spent


@pytest.mark.parametrize(("age_h", "stale"), [(35.99, False), (36.0, False), (36.01, True)])
def test_36_hour_threshold_boundary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    pass_result: list[int],
    age_h: float,
    stale: bool,
) -> None:
    _install(monkeypatch, _snapshot(_manifest(age_h=age_h)))
    assert client.get("/canary/status").status_code == (503 if stale else 200)
    assert canary.INDEX_MAX_AGE_HOURS == 36.0


def test_legacy_index_without_manifest_does_not_fail_the_canary(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pass_result: list[int]
) -> None:
    _install(monkeypatch, _snapshot(None, source="flat"))
    assert client.get("/canary/status").status_code == 200


def test_strict_switch_makes_a_missing_manifest_fail(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pass_result: list[int]
) -> None:
    monkeypatch.setenv(canary.INDEX_STRICT_ENV, "1")
    _install(monkeypatch, _snapshot(None, source="flat"))
    r = client.get("/canary/status")
    assert r.status_code == 503 and r.json()["assertion"] == "index_stale"
    _install(monkeypatch, _snapshot(_manifest(age_h=1.0)))  # strict does not fail a fresh build
    assert client.get("/canary/status").status_code == 200


def test_no_snapshot_yet_is_not_an_index_stale_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pass_result: list[int]
) -> None:
    monkeypatch.setenv(canary.INDEX_STRICT_ENV, "1")
    _install(monkeypatch, None)
    assert client.get("/canary/status").status_code == 200


def test_token_canary_endpoint_also_reports_index_stale(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pass_result: list[int]
) -> None:
    _install(monkeypatch, _snapshot(_manifest(age_h=40.0)))
    r = client.get("/canary", params={"token": TOKEN})
    assert r.status_code == 500
    body = r.json()
    assert body["status"] == "fail" and body["failing_assertion"] == "index_stale"
    assert pass_result == []


# --------------------------------------------------------------------------- /readyz


def test_readyz_reports_build_age_stores_and_items(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _snapshot(_manifest(age_h=3.5)))
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["index"]["build_id"] == "20261119T000000Z"
    assert body["index"]["age_hours"] == 3.5
    assert body["index"]["stores"] == 2 and body["index"]["items"] == 5
    assert body["index"]["built_at"].startswith("2026-11-20T08:30:00")
    assert body["index"]["reload"] == {"hot_reload": "off"}
    assert body["checks"]["index"].startswith("ok (build_id=20261119T000000Z, age_hours=3.5")


def test_readyz_says_legacy_explicitly_and_stays_ready(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _snapshot(None, source="flat"))
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"
    assert r.json()["checks"]["index"] == "legacy (no manifest)"
    assert r.json()["index"]["layout"] == "legacy (no manifest)"


def test_readyz_stays_200_for_a_very_old_build(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _snapshot(_manifest(age_h=500.0)))
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["index"]["age_hours"] == 500.0  # canary's job


def test_index_age_helper_treats_a_naive_timestamp_as_utc() -> None:
    m = _manifest(age_h=1.0)
    m.built_at = (NOW - timedelta(hours=2)).replace(tzinfo=None)
    assert index_age_hours(_snapshot(m), NOW) == pytest.approx(2.0)
    assert index_age_hours(_snapshot(None), NOW) is None
