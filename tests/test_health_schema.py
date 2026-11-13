"""/readyz schema-head gate: DB alembic_version must equal the head(s) shipped in the image.

Hermetic: in-memory SQLite engines stand in for Postgres (the gate issues one portable
``SELECT version_num FROM alembic_version``); no network, no real DATABASE_URL.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

import api.deps as deps
from api.main import app
from api.routes import health
from src.llm.resilience import ProbeCache

ROOT = Path(__file__).resolve().parents[1]
SECRET_URL = "postgresql://user:hunter2-secret@db.example.com:5432/postgres"


def _engine(revisions: list[str] | None) -> Any:
    """SQLite engine; ``revisions=None`` means the alembic_version table does not exist."""
    eng = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    if revisions is not None:
        with eng.begin() as conn:
            conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
            for rev in revisions:
                conn.execute(text("INSERT INTO alembic_version VALUES (:r)"), {"r": rev})
    return eng


@pytest.fixture
def heads(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin the image head to 0009 and return it."""
    _set_heads(monkeypatch, ("0009",))
    return "0009"


def _set_heads(monkeypatch: pytest.MonkeyPatch, value: tuple[str, ...]) -> None:
    monkeypatch.setattr(health, "_image_heads", lambda: value)


# ----------------------------------------------------------------------------- check_schema_head


def test_equal_is_ok(heads: str) -> None:
    assert health.check_schema_head(_engine(["0009"])) == (True, "ok (0009)")


def test_behind_fails_with_both_revisions_named(heads: str) -> None:
    ok, msg = health.check_schema_head(_engine(["0008"]))
    assert not ok and msg == "error: db at 0008, image expects 0009"


def test_ahead_fails(heads: str) -> None:
    ok, msg = health.check_schema_head(_engine(["0010"]))
    assert not ok and msg == "error: db at 0010, image expects 0009"


def test_empty_version_table_fails(heads: str) -> None:
    ok, msg = health.check_schema_head(_engine([]))
    assert not ok and msg == "error: db at none, image expects 0009"


def test_missing_table_fails_without_leaking_driver_text(heads: str) -> None:
    ok, msg = health.check_schema_head(_engine(None))
    assert not ok and msg == "error: db schema unreadable (OperationalError), image expects 0009"
    assert "no such table" not in msg  # exception TYPE only, never its text


def test_db_error_fails_and_never_echoes_the_url(heads: str) -> None:
    class Boom:
        def connect(self) -> Any:
            raise RuntimeError(f"could not connect to {SECRET_URL}")

    ok, msg = health.check_schema_head(Boom())
    assert not ok and msg.startswith("error: db schema unreadable (RuntimeError)")
    assert "hunter2" not in msg and "postgresql" not in msg


def test_multiple_heads_match_in_any_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_heads(monkeypatch, ("a1", "b2"))
    assert health.check_schema_head(_engine(["b2", "a1"]))[0] is True
    ok, msg = health.check_schema_head(_engine(["a1"]))  # only one of two heads applied
    assert not ok and msg == "error: db at a1, image expects a1, b2"
    assert not health.check_schema_head(_engine(["a1", "b2", "c3"]))[0]  # extra row


def test_unreadable_image_heads_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> tuple[str, ...]:
        raise FileNotFoundError(str(ROOT / "nope"))

    monkeypatch.setattr(health, "_image_heads", broken)
    ok, msg = health.check_schema_head(_engine(["0009"]))
    assert not ok and msg == "error: cannot read image migration heads (FileNotFoundError)"
    assert "nope" not in msg


def test_check_issues_exactly_one_select_and_no_ddl(heads: str) -> None:
    seen: list[str] = []
    eng = _engine(["0009"])
    from sqlalchemy import event

    event.listen(eng, "before_cursor_execute", lambda c, cur, stmt, *a: seen.append(stmt))
    assert health.check_schema_head(eng)[0]
    assert seen == ["SELECT version_num FROM alembic_version"]


def test_real_image_heads_come_from_the_repo_migrations() -> None:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    health._image_heads.cache_clear()
    expected = tuple(
        sorted(ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini"))).get_heads())
    )
    assert expected and health._image_heads() == expected


def test_dockerfile_ships_alembic_ini_and_scripts() -> None:
    """Without these COPYs the gate would fail closed on every deploy ('cannot read heads')."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY alembic.ini" in dockerfile and "COPY alembic/" in dockerfile
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert "alembic/" not in dockerignore and "alembic.ini" not in dockerignore


# ----------------------------------------------------------------------------- /readyz wiring


class _LLM:
    model = "m1"

    def probe(self) -> dict:
        return {"tier": "groq", "model": "m1", "latency_ms": 1}


@pytest.fixture
def ready_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
    monkeypatch.setattr(
        deps, "get_retriever", lambda: SimpleNamespace(dense=SimpleNamespace(index=None))
    )
    monkeypatch.setattr(deps, "get_catalogue_df", lambda: [])
    monkeypatch.setattr(deps, "get_llm", lambda: _LLM())
    monkeypatch.setattr(health, "_llm_probe_cache", ProbeCache(ttl_s=60))
    _set_heads(monkeypatch, ("0009",))
    return TestClient(app)


def test_readyz_ready_when_schema_equals_head(
    monkeypatch: pytest.MonkeyPatch, ready_client: TestClient
) -> None:
    monkeypatch.setattr(deps, "get_db_engine", lambda: _engine(["0009"]))
    r = ready_client.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"
    assert r.json()["checks"]["schema"] == "ok (0009)"


@pytest.mark.parametrize("revs", [["0008"], ["0010"], [], None])
def test_readyz_503_names_the_schema_check_when_not_at_head(
    monkeypatch: pytest.MonkeyPatch, ready_client: TestClient, revs: list[str] | None
) -> None:
    monkeypatch.setattr(deps, "get_db_engine", lambda: _engine(revs))
    r = ready_client.get("/readyz")
    body = r.json()
    assert r.status_code == 503 and body["status"] == "not ready"
    assert body["checks"]["database"] == "ok"  # reachable, just at the wrong revision
    assert (
        body["checks"]["schema"].startswith("error: ")
        and "image expects 0009" in body["checks"]["schema"]
    )


def test_readyz_without_engine_keeps_existing_behaviour(
    monkeypatch: pytest.MonkeyPatch, ready_client: TestClient
) -> None:
    monkeypatch.setattr(deps, "get_db_engine", lambda: None)
    r = ready_client.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"
    assert r.json()["checks"]["database"] == "not configured"
    assert r.json()["checks"]["schema"] == "not configured"


def test_readyz_db_down_skips_schema_and_masks_url(
    monkeypatch: pytest.MonkeyPatch, ready_client: TestClient
) -> None:
    class Down:
        def connect(self) -> Any:
            raise RuntimeError(f"connection to {SECRET_URL} refused")

    monkeypatch.setenv("DATABASE_URL", SECRET_URL)
    monkeypatch.setattr(deps, "get_db_engine", lambda: Down())
    r = ready_client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["checks"]["schema"] == "skipped (database unreachable)"
    assert "hunter2" not in r.text
