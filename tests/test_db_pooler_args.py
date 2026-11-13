"""E9: every engine that may talk to the pgbouncer pooler disables psycopg3 auto-prepare.

No network, no real DB: create_engine is captured with a monkeypatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import src.utils.db_url as db_url

_ROOT = Path(__file__).resolve().parent.parent
_URL = "postgresql://u:pw@db.example.invalid:6543/postgres"

# The only places allowed to call create_engine( directly.  Everything else must go through
# src.utils.db_url.create_db_engine so it inherits prepare_threshold=None.
_ALLOWED = {
    "src/utils/db_url.py",
    "alembic/env.py",
    # CI-only ephemeral Postgres on a direct connection (no pooler).
    "scripts/ops/check_migrations.py",
}


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_create_engine(url: Any, **kwargs: Any) -> object:
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("sqlalchemy.create_engine", fake_create_engine)
    return seen


def test_create_db_engine_disables_auto_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    db_url.create_db_engine(_URL, pool_size=5)
    assert seen["kwargs"]["connect_args"] == {"prepare_threshold": None}
    assert seen["kwargs"]["pool_size"] == 5


def test_create_db_engine_merges_caller_connect_args(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    db_url.create_db_engine(_URL, connect_args={"connect_timeout": 3})
    assert seen["kwargs"]["connect_args"] == {"prepare_threshold": None, "connect_timeout": 3}


def test_caller_connect_args_win_on_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    db_url.create_db_engine(_URL, connect_args={"prepare_threshold": 5})
    assert seen["kwargs"]["connect_args"] == {"prepare_threshold": 5}


def test_alembic_env_disables_auto_prepare() -> None:
    src = (_ROOT / "alembic" / "env.py").read_text()
    assert "connect_args=pooler_connect_args()" in src


def test_no_new_raw_create_engine_call_sites() -> None:
    offenders: list[str] = []
    for top in ("api", "src", "scripts", "alembic", "spaces", "qa", "eval", "evals", "brands"):
        base = _ROOT / top
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(_ROOT).as_posix()
            if rel in _ALLOWED or "/node_modules/" in rel:
                continue
            if "create_engine(" in path.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(rel)
    assert not offenders, f"use src.utils.db_url.create_db_engine instead: {offenders}"
