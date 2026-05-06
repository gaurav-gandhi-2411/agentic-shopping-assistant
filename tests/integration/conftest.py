"""Integration test fixtures.

Requires a running postgres:15 container with migrations applied.
See TESTING.md for setup instructions.

    DATABASE_URL=postgresql://postgres:test@localhost:5433/shopping \
        pytest tests/integration/

Tests are skipped automatically when DATABASE_URL is not set.
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEV_USER_ID = "00000000-0000-0000-0000-000000000001"

_RAW_URL = os.environ.get("DATABASE_URL", "")


def _normalise_url(url: str) -> str:
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


# ---------------------------------------------------------------------------
# Session-scoped engine + migration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pg_engine() -> Engine:
    if not _RAW_URL:
        pytest.skip("DATABASE_URL not set")
    engine = create_engine(_normalise_url(_RAW_URL), echo=False)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def run_migrations(pg_engine: Engine) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", str(pg_engine.url))
    command.upgrade(cfg, "head")


# ---------------------------------------------------------------------------
# Dev user — seeded once, idempotent
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dev_user_id(pg_engine: Engine, run_migrations: None) -> str:
    with pg_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO auth.users (id, email) "
                "VALUES (CAST(:uid AS uuid), 'dev@test.com') "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"uid": DEV_USER_ID},
        )
        conn.execute(
            text(
                "INSERT INTO users (id, email) "
                "VALUES (CAST(:uid AS uuid), 'dev@test.com') "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"uid": DEV_USER_ID},
        )
    return DEV_USER_ID


# ---------------------------------------------------------------------------
# Mock LLM + config
# ---------------------------------------------------------------------------

class _MockLLM:
    def generate(self, prompt: str) -> str:
        return "mock summary"


@pytest.fixture(scope="session")
def mock_llm() -> _MockLLM:
    return _MockLLM()


@pytest.fixture(scope="session")
def mock_config() -> dict:
    return {"memory": {"recent_turns": 4, "summary_trigger_turns": 2}}
