from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import create_engine, pool
from sqlalchemy.engine import URL

from alembic import context
from src.utils.db_url import parse_database_url, pooler_connect_args
from src.utils.redact import Redactor, install_log_redaction

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Mask DB passwords / API keys in alembic's own log handlers and error text.
install_log_redaction()

target_metadata = None


def _database_url() -> URL:
    """Return DATABASE_URL as a SQLAlchemy URL (psycopg3 driver, credentials encoded).

    Built from parts via parse_database_url, so a password with raw '@' etc. is safe.
    """
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise ValueError(
            "DATABASE_URL environment variable is required to run migrations. "
            "Example: DATABASE_URL=postgresql://user:pass@localhost/dbname alembic upgrade head"
        )
    return parse_database_url(url)


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (SQL script output mode)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    # prepare_threshold=None: DATABASE_URL may point at the pgbouncer transaction pooler (6543),
    # which breaks psycopg3 auto-prepared statements.
    connectable = create_engine(
        _database_url(),
        poolclass=pool.NullPool,
        connect_args=pooler_connect_args(),
    )
    try:
        connection = connectable.connect()
    except Exception as exc:
        # The interpreter prints uncaught tracebacks itself (no logging filter applies),
        # and driver errors echo the connect string.  Re-raise with masked text and no
        # chained original.
        raise RuntimeError(
            f"Could not connect to the database: {Redactor.from_env().redact(str(exc))}"
        ) from None
    with connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
