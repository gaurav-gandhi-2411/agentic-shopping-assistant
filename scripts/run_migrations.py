#!/usr/bin/env python3
"""Run Alembic migrations against DATABASE_URL.

Usage
-----
    # Apply all pending migrations:
    DATABASE_URL=postgresql://... python scripts/run_migrations.py

    # Apply with the auth shim (local Postgres — no Supabase auth.users):
    APPLY_AUTH_SHIM=true DATABASE_URL=postgresql://... python scripts/run_migrations.py
    # or equivalently:
    DATABASE_URL=postgresql://... python scripts/run_migrations.py --create-auth-shim

    # Print the SQL that would be executed without connecting:
    DATABASE_URL=postgresql://... python scripts/run_migrations.py --dry-run

The script reads DATABASE_URL from the environment.  Copy .env.example to
.env, fill in your Supabase connection string, then source it:

    cp .env.example .env
    # edit .env — set DATABASE_URL
    export $(grep -v '^#' .env | xargs)
    python scripts/run_migrations.py

See SUPABASE_SETUP.md for full setup instructions including the local
Docker dev-loop and the Supabase production workflow.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Run from repo root so alembic.ini is found regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))


def _normalise_url(url: str) -> str:
    """Add the +psycopg driver qualifier required by SQLAlchemy 2.x + psycopg3."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def _redact(url: str) -> str:
    """Replace the password in a connection string with *** for display."""
    return re.sub(r"(:)([^:@/][^:@]*)(@)", r"\1***\3", url)


def apply_auth_shim(engine) -> None:
    """Create the auth schema and a minimal auth.users stub.

    This is a no-op on Supabase (objects already exist).  Call it before
    running Alembic migrations on plain Postgres (local Docker, CI) where
    the auth schema is absent, because migration 0001 adds a FK that
    REFERENCES auth.users(id).

    Importable so tests/integration/conftest.py can reuse the same SQL
    rather than duplicating it.
    """
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS auth"))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS auth.users (
                id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email      TEXT,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Alembic migrations to the database at DATABASE_URL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the SQL that would be applied and print it — do not connect.",
    )
    parser.add_argument(
        "--create-auth-shim",
        action="store_true",
        help=(
            "Before running migrations, create the auth schema and a minimal "
            "auth.users stub.  Required on plain Postgres (local Docker/CI) "
            "because migration 0001 REFERENCES auth.users(id).  "
            "Also enabled by APPLY_AUTH_SHIM=true in the environment.  "
            "On Supabase, auth.users already exists — do not pass this flag."
        ),
    )
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        print(
            "ERROR: DATABASE_URL is not set.\n"
            "\n"
            "Export the variable before running this script:\n"
            "  export DATABASE_URL=postgresql://postgres.<ref>:<pass>@<host>:5432/postgres\n"
            "\n"
            "Or copy .env.example → .env, fill in the value, then:\n"
            "  export $(grep -v '^#' .env | xargs)\n"
            "\n"
            "See SUPABASE_SETUP.md for where to find the connection string.",
            file=sys.stderr,
        )
        sys.exit(1)

    use_shim = args.create_auth_shim or os.environ.get("APPLY_AUTH_SHIM", "").lower() in (
        "1", "true", "yes",
    )

    from alembic.config import Config

    from alembic import command

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", _normalise_url(db_url))

    if args.dry_run:
        # sql=True generates migration SQL without connecting to the database.
        # The auth shim is not needed here — no real FK validation occurs.
        print("=== Dry run — SQL that would be applied (not executed) ===\n")
        command.upgrade(cfg, "head", sql=True)
        print("\n=== End of dry run ===")
        return

    print(f"Connecting to: {_redact(db_url)}")

    if use_shim:
        from sqlalchemy import create_engine

        print("Creating auth schema shim (local Postgres mode)...")
        engine = create_engine(_normalise_url(db_url))
        try:
            apply_auth_shim(engine)
        finally:
            engine.dispose()
        print("Auth shim ready.")

    command.upgrade(cfg, "head")
    print("Migrations applied successfully.")


if __name__ == "__main__":
    main()
