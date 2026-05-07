#!/usr/bin/env python3
"""Run Alembic migrations against DATABASE_URL.

Usage
-----
    # Apply all pending migrations:
    DATABASE_URL=postgresql://... python scripts/run_migrations.py

    # Print the SQL that would be executed without applying it:
    DATABASE_URL=postgresql://... python scripts/run_migrations.py --dry-run

The script reads DATABASE_URL from the environment.  Copy .env.example to
.env, fill in your Supabase connection string, then source it:

    cp .env.example .env
    # edit .env — set DATABASE_URL
    export $(grep -v '^#' .env | xargs)
    python scripts/run_migrations.py

See SUPABASE_SETUP.md for full setup instructions.
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

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", _normalise_url(db_url))

    if args.dry_run:
        print("=== Dry run — SQL that would be applied (not executed) ===\n")
        command.upgrade(cfg, "head", sql=True)
        print("\n=== End of dry run ===")
    else:
        print(f"Connecting to: {_redact(db_url)}")
        command.upgrade(cfg, "head")
        print("Migrations applied successfully.")


if __name__ == "__main__":
    main()
