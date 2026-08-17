"""enable RLS with zero policies (deny-by-default) on the 5 unprotected tables

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-17

Item 264, defense in depth following 0007's grant revoke: demo_daily_stats,
demo_rate_limits, pairing_stats, saved_looks, and styling_events had no RLS
at all -- 0007 already closed the directly-reachable path (anon/authenticated
held zero grants after that migration), but RLS should still be enabled so a
future accidental GRANT (e.g. someone re-adding table access for a new
feature without thinking through row-level scope) doesn't silently reopen
full cross-tenant/anonymous access the way review-iq's TRUNCATE-grant
findings kept demonstrating all session.

Enabling RLS with ZERO policies is deliberate, not an oversight: Postgres's
own default when RLS is enabled and no policy matches a role is DENY, for
every command, to every non-superuser role. No app code needs a policy here
-- the backend connects as the `postgres` superuser (RLS does not apply to
the table owner or superuser roles), confirmed via DATABASE_URL's connection
role. If a future feature needs direct PostgREST row-level access to one of
these tables, add a real, narrow policy for it then -- do not add a
permissive USING(true) policy just to "make RLS work."

alembic_version is deliberately excluded: it's Alembic's own internal
version-tracking table (a single row), not application data, and Alembic
itself manages it outside any RLS-aware code path.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: Union[str, Sequence[str], None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = (
    "demo_daily_stats",
    "demo_rate_limits",
    "pairing_stats",
    "saved_looks",
    "styling_events",
)


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY")
