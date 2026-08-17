"""revoke anon/authenticated default PostgREST grants on every table

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-17

Every table in this database -- both the ones created by earlier alembic
migrations (which never issued an explicit GRANT, so Supabase's default
PostgREST auto-grant applied silently) and any created directly via the
dashboard -- carried the full default grant to `anon` and `authenticated`
(SELECT, INSERT, UPDATE, DELETE, TRUNCATE, TRIGGER, REFERENCES). Found
2026-08-17 during an estate-wide risk sweep: confirmed live via direct query
against production (information_schema.role_table_grants), same class of
finding review-iq's entire TRUNCATE-grant remediation was built around.

Live severity here, not theoretical: 5 tables (allowed_emails, conversations,
feedback, messages, users) have RLS enabled with real auth.uid()-based
policies, but RLS does not govern TRUNCATE. The other 6 tables
(alembic_version, demo_daily_stats, demo_rate_limits, pairing_stats,
saved_looks, styling_events) have NO RLS at all -- combined with the grant,
`anon` (the public key shipped in the frontend bundle) could DELETE any row
in any of them via a single unauthenticated PostgREST call
(DELETE /rest/v1/<table>) -- PostgREST has no verb mapping to TRUNCATE, so
DELETE was the actually-reachable vector, not TRUNCATE. demo_rate_limits and
demo_daily_stats specifically enforce DEMO_DAILY_REQUEST_CAP -- deleting rows
there resets the cap, meaning an anonymous caller could have driven uncapped
LLM spend, not just data loss.

Confirmed safe to revoke everything from both roles: frontend/lib/supabase/
{client,server}.ts use the Supabase client for auth only (grepped the whole
frontend for `.from(`, zero matches). The backend API connects to Postgres
directly as the `postgres` superuser (DATABASE_URL), never as anon/
authenticated -- neither role is used for any legitimate table access
anywhere in this codebase. Applied directly to production ahead of this
migration landing (Item 262, 2026-08-17) given the severity; this migration
exists so a fresh deploy (or anyone re-running migrations against a new
Supabase project) doesn't silently reintroduce the default grant.

Deliberately does not add RLS here -- that is a separate, already-tracked
follow-up (Item 264) for the 5 currently-unprotected tables. This migration
is grant-narrowing only, matching the revoke already live in production.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: Union[str, Sequence[str], None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = (
    "alembic_version",
    "allowed_emails",
    "conversations",
    "demo_daily_stats",
    "demo_rate_limits",
    "feedback",
    "messages",
    "pairing_stats",
    "saved_looks",
    "styling_events",
    "users",
)


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"REVOKE ALL ON public.{table} FROM anon, authenticated")


def downgrade() -> None:
    # Intentionally NOT restoring the default grant -- that grant was the
    # vulnerability this migration closes. If a future feature genuinely
    # needs direct PostgREST access to a specific table, add a narrow,
    # deliberate GRANT for exactly that table/role/privilege at that time.
    raise NotImplementedError(
        "This migration's downgrade is a no-op by design -- restoring the "
        "default anon/authenticated grant would reopen the vulnerability "
        "this migration exists to close. If a specific table genuinely "
        "needs PostgREST access, add a narrow GRANT for it explicitly "
        "instead of reverting this migration."
    )
