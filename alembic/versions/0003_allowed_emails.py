"""allowed_emails + allow-list enforcement hook

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-07

SUPABASE AUTH HOOK NOTE
-----------------------
This migration creates the allow-list table, the check helper, and the hook
function.  Wiring the hook to Supabase Auth requires a one-time dashboard step
that cannot be performed via SQL alone (Supabase Auth Hook API, 2026):

  Dashboard → Authentication → Hooks → Before User Created
  → select "Postgres Function" → choose public.before_user_created

On plain Postgres (local Docker / CI) there is no Supabase Auth to hook into,
so no signup enforcement is installed by this migration — the table and
functions are created but remain dormant until wired up.

A raw BEFORE INSERT trigger on auth.users was considered but rejected:
  - Supabase Auth sometimes inserts into auth.users using a service-role
    connection that bypasses row-level triggers during internal operations.
  - The managed Auth Hook is the officially supported mechanism and is not
    subject to those bypass paths.
  - Supabase explicitly warns that raw triggers on auth.users can break with
    Auth service updates.
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. allowed_emails
    #    Primary key is the email itself (TEXT) — avoids a surrogate UUID
    #    and makes the uniqueness constraint implicit.
    #    Mixed-case inserts are silently normalised to lowercase by the
    #    normalise_allowed_email trigger (see below).  The CHECK constraint
    #    is belt-and-suspenders: it should always pass after the trigger
    #    fires, but catches any path that bypasses the trigger.
    # ------------------------------------------------------------------
    op.create_table(
        "allowed_emails",
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("added_by", sa.UUID(), nullable=True),
        sa.Column(
            "added_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("note", sa.Text(), nullable=True),
        sa.CheckConstraint("email = lower(email)", name="allowed_emails_lowercase"),
        sa.ForeignKeyConstraint(["added_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("email"),
    )

    # BEFORE INSERT OR UPDATE trigger — lowercases the email column
    # automatically so callers don't need to remember.
    # The trigger runs before the CHECK constraint is evaluated, so
    # INSERT INTO allowed_emails VALUES ('User@Example.com') silently
    # stores 'user@example.com'.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.normalise_allowed_email()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $$
        BEGIN
            NEW.email := lower(NEW.email);
            RETURN NEW;
        END;
        $$;
        """
    )
    op.execute(
        """
        CREATE TRIGGER allowed_emails_normalise
        BEFORE INSERT OR UPDATE ON public.allowed_emails
        FOR EACH ROW EXECUTE FUNCTION public.normalise_allowed_email();
        """
    )

    # ------------------------------------------------------------------
    # 2. RLS
    #    SELECT is allowed for any authenticated user (auth.uid() IS NOT
    #    NULL) so the UI can show who's on the list.
    #    No INSERT / UPDATE / DELETE policies — mutations happen via the
    #    service-role key or SQL editor only.
    #    On plain Postgres (no auth.uid()), RLS is enabled but no policy
    #    is installed; superuser connections see all rows.
    # ------------------------------------------------------------------
    op.execute("ALTER TABLE allowed_emails ENABLE ROW LEVEL SECURITY;")

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_proc p
                JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = 'auth' AND p.proname = 'uid'
            ) THEN
                EXECUTE $pol$
                    CREATE POLICY allowed_emails_select ON allowed_emails
                        FOR SELECT USING (auth.uid() IS NOT NULL);
                $pol$;
            END IF;
        END
        $$;
        """
    )

    # ------------------------------------------------------------------
    # 3. check_email_allowed(email_to_check TEXT) → BOOLEAN
    #    SECURITY DEFINER so it can read allowed_emails regardless of the
    #    caller's row-level security context (the hook runs as the auth
    #    service, not as an authenticated user).
    #    STABLE — no side effects, result depends only on table contents.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.check_email_allowed(email_to_check TEXT)
        RETURNS BOOLEAN
        LANGUAGE sql
        SECURITY DEFINER
        STABLE
        SET search_path = public
        AS $$
            SELECT EXISTS (
                SELECT 1 FROM public.allowed_emails
                WHERE email = lower(email_to_check)
            );
        $$;
        """
    )

    # ------------------------------------------------------------------
    # 4. before_user_created(event jsonb) → jsonb
    #    Supabase Auth Hook function (Before User Created).
    #    Signature mandated by Supabase: receives a JSONB event, returns
    #    JSONB.  Return '{}'::jsonb to allow signup; return an error
    #    object to reject it.
    #
    #    Event payload shape:
    #      {
    #        "metadata": { ... },
    #        "user": { "id": "...", "email": "...", "app_metadata": {...} }
    #      }
    #
    #    Phone-based signups (no email) are passed through unchanged.
    #
    #    After migration: wire via
    #      Dashboard → Authentication → Hooks → Before User Created
    #      → Postgres Function → public.before_user_created
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.before_user_created(event jsonb)
        RETURNS jsonb
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public
        AS $$
        DECLARE
            signup_email TEXT;
        BEGIN
            signup_email := lower(event->'user'->>'email');

            -- Phone-only signups have no email; let them through.
            IF signup_email IS NULL OR signup_email = '' THEN
                RETURN '{}'::jsonb;
            END IF;

            IF NOT public.check_email_allowed(signup_email) THEN
                RETURN jsonb_build_object(
                    'error', jsonb_build_object(
                        'http_code', 403,
                        'message', 'This email is not on the access allow-list. '
                                   'Contact the team to request access.'
                    )
                );
            END IF;

            RETURN '{}'::jsonb;
        END;
        $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS public.before_user_created(jsonb);")
    op.execute("DROP FUNCTION IF EXISTS public.check_email_allowed(TEXT);")
    # Trigger is dropped implicitly when the table is dropped, but drop
    # the function explicitly since it lives in public schema.
    op.execute("DROP FUNCTION IF EXISTS public.normalise_allowed_email();")
    op.drop_table("allowed_emails")
