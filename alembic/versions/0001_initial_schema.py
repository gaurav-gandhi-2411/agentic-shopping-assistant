"""initial_schema

Revision ID: 0001
Revises:
Create Date: 2026-05-07 00:54:42.815258

RLS NOTE: RLS is enabled on all tables and policies are defined here,
but policies reference auth.uid() which only exists on Supabase.  On
plain Postgres (CI / local Docker) the policy block is skipped — RLS is
ENABLED but no policies are created, so superuser connections see all
rows.  Full RLS integration tests require a Supabase staging project and
are deferred to Phase 2 prompt 2.  See TESTING.md for details.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. Auth schema stub — only created when the real auth schema is
    #    absent (plain Postgres / CI).  Supabase already has auth.users.
    # ------------------------------------------------------------------
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.schemata
                WHERE schema_name = 'auth'
            ) THEN
                CREATE SCHEMA auth;
                CREATE TABLE auth.users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
            END IF;
        END
        $$;
        """
    )

    # ------------------------------------------------------------------
    # 2. users
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    # Cross-schema FK to auth.users cannot be expressed in SA metadata;
    # add via raw SQL so it works on both Supabase and plain Postgres.
    op.execute(
        """
        ALTER TABLE users
            ADD CONSTRAINT users_id_fkey
            FOREIGN KEY (id) REFERENCES auth.users(id) ON DELETE CASCADE;
        """
    )

    # ------------------------------------------------------------------
    # 3. conversations
    #    user_id is NOT NULL — every conversation must be owned.
    #    Tests seed a dev user row and pass its id explicitly.
    #    is_public: isolated by default, opt-in sharing.
    # ------------------------------------------------------------------
    op.create_table(
        "conversations",
        sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column(
            "is_public",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conversations_user_updated",
        "conversations",
        ["user_id", sa.text("updated_at DESC")],
    )

    # ------------------------------------------------------------------
    # 4. messages
    #    role restricted to ('user', 'assistant').
    #    Tool call inputs/outputs are serialised into tool_calls JSONB
    #    on the assistant row — not stored as separate 'tool' role rows.
    #    items JSONB holds the product card payload for assistant turns.
    #    filters JSONB holds the active filter snapshot for assistant turns
    #    (canonical location; no duplicate at conversation level).
    # ------------------------------------------------------------------
    op.create_table(
        "messages",
        sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("conversation_id", sa.UUID(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("items", postgresql.JSONB(), nullable=True, server_default=sa.text("'[]'")),
        sa.Column("tool_calls", postgresql.JSONB(), nullable=True, server_default=sa.text("'[]'")),
        sa.Column("filters", postgresql.JSONB(), nullable=True, server_default=sa.text("'{}'")),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="messages_role_check"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_messages_conversation_created",
        "messages",
        ["conversation_id", sa.text("created_at ASC")],
    )

    # ------------------------------------------------------------------
    # 5. feedback
    # ------------------------------------------------------------------
    op.create_table(
        "feedback",
        sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("message_id", sa.UUID(), nullable=False),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("rating IN (1, -1)", name="feedback_rating_check"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("message_id", name="uq_feedback_message_id"),
    )
    op.create_index("ix_feedback_message_id", "feedback", ["message_id"])

    # ------------------------------------------------------------------
    # 6. Row-level security
    # ------------------------------------------------------------------
    for table in ("users", "conversations", "messages", "feedback"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")

    # ------------------------------------------------------------------
    # 7. RLS policies — only created when auth.uid() exists (Supabase).
    #    On plain Postgres RLS is enabled but no policies are installed,
    #    so superuser connections are unrestricted.  See module docstring.
    # ------------------------------------------------------------------
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_proc p
                JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = 'auth' AND p.proname = 'uid'
            ) THEN
                -- users: own row only
                EXECUTE $pol$
                    CREATE POLICY users_self ON users
                        USING (id = auth.uid());
                $pol$;

                -- conversations: owner only (or public)
                EXECUTE $pol$
                    CREATE POLICY conversations_owner ON conversations
                        USING (user_id = auth.uid() OR is_public = true);
                $pol$;

                -- messages: via parent conversation
                EXECUTE $pol$
                    CREATE POLICY messages_owner ON messages
                        USING (
                            conversation_id IN (
                                SELECT id FROM conversations
                                WHERE user_id = auth.uid() OR is_public = true
                            )
                        );
                $pol$;

                -- feedback: via parent message -> conversation
                EXECUTE $pol$
                    CREATE POLICY feedback_owner ON feedback
                        USING (
                            message_id IN (
                                SELECT m.id FROM messages m
                                JOIN conversations c ON c.id = m.conversation_id
                                WHERE c.user_id = auth.uid() OR c.is_public = true
                            )
                        );
                $pol$;
            END IF;
        END
        $$;
        """
    )

    # ------------------------------------------------------------------
    # 8. Auth trigger — syncs Supabase Auth signups to public.users.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.handle_new_user()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public
        AS $func$
        BEGIN
            INSERT INTO public.users (id, email, created_at)
            VALUES (NEW.id, NEW.email, NEW.created_at)
            ON CONFLICT (id) DO NOTHING;
            RETURN NEW;
        END;
        $func$;
        """
    )

    # 9. Trigger — only when auth.users exists.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'auth' AND table_name = 'users'
            ) THEN
                EXECUTE $trig$
                    CREATE TRIGGER on_auth_user_created
                    AFTER INSERT ON auth.users
                    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
                $trig$;
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'auth' AND table_name = 'users'
            ) THEN
                DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
            END IF;
        END
        $$;
        """
    )
    op.execute("DROP FUNCTION IF EXISTS public.handle_new_user();")

    op.drop_index("ix_feedback_message_id", table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_messages_conversation_created", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_conversations_user_updated", table_name="conversations")
    op.drop_table("conversations")
    op.drop_table("users")

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.schemata
                WHERE schema_name = 'auth'
            ) AND NOT EXISTS (
                SELECT 1 FROM auth.users LIMIT 1
            ) THEN
                DROP TABLE IF EXISTS auth.users;
                DROP SCHEMA IF EXISTS auth;
            END IF;
        END
        $$;
        """
    )
