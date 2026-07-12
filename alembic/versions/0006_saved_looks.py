"""saved_looks table for anonymous-session-scoped look persistence and sharing

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-13

saved_looks — one row per saved look.  The UUID primary key doubles as the
  public share slug: GET /look/{id} renders the read-only shared board without
  any secondary slug column.  session_id tags the row to the anonymous demo
  session that created it; user_id is NULL for anonymous users and is set to the
  authenticated user's id when auth is present.

  snapshot is a self-contained JSONB payload (items[], rationale, cart_url,
  item_links, variant label) so the shared board can render without any
  catalogue look-up.  Immutable after insert — no UPDATE or DELETE paths exist
  in the application layer.

  ix_saved_looks_session on (session_id, created_at DESC) supports the common
  query "fetch this session's saved looks, newest first".
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: Union[str, Sequence[str], None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # saved_looks
    #   Append-only store for user-saved outfits.  id is gen_random_uuid()
    #   so it also serves as the public share slug — no separate slug column
    #   needed.  snapshot is self-contained: the shared board renders from
    #   this column alone without any catalogue query.
    #   user_id is NULL for anonymous demo sessions (the common case); FK to
    #   users is deferred to when auth is fully wired.
    # ------------------------------------------------------------------
    op.create_table(
        "saved_looks",
        sa.Column(
            "id",
            sa.dialects.postgresql.UUID(as_uuid=False),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column(
            "user_id",
            sa.dialects.postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        sa.Column("brand", sa.Text(), nullable=False),
        sa.Column("look_id", sa.Text(), nullable=True),
        sa.Column("occasion", sa.Text(), nullable=True),
        sa.Column("look_gender", sa.Text(), nullable=True),
        sa.Column("anchor_item_id", sa.Text(), nullable=True),
        sa.Column("look_total_inr", sa.Integer(), nullable=True),
        sa.Column("snapshot", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
    )
    # Session-scoped listing: "show me all looks I saved this session, newest first".
    op.create_index(
        "ix_saved_looks_session",
        "saved_looks",
        ["session_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    # Drop index before table (required).
    op.drop_index("ix_saved_looks_session", table_name="saved_looks")
    op.drop_table("saved_looks")
