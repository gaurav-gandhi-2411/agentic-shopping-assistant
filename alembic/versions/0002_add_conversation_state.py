"""add_conversation_state

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-07

Adds two columns to conversations:

  excluded_colours JSONB DEFAULT NULL
      Persists the colour-negation list set by the grounding node when a user
      expresses a colour sensitivity (e.g. "no red").  Without this column the
      value is dropped on every store.get(), causing the search node to ignore
      colour exclusions after the first server restart or cross-process load.

  summary TEXT DEFAULT NULL
      Cache slot for ConversationMemory._cached_summary.  Task 3 (Postgres-
      backed ConversationMemory) will write here; adding the column now avoids
      a third single-column migration.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column(
            "excluded_colours",
            postgresql.JSONB(),
            nullable=True,
            server_default=None,
        ),
    )
    op.add_column(
        "conversations",
        sa.Column("summary", sa.Text(), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "summary_message_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )


def downgrade() -> None:
    op.drop_column("conversations", "summary_message_count")
    op.drop_column("conversations", "summary")
    op.drop_column("conversations", "excluded_colours")
