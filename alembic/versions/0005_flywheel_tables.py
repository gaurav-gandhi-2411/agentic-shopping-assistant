"""styling_events + pairing_stats tables for the conversion-data flywheel

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-08

styling_events: append-only log of look interactions. Never UPDATE or DELETE.
  Each row is one event (look_shown, item_view, add_single, add_the_look,
  swap_slot, thumbs_up, thumbs_down) tagged with the full look context.
  look_total_inr is stored at event time (sum of filled_slots prices) so
  the dashboard query needs no join to recompute basket size.

pairing_stats: materialized aggregation keyed on (anchor_category, fill_category, occasion).
  Refreshed by the application (src/flywheel/stats.py) after every N new events.
  positive_rate is computed at query time: (add_the_look + thumbs_up) /
  (add_the_look + thumbs_up + thumbs_down + add_single_only).
  This formula lets thumbs_down AND single-item purchases drag the boost — not just thumbs_up.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: Union[str, Sequence[str], None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. styling_events
    #    Append-only event log — one row per user interaction with a look.
    #    Never UPDATE or DELETE rows; this is the raw signal source.
    #    look_total_inr is denormalised at write time so dashboard queries
    #    need no join to recompute basket size over time windows.
    # ------------------------------------------------------------------
    op.create_table(
        "styling_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=True),
        sa.Column("look_id", sa.Text(), nullable=False),
        sa.Column("anchor_item_id", sa.Text(), nullable=False),
        sa.Column("anchor_category", sa.Text(), nullable=False),
        sa.Column("filled_slots", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("occasion", sa.Text(), nullable=True),
        sa.Column("price_band", sa.Text(), nullable=True),
        sa.Column("brand", sa.Text(), nullable=True),
        sa.Column("look_total_inr", sa.Integer(), nullable=True),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    # Time-range queries (e.g. "last 30 days of events").
    op.create_index(
        "ix_styling_events_created_at",
        "styling_events",
        ["created_at"],
    )
    # Dashboard segmentation by occasion and brand without full-table scans.
    op.create_index(
        "ix_styling_events_occasion_brand",
        "styling_events",
        ["occasion", "brand"],
    )

    # ------------------------------------------------------------------
    # 2. pairing_stats
    #    Materialized aggregation keyed on (anchor_category, fill_category,
    #    occasion).  Refreshed via INSERT ... ON CONFLICT upserts in
    #    src/flywheel/stats.py — not via a DB trigger or scheduled job.
    #    No foreign keys to other tables: flywheel data is intentionally
    #    decoupled from auth/session tables to avoid cascading deletes.
    # ------------------------------------------------------------------
    op.create_table(
        "pairing_stats",
        sa.Column("anchor_category", sa.Text(), nullable=False),
        sa.Column("fill_category", sa.Text(), nullable=False),
        sa.Column("occasion", sa.Text(), nullable=False),
        sa.Column("looks_shown", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("add_the_look", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("thumbs_up", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("thumbs_down", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("add_single_only", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("anchor_category", "fill_category", "occasion"),
    )


def downgrade() -> None:
    # Drop in reverse creation order.  No foreign keys, so order is arbitrary
    # between the two new tables, but both must be dropped before any table
    # they might reference in future (none currently).
    op.drop_table("pairing_stats")
    op.drop_index("ix_styling_events_occasion_brand", table_name="styling_events")
    op.drop_index("ix_styling_events_created_at", table_name="styling_events")
    op.drop_table("styling_events")
