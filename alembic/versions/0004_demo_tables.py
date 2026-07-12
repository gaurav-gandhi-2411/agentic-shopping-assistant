"""demo_rate_limits + demo_daily_stats tables for anonymous demo session guards

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-07

demo_rate_limits — one row per (ip_hash, brand, UTC hour window).
  Stores a SHA-256[:32] hash of the client IP so raw addresses are never
  persisted.  The window_start column is always date_trunc('hour', NOW() AT
  TIME ZONE 'UTC') — the application enforces this; no DB trigger required.
  An index on window_start makes periodic cleanup scans fast.

demo_daily_stats — one row per (brand, UTC date).
  Accumulates request_count and cost_usd using INSERT ... ON CONFLICT upserts
  from the application layer.  NUMERIC(10, 6) gives six decimal places of
  precision for USD costs (micro-cent granularity).
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. demo_rate_limits
    #    Primary key is (ip_hash, brand, window_start) — one row per
    #    anonymous-IP × brand × UTC-hour.  Compound PK also acts as the
    #    unique index needed by the ON CONFLICT upsert in guards.py.
    # ------------------------------------------------------------------
    op.create_table(
        "demo_rate_limits",
        sa.Column("ip_hash", sa.Text(), nullable=False),
        sa.Column("brand", sa.Text(), nullable=False),
        sa.Column("window_start", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.PrimaryKeyConstraint("ip_hash", "brand", "window_start"),
    )
    # Secondary index so a cleanup job can efficiently delete rows older than
    # N hours without a full-table scan.
    op.create_index(
        "ix_demo_rate_limits_window",
        "demo_rate_limits",
        ["window_start"],
    )

    # ------------------------------------------------------------------
    # 2. demo_daily_stats
    #    Primary key is (brand, date) — one row per brand per UTC day.
    #    NUMERIC(10, 6) stores up to 9999.999999 USD with micro-cent
    #    precision — more than enough headroom for daily cost tracking.
    # ------------------------------------------------------------------
    op.create_table(
        "demo_daily_stats",
        sa.Column("brand", sa.Text(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "cost_usd",
            sa.Numeric(10, 6),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.PrimaryKeyConstraint("brand", "date"),
    )


def downgrade() -> None:
    # No foreign keys between the two tables, so order is arbitrary.
    op.drop_table("demo_daily_stats")
    op.drop_index("ix_demo_rate_limits_window", table_name="demo_rate_limits")
    op.drop_table("demo_rate_limits")
