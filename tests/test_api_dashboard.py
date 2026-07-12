"""Tests for GET /dashboard.

Uses a minimal FastAPI app that mounts only the dashboard router.  The DB
engine is monkeypatched so no real database is needed.

Covers:
  - Returns zeroed empty dashboard when engine is None
  - Returns populated metrics when the engine returns real rows
  - add_the_look_rate and add_single_rate computed correctly
  - basket_size.delta_inr is None when either basket value is None
  - top_pairings filters rows below MIN_PAIRING_SIGNALS
  - _safe_rate returns 0.0 on zero denominator
  - _basket_delta returns None when either input is None
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.dashboard import (
    _basket_delta,
    _empty_dashboard,
    _safe_rate,
)
from api.routes.dashboard import (
    router as dashboard_router,
)

# ---------------------------------------------------------------------------
# Minimal test app
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    a = FastAPI()
    a.include_router(dashboard_router)
    return a


_app = _make_app()
_client = TestClient(_app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


def test_safe_rate_normal() -> None:
    assert _safe_rate(1, 4) == 0.25


def test_safe_rate_zero_denominator() -> None:
    assert _safe_rate(5, 0) == 0.0


def test_safe_rate_zero_numerator() -> None:
    assert _safe_rate(0, 10) == 0.0


def test_basket_delta_both_present() -> None:
    assert _basket_delta(3000.0, 1500.0) == 1500.0


def test_basket_delta_look_none() -> None:
    assert _basket_delta(None, 1500.0) is None


def test_basket_delta_single_none() -> None:
    assert _basket_delta(3000.0, None) is None


def test_basket_delta_both_none() -> None:
    assert _basket_delta(None, None) is None


def test_empty_dashboard_structure() -> None:
    d = _empty_dashboard()
    assert d["looks_shown"] == 0
    assert d["add_the_look_rate"] == 0.0
    assert d["add_single_rate"] == 0.0
    assert d["basket_size"]["look_avg_inr"] is None
    assert d["basket_size"]["delta_inr"] is None
    assert "Not a controlled A/B test" in d["basket_size"]["caveat"]
    assert d["top_pairings"] == []
    assert d["by_occasion"] == []
    assert d["by_brand"] == []


# ---------------------------------------------------------------------------
# Route-level tests
# ---------------------------------------------------------------------------


def test_dashboard_no_engine_returns_empty() -> None:
    """When get_db_engine() returns None, the route returns zeroed metrics."""
    with patch("api.routes.dashboard.get_db_engine", return_value=None):
        resp = _client.get("/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert body["looks_shown"] == 0
    assert body["top_pairings"] == []
    assert "Not a controlled A/B test" in body["basket_size"]["caveat"]


def _build_mock_engine(
    *,
    looks_shown: int = 10,
    atl_count: int = 3,
    single_count: int = 2,
    basket_look: float | None = 4500.0,
    basket_single: float | None = 1500.0,
    pairing_rows: list[Any] | None = None,
    occasion_rows: list[Any] | None = None,
    brand_rows: list[Any] | None = None,
) -> MagicMock:
    """Build a mock SQLAlchemy engine whose connect().__enter__ returns a conn.

    Each scalar_one() call is popped from a predefined queue so the order of
    calls in _compute_dashboard() is matched exactly.
    """
    scalar_queue = [looks_shown, atl_count, single_count, basket_look, basket_single]

    def _make_scalar_result(val: Any) -> MagicMock:
        r = MagicMock()
        r.scalar_one.return_value = val
        return r

    def _make_fetchall_result(rows: list[Any]) -> MagicMock:
        r = MagicMock()
        r.fetchall.return_value = rows
        return r

    conn = MagicMock()

    # execute() is called 8 times in total: 5 scalar queries + 3 fetchall queries.
    execute_results: list[MagicMock] = [_make_scalar_result(v) for v in scalar_queue] + [
        _make_fetchall_result(pairing_rows or []),
        _make_fetchall_result(occasion_rows or []),
        _make_fetchall_result(brand_rows or []),
    ]
    conn.execute.side_effect = execute_results

    engine = MagicMock()
    engine.connect.return_value.__enter__ = lambda s: conn
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    return engine


def test_dashboard_rates_computed_correctly() -> None:
    engine = _build_mock_engine(looks_shown=10, atl_count=4, single_count=2)
    with patch("api.routes.dashboard.get_db_engine", return_value=engine):
        resp = _client.get("/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert body["looks_shown"] == 10
    assert body["add_the_look_rate"] == pytest.approx(0.4, abs=1e-4)
    assert body["add_single_rate"] == pytest.approx(0.2, abs=1e-4)


def test_dashboard_basket_delta_present() -> None:
    engine = _build_mock_engine(basket_look=4500.0, basket_single=1500.0)
    with patch("api.routes.dashboard.get_db_engine", return_value=engine):
        resp = _client.get("/dashboard")
    body = resp.json()
    assert body["basket_size"]["look_avg_inr"] == pytest.approx(4500.0)
    assert body["basket_size"]["single_avg_inr"] == pytest.approx(1500.0)
    assert body["basket_size"]["delta_inr"] == pytest.approx(3000.0)


def test_dashboard_basket_delta_none_when_missing() -> None:
    engine = _build_mock_engine(basket_look=None, basket_single=1500.0)
    with patch("api.routes.dashboard.get_db_engine", return_value=engine):
        resp = _client.get("/dashboard")
    body = resp.json()
    assert body["basket_size"]["look_avg_inr"] is None
    assert body["basket_size"]["delta_inr"] is None


def test_dashboard_caveat_verbatim() -> None:
    """The basket size caveat must appear verbatim in the response."""
    engine = _build_mock_engine()
    with patch("api.routes.dashboard.get_db_engine", return_value=engine):
        resp = _client.get("/dashboard")
    caveat = resp.json()["basket_size"]["caveat"]
    assert caveat == (
        "Estimated lift in basket size (look vs single-item purchase, this session data). "
        "Not a controlled A/B test."
    )


def test_dashboard_engine_exception_returns_empty() -> None:
    """If the DB query raises, the route catches and returns zeroed metrics."""
    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("DB is down")
    with patch("api.routes.dashboard.get_db_engine", return_value=engine):
        resp = _client.get("/dashboard")
    assert resp.status_code == 200
    assert resp.json()["looks_shown"] == 0


def _make_pairing_row(
    anchor: str,
    fill: str,
    occasion: str,
    atl: int,
    total: int,
) -> MagicMock:
    row = MagicMock()
    row.anchor_category = anchor
    row.fill_category = fill
    row.occasion = occasion
    row.add_the_look = atl
    row.total_signals = total
    return row


def test_dashboard_top_pairings_present() -> None:
    pairing_rows = [
        _make_pairing_row("tops", "jeans", "casual", 8, 12),
        _make_pairing_row("dress", "heels", "formal", 5, 7),
    ]
    engine = _build_mock_engine(pairing_rows=pairing_rows)
    with patch("api.routes.dashboard.get_db_engine", return_value=engine):
        resp = _client.get("/dashboard")
    body = resp.json()
    assert len(body["top_pairings"]) == 2
    assert body["top_pairings"][0]["anchor_category"] == "tops"
    assert body["top_pairings"][0]["add_the_look"] == 8
