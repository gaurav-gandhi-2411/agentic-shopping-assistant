"""Tests for GET /catalogue/{article_id}/similar auth handling.

Regression coverage for a live incident: the route used
``get_current_user_id`` (Supabase RS256-only), which threw an unhandled
exception (HTTP 500 + a browser CORS block, since no CORS headers are
attached to an unhandled-exception response) when a demo-mode HS256 session
token was presented. The fix swaps in ``get_current_user_id_or_demo`` — the
same dependency already used by the chat/image routes — so demo tokens are
accepted and garbage tokens cleanly 401 instead of 500.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
from api.demo.session import create_demo_token
from api.main import app

_MINIMAL_CONFIG: dict[str, Any] = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
}


def _catalogue_df() -> pd.DataFrame:
    """Return a tiny catalogue with an anchor item and one neighbour."""
    rows = [
        {
            "article_id": "111",
            "prod_name": "Floral Top",
            "display_name": "Floral Top (White Top)",
            "gender": "women",
            "price_inr": 999.0,
            "image_url": None,
            "detail_desc": "A white floral top.",
            "pdp_handle": None,
            "store": "hm",
            "facets": {
                "colour_group_name": "White",
                "product_type_name": "Top",
                "department_name": "Ladieswear",
            },
        },
        {
            "article_id": "222",
            "prod_name": "Linen Shirt",
            "display_name": "Linen Shirt (Beige)",
            "gender": "women",
            "price_inr": 1299.0,
            "image_url": None,
            "detail_desc": "A beige linen shirt.",
            "pdp_handle": None,
            "store": "hm",
            "facets": {
                "colour_group_name": "Beige",
                "product_type_name": "Shirt",
                "department_name": "Ladieswear",
            },
        },
    ]
    return pd.DataFrame(rows).set_index("article_id", drop=False)


class _MockDense:
    """Minimal stand-in for HybridRetriever.dense used by the /similar route."""

    def __init__(self, neighbours: list[tuple[str, float]]) -> None:
        self._neighbours = neighbours
        self.index = MagicMock(ntotal=len(neighbours) + 1)

    def search_by_id(self, article_id: str, top_k: int = 20) -> list[tuple[str, float]]:  # noqa: ARG002
        return self._neighbours


class _MockRetriever:
    def __init__(self, neighbours: list[tuple[str, float]]) -> None:
        self.dense = _MockDense(neighbours)


@pytest.fixture(autouse=True)
def inject_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject minimal deps so the FastAPI app starts without real indices."""
    monkeypatch.setattr(deps, "_config", _MINIMAL_CONFIG)
    monkeypatch.setattr(deps, "_catalogue_df", _catalogue_df())
    monkeypatch.setattr(deps, "_retriever", _MockRetriever([("222", 0.87)]))
    # Verification must run for real so we exercise the 401 path honestly —
    # JWT_VERIFICATION_DISABLED would short-circuit auth entirely.
    monkeypatch.delenv("JWT_VERIFICATION_DISABLED", raising=False)
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_JWT_SECRET", "test-demo-secret-at-least-32-bytes-long")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


def test_similar_accepts_demo_token(client: TestClient) -> None:
    """A valid HS256 demo token must be accepted (200), not blow up with 500."""
    token = create_demo_token(anon_id="anon:test-user", brand="hm")

    resp = client.get(
        "/catalogue/111/similar",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200, resp.text
    items = resp.json()
    assert isinstance(items, list)
    assert len(items) >= 1
    assert items[0]["article_id"] == "222"


def test_similar_rejects_garbage_token_with_401_not_500(client: TestClient) -> None:
    """An invalid/garbage bearer token must 401 cleanly, never 500."""
    resp = client.get(
        "/catalogue/111/similar",
        headers={"Authorization": "Bearer not.a.valid.jwt"},
    )

    assert resp.status_code == 401, resp.text


def test_similar_rejects_missing_auth_header(client: TestClient) -> None:
    """No Authorization header at all must 401, not 500."""
    resp = client.get("/catalogue/111/similar")

    assert resp.status_code == 401, resp.text


def test_similar_unknown_article_returns_404(client: TestClient) -> None:
    """An article_id absent from the catalogue must 404, using a valid demo token."""
    token = create_demo_token(anon_id="anon:test-user", brand="hm")

    resp = client.get(
        "/catalogue/does-not-exist/similar",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 404, resp.text


def test_item_detail_accepts_demo_token(client: TestClient) -> None:
    """GET /catalogue/{article_id} (item detail) must accept a demo token too —
    same root cause as /similar: a demo user browsing item detail must not 500."""
    token = create_demo_token(anon_id="anon:test-user", brand="hm")

    resp = client.get(
        "/catalogue/111",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["article_id"] == "111"
