"""GET /catalogue/{id}/similar takes retriever AND DataFrame from ONE snapshot per request.

A hot reload may install snapshot B while a request that started on A is in flight; the request
must finish entirely on A (neighbours from A's index, rows from A's catalogue).
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
from src.retrieval.snapshot import IndexSnapshot

_CONFIG: dict[str, Any] = {"retrieval": {"store_diversity": 0.0}}


def _row(aid: str, name: str) -> dict[str, Any]:
    return {
        "article_id": aid,
        "prod_name": name,
        "display_name": name,
        "gender": "women",
        "price_inr": 999.0,
        "image_url": None,
        "detail_desc": "d",
        "pdp_handle": None,
        "store": "fashor",
        "facets": {
            "colour_group_name": "Black",
            "product_type_name": "Top",
            "department_name": "x",
        },
    }


def _snapshot(rows: list[dict[str, Any]], neighbours: list[tuple[str, float]]) -> IndexSnapshot:
    df = pd.DataFrame(rows).set_index("article_id", drop=False)
    dense = MagicMock()
    dense.index = MagicMock(ntotal=len(rows) + 1)
    dense.search_by_id = lambda article_id, top_k=20: neighbours
    return IndexSnapshot(retriever=MagicMock(dense=dense), catalogue_df=df, build_id="x")


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(deps, "_config", _CONFIG)
    monkeypatch.setattr(deps, "_snapshot", None)
    monkeypatch.delenv("JWT_VERIFICATION_DISABLED", raising=False)
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_JWT_SECRET", "test-demo-secret-at-least-32-bytes-long")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")
    return TestClient(app, raise_server_exceptions=True)


def test_similar_finishes_on_the_snapshot_it_started_on(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _snapshot([_row("111", "A anchor"), _row("222", "A neighbour")], [("222", 0.9)])
    b = _snapshot(
        [_row("111", "B anchor"), _row("222", "B neighbour"), _row("999", "B only")],
        [("999", 0.9)],
    )
    deps.install_snapshot(a)
    real_get = deps.get_snapshot_or_none

    def get_then_swap() -> IndexSnapshot | None:
        snap = real_get()
        deps.install_snapshot(b)  # a reload lands right after the request took its snapshot
        return snap

    monkeypatch.setattr(deps, "get_snapshot_or_none", get_then_swap)
    token = create_demo_token(anon_id="anon:test-user", brand="hm")
    resp = client.get("/catalogue/111/similar", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    items = resp.json()
    assert [i["article_id"] for i in items] == ["222"]  # A's neighbours (B would say 999)
    assert items[0]["prod_name"] == "A neighbour"  # A's catalogue row, not B's
    assert deps.get_snapshot() is b  # the swap really happened

    # the NEXT request sees B
    monkeypatch.setattr(deps, "get_snapshot_or_none", real_get)
    resp2 = client.get("/catalogue/111/similar", headers={"Authorization": f"Bearer {token}"})
    assert [i["article_id"] for i in resp2.json()] == ["999"]
