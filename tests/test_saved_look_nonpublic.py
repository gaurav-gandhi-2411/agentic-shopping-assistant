"""Saved looks: items from non-public stores render as unavailable, with no outbound link (V3d ii).

The snapshot stored at save time is never rewritten; the rule is applied when a look is read.
Uses the in-memory fallback (no DB engine), so no database or network is touched.
"""
from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.look_render import render_snapshot
from api.routes.looks import router as looks_router
from src.storage.saved_looks import get_look_memory

NON_PUBLIC = ["myntra", "flipkart", "bhasinbrothers", "priyaasi"]


def _item(article_id: str, store: str | None, **kw: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "article_id": article_id, "prod_name": f"Item {article_id}", "price_inr": 999,
        "image_url": f"https://img.example/{article_id}.jpg", "slot_role": "complement",
        "pdp_handle": f"handle-{article_id}",
        "buy_url": f"https://shop.example/{store}/{article_id}", "store": store,
    }
    item.update(kw)
    return item


def _snapshot(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": items,
        "rationale": "A look.",
        "cart_url": "https://shop.example/cart?ids=all",
        "item_links": [
            {"article_id": i["article_id"], "name": i["prod_name"], "buy_url": i["buy_url"]}
            for i in items
        ],
        "variant_label": "Base",
    }


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(looks_router)
    app.state.engine = None  # in-memory fallback
    return TestClient(app)


def _save_and_get(client: TestClient, snapshot: dict[str, Any], brand: str | None = None):
    body = {"session_id": "s1", "brand": brand, "snapshot": snapshot}
    new_id = client.post("/looks", json=body).json()["id"]
    resp = client.get(f"/looks/{new_id}")
    assert resp.status_code == 200
    return new_id, resp.json()


@pytest.mark.parametrize("store", NON_PUBLIC)
def test_non_public_item_is_unavailable_with_no_outbound_link(client, store: str) -> None:
    snap = _snapshot([_item("a1", store), _item("a2", "snitch")])
    _, data = _save_and_get(client, snap)
    hidden, shown = data["snapshot"]["items"]
    assert hidden["unavailable"] is True
    assert hidden["buy_url"] is None and hidden["pdp_handle"] is None
    assert [lk["article_id"] for lk in data["snapshot"]["item_links"]] == ["a2"]
    assert data["snapshot"]["cart_url"] is None  # a whole-look cart link would include a1
    assert f"shop.example/{store}" not in json.dumps(data)
    assert "unavailable" not in shown and shown["buy_url"] == "https://shop.example/snitch/a2"


def test_unknown_or_missing_store_fails_closed(client) -> None:
    snap = _snapshot([_item("u1", "no-such-store"), _item("u2", None)])
    _, data = _save_and_get(client, snap, brand=None)  # brand coalesces to "unified"
    assert all(i["unavailable"] is True and i["buy_url"] is None for i in data["snapshot"]["items"])
    assert data["snapshot"]["item_links"] == []


def test_public_only_look_renders_exactly_as_saved(client) -> None:
    snap = _snapshot([_item("p1", "snitch"), _item("p2", "libas")])
    _, data = _save_and_get(client, snap)
    assert data["snapshot"] == snap


def test_legacy_single_brand_item_uses_look_brand(client) -> None:
    public = _snapshot([_item("l1", None)])
    _, data = _save_and_get(client, public, brand="snitch")
    assert "unavailable" not in data["snapshot"]["items"][0]
    private = _snapshot([_item("l2", None)])
    _, data = _save_and_get(client, private, brand="hm")
    assert data["snapshot"]["items"][0]["unavailable"] is True


def test_stored_snapshot_is_not_rewritten(client) -> None:
    snap = _snapshot([_item("a1", "myntra"), _item("a2", "snitch")])
    original = copy.deepcopy(snap)
    new_id, data = _save_and_get(client, snap)
    assert data["snapshot"]["items"][0]["unavailable"] is True
    stored = get_look_memory(new_id)
    assert stored is not None and stored["snapshot"] == original


def test_include_non_public_env_keeps_dev_behaviour(client, monkeypatch) -> None:
    monkeypatch.setenv("INCLUDE_NON_PUBLIC_STORES", "1")
    snap = _snapshot([_item("a1", "myntra")])
    _, data = _save_and_get(client, snap)
    assert data["snapshot"] == snap


def test_render_snapshot_does_not_mutate_its_input() -> None:
    snap = _snapshot([_item("a1", "flipkart")])
    before = copy.deepcopy(snap)
    out = render_snapshot(snap, "unified")
    assert snap == before and out is not snap


@pytest.mark.parametrize("bad", [{}, {"items": None}, {"items": "x"}, {"items": [None, 3, "s"]}])
def test_render_snapshot_tolerates_malformed_snapshots(bad: dict[str, Any]) -> None:
    assert render_snapshot(bad, None) == bad
