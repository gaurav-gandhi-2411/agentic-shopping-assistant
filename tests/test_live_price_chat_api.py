"""POST /chat with the live price check wired in (mock transport only; no real store)."""
from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
from api.main import app
from api.session import InMemorySessionStore
from src.pricing import live_check, metrics
from src.pricing.guards import ENV_FLAG, LiveConfig
from src.pricing.live_check import LivePriceChecker
from tests.test_api_chat import _DEFAULT_RESULT, _MINIMAL_CONFIG, _make_factory, _MockLLM


@pytest.fixture(autouse=True)
def _deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deps, "_session_store", InMemorySessionStore())
    monkeypatch.setattr(deps, "_llm", _MockLLM(["ok"]))
    monkeypatch.setattr(deps, "_config", _MINIMAL_CONFIG)
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")
    monkeypatch.delenv(ENV_FLAG, raising=False)
    metrics.reset()
    live_check.reset_checker()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


def _row(handle: str, store: str = "libas", price: float = 1000.0) -> dict[str, Any]:
    return {
        "article_id": f"{store}-{handle}", "prod_name": "Kurta", "display_name": "Kurta",
        "colour": "Red", "product_type": "Kurta", "department": "Women", "image_url": "u.jpg",
        "price_inr": price, "pdp_handle": handle, "store": store,
    }


def _result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {**_DEFAULT_RESULT, "retrieved_items": rows, "new_items_this_turn": True}


def _use_checker(monkeypatch: pytest.MonkeyPatch, handler, **cfg: Any) -> None:
    checker = LivePriceChecker(
        LiveConfig(enabled=True, min_interval_s=0.001, **cfg),
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr(live_check, "_enabled_checker", lambda: checker)


def _store_ok(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/robots.txt":
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")
    if request.url.path.endswith("/gone.js"):
        return httpx.Response(404)
    if request.url.path.endswith("/pricey.js"):
        return httpx.Response(200, json={"variants": [{"price": 150000, "available": True}]})
    return httpx.Response(200, json={"variants": [{"price": 100000, "available": True}]})


def test_flag_off_leaves_payload_unchanged(monkeypatch, client: TestClient) -> None:
    monkeypatch.setattr(deps, "get_agent_factory", _make_factory(_result([_row("gone")])))
    item = client.post("/chat", json={"message": "kurta"}).json()["items"][0]
    assert item["unavailable"] is False and item["price_may_have_changed"] is False
    assert item["indexed_price_inr"] is None and item["pdp_url"]


def test_shipped_config_yaml_default_is_off() -> None:
    import yaml

    with open("config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert LiveConfig.from_config(cfg).enabled is False
    assert cfg["live_price_check"]["budget_s"] == 1.5


def test_flag_on_updates_price_and_removes_unavailable(monkeypatch, client: TestClient) -> None:
    rows = [_row("gone"), _row("pricey"), _row("same")]
    monkeypatch.setattr(deps, "get_agent_factory", _make_factory(_result(rows)))
    _use_checker(monkeypatch, _store_ok, budget_s=1.0, burst=10)
    items = client.post("/chat", json={"message": "kurta"}).json()["items"]
    assert len(items) == 3  # count semantics unchanged
    assert [i["pdp_handle"] for i in items] == ["pricey", "same", "gone"]  # unavailable sinks
    pricey, same, gone = items
    assert pricey["price_inr"] == 1500.0 and pricey["indexed_price_inr"] == 1000.0
    assert same["price_inr"] == 1000.0 and same["indexed_price_inr"] is None
    assert gone["unavailable"] is True and gone["pdp_url"] is None


def test_slow_store_never_blocks_response_beyond_budget(monkeypatch, client: TestClient) -> None:
    import asyncio

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200)

    monkeypatch.setattr(deps, "get_agent_factory", _make_factory(_result([_row("a")])))
    _use_checker(monkeypatch, slow, budget_s=0.3)
    t0 = time.monotonic()
    resp = client.post("/chat", json={"message": "kurta"})
    assert time.monotonic() - t0 < 1.5
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["price_may_have_changed"] is True and item["price_inr"] == 1000.0
    assert item["pdp_url"]  # fail open: Buy stays unless positively unavailable


def test_ws_items_message_carries_live_check_and_sinks_unavailable(
    monkeypatch, client: TestClient
) -> None:
    import json

    rows = [_row("gone"), _row("pricey")]
    result = {
        **_result(rows),
        "final_answer": None,
        "current_plan": json.dumps({"action": "pending_respond", "prompt": "Describe."}),
    }

    class _Agent:
        def invoke(self, state: dict, **kwargs: Any) -> dict[str, Any]:
            return {**result, "messages": state.get("messages", [])}

    monkeypatch.setattr(deps, "get_agent_factory", lambda: lambda memory, streaming=False: _Agent())
    _use_checker(monkeypatch, _store_ok, budget_s=1.0, burst=10)
    with client.websocket_connect("/chat/stream") as ws:
        ws.send_json({"type": "user_message", "message": "kurta"})
        items = None
        for _ in range(40):
            msg = ws.receive_json()
            if msg["type"] == "items":
                items = msg["items"]
            if msg["type"] == "done":
                break
    assert items is not None and len(items) == 2
    assert [i["pdp_handle"] for i in items] == ["pricey", "gone"]
    assert items[0]["price_inr"] == 1500.0 and items[1]["unavailable"] is True


def test_non_public_store_card_never_fetched_over_api(monkeypatch, client: TestClient) -> None:
    seen: list[httpx.Request] = []

    def spy(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    rows = [_row("shirt-1", store="myntra"), _row("x", store="bhasinbrothers")]
    monkeypatch.setattr(deps, "get_agent_factory", _make_factory(_result(rows)))
    _use_checker(monkeypatch, spy)
    assert client.post("/chat", json={"message": "kurta"}).status_code == 200
    assert seen == []
