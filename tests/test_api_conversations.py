"""Tests for GET/POST/DELETE/PATCH /conversations."""
from __future__ import annotations

from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

import api.deps as deps
from api.main import app
from api.session import InMemorySessionStore


class _MockLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        return "ok"

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        yield "ok"

    def chat(self, messages: list[dict], **kwargs) -> str:
        return "ok"

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        yield "ok"


_MINIMAL_CONFIG = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
}

# Minimal session with two turns
def _make_session(llm: Any, config: dict) -> dict:
    from src.memory.conversation import ConversationMemory

    return {
        "messages": [
            {"role": "user", "content": "show me red dresses"},
            {"role": "assistant", "content": "Here are some red dresses."},
            {"role": "user", "content": "in blue instead"},
            {"role": "assistant", "content": "Here are some blue dresses."},
        ],
        "retrieved_items": [
            {
                "article_id": "111",
                "prod_name": "Floral Dress",
                "display_name": "Floral Dress (Blue)",
                "colour": "Blue",
                "product_type": "Dress",
                "department": "Ladieswear",
                "image_url": None,
                "detail_desc": None,
                "score": 0.9,
            }
        ],
        "filters": {"colour_group_name": "Blue", "product_type_name": "Dress"},
        "excluded_colours": None,
        "_memory": ConversationMemory(llm, config),
        "_summary": None,
        "_summary_message_count": 0,
    }


@pytest.fixture(autouse=True)
def inject_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemorySessionStore()
    llm = _MockLLM()
    monkeypatch.setattr(deps, "_session_store", store)
    monkeypatch.setattr(deps, "_llm", llm)
    monkeypatch.setattr(deps, "_config", _MINIMAL_CONFIG)
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")


@pytest.fixture
def client() -> TestClient:
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def seeded_cid(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pre-seed a two-turn conversation and return its ID."""
    store = deps.get_session_store()
    llm = deps.get_llm()
    config = deps.get_config()
    cid = "aaaabbbb-0000-0000-0000-000000000001"
    store.set(cid, _make_session(llm, config), deps.DEV_USER_ID)
    return cid


# ---------------------------------------------------------------------------
# GET /conversations
# ---------------------------------------------------------------------------

def test_list_conversations_empty(client: TestClient) -> None:
    resp = client.get("/conversations")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_conversations_returns_summary(client: TestClient, seeded_cid: str) -> None:
    resp = client.get("/conversations")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    item = data[0]
    assert item["conversation_id"] == seeded_cid
    assert item["message_count"] == 2
    assert "show me red dresses" in item["title"]
    assert item["last_message"] is not None


# ---------------------------------------------------------------------------
# GET /conversations/{id}
# ---------------------------------------------------------------------------

def test_get_conversation_detail(client: TestClient, seeded_cid: str) -> None:
    resp = client.get(f"/conversations/{seeded_cid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_id"] == seeded_cid
    assert len(data["messages"]) == 4
    first = data["messages"][0]
    assert first["role"] == "user"
    assert first["content"] == "show me red dresses"
    # id is None for in-memory session store (no DB backing)
    assert "id" in first
    assert len(data["retrieved_items"]) == 1
    assert data["retrieved_items"][0]["article_id"] == "111"


def test_get_conversation_404(client: TestClient) -> None:
    resp = client.get("/conversations/does-not-exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /conversations
# ---------------------------------------------------------------------------

def test_create_conversation(client: TestClient) -> None:
    resp = client.post("/conversations")
    assert resp.status_code == 201
    data = resp.json()
    assert "conversation_id" in data
    assert data["title"] == "New conversation"
    assert data["message_count"] == 0

    # Verify it now appears in the list
    list_resp = client.get("/conversations")
    ids = [c["conversation_id"] for c in list_resp.json()]
    assert data["conversation_id"] in ids


# ---------------------------------------------------------------------------
# DELETE /conversations/{id}
# ---------------------------------------------------------------------------

def test_delete_conversation(client: TestClient, seeded_cid: str) -> None:
    resp = client.delete(f"/conversations/{seeded_cid}")
    assert resp.status_code == 204

    # No longer in list
    list_resp = client.get("/conversations")
    assert list_resp.json() == []


def test_delete_conversation_404(client: TestClient) -> None:
    resp = client.delete("/conversations/does-not-exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /conversations/{id}
# ---------------------------------------------------------------------------

def test_patch_conversation_title(client: TestClient, seeded_cid: str) -> None:
    resp = client.patch(
        f"/conversations/{seeded_cid}",
        json={"title": "My custom title"},
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "My custom title"

    # Title persists on subsequent list
    list_resp = client.get("/conversations")
    assert list_resp.json()[0]["title"] == "My custom title"


def test_patch_conversation_404(client: TestClient) -> None:
    resp = client.patch("/conversations/does-not-exist", json={"title": "x"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# is_public field
# ---------------------------------------------------------------------------

def test_list_conversations_is_public_defaults_false(
    client: TestClient, seeded_cid: str
) -> None:
    resp = client.get("/conversations")
    assert resp.status_code == 200
    assert resp.json()[0]["is_public"] is False


def test_patch_conversation_is_public(client: TestClient, seeded_cid: str) -> None:
    resp = client.patch(
        f"/conversations/{seeded_cid}",
        json={"is_public": True},
    )
    assert resp.status_code == 200
    assert resp.json()["is_public"] is True

    # Persists on subsequent list
    list_resp = client.get("/conversations")
    assert list_resp.json()[0]["is_public"] is True


def test_patch_conversation_is_public_toggle_back(
    client: TestClient, seeded_cid: str
) -> None:
    client.patch(f"/conversations/{seeded_cid}", json={"is_public": True})
    resp = client.patch(f"/conversations/{seeded_cid}", json={"is_public": False})
    assert resp.status_code == 200
    assert resp.json()["is_public"] is False
