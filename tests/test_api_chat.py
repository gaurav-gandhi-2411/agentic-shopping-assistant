"""Tests for POST /chat.

Uses FastAPI's TestClient with lifespan disabled so no real indices are loaded.
Deps are injected via monkeypatch before each test.
"""
from __future__ import annotations

import uuid
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

import api.deps as deps
from api.main import app
from api.session import InMemorySessionStore

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockLLM:
    """Cycles through a list of canned responses; repeats the last when exhausted."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def _next(self) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        return self._next()

    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
        yield self._next()

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        yield self._next()


class _MockAgent:
    """Fake compiled LangGraph agent; invoke() returns a preset result dict."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def invoke(self, state: dict, **kwargs) -> dict[str, Any]:
        # Echo back the incoming messages so _persist_result sees a full history.
        result = dict(self._result)
        result.setdefault("messages", state.get("messages", []))
        return result


_MINIMAL_CONFIG = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
}

_DEFAULT_RESULT: dict[str, Any] = {
    "retrieved_items": [],
    "filters": {},
    "tool_calls": [{"router_decision": {"action": "search", "query": "hello"}}],
    "final_answer": "Here are some results.",
    "iteration": 1,
    "new_items_this_turn": False,
    "out_of_catalogue": False,
    "excluded_colours": None,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def inject_deps(monkeypatch: pytest.MonkeyPatch):
    """Inject a fresh session store, mock LLM, and minimal config before each test."""
    store = InMemorySessionStore()
    monkeypatch.setattr(deps, "_session_store", store)
    monkeypatch.setattr(deps, "_llm", _MockLLM(["ok"]))
    monkeypatch.setattr(deps, "_config", _MINIMAL_CONFIG)
    # Disable JWT verification so these tests run without Authorization headers.
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
    # Raise the rate limit high so individual tests never trip it.
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")
    # _retriever and _catalogue_df remain None; the agent factory is patched per-test.
    yield


@pytest.fixture
def client():
    # Instantiate without context manager so the lifespan is never triggered
    # and no real index files are required.
    yield TestClient(app, raise_server_exceptions=True)


def _make_factory(result: dict[str, Any]):
    """Return a get_agent_factory replacement that yields a mock agent."""
    agent = _MockAgent(result)

    def factory(memory: Any, streaming: bool = False) -> _MockAgent:
        return agent

    def get_factory() -> Any:
        return factory

    return get_factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_new_conversation_mints_uuid(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    """POST /chat without a conversation_id must return a valid UUID4."""
    monkeypatch.setattr(deps, "get_agent_factory", _make_factory(_DEFAULT_RESULT))

    resp = client.post("/chat", json={"message": "hello"})

    assert resp.status_code == 200
    data = resp.json()
    cid = data["conversation_id"]
    # Must be a valid UUID4
    parsed = uuid.UUID(cid, version=4)
    assert str(parsed) == cid
    assert data["response"] == "Here are some results."


def test_continuation_reuses_session(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    """Two consecutive turns with the same conversation_id must share the session."""
    turn1_result = {**_DEFAULT_RESULT, "final_answer": "Turn 1 answer."}
    turn2_result = {**_DEFAULT_RESULT, "final_answer": "Turn 2 answer."}

    # Use a stateful factory so the second call returns a different result.
    results = [turn1_result, turn2_result]
    call_count = [0]

    def stateful_factory() -> Any:
        def factory(memory: Any, streaming: bool = False) -> _MockAgent:
            idx = min(call_count[0], len(results) - 1)
            call_count[0] += 1
            return _MockAgent(results[idx])
        return factory

    monkeypatch.setattr(deps, "get_agent_factory", stateful_factory)

    r1 = client.post("/chat", json={"message": "show me red dresses"})
    assert r1.status_code == 200
    cid = r1.json()["conversation_id"]

    r2 = client.post("/chat", json={"conversation_id": cid, "message": "in blue instead"})
    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == cid
    assert r2.json()["response"] == "Turn 2 answer."

    # Session must still exist in the store with accumulated messages.
    session = deps.get_session_store().get(cid, deps.DEV_USER_ID)
    assert session is not None
    user_messages = [m for m in session["messages"] if m["role"] == "user"]
    assert len(user_messages) == 2


def test_agent_error_returns_500(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    """If agent.invoke() raises, the endpoint must return HTTP 500."""

    class _BrokenAgent:
        def invoke(self, state: dict, **kwargs) -> dict:
            raise RuntimeError("index is corrupt")

    def get_factory() -> Any:
        def factory(memory: Any, streaming: bool = False) -> _BrokenAgent:
            return _BrokenAgent()
        return factory

    monkeypatch.setattr(deps, "get_agent_factory", get_factory)

    resp = client.post("/chat", json={"message": "hello"})

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal server error"


def test_items_returned_when_new_items_this_turn(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    """When new_items_this_turn=True the response must include the item list."""
    result_with_items = {
        **_DEFAULT_RESULT,
        "new_items_this_turn": True,
        "retrieved_items": [
            {
                "article_id": "111222333",
                "prod_name": "Slim Trousers",
                "display_name": "Slim Trousers (Black Trousers)",
                "colour": "Black",
                "product_type": "Trousers",
                "department": "Ladieswear",
                "image_url": None,
                "detail_desc": "Slim fit trousers in black.",
                "score": 0.92,
            }
        ],
    }
    monkeypatch.setattr(deps, "get_agent_factory", _make_factory(result_with_items))

    resp = client.post("/chat", json={"message": "show me black trousers"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["new_items_this_turn"] is True
    assert len(data["items"]) == 1
    assert data["items"][0]["article_id"] == "111222333"
    assert data["items"][0]["display_name"] == "Slim Trousers (Black Trousers)"


def test_chat_persists_filters_across_calls(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    """Filter state from turn 1 must be present in the session after turn 2.

    Locks in the Phase 0 search_node filter-persistence fix: the agent merges
    incoming state filters with the new turn's filters, so the dress product_type
    from turn 1 survives when the user says 'in blue instead' on turn 2.
    """
    # Turn 1: agent sets both product_type and colour filters.
    turn1_result = {
        **_DEFAULT_RESULT,
        "filters": {"product_type_name": "Dress", "colour_group_name": "Red"},
        "final_answer": "Here are some red dresses.",
    }
    # Turn 2: agent simulates the merge — dress filter preserved, colour updated.
    turn2_result = {
        **_DEFAULT_RESULT,
        "filters": {"product_type_name": "Dress", "colour_group_name": "Blue"},
        "final_answer": "Here are some blue dresses.",
    }

    results = [turn1_result, turn2_result]
    call_count = [0]

    def stateful_factory() -> Any:
        def factory(memory: Any, streaming: bool = False) -> _MockAgent:
            idx = min(call_count[0], len(results) - 1)
            call_count[0] += 1
            return _MockAgent(results[idx])
        return factory

    monkeypatch.setattr(deps, "get_agent_factory", stateful_factory)

    r1 = client.post("/chat", json={"message": "show me red dresses"})
    assert r1.status_code == 200
    cid = r1.json()["conversation_id"]

    # Verify turn 1 stored the expected filters.
    session_after_t1 = deps.get_session_store().get(cid, deps.DEV_USER_ID)
    assert session_after_t1 is not None
    assert session_after_t1["filters"]["product_type_name"] == "Dress"
    assert session_after_t1["filters"]["colour_group_name"] == "Red"

    r2 = client.post("/chat", json={"conversation_id": cid, "message": "in blue instead"})
    assert r2.status_code == 200

    session_after_t2 = deps.get_session_store().get(cid, deps.DEV_USER_ID)
    assert session_after_t2 is not None
    # product_type_name must survive the colour refinement.
    assert session_after_t2["filters"]["product_type_name"] == "Dress"
    assert session_after_t2["filters"]["colour_group_name"] == "Blue"


def test_chat_handles_ooc_query(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    """An out-of-catalogue query must return out_of_catalogue=True and no items.

    The mock agent simulates what the graph does when the LLM router returns an
    OOC routing decision (e.g. for 'show me face cream').
    """
    ooc_result = {
        **_DEFAULT_RESULT,
        "out_of_catalogue": True,
        "new_items_this_turn": False,
        "retrieved_items": [],
        "filters": {},
        "tool_calls": [
            {"router_decision": {"action": "out_of_catalogue", "query": "show me face cream"}}
        ],
        "final_answer": (
            "I can only help with clothing and fashion items. "
            "Face cream is outside my catalogue."
        ),
    }
    monkeypatch.setattr(deps, "get_agent_factory", _make_factory(ooc_result))

    resp = client.post("/chat", json={"message": "show me face cream"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["out_of_catalogue"] is True
    assert data["items"] == []
    assert data["new_items_this_turn"] is False
    assert "face cream" in data["response"].lower() or "catalogue" in data["response"].lower()
