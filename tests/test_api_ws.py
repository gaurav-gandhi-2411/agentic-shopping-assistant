"""Tests for WS /chat/stream.

Uses FastAPI's TestClient WebSocket support.  Deps are injected via
monkeypatch; lifespan runs with real index files (same as test_api_chat.py).
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

import api.deps as deps
from api.main import app
from api.session import InMemorySessionStore


# ---------------------------------------------------------------------------
# Mock helpers  (mirrors test_api_chat.py)
# ---------------------------------------------------------------------------

class _MockLLM:
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
        # Split the next response into individual words to simulate multi-token output.
        text = self._next()
        for word in text.split():
            yield word + " "

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        yield self._next()


class _MockAgent:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def invoke(self, state: dict, **kwargs) -> dict[str, Any]:
        r = dict(self._result)
        r.setdefault("messages", state.get("messages", []))
        return r


class _SlowMockAgent:
    """Agent that sleeps delay_s seconds before returning (for cancel tests)."""

    def __init__(self, delay_s: float = 0.5) -> None:
        self._delay = delay_s

    def invoke(self, state: dict, **kwargs) -> dict[str, Any]:
        time.sleep(self._delay)
        return {
            "messages": state.get("messages", []),
            "retrieved_items": [],
            "filters": {},
            "tool_calls": [],
            "final_answer": "should not be seen",
            "iteration": 0,
            "new_items_this_turn": False,
            "out_of_catalogue": False,
            "excluded_colours": None,
            "current_plan": None,
        }


_MINIMAL_CONFIG = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
}

# A result that simulates a normal search turn with pending_respond.
_SEARCH_RESULT: dict[str, Any] = {
    "retrieved_items": [
        {
            "article_id": "999888777",
            "prod_name": "Test Jacket",
            "display_name": "Test Jacket (Blue Jacket)",
            "colour": "Blue",
            "product_type": "Jacket",
            "department": "Ladieswear",
            "image_url": None,
            "detail_desc": "A nice jacket.",
            "score": 0.85,
        }
    ],
    "filters": {"colour_group_name": "Blue"},
    "tool_calls": [
        {"router_decision": {"action": "search", "query": "blue jacket"}},
        {"search": {"query": "blue jacket", "filters": {}}},
    ],
    "final_answer": None,
    "current_plan": json.dumps({"action": "pending_respond", "prompt": "Describe these items."}),
    "iteration": 1,
    "new_items_this_turn": True,
    "out_of_catalogue": False,
    "excluded_colours": None,
}

# A result that simulates an OOC turn (pending_answer, no items).
_OOC_RESULT: dict[str, Any] = {
    "retrieved_items": [],
    "filters": {},
    "tool_calls": [
        {"router_decision": {"action": "out_of_catalogue", "query": "face cream"}},
        {"search_ooc": {"query": "face cream", "category": "beauty or cosmetics"}},
    ],
    "final_answer": None,
    "current_plan": json.dumps({
        "action": "pending_answer",
        "text": (
            "I don't carry beauty or cosmetics products — this catalogue is clothing only. "
            "I can help with dresses, tops, trousers, jackets, knitwear, and accessories."
        ),
    }),
    "iteration": 1,
    "new_items_this_turn": False,
    "out_of_catalogue": True,
    "excluded_colours": None,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def inject_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemorySessionStore()
    monkeypatch.setattr(deps, "_session_store", store)
    monkeypatch.setattr(deps, "_llm", _MockLLM(["Great blue jackets for you!"]))
    monkeypatch.setattr(deps, "_config", _MINIMAL_CONFIG)
    # Disable JWT verification so WS tests connect without ?token= query param.
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
    # Raise the rate limit high so individual tests never trip it.
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")


@pytest.fixture
def client() -> TestClient:
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _make_factory(agent: Any) -> Any:
    def get_factory() -> Any:
        def factory(memory: Any, streaming: bool = False) -> Any:
            return agent
        return factory
    return get_factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ws_full_turn_message_sequence(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    """Normal search turn: verify server sends session → routing → tool → items → tokens → done."""
    monkeypatch.setattr(deps, "get_agent_factory", _make_factory(_MockAgent(_SEARCH_RESULT)))
    # LLM streams three-word response
    monkeypatch.setattr(deps, "_llm", _MockLLM(["Great blue jackets!"]))

    with client.websocket_connect("/chat/stream") as ws:
        ws.send_json({"type": "user_message", "message": "show me blue jackets"})

        # session ack
        msg = ws.receive_json()
        assert msg["type"] == "session"
        cid = msg["conversation_id"]
        assert len(cid) == 36  # UUID4

        # routing
        msg = ws.receive_json()
        assert msg["type"] == "routing"
        assert msg["decision"]["action"] == "search"

        # tool_start events (router_decision is skipped; search is emitted)
        tool_types = []
        items_msg = None
        token_parts = []
        done_msg = None

        while True:
            msg = ws.receive_json()
            if msg["type"] == "tool_start":
                tool_types.append(msg["tool"])
            elif msg["type"] == "items":
                items_msg = msg
            elif msg["type"] == "token":
                token_parts.append(msg["text"])
            elif msg["type"] == "done":
                done_msg = msg
                break
            else:
                pytest.fail(f"Unexpected message type: {msg['type']}")

        assert "search" in tool_types

        assert items_msg is not None
        assert len(items_msg["items"]) == 1
        assert items_msg["items"][0]["article_id"] == "999888777"

        assert len(token_parts) >= 1
        full_text = "".join(token_parts)
        assert full_text == done_msg["final_state"]["response"], (
            "Token stream must exactly reconstruct done.final_state.response"
        )

        assert done_msg is not None
        assert done_msg["final_state"]["new_items_this_turn"] is True
        assert done_msg["final_state"]["out_of_catalogue"] is False


def test_ws_ooc_turn(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    """OOC query: done.final_state.out_of_catalogue must be True; no items emitted."""
    monkeypatch.setattr(deps, "get_agent_factory", _make_factory(_MockAgent(_OOC_RESULT)))

    with client.websocket_connect("/chat/stream") as ws:
        ws.send_json({"type": "user_message", "message": "show me face cream"})

        msg = ws.receive_json()
        assert msg["type"] == "session"

        items_seen = False
        token_parts = []
        done_msg = None

        while True:
            msg = ws.receive_json()
            if msg["type"] == "items":
                items_seen = True
            elif msg["type"] == "token":
                token_parts.append(msg["text"])
            elif msg["type"] == "done":
                done_msg = msg
                break
            elif msg["type"] in ("routing", "tool_start"):
                pass  # expected, ignore
            else:
                pytest.fail(f"Unexpected: {msg['type']}")

        assert not items_seen, "OOC turn must not emit items"
        assert done_msg is not None
        assert done_msg["final_state"]["out_of_catalogue"] is True
        assert done_msg["final_state"]["new_items_this_turn"] is False
        full_text = "".join(token_parts)
        assert "catalogue" in full_text.lower()


def test_ws_cancel_before_agent_completes(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    """Cancel must produce WSCancelledMessage well before the agent finishes (500ms).

    The mock agent sleeps 500ms.  We send cancel immediately after the session
    ack.  The cancelled message must arrive in under 300ms — long before the
    agent would complete.
    """
    monkeypatch.setattr(
        deps, "get_agent_factory", _make_factory(_SlowMockAgent(delay_s=0.5))
    )

    with client.websocket_connect("/chat/stream") as ws:
        ws.send_json({"type": "user_message", "message": "show me jackets"})

        msg = ws.receive_json()
        assert msg["type"] == "session"

        # Send cancel immediately after receiving the session ack.
        t0 = time.monotonic()
        ws.send_json({"type": "cancel"})

        cancelled = ws.receive_json()
        elapsed = time.monotonic() - t0

        assert cancelled["type"] == "cancelled", f"Expected cancelled, got {cancelled}"
        assert elapsed < 0.3, (
            f"Cancel took {elapsed:.3f}s — should arrive well before the 500ms agent sleep"
        )
