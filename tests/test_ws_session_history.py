"""Regression test for the WS session-history overwrite bug.

api/routes/chat.py::_persist_result does ``session["messages"] = result.get("messages", ...)``
assuming `result["messages"]` carries the FULL accumulated message list. That's true for
POST /chat (agent.invoke() returns the graph's accumulated `messages` via the
`operator.add` reducer). But the WS streaming path used to REBUILD `result["messages"]`
to a single-element list (just the assistant's streamed reply) right before calling
_persist_result — truncating multi-turn session history on every streamed turn, which is
the primary frontend path.

This test drives two consecutive WS turns on the SAME conversation through the real
`ws_chat` route (demo-anon session + ticket auth, exactly like the frontend) with a fake
streaming-mode agent standing in for the compiled LangGraph graph (no real index needed —
this bug is about message-list plumbing in api/routes/chat.py, not retrieval), and asserts
the persisted session ends up with exactly 4 messages in order: user1, assistant1, user2,
assistant2.
"""
from __future__ import annotations

import json
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

import api.deps as deps
from api.main import app
from api.session import InMemorySessionStore

_MINIMAL_CONFIG: dict = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
}


class _FakeStreamingAgent:
    """Mimics the streaming-mode graph contract used by respond_node/outfit_node/
    clarify_node: `messages` is returned UNCHANGED (i.e. equal to the full
    accumulated input — prior session history + this turn's user message) because
    the assistant's reply text isn't known yet; it hands off a "pending_respond"
    plan for api/routes/chat.py::ws_chat to stream via llm.generate_stream and
    append itself.
    """

    def invoke(self, state: dict, **kwargs: Any) -> dict:
        return {
            "messages": state["messages"],
            "retrieved_items": [],
            "filters": {},
            "tool_calls": [],
            "current_plan": json.dumps({"action": "pending_respond", "prompt": "reply please"}),
            "final_answer": None,
            "new_items_this_turn": False,
            "out_of_catalogue": False,
            "excluded_colours": None,
        }


class _MockLLM:
    """Cycles through canned replies for generate_stream (bridged via _iter_tokens);
    yields the whole reply as a single "token" chunk — content is irrelevant, only
    the resulting session["messages"] shape is asserted.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self._idx = 0

    def _next(self) -> str:
        r = self._replies[min(self._idx, len(self._replies) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str = None, **kwargs: Any) -> str:
        return self._next()

    def generate_stream(self, prompt: str, system: str = None, **kwargs: Any) -> Iterator[str]:
        yield self._next()

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        yield self._next()


@pytest.fixture
def demo_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Wires a fake streaming agent + mock LLM into api.deps (no real index/graph
    compile needed) and enables the demo-anon auth path, exactly like the deployed
    service's frontend-facing WS path.
    """
    llm = _MockLLM(["Turn one reply.", "Turn two reply."])

    monkeypatch.setattr(deps, "_session_store", InMemorySessionStore())
    monkeypatch.setattr(deps, "_llm", llm)
    monkeypatch.setattr(deps, "_config", _MINIMAL_CONFIG)
    monkeypatch.setattr(deps, "_db_engine", None)  # skip daily-cap DB checks

    def _factory() -> Any:
        agent = _FakeStreamingAgent()

        def factory(memory: Any, streaming: bool = False) -> _FakeStreamingAgent:
            return agent

        return factory

    monkeypatch.setattr(deps, "get_agent_factory", _factory)

    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_JWT_SECRET", "ws-session-history-test-secret")
    monkeypatch.delenv("JWT_VERIFICATION_DISABLED", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")

    # api.routes.chat._DEMO_SESSIONS is a module-level dict keyed by conversation_id;
    # reset it so tests never see state left behind by another test/run.
    import api.routes.chat as chat_module

    monkeypatch.setattr(chat_module, "_DEMO_SESSIONS", {})

    return TestClient(app, raise_server_exceptions=True)


def _mint_ticket(client: TestClient, session_token: str) -> str:
    resp = client.post(
        "/auth/ws-ticket", headers={"Authorization": f"Bearer {session_token}"}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["ticket"]


def _run_ws_turn(
    client: TestClient, ticket: str, message: str, conversation_id: str | None
) -> str:
    """Open one fresh WS connection (per-turn reconnect, matching the browser),
    send one user message, and return the conversation_id from the response frames.
    Blocks until the "done" frame or an "error" frame.
    """
    with client.websocket_connect(f"/chat/stream?ticket={ticket}") as ws:
        payload: dict = {"type": "user_message", "message": message}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        ws.send_json(payload)

        cid: str | None = conversation_id
        for _ in range(200):
            frame = ws.receive_json()
            ftype = frame.get("type")
            if ftype == "session":
                cid = frame["conversation_id"]
            elif ftype == "error":
                raise AssertionError(f"WS error frame: {frame}")
            elif ftype in ("done", "cancelled"):
                break
        assert cid is not None
        return cid


def test_two_ws_turns_leave_full_ordered_history(demo_client: TestClient) -> None:
    """Two consecutive WS turns on the same conversation must leave
    session["messages"] with exactly 4 entries in order: u1, a1, u2, a2 — not
    truncated to a single assistant reply per turn.
    """
    resp = demo_client.post("/demo/session")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    session_token = body["session_token"]
    first_ticket = body["ws_ticket"]

    import api.routes.chat as chat_module

    conversation_id = _run_ws_turn(demo_client, first_ticket, "hello there", None)

    ticket = _mint_ticket(demo_client, session_token)
    conversation_id = _run_ws_turn(demo_client, ticket, "and now this", conversation_id)

    session = chat_module._DEMO_SESSIONS[conversation_id]
    messages = session["messages"]

    assert len(messages) == 4, f"expected 4 messages, got {len(messages)}: {messages}"
    assert messages[0] == {"role": "user", "content": "hello there"}
    assert messages[1] == {"role": "assistant", "content": "Turn one reply."}
    assert messages[2] == {"role": "user", "content": "and now this"}
    assert messages[3] == {"role": "assistant", "content": "Turn two reply."}
