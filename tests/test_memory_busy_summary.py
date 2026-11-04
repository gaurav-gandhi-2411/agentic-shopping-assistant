"""B1: a busy-message 'summary' from the LLM chain must never be cached or persisted."""
from __future__ import annotations

from src.llm.resilience import LLM_BUSY_MESSAGE
from src.memory.conversation import ConversationMemory

CONFIG = {"memory": {"recent_turns": 2, "summary_trigger_turns": 4}}
MSGS = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(6)]


class _LLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    def generate(self, prompt: str, **kw: object) -> str:
        self.calls += 1
        return self.replies.pop(0)


def test_busy_message_is_not_cached_or_persisted_and_next_turn_retries():
    llm = _LLM([LLM_BUSY_MESSAGE, "- likes red"])
    mem = ConversationMemory(llm, CONFIG)
    state: dict = {}
    ctx = mem.get_context(MSGS, state)
    assert ctx == MSGS[-2:]  # no summary message at all
    assert mem._cached_summary is None and "_summary" not in state

    ctx = mem.get_context(MSGS, state)  # later turn retries
    assert llm.calls == 2
    assert state["_summary"] == "- likes red" and "likes red" in ctx[0]["content"]


def test_busy_message_keeps_previous_summary():
    llm = _LLM([LLM_BUSY_MESSAGE])
    mem = ConversationMemory(llm, CONFIG)
    mem.restore_summary("- old summary", 0)
    state: dict = {}
    ctx = mem.get_context(MSGS, state)
    assert "old summary" in ctx[0]["content"]
    assert mem._summary_computed_at == 0 and "_summary" not in state
