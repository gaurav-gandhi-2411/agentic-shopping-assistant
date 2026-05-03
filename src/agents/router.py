"""
Router backend — LLM-based prompt routing.

Implements decide(state) -> dict, returning the same format as router_node in
graph.py: {"current_plan": <json str>, "tool_calls": [...]}.

Note: out-of-catalogue short-circuit is handled in graph.py's router_node
closure (_detect_ooc) and is not duplicated here.
"""
from __future__ import annotations

import json
from typing import Callable, Protocol


class RouterBackend(Protocol):
    def decide(self, state: dict) -> dict: ...


class LLMRouterBackend:
    """Wraps the existing prompt-based routing logic."""

    def __init__(
        self,
        llm,
        memory,
        prompt_template: str,
        format_items_brief: Callable,
        format_messages: Callable,
        parse_response: Callable,
    ):
        self._llm = llm
        self._memory = memory
        self._prompt = prompt_template
        self._format_items_brief = format_items_brief
        self._format_messages = format_messages
        self._parse_response = parse_response

    def decide(self, state: dict) -> dict:
        tool_calls = state.get("tool_calls", [])
        last_action = "none"
        for tc in reversed(tool_calls):
            key = list(tc.keys())[0]
            if key != "router_decision":
                last_action = key
                break

        context = self._memory.get_context(state.get("messages", []))
        prompt = self._prompt.format(
            last_action=last_action,
            items_retrieved=len(state.get("retrieved_items", [])),
            retrieved_summary=self._format_items_brief(state.get("retrieved_items", [])),
            current_filters=json.dumps(state.get("filters", {})),
            user_query=state["user_query"],
            conversation=self._format_messages(context),
        )
        raw = self._llm.generate(prompt)
        parsed = self._parse_response(raw, state["user_query"])
        return {
            "current_plan": json.dumps(parsed),
            "tool_calls": tool_calls + [{"router_decision": parsed}],
        }


def get_router_backend(
    config: dict,
    llm,
    memory,
    catalogue_df=None,
    prompt_template: str = "",
    format_items_brief: Callable = None,
    format_messages: Callable = None,
    parse_response: Callable = None,
) -> LLMRouterBackend:
    return LLMRouterBackend(
        llm, memory, prompt_template,
        format_items_brief, format_messages, parse_response,
    )
