from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    # Conversation
    messages: Annotated[list[dict], operator.add]   # role/content dicts, accumulated
    user_query: str                                  # most recent user input

    # Agent internals
    current_plan: str | None                         # JSON string of last router decision
    tool_calls: list[dict]                           # tool call history within this turn
    retrieved_items: list[dict]                      # latest retrieval results — persisted
                                                     # across turns so compare can reference
                                                     # prior results (e.g. "compare first two")
    filters: dict                                    # accumulated facet filters

    # Output
    final_answer: str | None
    iteration: int
    new_items_this_turn: bool | None   # True only when search/compare/outfit produced new items
    out_of_catalogue: bool | None      # True when query is clearly outside H&M's product domain
