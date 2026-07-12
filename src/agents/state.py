import operator
from typing import Annotated, Any, TypedDict


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
    excluded_colours: list[str] | None  # Colours to exclude (parsed from negation queries)

    # Outfit composition context — set by outfit_node, persisted for flywheel event logging
    occasion: str | None        # one of the 12 occasion slugs
    look_gender: str | None     # "men" | "women" | "unisex"
    look_id: str | None         # UUID of the current composed look
    outfit_rationale: str | None    # grounded rationale for the base variant
    outfit_variants: list | None    # list of variant look dicts (1-3)
    budget_total_inr: float | None  # sum of shown item prices for the base variant
    suppressed_slots: list[dict] | None  # [{"slot": ..., "reason": ...}] — see
                                          # composer.compose_outfit's docstring

    # Phase B Part 2 — cross-gender PARTNER styling. Set by outfit_node ONLY when
    # this turn composed a companion look for the user's partner; None/omitted
    # for the primary look. "look_role" doubles as the frontend discriminator
    # ("partner" vs primary/omitted) — see api/schemas.py ChatResponse.
    look_role: str | None
    look_title: str | None
    coordinated_with: str | None

    # Conversation memory — the ConversationMemory instance for this conversation.
    # Injected into the initial state so the compiled graph singleton can access it
    # without needing memory as a constructor argument.  No node returns this field;
    # it persists unchanged through the full graph execution.
    _memory: Any

    # Summary state — written by get_context() when a summary is (re)computed;
    # propagated through the graph so _persist_result can sync to the session dict.
    _summary: str | None
    _summary_message_count: int

    # Colour refinement chips — set by search_node; cleared each turn and repopulated
    # with distinct colours from the current result set for front-end chip rendering.
    suggestion_chips: list[str] | None

    # Anchor item for buy-similar searches — stored after image upload by image_style.py.
    # The anchor is the CLIP-nearest catalogue item to the uploaded image.
    # search_node uses this for dense similarity retrieval when "similar/like this" is detected.
    anchor_article_id: str | None

    # "Owned anchor" feature — True when anchor_article_id refers to an item the USER
    # OWNS (uploaded a photo of it) rather than a catalogue item for sale. Set by
    # image_style.py alongside anchor_article_id; consulted by outfit_node so a
    # follow-up "Style this <item>" / re-compose never silently re-tags the user's
    # own garment as buyable. Defaults to False for text-only sessions.
    anchor_is_owned: bool

    # P3 body-type-aware guidance — set by outfit_node ONLY when a body type/
    # modifier was volunteered by the user (opt-in, never a gate on results).
    # Like `occasion`, NOT auto-persisted across turns by the session dict
    # (see graph.py's _reconstruct_body_type_from_history docstring); recovered
    # each turn from conversation history the same way occasion is.
    body_type: str | None
    body_modifiers: list[str] | None

    # P2 couple-from-scratch (graph.py::_compose_couple_from_scratch) — set
    # ONLY when a turn composes a SECOND, partner-gender look alongside the
    # primary one (no session anchor existed yet, but an occasion was
    # genuinely named — see router_node's "occasion_explicit" plan flag).
    # These mirror the existing primary-look fields above (retrieved_items,
    # look_id, occasion, look_gender, outfit_rationale, budget_total_inr,
    # suppressed_slots, look_role, look_title, coordinated_with) one-for-one,
    # for the SECOND board — downstream serialization (api/routes/chat.py /
    # frontend) reads these to render it. None/omitted for every other turn.
    partner_retrieved_items: list[dict] | None
    partner_look_id: str | None
    partner_occasion: str | None
    partner_look_gender: str | None
    partner_outfit_rationale: str | None
    partner_budget_total_inr: float | None
    partner_suppressed_slots: list[dict] | None
    partner_look_role: str | None
    partner_look_title: str | None
    partner_coordinated_with: str | None
