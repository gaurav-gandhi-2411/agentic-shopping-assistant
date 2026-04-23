import json
import re

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.agents.state import AgentState
from src.agents.tools import apply_filter, clarify, compare_items, search_catalogue, suggest_outfit
from src.llm.client import LLMClient
from src.memory.conversation import ConversationMemory
from src.retrieval.hybrid_search import HybridRetriever


ROUTER_PROMPT = """\
You are a shopping assistant planner. Given the conversation so far and the latest user query,
decide the NEXT action. Respond with ONE of the following JSON objects and nothing else:

{{"action": "search", "query": "<search string>", "filters": {{}}}}
{{"action": "compare", "article_ids": ["<id1>", "<id2>"]}}
{{"action": "filter", "key": "<facet>", "value": "<value>"}}
{{"action": "outfit", "article_id": "<article_id>"}}
{{"action": "clarify", "question": "<clarification for the user>"}}
{{"action": "respond"}}

STRICT RULES — follow in order:
1. If last_action is "compare"  →  you MUST output {{"action": "respond"}}
2. If last_action is "filter"   →  you MUST output {{"action": "search", ...}}
3. If items_retrieved > 0 AND last_action is "search"  →  output {{"action": "respond"}}
4. Use "search" for a new information need (first turn or after "filter").
5. Use "filter" to narrow results by colour, type, etc. — only once per turn.
   Filter values MUST be exact catalogue values. Valid colours: Black, White, Dark Blue,
   Grey, Red, Blue, Light Blue, Dark Red, Light Pink, Beige, Light Beige, Dark Grey.
   Never use descriptive terms (e.g. "Lightweight", "Breathable") as filter values.
6. Use "compare" when the user explicitly asks to compare items.
7. Use "outfit" when the user says "style this with", "what goes with", "build an outfit around",
   "complete the look", or asks for complementary items. Look up the article_id of the seed item
   from Current retrieved items by matching the item name the user refers to. If unclear, use
   the first item in Current retrieved items.
8. Use "clarify" ONLY if the query is completely incomprehensible OR is missing info
   so essential that no useful result is possible:
   - Gender ambiguity on a gender-neutral category ("a jacket" with no prior context)
   - Explicit request for price/size data the catalogue does not contain
   - Completely off-topic (not a shopping query at all)
   Do NOT clarify for any of these — interpret and search instead:
   - Follow-up refinements: "something more casual", "in blue", "cheaper", "simpler"
   - Style words: "elegant", "minimal", "edgy", "relaxed", "chic"
   - Occasion words: "date night", "beach", "office", "brunch", "gym"
   - Vague adjectives of any kind — commit to a best-effort search; the user can refine
   Default rule: if you are not certain clarification is essential, output "search".
9. NEVER repeat the same action twice in a row.

Last action taken: {last_action}
Items retrieved so far: {items_retrieved}

Available facets: colour_group_name, product_type_name, department_name, index_group_name, garment_group_name

Current retrieved items (if any):
{retrieved_summary}

Current filters: {current_filters}
Latest user query: {user_query}

Recent conversation:
{conversation}

Respond with ONLY the JSON object. No explanation."""


RESPOND_PROMPT = """\
You are a friendly, concise shopping assistant. The user asked: "{user_query}"

Here are the items we retrieved to help answer:
{items}

Write a helpful 2-4 sentence response that directly answers the user's question \
and naturally weaves in the best 2-3 items. If the user asked for a comparison, \
highlight the key differences. Do not invent attributes not in the retrieved items."""


def _format_items_brief(items: list[dict]) -> str:
    """Compact summary for the router prompt — includes IDs so the LLM can cite them."""
    if not items:
        return "None"
    lines = []
    for i, item in enumerate(items[:5], 1):
        lines.append(
            f"{i}. [ID: {item['article_id']}] {item['display_name']} — {item['product_type']}"
        )
    return "\n".join(lines)


def _format_items_for_response(items: list[dict]) -> str:
    if not items:
        return "No items retrieved."
    lines = []
    for item in items[:5]:
        desc = item["detail_desc"]
        short = desc[:120].rstrip() + "…" if len(desc) > 120 else desc
        lines.append(
            f"- {item['display_name']}\n"
            f"  Type: {item['product_type']}, Colour: {item['colour']}\n"
            f"  {short}"
        )
    return "\n".join(lines)


def _format_messages(messages: list[dict]) -> str:
    if not messages:
        return "(no prior conversation)"
    parts = []
    for m in messages[-6:]:
        role = m.get("role", "user").title()
        content = m.get("content", "")[:300]
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _parse_router_response(text: str, fallback_query: str) -> dict:
    """Extract first JSON object from LLM output; fall back to search on any parse failure.

    Uses brace-depth tracking so nested objects (e.g. "filters": {}) are included.
    """
    # Try the whole text first (LLM may return pure JSON)
    try:
        parsed = json.loads(text.strip())
        if "action" in parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Walk the string to find the outermost {...} block
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if "action" in parsed:
                            return parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

    return {"action": "search", "query": fallback_query}


def build_graph(
    retriever: HybridRetriever,
    catalogue_df: pd.DataFrame,
    llm: LLMClient,
    memory: ConversationMemory,
    config: dict,
    streaming_mode: bool = False,
):
    max_iterations = config["agent"]["max_iterations"]
    top_k = config["retrieval"]["final_k"]

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def router_node(state: AgentState) -> dict:
        # Extract the last non-router tool action to give the model explicit state cues.
        tool_calls = state.get("tool_calls", [])
        last_action = "none"
        for tc in reversed(tool_calls):
            key = list(tc.keys())[0]
            if key != "router_decision":
                last_action = key
                break

        context = memory.get_context(state.get("messages", []))
        prompt = ROUTER_PROMPT.format(
            last_action=last_action,
            items_retrieved=len(state.get("retrieved_items", [])),
            retrieved_summary=_format_items_brief(state.get("retrieved_items", [])),
            current_filters=json.dumps(state.get("filters", {})),
            user_query=state["user_query"],
            conversation=_format_messages(context),
        )
        raw = llm.generate(prompt)
        parsed = _parse_router_response(raw, state["user_query"])
        return {
            "current_plan": json.dumps(parsed),
            "tool_calls": tool_calls + [{"router_decision": parsed}],
        }

    # Build a set of valid values per facet once at graph-construction time.
    _valid_facet_values: dict[str, set[str]] = {
        col: set(catalogue_df[col].dropna().str.lower().unique())
        for col in [
            "colour_group_name", "product_type_name",
            "department_name", "index_group_name", "garment_group_name",
        ]
        if col in catalogue_df.columns
    }

    SEASONAL_PRODUCT_TYPES = {
        "winter": {
            "include": {"jacket", "coat", "sweater", "hoodie", "cardigan", "pullover",
                        "scarf", "beanie", "gloves", "boots", "jumper"},
            "exclude": {"shorts", "bikini", "swimsuit", "swimwear", "vest top",
                        "tank top", "sandals", "flip flops", "summer dress"},
        },
        "summer": {
            "include": {"shorts", "bikini", "swimsuit", "swimwear", "vest top",
                        "tank top", "sandals", "sundress", "summer dress", "t-shirt"},
            "exclude": {"coat", "parka", "wool jumper", "winter jacket",
                        "scarf", "beanie", "boots"},
        },
        "spring": {
            "include": {"light jacket", "cardigan", "blouse"},
            "exclude": set(),
        },
        "autumn": {
            "include": {"jacket", "cardigan", "sweater", "boots"},
            "exclude": {"bikini", "swimsuit", "shorts"},
        },
        "fall": {
            "include": {"jacket", "cardigan", "sweater", "boots"},
            "exclude": {"bikini", "swimsuit", "shorts"},
        },
    }

    OCCASION_EXCLUSIONS = {
        "date night": {
            "exclude_department": {"nightwear", "expressive lingerie", "lingerie",
                                   "underwear", "swimwear"},
            "exclude_product_type": {"night gown", "pajamas", "pyjamas",
                                     "underwear bottom", "bra", "briefs",
                                     "bikini", "swimsuit"},
        },
        "evening out": {
            "exclude_department": {"nightwear", "lingerie", "swimwear"},
            "exclude_product_type": {"night gown", "pajamas", "bra"},
        },
        "party": {
            "exclude_department": {"nightwear", "lingerie", "swimwear"},
            "exclude_product_type": {"night gown", "pajamas", "bra"},
        },
        "wedding": {
            "exclude_department": {"nightwear", "lingerie", "swimwear"},
            "exclude_product_type": {"night gown", "pajamas", "bra", "bikini"},
        },
        "cocktail": {
            "exclude_department": {"nightwear", "lingerie", "swimwear"},
            "exclude_product_type": {"night gown", "pajamas", "bra"},
        },
        "office": {
            "exclude_department": {"nightwear", "lingerie", "swimwear"},
            "exclude_product_type": {"night gown", "pajamas", "bra", "bikini", "shorts"},
        },
        "meeting": {
            "exclude_department": {"nightwear", "lingerie", "swimwear"},
            "exclude_product_type": {"night gown", "pajamas", "bra"},
        },
        "work": {
            "exclude_department": {"nightwear", "lingerie", "swimwear"},
            "exclude_product_type": {"night gown", "pajamas", "bra"},
        },
        "beach": {
            "exclude_department": set(),
            "exclude_product_type": {"coat", "heavy jacket", "sweater"},
        },
        "brunch": {
            "exclude_department": {"nightwear", "lingerie"},
            "exclude_product_type": {"night gown", "pajamas", "bra", "bikini"},
        },
    }

    def _is_refinement_search(
        query: str, prior_items: list[dict], filters: dict
    ) -> bool:
        """True when the new query refines rather than pivots from prior results.

        Checks whether the dominant product_type of prior results appears in the
        new search query string or in the inherited filters.  A pivot ("show me
        shirts" after dresses) returns False so dedup is skipped.
        """
        if not prior_items:
            return False
        from collections import Counter
        types = [it.get("product_type", "") for it in prior_items if it.get("product_type")]
        if not types:
            return False
        dominant = Counter(types).most_common(1)[0][0].lower()
        if dominant in query.lower():
            return True
        return any(str(v).lower() == dominant for v in filters.values())

    def search_node(state: AgentState) -> dict:
        plan = json.loads(state.get("current_plan") or "{}")
        query = plan.get("query", state["user_query"])
        # Merge accumulated state filters with any new filters the router specified.
        merged = {**state.get("filters", {}), **plan.get("filters", {})}

        # Auto-extract facet filters from the query when the LLM omitted them.
        # LLM-emitted filters take precedence (facets already in merged are skipped).
        # Longest value matched first within each facet so "dark blue" beats "blue",
        # "t-shirt" beats "shirt", etc.
        query_lower = query.lower()
        for facet_name, facet_vals in _valid_facet_values.items():
            if facet_name in merged:
                continue
            for val in sorted(facet_vals, key=len, reverse=True):
                if re.search(r"\b" + re.escape(val) + r"\b", query_lower):
                    merged = {**merged, facet_name: val}
                    break

        # Detect season words for post-retrieval soft exclusion.
        excluded_types: set[str] = set()
        for season, types in SEASONAL_PRODUCT_TYPES.items():
            if re.search(r"\b" + season + r"\b", query_lower):
                excluded_types = types["exclude"]
                break

        # Detect occasion words for post-retrieval soft exclusion.
        occasion_excludes_dept: set[str] = set()
        occasion_excludes_type: set[str] = set()
        for occ, occ_cfg in OCCASION_EXCLUSIONS.items():
            if re.search(r"\b" + re.escape(occ) + r"\b", query_lower):
                occasion_excludes_dept |= occ_cfg.get("exclude_department", set())
                occasion_excludes_type |= occ_cfg.get("exclude_product_type", set())

        prior_items = state.get("retrieved_items", [])
        prior_ids = {it["article_id"] for it in prior_items}
        refinement = _is_refinement_search(query, prior_items, merged)

        # Fetch a larger pool on refinement turns so we have candidates to dedup.
        fetch_k = top_k * 3 if (refinement and prior_ids) else top_k
        result = search_catalogue(query, merged or None, retriever, fetch_k)

        # If filters produced no results (LLM invented an invalid value), retry
        # without filters and clear the bad accumulated filters so they don't
        # contaminate future turns.
        effective_filters = merged
        if not result["items"] and merged:
            result = search_catalogue(query, None, retriever, fetch_k)
            effective_filters = {}

        # Combined soft exclusion: seasonal product types + occasion department/type.
        # Falls back to the original list if fewer than 3 items survive (too aggressive).
        if excluded_types or occasion_excludes_dept or occasion_excludes_type:
            filtered_items = []
            for item in result["items"]:
                dept  = item.get("department", "").lower()
                ptype = item.get("product_type", "").lower()
                if dept in occasion_excludes_dept:
                    continue
                if any(ex in ptype for ex in occasion_excludes_type):
                    continue
                if any(ex in ptype for ex in excluded_types):
                    continue
                filtered_items.append(item)
            if len(filtered_items) >= 3:
                result["items"] = filtered_items

        # For refinement turns: return items the user hasn't already seen.
        # Fallback to unfiltered list if fewer than 2 fresh candidates exist.
        if refinement and prior_ids:
            fresh = [it for it in result["items"] if it["article_id"] not in prior_ids]
            items_out = fresh[:top_k] if len(fresh) >= 2 else result["items"][:top_k]
        else:
            items_out = result["items"][:top_k]

        update: dict = {
            "retrieved_items": items_out,
            "iteration": state.get("iteration", 0) + 1,
            "tool_calls": state.get("tool_calls", []) + [
                {"search": {"query": query, "filters": merged}}
            ],
        }
        if effective_filters != merged:
            update["filters"] = effective_filters
        return update

    def compare_node(state: AgentState) -> dict:
        plan = json.loads(state.get("current_plan") or "{}")
        article_ids = plan.get("article_ids", [])

        # Fallback: if the LLM didn't extract explicit IDs (e.g. "compare the first two"),
        # use the first 2 items already stored in state — they are ranked by relevance and
        # persist across turns, so this correctly references the last set shown to the user.
        if not article_ids and state.get("retrieved_items"):
            article_ids = [r["article_id"] for r in state["retrieved_items"][:2]]

        result = compare_items(article_ids, catalogue_df)
        # Keep existing items if compare found nothing (e.g. bad IDs)
        new_items = result["items"] if result["items"] else state.get("retrieved_items", [])
        return {
            "retrieved_items": new_items,
            "iteration": state.get("iteration", 0) + 1,
            "tool_calls": state.get("tool_calls", []) + [
                {"compare": {"article_ids": article_ids}}
            ],
        }

    def filter_node(state: AgentState) -> dict:
        plan = json.loads(state.get("current_plan") or "{}")
        key = plan.get("key", "")
        value = plan.get("value", "")

        # Reject filters whose value doesn't exist in the catalogue — prevents
        # the LLM from applying invented values (e.g. colour "Lightweight") that
        # would silently zero-out every subsequent search.
        valid_vals = _valid_facet_values.get(key, set())
        if valid_vals and value.lower() not in valid_vals:
            return {
                "iteration": state.get("iteration", 0) + 1,
                "tool_calls": state.get("tool_calls", []) + [
                    {"filter_rejected": {key: value}}
                ],
            }

        new_filters = apply_filter(state.get("filters", {}), key, value)
        return {
            "filters": new_filters,
            "iteration": state.get("iteration", 0) + 1,
            "tool_calls": state.get("tool_calls", []) + [{"filter": {key: value}}],
        }

    def clarify_node(state: AgentState) -> dict:
        plan = json.loads(state.get("current_plan") or "{}")
        question = plan.get("question", "Could you clarify your request?")
        result = clarify(question)
        answer = result["clarification_question"]
        if streaming_mode:
            # In streaming mode the API streams the answer; store it for the caller.
            return {
                "current_plan": json.dumps({"action": "pending_answer", "text": answer}),
                "final_answer": None,
                "messages": [],
            }
        return {
            "final_answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
        }

    def outfit_node(state: AgentState) -> dict:
        plan = json.loads(state.get("current_plan") or "{}")
        article_id = plan.get("article_id", "")

        # Fallback: use first retrieved item when the LLM didn't extract an explicit ID.
        if not article_id and state.get("retrieved_items"):
            article_id = state["retrieved_items"][0]["article_id"]

        result = suggest_outfit(article_id, catalogue_df, retriever)
        seed = result.get("seed_item")
        complements = result.get("complements", [])
        rationale = result.get("outfit_rationale", "")

        items_out = ([seed] if seed else []) + complements
        answer = f"**Outfit suggestion**\n\n{rationale}"

        update: dict = {
            "retrieved_items": items_out,
            "tool_calls": state.get("tool_calls", []) + [{"outfit": {"article_id": article_id}}],
        }
        if streaming_mode:
            update["current_plan"] = json.dumps({"action": "pending_answer", "text": answer})
            update["final_answer"] = None
            update["messages"] = []
        else:
            update["final_answer"] = answer
            update["messages"] = [{"role": "assistant", "content": answer}]
        return update

    def respond_node(state: AgentState) -> dict:
        prompt = RESPOND_PROMPT.format(
            user_query=state["user_query"],
            items=_format_items_for_response(state.get("retrieved_items", [])),
        )
        if streaming_mode:
            # In streaming mode the API does the LLM call with token streaming;
            # store the formatted prompt so the API can pick it up.
            return {
                "current_plan": json.dumps({"action": "pending_respond", "prompt": prompt}),
                "final_answer": None,
                "messages": [],
            }
        answer = llm.generate(prompt)
        return {
            "final_answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_decision(state: AgentState) -> str:
        # Hard cap: always respond if we've hit the iteration limit.
        if state.get("iteration", 0) >= max_iterations:
            return "respond"

        # Deterministic loop-termination: after any tool that produces retrieved_items
        # (search or compare), force respond regardless of what the LLM output.
        # "filter" is intentionally excluded — it updates state.filters only; a search
        # must follow to apply those filters before we can respond with fresh results.
        tool_calls = state.get("tool_calls", [])
        last_tool = "none"
        for tc in reversed(tool_calls):
            key = list(tc.keys())[0]
            if key != "router_decision":
                last_tool = key
                break
        if last_tool in {"search", "compare"} and state.get("retrieved_items"):
            return "respond"

        try:
            plan = json.loads(state.get("current_plan") or "{}")
            action = plan.get("action", "search")
        except (json.JSONDecodeError, TypeError):
            action = "search"
        valid = {"search", "compare", "filter", "clarify", "respond", "outfit"}
        return action if action in valid else "search"

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("search", search_node)
    builder.add_node("compare", compare_node)
    builder.add_node("filter", filter_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("outfit", outfit_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", route_decision)
    builder.add_edge("search", "router")
    builder.add_edge("compare", "router")
    builder.add_edge("filter", "router")
    builder.add_edge("clarify", END)
    builder.add_edge("outfit", END)
    builder.add_edge("respond", END)

    return builder.compile()
