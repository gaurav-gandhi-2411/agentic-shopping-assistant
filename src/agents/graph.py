import json
import re

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.agents.grounding import validate_response
from src.agents.reranker import rerank
from src.agents.state import AgentState
from src.agents.tools import apply_filter, clarify, compare_items, search_catalogue, suggest_outfit
from src.llm.client import LLMClient
from src.memory.conversation import ConversationMemory
from src.retrieval.hybrid_search import HybridRetriever


_COMPARE_INTENT = re.compile(
    r"\bcompare\b|\bdifference\s+between\b|\bvs\b|\bversus\b", re.IGNORECASE
)

_BEACH_SUMMER_RE = re.compile(
    r"\b(beach|summer|vacation|holiday|resort)\b", re.IGNORECASE
)

# Structured OOC category map: category label → list of trigger words.
# Checked in search_node BEFORE any retrieval; fires a canned "not in catalogue" response.
# Structured OOC category map: checked in insertion order.
# More specific categories (pet supplies, electronics) before broader ones (food) to
# prevent "dog food" from matching "food" in the food/drink category.
_OOC_CATEGORIES: dict[str, list[str]] = {
    "pet supplies": [
        "dog food", "cat food", "pet food", "pet supplies", "pet toy",
        "dog collar", "cat litter", "pet treat",
    ],
    "electronics": [
        "laptop", "computer", "smartphone", "tablet", "headphones", "camera",
        "television", " tv ", "gadget", "phone case", "charger", "earphones",
        "speaker", "monitor", "keyboard", "mouse",
    ],
    "beauty or cosmetics": [
        "lipstick", "lipsticks", "mascara", "eyeshadow", "eye shadow",
        "foundation", "concealer", "eyeliner", "nail polish", "nail varnish",
        "perfume", "fragrance", "cologne", "deodorant", "moisturizer",
        "moisturiser", "skincare", "serum", "serums", "makeup", "make up",
        "face cream", "blush", "bronzer", "highlighter", "contour",
        "bb cream", "cc cream", "toner", "face mask", "sheet mask",
    ],
    "home and furniture": [
        "pillow", "bedsheet", "bed sheet", "towel", "blanket", "rug",
        "curtain", "furniture", "vase", "candle", "sofa", "couch",
        "mattress", "bookshelf", "lamp", "duvet", "comforter",
    ],
    "food and drink": [
        " food ", "snack", "drink", "coffee", "tea", "recipe", "restaurant",
        "grocery", "meal", "cuisine",
    ],
}


def _detect_ooc(query: str) -> str | None:
    """Return the OOC category label if query contains a known non-clothing keyword, else None."""
    q = query.lower()
    for category, words in _OOC_CATEGORIES.items():
        if any(w in q for w in words):
            return category
    return None

_LAST_N_RE = re.compile(r"\blast\s+(two|three|four|five|[2-5])\b", re.IGNORECASE)
_FIRST_N_RE = re.compile(r"\b(?:first|top)\s+(two|three|four|five|[2-5])\b", re.IGNORECASE)
_IDX_PAIR_RE = re.compile(r"\b(\d)\s+and\s+(\d)\b")
_ORD_PAIR_RE = re.compile(r"\b(\d)(?:st|nd|rd|th)\s+and\s+(\d)(?:st|nd|rd|th)\b", re.IGNORECASE)
_WORD_TO_INT = {"two": 2, "three": 3, "four": 4, "five": 5}


def _select_items_for_compare(user_query: str, items: list[dict]) -> list[dict]:
    """Pick items from the retrieved list based on user's selection modifier."""
    q = user_query.lower()
    n = len(items)

    m = _LAST_N_RE.search(q)
    if m:
        token = m.group(1).lower()
        count = _WORD_TO_INT[token] if token in _WORD_TO_INT else int(token)
        return items[max(0, n - count):]

    m = _FIRST_N_RE.search(q)
    if m:
        token = m.group(1).lower()
        count = _WORD_TO_INT[token] if token in _WORD_TO_INT else int(token)
        return items[:count]

    m = _ORD_PAIR_RE.search(q) or _IDX_PAIR_RE.search(q)
    if m:
        i, j = int(m.group(1)), int(m.group(2))
        selected = [items[idx - 1] for idx in (i, j) if 1 <= idx <= n]
        return selected if len(selected) == 2 else items[:2]

    return items[:2]

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
0. COMPARE PRIORITY (highest): If the user says "compare", "compare the", "compare those",
   "difference between", "vs", or "versus" AND items_retrieved > 0 →
   ALWAYS output {{"action": "compare", "article_ids": []}}. This overrides rules 1–9.
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
   - Completely off-topic (not a shopping query at all)
   Do NOT clarify for any of these — interpret and search instead:
   - Follow-up refinements: "something more casual", "in blue", "cheaper", "simpler"
   - Style words: "elegant", "minimal", "edgy", "relaxed", "chic"
   - Occasion words: "date night", "beach", "office", "brunch", "gym"
   - Vague adjectives of any kind — commit to a best-effort search; the user can refine

   AVAILABLE DATA — the system has: display_name, colour_group_name, product_type_name,
   department_name, detail_desc (short product description).
   The system does NOT have: price, size, material/fabric composition, weight, fit,
   in-stock status, seller rating.
   When the user asks about an attribute the system does not have (price, fabric, size,
   fit, stock, rating), output {{"action": "respond"}} — do NOT clarify. The respond layer
   will deliver the "I don't have that information" message using its grounding rules.

   Default rule: if you are not certain clarification is essential, output "respond" or "search".
9. NEVER repeat the same action twice in a row.

SEASONAL / OCCASION QUERY REWRITING:
When constructing the search "query" field, always expand it with 2-3 relevant category words:
- "winter" / "cold" / "snow" / "cosy" / "cozy": append "sweater coat jacket knitwear"
- "beach" specifically: append "swimwear bikini cover-up sundress light dress"
- "summer" generally (no beach): append "summer dress light top t-shirt"
- Do NOT add "shorts" to beach or summer queries — "shorts" triggers athletic/sport matches.
- "autumn" / "fall" / "rainy": append "jacket coat knitwear"
- "office" / "work" / "meeting": append "blazer trousers shirt dress"
- "date night" / "evening" / "cocktail": append "dress blouse blazer evening elegant"
Never pass the raw user query unchanged when a seasonal or occasion context is present.

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
You are a shopping assistant. Answer the user's question using ONLY the attributes \
listed below for each item. Do not invent or infer any facts not present in the data.

STRICT RULES:
- Do NOT mention price, cost, cheaper, expensive, affordable, budget, sale, or discount. \
No price data exists. If asked about price, say "I don't have pricing information."
- Do NOT mention size, fit, runs big/small, or true-to-size. No size data exists.
- Do NOT claim fabric performance (breathable, sweat-wicking, waterproof, warm, cold) \
unless those exact words appear in the item description below.
- Do NOT compare items on attributes that are not listed (no price comparisons, no size comparisons).
- Keep response to 2-3 sentences. Mention at most 2-3 items by name.
- Use only facts directly from the provided item attributes.

MISSING ATTRIBUTE HANDLING — if the user asked about something we don't have:
- fabric / material / composition → "I don't have fabric information for these items — \
check the product details on the site."
- price / cost / sale → "I don't have pricing information — check the product page."
- size / fit / sizing → "I don't have size or fit information — check the product page."
- stock / availability → "I don't have stock information — check the product page."
Follow with one optional sentence about what IS visible (colour, type, description excerpt).

User question: "{user_query}"

Available item attributes:
{items}

Write your response now."""


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
        desc = item.get("detail_desc") or ""
        short = desc[:150].rstrip() + "..." if len(desc) > 150 else desc
        lines.append(
            f"- display_name: {item.get('display_name', '')}\n"
            f"  colour: {item.get('colour', '')} | "
            f"type: {item.get('product_type', '')} | "
            f"department: {item.get('department', '')}\n"
            f"  description: {short}"
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

    # Gender keyword → index_group_name value mapping.
    # Applied in search_node before the general auto-facet extractor so
    # gender intent in the query always wins over ambiguous facet matches.
    _GENDER_MAP: dict[str, str] = {
        r"\bmen\b": "Menswear", r"\bmens\b": "Menswear",
        r"\bman\b": "Menswear", r"\bmale\b": "Menswear",
        r"\bwomen\b": "Ladieswear", r"\bwomens\b": "Ladieswear",
        r"\bwoman\b": "Ladieswear", r"\bfemale\b": "Ladieswear",
        r"\bladies\b": "Ladieswear", r"\bladieswear\b": "Ladieswear",
        r"\bkid\b": "Baby/Children", r"\bkids\b": "Baby/Children",
        r"\bchild\b": "Baby/Children", r"\bchildren\b": "Baby/Children",
        r"\bbaby\b": "Baby/Children",
        r"\bteen\b": "Divided", r"\bteens\b": "Divided",
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
        raw_query = state["user_query"]
        query = plan.get("query", raw_query)

        # Out-of-catalogue detection: keyword check on original user query.
        # Uses structured keyword list rather than score threshold — MiniLM similarity
        # is too noisy to separate in-catalogue from out-of-catalogue reliably.
        ooc_category = _detect_ooc(raw_query)
        if ooc_category:
            print(f"[search] OOC detected ({ooc_category!r}): {raw_query!r}")
            return {
                "retrieved_items": [],
                "new_items_this_turn": False,
                "out_of_catalogue": True,
                "iteration": state.get("iteration", 0) + 1,
                "tool_calls": state.get("tool_calls", []) + [
                    {"search_ooc": {"query": raw_query, "category": ooc_category}}
                ],
            }
        # Merge accumulated state filters with any new filters the router specified.
        merged = {**state.get("filters", {}), **plan.get("filters", {})}

        # Gender keyword extraction — applied before general auto-facet so explicit
        # gender words ("men's shoes", "women's jacket") always set the right group.
        if "index_group_name" not in merged:
            raw_lower = raw_query.lower()
            for pattern, group_val in _GENDER_MAP.items():
                if re.search(pattern, raw_lower, re.IGNORECASE):
                    merged = {**merged, "index_group_name": group_val.lower()}
                    break

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

        prior_items = state.get("retrieved_items", [])
        prior_ids = {it["article_id"] for it in prior_items}
        refinement = _is_refinement_search(query, prior_items, merged)

        # Always fetch 20 candidates for the LLM reranker.
        result = search_catalogue(query, merged or None, retriever, 20)

        # Gender filter is applied when it was extracted from this query (not inherited).
        # Keep it explicit so we can handle zero-stock gracefully below.
        gender_filter_applied = "index_group_name" in merged and "index_group_name" not in {
            k: v for k, v in (state.get("filters") or {}).items()
        }
        gender_value = merged.get("index_group_name", "") if gender_filter_applied else ""

        # Filter retry logic: different behaviour for gender vs other filters.
        effective_filters = merged
        if not result["items"] and merged:
            if gender_filter_applied:
                # Gender filter produced 0 results — do NOT silently fall back to
                # the other gender. Try dropping non-gender filters if any exist,
                # keeping gender filter in place so the message stays accurate.
                other_filters = {k: v for k, v in merged.items() if k != "index_group_name"}
                if other_filters:
                    gender_only = {"index_group_name": gender_value}
                    retry = search_catalogue(query, gender_only, retriever, 20)
                    result = retry
                    effective_filters = gender_only
                # If still 0 items (no menswear footwear at all) keep effective_filters=merged
                # → respond_node will emit an explicit "no stock" message.
            else:
                # No gender filter — drop invalid facet filters (LLM invented bad values)
                result = search_catalogue(query, None, retriever, 20)
                effective_filters = {}

        # Sparse/zero stock warning: fires when gender filter is applied and fewer
        # than 5 items matched (including 0).
        few_gender_results = gender_filter_applied and len(result["items"]) < 5

        # For refinement turns: exclude items the user has already seen before reranking.
        candidates = result["items"]
        if refinement and prior_ids:
            fresh = [it for it in candidates if it["article_id"] not in prior_ids]
            if len(fresh) >= 2:
                candidates = fresh

        items_out = rerank(query, candidates, llm, top_k=top_k)

        # Dedup by (prod_name, colour): H&M lists same product in many colours;
        # backfill from the wider candidates pool if dedup drops below top_k.
        seen_prod: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for item in items_out:
            key = (item.get("prod_name", item["display_name"]).strip().lower(), item["colour"].lower())
            if key not in seen_prod:
                seen_prod.add(key)
                deduped.append(item)
        if len(deduped) < top_k:
            seen_ids_dedup = {it["article_id"] for it in deduped}
            for item in candidates:
                if len(deduped) >= top_k:
                    break
                key = (item.get("prod_name", item["display_name"]).strip().lower(), item["colour"].lower())
                if item["article_id"] not in seen_ids_dedup and key not in seen_prod:
                    seen_prod.add(key)
                    deduped.append(item)
        items_out = deduped

        # Beach/summer queries: cap at 2 items per product_type to ensure variety
        # (e.g. prevent 4 bikinis when swimwear dominates retrieval).
        if _BEACH_SUMMER_RE.search(query):
            diverse: list[dict] = []
            type_counts: dict[str, int] = {}
            seen_diverse = {it["article_id"] for it in diverse}
            for item in items_out:
                pt = item.get("product_type", "").lower()
                if type_counts.get(pt, 0) < 2:
                    diverse.append(item)
                    type_counts[pt] = type_counts.get(pt, 0) + 1
            # Fill remaining slots from the wider candidates pool if reranker gave few types
            if len(diverse) < top_k:
                seen_diverse = {it["article_id"] for it in diverse}
                for item in candidates:
                    if len(diverse) >= top_k:
                        break
                    if item["article_id"] in seen_diverse:
                        continue
                    pt = item.get("product_type", "").lower()
                    if type_counts.get(pt, 0) < 2:
                        diverse.append(item)
                        type_counts[pt] = type_counts.get(pt, 0) + 1
                        seen_diverse.add(item["article_id"])
            items_out = diverse[:top_k]

        search_meta: dict = {"query": query, "filters": merged}
        if few_gender_results:
            search_meta["few_gender_results"] = True
            search_meta["gender_group"] = merged.get("index_group_name", "")
        update: dict = {
            "retrieved_items": items_out,
            "new_items_this_turn": True,
            "iteration": state.get("iteration", 0) + 1,
            "tool_calls": state.get("tool_calls", []) + [{"search": search_meta}],
        }
        if effective_filters != merged:
            update["filters"] = effective_filters
        return update

    def compare_node(state: AgentState) -> dict:
        plan = json.loads(state.get("current_plan") or "{}")
        article_ids = plan.get("article_ids", [])
        retrieved = state.get("retrieved_items", [])

        # Parse selection modifier ("last two", "first two", "2 and 4", etc.)
        # then override the LLM-extracted IDs so positional references are honoured.
        if retrieved:
            selected = _select_items_for_compare(state.get("user_query", ""), retrieved)
            if selected:
                article_ids = [it["article_id"] for it in selected]

        # Edge case: not enough items to compare
        if len(retrieved) < 2 and not article_ids:
            answer = "I only have one item to compare — please search for more items first."
            if streaming_mode:
                return {
                    "current_plan": json.dumps({"action": "pending_answer", "text": answer}),
                    "final_answer": None,
                    "messages": [],
                }
            return {
                "final_answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
            }

        result = compare_items(article_ids, catalogue_df)
        # Keep existing items if compare found nothing (e.g. bad IDs)
        new_items = result["items"] if result["items"] else state.get("retrieved_items", [])
        return {
            "retrieved_items": new_items,
            "new_items_this_turn": True,
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

        # Guard: if still no seed (e.g. "outfits for date night" misrouted here with no
        # prior items), return a graceful prompt rather than "Item not found."
        if not article_id:
            answer = (
                "To build an outfit, click 'Style this' on a specific item — "
                "or tell me which item you'd like to style."
            )
            if streaming_mode:
                return {
                    "current_plan": json.dumps({"action": "pending_answer", "text": answer}),
                    "final_answer": None,
                    "messages": [],
                }
            return {
                "final_answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
            }

        result = suggest_outfit(article_id, catalogue_df, retriever)
        seed = result.get("seed_item")
        complements = result.get("complements", [])
        rationale = result.get("outfit_rationale", "")
        empty_slots = result.get("empty_slots", [])

        items_out = ([seed] if seed else []) + complements
        answer = f"**Outfit suggestion**\n\n{rationale}"
        if empty_slots:
            answer += "\n\n_Some outfit slots (e.g. shoes or accessories) have limited matching items in this catalogue._"

        update: dict = {
            "retrieved_items": items_out,
            "new_items_this_turn": True,
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
        # Out-of-catalogue shortcut: skip LLM, return a canned concise message.
        if state.get("out_of_catalogue"):
            ooc_cat = next(
                (tc["search_ooc"].get("category", "") for tc in state.get("tool_calls", [])
                 if "search_ooc" in tc),
                "",
            )
            if ooc_cat:
                answer = (
                    f"I don't carry {ooc_cat} products — this catalogue is clothing only. "
                    f"I can help with dresses, tops, trousers, jackets, knitwear, and accessories."
                )
            else:
                answer = (
                    "I don't have that in this catalogue. "
                    "I can help with clothing like dresses, tops, trousers, jackets, and outerwear."
                )
            if streaming_mode:
                return {
                    "current_plan": json.dumps({"action": "pending_answer", "text": answer}),
                    "final_answer": None,
                    "messages": [],
                }
            return {
                "final_answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
            }

        items = state.get("retrieved_items", [])

        # Check if gender filter returned sparse/zero results.
        few_gender = any(
            tc.get("search", {}).get("few_gender_results")
            for tc in state.get("tool_calls", [])
        )
        gender_group = next(
            (tc["search"].get("gender_group", "") for tc in state.get("tool_calls", [])
             if tc.get("search", {}).get("few_gender_results")),
            "",
        )

        # Zero-stock gender case: skip LLM, return a direct explicit message.
        if few_gender and gender_group and not items:
            answer = (
                f"This catalogue has no {gender_group} items matching your query. "
                f"The H&M {gender_group} range here is very limited — "
                f"try the main H&M site for a broader selection."
            )
            if streaming_mode:
                return {
                    "current_plan": json.dumps({"action": "pending_answer", "text": answer}),
                    "final_answer": None,
                    "messages": [],
                }
            return {
                "final_answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
            }

        prompt = RESPOND_PROMPT.format(
            user_query=state["user_query"],
            items=_format_items_for_response(items),
        )
        if few_gender and gender_group:
            prompt += (
                f"\n\nNote: this catalogue has limited {gender_group} stock. "
                f"Mention this briefly at the end of your response."
            )
        if streaming_mode:
            # In streaming mode the app streams the LLM call; store the prompt for pickup.
            return {
                "current_plan": json.dumps({"action": "pending_respond", "prompt": prompt}),
                "final_answer": None,
                "messages": [],
            }
        answer = llm.generate(prompt)
        answer, flags = validate_response(answer, items)
        if flags:
            print(f"[grounding] flags={flags} query={state['user_query']!r}")
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

        # Out-of-catalogue short-circuit: don't re-run search, go straight to respond.
        if state.get("out_of_catalogue"):
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
        # Compare-intent guard: user explicitly wants a comparison → force compare
        # regardless of what the LLM plan says.  Skipped once compare already ran
        # (last_tool == "compare") so we don't loop.
        if (last_tool != "compare"
                and _COMPARE_INTENT.search(state.get("user_query", ""))
                and state.get("retrieved_items")):
            return "compare"

        # Always respond after compare, even if it returned no items
        # (prevents infinite loops when compare is called with empty state).
        if last_tool == "compare":
            return "respond"
        if last_tool == "search" and state.get("retrieved_items"):
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
