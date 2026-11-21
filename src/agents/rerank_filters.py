"""Deterministic post-rerank rules (beach / occasion / season / colour / diversity).

These are the hard rules the old LLM reranker prompt carried (src/agents/reranker.py @ ad17f47),
moved out of the prompt into code so they hold on every outcome (model order or retrieval-order
fallback) and need no prompt text; the compact variant-C prompt has none of them.

`rule_violations(query, item)` is the single definition of a violation; `apply_rules` removes
violators from an ordered candidate list, and the evaluation scripts count violations with the
same function. Matching is on word boundaries over the item's display name, product type and
colour (descriptions only for the beach dress-style rule, like the old "described as" wording).
Rules the old prompt only PREFERRED (rich colours, beach items first) are not hard filters; the
beach preference is kept as a stable reorder.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

# Occasions where sleepwear/underwear/swimwear is never right (old prompt: "Occasion context").
_FORMAL_Q = re.compile(
    r"\b(date[\s-]?night|evening\s+out|wedding|cocktail|sangeet|haldi|mehendi|mehndi|reception"
    r"|engagement)\b", re.I)
# Old `_DATE_NIGHT_RE` (the colour-diversity swap) plus the rule block it belongs to.
_DATE_NIGHT_Q = re.compile(
    r"\b(date\s+night|date-night|evening\s+out|evening|cocktail|wedding|gala)\b", re.I)
_BEACH_Q = re.compile(r"\b(beach|pool|poolside|seaside)\b", re.I)
_WINTER_Q = re.compile(r"\b(winter|autumn|fall)\b", re.I)
_SUMMER_Q = re.compile(r"\b(summer|beach)\b", re.I)
_PROF_Q = re.compile(r"\b(office|work|workwear|meeting)\b", re.I)

_NIGHT_SWIM = re.compile(
    r"\b(sleepwear|nightwear|nightdress|nighty|night\s+suit|pyjamas?|pajamas?|lounge\s?wear"
    r"|underwear|lingerie|bra|briefs|swimwear|swimsuit|swimming|bikini|monokini|swim)\b", re.I)
_SWIM = re.compile(r"\b(swimwear|swimsuit|swimming|bikini|monokini|swim|board\s+shorts)\b", re.I)
_CASUAL_NEUTRAL = re.compile(
    r"\b(beige|camel|brown|pastel|baby\s+pink|light\s+pink|peach|mint|lilac|lavender"
    r"|powder\s+blue|light\s+blue)\b", re.I)
_CASUAL_TYPES = re.compile(r"\b(sweaters?|sweatshirts?|hoodies?|t-?shirts?|tees?)\b", re.I)
_BEACH_DRESS_STYLE = re.compile(
    r"\b(bodycon|shift|midi|formal|cocktail|structured|tailored)\b", re.I)
_BEACH_HEAVY = re.compile(
    r"\b(blouse|blazers?|trousers?|jeans|coats?|overcoats?|knitwear|sweaters?|sweatshirts?)\b",
    re.I)
_SAND = re.compile(r"\b(beige|sand)\b", re.I)
_SUNDRESS = re.compile(r"\b(sundress|sun\s+dress|beach\s+dress|cover-?up|sarong|tank)\b", re.I)
_BEACH_PREFER = re.compile(
    r"\b(swimwear|bikini|swimsuit|cover-?up|sundress|beach\s+dress|tank|board\s+shorts|sarong"
    r"|flip-?flops?|beach\s+bag)\b", re.I)
_WINTER_BAD = re.compile(r"\b(shorts|swimwear|swimsuit|bikini|monokini|sandals?|flip-?flops?)\b", re.I)
_SUMMER_BAD = re.compile(r"\b(coats?|overcoats?|parkas?|puffer|sweaters?|knitwear)\b", re.I)
_DRESS = re.compile(r"\b(dress|gown)\b", re.I)

# Colour words a query can name (a named colour is a hard constraint, old prompt rule).
COLOUR_WORDS = (
    "black", "white", "red", "blue", "green", "yellow", "pink", "purple", "orange", "grey", "gray",
    "brown", "beige", "navy", "maroon", "burgundy", "gold", "silver", "cream", "ivory", "teal",
    "mustard", "olive", "peach", "lavender", "turquoise", "magenta",
)
_COLOUR_Q = re.compile(r"\b(" + "|".join(COLOUR_WORDS) + r")\b", re.I)


def _text(it: dict[str, Any]) -> str:
    return f"{it.get('display_name') or ''} {it.get('product_type') or ''}"


def _colour(it: dict[str, Any]) -> str:
    return str(it.get("colour") or "").lower()


def named_colours(query: str) -> set[str]:
    """Colour words the query names (gray/grey folded)."""
    return {("grey" if c.lower() == "gray" else c.lower()) for c in _COLOUR_Q.findall(query)}


def _colour_matches(it_colour: str, wanted: set[str]) -> bool:
    c = it_colour.replace("gray", "grey")
    return any(w in c for w in wanted)


def rule_violations(query: str, it: dict[str, Any]) -> list[str]:
    """Names of the hard rules this item breaks for this query ([] = compliant)."""
    out: list[str] = []
    text, colour = _text(it), _colour(it)
    desc = f"{text} {it.get('detail_desc') or ''}"
    wanted = named_colours(query)
    q = query.lower()
    swim = bool(_SWIM.search(text))
    if (_FORMAL_Q.search(query) or _PROF_Q.search(query)) and _NIGHT_SWIM.search(text):
        out.append("occasion_sleep_swim")
    if _DATE_NIGHT_Q.search(query):
        # a colour the user asked for is never "casual" (e.g. "light blue" for query "blue")
        if _CASUAL_NEUTRAL.search(colour) and not (wanted and _colour_matches(colour, wanted)):
            out.append("date_night_casual_colour")
        ct = _CASUAL_TYPES.search(text)
        if ct and ct.group(0).lower().rstrip("s") not in q:
            out.append("date_night_casual_type")
    if _BEACH_Q.search(query):
        if _DRESS.search(text) and _BEACH_DRESS_STYLE.search(desc):
            out.append("beach_dress_style")
        if _BEACH_HEAVY.search(text):
            out.append("beach_heavy_item")
        if (_SAND.search(colour) or _SAND.search(text)) and not swim and not _SUNDRESS.search(text):
            out.append("beach_sand")
    if _WINTER_Q.search(query) and _WINTER_BAD.search(text):
        out.append("winter_light_item")
    if _SUMMER_Q.search(query) and _SUMMER_BAD.search(text):
        out.append("summer_heavy_item")
    if wanted and colour and not _colour_matches(colour, wanted):
        out.append("colour_constraint")
    return out


def _diversity_swap(query: str, shown: list[dict], pool: Sequence[dict]) -> tuple[list[dict], bool]:
    """Old `_enforce_colour_diversity`: date-night picks all one colour -> swap the last pick for
    the best other-colour candidate. NEW guard: skipped when the query names a colour (the old
    prompt said 'unless the query explicitly names one colour'; the old code ignored that)."""
    if not _DATE_NIGHT_Q.search(query) or named_colours(query) or not shown:
        return shown, False
    colours = [_colour(it) for it in shown]
    if len(set(colours)) >= 2:
        return shown, False
    seen = {it["article_id"] for it in shown}
    for cand in pool:
        if cand["article_id"] not in seen and _colour(cand) != colours[0]:
            return shown[:-1] + [cand], True
    return shown, False


def apply_rules(
    query: str, ordered: list[dict], top_k: int
) -> tuple[list[dict], dict[str, Any]]:
    """`ordered` = the full candidate list, best first. Returns (top_k items, info).

    1. drop every item with a rule violation (never to empty: if all violate, keep the order and
       report `skipped`);  2. beach queries: stable-move preferred beach items to the front;
    3. take top_k;  4. colour-diversity swap from the remaining compliant candidates.
    The result can hold fewer than top_k items when fewer compliant candidates exist."""
    ok = [it for it in ordered if not rule_violations(query, it)]
    info: dict[str, Any] = {"n_removed": len(ordered) - len(ok), "skipped": False,
                            "diversity_swap": False}
    if not ok:  # nothing compliant: a rule set that rejects the whole pool is not trusted
        info.update(n_removed=0, skipped=True)
        ok = list(ordered)
    if _BEACH_Q.search(query):
        ok = [it for it in ok if _BEACH_PREFER.search(_text(it))] + [
            it for it in ok if not _BEACH_PREFER.search(_text(it))]
    shown, swapped = _diversity_swap(query, ok[:top_k], ok[top_k:])
    info["diversity_swap"] = swapped
    return shown, info
