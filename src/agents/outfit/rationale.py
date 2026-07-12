"""Grounded stylist rationale generation for outfit looks.

Design contract
---------------
- ONE batched LLM call for all variants (cost-aware).
- Falls back to a deterministic template rationale on any LLM or parse failure.
- Every LLM-generated rationale is passed through validate_rationale (grounding
  gate) before being returned; all-dropped fallback substitutes the template.
- Works on the $0 Ollama path and in DEMO_MODE (no live LLM required for tests
  because the fallback path is always available).
- No new logging beyond what the LLM client already logs.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import TYPE_CHECKING

from src.agents.grounding import validate_rationale
from src.agents.outfit.body_type import (
    BASE_SHAPES,
    POSITIVE_TEMPLATES,
    contains_banned_framing,
)

if TYPE_CHECKING:
    from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


def _safe_str(val: object) -> str:
    """Return str(val) unless val is None, float NaN, or the sentinel string 'nan'."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    s = str(val)
    return "" if s.lower() == "nan" else s


# ── Display-attribute grounding (live defect #6, 2026-07-10 sweep) ─────────────
# Some stores' product_type is a raw multi-garment CATEGORY string ("Blue
# Blazers, Waistcoats and Suits", "Kurtas, Ethnic Sets and Bottoms") — read
# verbatim into a stylist note it produces prose like "the blue blazers,
# waistcoats and suits and black footwear keeps the focus" (wrong items, wrong
# colour, broken grammar). These helpers ground the note in the words the USER
# actually sees: the garment noun and colour word from the product's own name,
# falling back to product_type only when it is a clean single noun.

# Scanned in order — multi-word and more-specific nouns first so "kurta set"
# wins over "kurta" and "nehru jacket" over "jacket".
_NOTE_GARMENT_NOUNS: tuple[str, ...] = (
    "nehru jacket", "kurta set", "co-ord set", "t-shirt", "sherwani", "waistcoat",
    "anarkali", "lehenga", "saree", "dupatta", "palazzo", "churidar", "dhoti",
    "kurti", "kurta", "blazer", "trousers", "jeans", "chinos", "shorts", "skirt",
    "gown", "dress", "shirt", "jacket", "coat", "sweater", "cardigan", "shrug",
    "heels", "sandals", "loafers", "oxfords", "boots", "sneakers", "juttis",
    "mojaris", "shoes", "belt", "watch", "clutch", "necklace", "earrings", "top",
)

# Nouns that take a plural verb ("the trousers keep…", not "keeps").
_PLURAL_NOUNS: frozenset[str] = frozenset({
    "trousers", "jeans", "chinos", "shorts", "heels", "sandals", "loafers",
    "oxfords", "boots", "sneakers", "juttis", "mojaris", "shoes", "earrings",
})

# Colour words as they appear in real product names — multi-word first.
_NOTE_COLOUR_WORDS: tuple[str, ...] = (
    "navy blue", "off white", "sea green", "dark red", "dark green", "dark blue",
    "light beige", "light pink", "wine", "maroon", "burgundy", "rust", "teal",
    "olive", "mustard", "lavender", "peach", "cream", "beige", "khaki",
    "charcoal", "turquoise", "navy", "black", "white", "grey", "gold", "silver",
    "red", "blue", "green", "purple", "pink", "orange", "yellow", "brown",
)


def _first_vocab_match(text: str, vocab: tuple[str, ...]) -> str | None:
    """Return the vocab entry appearing EARLIEST in text (longest wins ties)."""
    best: tuple[int, int, str] | None = None
    low = text.lower()
    for word in vocab:
        m = re.search(rf"\b{re.escape(word)}\b", low)
        if m and (best is None or (m.start(), -len(word)) < (best[0], best[1])):
            best = (m.start(), -len(word), word)
    return best[2] if best else None


def _display_noun(product_type: object, prod_name: object) -> str:
    """A short garment noun safe to put in prose. Prefers the product NAME's own
    garment word; falls back to a clean (<=2 words, no comma) product_type."""
    name_noun = _first_vocab_match(_safe_str(prod_name), _NOTE_GARMENT_NOUNS)
    if name_noun:
        return name_noun
    pt = _safe_str(product_type).strip().lower()
    if pt and "," not in pt and len(pt.split()) <= 2:
        return pt
    pt_noun = _first_vocab_match(pt, _NOTE_GARMENT_NOUNS)
    return pt_noun or "piece"


def _display_colour(colour: object, prod_name: object) -> str:
    """The colour word shown to the user. Prefers the colour named in the product
    NAME (a "Wine … Kurta" must not be narrated as "purple" just because its
    catalogue colour-group is Purple); falls back to the colour attribute."""
    name_colour = _first_vocab_match(_safe_str(prod_name), _NOTE_COLOUR_WORDS)
    if name_colour:
        return name_colour
    return _safe_str(colour).lower().strip()

# ── Prompt template ────────────────────────────────────────────────────────────

_RATIONALE_SYSTEM = """\
You are a concise fashion stylist assistant. You will receive a JSON list of
outfit "fact-sheets". Each fact-sheet has: occasion, gender, seed_colour,
seed_type, complement_pairs (list of {slot, colour, type}), and colour_harmony.
Return a JSON array of rationale strings — one per outfit, in the same order.
Each rationale must:
  - Be 1–3 sentences.
  - Reference ONLY the provided colours, product types, slots, and occasion —
    PLUS coordinates_with_anchor_colour/coordinates_with_anchor_type when a
    fact-sheet includes them (see below).
  - NOT mention price, fabric composition, brand, size, or fit — EXCEPT
    budget_inr (see below), which is the user's OWN stated budget, not an
    item's price.
  - Sound like a warm personal stylist, not a spec sheet.

If a fact-sheet includes coordinates_with_anchor_colour and/or
coordinates_with_anchor_type, this is a PARTNER companion look for the other
half of a couple — briefly mention how it coordinates with that anchor item
(e.g. its colour), in addition to the usual rationale.

A fact-sheet may also include, ONLY when genuinely known:
  - occasion_register_hint: a short phrase describing the occasion's register
    (e.g. "bright daytime ceremony", "glamorous evening event") — use it to
    keep your wording occasion-correct (e.g. don't call a haldi look "moody"
    or a reception look "casual"), but never state it as a new fact beyond
    what the rest of the fact-sheet already grounds.
  - user_context: a short snippet of the user's own words from this
    conversation. You may echo their stated occasion/request back in your own
    phrasing (e.g. "for the sangeet you mentioned") — never introduce a new
    fact from it beyond restating what they actually said.
  - budget_inr: the user's own stated budget in INR. You may mention the look
    was kept within/around this budget — never invent a different number, and
    never state it as an item's individual price.
  - anchor_is_owned: true when the seed item is a garment the user already
    owns (uploaded by photo), not a catalogue item for sale — you may say
    something like "anchored on your own {seed_type}" instead of implying it
    can be bought.
  - body_type / body_type_style_hint: the user volunteered a body shape (this
    is OPT-IN — they asked for it). Use celebratory, body-positive language
    only: "balances", "highlights", "flows", "frames", "celebrates", "draws
    the eye", "creates one long line". NEVER use language that implies hiding,
    fixing, correcting, concealing, or camouflaging any part of the body, and
    never call any body shape or size "wrong" or "unflattering". Talk about
    what the GARMENT's cut/line does, never about a body "needing" anything.

Never state a fact that is not present in the fact-sheet.

Respond with ONLY the JSON array and nothing else."""

_RATIONALE_USER = "Fact-sheets:\n{fact_sheets_json}"

# One-line occasion register hints (Wave 7 wedding-occasion expansion) — keeps
# the LLM rationale's wording occasion-correct (e.g. never call a haldi look
# "moody", never call a reception look "casual") without inventing new facts.
# Only added for occasions where the generic wording is otherwise ambiguous;
# every other occasion omits the key (backward compatible — no behaviour
# change for existing call sites).
_OCCASION_REGISTER_HINTS: dict[str, str] = {
    "sangeet": "embellished evening event",
    "haldi": "bright daytime ceremony",
    "mehendi": "green-themed daytime ceremony",
    "reception": "glamorous evening event",
    "engagement": "elegant semi-formal ceremony",
}


# ── Public API ─────────────────────────────────────────────────────────────────


def build_fact_sheet(
    look: dict,
    *,
    partner_context: dict[str, str] | None = None,
    user_context: str | None = None,
    budget_inr: float | None = None,
    anchor_is_owned: bool = False,
    body_type: str | None = None,
    body_modifiers: list[str] | None = None,
) -> dict:
    """Extract only real, grounded attributes from a look dict.

    Never includes item price, fabric, brand, or size — only
    colour/type/slot/occasion, plus the additive session-level facts below
    (only added when genuinely known, so backward compatible with every
    existing call site that doesn't pass them).

    Args:
        look: A dict in the same shape as compose_outfit's return value.
        partner_context: Phase B Part 2 cross-gender partner styling — when
            given (keys "anchor_colour", "anchor_type"), adds
            "coordinates_with_anchor_colour"/"coordinates_with_anchor_type" so
            the LLM can mention how this companion look coordinates with the
            ORIGINAL anchor look. None for regular (non-partner) looks.
        user_context: a short snippet of the user's own words for this turn
            (e.g. their raw query) — lets the rationale reference their
            stated occasion/request in their own phrasing. None omits the key.
        budget_inr: the user's own stated budget in INR (the session-level
            constraint, NOT an item price) — lets the rationale mention the
            look was kept within budget. None/falsy omits the key.
        anchor_is_owned: True when this look's seed item is a garment the
            user already owns (uploaded by photo) rather than a catalogue
            item for sale. False omits the key.
        body_type: P3 — one of src.agents.outfit.body_type.BASE_SHAPE_SLUGS,
            volunteered by the user (opt-in). None omits both body_type keys.
        body_modifiers: P3 — "petite"/"tall"/"plus_size", or None.

    Returns:
        A compact fact-sheet dict safe to include in an LLM prompt. Includes
        "occasion_register_hint" when the look's occasion has a dedicated hint
        in _OCCASION_REGISTER_HINTS (omitted otherwise).
    """
    seed = look.get("seed_item") or {}
    complements = look.get("complements") or []
    occasion = look.get("occasion") or "casual"
    gender = look.get("gender") or "women"

    seed_name = seed.get("prod_name") or seed.get("display_name")
    seed_colour = _display_colour(seed.get("colour"), seed_name)
    seed_type = _display_noun(seed.get("product_type"), seed_name)

    complement_pairs = []
    for comp in complements:
        slot = comp.get("_slot") or ""
        comp_name = comp.get("prod_name") or comp.get("display_name")
        colour = _display_colour(comp.get("colour"), comp_name)
        pt = _display_noun(comp.get("product_type"), comp_name)
        if slot or colour or pt:
            complement_pairs.append({"slot": slot, "colour": colour, "type": pt})

    # Derive a simple colour-harmony label from coherence.py logic
    # (cheap string comparison — no external call)
    harmony = _colour_harmony_label(
        seed_colour,
        [c["colour"] for c in complement_pairs],
        occasion,
    )

    fact_sheet: dict = {
        "occasion": occasion.replace("_", " "),
        "gender": gender,
        "seed_colour": seed_colour,
        "seed_type": seed_type,
        "complement_pairs": complement_pairs,
        "colour_harmony": harmony,
    }
    register_hint = _OCCASION_REGISTER_HINTS.get(occasion)
    if register_hint:
        fact_sheet["occasion_register_hint"] = register_hint

    if partner_context:
        anchor_colour = (partner_context.get("anchor_colour") or "").strip()
        anchor_type = (partner_context.get("anchor_type") or "").strip()
        if anchor_colour:
            fact_sheet["coordinates_with_anchor_colour"] = anchor_colour
        if anchor_type:
            fact_sheet["coordinates_with_anchor_type"] = anchor_type

    if user_context and user_context.strip():
        # Trimmed to keep the batched prompt compact — a short conversational
        # snippet is all the LLM needs to echo the user's own phrasing back.
        fact_sheet["user_context"] = user_context.strip()[:300]
    if budget_inr:
        fact_sheet["budget_inr"] = budget_inr
    if anchor_is_owned:
        fact_sheet["anchor_is_owned"] = True

    if body_type and body_type in BASE_SHAPES:
        fact_sheet["body_type"] = body_type.replace("_", " ")
        style_hint = POSITIVE_TEMPLATES.get(body_type)
        if style_hint:
            fact_sheet["body_type_style_hint"] = style_hint
    if body_modifiers:
        fact_sheet["body_modifiers"] = [m.replace("_", " ") for m in body_modifiers]

    return fact_sheet


def _partner_whitelist_tokens(partner_context: dict[str, str] | None) -> set[str] | None:
    """Return grounding-whitelist tokens derived from partner_context, or None.

    See validate_rationale's extra_whitelist_tokens param — the ORIGINAL
    anchor look's colour/type words live in a DIFFERENT look's items, so they
    must be explicitly whitelisted for the partner look's own grounding check.
    """
    if not partner_context:
        return None
    tokens: set[str] = set()
    for key in ("anchor_colour", "anchor_type"):
        val = (partner_context.get(key) or "").lower().strip()
        if val:
            tokens.update(val.split())
            tokens.add(val)
    return tokens or None


def generate_rationales(
    looks: list[dict],
    llm: "LLMClient",
    *,
    occasion: str,
    gender: str,
    partner_context: dict[str, str] | None = None,
    user_context: str | None = None,
    budget_inr: float | None = None,
    anchor_is_owned: bool = False,
    body_type: str | None = None,
    body_modifiers: list[str] | None = None,
) -> list[str]:
    """Generate a grounded stylist rationale for each look in a single LLM call.

    Strategy:
    1. Build a fact-sheet per look (colours/types/slots/occasion only, plus the
       additive session-level facts below when known).
    2. Make ONE batched LLM call requesting a JSON list of rationale strings.
    3. Parse the JSON list; on any error (parse failure, wrong length, empty)
       fall back to template_rationale for each affected look.
    4. Run each rationale through validate_rationale; if the grounding gate
       empties a rationale, substitute the deterministic template.

    Args:
        looks:    List of look dicts (compose_outfit output shape).
        llm:      LLMClient to use for generation.
        occasion: Occasion slug (used for template fallback labelling).
        gender:   Gender string (used for fact-sheet context).
        partner_context: Phase B Part 2 — see build_fact_sheet. Applied to
            EVERY look in this batch (callers only pass partner_context when
            generating rationale for a single partner look).
        user_context: see build_fact_sheet. Applied to EVERY look in this
            batch — it's the same conversational turn for all of them.
        budget_inr: see build_fact_sheet. Also exempts the "budget" keyword
            from the grounding gate's price scrub (see validate_rationale).
        anchor_is_owned: see build_fact_sheet.
        body_type: see build_fact_sheet. Also selects the body-positive
            template fallback (see _body_type_template_rationale) whenever the
            LLM rationale is unavailable or fails the banned-framing gate.
        body_modifiers: see build_fact_sheet.

    Returns:
        A list of rationale strings, one per look (same order, same length).
    """
    if not looks:
        return []

    fact_sheets = [
        build_fact_sheet(
            look,
            partner_context=partner_context,
            user_context=user_context,
            budget_inr=budget_inr,
            anchor_is_owned=anchor_is_owned,
            body_type=body_type,
            body_modifiers=body_modifiers,
        )
        for look in looks
    ]

    # Attempt ONE batched LLM call
    llm_rationales: list[str | None] = [None] * len(looks)
    try:
        prompt = _RATIONALE_USER.format(
            fact_sheets_json=json.dumps(fact_sheets, ensure_ascii=False, indent=2)
        )
        raw = llm.generate(prompt, system=_RATIONALE_SYSTEM)
        parsed = _parse_json_list(raw)
        if isinstance(parsed, list) and len(parsed) == len(looks):
            for i, r in enumerate(parsed):
                if isinstance(r, str) and r.strip():
                    llm_rationales[i] = r.strip()
        else:
            logger.warning(
                "[rationale] LLM returned %s items, expected %d — using templates",
                len(parsed) if isinstance(parsed, list) else "non-list",
                len(looks),
            )
    except Exception as exc:
        logger.warning("[rationale] LLM call failed (%s) — using templates for all", exc)

    # Apply grounding gate; fall back to template on failure
    extra_whitelist_tokens = _partner_whitelist_tokens(partner_context)
    results: list[str] = []
    for i, look in enumerate(looks):
        all_items = _look_all_items(look)
        occ = look.get("occasion") or occasion

        llm_text = llm_rationales[i]
        # P3 HARD GUARDRAIL: this is the single choke point every generated
        # rationale passes through. Any banned body-framing word (see
        # body_type.BANNED_FRAMING_WORDS) discards the LLM text outright —
        # checked BEFORE the grounding gate since a banned-word hit is a tone
        # violation, not a grounding one, and must never reach the caller
        # regardless of whether this specific look has a body_type set (an
        # LLM can hallucinate this register even for a body-type-agnostic
        # look).
        if llm_text and contains_banned_framing(llm_text):
            logger.warning(
                "[rationale] banned body-framing language detected for look %d — "
                "using body-positive template fallback",
                i,
            )
            results.append(_body_type_template_rationale(look, body_type))
            continue

        if llm_text:
            # The prompt only ever saw the fact-sheet's cleaned nouns/colours
            # (_display_noun/_display_colour), which may not be raw item
            # attribute tokens — whitelist them so the grounding gate doesn't
            # drop sentences that reference the prompt's own facts.
            fact_tokens = {
                t for pair in fact_sheets[i].get("complement_pairs", [])
                for t in (pair.get("colour", ""), pair.get("type", "")) if t
            }
            fact_tokens.update(
                t for t in (fact_sheets[i].get("seed_colour"), fact_sheets[i].get("seed_type"))
                if t
            )
            look_whitelist = (extra_whitelist_tokens or set()) | fact_tokens
            cleaned, flags = validate_rationale(
                llm_text, all_items, occ, look_whitelist, budget_inr=budget_inr
            )
            grounding_flags = [f for f in flags if f.startswith("rationale:all_dropped")]
            if grounding_flags:
                logger.debug(
                    "[rationale] grounding dropped all sentences for look %d — using template",
                    i,
                )
                results.append(_body_type_template_rationale(look, body_type))
            else:
                results.append(cleaned)
        else:
            results.append(_body_type_template_rationale(look, body_type))

    return results


def _body_type_template_rationale(look: dict, body_type: str | None) -> str:
    """Template fallback for a body-type turn — grounded template + POSITIVE_TEMPLATES.

    Used whenever the LLM rationale is unavailable, fails grounding, or trips
    the banned-framing guardrail. When body_type is None (or unknown), this is
    identical to template_rationale(look) — fully backward compatible with
    every non-body-type call site.
    """
    base = template_rationale(look)
    if body_type and body_type in POSITIVE_TEMPLATES:
        return f"{base} {POSITIVE_TEMPLATES[body_type]}"
    return base


def template_rationale(look: dict) -> str:
    """Build a deterministic template rationale from real look attributes.

    Fully grounded by construction — only references colours, product types,
    slots, and occasion that are actually in the look.

    Args:
        look: A look dict in compose_outfit output shape.

    Returns:
        A short (1–2 sentence) rationale string.
    """
    seed = look.get("seed_item") or {}
    complements = look.get("complements") or []
    occasion = (look.get("occasion") or "casual").replace("_", " ")

    seed_name = seed.get("prod_name") or seed.get("display_name")
    seed_colour = _display_colour(seed.get("colour"), seed_name) or "classic"
    seed_type = _display_noun(seed.get("product_type"), seed_name)

    if complements:
        comp_names = []
        last_noun = ""
        for c in complements[:2]:  # mention at most 2
            c_name = c.get("prod_name") or c.get("display_name")
            ct = _display_noun(c.get("product_type"), c_name)
            cc = _display_colour(c.get("colour"), c_name)
            if ct == "piece":  # no recognisable garment noun — skip, don't babble
                continue
            comp_names.append(f"{cc} {ct}" if cc else ct)
            last_noun = ct
        if comp_names:
            complement_str = " and ".join(comp_names)
            # "the trousers keep…" vs "the waistcoat keeps…" — plural agreement
            verb = "keep" if (len(comp_names) > 1 or last_noun in _PLURAL_NOUNS) else "keeps"
            return (
                f"The {seed_colour} {seed_type} anchors this {occasion} look; "
                f"the {complement_str} {verb} the focus on the hero piece."
            )

    return (
        f"A {seed_colour} {seed_type} styled for {occasion} — "
        f"a clean, occasion-appropriate choice."
    )


# ── Private helpers ────────────────────────────────────────────────────────────


def _colour_harmony_label(
    seed_colour: str,
    complement_colours: list[str],
    occasion: str,
) -> str:
    """Return a simple harmony label without importing coherence.py at call time."""
    _NEUTRALS = frozenset({"black", "white", "grey", "beige", "off white", "silver"})
    if not complement_colours:
        return "monochromatic"
    c0 = complement_colours[0]
    if c0 == seed_colour:
        return "monochromatic"
    if c0 in _NEUTRALS or seed_colour in _NEUTRALS:
        return "neutral_accent"
    return "contrasting"


def _parse_json_list(raw: str) -> list:
    """Extract the first JSON array from an LLM response string.

    Returns an empty list on any parse failure.
    """
    raw = raw.strip()
    # Try direct parse first
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Walk and find the first [...] block
    start = raw.find("[")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(raw[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
            if in_str:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(raw[start: i + 1])
                        if isinstance(result, list):
                            return result
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

    return []


def _look_all_items(look: dict) -> list[dict]:
    """Return all items in a look as a flat list for grounding checks."""
    seed = look.get("seed_item")
    complements = look.get("complements") or []
    items = []
    if seed:
        items.append(seed)
    items.extend(complements)
    return items
