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
from typing import TYPE_CHECKING

from src.agents.grounding import validate_rationale

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

# ── Prompt template ────────────────────────────────────────────────────────────

_RATIONALE_SYSTEM = """\
You are a concise fashion stylist assistant. You will receive a JSON list of
outfit "fact-sheets". Each fact-sheet has: occasion, gender, seed_colour,
seed_type, complement_pairs (list of {slot, colour, type}), and colour_harmony.
Return a JSON array of rationale strings — one per outfit, in the same order.
Each rationale must:
  - Be 1–3 sentences.
  - Reference ONLY the provided colours, product types, slots, and occasion.
  - NOT mention price, fabric composition, brand, size, or fit.
  - Sound like a warm personal stylist, not a spec sheet.
Respond with ONLY the JSON array and nothing else."""

_RATIONALE_USER = "Fact-sheets:\n{fact_sheets_json}"


# ── Public API ─────────────────────────────────────────────────────────────────


def build_fact_sheet(look: dict) -> dict:
    """Extract only real, grounded attributes from a look dict.

    Never includes price, fabric, brand, or size — only colour/type/slot/occasion.

    Args:
        look: A dict in the same shape as compose_outfit's return value.

    Returns:
        A compact fact-sheet dict safe to include in an LLM prompt.
    """
    seed = look.get("seed_item") or {}
    complements = look.get("complements") or []
    occasion = look.get("occasion") or "casual"
    gender = look.get("gender") or "women"

    seed_colour = _safe_str(seed.get("colour")).lower().strip()
    seed_type = _safe_str(seed.get("product_type")).lower().strip()

    complement_pairs = []
    for comp in complements:
        slot = comp.get("_slot") or ""
        colour = _safe_str(comp.get("colour")).lower().strip()
        pt = _safe_str(comp.get("product_type")).lower().strip()
        if slot or colour or pt:
            complement_pairs.append({"slot": slot, "colour": colour, "type": pt})

    # Derive a simple colour-harmony label from coherence.py logic
    # (cheap string comparison — no external call)
    harmony = _colour_harmony_label(
        seed_colour,
        [c["colour"] for c in complement_pairs],
        occasion,
    )

    return {
        "occasion": occasion.replace("_", " "),
        "gender": gender,
        "seed_colour": seed_colour,
        "seed_type": seed_type,
        "complement_pairs": complement_pairs,
        "colour_harmony": harmony,
    }


def generate_rationales(
    looks: list[dict],
    llm: "LLMClient",
    *,
    occasion: str,
    gender: str,
) -> list[str]:
    """Generate a grounded stylist rationale for each look in a single LLM call.

    Strategy:
    1. Build a fact-sheet per look (colours/types/slots/occasion only).
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

    Returns:
        A list of rationale strings, one per look (same order, same length).
    """
    if not looks:
        return []

    fact_sheets = [build_fact_sheet(look) for look in looks]

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
    results: list[str] = []
    for i, look in enumerate(looks):
        all_items = _look_all_items(look)
        occ = look.get("occasion") or occasion

        llm_text = llm_rationales[i]
        if llm_text:
            cleaned, flags = validate_rationale(llm_text, all_items, occ)
            grounding_flags = [f for f in flags if f.startswith("rationale:all_dropped")]
            if grounding_flags:
                logger.debug(
                    "[rationale] grounding dropped all sentences for look %d — using template",
                    i,
                )
                results.append(template_rationale(look))
            else:
                results.append(cleaned)
        else:
            results.append(template_rationale(look))

    return results


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

    seed_colour = _safe_str(seed.get("colour")).lower() or "classic"
    seed_type = _safe_str(seed.get("product_type")).lower() or "piece"

    if complements:
        comp_names = []
        for c in complements[:2]:  # mention at most 2
            ct = (_safe_str(c.get("product_type")) or _safe_str(c.get("display_name"))).lower()
            cc = _safe_str(c.get("colour")).lower()
            if ct and cc:
                comp_names.append(f"{cc} {ct}")
            elif ct:
                comp_names.append(ct)
        if comp_names:
            complement_str = " and ".join(comp_names)
            return (
                f"The {seed_colour} {seed_type} anchors this {occasion} look; "
                f"the {complement_str} keeps the focus on the hero piece."
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
