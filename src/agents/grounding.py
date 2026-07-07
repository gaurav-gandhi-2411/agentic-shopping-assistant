from __future__ import annotations

import re

FORBIDDEN_PATTERNS: dict[str, list[str]] = {
    "price": [
        r"\bprice\b", r"\bcost\b", r"\bcheaper\b", r"\bexpensive\b",
        r"\baffordable\b", r"\bbudget\b", r"\bsale\b", r"\bdiscount\b",
        r"[\$€£]\d",
    ],
    "size": [
        r"\bruns big\b", r"\bruns small\b", r"\btrue to size\b",
        r"\bxs\b", r"\bxl\b", r"\bxxl\b",
        r"\btight fit\b", r"\bloose fit\b",
    ],
    "material_feel": [
        r"\bbreathable\b", r"\bsweat.wicking\b", r"\bwaterproof\b",
        r"\bwater.resistant\b", r"\bcool touch\b",
        r"\bwarm\b", r"\bcold\b",
    ],
}

_FALLBACK_MSGS: dict[str, str] = {
    "price": (
        "I don't have pricing information — you can check the product page for current prices."
    ),
    "size": (
        "I don't have size or fit information — check the product page for sizing guides."
    ),
    "material_feel": (
        "I don't have fabric performance details beyond what's listed in the product description."
    ),
}


def validate_response(
    response: str,
    retrieved_items: list[dict],
    *,
    allow_budget_mentions: bool = False,
) -> tuple[str, list[str]]:
    """Scan LLM response for ungrounded attribute claims.

    Returns (cleaned_response, flags).  Each flag is "category:pattern".
    A sentence containing a forbidden keyword is replaced with a standard
    disclaimer unless the keyword appears in any retrieved item's field values.
    Repeated fallback messages for the same category are deduplicated.

    Args:
        allow_budget_mentions: when True, exempts ONLY the "\\bbudget\\b" price
            pattern from the scrub — used by validate_rationale when the
            fact-sheet genuinely includes the user's own stated budget_inr, so
            a rationale can say "kept within your budget" without being
            scrubbed as an ungrounded price claim. Other price words (cost,
            cheaper, expensive, sale, discount) are still scrubbed since
            nothing grounds those as true.
    """
    if not response:
        return response, []

    backing = " ".join(
        " ".join(str(v or "") for v in it.values()) for it in retrieved_items
    ).lower()

    sentences = re.split(r"(?<=[.!?])\s+", response.strip())
    cleaned: list[str] = []
    flags: list[str] = []
    used_fallbacks: set[str] = set()

    for sentence in sentences:
        s_lower = sentence.lower()
        hit_cat: str | None = None
        hit_pat: str | None = None

        for category, patterns in FORBIDDEN_PATTERNS.items():
            for pat in patterns:
                if allow_budget_mentions and pat == r"\bbudget\b":
                    continue
                if re.search(pat, s_lower) and not re.search(pat, backing):
                    hit_cat = category
                    hit_pat = pat
                    break
            if hit_cat:
                break

        if hit_cat:
            flags.append(f"{hit_cat}:{hit_pat}")
            fallback = _FALLBACK_MSGS[hit_cat]
            if fallback not in used_fallbacks:
                cleaned.append(fallback)
                used_fallbacks.add(fallback)
        else:
            cleaned.append(sentence)

    return " ".join(cleaned), flags


# ── Rationale grounding ────────────────────────────────────────────────────────

# Conservative vocabulary of colour words we can confidently identify.
# Only flag tokens in this set — do NOT flag generic styling words.
_KNOWN_COLOURS: frozenset[str] = frozenset({
    "red", "dark red", "blue", "dark blue", "light blue", "navy",
    "green", "dark green", "yellow", "dark yellow", "light yellow",
    "orange", "dark orange", "light orange", "pink", "light pink",
    "dark pink", "purple", "dark purple", "grey", "gray", "dark grey",
    "light grey", "black", "white", "off white", "beige", "light beige",
    "brown", "gold", "silver", "turquoise", "dark turquoise", "khaki",
    "mustard", "maroon", "coral", "teal", "olive", "cream", "ivory",
    "magenta", "indigo", "violet",
})

# Common garment/product-type words we can confidently identify.
# Same conservative principle — only flag tokens that are clearly product types.
_KNOWN_PRODUCT_TYPES: frozenset[str] = frozenset({
    "shirt", "t-shirt", "tshirt", "blouse", "top", "crop top", "tank top",
    "sweater", "sweatshirt", "cardigan", "hoodie", "polo",
    "trousers", "jeans", "shorts", "skirt", "leggings",
    "dress", "gown", "jumpsuit", "playsuit", "dungarees",
    "jacket", "coat", "blazer", "waistcoat", "parka", "anorak",
    "kurta", "kurti", "kameez", "tunic", "kaftan",
    "lehenga", "saree", "anarkali", "sharara", "salwar",
    "churidar", "palazzo", "dhoti", "sherwani", "bandhgala",
    "dupatta", "stole", "scarf",
    "shoes", "sandals", "boots", "heels", "flats", "sneakers",
    "juttis", "jutti", "mojaris", "mojari", "kolhapuris", "wedges", "loafers",
    "bag", "clutch", "handbag", "purse",
    "jewellery", "jewelry", "necklace", "earrings", "bracelet",
})


# Matches a rupee figure however it's written: "₹5,000", "Rs. 5000", "INR 5000",
# "5000 rupees" — captures just the numeric group so it can be parsed as a float.
_RUPEE_AMOUNT_RE = re.compile(
    r"(?:[₹]|\bRs\.?\b|\bINR\b)\s*([\d][\d,]*(?:\.\d+)?)"
    r"|([\d][\d,]*(?:\.\d+)?)\s*(?:\bINR\b|\brupees?\b)",
    re.IGNORECASE,
)


def _extract_rupee_amounts(text: str) -> list[float]:
    """Pull every rupee figure out of a sentence as a float (commas stripped)."""
    amounts: list[float] = []
    for m in _RUPEE_AMOUNT_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        if raw:
            amounts.append(float(raw.replace(",", "")))
    return amounts


def validate_rationale(
    text: str,
    look_items: list[dict],
    occasion: str | None,
    extra_whitelist_tokens: set[str] | None = None,
    *,
    budget_inr: float | None = None,
) -> tuple[str, list[str]]:
    """Grounding gate for LLM-generated outfit rationales.

    Builds a whitelist from the actual look items (colours, product types, slot
    names) plus the occasion string, then scans the rationale sentence-by-sentence.
    A sentence is DROPPED if it names a colour or product-type word that is NOT in
    the whitelist.  Generic styling words ("balance", "hero", "neutral", "elevate",
    "anchor", "tone", "piece", "look", "outfit", "style", etc.) are never flagged.

    Also applies the same price/size/material scrubbing as validate_response,
    except the "budget" keyword is exempted when budget_inr is genuinely known
    (see validate_response's allow_budget_mentions) — other price words (cost,
    cheaper, expensive, sale, discount) are still scrubbed unconditionally.

    Args:
        extra_whitelist_tokens: additional lowercase tokens/phrases to whitelist
            beyond this look's own items — used by the cross-gender PARTNER
            styling feature (Phase B Part 2) so a companion-look rationale can
            reference the ORIGINAL anchor look's colour/product-type (which
            live in a DIFFERENT look's items, not this one) without being
            flagged as an ungrounded claim.
        budget_inr: the user's own stated budget in INR, when known. Affects the
            "budget" keyword exemption above, and is also used to catch a
            fabricated rupee figure: if a sentence states a ₹/Rs/INR amount that
            doesn't match budget_inr (within tolerance), the sentence is dropped
            as an ungrounded numeric claim — the LLM could otherwise say "under
            ₹5000" when the real budget is ₹3000.

    Returns (cleaned_text, flags) where flags use the same "category:token" format.
    If the cleaned text would be empty (all sentences dropped), returns the original
    text and includes a "rationale:all_dropped" flag so the caller can fall back.
    """
    if not text:
        return text, []

    # Build whitelist from real look attributes
    whitelist_tokens: set[str] = set()
    for item in look_items:
        colour = (item.get("colour") or "").lower().strip()
        if colour:
            whitelist_tokens.update(colour.split())
            whitelist_tokens.add(colour)  # multi-word colour as phrase too
        pt = (item.get("product_type") or "").lower().strip()
        if pt:
            whitelist_tokens.update(pt.split())
            whitelist_tokens.add(pt)
        slot = (item.get("_slot") or "").lower().strip()
        if slot:
            whitelist_tokens.add(slot)

    if occasion:
        whitelist_tokens.update(occasion.lower().replace("_", " ").split())
        whitelist_tokens.add(occasion.lower())

    if extra_whitelist_tokens:
        whitelist_tokens.update(t.lower().strip() for t in extra_whitelist_tokens if t)

    # Apply price/size/material scrubbing (reuse existing logic)
    # We treat look_items as retrieved_items for this purpose
    scrubbed, flags = validate_response(
        text, look_items, allow_budget_mentions=budget_inr is not None
    )
    if not scrubbed.strip():
        return text, flags + ["rationale:all_dropped"]

    # Now scan each sentence for ungrounded colour or product-type claims
    sentences = re.split(r"(?<=[.!?])\s+", scrubbed.strip())
    cleaned: list[str] = []

    for sentence in sentences:
        s_lower = sentence.lower()
        drop_reason: str | None = None

        # Check for colour tokens NOT in whitelist
        for colour_phrase in sorted(_KNOWN_COLOURS, key=len, reverse=True):
            if colour_phrase in s_lower:
                colour_words = colour_phrase.split()
                # Only flag if the colour phrase is NOT grounded in the whitelist
                if colour_phrase not in whitelist_tokens and not all(
                    w in whitelist_tokens for w in colour_words
                ):
                    drop_reason = f"rationale:ungrounded_colour:{colour_phrase}"
                    break

        if drop_reason is None:
            # Check for product-type tokens NOT in whitelist
            for pt_phrase in sorted(_KNOWN_PRODUCT_TYPES, key=len, reverse=True):
                if pt_phrase in s_lower:
                    pt_words = pt_phrase.split()
                    if pt_phrase not in whitelist_tokens and not all(
                        w in whitelist_tokens for w in pt_words
                    ):
                        # Also check single-word against whitelist individually
                        if not any(w in whitelist_tokens for w in pt_words):
                            drop_reason = f"rationale:ungrounded_type:{pt_phrase}"
                            break

        # Check for a fabricated budget figure: a ₹/Rs/INR amount that doesn't
        # match the user's real budget_inr (within a small rounding tolerance).
        if drop_reason is None and budget_inr is not None:
            for amount in _extract_rupee_amounts(sentence):
                tolerance = max(50.0, 0.02 * budget_inr)  # allow rounding, e.g. "under ₹5000"
                if abs(amount - budget_inr) > tolerance:
                    drop_reason = f"rationale:budget_mismatch:{amount:g}"
                    break

        if drop_reason:
            flags.append(drop_reason)
            # Do not append sentence — it's dropped
        else:
            cleaned.append(sentence)

    if not cleaned:
        # All sentences were dropped — signal caller to use template fallback
        return text, flags + ["rationale:all_dropped"]

    return " ".join(cleaned), flags
