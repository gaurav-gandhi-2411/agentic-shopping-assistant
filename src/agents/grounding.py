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
) -> tuple[str, list[str]]:
    """Scan LLM response for ungrounded attribute claims.

    Returns (cleaned_response, flags).  Each flag is "category:pattern".
    A sentence containing a forbidden keyword is replaced with a standard
    disclaimer unless the keyword appears in any retrieved item's field values.
    Repeated fallback messages for the same category are deduplicated.
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
