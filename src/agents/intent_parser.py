"""
IntentParser (F3) — deterministic, zero-LLM query → structured intent.

parse_intent(raw_query) → IntentV1
merge_with_context(intent, session_context) → IntentV1

Intentionally has ZERO project imports so it can be tested and used in
isolation without loading the catalogue, index, or LLM layers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IntentV1:
    """Structured intent extracted from a single user turn."""

    garment_type: str | None  # "dress" | "kurti" | "shorts" | ... | None
    gender: str | None  # "women" | "men" | "unisex" | None
    colour: str | None  # canonical colour string | None
    occasion: str | None  # canonical occasion slug | None
    budget_max_inr: int | None  # upper budget bound in INR, or None
    store_filter: list[str] = field(default_factory=list)  # explicit store mentions
    raw_query: str = ""  # original text, NEVER modified
    is_product_query: bool = False  # True → product search path
    # P3 body-type-aware guidance (see src/agents/outfit/body_type.py for the
    # full registry). body_type is one of BASE_SHAPE_SLUGS there, or None.
    body_type: str | None = None
    body_modifiers: list[str] = field(default_factory=list)
    # True when the user asked a body-type QUESTION with no shape stated
    # ("what suits my body type") — routes to a deterministic clarify message
    # rather than product search (see graph.py's router_node short-circuit).
    wants_body_type_guidance: bool = False


# ---------------------------------------------------------------------------
# Compound garment terms  (longest first so "dress shirt" beats "shirt")
# ---------------------------------------------------------------------------

_COMPOUND_TERMS: dict[str, str] = {
    "dress shirt": "shirt",
    "co-ord set": "coord",
    "co ord set": "coord",
    "coord set": "coord",
    "co-ord": "coord",
    "dungaree dress": "dress",
    "shirt dress": "dress",
    "jacket dress": "dress",
    "skirt suit": "coord",
    # "palazzo pants" prevents "pants" (trousers rule) from overriding "palazzo" via
    # rightmost-match logic when both appear in the same two-word phrase.
    "palazzo pants": "palazzo",
}

_COMPOUND_SORTED: list[tuple[str, str]] = sorted(
    _COMPOUND_TERMS.items(), key=lambda kv: len(kv[0]), reverse=True
)

# ---------------------------------------------------------------------------
# Garment rules — order matters; first list order drives "last match wins"
# via position tracking (same algorithm as normalizer.py).
# ---------------------------------------------------------------------------

_GARMENT_RULES: list[tuple[str, str]] = [
    (r"\bshorts?\b", "shorts"),
    (r"\bminiskirt\b|\bmini skirt\b", "skirt"),
    (r"\bskirt\b", "skirt"),
    (r"\btrouser\b|\btrousers\b|\bpants\b|\bchino\b|\bchinos\b", "trousers"),
    (r"\bjean\b|\bjeans\b|\bdenim\b", "jeans"),
    (r"\bsarees?\b|\bsari\b", "saree"),
    (r"\blehenga\b", "lehenga"),
    (r"\banarkali\b", "anarkali"),
    (r"\bsharara\b", "sharara"),
    (r"\bpalazzo\b", "palazzo"),
    (r"\bkurti\b", "kurti"),
    (r"\bkurta\b", "kurta"),
    (r"\bdupatta\b", "dupatta"),
    (r"\bsalwar\b", "salwar"),
    (r"\bmonokini\b|\bswimsuit\b|\bbikini\b|\bswimwear\b", "swimwear"),
    (r"\bjumpsuit\b|\bplaysuit\b", "jumpsuit"),
    (r"\bblazers?\b", "blazer"),
    (
        r"\bjacket\b|\bcoat\b|\bbomber\b|\bpuffer\b|\bwindcheater\b|\bparka\b|\banorak\b",
        "outerwear",
    ),
    (r"\bsweater\b|\bsweatshirt\b|\bhoodie\b|\bcardigan\b|\bknitwear\b", "knitwear"),
    (r"\bdress(?:es)?\b|\bgown\b", "dress"),
    # Negative lookbehind prevents matching the "shirt" fragment inside "t-shirt"/"tshirt".
    (r"(?<!t-)(?<!t)\bshirt\b", "shirt"),
    (r"\bblouse\b", "blouse"),
    (r"\btunic\b", "tunic"),
    (r"\bt-shirt\b|\btshirt\b|\btee\b", "top"),
    (r"\btop\b", "top"),
    (r"\bvest\b|\btank\b", "vest"),
    (
        r"\bfootwear\b|\bshoe\b|\bshoes\b|\bsandal\b|\bsandals\b|\bsneaker\b|\bsneakers\b"
        r"|\bheels?\b|\bboot\b|\bboots\b|\bflats?\b|\bslipper\b|\bslippers\b",
        "footwear",
    ),
    (r"\bhandbag\b|\btote\b|\bcrossbody\b|\bpurse\b|\bclutch\b|\bbag\b", "bag"),
    (r"\bco-?ord\b", "coord"),
    (r"\bkaftan\b", "kaftan"),
    (r"\bbodysuit\b|\blingerie\b|\bbra\b", "innerwear"),
    (r"\bnightgown\b|\bnight\s+gown\b|\bpyjama\b|\bpajama\b|\bnightsuit\b", "nightwear"),
    # 2026-07-11 catalogue-gap follow-up: the catalogue's single sherwani was
    # given its own product_type_name facet (scripts/patch_thin_category_
    # facets.py) so a "sherwani for X" query can hard-filter to it instead of
    # relying on embedding similarity alone. "bandhgala" has zero current
    # inventory but costs nothing to recognise for when/if it's added.
    (r"\bsherwanis?\b", "sherwani"),
    (r"\bbandhgalas?\b", "bandhgala"),
]

_COMPILED_GARMENT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), gtype) for pattern, gtype in _GARMENT_RULES
]

# Preposition barrier — stop garment search after "for/under/with/to"
_BARRIER_RE = re.compile(r"\b(for|under|with|to)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Gender rules — scan in order. EXPLICIT gender words/relations (either gender)
# always win over garment-implied heuristics: "kurta" is unisex in this
# catalogue (it stocks men's kurtas), so "kurta for men" must not fall through
# to a garment-based "women" guess before the explicit "men" word is checked.
# Garment-implied rules only fire as a fallback when no explicit word matched
# (2026-07-11: found live via a post-deploy proof — "printed kurta for men"
# was resolving to gender=women in production because the old ordering
# checked "kurta" as a women-marker before the explicit "men" rule).
# ---------------------------------------------------------------------------

_GENDER_MAP: list[tuple[str, str]] = [
    # Explicit women signals — checked before men to avoid "women" containing
    # "men" substring (moot under \b word-boundary regex, but kept as belt-
    # and-braces since it costs nothing).
    (
        r"\bwomen\b|\bwoman\b|\bwomens\b|\bwomen's\b|\bwoman's\b"
        r"|\bladies\b|\bfemale\b|\bgirl\b|\bher\b|\bshe\b",
        "women",
    ),
    (
        r"\bfor\s+(?:my\s+)?wife\b|\bfor\s+(?:my\s+)?girlfriend\b"
        r"|\bfor\s+(?:my\s+)?mum\b|\bfor\s+(?:my\s+)?mom\b|\bfor\s+(?:my\s+)?daughter\b",
        "women",
    ),
    # Explicit men signals
    (r"\bmen\b|\bman\b|\bmens\b|\bmen's\b|\bmale\b|\bhim\b|\bhe\b", "men"),
    (
        r"\bfor\s+(?:my\s+)?husband\b|\bfor\s+(?:my\s+)?boyfriend\b"
        r"|\bfor\s+(?:my\s+)?dad\b|\bfor\s+(?:my\s+)?father\b"
        r"|\bfor\s+(?:my\s+)?son\b|\bfor\s+(?:my\s+)?brother\b",
        "men",
    ),
    # Garment-implied fallback (only reached when no explicit gender word
    # matched above). "kurta" is deliberately excluded — this catalogue
    # stocks men's kurtas (see gold_020/gold_028 in the strict gold set), so
    # it is NOT a reliable gender signal; "kurti" is a genuinely women-
    # specific term and stays.
    (r"\bsherwani\b|\bbandhgala\b", "men"),
    (
        r"\bsarees?\b|\bkurti\b|\blehenga\b|\bdupatta\b"
        r"|\banarkali\b|\bsharara\b|\bpalazzo\b",
        "women",
    ),
]

_COMPILED_GENDER: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), gender) for pattern, gender in _GENDER_MAP
]

# ---------------------------------------------------------------------------
# Colour map — sorted longest-first so "dark blue" beats "blue"
# ---------------------------------------------------------------------------

_COLOUR_MAP: dict[str, str] = {
    "black": "Black",
    "white": "White",
    "red": "Red",
    "blue": "Blue",
    "dark blue": "Dark Blue",
    "light blue": "Light Blue",
    "navy": "Navy Blue",
    "navy blue": "Navy Blue",
    "grey": "Grey",
    "gray": "Grey",
    "dark grey": "Dark Grey",
    "light grey": "Light Grey",
    "pink": "Pink",
    "light pink": "Light Pink",
    "green": "Green",
    "dark green": "Dark Green",
    "yellow": "Yellow",
    "orange": "Orange",
    "purple": "Purple",
    "beige": "Beige",
    "cream": "Cream",
    "off white": "Off White",
    "brown": "Brown",
    "khaki": "Khaki",
    "turquoise": "Turquoise",
    # Phase A colour-backfill extension (2026-07-06) — common catalogue colours that
    # were missing from the base map, added so query-side colour parsing and the
    # catalogue-side colour backfill (src/catalogue/cleaning.py) share one vocabulary.
    #
    # 2026-07-11 correction: these nine synonyms (mustard/burgundy/maroon/charcoal/
    # peach/olive/teal/cream/lavender) were originally collapsed onto an approximate
    # NEIGHBOURING catalogue value (e.g. mustard -> Yellow) instead of the catalogue's
    # own distinct value — live-proven: the catalogue has 58 distinct colour_group_name
    # values, and each of these nine has its OWN well-populated bucket (Navy Blue=770,
    # Mustard=295, Burgundy=136, Maroon=478, Charcoal=83, Peach=244, Olive=239, Teal=179,
    # Cream=131, Lavender=91 items) that a query naming that colour was silently never
    # matching. Now maps to the catalogue's own exact value; _COLOUR_FAMILY below
    # additionally widens the RETRIEVAL FILTER (not this display/scoring value) to the
    # near-synonym neighbours so genuinely mistagged items aren't newly excluded.
    "mustard": "Mustard",
    "burgundy": "Burgundy",
    "maroon": "Maroon",
    "wine": "Dark Red",  # no exact catalogue value for "wine" — Dark Red remains closest
    "lavender": "Lavender",
    "charcoal": "Charcoal",
    "peach": "Peach",
    "olive": "Olive",
    "teal": "Teal",
    "rust": "Rust",
}

_COLOUR_SORTED: list[tuple[str, str]] = sorted(
    _COLOUR_MAP.items(), key=lambda kv: len(kv[0]), reverse=True
)

# Colour-family groups for RETRIEVAL FILTERING only (see 2026-07-11 correction above) —
# when a query's colour resolves to one of these keys, the filter widens to the whole
# tuple (isin match, see hybrid_search._facet_value_matches) instead of the single exact
# value, so "navy blue blazer" also catches items the catalogue tagged plain "Dark Blue".
# Deliberately NOT used for display/scoring (intent.colour, colour_score) — a rationale
# should still say "navy", not "navy or dark blue".
_COLOUR_FAMILY: dict[str, tuple[str, ...]] = {
    "Navy Blue": ("Navy Blue", "Dark Blue"),
    "Mustard": ("Mustard", "Yellow"),
    "Burgundy": ("Burgundy", "Dark Red", "Maroon"),
    "Maroon": ("Maroon", "Dark Red", "Burgundy"),
    "Dark Red": ("Dark Red", "Maroon", "Burgundy"),
    "Charcoal": ("Charcoal", "Dark Grey"),
    "Peach": ("Peach", "Light Pink"),
    "Olive": ("Olive", "Khaki"),
    "Teal": ("Teal", "Turquoise", "Turquoise Blue"),
    "Cream": ("Cream", "Light Beige"),
    "Lavender": ("Lavender", "Purple"),
}


def colour_filter_values(colour: str | None) -> tuple[str, ...] | str | None:
    """Return the retrieval-filter value for a canonical colour: the family
    tuple if `colour` is a known-fragmented catalogue bucket (see
    _COLOUR_FAMILY), else the colour unchanged. None passes through as None.
    """
    if not colour:
        return colour
    return _COLOUR_FAMILY.get(colour, colour)

# ---------------------------------------------------------------------------
# Occasion map — sorted longest-first
# ---------------------------------------------------------------------------

_OCCASION_MAP: dict[str, str] = {
    "smart casual": "smart_casual",
    "date night": "party_evening",
    "ring ceremony": "engagement",
    "wedding": "wedding_guest",
    "shaadi": "wedding_guest",
    "sangeet": "sangeet",
    "haldi": "haldi",
    "mehendi": "mehendi",
    "mehndi": "mehendi",
    "reception": "reception",
    "cocktail": "reception",
    "engagement": "engagement",
    "roka": "engagement",
    "sagai": "engagement",
    "puja": "festive_puja",
    "festive": "festive_puja",
    "ethnic": "traditional_ethnic",
    "traditional": "traditional_ethnic",
    "party": "party_evening",
    "evening": "party_evening",
    "office": "office",
    "work": "office",
    "casual": "casual",
    "formal": "office",
    "beach": "casual",
    "brunch": "casual",
}

_OCCASION_SORTED: list[tuple[str, str]] = sorted(
    _OCCASION_MAP.items(), key=lambda kv: len(kv[0]), reverse=True
)

# ---------------------------------------------------------------------------
# P3: body-type map — sorted longest-first (same algorithm as colour/occasion).
#
# Kept as a LOCAL copy rather than importing src.agents.outfit.body_type.
# SYNONYMS, deliberately: this module's docstring states a zero-project-import
# invariant so IntentParser stays testable/usable without loading the
# catalogue/index/LLM layers. Keep this in sync with body_type.py's SYNONYMS
# dict when either changes (same established pattern as _OCCASION_MAP here
# duplicating occasions.py's OCCASIONS slugs).
# ---------------------------------------------------------------------------

_BODY_TYPE_MAP: dict[str, str] = {
    # pear
    "pear shaped": "pear",
    "pear-shaped": "pear",
    "pear shape": "pear",
    "pear": "pear",
    "triangle body": "pear",
    "curvy hips": "pear",
    "hip-forward": "pear",
    "hip forward": "pear",
    # apple
    "apple shaped": "apple",
    "apple-shaped": "apple",
    "apple shape": "apple",
    "apple": "apple",
    "round shape": "apple",
    "rounder middle": "apple",
    "midsection-forward": "apple",
    # hourglass
    "hourglass shaped": "hourglass",
    "hourglass-shaped": "hourglass",
    "hourglass shape": "hourglass",
    "hourglass": "hourglass",
    "balanced curves": "hourglass",
    # rectangle
    "rectangle shaped": "rectangle",
    "rectangle-shaped": "rectangle",
    "rectangle shape": "rectangle",
    "rectangle": "rectangle",
    "straight frame": "rectangle",
    "athletic frame": "rectangle",
    "athletic build": "rectangle",
    # inverted triangle
    "inverted triangle": "inverted_triangle",
    "inverted-triangle": "inverted_triangle",
    "broad-shouldered": "inverted_triangle",
    "broad shouldered": "inverted_triangle",
    # modifiers
    "plus size": "plus_size",
    "plus-size": "plus_size",
    "curvy": "plus_size",
    "petite": "petite",
    "tall": "tall",
}

_BASE_SHAPE_SLUGS: frozenset[str] = frozenset(
    {"pear", "apple", "hourglass", "rectangle", "inverted_triangle"}
)
_MODIFIER_SLUGS: frozenset[str] = frozenset({"petite", "tall", "plus_size"})

_BODY_TYPE_SORTED: list[tuple[str, str]] = sorted(
    _BODY_TYPE_MAP.items(), key=lambda kv: len(kv[0]), reverse=True
)

# "What suits my body type" / "which styles suit me" / "what should I wear
# for my body type" — a body-type QUESTION with no shape stated.
_BODY_TYPE_QUESTION_RE = re.compile(
    r"\bwhat\s+suits\s+my\s+body(?:\s+type)?\b"
    r"|\bwhich\s+styles?\s+suit(?:s)?\s+me\b"
    r"|\bwhat\s+should\s+i\s+wear\s+for\s+my\s+body(?:\s+type)?\b"
    r"|\bwhat.?s\s+my\s+body\s+type\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Store names
# ---------------------------------------------------------------------------

_STORES: frozenset[str] = frozenset(
    {"myntra", "flipkart", "snitch", "fashor", "powerlook", "virgio",
     "berrylush", "globalrepublic", "libas"}
)

# ---------------------------------------------------------------------------
# Budget regex patterns
# ---------------------------------------------------------------------------

# "under ₹1000", "below 2000", "less than 1500", "up to 3000", "max 500", "within 800"
_BUDGET_EXACT_RE = re.compile(
    r"(?:under|below|less\s+than|up\s+to|max|within|upto)\s*[₹rs\.]*\s*(\d[\d,]*)",
    re.IGNORECASE,
)

# "around 1000", "about 1500", "approximately 800"
_BUDGET_APPROX_RE = re.compile(
    r"(?:around|about|approximately)\s*[₹rs\.]*\s*(\d[\d,]*)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Product-query signals
# ---------------------------------------------------------------------------

_BUY_SIGNAL_RE = re.compile(
    # "need" and "want" are intentionally omitted here — they are too broad
    # ("I need advice", "I want to chat") would false-positive.  Garment-type
    # detection already handles "I need a jacket" / "I want a dress" via the
    # noun itself.  Explicit verbs below are unambiguously shopping-intent signals.
    r"\b(buy|shop|find|show|get|looking\s+for|search\s+for"
    r"|suggest|recommend|help\s+me\s+(?:buy|find|get|pick))\b",
    re.IGNORECASE,
)

_REFINEMENT_RE = re.compile(
    r"\b(in\s+(?:blue|red|black|white|green|yellow|pink|purple|grey|beige|orange|brown)"
    r"|cheaper|more\s+formal|more\s+casual|different\s+colo(?:u)?r"
    r"|change\s+colo(?:u)?r|similar|something\s+like\s+this|like\s+these)\b",
    re.IGNORECASE,
)

# Catch "show me more" as a product continuation
_MORE_RE = re.compile(r"\bshow\s+me\s+more\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_garment_type(text_lower: str) -> str | None:
    """Run compound-term lookup then position-aware garment rule scan.

    Mirrors the normalizer.py algorithm:
    1. Compound table (longest-first, first match wins).
    2. Full rule scan with position tracking.
    3. Preposition barrier — discard matches after the barrier.
    4. Rightmost (last) match wins.
    """
    # Step 1: compound table
    for phrase, gtype in _COMPOUND_SORTED:
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        if re.search(pattern, text_lower, re.IGNORECASE):
            return gtype

    # Step 2: collect all positions
    matches: list[tuple[int, str]] = []
    for compiled_re, gtype in _COMPILED_GARMENT_RULES:
        for m in compiled_re.finditer(text_lower):
            matches.append((m.start(), gtype))

    if not matches:
        return None

    # Step 3: preposition barrier
    earliest_pos = min(pos for pos, _ in matches)
    barrier_match = _BARRIER_RE.search(text_lower, earliest_pos)
    if barrier_match:
        barrier_pos = barrier_match.start()
        matches = [(pos, gt) for pos, gt in matches if pos < barrier_pos]

    if not matches:
        return None

    # Step 4: rightmost match
    _, winning_gtype = max(matches, key=lambda t: t[0])
    return winning_gtype


def _extract_gender(text_lower: str) -> str | None:
    """Return first gender match scanning women patterns before men."""
    for compiled_re, gender in _COMPILED_GENDER:
        if compiled_re.search(text_lower):
            return gender
    return None


def _extract_colour(text_lower: str) -> str | None:
    """Return canonical colour for longest matching phrase."""
    for phrase, canonical in _COLOUR_SORTED:
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        if re.search(pattern, text_lower):
            return canonical
    return None


def _extract_occasion(text_lower: str) -> str | None:
    """Return canonical occasion slug for longest matching phrase."""
    for phrase, slug in _OCCASION_SORTED:
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        if re.search(pattern, text_lower):
            return slug
    return None


def _extract_body_type(text_lower: str) -> tuple[str | None, list[str]]:
    """Return (base_type, modifiers) via longest-first, non-overlapping matching.

    Mirrors _extract_colour/_extract_occasion's longest-phrase-first algorithm,
    but allows MULTIPLE non-overlapping matches (a message can carry both a
    base shape and a modifier, e.g. "petite pear") rather than a single
    winner. A longer phrase's matched span is never re-claimed by a shorter
    phrase nested inside it (e.g. "curvy hips" wins over standalone "curvy").
    """
    consumed: list[tuple[int, int]] = []
    ordered: list[tuple[int, str]] = []

    for phrase, slug in _BODY_TYPE_SORTED:
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        for m in re.finditer(pattern, text_lower):
            span = (m.start(), m.end())
            if any(not (span[1] <= c[0] or span[0] >= c[1]) for c in consumed):
                continue
            consumed.append(span)
            ordered.append((span[0], slug))

    ordered.sort(key=lambda t: t[0])
    slugs_in_order = [slug for _, slug in ordered]

    base = next((s for s in slugs_in_order if s in _BASE_SHAPE_SLUGS), None)
    modifiers: list[str] = []
    for s in slugs_in_order:
        if s in _MODIFIER_SLUGS and s not in modifiers:
            modifiers.append(s)
    return base, modifiers


def _extract_budget(raw_query: str) -> int | None:
    """Extract upper budget bound in INR.

    Exact phrases (under/below/less than/up to/max/within) → exact value.
    Approximate phrases (around/about/approximately) → value × 1.3 (30% buffer).
    """
    m = _BUDGET_EXACT_RE.search(raw_query)
    if m:
        return int(m.group(1).replace(",", ""))
    m = _BUDGET_APPROX_RE.search(raw_query)
    if m:
        base = int(m.group(1).replace(",", ""))
        return int(base * 1.3)
    return None


def _extract_stores(text_lower: str) -> list[str]:
    """Return list of store names mentioned in the query (whole-word match)."""
    found: list[str] = []
    for store in sorted(_STORES):  # deterministic order
        pattern = r"\b" + re.escape(store) + r"\b"
        if re.search(pattern, text_lower):
            found.append(store)
    return found


def _is_product_query(
    text_lower: str,
    garment_type: str | None,
    occasion: str | None,
    store_filter: list[str],
) -> bool:
    """Determine whether the query belongs on the product search path."""
    if garment_type is not None:
        return True
    if occasion is not None:
        return True
    if store_filter:
        return True
    if _BUY_SIGNAL_RE.search(text_lower):
        return True
    if _REFINEMENT_RE.search(text_lower):
        return True
    if _MORE_RE.search(text_lower):
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_intent(raw_query: str) -> IntentV1:
    """Parse a raw user query into a structured IntentV1.

    Deterministic, zero LLM, stdlib only.

    Parameters
    ----------
    raw_query:
        The original user message exactly as received.

    Returns
    -------
    IntentV1
        All extracted fields; unmatched fields are None / empty list / False.
        raw_query is always preserved verbatim.
    """
    text_lower = raw_query.lower()

    garment_type = _extract_garment_type(text_lower)
    gender = _extract_gender(text_lower)
    colour = _extract_colour(text_lower)
    occasion = _extract_occasion(text_lower)
    budget_max_inr = _extract_budget(raw_query)
    store_filter = _extract_stores(text_lower)
    is_product = _is_product_query(text_lower, garment_type, occasion, store_filter)
    body_type, body_modifiers = _extract_body_type(text_lower)
    wants_body_type_guidance = bool(_BODY_TYPE_QUESTION_RE.search(text_lower))

    return IntentV1(
        garment_type=garment_type,
        gender=gender,
        colour=colour,
        occasion=occasion,
        budget_max_inr=budget_max_inr,
        store_filter=store_filter,
        raw_query=raw_query,
        is_product_query=is_product,
        body_type=body_type,
        body_modifiers=body_modifiers,
        wants_body_type_guidance=wants_body_type_guidance,
    )


def merge_with_context(intent: IntentV1, session_context: dict) -> IntentV1:
    """Merge a new turn's intent with accumulated session context.

    Fields carried forward from session_context when the new intent does not
    specify them: garment_type, gender, colour, occasion, budget_max_inr.

    Never overwrites a field already populated by the new intent.
    Always preserves raw_query from the new intent.

    Parameters
    ----------
    intent:
        The IntentV1 produced by parse_intent() for the current turn.
    session_context:
        Dict with keys "garment_type", "gender", "colour", "occasion",
        "budget_max_inr" (all str | None, budget_max_inr is int | None),
        "body_type" (str | None), "body_modifiers" (list[str] | None) from
        the prior resolved intent.

    Returns
    -------
    IntentV1
        A new IntentV1 with context-inherited fields filled in where the
        current turn left them None.
    """
    return IntentV1(
        garment_type=intent.garment_type
        if intent.garment_type is not None
        else session_context.get("garment_type"),
        gender=intent.gender
        if intent.gender is not None
        else session_context.get("gender"),
        colour=intent.colour
        if intent.colour is not None
        else session_context.get("colour"),
        occasion=intent.occasion
        if intent.occasion is not None
        else session_context.get("occasion"),
        budget_max_inr=intent.budget_max_inr
        if intent.budget_max_inr is not None
        else session_context.get("budget_max_inr"),
        store_filter=intent.store_filter,
        raw_query=intent.raw_query,
        is_product_query=intent.is_product_query,
        body_type=intent.body_type
        if intent.body_type is not None
        else session_context.get("body_type"),
        body_modifiers=intent.body_modifiers
        if intent.body_modifiers
        else (session_context.get("body_modifiers") or []),
        wants_body_type_guidance=intent.wants_body_type_guidance,
    )
