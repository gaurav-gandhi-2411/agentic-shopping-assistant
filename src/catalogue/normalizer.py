"""
GarmentNormalizer — deterministic keyword/rule based garment type normalizer.

Derives a reliable garment_type, coarse category, and type_confidence from
a product name + optional store label + optional brand, using only stdlib.
No LLM. No project imports.

Mandatory spot-check results (verified by test_normalizer.py):
    "Shorts For Under Dresses"          -> shorts     (high)
    "Dress Shirt"                       -> shirt      (high)  [compound table]
    "Jacket Dress"                      -> dress      (high)  [rightmost-noun]
    "Mini Skirt"                        -> skirt      (high)  [compound table]
    "Co-Ord Set"                        -> coord      (high)  [compound table]
    "DressBerry Women Black Shorts"     -> shorts     (high)  [brand-strip]
    "20Dresses Crop Jacket"             -> outerwear  (high)  [brand-strip + rightmost]
    "DressBerry Sweater"                -> knitwear   (high)  [brand-strip]
    "Black Floral Maxi Dress"           -> dress      (high)
    "Kurti For Women"                   -> kurti      (high)  [barrier before "For Women" stops no garment noun]
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class NormalizationResult:
    """Container for a single garment-type normalization result."""

    garment_type: str | None  # e.g. "dress", "shorts", "kurti", None
    category: str  # "apparel" | "footwear" | "accessories" | "outerwear" | "unknown"
    type_confidence: str  # "high" | "medium" | "low" | "unknown"


# ---------------------------------------------------------------------------
# Compound-term lookup table (longest phrase wins — scan order is longest first)
# ---------------------------------------------------------------------------

_COMPOUND_TERMS: dict[str, str] = {
    "dungaree dress": "dress",
    "dress material": "fabric_material",  # unstitched fabric bolts, not wearable garments
    "shirt dress": "dress",
    "blouse piece": "fabric_material",    # raw fabric sold with sarees
    "skirt suit": "coord",
    "co-ord set": "coord",
    "co ord set": "coord",
    "coord set": "coord",
    "dress shirt": "shirt",
    "co-ord": "coord",
    "unstitched": "fabric_material",      # any "unstitched X" title = raw material
}

# Pre-sorted longest → shortest so the first match wins when phrases overlap
_COMPOUND_SORTED: list[tuple[str, str]] = sorted(
    _COMPOUND_TERMS.items(), key=lambda kv: len(kv[0]), reverse=True
)

# ---------------------------------------------------------------------------
# Garment rule list  (order matters for the position scan)
# ---------------------------------------------------------------------------

_GARMENT_RULES: list[tuple[str, str, str]] = [
    # Bottoms & shorts — specific first so "under dresses" purpose clause doesn't win
    (r"\bshorts?\b", "shorts", "apparel"),
    (r"\bminiskirt\b|\bmini skirt\b", "skirt", "apparel"),
    (r"\bskirt\b", "skirt", "apparel"),
    (r"\btrouser\b|\btrousers\b|\bpants\b|\bchino\b|\bchinos\b", "trousers", "apparel"),
    (r"\bjean\b|\bjeans\b|\bdenim\b", "jeans", "apparel"),
    # Ethnic
    (r"\bsarees?\b|\bsari\b", "saree", "apparel"),
    (r"\blehenga\b", "lehenga", "apparel"),
    (r"\banarkali\b", "anarkali", "apparel"),
    (r"\bsharara\b", "sharara", "apparel"),
    (r"\bpalazzo\b", "palazzo", "apparel"),
    (r"\bkurti\b", "kurti", "apparel"),
    (r"\bkurta\b", "kurta", "apparel"),
    (r"\bdupatta\b", "dupatta", "apparel"),
    (r"\bsalwar\b", "salwar", "apparel"),
    # Swimwear
    (r"\bmonokini\b|\bswimsuit\b|\bbikini\b|\bswimwear\b", "swimwear", "apparel"),
    # One-piece
    (r"\bjumpsuit\b|\bplaysuit\b", "jumpsuit", "apparel"),
    # Dungarees (bib-overalls style; treated as jumpsuit-adjacent)
    (r"\bdungaree\b|\bdungarees\b", "jumpsuit", "apparel"),
    # Outerwear — blazer before coat/jacket so "blazer" is specific
    (r"\bblazer\b", "blazer", "outerwear"),
    (
        r"\bjacket\b|\bcoat\b|\bbomber\b|\bpuffer\b|\bwindcheater\b|\bparka\b|\banorak\b",
        "outerwear",
        "outerwear",
    ),
    # Knitwear
    (r"\bsweater\b|\bsweatshirt\b|\bhoodie\b|\bcardigan\b|\bknitwear\b", "knitwear", "apparel"),
    # Dress — AFTER shorts/skirts so "shorts for under dresses" doesn't pick up "dress"
    (r"\bdress(?:es)?\b|\bgown\b", "dress", "apparel"),
    # Tops (shirt after dress so "dress shirt" handled by compound table,
    # but standalone "shirt" still maps correctly).
    # Negative lookbehind prevents matching the "shirt" inside "t-shirt" or "tshirt".
    (r"(?<!t-)(?<!t)\bshirt\b", "shirt", "apparel"),
    (r"\bblouse\b", "blouse", "apparel"),
    (r"\btunic\b", "tunic", "apparel"),
    (r"\bt-shirt\b|\btshirt\b|\btee\b", "top", "apparel"),
    (r"\btop\b", "top", "apparel"),
    (r"\bvest\b|\btank\b", "vest", "apparel"),
    # Footwear
    (
        r"\bfootwear\b|\bshoe\b|\bshoes\b|\bsandal\b|\bsandals\b|\bsneaker\b|\bsneakers\b"
        r"|\bheels?\b|\bboot\b|\bboots\b|\bflats?\b|\bslipper\b|\bslippers\b",
        "footwear",
        "footwear",
    ),
    # Bags
    (r"\bhandbag\b|\btote\b|\bcrossbody\b|\bpurse\b|\bclutch\b|\bbag\b", "bag", "accessories"),
    # Coord set (catch-all for coord after compound table)
    (r"\bco-?ord\b", "coord", "apparel"),
    # Kaftan
    (r"\bkaftan\b", "kaftan", "apparel"),
    # Bodysuit / lingerie
    (r"\bbodysuit\b|\blingerie\b|\bbra\b|\bpanty\b|\bpanties\b", "innerwear", "apparel"),
    # Night wear
    (
        r"\bnightgown\b|\bnight\s+gown\b|\bpyjama\b|\bpajama\b|\bnightsuit\b",
        "nightwear",
        "apparel",
    ),
]

# Pre-compiled for performance
_COMPILED_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(pattern, re.IGNORECASE), gtype, cat)
    for pattern, gtype, cat in _GARMENT_RULES
]

# Preposition barrier pattern — anchors where the garment noun search stops
_BARRIER_RE = re.compile(r"\b(for|under|with|to)\b", re.IGNORECASE)

# Saree word / "blouse piece" phrase — co-occurrence means a finished saree sold
# with a bundled blouse fabric swatch (see Step 1.5 override below). NOT used to
# exempt bare "saree" mentions on their own (e.g. "Saree Mall" is a brand name,
# and its "Unstitched Dress Material" products correctly stay fabric_material).
_SAREE_WORD_RE = re.compile(r"\bsarees?\b|\bsari\b", re.IGNORECASE)
_BLOUSE_PIECE_RE = re.compile(r"blouse\s*piece", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_garment_type(
    prod_name: str,
    product_type_name: str | None = None,
    brand: str | None = None,
) -> NormalizationResult:
    """Derive garment_type, category, and type_confidence from product metadata.

    Algorithm
    ---------
    1. Brand-prefix strip — remove leading brand token from the lowercased name.
    2. Compound-term lookup — exact whole-word phrase match for ambiguous combos
       such as "dress shirt" → "shirt".
    3. Full garment-rule scan with position tracking.
    4. Preposition barrier — discard garment nouns that appear after the first
       occurrence of "for/under/with/to" following the earliest garment match.
    5. Select the rightmost remaining match (head noun in a compound).
    6. Fall back to product_type_name (store label) if name yields nothing.

    Parameters
    ----------
    prod_name:
        Raw product title from the catalogue feed.
    product_type_name:
        Optional store-assigned category label (used as a fallback).
    brand:
        Optional brand name used to strip a leading brand token from the title.

    Returns
    -------
    NormalizationResult
        garment_type  — canonical garment string or None
        category      — "apparel" | "footwear" | "accessories" | "outerwear" | "unknown"
        type_confidence — "high" | "medium" | "low" | "unknown"
    """
    # ── Step 1: brand-prefix strip ──────────────────────────────────────────────
    name_lower = prod_name.lower().lstrip(" \t\n\r,.-_")
    if brand:
        brand_lower = brand.lower().strip()
        skip_brands = {"unknown", "mixed", "n/a", ""}
        if brand_lower not in skip_brands:
            brand_prefix_re = re.compile(
                r"^" + re.escape(brand_lower) + r"[\s\-_,]+", re.IGNORECASE
            )
            name_lower = brand_prefix_re.sub("", name_lower)

    residual = name_lower

    # ── Step 1.5: finished-saree-with-blouse-piece override ─────────────────────
    # A saree word co-occurring with "blouse piece" is always a finished, shoppable
    # saree — regardless of whether the blouse piece itself is "unstitched" (the
    # dominant real-world pattern: "Saree With Unstitched Blouse Piece") and
    # regardless of noun position (some listings use "&" instead of "with", so
    # there is no preposition barrier to demote the trailing "blouse" noun — e.g.
    # "Sangria Blue Striped Saree & Embellished Blouse Piece"). This must be
    # checked BEFORE the generic compound-term loop and the rightmost-noun rule,
    # both of which would otherwise let "unstitched"/"blouse" win.
    if _SAREE_WORD_RE.search(residual) and _BLOUSE_PIECE_RE.search(residual):
        return NormalizationResult(garment_type="saree", category="apparel", type_confidence="high")

    # ── Step 2: compound-term lookup ────────────────────────────────────────────
    for phrase, gtype in _COMPOUND_SORTED:
        # Whole-word phrase match anywhere in the residual
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        if re.search(pattern, residual, re.IGNORECASE):
            # Derive category from the matched garment type
            cat = _category_for(gtype)
            return NormalizationResult(
                garment_type=gtype,
                category=cat,
                type_confidence="high",
            )

    # ── Step 3: collect all garment noun matches with positions ─────────────────
    matches: list[tuple[int, str, str]] = []  # (start_pos, garment_type, category)
    for compiled_re, gtype, cat in _COMPILED_RULES:
        for m in compiled_re.finditer(residual):
            matches.append((m.start(), gtype, cat))

    if not matches:
        # Fall through to product_type_name fallback below
        return _fallback_product_type(product_type_name)

    # ── Step 4: preposition barrier ─────────────────────────────────────────────
    earliest_match_pos = min(pos for pos, _, _ in matches)
    barrier_match = _BARRIER_RE.search(residual, earliest_match_pos)
    if barrier_match:
        barrier_pos = barrier_match.start()
        matches = [(pos, gt, cat) for pos, gt, cat in matches if pos < barrier_pos]

    if not matches:
        return _fallback_product_type(product_type_name)

    # ── Step 5: select rightmost match (head noun) ──────────────────────────────
    _, winning_gtype, winning_cat = max(matches, key=lambda t: t[0])
    return NormalizationResult(
        garment_type=winning_gtype,
        category=winning_cat,
        type_confidence="high",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _category_for(garment_type: str) -> str:
    """Return the coarse category string for a canonical garment_type token."""
    if garment_type in {"footwear"}:
        return "footwear"
    if garment_type in {"bag"}:
        return "accessories"
    if garment_type in {"blazer", "outerwear"}:
        return "outerwear"
    if garment_type in {"fabric_material"}:
        return "raw_material"
    return "apparel"


def _fallback_product_type(product_type_name: str | None) -> NormalizationResult:
    """Scan product_type_name through garment rules without barrier logic.

    Returns confidence="medium" when a match is found, "unknown" otherwise.
    """
    if not product_type_name:
        return NormalizationResult(garment_type=None, category="unknown", type_confidence="unknown")

    label_lower = product_type_name.lower().strip()

    # Finished-saree-with-blouse-piece override — see Step 1.5 in normalize_garment_type.
    if _SAREE_WORD_RE.search(label_lower) and _BLOUSE_PIECE_RE.search(label_lower):
        return NormalizationResult(garment_type="saree", category="apparel", type_confidence="medium")

    # Check compound terms first
    for phrase, gtype in _COMPOUND_SORTED:
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        if re.search(pattern, label_lower, re.IGNORECASE):
            cat = _category_for(gtype)
            return NormalizationResult(garment_type=gtype, category=cat, type_confidence="medium")

    # Then rule scan (no barrier on store label — it's a short categorical string)
    matches: list[tuple[int, str, str]] = []
    for compiled_re, gtype, cat in _COMPILED_RULES:
        for m in compiled_re.finditer(label_lower):
            matches.append((m.start(), gtype, cat))

    if not matches:
        return NormalizationResult(garment_type=None, category="unknown", type_confidence="unknown")

    _, winning_gtype, winning_cat = max(matches, key=lambda t: t[0])
    return NormalizationResult(
        garment_type=winning_gtype,
        category=winning_cat,
        type_confidence="medium",
    )
