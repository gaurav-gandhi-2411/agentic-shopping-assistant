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
    # 2026-07-13 fix: "kurta and pyjama"/"pajama" SET titles (e.g. "Men Kurta and
    # Pyjama Set Dupion Silk") were resolving to garment_type="nightwear" —
    # "pyjama"/"pajama" is rightmost and wins the position scan over "kurta"
    # (see _GARMENT_RULES's nightwear rule). These are ethnic kurta-pajama sets,
    # not nightwear. Deliberately narrow to the kurta-combination phrases only —
    # bare "pyjama set"/"pajama set" (no "kurta") must still resolve to
    # nightwear (genuine men's pyjama-only sets). Mirrored in
    # src/agents/intent_parser.py's _COMPOUND_TERMS (same algorithm — see this
    # module's docstring).
    "kurta and pyjama": "kurta",
    "kurta pyjama": "kurta",
    "kurta pajama": "kurta",
    "kurta and pajama": "kurta",
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
    # Wedding/formal ethnic wear (2026-07-19, men's-ethnic-depth wave). Sherwani,
    # bandhgala, jodhpuri suit, and indo-western are kept as "apparel" (not
    # "outerwear") for the same reason kurta/lehenga/anarkali above are: each is
    # the complete/main garment of the outfit, not a layer worn over another
    # garment the way a blazer/jacket is. No separate "ethnic_wear" category
    # exists in this schema, so these share "apparel" with the ethnic rules above.
    (r"\bsherwani\b", "sherwani", "apparel"),
    # Real feeds use both spellings ("Black Bandhgala", "Grey Bandgala Suit").
    (r"\bband(?:h)?gala\b", "bandhgala", "apparel"),
    (r"\bjodhpuri\s+suits?\b", "jodhpuri_suit", "apparel"),
    # Covers "indo western", "indo-western", and "indowestern" (no separator).
    (r"\bindo\s*-?\s*western\b", "indowestern", "apparel"),
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
        r"|\bheels?\b|\bboot\b|\bboots\b|\bflats?\b|\bslipper\b|\bslippers\b"
        # 2026-07-19 fix: ethnic-footwear brands (kraftojodhpur, houseofvian,
        # 5-elements, taurjuttis, fizzygoblet) use jutti/mojari/kolhapuri/chappal
        # as their footwear noun — none of these were in the rule set, so titles
        # like "Amber Jute Men's Sherwani Jutti" fell through to "sherwani" (the
        # only recognized noun) instead of footwear. "kolhapuris" plural is listed
        # explicitly (not just "kolhapuri") because real titles ("Rangeena
        # Kolhapuris & Handbag Combo") use the plural and \bkolhapuri\b does not
        # match inside "kolhapuris" (no word boundary between "i" and the "s").
        r"|\bjutti\b|\bjuttis\b|\bmojari\b|\bmojaris\b|\bkolhapuri\b|\bkolhapuris\b"
        r"|\bchappal\b|\bchappals\b",
        "footwear",
        "footwear",
    ),
    # Bags
    (r"\bhandbag\b|\btote\b|\bcrossbody\b|\bpurse\b|\bclutch\b|\bbag\b", "bag", "accessories"),
    # Jewellery / fine accessories (2026-07-19 jewellery-inventory-gap wave).
    # theamethyststore/southtemplejewellery/daivik are dedicated jewellery
    # stores whose titles put an apparel-fragment word ("short", "shirt",
    # "saree") *before* the actual jewellery noun — e.g. "Simrath Short
    # Necklace Set", "James Shirt Button Clip", "Laksmi Lotus Saree Pin".
    # Confirmed live: 1,488 theamethyststore rows ("Short Necklace Set" ->
    # garment_type="shorts"), 60 theamethyststore rows ("Shirt Button Clip"
    # -> "shirt"), 57 southtemplejewellery rows ("Short Necklace Set" variants
    # -> "shorts", including via the store's own product_type_name label
    # "Short Necklaces" being re-scanned by the _fallback_product_type path),
    # and 56 daivik rows ("... Short ... Necklace/Haram ...") + 11 daivik rows
    # ("Saree Pin" -> "saree"). None of these jewellery nouns existed in the
    # rule set at all, so the coincidental apparel-fragment word was the ONLY
    # match and won by default. Adding the real jewellery nouns here is
    # sufficient — the existing rightmost-noun scan (Step 5) already prefers
    # the noun that sits after "short"/"shirt"/"saree" in every confirmed
    # title, no separate override step needed (contrast with the footwear-led
    # combo override above, where the bag noun sits to the *right* of the
    # noun that should win).
    # "necklace"/"earrings"/"jhumka" use the exact garment_type strings
    # src/agents/outfit/slots.py's _ACCESSORY_JEWELLERY_FAMILY already expects
    # (that constant predates this fix and was previously unreachable for
    # these brands); other jewellery nouns fall to generic "jewellery", which
    # that same frozenset also recognizes.
    (r"\bnecklaces?\b", "necklace", "accessories"),
    (r"\bearrings?\b", "earrings", "accessories"),
    (r"\bjhumkas?\b", "jhumka", "accessories"),
    (r"\bharams?\b", "jewellery", "accessories"),
    (r"\bpendants?\b", "jewellery", "accessories"),
    (r"\bchains?\b", "jewellery", "accessories"),
    (r"\bpins?\b", "jewellery", "accessories"),
    (r"\bclips?\b", "jewellery", "accessories"),
    (r"\bkasumalai\b", "jewellery", "accessories"),
    (r"\bguttapusalu\b|\bguttaspusalu\b", "jewellery", "accessories"),
    # "Jada Billa"/"Jadabilla"/"JadaBillai" (hair-ornament, daivik).
    (r"\bjada\s*billa(?:i)?\b", "jewellery", "accessories"),
    (r"\bmattal\b", "jewellery", "accessories"),
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

# Footwear noun / bag noun — co-occurrence means a footwear-led combo listing
# (see Step 1.6 override below), e.g. "Firdaus Juttis & Clutch Combo" and
# "Rangeela Kolhapuris & Handbag Combo" (real kraftojodhpur/houseofvian/
# 5-elements titles, 2026-07-19 ethnic-footwear wave). The bag noun ("clutch",
# "handbag") is always the second, bundled item and sits to the right of the
# footwear noun, so it would otherwise win the rightmost-noun scan in Step 5.
_FOOTWEAR_WORD_RE = re.compile(
    r"\bfootwear\b|\bshoe\b|\bshoes\b|\bsandal\b|\bsandals\b|\bsneaker\b|\bsneakers\b"
    r"|\bheels?\b|\bboot\b|\bboots\b|\bflats?\b|\bslipper\b|\bslippers\b"
    r"|\bjutti\b|\bjuttis\b|\bmojari\b|\bmojaris\b|\bkolhapuri\b|\bkolhapuris\b"
    r"|\bchappal\b|\bchappals\b",
    re.IGNORECASE,
)
_BAG_WORD_RE = re.compile(
    r"\bhandbag\b|\btote\b|\bcrossbody\b|\bpurse\b|\bclutch\b|\bbag\b", re.IGNORECASE
)


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
       1.5/1.6. Early-return overrides for finished-saree-with-blouse-piece and
       footwear-led bag combos (see inline comments) that would otherwise be
       mis-resolved by the generic rightmost-noun scan below.
    2. Compound-term lookup — exact whole-word phrase match for ambiguous combos
       such as "dress shirt" → "shirt".
    3. Full garment-rule scan with position tracking.
       3.5. Jewellery-noun precedence — drop "shorts"/"shirt"/"saree"/"top"
       matches outright when a real jewellery noun (necklace/earrings/jhumka/
       jewellery) is also present, regardless of which side it falls on.
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

    # ── Step 1.6: footwear-led combo override ───────────────────────────────────
    # A footwear noun (jutti/mojari/kolhapuri/chappal/shoe/etc.) co-occurring with
    # a bag noun (clutch/handbag/potli-adjacent "bag") is a footwear item bundled
    # with an accessory add-on, not a bag — the footwear is the lead product being
    # sold (e.g. "Firdaus Juttis & Clutch Combo"). Only fires when the footwear
    # noun appears before the bag noun, matching the real combo-title convention;
    # this must run before Step 5's rightmost-noun scan, which would otherwise let
    # the trailing bag noun win.
    _footwear_match = _FOOTWEAR_WORD_RE.search(residual)
    _bag_match = _BAG_WORD_RE.search(residual)
    if _footwear_match and _bag_match and _footwear_match.start() < _bag_match.start():
        return NormalizationResult(garment_type="footwear", category="footwear", type_confidence="high")

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

    # ── Step 3.5: jewellery-noun precedence over apparel-fragment words ────────
    # "shorts"/"shirt"/"saree"/"top" are generic apparel-fragment keywords that
    # collide with dedicated-jewellery titles where they are a modifier of the
    # jewellery noun, not a head noun — and unlike the footwear-led-combo case,
    # the jewellery noun can land on EITHER side of the fragment word (e.g.
    # "Simrath Short Necklace Set" — noun after; "Antique Gold-Plated Temple
    # Necklace Set - Bridal Short Design K-1835" — noun before; "South Indian
    # Laxmi Jhumkas - Gold-Plated Ruby Floral Top R-2733" — "top" trailing).
    # So this can't be solved by the rightmost-noun scan alone; whenever a real
    # jewellery noun is present anywhere in the title, the fragment match is
    # dropped outright. Confirmed live (2026-07-19 jewellery-inventory-gap
    # wave): theamethyststore (1,488 "shorts" + 60 "shirt" + 9 "saree"),
    # southtemplejewellery (57 "shorts" + 1 "top"), daivik (56 "shorts" + 11
    # "saree"). Safe beyond these 3 brands too: a real shorts/shirt/saree/top
    # garment listing essentially never also contains a necklace/earring/
    # jhumka/jewellery noun in the same title.
    _APPAREL_FRAGMENT_GTYPES = {"shorts", "shirt", "saree", "top"}
    _JEWELLERY_GTYPES = {"jewellery", "necklace", "earrings", "jhumka"}
    if any(gt in _JEWELLERY_GTYPES for _, gt, _ in matches) and any(
        gt in _APPAREL_FRAGMENT_GTYPES for _, gt, _ in matches
    ):
        matches = [(pos, gt, cat) for pos, gt, cat in matches if gt not in _APPAREL_FRAGMENT_GTYPES]

    if not matches:
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
    if garment_type in {"bag", "jewellery", "necklace", "earrings", "jhumka"}:
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

    # Jewellery-noun precedence over apparel-fragment words — see Step 3.5 in
    # normalize_garment_type (same collision, e.g. store label "Short Necklaces").
    _apparel_fragment_gtypes = {"shorts", "shirt", "saree", "top"}
    _jewellery_gtypes = {"jewellery", "necklace", "earrings", "jhumka"}
    if any(gt in _jewellery_gtypes for _, gt, _ in matches) and any(
        gt in _apparel_fragment_gtypes for _, gt, _ in matches
    ):
        matches = [
            (pos, gt, cat) for pos, gt, cat in matches if gt not in _apparel_fragment_gtypes
        ]

    if not matches:
        return NormalizationResult(garment_type=None, category="unknown", type_confidence="unknown")

    _, winning_gtype, winning_cat = max(matches, key=lambda t: t[0])
    return NormalizationResult(
        garment_type=winning_gtype,
        category=winning_cat,
        type_confidence="medium",
    )
