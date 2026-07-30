from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.agents.outfit import body_type as body_type_module
from src.agents.outfit.occasions import EITHER, ETHNIC_HEAVY, ETHNIC_ONLY, get_occasion

# ── Anchor type detection — keyword sets keyed on product_type_name (lowercase) ──

ETHNIC_TOP_KEYWORDS: frozenset[str] = frozenset({
    "kurta", "kurti", "kameez", "tunic", "kaftan",
})
ETHNIC_ONE_PIECE_KEYWORDS: frozenset[str] = frozenset({
    "lehenga", "saree", "anarkali", "suit-set", "suit set", "sharara set",
    "salwar kameez", "palazzo set", "ethnic dress", "gown",
})
ETHNIC_BOTTOM_KEYWORDS: frozenset[str] = frozenset({
    "palazzo", "palazzos", "churidar", "salwar", "sharara", "pyjama", "dhoti",
    # 2026-07-25 (set-not-single follow-up): "pajama" (American spelling) is a
    # real, prevalent alternate to "pyjama" in this catalogue -- 1,386 rows
    # use it, including "kurta pajama" bare-juxtaposition listings (1,348
    # rows) that were silently invisible to every noun-counting check below.
    "pajama",
})
WESTERN_TOP_KEYWORDS: frozenset[str] = frozenset({
    "shirt", "t-shirt", "tshirt", "top", "blouse", "sweater", "sweatshirt",
    "tank top", "crop top", "polo",
    # Wave 9 (2026-07-23, gym occasion): the new activewear catalogue's own
    # product_type_name facet is the literal string "sports_bra" (underscore,
    # no space) — "sports bra" (space) also catches the freeform-name variant.
    # Deliberately NOT a bare "bra" (2,509 catalogue rows contain that
    # substring, including unrelated jewellery like "Brass Necklace").
    "sports_bra", "sports bra",
})
WESTERN_BOTTOM_KEYWORDS: frozenset[str] = frozenset({
    "trousers", "jeans", "shorts", "skirt", "jeggings",
    # 2026-07-11: "pant"/"pants" is a distinct catalogue naming convention from
    # "trousers" ("Kurta with Pant & Dupatta") — same garment class, was
    # previously unrecognised by this keyword set entirely.
    "pant", "pants",
    # Wave 9 (2026-07-23, gym occasion): "leggings"/"joggers"/"skort" are new
    # product_type_name facet values from the activewear catalogue merge with
    # no prior keyword coverage here — without these, classify_anchor/
    # classify_item resolve them to "unknown", which SLOT_ALLOWED_CLASSES
    # then hard-rejects from ever filling a "bottom" slot at all. (track_pants/
    # cargo_pants already resolve via the existing "pants" substring match.)
    "leggings", "joggers", "skort",
})
WESTERN_ONE_PIECE_KEYWORDS: frozenset[str] = frozenset({
    "dress", "jumpsuit", "playsuit", "dungarees", "co-ord",
})
OUTERWEAR_KEYWORDS: frozenset[str] = frozenset({
    "jacket", "coat", "blazer", "cardigan", "nehru jacket", "waistcoat",
    "parka", "anorak", "sherwani", "bandhgala",
})
FOOTWEAR_KEYWORDS: frozenset[str] = frozenset({
    "shoes", "sandals", "boots", "heels", "flats", "sneakers",
    "juttis", "jutti", "mojaris", "mojari", "kolhapuris", "kolhapuri",
    "wedges", "loafers", "pumps",
})
MEN_FORMALWEAR_KEYWORDS: frozenset[str] = frozenset({
    "sherwani", "bandhgala", "nehru jacket",
})

# ── Accessory sub-families (Phase B Part 1) ─────────────────────────────────
# Kept as separate small families (rather than one flat ACCESSORY_KEYWORDS set)
# so accessory_query_matches() can require a candidate to share a FAMILY with
# the slot's own query — e.g. a dupatta-seeking slot must never accept a
# handbag, and a "belt watch cap" slot must never accept a dupatta.
_ACCESSORY_DUPATTA_FAMILY: frozenset[str] = frozenset({"dupatta", "stole", "scarf"})
_ACCESSORY_BAG_FAMILY: frozenset[str] = frozenset({"bag", "handbag", "sling", "clutch", "tote"})
_ACCESSORY_JEWELLERY_FAMILY: frozenset[str] = frozenset(
    # "pendant" added 2026-07-25 (out-of-sample validation finding): a
    # "Pendant" product_type row (126 in the catalogue, all jewellery) slipped
    # past the new accessory-exclusion gate for "office outfit for men" —
    # was never in ACCESSORY_KEYWORDS at all before this.
    {"jewellery", "jewelry", "jhumka", "earrings", "necklace", "bangle", "pendant"}
)
_ACCESSORY_BELT_WATCH_FAMILY: frozenset[str] = frozenset({"belt", "watch"})
_ACCESSORY_EYEWEAR_CAP_FAMILY: frozenset[str] = frozenset({"sunglasses", "cap"})
_ACCESSORY_MENSWEAR_FORMAL_FAMILY: frozenset[str] = frozenset({"pocket square", "safa"})

_ACCESSORY_FAMILIES: tuple[frozenset[str], ...] = (
    _ACCESSORY_DUPATTA_FAMILY,
    _ACCESSORY_BAG_FAMILY,
    _ACCESSORY_JEWELLERY_FAMILY,
    _ACCESSORY_BELT_WATCH_FAMILY,
    _ACCESSORY_EYEWEAR_CAP_FAMILY,
    _ACCESSORY_MENSWEAR_FORMAL_FAMILY,
)

# Union of every family — used by classify_item() to detect "this candidate IS
# an accessory of some kind" before checking WHICH family it belongs to.
ACCESSORY_KEYWORDS: frozenset[str] = frozenset().union(*_ACCESSORY_FAMILIES)

# Sub-families that are genuinely unisex in this catalogue (sunglasses, belts,
# watches, caps) — used ONLY as a narrow opt-in fallback when a gendered
# accessory search returns nothing (see is_gender_neutral_accessory below).
_GENDER_NEUTRAL_ACCESSORY_FAMILIES: tuple[frozenset[str], ...] = (
    _ACCESSORY_EYEWEAR_CAP_FAMILY,
    _ACCESSORY_BELT_WATCH_FAMILY,
)

# Western marker words for classes classify_anchor() never flags as "western"
# (footwear/outerwear/unknown) — e.g. is_western_item("Sneakers") is False
# because classify_anchor("Sneakers") returns "footwear", not one of the three
# western_* classes. Used ONLY by ethnic-occasion coherence gates so a sangeet
# look can never accept a pair of sneakers or a denim jacket into the
# footwear/outerwear slot (see is_western_marker_item + coherence.py).
_WESTERN_MARKER_KEYWORDS: frozenset[str] = frozenset(
    {"sneaker", "sneakers", "denim", "bomber", "hoodie", "blazer", "t-shirt", "tshirt"}
)

# Small, conservative novelty/costume denylist (Phase B Part 1 quality guard).
# "cosplay"/"costume" are checked as bare substrings (no real-catalogue false
# positives found — see offline check). The instrument/object words are only
# treated as novelty when paired with "shape"/"shaped" in the same name, so a
# legitimate "V-shape Waist Jegging" is NOT rejected (it contains no object
# word), while "Luxury Piano Shape Statement Handbag" IS rejected.
_NOVELTY_GENERAL_MARKERS: frozenset[str] = frozenset({"cosplay", "costume"})
_NOVELTY_OBJECT_WORDS: frozenset[str] = frozenset(
    {
        "piano", "guitar", "violin", "football", "rhino", "puppy", "dachshund",
        "unicorn", "flamingo", "telephone",
    }
)


def _contains_word(text: str, phrase: str) -> bool:
    """Word-boundary substring match — plain `phrase in text` would let short
    accessory keywords like "bag"/"cap" false-positive inside unrelated words
    (e.g. "Paperbag Waist Pants" contains "bag"; "Capri" contains "cap").  This
    was the live-proven root cause of a pair of trousers filling an
    "accessory" slot — the slot's "bag handbag" query text-matched "paperbag".
    """
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def classify_item(product_type: str, prod_name: str = "") -> str:
    """Classify a CANDIDATE item (not just an anchor) into a slot-compatible class.

    Same classes as classify_anchor(), PLUS "accessory" for bags/belts/watches/
    dupattas/jewellery/etc.  classify_anchor() never returns "accessory" because
    in this catalogue an accessory is never used as a look ANCHOR; this sibling
    function is used for candidate-side slot-type gating in
    composer._find_best_candidate, where accessory candidates DO occur.

    Trusts the catalogue's own `product_type` facet FIRST (checked ALONE, with
    the freeform name blanked out), falling back to the combined product_type +
    name keyword scan (classify_anchor's original behaviour) only when
    product_type alone doesn't resolve to a known class.  This matters because
    many real listings are co-ord/bundle sets whose freeform NAME mentions
    other garment parts too — e.g. product_type="top" but name "... Crop Top
    WITH PALAZZO", or product_type="kurta" but name "... Kurta WITH TROUSERS &
    DUPATTA".  Scanning the combined text let those name-only mentions
    ("palazzo"/"trousers") override the authoritative product_type and put a
    top-typed item in a "bottom" slot — a live-proven variant of the same class
    of bug as the "Paperbag Waist Pants" substring collision, generalised
    beyond "bag"-in-text collisions to "other-garment-word"-in-text collisions.

    Uses word-boundary matching (see _contains_word) so "Paperbag Waist Pants"
    is never misclassified as an accessory just because "bag" is a substring.
    """
    pt = product_type.lower()
    name = prod_name.lower()

    if any(_contains_word(pt, kw) for kw in ACCESSORY_KEYWORDS):
        return "accessory"
    pt_only_class = classify_anchor(product_type, "")
    if pt_only_class != "unknown":
        return pt_only_class

    combined = pt + " " + name
    if any(_contains_word(combined, kw) for kw in ACCESSORY_KEYWORDS):
        return "accessory"
    return classify_anchor(product_type, prod_name)


# Slot name -> the set of classify_item() classes allowed to fill it.  A
# candidate whose class is not in this set is rejected before scoring —
# this is what makes it impossible for a bottom-classified item (e.g. those
# paperbag-waist trousers) to ever fill the "accessory" slot, regardless of
# what the retrieval layer's text/embedding similarity happened to surface.
# The five sets are pairwise disjoint by construction (see
# TestSlotAllowedClassesDisjoint in tests/test_outfit_package.py).
SLOT_ALLOWED_CLASSES: dict[str, frozenset[str]] = {
    "top": frozenset({"western_top", "ethnic_top"}),
    "bottom": frozenset({"western_bottom", "ethnic_bottom"}),
    "footwear": frozenset({"footwear"}),
    "outerwear": frozenset({"outerwear"}),
    "accessory": frozenset({"accessory"}),
}


# classify_item() classes that can never stand alone as "an outfit" for a
# generic "outfit/look/wear" ask (see graph.py's + eval_strict.py's
# accessory-exclusion gate, both of which import this) — a lone shoe fails
# that bar the same way a lone bag does (2026-07-30 fix: "footwear" is its
# own disjoint class from "accessory" in SLOT_ALLOWED_CLASSES above, so the
# gate's original `!= "accessory"` check never caught a standalone footwear
# item like "Ladies Triveni Kolhapuri Chappal" ranking for "haldi look").
NON_OUTFIT_ITEM_CLASSES: frozenset[str] = frozenset({"accessory", "footwear"})


def is_slot_type_allowed(slot_name: str, product_type: str, prod_name: str = "") -> bool:
    """Hard slot-type gate: reject a candidate whose classified item-type doesn't
    belong to the slot it's being considered for.

    Unknown slot names (should never occur — every SlotSpec.slot_name is one of
    the five keys in SLOT_ALLOWED_CLASSES) fall back to permissive True rather
    than silently rejecting everything.
    """
    allowed = SLOT_ALLOWED_CLASSES.get(slot_name)
    if allowed is None:
        return True
    return classify_item(product_type, prod_name) in allowed


def accessory_query_matches(query: str, product_type: str, prod_name: str) -> bool:
    """Return True if an accessory candidate's text shares a FAMILY with the
    slot query (e.g. a "dupatta ethnic dupatta" query must not accept a
    handbag, and a "belt watch cap" query must not accept a dupatta).

    Permissive (returns True) when the query doesn't recognisably target one of
    the known accessory families — avoids over-rejecting queries not covered
    by this list.  Every accessory SlotSpec.search_query in get_fill_slots()
    below does map onto at least one family (see offline check).
    """
    q = query.lower()
    combined = (product_type + " " + prod_name).lower()
    matched_families = [
        fam for fam in _ACCESSORY_FAMILIES if any(_contains_word(q, kw) for kw in fam)
    ]
    if not matched_families:
        return True
    return any(any(_contains_word(combined, kw) for kw in fam) for fam in matched_families)


def split_accessory_query_by_family(query: str) -> list[str]:
    """Split a multi-family accessory SlotSpec.search_query into one retrieval
    sub-query per accessory family it spans, each retaining that family's own
    matched keyword(s) plus every non-family (register/occasion/gender) word
    from the original query.

    Root-cause fix (2026-07-19, bridal-look jewellery gap): a single accessory
    slot's search_query frequently mixes multiple families in one string —
    e.g. the bridal ethnic_one_piece slot's "dupatta jewellery clutch ethnic
    accessory" spans the DUPATTA + BAG + JEWELLERY families at once.
    Retrieving that combined text as ONE query lets whichever family's
    vocabulary has the strongest lexical/semantic overlap with the query text
    dominate the retrieval window so completely that the other families'
    candidates never enter even a very wide pool — live-proven: the combined
    bridal accessory query returned ZERO jewellery items in the top 500 dense
    hits and top 300 sparse hits against the real unified catalogue, despite
    18,796 real jewellery rows existing and the literal word "jewellery"
    being IN the query (dupatta/clutch listings very often literally repeat
    "ethnic"/"embroidered"/"accessory" in their own catalogue text, so they
    win on lexical/embedding overlap alone). This is a retrieval-POOL defect,
    not a downstream gating defect — accessory_query_matches() already
    permits every one of these families through the slot-type gate; they
    just never arrive as candidates in the first place.

    Splitting into one sub-query per matched family (verified offline: an
    isolated "jewellery ethnic accessory festive embroidered" query surfaces
    18/40 genuine jewellery hits where the combined query surfaced 0/40) and
    merging the resulting pools before scoring gives every family a genuine,
    undiluted retrieval shot. The existing scoring formula in
    _score_candidates (colour/fabric/body-type deltas, occasion register)
    still picks the actual winner — jewellery is never forced to win, only
    given a fair chance to compete.

    Returns an empty list (signalling "no split needed — use the query
    text as-is") when the query matches 0 or 1 accessory family — every
    existing single-family accessory SlotSpec (e.g. "dupatta ethnic dupatta",
    "pocket square safa ethnic accessory") is therefore fully unaffected by
    this function; only genuinely multi-family queries are split.
    """
    q = query.lower()
    matched: list[tuple[frozenset[str], list[str]]] = []
    residual = q
    for family in _ACCESSORY_FAMILIES:
        words_present = [kw for kw in family if _contains_word(q, kw)]
        if words_present:
            matched.append((family, words_present))
            for w in words_present:
                residual = re.sub(rf"\b{re.escape(w)}\b", " ", residual)

    if len(matched) <= 1:
        return []

    residual = re.sub(r"\s+", " ", residual).strip()
    return [
        " ".join(words) + (f" {residual}" if residual else "") for _, words in matched
    ]


def is_gender_neutral_accessory(product_type: str, prod_name: str = "") -> bool:
    """Return True for accessory sub-types that are genuinely unisex in this
    catalogue (sunglasses, belts, watches, caps).

    Used ONLY as a narrow, opt-in fallback in composer._find_best_candidate
    when a slot's gendered search returns zero results — never for garments
    (tops, bottoms, footwear, outerwear), where gender ambiguity is a leak to
    close, not a feature to exploit.
    """
    combined = (product_type + " " + prod_name).lower()
    return any(
        any(_contains_word(combined, kw) for kw in fam)
        for fam in _GENDER_NEUTRAL_ACCESSORY_FAMILIES
    )


# A bag-family word alongside an object/instrument/animal word is the second
# (in addition to "shape"/"shaped") signal that a candidate is a novelty item —
# catches real catalogue rows like "Designer Dachshund Crossbody Bag" that
# don't happen to use the literal word "shape".  Checked against the real
# catalogue: "football" (49 rows, all "Football Shoes"/"Football Shorts") and
# "flamingo" (colour-name rows, "Flamingo Pink ... Shirt") never co-occur with
# any of these bag words, so this AND-combination has zero false positives
# there (see offline audit).
_NOVELTY_BAG_WORDS: frozenset[str] = frozenset(
    {"bag", "handbag", "clutch", "crossbody", "tote", "purse"}
)


def is_novelty_item(prod_name: str) -> bool:
    """Reject conservative novelty/costume items that should never fill a real
    outfit slot (e.g. "Luxury Piano Shape Statement Handbag", "Designer
    Dachshund Crossbody Bag").

    Small, deliberately conservative denylist — false negatives (a novelty item
    that slips through) are safer than false positives that reject real
    garments.  Checked against the real catalogue: no "V-shape Waist Jegging"-
    style false positive (no object word present), no "Novelty Town" brand-name
    false positive (that brand word is not in this denylist at all), and no
    "Football Shoes"/"Flamingo Pink Shirt" false positive (object word present
    but no bag-family word alongside it).
    """
    name = (prod_name or "").lower()
    if any(_contains_word(name, kw) for kw in _NOVELTY_GENERAL_MARKERS):
        return True
    has_object_word = any(_contains_word(name, kw) for kw in _NOVELTY_OBJECT_WORDS)
    if has_object_word and (
        "shape" in name
        or "shaped" in name
        or any(_contains_word(name, kw) for kw in _NOVELTY_BAG_WORDS)
    ):
        return True
    return False


# is_kids_item / _KIDS_MARKER_RE promoted to src.catalogue.cleaning (2026-07-12)
# so the retrieval layer (hybrid_search.py) and plain-search node (graph.py) can
# apply it as an unconditional hard exclusion without importing from the agents
# layer. Import from there — see that module's docstring for the S5 fix history.

# Phase B: an adult, correctly-gendered item can still be casual-register and
# leak into a formal/office look — _default_bottom_query() drops "jeans"/
# "skirt" from the QUERY text for formality>=3 occasions, but that only
# shapes retrieval ranking; it does not stop a casual item from being
# retrieved via other query terms (register tokens, anchor colour, etc.) and
# then accepted as the best-scoring candidate. Live-proven: "black top for
# office for women" -> Style this -> bottom slot filled with "ONLY Women Blue
# Solid Denim Mini Skirts" (adult item, correctly gendered, NOT caught by
# is_kids_item). Deliberately narrow, word-boundary keyword list — conservative
# false-negatives (a casual item slipping through on a word not in this list)
# are safer than false positives that reject legitimate formal items. "denim"
# is checked standalone (not just "denim skirt"/"denim jeans") per design: a
# hypothetical "denim-look tailored trouser" is still rejected — keeping the
# rule simple beats trying to carve out fabric-look exceptions.
_CASUAL_MARKER_RE = re.compile(
    r"\b(denim|jeans|mini\s+skirts?|shorts?|joggers?|cargo|distressed|ripped)\b",
    re.IGNORECASE,
)


def is_casual_marker_item(prod_name: str) -> bool:
    """Return True if `prod_name` carries a casual/denim-register marker word.

    Checked as an ADDITIONAL gate in composer._find_best_candidate for
    formal occasions (occasion.formality >= 3), alongside (never instead of)
    the gender/slot-type/novelty/kids gates — see module docstring on
    _CASUAL_MARKER_RE for the live regression this closes.
    """
    return bool(_CASUAL_MARKER_RE.search(prod_name or ""))


# Rugged/athletic footwear register — live-proven miss (sweep 2026-07-10,
# relevance-adjacent): a sangeet "his look" footwear slot filled with ₹759
# combat boots. The generic casual-marker gate above covers garments (denim/
# cargo/joggers) but had no footwear vocabulary at all. Plain "boots" is
# included deliberately: for Indian festive/formal occasions the appropriate
# menswear registers are oxfords/derbies/loafers/monks/mojaris/juttis — even a
# dress Chelsea boot is an edge case not worth the combat-boot false accepts.
_RUGGED_FOOTWEAR_RE = re.compile(
    r"\b(boots?|sneakers?|trainers?|running|walking|sports?|training|trekking|hiking|"
    r"football|badminton|gym|flip[\s-]?flops?|slippers?|sliders?|crocs?|clogs?)\b",
    re.IGNORECASE,
)


def is_rugged_footwear_item(prod_name: str) -> bool:
    """True if `prod_name` reads as rugged/athletic/at-home footwear — never
    acceptable in a formality >= 3 look's footwear slot."""
    return bool(_RUGGED_FOOTWEAR_RE.search(prod_name or ""))


# Wave 9 (2026-07-23, gym occasion): the INVERSE selection to
# _RUGGED_FOOTWEAR_RE above — a gym look's footwear slot must ONLY accept a
# genuine athletic/sport shoe, never a jutti/mojari/formal oxford/wedding
# heel as a fallback. Deliberately excludes flip-flops/slippers/sliders/
# crocs/clogs (at-home, not gym-appropriate) that _RUGGED_FOOTWEAR_RE lumps
# in with genuine athletic wear. Catalogue audit against the real unified
# catalogue (data/processed/unified/catalogue.parquet): 0 women's rows and 20
# men's rows (all store=flipkart, "Running Shoes"/"Sneakers"/"Training & Gym
# Shoes For Men") match this pattern — a women's gym look's footwear slot is
# therefore expected to go through honest suppression (composer.
# _suppression_reason) far more often than not; that is the correct honest
# behaviour for genuinely thin inventory, never a bug to paper over with a
# non-athletic substitute.
_ATHLETIC_FOOTWEAR_RE = re.compile(
    r"\b(sneakers?|trainers?|running|sports?|training|gym|athletic|workout)\b",
    re.IGNORECASE,
)


def is_athletic_footwear_item(prod_name: str) -> bool:
    """True if `prod_name` reads as a genuine athletic/sport shoe — the ONLY
    footwear register a gym look's footwear slot accepts (see coherence.py's
    athletic-register gate)."""
    return bool(_ATHLETIC_FOOTWEAR_RE.search(prod_name or ""))


# Phase B (product gap 2): a multi-piece SET listing ("Anarkali Sharara Set",
# "Kurta Set with Dupatta", a "Co-Ord Set") is a WHOLE OUTFIT, not a single
# garment — it must never fill a single complement slot (bottom/top/
# outerwear/accessory/etc.), even though it may still be used as a look's own
# SEED/anchor (a kurta set as a look's hero item is fine — compose_outfit's
# seed resolution never calls composer._find_best_candidate, so this gate
# never touches the seed).
#
# Live-proven root cause: product_type_name="sharara" alone (the catalogue's
# OWN facet) classify_item()'s pt-first shortcut resolves straight to
# "ethnic_bottom" WITHOUT ever inspecting the freeform name — so a
# "Quirky Floral Printed Cotton Anarkali Sharara Set" (a 2-piece anarkali top
# + sharara bottom SET) slipped past the bottom slot's hard slot-type gate
# and filled an "office look" bottom slot.
#
# Signals, any sufficient (all verified against the real unified catalogue's
# own "Set"-in-name product rows, and against every "set-not-single" hand
# label in eval/fixtures/strict_gold_labels.yaml — see offline check):
#   1. product_type_name is one of the catalogue's own dedicated set-type
#      values: "Suits"/"Suit Set(s)" (2-3 piece ethnic suit sets: kurta +
#      bottom [+ dupatta]), "coord"/"Co-Ord" (western co-ord sets),
#      "Sets"/"Track-Suit" (misc matching sets e.g. "Winter Set", "Cord Set").
#   2. Freeform name contains "with" AND mentions >= 2 DISTINCT garment nouns
#      (e.g. "Kaftan Kurta with Abstract Patchwork Palazzo") — this
#      catalogue's dominant multi-piece convention, and very often never uses
#      the literal word "set"/"sets" at all (2026-07-11 fix).
#   3. Freeform name contains the word "set(s)" (and is NOT a "Set of N"
#      same-item PACK — e.g. "TAG 7 Women Set of 2 ... Palazzos", a 2-pack of
#      the SAME garment, correctly excluded) AND mentions >= 1 garment noun.
#      2026-07-25 fix: originally required >=2 distinct nouns here too (the
#      same bar as "with"), but that was over-conservative and caused 16
#      real strict-gold misses — this catalogue's single most common
#      multi-piece convention is a bare "<Garment> Set" suffix that never
#      names the second piece in the product NAME at all (e.g. "Orange
#      Floral Printed Cotton Straight Kurta Set", "Plus Size Pink Printed
#      Cotton Straight Kurta Set") — the word "Set" itself is already the
#      strong signal once the "Set of N" pack pattern is excluded.
#   4. Freeform name contains an explicit "N-Piece"/"N Piece" count (e.g.
#      "Sea Green Winter Ethnic 3-Piece Set") — sufficient on its own, since
#      some of these name NO garment noun at all in the truncated name.
#   5. Freeform name contains the literal word "co-ord(s)" — inherently a
#      2-piece-by-definition term, sufficient on its own. Needed because the
#      product_type_name facet often captures only the hero piece (e.g.
#      "URBANIC...Hooded Co-Ords Set" carries product_type_name="top", never
#      matching signal 1's "coord" facet check).
#   6. A recognised TOP noun immediately followed by a recognised BOTTOM/
#      companion noun with NO connector word at all (e.g. "Green Kurta
#      Pajama", "kurta pajama"/"kurta pyjama" alone account for 2,459
#      catalogue rows) — this catalogue's third naming convention. Scoped
#      NARROWLY to TOP-then-BOTTOM specifically (not the full distinct-noun
#      union signals 2/3 use) to avoid false-firing on "Anarkali Kurta" — a
#      ONE_PIECE+TOP synonym pair naming ONE single garment, not two.
_SET_PRODUCT_TYPES: frozenset[str] = frozenset({
    "suits", "suit set", "suit sets", "coord", "co-ord", "sets", "track-suit",
})

_SET_WORD_RE = re.compile(r"\bsets?\b", re.IGNORECASE)
_SET_OF_N_RE = re.compile(r"\bset\s+of\s+\d+\b", re.IGNORECASE)
_N_PIECE_RE = re.compile(r"\b\d+[\s-]?piece\b", re.IGNORECASE)
_COORD_WORD_RE = re.compile(r"\bco-?ords?\b", re.IGNORECASE)
_TOP_THEN_BOTTOM_RE = re.compile(
    r"\b(?:kurta|kurti|kameez|tunic|shirt|top|blouse)\s+"
    r"(?:pajama|pyjama|palazzos?|churidar|salwar|sharara|dhoti|trousers?|pants?|dupatta)\b",
    re.IGNORECASE,
)

# Garment-noun vocabulary reused from the anchor-classification keyword sets
# above — used ONLY to count how many distinct garment types a name mentions.
_SET_GARMENT_NOUN_KEYWORDS: frozenset[str] = (
    ETHNIC_TOP_KEYWORDS
    | ETHNIC_ONE_PIECE_KEYWORDS
    | ETHNIC_BOTTOM_KEYWORDS
    | WESTERN_TOP_KEYWORDS
    | WESTERN_BOTTOM_KEYWORDS
    | WESTERN_ONE_PIECE_KEYWORDS
    | OUTERWEAR_KEYWORDS
    | frozenset({"dupatta"})
)

_WITH_RE = re.compile(r"\bwith\b", re.IGNORECASE)


def is_multi_piece_set(product_type: str, prod_name: str) -> bool:
    """Return True if this item is a multi-piece SET listing (a whole outfit)
    rather than a single garment. See the module comment above
    _SET_PRODUCT_TYPES for the six signals checked, any one sufficient.
    """
    pt = (product_type or "").lower().strip()
    if pt in _SET_PRODUCT_TYPES:
        return True
    name = (prod_name or "").lower()
    if _N_PIECE_RE.search(name) or _COORD_WORD_RE.search(name) or _TOP_THEN_BOTTOM_RE.search(name):
        return True
    has_set = _SET_WORD_RE.search(name) and not _SET_OF_N_RE.search(name)
    has_with = _WITH_RE.search(name)
    if not (has_set or has_with):
        return False
    distinct_nouns = {kw for kw in _SET_GARMENT_NOUN_KEYWORDS if _contains_word(name, kw)}
    if has_set:
        return len(distinct_nouns) >= 1
    return len(distinct_nouns) >= 2


# 2026-07-25 (strict-eval attribute-contradiction follow-up, now the largest
# CODE-FIXABLE miss bucket at 22): no deterministic fit/silhouette-matching
# gate existed anywhere in the plain-search path before this — retrieval
# relies entirely on embedding similarity, which frequently ranks a "Slim
# Fit" item highly for a "straight fit" query since the two phrases are
# semantically close in embedding space despite being product-listing
# OPPOSITES in this catalogue's own marketing vocabulary.
#
# Two representations, chosen per group based on whether the group's members
# are genuinely N-way mutually exclusive or actually two *camps* of near-
# synonyms opposing each other:
#
# - Flat groups (_ATTRIBUTE_CONTRADICTION_FLAT_GROUPS): any two DIFFERENT
#   members oppose each other. Correct only when every member is genuinely
#   distinct from every other (RISE, BREASTED, NECKLINE) — a garment has
#   exactly one neckline shape, one rise, one breasted style.
# - Camp-pair groups (_ATTRIBUTE_CONTRADICTION_CAMP_PAIRS): two frozensets
#   per pair; only CROSS-camp words oppose, same-camp words are compatible
#   synonyms. Required for FIT and SILHOUETTE, where a flat group produced a
#   real false positive: "anarkali" and "a-line" were both dumped into one
#   "silhouette" group as if mutually exclusive, but an anarkali kurta IS
#   a-line by definition (verified against real catalogue desc text, e.g.
#   article 7797797454046 "...heritage anarkali style with a graceful
#   flared silhouette..." — anarkali literally means flared/a-line here).
#   The genuine opposition is flared-family vs fitted/straight-family.
#
# Grounded in the ACTUAL contradiction pairs found across every
# attribute-contradiction hand label in strict_gold_labels.yaml (not
# invented) — verified real hit-rate per word against
# data/processed/unified/catalogue.parquet before inclusion (all >=12 rows,
# most in the hundreds-to-thousands).
_ATTRIBUTE_CONTRADICTION_FLAT_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"high waisted", "high-waisted", "high rise", "mid rise", "low rise"}),
    frozenset({"single breasted", "double breasted"}),
    frozenset({
        "v-neck", "halter neck", "boat neck", "round neck", "square neck",
        "mandarin collar", "scoop neck", "sweetheart neck",
    }),
)

_ATTRIBUTE_CONTRADICTION_CAMP_PAIRS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    # FIT tightness: "slim"/"skinny" (tight camp) vs the mutually-compatible
    # "straight"/"regular"/"relaxed"/"oversized"/"loose"/"tailored" camp.
    (
        frozenset({"slim fit", "skinny fit"}),
        frozenset({
            "straight fit", "regular fit", "relaxed fit", "tailored fit",
            "oversized", "loose fit",
        }),
    ),
    # SILHOUETTE flare: "a-line"/"anarkali"/"fit and flare" are mutually
    # COMPATIBLE (same flared family — anarkali kurtas are a-line by
    # definition), opposing the straight/regular camp. "regular fit" belongs
    # here too (distinct dimension from FIT-tightness above) — real hand
    # label evidence: "'Regular Fit' contradicts 'a-line'" (a kurta's
    # overall cut is either flared/a-line or straight/regular, not both).
    (
        frozenset({"a-line", "anarkali", "fit and flare", "fit & flare"}),
        frozenset({"straight cut", "straight fit", "regular fit"}),
    ),
    # SILHOUETTE fitted: "bodycon" is its OWN pole (2026-07-25, out-of-sample
    # validation finding), not folded into the straight/regular camp above —
    # an item with the structured facet line "Silhouette: Straight kurta"
    # surfaced for a "bodycon kurta" query in the held-out set. Bodycon
    # (body-hugging throughout) and straight (hangs straight, unflared but
    # NOT tight) are genuinely distinct silhouettes for a kurta, even though
    # both oppose the SAME flared camp above — treating them as compatible
    # with each other was too coarse. Opposes both other poles.
    #
    # "silhouette: straight" (the structured facet-line phrasing, not bare
    # "straight kurta" prose) is deliberately the ONLY straight-family
    # trigger added here — a bare "straight kurta"/"straight kurti" phrase
    # appears in 2252/14184 (16%) of catalogue kurta rows (the default,
    # most-common silhouette description for kurtas generally), which would
    # have been a badly over-broad exclusion for a single niche query
    # pattern; the structured "Silhouette: Straight" facet line is a much
    # narrower, catalogue-verified signal (12/14184 rows).
    (
        frozenset({"bodycon"}),
        frozenset({"a-line", "anarkali", "fit and flare", "fit & flare"}),
    ),
    (
        frozenset({"bodycon"}),
        frozenset({"straight cut", "straight fit", "regular fit", "silhouette: straight"}),
    ),
)


def is_attribute_contradiction(query_text: str, item_name: str, item_desc: str) -> bool:
    """Return True if the CANDIDATE ITEM's own name/desc explicitly states a
    fit/rise/breasted-style/silhouette/neckline word that OPPOSES a word the
    QUERY itself explicitly stated — e.g. query "straight fit kurta" +
    item name "...Slim Fit Kurta".

    Deliberately conservative in both directions:
      - Only fires when the QUERY names a tracked word at all (a query with
        no stated fit/silhouette preference can never trigger this).
      - If the item's own text ALSO contains the query's exact word anywhere
        (even alongside an opposing word — e.g. boilerplate mentioning
        several fit options), that's treated as a genuine match, never a
        contradiction — same "explicit confirmation wins" precedent as the
        query's own hand-labeling rubric.
      - "Unstated, not contradicted" is never penalised here (mirrors the
        hand-labeling rubric's point 2) — an item with NO word from a group
        at all is never flagged for that group.
      - Near-synonyms (e.g. "anarkali" vs "a-line") never oppose each other
        — see _ATTRIBUTE_CONTRADICTION_CAMP_PAIRS.
    """
    query_lower = (query_text or "").lower()
    text = f"{item_name or ''} {item_desc or ''}".lower()

    for group in _ATTRIBUTE_CONTRADICTION_FLAT_GROUPS:
        stated = next((w for w in group if _contains_word(query_lower, w)), None)
        if stated is None:
            continue
        if _contains_word(text, stated):
            continue  # item explicitly confirms the query's own word — never a contradiction
        if any(w != stated and _contains_word(text, w) for w in group):
            return True

    for camp_a, camp_b in _ATTRIBUTE_CONTRADICTION_CAMP_PAIRS:
        for stated_camp, opposing_camp in ((camp_a, camp_b), (camp_b, camp_a)):
            stated = next((w for w in stated_camp if _contains_word(query_lower, w)), None)
            if stated is None:
                continue
            if any(_contains_word(text, w) for w in stated_camp):
                continue  # item confirms the query's own camp — never a contradiction
            if any(_contains_word(text, w) for w in opposing_camp):
                return True

    return False


def is_western_marker_item(product_type: str, prod_name: str = "") -> bool:
    """Return True if a footwear/outerwear/unknown-class item carries an
    explicit WESTERN marker word (sneaker, denim, bomber, hoodie, blazer,
    t-shirt).  Layered ON TOP of is_western_item (which only covers
    western_top/bottom/one_piece) so an ethnic-only occasion (e.g. sangeet)
    can never accept a pair of sneakers or a denim jacket via the
    footwear/outerwear slot — used only by coherence.py's ethnic gates.
    """
    combined = (product_type + " " + prod_name).lower()
    return any(_contains_word(combined, kw) for kw in _WESTERN_MARKER_KEYWORDS)


def resolve_look_gender(
    *,
    intent_gender: str | None,
    session_gender: str | None,
    catalogue_df: pd.DataFrame,
    anchor_id: str | None,
    brand_gender_default: str,
) -> str:
    """Resolve which gender ("men" | "women") to steer a look composition with.

    Precedence (first concrete "men"/"women" signal wins):
      1. intent_gender — explicit gender parsed from the user's own text this turn.
      2. session_gender — gender context carried over from prior turns in the
         same conversation (e.g. a previous "men's shirts" search set filters).
      3. The anchor item's own catalogue `gender` column, when it resolves to
         "men" or "women" — critical for the image-upload owned-anchor path: a
         photo of a men's garment must never silently compose a women's-default
         look just because the brand's configured default happens to be
         "women". Shared by api/routes/image_style.py and graph.py's
         outfit_node so both paths resolve gender identically.
      4. brand_gender_default (config-level fallback; "mixed"/anything else
         coerces to "women" as the least-committal default — never guessed
         per-item).

    Never returns anything other than "men"/"women" — composition always needs
    a concrete slice; "unknown" is never returned here (see gender_allowed for
    how per-item "unknown" rows are excluded downstream).
    """
    if intent_gender in ("men", "women"):
        return intent_gender
    if session_gender in ("men", "women"):
        return session_gender
    if anchor_id is not None and "gender" in catalogue_df.columns and "article_id" in catalogue_df.columns:
        match = catalogue_df.loc[catalogue_df["article_id"] == anchor_id, "gender"]
        if not match.empty and match.iloc[0] is not None:
            anchor_gender = str(match.iloc[0]).lower()
            if anchor_gender in ("men", "women"):
                return anchor_gender
    resolved_default = brand_gender_default or "women"
    return "women" if resolved_default not in ("men", "women") else resolved_default

# Occasions where footwear is required (formality >= 3, ethnic)
# Wave 8: diwali/navratri/karva_chauth/eid added (all formality >= 3, ethnic
# events warrant required footwear). raksha_bandhan deliberately excluded —
# formality 2/EITHER, casual-festive register, footwear stays optional like
# casual/smart_casual/party_evening.
_FORMAL_ETHNIC_OCCASIONS: frozenset[str] = frozenset({
    "sangeet", "haldi", "mehendi", "festive_puja", "wedding_guest",
    "traditional_ethnic", "reception", "engagement",
    "diwali", "navratri", "karva_chauth", "eid",
})

# Women-only ethnic categories — hard reject for men's looks regardless of gender field
WOMEN_ONLY_ETHNIC_KEYWORDS: frozenset[str] = frozenset({
    "dupatta", "saree", "lehenga",
})

# Fabric/embellishment keywords for haldi/mehendi vs sangeet/reception scoring
SANGEET_EMBELLISHMENT_KEYWORDS: frozenset[str] = frozenset({
    "embroidered", "embroidery", "sequin", "zari", "embellished",
    "heavy work", "bridal", "mirror work", "thread work", "beaded",
    "resham", "gota", "kundan",
})
HALDI_LIGHTWEIGHT_KEYWORDS: frozenset[str] = frozenset({
    "cotton", "floral", "tie-dye", "georgette", "chiffon", "printed",
    "casual", "lightweight", "summer", "marigold", "yellow", "daisy",
})

# formality_softener override values (the sibling intent-parser field a
# different fix surfaces from queries like "something comfortable for
# sangeet dancing" or "not too flashy"). Reuses SANGEET_EMBELLISHMENT_KEYWORDS/
# HALDI_LIGHTWEIGHT_KEYWORDS verbatim rather than duplicating them — the
# override's "embellished vs plain/lightweight" text signal is semantically
# identical to sangeet/haldi's existing keyword scan, only the SIGN differs.
FORMALITY_SOFTENER_VALUES: frozenset[str] = frozenset({"minimalist", "comfortable"})


def classify_anchor(product_type: str, prod_name: str = "") -> str:
    """Return anchor class: ethnic_top | ethnic_one_piece | ethnic_bottom |
    western_top | western_bottom | western_one_piece | outerwear | footwear | unknown."""
    pt = product_type.lower()
    name = prod_name.lower()
    combined = pt + " " + name

    if any(kw in combined for kw in ETHNIC_ONE_PIECE_KEYWORDS):
        return "ethnic_one_piece"
    if any(kw in combined for kw in ETHNIC_TOP_KEYWORDS):
        return "ethnic_top"
    if any(kw in combined for kw in ETHNIC_BOTTOM_KEYWORDS):
        return "ethnic_bottom"
    if any(kw in combined for kw in MEN_FORMALWEAR_KEYWORDS):
        return "men_formalwear"
    if any(kw in combined for kw in OUTERWEAR_KEYWORDS):
        return "outerwear"
    if any(kw in combined for kw in FOOTWEAR_KEYWORDS):
        return "footwear"
    if any(kw in combined for kw in WESTERN_ONE_PIECE_KEYWORDS):
        return "western_one_piece"
    if any(kw in combined for kw in WESTERN_BOTTOM_KEYWORDS):
        return "western_bottom"
    if any(kw in combined for kw in WESTERN_TOP_KEYWORDS):
        return "western_top"
    return "unknown"


def is_ethnic_item(product_type: str, prod_name: str = "") -> bool:
    """Return True if item is ethnic (kurta, lehenga, saree, etc.)."""
    anchor_class = classify_anchor(product_type, prod_name)
    return anchor_class in ("ethnic_top", "ethnic_one_piece", "ethnic_bottom", "men_formalwear")


def is_western_item(product_type: str, prod_name: str = "") -> bool:
    anchor_class = classify_anchor(product_type, prod_name)
    return anchor_class in ("western_top", "western_bottom", "western_one_piece")


@dataclass
class SlotSpec:
    """Definition of one complement slot to fill."""

    slot_name: str           # e.g. "bottom", "accessory", "footwear"
    search_query: str        # query terms to find candidates
    required: bool = True    # if True, empty slot is a hard failure; if False, optional


def gender_allowed(item_gender: str, look_gender: str) -> bool:
    """Return True if item gender is compatible with look gender.

    "unknown" is excluded from all gendered (men/women) looks — never guessed in.
    "unisex" look accepts everything.
    """
    ig = (item_gender or "unknown").lower()
    lg = look_gender.lower()
    if lg in ("men", "women"):
        return ig == lg
    return True  # unisex


def _occasion_register_tokens(occasion_slug: str) -> str:
    """Return extra register tokens appended to every slot's search_query so
    retrieval favours occasion-appropriate garments — e.g. an office bottom
    slot should surface tailored trousers, not a denim skirt.

    haldi/mehendi keep their own dedicated registers rather than the generic
    ethnic-festive one, since they already have a dedicated lightweight/floral
    colour+fabric bias (colour_score, fabric_score_delta) that would conflict
    with "embroidered" (haldi/mehendi favour light, undone-up looks). reception
    gets its own embellished-evening register, mirroring sangeet's bias.

    Wave 8: diwali/navratri/karva_chauth/raksha_bandhan/eid each get a
    dedicated register mirroring their own colour_score palette override
    above, so retrieval query text and colour scoring stay aligned (same
    pattern as haldi/mehendi/reception).

    Wave 9: gym gets a dedicated "activewear athletic gym sport" register —
    the generic EITHER/formality<3 fallback below would just return "casual",
    which is too weak a signal to steer retrieval away from ordinary
    lounge/casual wear toward genuine activewear.
    """
    if occasion_slug == "haldi":
        return "cotton floral"
    if occasion_slug == "mehendi":
        return "green floral festive"
    if occasion_slug == "reception":
        return "embellished formal evening"
    if occasion_slug == "engagement":
        return "elegant festive"
    if occasion_slug == "diwali":
        return "festive glam gold embellished"
    if occasion_slug == "navratri":
        return "chaniya choli bright colourful dance"
    if occasion_slug == "karva_chauth":
        return "red traditional bridal ethnic"
    if occasion_slug == "raksha_bandhan":
        return "casual festive light"
    if occasion_slug == "eid":
        return "pastel elegant festive"
    if occasion_slug == "gym":
        return "activewear athletic gym sport"
    occ = get_occasion(occasion_slug)
    if occ.ethnic_lean in (ETHNIC_HEAVY, ETHNIC_ONLY):
        return "festive embroidered"
    if occ.formality >= 3:
        return "formal tailored"
    return "casual"


def _append_register(slots: list[SlotSpec], occasion_slug: str) -> list[SlotSpec]:
    """Append the occasion's register tokens to every slot's search_query."""
    register = _occasion_register_tokens(occasion_slug)
    return [SlotSpec(s.slot_name, f"{s.search_query} {register}", s.required) for s in slots]


def _default_bottom_query(occasion_slug: str) -> str:
    """Return the base "bottom" slot query for a western top/outerwear anchor.

    Register-token appending alone ("... formal tailored") isn't enough to keep
    a denim/casual skirt out of a formal look — the literal words "jeans"/
    "skirt" are still IN the query text, so BM25/dense retrieval still surfaces
    them strongly.  For formality>=3, non-ethnic occasions (office, party_
    evening) this drops "jeans"/"skirt" from the query entirely so retrieval is
    steered toward tailored trousers, matching "office bottom must retrieve
    trousers, not a denim skirt".

    Phase B pool-composition fix: "trousers" alone (plus the register's
    generic "formal tailored") is a weak western-only signal — in a larger/
    differently-composed catalogue than this repo's own offline unified
    index, "formal"/"tailored" score close to embroidered ETHNIC formal wear
    too (a "Suit Set"/"Sharara Set" listing's own description text uses
    "formal"/"tailored" register words just as often), which can let an
    ethnic-heavy top-40 retrieval window dominate a western-register bottom
    slot even before any gate runs.  "pants"/"western" add unambiguous
    western-vocabulary lexical weight (BM25) without removing anything.

    Wave 9 gym override: "trousers jeans skirt" (the generic casual fallback
    below) is actively wrong for a gym look's bottom slot — none of those
    three are gym-appropriate, and "jeans"/"skirt" would pull the retrieval
    window toward exactly the wrong register. Returns explicit activewear
    bottom vocabulary instead, mirroring the office/party_evening branch's
    same "replace, don't just append" approach above.
    """
    if occasion_slug == "gym":
        return "leggings joggers track pants cargo pants shorts activewear"
    occ = get_occasion(occasion_slug)
    if occ.ethnic_lean == EITHER and occ.formality >= 3:
        return "trousers pants tailored western formal office wear"
    return "trousers jeans skirt"


def _default_footwear_query(occasion_slug: str, is_men: bool) -> str:
    """Return the base "footwear" slot query for a western top/bottom anchor.

    Wave 9 gym override: the generic "sneakers flats heels casual shoes"
    query text would help retrieval surface exactly the non-athletic footwear
    (flats/heels/loafers) that coherence.py's athletic-register gate then
    correctly rejects — this doesn't relax that gate, it only improves the
    odds of a genuine athletic shoe reaching the candidate pool in the first
    place when one exists (see is_athletic_footwear_item's catalogue-audit
    docstring for how thin that inventory actually is, especially for
    women).
    """
    if occasion_slug == "gym":
        return "sneakers sports shoes running shoes training shoes gym shoes trainers"
    return "sneakers casual shoes loafers men" if is_men else "sneakers flats heels casual shoes women"


def get_fill_slots(
    anchor_class: str,
    gender: str,
    occasion_slug: str,
    body_type: str | None = None,
    body_modifiers: list[str] | None = None,
) -> list[SlotSpec]:
    """Return ordered list of SlotSpecs to fill for a given anchor + gender + occasion.

    Gender: "men" | "women" | "unisex" (treated as women for ethnic, men for men's brands).

    Thin wrapper around _get_fill_slots_base(): appends occasion-register
    tokens (see _occasion_register_tokens) to every slot's search_query so
    retrieval is occasion-aware (formal tailored / festive embroidered /
    casual), without touching the base per-anchor-class slot definitions.

    P3: when body_type/body_modifiers are known, ALSO appends the body type's
    query-augmentation tokens (src.agents.outfit.body_type.query_tokens) —
    mirrors _occasion_register_tokens exactly. No-op (empty string appended)
    when body_type is None and body_modifiers is empty, so this is fully
    backward compatible with every existing call site.
    """
    slots = _get_fill_slots_base(anchor_class, gender, occasion_slug)
    slots = _append_register(slots, occasion_slug)
    return _append_body_type_tokens(slots, body_type, body_modifiers)


def _append_body_type_tokens(
    slots: list[SlotSpec], body_type: str | None, body_modifiers: list[str] | None
) -> list[SlotSpec]:
    """Append body-type query-augmentation tokens to every slot's search_query."""
    tokens = body_type_module.query_tokens(body_type, body_modifiers)
    if not tokens:
        return slots
    return [SlotSpec(s.slot_name, f"{s.search_query} {tokens}", s.required) for s in slots]


def _get_fill_slots_base(anchor_class: str, gender: str, occasion_slug: str) -> list[SlotSpec]:
    """Original per-anchor-class slot definitions (pre occasion-register tokens)."""
    g = gender.lower()
    is_men = g == "men"

    if anchor_class == "ethnic_top":
        if is_men:
            return [
                SlotSpec("bottom", "churidar pyjama dhoti ethnic bottom", required=True),
                SlotSpec("outerwear", "nehru jacket waistcoat ethnic waistcoat", required=False),
                SlotSpec(
                    "footwear", "mojaris juttis kolhapuris ethnic footwear",
                    required=occasion_slug in _FORMAL_ETHNIC_OCCASIONS,
                ),
            ]
        else:
            return [
                SlotSpec("bottom", "palazzo churidar salwar sharara ethnic bottom", required=True),
                SlotSpec("accessory", "dupatta ethnic dupatta", required=True),
                SlotSpec(
                    "footwear", "juttis heels wedges ethnic footwear",
                    required=occasion_slug in _FORMAL_ETHNIC_OCCASIONS,
                ),
            ]

    if anchor_class == "ethnic_one_piece":
        # lehenga / saree / anarkali / suit-set — never top/bottom
        return [
            SlotSpec("accessory", "dupatta jewellery clutch ethnic accessory", required=True),
            SlotSpec("footwear", "heels juttis ethnic footwear", required=True),
        ]

    if anchor_class == "men_formalwear":
        # sherwani / bandhgala
        return [
            SlotSpec("bottom", "churidar pyjama ethnic bottom", required=True),
            SlotSpec("footwear", "mojaris juttis ethnic footwear", required=True),
            SlotSpec("accessory", "pocket square safa ethnic accessory", required=False),
        ]

    if anchor_class == "ethnic_bottom":
        # sharara/palazzo as anchor → need ethnic top + dupatta
        if is_men:
            return [
                SlotSpec("top", "kurta ethnic top", required=True),
                SlotSpec(
                    "footwear", "mojaris juttis ethnic footwear",
                    required=occasion_slug in _FORMAL_ETHNIC_OCCASIONS,
                ),
            ]
        else:
            return [
                SlotSpec("top", "kurta kurti ethnic top kameez", required=True),
                SlotSpec("accessory", "dupatta ethnic dupatta", required=True),
                SlotSpec(
                    "footwear", "juttis heels ethnic footwear",
                    required=occasion_slug in _FORMAL_ETHNIC_OCCASIONS,
                ),
            ]

    if anchor_class == "outerwear":
        return [
            SlotSpec("top", "top shirt blouse", required=True),
            SlotSpec("bottom", _default_bottom_query(occasion_slug), required=True),
        ]

    if anchor_class == "western_one_piece":
        return [
            SlotSpec("outerwear", "jacket cardigan blazer", required=False),
            SlotSpec("footwear", "shoes sandals boots heels", required=False),
            SlotSpec("accessory", "bag handbag", required=False),
        ]

    if anchor_class == "western_bottom":
        footwear_query = _default_footwear_query(occasion_slug, is_men)
        return [
            SlotSpec("top", "top shirt blouse", required=True),
            SlotSpec("outerwear", "jacket blazer coat cardigan", required=False),
            SlotSpec("footwear", footwear_query, required=False),
        ]

    # Default: western_top / unknown
    footwear_query = _default_footwear_query(occasion_slug, is_men)
    accessory_query = (
        "belt watch cap men accessory" if is_men else "handbag sling bag earrings women accessory"
    )
    return [
        SlotSpec("bottom", _default_bottom_query(occasion_slug), required=True),
        SlotSpec("outerwear", "jacket blazer coat cardigan", required=False),
        SlotSpec("footwear", footwear_query, required=False),
        SlotSpec("accessory", accessory_query, required=False),
    ]


def fabric_score_delta(
    item: dict, occasion_slug: str, formality_override: str | None = None
) -> float:
    """Return a score adjustment based on fabric/embellishment keywords for haldi vs sangeet.

    Base behaviour (formality_override absent — unchanged, backward compatible):
    - sangeet/reception/diwali: embellished items score +0.1; lightweight items
      score -0.1. Diwali joins this group (Wave 8) — a festival-of-lights
      evening register reads closer to sangeet/reception's embellished-glam
      bias than haldi/mehendi's undone-up daytime bias.
    - haldi/mehendi/raksha_bandhan: lightweight/floral items score +0.1;
      embellished items score -0.1. Raksha Bandhan joins this group (Wave 8) —
      formality 2/casual-festive reads closer to haldi's light, low-key bias.
    - all other occasions (including wedding_guest, navratri, karva_chauth,
      eid): 0.0 — these have their own dedicated colour_score palette overrides
      instead, and no strong embellishment-vs-lightweight signal.

    formality_override ("minimalist" | "comfortable", see FORMALITY_SOFTENER_VALUES —
    the `formality_softener` value a sibling intent-parser fix surfaces from
    queries like "something comfortable for sangeet dancing" or "not too
    flashy"): when present, OVERRIDES the base occasion-driven sign entirely —
    embellished/heavy items always score -0.1 and lightweight/plain items
    always score +0.1, regardless of occasion_slug. This fixes two confirmed
    live defects: (1) sangeet's base rule unconditionally boosts embellishment
    even when the query explicitly asks for something comfortable/low-key,
    with no way to suppress the bonus; (2) wedding_guest never got ANY
    embellishment-awareness because it isn't in the base sangeet/haldi/
    mehendi/reception occasion list — the override applies uniformly to
    wedding_guest (and any other occasion) once the user has signalled they
    want a toned-down look. wedding_guest's baseline (no override) behaviour
    stays 0.0, unchanged.

    Keyword check is heuristic — searches prod_name + detail_desc.
    """
    text = (
        (item.get("prod_name") or "") + " " + (item.get("detail_desc") or "")
    ).lower()
    has_embellishment = any(kw in text for kw in SANGEET_EMBELLISHMENT_KEYWORDS)
    has_lightweight = any(kw in text for kw in HALDI_LIGHTWEIGHT_KEYWORDS)

    if formality_override in FORMALITY_SOFTENER_VALUES:
        if has_embellishment:
            return -0.1
        if has_lightweight:
            return 0.1
        return 0.0

    if occasion_slug not in (
        "sangeet", "haldi", "mehendi", "reception", "diwali", "raksha_bandhan",
    ):
        return 0.0

    if occasion_slug in ("sangeet", "reception", "diwali"):
        if has_embellishment:
            return 0.1
        if has_lightweight:
            return -0.1
    else:  # haldi, mehendi, raksha_bandhan
        if has_lightweight:
            return 0.1
        if has_embellishment:
            return -0.1
    return 0.0
