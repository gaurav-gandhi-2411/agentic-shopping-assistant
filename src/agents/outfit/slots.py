from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

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
    "palazzo", "churidar", "salwar", "sharara", "pyjama", "dhoti",
})
WESTERN_TOP_KEYWORDS: frozenset[str] = frozenset({
    "shirt", "t-shirt", "tshirt", "top", "blouse", "sweater", "sweatshirt",
    "tank top", "crop top", "polo",
})
WESTERN_BOTTOM_KEYWORDS: frozenset[str] = frozenset({
    "trousers", "jeans", "shorts", "skirt", "jeggings",
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
    {"jewellery", "jewelry", "jhumka", "earrings", "necklace", "bangle"}
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


# S5 fix: juniors/kids garments mislabeled as adult inventory. This catalogue's
# `gender` column derives from `index_group_name`, and juniors/girls/boys/kids
# SKUs are indexed under "Ladieswear"/"Menswear" alongside genuinely adult
# items (verified: "M&H Juniors Girls Blue Straight Knee Length Denim Skirts"
# and "Juniors by Lifestyle Kids-Girls White Pure Cotton Print Top" both carry
# gender="women", index_group_name="Ladieswear" in data/processed/unified/
# catalogue.parquet) — so gender_allowed() alone lets them through into ADULT
# outfit slots. Live-proven: an office look's bottom slot filled with the
# Juniors denim-skirt item above. Deliberately narrow (four markers, not a
# broader age/size heuristic) to avoid rejecting real adult inventory whose
# name happens to share a word.
_KIDS_MARKER_RE = re.compile(r"\b(junior|juniors|girl|girls|boy|boys|kid|kids)\b", re.IGNORECASE)


def is_kids_item(prod_name: str) -> bool:
    """Return True if `prod_name` carries a juniors/girls/boys/kids marker.

    Checked as an ADDITIONAL gate in composer._find_best_candidate, alongside
    (never instead of) the gender/slot-type/novelty gates — see module
    docstring on _KIDS_MARKER_RE for why the gender column alone isn't enough.
    """
    return bool(_KIDS_MARKER_RE.search(prod_name or ""))


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
_FORMAL_ETHNIC_OCCASIONS: frozenset[str] = frozenset({
    "sangeet", "haldi_mehendi", "festive_puja", "wedding_guest", "traditional_ethnic",
})

# Women-only ethnic categories — hard reject for men's looks regardless of gender field
WOMEN_ONLY_ETHNIC_KEYWORDS: frozenset[str] = frozenset({
    "dupatta", "saree", "lehenga",
})

# Fabric/embellishment keywords for haldi_mehendi vs sangeet scoring
SANGEET_EMBELLISHMENT_KEYWORDS: frozenset[str] = frozenset({
    "embroidered", "embroidery", "sequin", "zari", "embellished",
    "heavy work", "bridal", "mirror work", "thread work", "beaded",
    "resham", "gota", "kundan",
})
HALDI_LIGHTWEIGHT_KEYWORDS: frozenset[str] = frozenset({
    "cotton", "floral", "tie-dye", "georgette", "chiffon", "printed",
    "casual", "lightweight", "summer", "marigold", "yellow", "daisy",
})


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

    haldi_mehendi keeps its own "cotton floral" register rather than the
    generic ethnic-festive one, since it already has a dedicated lightweight/
    floral colour+fabric bias (colour_score, fabric_score_delta) that would
    conflict with "embroidered" (haldi favours light, undone-up looks).
    """
    if occasion_slug == "haldi_mehendi":
        return "cotton floral"
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
    """
    occ = get_occasion(occasion_slug)
    if occ.ethnic_lean == EITHER and occ.formality >= 3:
        return "trousers"
    return "trousers jeans skirt"


def get_fill_slots(anchor_class: str, gender: str, occasion_slug: str) -> list[SlotSpec]:
    """Return ordered list of SlotSpecs to fill for a given anchor + gender + occasion.

    Gender: "men" | "women" | "unisex" (treated as women for ethnic, men for men's brands).

    Thin wrapper around _get_fill_slots_base(): appends occasion-register
    tokens (see _occasion_register_tokens) to every slot's search_query so
    retrieval is occasion-aware (formal tailored / festive embroidered /
    casual), without touching the base per-anchor-class slot definitions.
    """
    slots = _get_fill_slots_base(anchor_class, gender, occasion_slug)
    return _append_register(slots, occasion_slug)


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
        footwear_query = (
            "sneakers casual shoes loafers men"
            if is_men
            else "sneakers flats heels casual shoes women"
        )
        return [
            SlotSpec("top", "top shirt blouse", required=True),
            SlotSpec("outerwear", "jacket blazer coat cardigan", required=False),
            SlotSpec("footwear", footwear_query, required=False),
        ]

    # Default: western_top / unknown
    footwear_query = (
        "sneakers casual shoes loafers men"
        if is_men
        else "sneakers flats heels casual shoes women"
    )
    accessory_query = (
        "belt watch cap men accessory" if is_men else "handbag sling bag earrings women accessory"
    )
    return [
        SlotSpec("bottom", _default_bottom_query(occasion_slug), required=True),
        SlotSpec("outerwear", "jacket blazer coat cardigan", required=False),
        SlotSpec("footwear", footwear_query, required=False),
        SlotSpec("accessory", accessory_query, required=False),
    ]


def fabric_score_delta(item: dict, occasion_slug: str) -> float:
    """Return a score adjustment based on fabric/embellishment keywords for haldi vs sangeet.

    For sangeet: embellished items score +0.1; lightweight items score -0.1.
    For haldi_mehendi: lightweight/floral items score +0.1; embellished items score -0.1.
    For all other occasions: 0.0.

    Keyword check is heuristic — searches prod_name + detail_desc.
    """
    if occasion_slug not in ("sangeet", "haldi_mehendi"):
        return 0.0

    text = (
        (item.get("prod_name") or "") + " " + (item.get("detail_desc") or "")
    ).lower()

    has_embellishment = any(kw in text for kw in SANGEET_EMBELLISHMENT_KEYWORDS)
    has_lightweight = any(kw in text for kw in HALDI_LIGHTWEIGHT_KEYWORDS)

    if occasion_slug == "sangeet":
        if has_embellishment:
            return 0.1
        if has_lightweight:
            return -0.1
    else:  # haldi_mehendi
        if has_lightweight:
            return 0.1
        if has_embellishment:
            return -0.1
    return 0.0
