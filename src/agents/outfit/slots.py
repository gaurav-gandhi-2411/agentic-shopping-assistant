from __future__ import annotations

from dataclasses import dataclass

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


def get_fill_slots(anchor_class: str, gender: str, occasion_slug: str) -> list[SlotSpec]:
    """Return ordered list of SlotSpecs to fill for a given anchor + gender + occasion.

    Gender: "men" | "women" | "unisex" (treated as women for ethnic, men for men's brands).
    """
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
            SlotSpec("bottom", "trousers jeans skirt", required=True),
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
        SlotSpec("bottom", "trousers jeans skirt", required=True),
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
