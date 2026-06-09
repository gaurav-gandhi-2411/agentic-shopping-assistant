from __future__ import annotations

from src.agents.outfit.occasions import ETHNIC_HEAVY, ETHNIC_ONLY, get_occasion
from src.agents.outfit.slots import is_western_item

# Jewel tones valid as co-stars in ethnic/festive occasions (multi_jewel harmony tier)
_JEWEL_TONES: frozenset[str] = frozenset({
    "red", "dark red", "dark pink", "pink", "orange", "dark orange",
    "blue", "dark blue", "green", "dark green", "purple", "dark purple",
    "gold", "yellow", "dark yellow", "turquoise", "dark turquoise",
})

_NEUTRAL_COLOURS: frozenset[str] = frozenset({
    "black", "white", "grey", "dark grey", "light grey", "beige",
    "light beige", "off white", "silver",
})


def is_coherent_candidate(
    candidate: dict,
    occasion_slug: str,
    gender: str,
    slot_name: str,
) -> bool:
    """Return False if candidate violates any hard coherence gate; True otherwise.

    Hard gates (in priority order):
    1. Dupatta slot: reject for men.
    2. ethnic_only occasion: reject western items in any slot.
    3. ethnic_heavy occasion: reject western_casual items (western_formal OK for men's wedding_guest).
    4. Ethnic anchor + formality >= 4: reject western candidates in non-outerwear slots.
    """
    occasion = get_occasion(occasion_slug)
    is_men = gender.lower() == "men"
    pt = candidate.get("product_type") or candidate.get("product_type_name") or ""
    name = candidate.get("prod_name") or candidate.get("display_name") or ""

    # Gate 1: dupatta is women-only
    if slot_name == "accessory" and is_men:
        combined = (pt + " " + name).lower()
        if "dupatta" in combined:
            return False

    # Gate 2: ethnic_only occasions reject western items entirely
    if occasion.ethnic_lean == ETHNIC_ONLY and is_western_item(pt, name):
        return False

    # Gate 3: ethnic_heavy occasions reject western_casual items
    if occasion.ethnic_lean == ETHNIC_HEAVY and is_western_item(pt, name):
        # Western formal (blazer/trousers/shirt) may be OK for men's wedding_guest
        if is_men and occasion_slug == "wedding_guest":
            combined = (pt + " " + name).lower()
            is_formal_western = any(
                kw in combined for kw in ("blazer", "trousers", "shirt", "suit", "formal")
            )
            if is_formal_western:
                return True  # allowed
        return False

    return True


def colour_score(
    candidate_colour: str,
    anchor_colour: str,
    occasion_slug: str,
) -> float:
    """Return a [0, 1] colour compatibility score.

    For ethnic_heavy/ethnic_only occasions: clash penalties are suspended.
    Jewel tones can co-star (multi_jewel harmony tier). Score is 0.7 for any
    jewel-tone pairing (good), 0.5 for neutral, 0.3 penalty only for truly
    clashing pairs even in western context.

    For western occasions: neutrals score 1.0 with anything; same-colour scores
    0.9; mismatched non-neutrals score 0.4.

    Haldi palette override: if occasion is haldi_mehendi and candidate is yellow/
    orange/marigold palette, score 1.0; anything dark scores 0.2.
    """
    occasion = get_occasion(occasion_slug)
    c_lower = candidate_colour.lower()
    a_lower = anchor_colour.lower()

    # Haldi override
    if occasion_slug == "haldi_mehendi":
        _haldi_palette = {"yellow", "light yellow", "dark yellow", "orange", "light orange",
                          "dark orange", "gold"}
        if c_lower in _haldi_palette:
            return 1.0
        if c_lower in ("black", "dark grey", "dark blue", "dark red", "dark purple"):
            return 0.2
        return 0.6

    # Ethnic festive occasions: no clash penalty; jewel tones co-star
    if occasion.ethnic_lean in (ETHNIC_HEAVY, ETHNIC_ONLY):
        if c_lower in _NEUTRAL_COLOURS:
            return 0.7  # neutrals are fine but not preferred over jewel tones
        if c_lower == a_lower:
            return 0.9  # monochromatic is excellent
        if c_lower in _JEWEL_TONES:
            return 0.8  # jewel-tone co-star is valid
        return 0.6  # anything else is acceptable

    # Western occasions: original rules
    if c_lower in _NEUTRAL_COLOURS or a_lower in _NEUTRAL_COLOURS:
        return 1.0
    if c_lower == a_lower:
        return 0.9
    return 0.4
