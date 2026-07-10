from __future__ import annotations

import re

from src.agents.outfit.occasions import ETHNIC_HEAVY, ETHNIC_ONLY, get_occasion
from src.agents.outfit.slots import (
    WOMEN_ONLY_ETHNIC_KEYWORDS,
    gender_allowed,
    is_ethnic_item,
    is_western_item,
    is_western_marker_item,
)

# Phase B (product gap 1): symmetric to the ETHNIC_ONLY gate below (gate 2) —
# a WESTERN-REGISTER occasion's slot candidates must never be ethnic-festive
# wear.  Currently only "office" (occasions.py: formality=3, ethnic_lean=
# EITHER) — the only occasion slug in this catalogue's model whose register is
# strictly business/tailored-western.  "work"/"formal" free text both map to
# the SAME "office" slug via intent_parser._OCCASION_MAP, so no other slug
# needs this.  Deliberately EXCLUDES party_evening (also EITHER-lean but not
# reported/asked for) and every ETHNIC_HEAVY/ETHNIC_ONLY occasion (haldi/
# mehendi/festive_puja/wedding_guest/sangeet/traditional_ethnic/reception) —
# those are already correctly ethnic-leaning and must never be gated toward western.
_WESTERN_REGISTER_OCCASIONS: frozenset[str] = frozenset({"office"})

# Live-proven: an "office look for women" board's bottom slot filled with a
# "Quirky Floral Printed Cotton Anarkali Sharara Set" — ethnic AND carrying an
# explicit festive/quirky marker.  Checked as a simple word-boundary denylist,
# same conservative style as _WESTERN_MARKER_KEYWORDS in slots.py.
_FESTIVE_MARKER_RE = re.compile(r"\b(quirky|festive)\b", re.IGNORECASE)


def is_western_register_occasion(occasion_slug: str) -> bool:
    """Return True if `occasion_slug` is one of the western-register
    occasions gated by is_coherent_candidate's gate 4 (see
    _WESTERN_REGISTER_OCCASIONS above). Exposed publicly so composer.py's
    pool-underflow retrieval fallback can key off the SAME occasion set
    used by the coherence gate, rather than duplicating the list.
    """
    return occasion_slug in _WESTERN_REGISTER_OCCASIONS

# Muted/earthy tones added for Phase B Part 1 (real catalogue colour audit — see
# offline check).  These coordinate well with each other in BOTH ethnic (jewel/
# warm co-star) and western (colour-story) styling — unlike the primary-hue
# _JEWEL_TONES below (red/blue/green/...), which really DO clash with each
# other in a western context (see test_western_mismatch_scores_low), these
# muted tones are a dedicated "coordinating" tier so e.g. a rust anchor +
# navy blue complement scores as a colour story, not a random clash.
_MUTED_COORDINATING_COLOURS: frozenset[str] = frozenset({
    "navy blue", "mustard", "burgundy", "lavender", "peach",
    "maroon", "olive", "teal", "wine", "rust",
})

# Jewel tones valid as co-stars in ethnic/festive occasions (multi_jewel harmony tier)
_JEWEL_TONES: frozenset[str] = frozenset({
    "red", "dark red", "dark pink", "pink", "orange", "dark orange",
    "blue", "dark blue", "green", "dark green", "purple", "dark purple",
    "gold", "yellow", "dark yellow", "turquoise", "dark turquoise",
}) | _MUTED_COORDINATING_COLOURS

_NEUTRAL_COLOURS: frozenset[str] = frozenset({
    "black", "white", "grey", "dark grey", "light grey", "beige",
    "light beige", "off white", "silver", "cream", "khaki", "charcoal",
})

# ── Couple-harmony palette (Phase B Part 2: cross-gender partner styling) ──────
# Maps an anchor look's colour to a small set of COMPLEMENTARY (never identical)
# partner-look colours.  Deliberately reuses/extends the muted-coordinating and
# neutral tiers above so a rust anchor + navy/cream partner look reads as the
# same deliberate colour story that colour_score() already treats as harmony —
# a companion look must coordinate WITH the anchor, not clash with it, but a
# matching-matching (identical hue) couple look is not the goal here.
COUPLE_HARMONY_MAP: dict[str, tuple[str, ...]] = {
    "rust": ("navy blue", "cream", "olive"),
    "black": ("burgundy", "grey", "white"),
    "white": ("navy blue", "black", "beige"),
    "navy blue": ("cream", "rust", "grey"),
    "dark blue": ("cream", "grey", "beige"),
    "blue": ("cream", "grey", "navy blue"),
    "red": ("navy blue", "charcoal", "cream"),
    "dark red": ("navy blue", "charcoal", "cream"),
    "pink": ("grey", "navy blue", "charcoal"),
    "light pink": ("grey", "navy blue", "charcoal"),
    "dark pink": ("grey", "navy blue", "charcoal"),
    "maroon": ("beige", "grey", "cream"),
    "burgundy": ("grey", "cream", "navy blue"),
    "wine": ("grey", "cream", "navy blue"),
    "mustard": ("navy blue", "charcoal", "olive"),
    "olive": ("cream", "mustard", "rust"),
    "teal": ("cream", "grey", "coral"),
    "green": ("cream", "beige", "grey"),
    "dark green": ("cream", "beige", "grey"),
    "purple": ("grey", "silver", "cream"),
    "dark purple": ("grey", "silver", "cream"),
    "yellow": ("navy blue", "charcoal", "grey"),
    "dark yellow": ("navy blue", "charcoal", "grey"),
    "beige": ("navy blue", "olive", "burgundy"),
    "cream": ("navy blue", "olive", "burgundy"),
    "gold": ("navy blue", "burgundy", "charcoal"),
    "orange": ("navy blue", "charcoal", "cream"),
    "dark orange": ("navy blue", "charcoal", "cream"),
    "grey": ("burgundy", "navy blue", "black"),
    "dark grey": ("burgundy", "navy blue", "black"),
    "peach": ("grey", "navy blue", "olive"),
    "lavender": ("grey", "charcoal", "navy blue"),
}

# Safe neutral fallback for any anchor colour not in COUPLE_HARMONY_MAP above —
# a companion look always gets SOME palette guidance rather than none.
_DEFAULT_COUPLE_HARMONY: tuple[str, ...] = ("navy blue", "grey", "charcoal")


def couple_harmony_palette(anchor_colour: str) -> tuple[str, ...]:
    """Return complementary partner-look colours for a given anchor colour.

    Deliberately EXCLUDES the anchor's own colour — a coordinated couple look
    (Phase B Part 2) reads through complementary/muted-neutral tones, not
    identical-hue matching-matching. Falls back to a safe neutral default
    (navy/grey/charcoal) for any anchor colour not in COUPLE_HARMONY_MAP.
    """
    key = (anchor_colour or "").lower().strip()
    return COUPLE_HARMONY_MAP.get(key, _DEFAULT_COUPLE_HARMONY)


def is_coherent_candidate(
    candidate: dict,
    occasion_slug: str,
    gender: str,
    slot_name: str,
    *,
    skip_gender_gate: bool = False,
) -> bool:
    """Return False if candidate violates any hard coherence gate; True otherwise.

    Hard gates (in priority order):
    0. Women-only ethnic categories (dupatta/saree/lehenga): hard reject for men.
    0b. Per-item gender mismatch: unknown gender is excluded from gendered looks.
    1. Dupatta slot: reject for men (belt-and-suspenders, also caught by gate 0).
    2. ethnic_only occasion: reject western items in any slot.
    3. ethnic_heavy occasion: reject western_casual items (western_formal OK for men's
       wedding_guest, and for either gender's reception — indo-western glam register).
    4. western_register occasion (office): reject ethnic items and festive/quirky markers.

    Args:
        skip_gender_gate: When True, skips gate 0b ONLY.  Set by composer.
            _find_best_candidate for the narrow gender-neutral-accessory
            fallback path (sunglasses/belt/watch/cap with gender="unknown"),
            where the caller has ALREADY verified the item is a genuinely
            unisex accessory sub-type — every other gate still runs unchanged.
    """
    occasion = get_occasion(occasion_slug)
    is_men = gender.lower() == "men"
    pt = candidate.get("product_type") or candidate.get("product_type_name") or ""
    name = candidate.get("prod_name") or candidate.get("display_name") or ""

    # Gate 0: women-only ethnic categories are a hard reject for men's looks,
    # belt-and-suspenders even if gender derivation missed them.
    if is_men:
        combined = (pt + " " + name).lower()
        if any(kw in combined for kw in WOMEN_ONLY_ETHNIC_KEYWORDS):
            return False

    # Gate 0b: per-item gender mismatch — unknown is excluded from gendered looks.
    if not skip_gender_gate:
        candidate_gender = (candidate.get("gender") or "unknown").lower()
        if not gender_allowed(candidate_gender, gender):
            return False

    # Gate 1: dupatta is women-only
    if slot_name == "accessory" and is_men:
        combined = (pt + " " + name).lower()
        if "dupatta" in combined:
            return False

    # Gate 2: ethnic_only occasions reject western items entirely.  is_western_item
    # only covers western_top/bottom/one_piece — is_western_marker_item is layered
    # on top so a footwear/outerwear/unknown-class item that's EXPLICITLY western
    # (sneakers, denim, bomber, hoodie, blazer, t-shirt) is caught too: a sangeet
    # look must never accept a pair of sneakers or a denim jacket.
    if occasion.ethnic_lean == ETHNIC_ONLY and (
        is_western_item(pt, name) or is_western_marker_item(pt, name)
    ):
        return False

    # Gate 3: ethnic_heavy occasions reject western_casual items (same marker
    # extension as gate 2 above).
    if occasion.ethnic_lean == ETHNIC_HEAVY and (
        is_western_item(pt, name) or is_western_marker_item(pt, name)
    ):
        # Western formal (blazer/trousers/shirt) may be OK for men's wedding_guest
        if is_men and occasion_slug == "wedding_guest":
            combined = (pt + " " + name).lower()
            is_formal_western = any(
                kw in combined for kw in ("blazer", "trousers", "shirt", "suit", "formal")
            )
            if is_formal_western:
                return True  # allowed
        # Reception is an indo-western glam evening register for EITHER gender
        # (see _anchor_query_for_occasion's "reception" query) — a formal gown
        # is also allowed alongside blazer/trousers/shirt/suit.
        if occasion_slug == "reception":
            combined = (pt + " " + name).lower()
            is_formal_western = any(
                kw in combined
                for kw in ("blazer", "trousers", "shirt", "suit", "formal", "gown")
            )
            if is_formal_western:
                return True  # allowed
        return False

    # Gate 4 (Phase B product gap 1): WESTERN_REGISTER occasions (currently
    # only "office" — see _WESTERN_REGISTER_OCCASIONS docstring) reject
    # ethnic-classified items AND anything carrying an explicit festive/quirky
    # marker, symmetric to gates 2/3 above.  Live-proven: an office look's
    # bottom slot filled with a "Quirky Floral Printed Cotton Anarkali
    # Sharara Set" (ethnic + festive).
    if occasion_slug in _WESTERN_REGISTER_OCCASIONS and (
        is_ethnic_item(pt, name) or _FESTIVE_MARKER_RE.search(name)
    ):
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

    Haldi palette override: if occasion is haldi and candidate is yellow/
    orange/marigold palette, score 1.0; anything dark scores 0.2.

    Mehendi palette override: green/mint/olive palette scores 1.0 (mirrors the
    haldi override's shape); anything dark scores 0.2.

    Reception palette override: jewel/dark-glam tones score highest (reuses
    the ethnic_heavy _JEWEL_TONES tier below rather than inventing new scoring
    machinery); pale/light casual tones score low — an evening indo-western
    reception look should read glam, not light-daytime pastel.
    """
    occasion = get_occasion(occasion_slug)
    c_lower = candidate_colour.lower()
    a_lower = anchor_colour.lower()

    # Haldi override
    if occasion_slug == "haldi":
        _haldi_palette = {"yellow", "light yellow", "dark yellow", "orange", "light orange",
                          "dark orange", "gold"}
        if c_lower in _haldi_palette:
            return 1.0
        if c_lower in ("black", "dark grey", "dark blue", "dark red", "dark purple"):
            return 0.2
        return 0.6

    # Mehendi override
    if occasion_slug == "mehendi":
        _mehendi_palette = {"green", "dark green", "light green", "olive", "teal",
                             "turquoise", "dark turquoise", "mint"}
        if c_lower in _mehendi_palette:
            return 1.0
        if c_lower in ("black", "dark grey", "dark blue", "dark red", "dark purple"):
            return 0.2
        return 0.6

    # Reception override
    if occasion_slug == "reception":
        _pale_casual = {"light pink", "light yellow", "light blue", "light grey",
                         "light beige", "light orange", "beige", "off white"}
        if c_lower in _JEWEL_TONES or c_lower in ("black", "dark grey", "charcoal"):
            return 0.9
        if c_lower in _pale_casual:
            return 0.3
        if c_lower == a_lower:
            return 0.85
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

    # Western occasions: original rules, extended with a "muted coordinating"
    # tier (Phase B Part 1) so two earthy/jewel-warm tones (e.g. rust + navy
    # blue) read as a deliberate colour story rather than a random clash — this
    # is a SEPARATE, smaller set from _JEWEL_TONES precisely so bright primary
    # pairs (red+blue, both already in _JEWEL_TONES) keep clashing here, as
    # tested by test_western_mismatch_scores_low.
    if c_lower in _NEUTRAL_COLOURS or a_lower in _NEUTRAL_COLOURS:
        return 1.0
    if c_lower == a_lower:
        return 0.9
    if c_lower in _MUTED_COORDINATING_COLOURS and a_lower in _MUTED_COORDINATING_COLOURS:
        return 0.75
    return 0.4
