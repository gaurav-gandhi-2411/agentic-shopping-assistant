from __future__ import annotations

from dataclasses import dataclass

ETHNIC_ONLY = "ethnic_only"
ETHNIC_HEAVY = "ethnic_heavy"
EITHER = "either"


@dataclass(frozen=True)
class Occasion:
    slug: str
    formality: int       # 1 (casual) – 5 (most formal)
    ethnic_lean: str     # ETHNIC_ONLY | ETHNIC_HEAVY | EITHER


OCCASIONS: dict[str, Occasion] = {
    "casual":              Occasion("casual",              1, EITHER),
    "smart_casual":        Occasion("smart_casual",        2, EITHER),
    "office":              Occasion("office",              3, EITHER),
    "haldi":               Occasion("haldi",               3, ETHNIC_ONLY),
    "mehendi":             Occasion("mehendi",             3, ETHNIC_ONLY),
    "party_evening":       Occasion("party_evening",       4, EITHER),
    "festive_puja":        Occasion("festive_puja",        4, ETHNIC_HEAVY),
    "wedding_guest":       Occasion("wedding_guest",       4, ETHNIC_HEAVY),
    "engagement":          Occasion("engagement",          4, EITHER),
    "sangeet":             Occasion("sangeet",             5, ETHNIC_ONLY),
    "traditional_ethnic":  Occasion("traditional_ethnic",  5, ETHNIC_ONLY),
    "reception":           Occasion("reception",           5, ETHNIC_HEAVY),
    # Wave 8 festival-occasion expansion — siblings of festive_puja, not
    # replacements (see festive_puja docstring note above OCCASIONS).
    "diwali":              Occasion("diwali",              4, ETHNIC_HEAVY),
    # Distinct from sangeet (a wedding event): navratri/garba is a standalone
    # dance-festival occasion, so ETHNIC_ONLY at formality 3, not 5.
    "navratri":            Occasion("navratri",            3, ETHNIC_ONLY),
    "karva_chauth":        Occasion("karva_chauth",        4, ETHNIC_ONLY),
    # Casual-festive, not heavy ethnic — either register is acceptable.
    "raksha_bandhan":      Occasion("raksha_bandhan",      2, EITHER),
    "eid":                 Occasion("eid",                 4, ETHNIC_HEAVY),
}

# Legacy slug aliases — kept in ONE place so a persisted session's occasion
# string from before "haldi_mehendi" was split into "haldi"/"mehendi" still
# resolves to a real Occasion instead of silently falling back to "casual".
# "haldi" is the compat target (not "mehendi") since it was the alphabetically
# first half of the old combined slug and carries the same formality/ethnic_lean.
_OCCASION_ALIASES: dict[str, str] = {
    "haldi_mehendi": "haldi",
}


def get_occasion(slug: str) -> Occasion:
    """Return Occasion for slug; fall back to casual if unknown.

    Resolves legacy slug aliases (see _OCCASION_ALIASES) before lookup so
    sessions persisted before an occasion slug was renamed/split still resolve
    to a real Occasion.
    """
    slug = _OCCASION_ALIASES.get(slug, slug)
    return OCCASIONS.get(slug, OCCASIONS["casual"])
