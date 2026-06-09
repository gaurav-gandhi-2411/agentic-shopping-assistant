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
    "haldi_mehendi":       Occasion("haldi_mehendi",       3, ETHNIC_ONLY),
    "party_evening":       Occasion("party_evening",       4, EITHER),
    "festive_puja":        Occasion("festive_puja",        4, ETHNIC_HEAVY),
    "wedding_guest":       Occasion("wedding_guest",       4, ETHNIC_HEAVY),
    "sangeet":             Occasion("sangeet",             5, ETHNIC_ONLY),
    "traditional_ethnic":  Occasion("traditional_ethnic",  5, ETHNIC_ONLY),
}


def get_occasion(slug: str) -> Occasion:
    """Return Occasion for slug; fall back to casual if unknown."""
    return OCCASIONS.get(slug, OCCASIONS["casual"])
