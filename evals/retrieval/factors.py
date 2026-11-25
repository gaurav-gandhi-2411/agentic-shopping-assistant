"""Factors run on the two best variants (chosen by highest nDCG@5 point estimate).

F1  sets demoted vs not, ethnic-wear queries only.
    Flag (deterministic, from the fixture's occasion + garment type, no model involved):
    occasion in ETHNIC_OCCASIONS and the stated garment is a single ethnic garment, i.e. its
    category is in ETHNIC_SINGLE_GARMENTS (no "set", no generic "outfit"/"ethnic wear"). Those
    are exactly the queries where production's single-garment gate fires (generic asks and
    set/combo words skip it). Arm "sets demoted" re-runs the variant on the pool after
    production's `_prefer_single_garment_pool(items, 5)` (imported, not reimplemented) applied to
    the FROZEN top-20: a stable partition singles-first; when >= 5 singles exist the sets are
    dropped from the pool (production's behaviour), otherwise they are demoted behind the
    singles. Arm "not demoted" is the frozen pool as-is (current main-comparison result).
    HOW POOL-MEMBERSHIP CHANGES ARE JUDGED: every candidate keeps the label it already has (one
    judging pass over the frozen 20 serves every arm, so no new label is ever needed), and nDCG's
    ideal ranking is taken from the FULL frozen pool in BOTH arms, so removing sets can never
    shrink the ideal and flatter the arm. The pool-loss cost is reported next to it: items
    removed, and how many of those had label >= 1 (relevant items the demotion threw away).

F2  gender-unspecified queries (fixtures with gender "either", 12): current result vs a
    gender-interleaved pool. Interleaving alternates women's / men's items in retrieval-rank
    order, starting with the gender of the best-ranked gendered item; items whose catalogue
    gender is neither (unisex/unknown) keep their original positions; when one gender runs out
    the other's items follow in rank order. Same pool, same labels, only the order changes; the
    variant is re-run on the reordered pool (for B the order IS the result).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

ETHNIC_OCCASIONS = frozenset({"sangeet", "haldi", "mehendi", "wedding_guest", "festive_diwali"})
ETHNIC_SINGLE_GARMENTS = frozenset(
    {"kurta", "kurti", "anarkali", "anarkali gown", "lehenga", "saree", "sherwani", "bandhgala"}
)
TOP_K = 5


def is_set_demotion_query(case: Mapping[str, Any]) -> bool:
    """F1 flag: ethnic occasion AND a single ethnic garment named (see module docstring)."""
    category = str((case.get("constraints") or {}).get("category") or "").lower()
    return case.get("occasion") in ETHNIC_OCCASIONS and category in ETHNIC_SINGLE_GARMENTS


def is_gender_unspecified(case: Mapping[str, Any]) -> bool:
    """F2 flag: the query states no gender."""
    return case.get("gender") == "either"


def demote_sets(items: Sequence[dict[str, Any]], top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Production's single-garment pool gate applied to a frozen pool (stable partition)."""
    from src.agents.graph import _prefer_single_garment_pool

    return list(_prefer_single_garment_pool(list(items), top_k))


def pool_loss(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    labels: Mapping[str, float],
) -> dict[str, int]:
    """Pool-membership accounting for F1: items removed, and how many carried label >= 1."""
    kept = {str(it["article_id"]) for it in after}
    removed = [str(it["article_id"]) for it in before if str(it["article_id"]) not in kept]
    return {
        "n_before": len(before),
        "n_after": len(after),
        "n_removed": len(removed),
        "n_removed_relevant": sum(1 for a in removed if labels.get(a, 0) >= 1),
    }


def interleave_gender(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Alternate women's and men's items in retrieval-rank order (see module docstring)."""
    women = [it for it in items if it.get("gender") == "women"]
    men = [it for it in items if it.get("gender") == "men"]
    gendered = [it for it in items if it.get("gender") in ("women", "men")]
    if not gendered:
        return list(items)
    first, second = (women, men) if gendered[0].get("gender") == "women" else (men, women)
    merged: list[dict[str, Any]] = []
    for i in range(max(len(first), len(second))):
        if i < len(first):
            merged.append(first[i])
        if i < len(second):
            merged.append(second[i])
    out: list[dict[str, Any]] = []
    fill = iter(merged)
    for it in items:
        out.append(it if it.get("gender") not in ("women", "men") else next(fill))
    return out
