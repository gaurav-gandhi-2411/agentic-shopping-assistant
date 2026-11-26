"""Build-time gender correction: product name's leading word vs the gender label.

Pre-registered rule: docs/ops/incremental-build-rules.md section 5. Pure functions, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MEN = frozenset({"men", "mens", "man", "gents", "gent", "male", "boys", "boy"})
_WOMEN = frozenset({"women", "womens", "woman", "ladies", "lady", "female", "girls", "girl"})
_LABELS = ("men", "women")

# Outcomes of ``resolve_gender``.
CORRECTED = "corrected"
NO_CONFLICT = "no_conflict"
AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class GenderDecision:
    """Result for one row. ``after`` equals ``before`` unless ``outcome == CORRECTED``."""

    outcome: str
    before: str
    after: str
    reason: str


def _words(text: object) -> list[str]:
    return re.findall(r"[a-z]+", str(text or "").lower())


def _gender_of_words(words: list[str]) -> set[str]:
    found: set[str] = set()
    if _MEN.intersection(words):
        found.add("men")
    if _WOMEN.intersection(words):
        found.add("women")
    return found


def leading_gender(name: object) -> str | None:
    """Gender named by the FIRST word of ``name`` ("men"/"women"), else None."""
    words = _words(name)
    if not words:
        return None
    found = _gender_of_words(words[:1])
    return next(iter(found)) if found else None


def resolve_gender(
    name: object, label: object, product_type: object, tags: list[str] | tuple[str, ...]
) -> GenderDecision:
    """Apply the registered rule to one row; see the rules doc for the order of the steps."""
    before = str(label or "").strip().lower()
    g = leading_gender(name)
    if g is None or before not in _LABELS or before == g:
        return GenderDecision(NO_CONFLICT, before, before, "no_conflict")
    type_words = _words(product_type)
    tag_words = [w for t in tags for w in _words(t)]
    if "unisex" in _words(name) + type_words + tag_words:
        return GenderDecision(AMBIGUOUS, before, before, "unisex_marker")
    for source, words in (("product_type", type_words), ("tags", tag_words)):
        evidence = _gender_of_words(words)
        if len(evidence) == 2:
            return GenderDecision(AMBIGUOUS, before, before, f"{source}_both_genders")
        if evidence == {g}:
            return GenderDecision(CORRECTED, before, g, f"{source}_agrees_with_name")
        if evidence:
            return GenderDecision(AMBIGUOUS, before, before, f"{source}_supports_label")
    return GenderDecision(CORRECTED, before, g, "leading_word_no_other_evidence")
