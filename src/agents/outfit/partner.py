"""Cross-gender PARTNER styling — Phase B Part 2.

Detects an EXPLICIT request to style a coordinated companion look for the
user's partner ("what should my husband wear with this?", "style my wife to
match", "his and hers", "couple outfit for a wedding") and composes a
separate, LABELLED look in the partner's gender that harmonises with the
current session anchor/look — complementary (not identical) colours, the same
occasion, and the same formality tier.

Design notes
------------
- Detection is regex-first and deliberately conservative (see the regexes
  below) — ambiguous phrasing ("also show me shirts", "women's shirts for me
  too") must never fire.  This mirrors graph.py's existing deterministic-core
  routing (e.g. ``_LOOK_REFINEMENT_RE``, ``_STYLE_ANCHOR_RE``).
- "for him"/"for her" alone is the weakest signal in the spec and only fires
  alongside an explicit styling/coordination verb in the SAME query —
  otherwise a plain gendered product search ("show me shirts for him") would
  misfire as a partner-styling request.
- The companion look is composed with the SAME ``compose_outfit`` machinery
  used for the primary look (gender hard filter, slot-type gates, honest slot
  suppression all apply unchanged) — this module only resolves intent/gender
  and derives the coordinating colour palette + a partner-gender seed query;
  ``composer.compose_outfit`` does the actual retrieval + slot filling.
- ``build_coordinated_with_text`` is fully DETERMINISTIC (no LLM call) —
  every colour/type token it references comes directly from the anchor item
  or the composed partner look itself, so it is grounded by construction and
  cheap (cost-aware default: no extra LLM round-trip just for this one line).
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

import pandas as pd

from src.agents.outfit.coherence import couple_harmony_palette
from src.agents.outfit.composer import _anchor_matches_occasion, compose_outfit
from src.agents.outfit.occasions import ETHNIC_HEAVY, ETHNIC_ONLY, get_occasion
from src.agents.outfit.slots import gender_allowed
from src.retrieval.hybrid_search import HybridRetriever

# ── Intent detection ────────────────────────────────────────────────────────

# Words that directly imply a concrete partner gender.
# "groom"/"bride" are deliberately NOT here (see _GROOM_RE/_BRIDE_RE below) —
# husband/wife/boyfriend/girlfriend are near-always used in a possessive,
# "my partner" sense, but "sherwani for groom"/"lehenga for bride" are
# extremely common PLAIN single-item search phrasings (occasion-role
# descriptors, same as "for men") with zero partner-coordination intent.
# Live-proven 2026-07-11: "sherwani for groom" misrouted to a partner-styling
# clarify question instead of ever searching.
_MEN_WORD_RE = re.compile(r"\b(husband|boyfriend)\b", re.IGNORECASE)
_WOMEN_WORD_RE = re.compile(r"\b(wife|girlfriend)\b", re.IGNORECASE)

# Weaker signal: only counts alongside an explicit styling/coordination verb
# in the SAME query — mirrors the "for him"/"for her" treatment below.
_GROOM_RE = re.compile(r"\bgroom\b", re.IGNORECASE)
_BRIDE_RE = re.compile(r"\bbride\b", re.IGNORECASE)

# Words that name a partner WITHOUT implying a specific gender — resolved
# against the anchor look's own gender (opposite of it) by resolve_partner_gender.
# "fiance(e)" is intentionally bucketed here rather than gendered directly:
# plain-ASCII "fiance"/"fiancee" and the accented "fiancé"/"fiancée" forms are
# not reliably distinguishable by gender without depending on an accent most
# users won't type — treating it as "opposite" (anchor-derived) is the safer,
# conservative choice consistent with "no guessing" per the intent spec.
_OPPOSITE_WORD_RE = re.compile(r"\b(partner|couple|fianc[ée]e?)\b", re.IGNORECASE)
_HIS_AND_HERS_RE = re.compile(r"\bhis\s+and\s+hers\b", re.IGNORECASE)

# Weakest signal: only counts alongside an explicit styling/coordination verb
# in the SAME query (see module docstring).
_FOR_HIM_RE = re.compile(r"\bfor\s+him\b", re.IGNORECASE)
_FOR_HER_RE = re.compile(r"\bfor\s+her\b", re.IGNORECASE)
_STYLING_VERB_RE = re.compile(
    r"\b(style|wear|match|matching|coordinate|coordinated|outfit|dress|look)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PartnerIntent:
    """Result of scanning a user query for explicit partner-styling intent.

    Attributes:
        matched: True when an explicit partner-styling signal was found.
        gender_hint: "men" | "women" | "opposite" | None. "opposite" means the
            triggering word (partner/couple/fiance(e)/"his and hers") doesn't
            imply a gender on its own — resolve_partner_gender resolves it
            against the current look/anchor's own gender.
        matched_phrase: the literal phrase that triggered the match (for logging).
    """

    matched: bool
    gender_hint: str | None
    matched_phrase: str | None


def detect_partner_intent(raw_query: str) -> PartnerIntent:
    """Return PartnerIntent for an explicit cross-gender partner-styling request.

    Fires ONLY on:
      - An explicit relationship noun: husband, wife, boyfriend, girlfriend,
        fiance(e), partner, couple.
      - The literal phrase "his and hers".
      - "for him"/"for her"/"groom"/"bride" WHEN a styling/coordination verb
        ("style", "wear", "match", "coordinate", "outfit", "dress", "look")
        is ALSO present in the same query — "groom"/"bride" alone are plain
        occasion-role descriptors ("sherwani for groom"), not a partner
        signal on their own; "the groom needs a matching look" IS one.

    Never fires on ambiguous phrasing with no relationship signal at all —
    e.g. "also show me shirts" or "women's shirts for me too".
    """
    if _HIS_AND_HERS_RE.search(raw_query):
        return PartnerIntent(True, "opposite", "his and hers")

    m = _MEN_WORD_RE.search(raw_query)
    if m:
        return PartnerIntent(True, "men", m.group(1).lower())

    m = _WOMEN_WORD_RE.search(raw_query)
    if m:
        return PartnerIntent(True, "women", m.group(1).lower())

    m = _OPPOSITE_WORD_RE.search(raw_query)
    if m:
        return PartnerIntent(True, "opposite", m.group(1).lower())

    has_styling_verb = bool(_STYLING_VERB_RE.search(raw_query))
    if has_styling_verb and _FOR_HIM_RE.search(raw_query):
        return PartnerIntent(True, "men", "for him")
    if has_styling_verb and _FOR_HER_RE.search(raw_query):
        return PartnerIntent(True, "women", "for her")
    if has_styling_verb and _GROOM_RE.search(raw_query):
        return PartnerIntent(True, "men", "groom")
    if has_styling_verb and _BRIDE_RE.search(raw_query):
        return PartnerIntent(True, "women", "bride")

    return PartnerIntent(False, None, None)


def resolve_partner_gender(gender_hint: str, anchor_gender: str) -> str:
    """Resolve a PartnerIntent.gender_hint ("men"/"women"/"opposite") to a
    concrete "men"/"women" using the session's current look/anchor gender.

    "opposite" (partner/couple/fiance(e)/"his and hers") always resolves to
    the gender NOT worn by the anchor — a companion look is, by definition,
    for the other half of the pair.
    """
    if gender_hint in ("men", "women"):
        return gender_hint
    return "men" if anchor_gender == "women" else "women"


# ── Companion-look composition ──────────────────────────────────────────────

# Occasion-aware core garment for the partner's SEED item (requirement 3:
# "seed the companion look with a partner-gender core garment — men →
# shirt/kurta by occasion ethnicity; women → dress/kurta similarly").
def _partner_seed_query(occasion_slug: str, partner_gender: str, palette: tuple[str, ...]) -> str:
    """Build the retrieval query for the partner's own seed/anchor item."""
    occ = get_occasion(occasion_slug)
    is_men = partner_gender == "men"
    if occ.ethnic_lean in (ETHNIC_HEAVY, ETHNIC_ONLY):
        core = "sherwani kurta festive embellished ethnic" if is_men else (
            "kurta kurti anarkali lehenga festive ethnic"
        )
    else:
        core = "shirt casual" if is_men else "dress top casual"
    colour_terms = " ".join(palette[:3])
    return f"{core} {colour_terms}"


def _empty_partner_result(occasion_slug: str, gender: str, reason: str) -> dict:
    """Same shape as composer._empty_result — used when no seed can be found."""
    return {
        "look_id": str(uuid.uuid4()),
        "seed_item": None,
        "complements": [],
        "outfit_rationale": reason,
        "empty_slots": [],
        "suppressed_slots": [],
        "occasion": occasion_slug,
        "gender": gender,
        "budget_total_inr": None,
    }


def compose_partner_look(
    catalogue_df: pd.DataFrame,
    retriever: HybridRetriever,
    *,
    anchor_item: dict,
    occasion_slug: str,
    partner_gender: str,
    budget_inr: float | None = None,
) -> dict:
    """Compose a coordinated companion look in ``partner_gender`` that
    harmonises with ``anchor_item`` (the current session look/anchor).

    Steps:
      1. Derive a complementary colour palette from the anchor's own colour
         (couple_harmony_palette — NEVER the anchor's exact colour).
      2. Retrieve a partner-gender core garment seed, preferring a palette
         colour when one is available in the retrieved candidate pool.
      3. Hand off to compose_outfit (SAME gender hard filter, slot-type
         gates, coherence gates, honest suppression as the primary look) to
         fill the remaining slots.

    Args:
        budget_inr: optional total-look budget cap, in INR, for the PARTNER's
            own look. Mirrors composer.compose_outfit's occasion-driven-anchor
            budget gate (commit 85078b1 — "sangeet look under ₹8000" boarded a
            ₹9,900 lehenga ANCHOR before this fix): the seed-candidate
            selection below rejects over-budget candidates BEFORE any
            complement is even considered, and the same cap is forwarded to
            the internal compose_outfit call so complements are budget-
            squeezed too. None (default) is a full no-op, same as before.

    Returns a dict in the same shape as compose_outfit's return value
    (look_id, seed_item, complements, outfit_rationale, empty_slots,
    suppressed_slots, occasion, gender, budget_total_inr). seed_item is None
    when no partner-gender candidate exists for this occasion (or none fits
    ``budget_inr``) in the catalogue — callers should respond honestly in
    that case (mirrors compose_outfit's own occasion-driven-entry failure
    mode).
    """
    anchor_colour = (anchor_item.get("colour") or "").lower().strip()
    palette = couple_harmony_palette(anchor_colour)
    seed_query = _partner_seed_query(occasion_slug, partner_gender, palette)

    candidates = retriever.search(seed_query, top_k=15, filters={"gender": partner_gender})
    valid = [
        c
        for c in candidates
        if _anchor_matches_occasion(c, occasion_slug)
        and gender_allowed((c.get("gender") or "unknown").lower(), partner_gender)
    ]
    # Budget gate (mirrors compose_outfit's occasion-driven anchor path, commit
    # 85078b1): reject over-budget seed candidates before any complement is
    # considered, so the companion look never boards a seed that already
    # blows the stated cap on its own.
    _pre_budget_valid = valid
    if budget_inr is not None:
        valid = [c for c in valid if (c.get("price_inr") or 0.0) <= budget_inr]
    if not valid:
        if budget_inr is not None and _pre_budget_valid:
            # Occasion/gender-valid candidates existed but ALL were over
            # budget — honest budget-specific message, never a silent
            # fall-back to an over-budget seed.
            reason = (
                f"No {partner_gender}'s {occasion_slug.replace('_', ' ')} pieces within "
                f"₹{budget_inr:,.0f} in our partner stores yet for a companion look — "
                f"try a higher budget."
            )
        else:
            reason = (
                f"No {partner_gender}'s items found in this catalogue for a "
                f"{occasion_slug.replace('_', ' ')} companion look."
            )
        return _empty_partner_result(occasion_slug, partner_gender, reason)

    # Prefer a candidate whose colour is IN the harmony palette; fall back to
    # the top-ranked candidate otherwise (still occasion/gender/budget-valid).
    def _colour_rank(item: dict) -> int:
        return 0 if (item.get("colour") or "").lower() in palette else 1

    valid.sort(key=_colour_rank)
    seed_item = valid[0]

    return compose_outfit(
        catalogue_df,
        retriever,
        seed_article_id=seed_item["article_id"],
        occasion_slug=occasion_slug,
        gender=partner_gender,
        budget_inr=budget_inr,
    )


# ── Couple-from-scratch orchestration (P2) ──────────────────────────────────


def compose_couple_look(
    catalogue_df: pd.DataFrame,
    retriever: HybridRetriever,
    *,
    occasion_slug: str,
    partner_gender: str,
    budget_inr: float | None = None,
    brand_gender_default: str = "women",
) -> tuple[dict, dict]:
    """Compose a FROM-SCRATCH pair of coordinated looks when no session anchor
    exists yet — e.g. "style us as a couple for a reception" with nothing
    styled this session.

    ``compose_partner_look`` always needs an existing ``anchor_item`` to
    coordinate with; this function bootstraps one by composing a PRIMARY look
    first (occasion-driven, no seed_article_id — compose_outfit picks its own
    anchor) in the gender OPPOSITE ``partner_gender``, then feeds that look's
    own seed_item into the existing ``compose_partner_look`` for
    ``partner_gender``. This keeps the pair always mixed-gender regardless of
    how ``partner_gender`` was resolved (concrete "husband"/"wife" hint, or
    the "opposite" hint from "couple"/"his and hers" resolved against the
    brand's gender default — see graph.py's router_node and
    resolve_partner_gender) — no separate "always compose women's look first"
    special case is needed.

    Budget assumption (explicit, not silently chosen): ``budget_inr``, if
    given, is treated as a PER-PERSON cap applied INDEPENDENTLY to EACH look,
    not a combined couple total split in half. "under ₹15000" for a couple
    look most naturally reads as "each of us has up to ₹15000" — a 50/50 split
    has no principled basis (one look may legitimately need more slots filled
    than the other, e.g. jewellery-heavy ethnic womenswear vs. a men's kurta).

    Args:
        occasion_slug: shared occasion for both looks.
        partner_gender: the ALREADY-RESOLVED partner gender ("men"/"women") —
            i.e. resolve_partner_gender's output, not a raw PartnerIntent hint.
        budget_inr: optional PER-PERSON budget cap (see assumption above),
            forwarded unchanged to both compose_outfit and compose_partner_look.
        brand_gender_default: forwarded to the primary look's compose_outfit
            call (used only if its own gender ever resolves to "unisex").

    Returns:
        (primary_look, partner_look) — each shaped exactly like
        compose_outfit's return dict (look_id, seed_item, complements,
        outfit_rationale, empty_slots, suppressed_slots, occasion, gender,
        budget_total_inr). If the primary look's seed_item is None (no
        catalogue anchor at all for this occasion/gender/budget), partner_look
        is an honest empty result too — it is never composed against a look
        that doesn't exist.
    """
    primary_gender = "women" if partner_gender == "men" else "men"

    primary_look = compose_outfit(
        catalogue_df,
        retriever,
        occasion_slug=occasion_slug,
        gender=primary_gender,
        budget_inr=budget_inr,
        brand_gender_default=brand_gender_default,
    )
    if primary_look.get("seed_item") is None:
        reason = (
            f"No {primary_gender}'s items found in this catalogue for a "
            f"{occasion_slug.replace('_', ' ')} couple look."
        )
        return primary_look, _empty_partner_result(occasion_slug, partner_gender, reason)

    partner_look = compose_partner_look(
        catalogue_df,
        retriever,
        anchor_item=primary_look["seed_item"],
        occasion_slug=occasion_slug,
        partner_gender=partner_gender,
        budget_inr=budget_inr,
    )
    return primary_look, partner_look


# ── Deterministic "coordinated_with" text ───────────────────────────────────

# Human-readable formality/occasion register label used in the
# "coordinated_with" sentence — e.g. "at the same smart-casual level".
_FORMALITY_LEVEL_LABELS: dict[str, str] = {
    "casual": "casual",
    "smart_casual": "smart-casual",
    "office": "office-formal",
    "haldi": "haldi festive",
    "mehendi": "mehendi festive",
    "party_evening": "evening formal",
    "festive_puja": "festive",
    "wedding_guest": "wedding formal",
    "engagement": "engagement festive",
    "sangeet": "sangeet festive",
    "traditional_ethnic": "traditional festive",
    "reception": "reception formal",
}


def _formality_level_label(occasion_slug: str) -> str:
    return _FORMALITY_LEVEL_LABELS.get(occasion_slug, occasion_slug.replace("_", " "))


def build_coordinated_with_text(anchor_item: dict, partner_look: dict, occasion_slug: str) -> str:
    """Build the board-level "coordinated_with" sentence for a partner look.

    Deterministic (no LLM call) — every colour/type token referenced is taken
    directly from the anchor item or the composed partner look itself, so
    this is grounded by construction (see module docstring).

    Args:
        anchor_item: the session's current look/anchor item dict.
        partner_look: compose_partner_look's return value.
        occasion_slug: the occasion slug shared by both looks.

    Returns:
        A short sentence, e.g. "Coordinated with the rust dress — navy and
        cream complement it at the same smart-casual level."
    """
    anchor_colour = (anchor_item.get("colour") or "").lower().strip()
    anchor_type = (
        (anchor_item.get("product_type") or anchor_item.get("prod_name") or "item")
    ).lower().strip()
    anchor_desc = f"{anchor_colour} {anchor_type}".strip() if anchor_colour else anchor_type

    seed = partner_look.get("seed_item") or {}
    seed_colour = (seed.get("colour") or "").lower().strip()
    complement_colours = [
        (c.get("colour") or "").lower().strip()
        for c in (partner_look.get("complements") or [])
        if c.get("colour")
    ]

    ordered_colours: list[str] = []
    for colour in [seed_colour, *complement_colours]:
        if colour and colour not in ordered_colours:
            ordered_colours.append(colour)
    colour_text = " and ".join(ordered_colours[:2]) if ordered_colours else "a complementary palette"

    level = _formality_level_label(occasion_slug)
    return f"Coordinated with the {anchor_desc} — {colour_text} complement it at the same {level} level."
