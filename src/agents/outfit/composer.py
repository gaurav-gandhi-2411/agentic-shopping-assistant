from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass

import pandas as pd

from src.agents.outfit.body_type import body_type_score_delta
from src.agents.outfit.body_type import query_tokens as body_type_query_tokens
from src.agents.outfit.coherence import (
    colour_score,
    is_coherent_candidate,
    is_western_register_occasion,
)
from src.agents.outfit.occasions import ETHNIC_HEAVY, ETHNIC_ONLY, get_occasion
from src.agents.outfit.slots import (
    accessory_query_matches,
    classify_anchor,
    fabric_score_delta,
    gender_allowed,
    get_fill_slots,
    is_casual_marker_item,
    is_ethnic_item,
    is_gender_neutral_accessory,
    is_kids_item,
    is_multi_piece_set,
    is_novelty_item,
    is_slot_type_allowed,
    is_western_item,
)
from src.retrieval.hybrid_search import HybridRetriever, normalize_prod_name

logger = logging.getLogger(__name__)

# Transparent flywheel boost constant — documented here and in ADR.
# When >= FLYWHEEL_MIN_SIGNALS pairings observed, score × (1 + FLYWHEEL_ALPHA × positive_rate).
# Max +25% boost from conversion signal; 0 at cold-start.
FLYWHEEL_ALPHA: float = 0.25
FLYWHEEL_MIN_SIGNALS: int = 10

# Cross-store styling is a shipped product bar (Phase F).  The strict per-item gender gate
# (see gender_allowed) narrowed the complement candidate pool and, as a side effect, removed
# the accidental store diversity that used to come from unknown-gender rows in other stores.
# This is a SOFT preference — a multiplicative penalty, not a filter — so the best same-store
# candidate still wins when no other-store candidate is within striking distance.  0.85 means
# a same-store candidate needs to score >~18% higher than the best other-store candidate to
# still be picked; near-tied candidates from a new store win instead.
STORE_DIVERSITY_PENALTY: float = 0.85


@dataclass
class PairingStat:
    """Conversion signal counts for one (anchor_class, fill_slot, occasion) triple.

    Injected by the flywheel phase; zero-valued at cold-start.
    """

    add_the_look: int = 0
    thumbs_up: int = 0
    thumbs_down: int = 0
    add_single_only: int = 0


def _infer_gender(item: dict, brand_gender_default: str) -> str:
    """Infer gender from item's explicit gender field, then index_group_name, then brand default."""
    # Check explicit gender column first (written at index build time)
    derived = (item.get("gender") or "").lower()
    if derived in ("men", "women"):
        return derived
    # Fall back to index_group_name / department
    ig = (item.get("index_group_name") or item.get("department") or "").lower()
    if "menswear" in ig or ig == "men":
        return "men"
    if "ladieswear" in ig or "women" in ig or "ladies" in ig:
        return "women"
    return brand_gender_default or "women"


def compose_outfit(
    catalogue_df: pd.DataFrame,
    retriever: HybridRetriever,
    *,
    seed_article_id: str | None = None,
    occasion_slug: str = "casual",
    gender: str = "women",
    budget_inr: float | None = None,
    pairing_stats: dict | None = None,  # {(anchor_cat, fill_cat, occasion): PairingStat}
    brand_gender_default: str = "women",
    owned_anchor: bool = False,
    exclude_ids: set[str] | None = None,
    body_type: str | None = None,
    body_modifiers: list[str] | None = None,
) -> dict:
    """Compose an outfit for a given occasion, optionally anchored to a seed item.

    When seed_article_id is None, finds the best anchor from the catalogue for
    the occasion using a retrieval query, then fills slots around it.

    Args:
        owned_anchor: When True, the resolved seed item is stamped ``_owned=True``
            (e.g. the seed came from a user-uploaded photo of an item they already
            own). Owned seeds are never for sale — ``build_cart_action`` excludes
            them from cart/link/total machinery, and ``ItemSummary.from_agent_item``
            suppresses their ``pdp_url``. Complements are never owned.
        exclude_ids: article_ids that must NEVER be picked as a complement, in
            addition to the seed itself. Used by ``compose_biased_look`` so a
            variant hard-excludes the BASE look's complement ids — guaranteeing
            a visibly different look whenever the candidate pool has an
            alternative, instead of relying only on score nudges to change the
            winner (Phase B Part 1: "guaranteed-distinct variants").
        body_type: P3 — one of src.agents.outfit.body_type.BASE_SHAPE_SLUGS, or
            None. Opt-in bias only (never a filter): nudges complement scoring
            via body_type_score_delta and augments slot/anchor retrieval
            queries via body_type.query_tokens. None (default) is a full no-op.
        body_modifiers: P3 — list of MODIFIER_SLUGS ("petite"/"tall"/
            "plus_size"), or None. Composes with body_type (union of
            recommend/deprioritize keyword sets — see body_type_score_delta).

    Returns a dict with: look_id, seed_item, complements, outfit_rationale,
    empty_slots, suppressed_slots, occasion, gender, budget_total_inr.
    """
    look_id = str(uuid.uuid4())

    # ── Step 1: resolve seed item ───────────────────────────────────────────
    if seed_article_id:
        indexed = catalogue_df.set_index("article_id")
        if seed_article_id not in indexed.index:
            return _empty_result(look_id, occasion_slug, gender, "Item not found.")
        row = indexed.loc[seed_article_id]
        facets = row["facets"] if isinstance(row.get("facets"), dict) else {}
        seed_item = _row_to_item(seed_article_id, row, facets, role="seed")
    else:
        # Occasion-driven entry: retrieve an anchor appropriate for the occasion.
        # Push the gender into the retriever filter (same fix as _find_best_candidate
        # below) so the candidate window itself is already single-gender — the
        # gender_allowed() re-check right below is now belt-and-suspenders, not the
        # only gate.  Skipped for "unisex" (no single-gender filter makes sense).
        anchor_query = _anchor_query_for_occasion(occasion_slug, gender)
        _bt_tokens = body_type_query_tokens(body_type, body_modifiers)
        if _bt_tokens:
            anchor_query = f"{anchor_query} {_bt_tokens}"
        anchor_filters = {"gender": gender} if gender in ("men", "women") else None
        candidates = retriever.search(anchor_query, top_k=10, filters=anchor_filters)
        # Filter by occasion coherence AND gender compatibility. S5 fix: also
        # reject juniors/girls/boys/kids items as a look ANCHOR — the same
        # catalogue mislabeling that lets them fill complement slots (see
        # is_kids_item) would otherwise let one become the seed itself.
        valid = [
            c
            for c in candidates
            if _anchor_matches_occasion(c, occasion_slug)
            and gender_allowed(
                (c.get("gender") or "unknown").lower(),
                gender if gender != "unisex" else "unisex",
            )
            and not is_kids_item(c.get("prod_name") or c.get("display_name") or "")
        ]
        # Budget gate (live-proven bug: "I'm pear-shaped, sangeet look under
        # ₹8000" boarded a ₹9,900 lehenga ANCHOR — this filter previously
        # gated occasion/gender/kids only, never price, so `valid[0]` could
        # pick an over-budget item and the look violated the cap before a
        # single complement was even considered). ONLY applied here — the
        # explicit seed_article_id path above (a user's "Style this <item>"
        # choice) must never be budget-rejected; the user already chose that
        # exact item, so only complements get budget-squeezed for that path,
        # same as before this fix.
        _pre_budget_valid = valid
        if budget_inr is not None:
            valid = [c for c in valid if (c.get("price_inr") or 0.0) <= budget_inr]
        if not valid:
            if budget_inr is not None and _pre_budget_valid:
                # Occasion/gender-valid anchors existed but ALL were over budget —
                # honest budget-specific message, never a silent fall-back to an
                # over-budget anchor (mirrors _no_anchor_msg's style below).
                _no_budget_anchor_msg = (
                    f"No {gender} {occasion_slug.replace('_', ' ')} pieces within "
                    f"₹{budget_inr:,.0f} in our partner stores yet — try a higher budget."
                )
                return _empty_result(look_id, occasion_slug, gender, _no_budget_anchor_msg)
            _no_anchor_msg = (
                f"No {gender} items found in this catalogue for a "
                f"{occasion_slug.replace('_', ' ')} look. "
                f"Try a catalogue that stocks {gender}'s fashion."
            )
            return _empty_result(look_id, occasion_slug, gender, _no_anchor_msg)
        seed_item = valid[0]
        seed_item["_role"] = "seed"
        seed_article_id = seed_item["article_id"]

    if owned_anchor:
        seed_item["_owned"] = True

    # Resolve gender from seed item if not explicitly provided
    effective_gender = (
        gender if gender != "unisex" else _infer_gender(seed_item, brand_gender_default)
    )

    anchor_product_type = seed_item.get("product_type") or ""
    anchor_name = seed_item.get("prod_name") or seed_item.get("display_name") or ""
    anchor_class = classify_anchor(anchor_product_type, anchor_name)
    anchor_colour = (seed_item.get("colour") or "").lower()

    fill_slots = get_fill_slots(
        anchor_class, effective_gender, occasion_slug, body_type, body_modifiers
    )

    # ── Step 2: fill each slot ──────────────────────────────────────────────
    complements: list[dict] = []
    empty_slots: list[str] = []
    suppressed_slots: list[dict] = []
    seen_ids: set[str] = {seed_article_id} | (exclude_ids or set())
    seen_prod_colour: set[tuple[str, str]] = set()
    seen_prod_colour.add((normalize_prod_name(anchor_name), anchor_colour))

    # Stores already represented in this look (seed + chosen complements so far).  Fed to
    # _find_best_candidate so it can apply the soft STORE_DIVERSITY_PENALTY.
    seen_stores: set[str] = set()
    seed_store = seed_item.get("store")
    if seed_store:
        seen_stores.add(str(seed_store).lower())

    # "Owned anchor" budget fix: an owned seed (the user's own garment, uploaded via
    # photo) is never for sale, so its price must NOT count against the user's stated
    # budget — the budget buys complements only. A non-owned seed still counts from
    # the start, as before.
    running_total = 0.0 if owned_anchor else (seed_item.get("price_inr") or 0.0)

    for slot_spec in fill_slots:
        candidate = _find_best_candidate(
            query=slot_spec.search_query,
            slot_name=slot_spec.slot_name,
            occasion_slug=occasion_slug,
            gender=effective_gender,
            anchor_colour=anchor_colour,
            seen_ids=seen_ids,
            seen_prod_colour=seen_prod_colour,
            retriever=retriever,
            budget_remaining=budget_inr - running_total if budget_inr is not None else None,
            pairing_stats=pairing_stats,
            anchor_class=anchor_class,
            seen_stores=seen_stores,
            body_type=body_type,
            body_modifiers=body_modifiers,
        )
        if candidate:
            candidate["_slot"] = slot_spec.slot_name
            # RED 1a/1e/B4a/B4b: without this, ItemSummary.slot_role is None for every
            # complement (api/schemas.py reads item["_role"]) and the frontend OutfitBoard
            # filters complements out of the rendered look, showing only the seed card.
            candidate["_role"] = "complement"
            complements.append(candidate)
            seen_ids.add(candidate["article_id"])
            seen_prod_colour.add(
                (
                    normalize_prod_name(
                        candidate.get("prod_name") or candidate.get("display_name") or ""
                    ),
                    (candidate.get("colour") or "").lower(),
                )
            )
            running_total += candidate.get("price_inr") or 0.0
            candidate_store = candidate.get("store")
            if candidate_store:
                seen_stores.add(str(candidate_store).lower())
        else:
            # Honest slot suppression (Phase B Part 1): no valid candidate survived
            # the gender/slot-type/coherence/budget gates for this slot.  We do NOT
            # fill it cross-gender or with an off-vocabulary item — record why it's
            # missing so the frontend can show an honest note instead of silently
            # dropping the slot.  A 3-item correct look beats a 5-item wrong one.
            suppressed_slots.append(
                {"slot": slot_spec.slot_name, "reason": _suppression_reason(
                    slot_spec.slot_name, effective_gender,
                )}
            )
            if slot_spec.required:
                empty_slots.append(slot_spec.slot_name)

    # ── Step 3: build rationale ─────────────────────────────────────────────
    comp_names = [c.get("display_name") or c.get("prod_name") for c in complements]
    seed_display = seed_item.get("display_name") or seed_item.get("prod_name") or "item"
    if comp_names:
        rationale = (
            f"**{occasion_slug.replace('_', ' ').title()} look** — "
            f"Styled **{seed_display}** with {', '.join(comp_names)}."
        )
    else:
        rationale = f"Showing **{seed_display}** — no complementary items found for this occasion."

    budget_total = running_total if running_total > 0 else None

    logger.debug(
        "[composer] look_id=%s occasion=%s gender=%s anchor_class=%s slots=%s empty=%s budget_total=%s",
        look_id,
        occasion_slug,
        effective_gender,
        anchor_class,
        [c["_slot"] for c in complements],
        empty_slots,
        budget_total,
    )

    return {
        "look_id": look_id,
        "seed_item": seed_item,
        "complements": complements,
        "outfit_rationale": rationale,
        "empty_slots": empty_slots,
        "suppressed_slots": suppressed_slots,
        "occasion": occasion_slug,
        "gender": effective_gender,
        "budget_total_inr": budget_total,
    }


def _suppression_reason(slot_name: str, gender: str) -> str:
    """Return a short, honest, user-facing reason a slot was suppressed (no
    candidate survived the gender/slot-type/coherence/budget gates).

    Deliberately generic rather than quoting a specific inventory count — we
    don't have a live per-store count at request time, and a wrong number would
    be worse than a plain "not yet" statement.
    """
    labels = {
        "footwear": "footwear",
        "outerwear": "outerwear",
        "accessory": "accessories",
        "bottom": "bottoms",
        "top": "tops",
    }
    label = labels.get(slot_name, slot_name)
    return f"No {gender}'s {label} in our partner stores yet"


def compose_outfit_variants(
    catalogue_df: pd.DataFrame,
    retriever: HybridRetriever,
    *,
    seed_article_id: str | None = None,
    occasion_slug: str = "casual",
    gender: str = "women",
    budget_inr: float | None = None,
    pairing_stats: dict | None = None,
    brand_gender_default: str = "women",
    owned_anchor: bool = False,
    body_type: str | None = None,
    body_modifiers: list[str] | None = None,
) -> list[dict]:
    """Compose 2-3 look variants around the same seed and occasion.

    Variant 0 (Base):  standard compose_outfit output — unchanged behaviour.
    Variant 1 (Colour story): biases complement colour selection toward an
        alternate palette while remaining coherent and occasion-correct.
    Variant 2 (Dressier/Lighter): shifts the colour-harmony scoring to prefer
        higher-formality complements for Western/casual occasions, or lower-
        formality for ethnic/formal occasions — always within the same occasion
        family, never changing the occasion slug or gender.

    Coherence guarantees:
    - All variants run through the SAME gender gates, occasion gates,
      no-duplicate-slot, and no-dead-image rules as the base.
    - If a distinct coherent variant cannot be produced, returns fewer items
      (1 or 2) rather than shipping an incoherent look.
    - Deterministic: seed=42 applied via score jitter ordering.

    Each returned look dict carries: look_id, seed_item, complements,
    outfit_rationale, empty_slots, occasion, gender, budget_total_inr.
    The `rationale` field is populated by the caller after LLM generation.

    Args:
        catalogue_df:        Catalogue DataFrame.
        retriever:           HybridRetriever instance.
        seed_article_id:     Anchor item; None triggers occasion-driven selection.
        occasion_slug:       Occasion slug string.
        gender:              "men" | "women" | "unisex".
        budget_inr:          Optional budget cap in INR.
        pairing_stats:       Flywheel pairing stats dict (may be None).
        brand_gender_default: Gender default for the brand catalogue.
        owned_anchor:        When True, the seed item in EVERY variant is stamped
            ``_owned=True`` (see ``compose_outfit`` docstring). Complements are
            never owned.
        body_type:           P3 — see ``compose_outfit`` docstring. Applied to
            every variant (base + biased).
        body_modifiers:      P3 — see ``compose_outfit`` docstring.

    Returns:
        List of 1–3 look dicts.  Always non-empty (falls back to base-only).
    """
    # Variant 0: base look — standard behaviour
    base = compose_outfit(
        catalogue_df,
        retriever,
        seed_article_id=seed_article_id,
        occasion_slug=occasion_slug,
        gender=gender,
        budget_inr=budget_inr,
        pairing_stats=pairing_stats,
        brand_gender_default=brand_gender_default,
        owned_anchor=owned_anchor,
        body_type=body_type,
        body_modifiers=body_modifiers,
    )
    base["variant_label"] = "Base"

    # If the base look has no seed, we can't produce meaningful variants
    if base.get("seed_item") is None:
        return [base]

    variants: list[dict] = [base]

    # S6 fix: sequential composition. Each subsequent variant must exclude
    # every complement id already used by an EARLIER accepted variant (not
    # just the base's) — otherwise two independently-biased variants that both
    # only have ONE non-base alternative per slot converge on the identical
    # complement set (live-proven: "Colour Palette" and "Dressed Up" landing on
    # the same navy blazer + white bag once the base's own items were the only
    # ones excluded from each). `accumulated_ids` tracks every complement id
    # used by any variant accepted so far, seeded with the base's own ids.
    accumulated_ids: set[str] = {c["article_id"] for c in (base.get("complements") or [])}

    # ── Variant 1: Alternate colour story ───────────────────────────────────
    # Build a biased candidate selector that prefers a different colour palette.
    # We wrap the retriever with a scoring override that flips colour preferences.
    alt_colour_look = compose_biased_look(
        catalogue_df=catalogue_df,
        retriever=retriever,
        base_look=base,
        seed_article_id=seed_article_id,
        occasion_slug=occasion_slug,
        gender=gender,
        budget_inr=budget_inr,
        pairing_stats=pairing_stats,
        brand_gender_default=brand_gender_default,
        bias_mode="alternate_colour",
        owned_anchor=owned_anchor,
        extra_exclude_ids=accumulated_ids,
        body_type=body_type,
        body_modifiers=body_modifiers,
    )
    if alt_colour_look and _is_distinct_look(alt_colour_look, variants):
        alt_colour_look["variant_label"] = "Colour story"
        variants.append(alt_colour_look)
        accumulated_ids |= {c["article_id"] for c in (alt_colour_look.get("complements") or [])}

    # ── Variant 2: Dressier or Lighter lean ──────────────────────────────────
    # Shift formality within the same occasion family.
    formality_look = compose_biased_look(
        catalogue_df=catalogue_df,
        retriever=retriever,
        base_look=base,
        seed_article_id=seed_article_id,
        occasion_slug=occasion_slug,
        gender=gender,
        budget_inr=budget_inr,
        pairing_stats=pairing_stats,
        brand_gender_default=brand_gender_default,
        bias_mode="formality_shift",
        owned_anchor=owned_anchor,
        extra_exclude_ids=accumulated_ids,
        body_type=body_type,
        body_modifiers=body_modifiers,
    )
    if formality_look and _is_distinct_look(formality_look, variants):
        # Label: ethnic/formal occasions get "Lighter"; western/casual get "Dressier"
        from src.agents.outfit.occasions import ETHNIC_HEAVY, ETHNIC_ONLY, get_occasion
        occ = get_occasion(occasion_slug)
        is_formal_ethnic = occ.ethnic_lean in (ETHNIC_HEAVY, ETHNIC_ONLY) or occ.formality >= 4
        formality_look["variant_label"] = "Lighter" if is_formal_ethnic else "Dressier"
        variants.append(formality_look)
        accumulated_ids |= {c["article_id"] for c in (formality_look.get("complements") or [])}

    return variants


def _is_distinct_look(look: dict, others: dict | list[dict]) -> bool:
    """Return True if `look` has at least one complement not present in ANY
    look in `others`.

    `others` may be a single look dict (legacy 2-look comparison) or a list of
    look dicts — checked against EVERY previously-accepted variant (S6 fix),
    not just the base, so a third variant that happens to converge on the same
    complement set as an earlier NON-BASE variant is correctly rejected too
    (a duplicate that matches variant 1 but differs from the base was
    previously accepted as "distinct" since only the base was checked).
    """
    other_looks = [others] if isinstance(others, dict) else others
    other_ids: set[str] = set()
    for o in other_looks:
        other_ids |= {c["article_id"] for c in (o.get("complements") or [])}
    look_ids = {c["article_id"] for c in (look.get("complements") or [])}
    return bool(look_ids - other_ids)


def compose_biased_look(
    *,
    catalogue_df: pd.DataFrame,
    retriever: HybridRetriever,
    base_look: dict,
    seed_article_id: str | None,
    occasion_slug: str,
    gender: str,
    budget_inr: float | None,
    pairing_stats: dict | None,
    brand_gender_default: str,
    bias_mode: str,
    owned_anchor: bool = False,
    extra_exclude_ids: set[str] | None = None,
    body_type: str | None = None,
    body_modifiers: list[str] | None = None,
) -> dict | None:
    """Compose a variant look using a biased retriever wrapper.

    bias_mode options:
    - "alternate_colour": prefers complements whose colour differs from the base
      complements (shifts the colour story without abandoning coherence).
    - "formality_shift": nudges scoring toward embellished/formal items for
      casual occasions, or lightweight/casual items for formal ethnic occasions.
    - "ethnic_shift": nudges scoring toward ethnic garment types/keywords (kurta,
      lehenga, saree, dupatta, ...) and away from purely western items — used for
      "make this look more ethnic" refinement turns. Gender/occasion/coherence
      gates are unchanged; this only re-scores candidates within the same gates.

    Args:
        extra_exclude_ids: additional article_ids to hard-exclude on top of the
            base look's own complement ids (S6 fix) — the caller passes every
            complement id used by PRIOR variants in the same
            compose_outfit_variants() call, so a third/later variant can never
            converge on a complement set identical to an earlier NON-BASE
            variant. Ignored for bias_mode="ethnic_shift" (see the exclude_ids
            comment below — that mode never hard-excludes).

    Returns None if the variant cannot be composed coherently, or if it would
    be identical to the base.
    """
    # Guaranteed-distinct variants (Phase B Part 1): for the two FIXED variants
    # ("alternate_colour", "formality_shift"), hard-exclude the base look's OWN
    # complement article_ids up front, rather than relying only on the bias
    # score nudges to (maybe) promote a different winner. Whenever the candidate
    # pool has ANY alternative for a slot, this guarantees the variant differs
    # from the base for that slot — _is_distinct_look (the caller) still gates on
    # "differs from base" so a too-thin pool honestly yields fewer variants
    # instead of a padded duplicate.
    #
    # "ethnic_shift" is NOT a fixed variant — it's the ad-hoc "make this look
    # more ethnic" refinement bias (graph.py). Its job is to shift STYLE within
    # the same gates, not to guarantee a different item; when the pool's only
    # gender-valid candidate for a slot is the one already in the base look,
    # ethnic_shift must still return it rather than suppressing the slot (see
    # TestComposeBiasedLookEthnicShiftGating.test_ethnic_shift_never_bypasses_gender_gate).
    exclude_ids = (
        {c["article_id"] for c in (base_look.get("complements") or [])}
        if bias_mode != "ethnic_shift"
        else set()
    )
    if bias_mode != "ethnic_shift" and extra_exclude_ids:
        exclude_ids = exclude_ids | extra_exclude_ids
    biased_retriever = _BiasedRetriever(
        retriever=retriever,
        bias_mode=bias_mode,
        base_complement_colours={
            (c.get("colour") or "").lower()
            for c in (base_look.get("complements") or [])
        },
        occasion_slug=occasion_slug,
        gender=gender,
        # Deterministic seed for reproducibility
        _seed=42,
    )
    try:
        look = compose_outfit(
            catalogue_df,
            biased_retriever,  # type: ignore[arg-type]
            seed_article_id=seed_article_id,
            occasion_slug=occasion_slug,
            gender=gender,
            budget_inr=budget_inr,
            pairing_stats=pairing_stats,
            brand_gender_default=brand_gender_default,
            owned_anchor=owned_anchor,
            exclude_ids=exclude_ids,
            body_type=body_type,
            body_modifiers=body_modifiers,
        )
        if look.get("seed_item") is None:
            return None
        return look
    except Exception as exc:
        logger.debug("[composer] variant compose_with_bias failed (%s): %s", bias_mode, exc)
        return None


# Western hint words that identify which slot a get_fill_slots() search_query
# targets (see slots.py's default/western_bottom/outerwear/western_one_piece
# branches) so ethnic_shift can append the matching ethnic vocabulary. These are
# intentionally kept in sync with the literal query strings in slots.py.
_BOTTOM_HINT_WORDS: frozenset[str] = frozenset({"trousers", "jeans", "skirt", "jeggings"})
_OUTERWEAR_HINT_WORDS: frozenset[str] = frozenset({"jacket", "blazer", "coat", "cardigan"})
_FOOTWEAR_HINT_WORDS: frozenset[str] = frozenset(
    {"sneakers", "flats", "heels", "loafers", "shoes", "wedges", "pumps"}
)
_ACCESSORY_HINT_WORDS: frozenset[str] = frozenset(
    {"bag", "handbag", "sling", "earrings", "belt", "watch", "cap"}
)
_TOP_HINT_WORDS: frozenset[str] = frozenset({"shirt", "blouse"})


def _ethnic_shift_query(query: str, gender: str) -> str:
    """Augment a western-leaning slot search_query with ethnic-leaning terms.

    get_fill_slots() (src/agents/outfit/slots.py) derives each slot's search_query
    purely from the seed item's anchor_class, never from occasion or bias mode —
    so a western-anchored look (e.g. a blouse) always retrieves a western-only
    candidate pool. _BiasedRetriever's ethnic_shift re-scoring can only promote
    candidates that already exist in that pool, so without this augmentation the
    pool never contains an ethnic item to promote and "make this look more
    ethnic" silently yields an all-western result.

    Detects which slot the query targets via its known western hint words and
    appends the matching ethnic vocabulary, gendered the same way the ethnic
    branches of slots.py already gender their own queries. Queries that don't
    match any known western hint (already-ethnic queries, e.g. slots.py's own
    "kurta ethnic top") are returned unchanged.
    """
    words = set(query.lower().split())
    is_men = gender.lower() == "men"
    extra_terms: list[str] = []
    if words & _BOTTOM_HINT_WORDS:
        extra_terms.append(
            "churidar pyjama ethnic bottom"
            if is_men
            else "palazzo churidar salwar sharara ethnic bottom"
        )
    if words & _OUTERWEAR_HINT_WORDS:
        extra_terms.append(
            "nehru jacket waistcoat ethnic waistcoat"
            if is_men
            else "nehru jacket ethnic jacket shrug"
        )
    if words & _FOOTWEAR_HINT_WORDS:
        extra_terms.append(
            "mojaris juttis kolhapuris ethnic footwear"
            if is_men
            else "juttis kolhapuris ethnic sandals"
        )
    if words & _ACCESSORY_HINT_WORDS:
        extra_terms.append(
            "pocket square safa ethnic accessory"
            if is_men
            else "dupatta jhumka ethnic jewellery"
        )
    if words & _TOP_HINT_WORDS:
        extra_terms.append("kurta ethnic top" if is_men else "kurta kurti ethnic top")

    if not extra_terms:
        return query
    return query + " " + " ".join(extra_terms)


class _BiasedRetriever:
    """Thin wrapper around HybridRetriever that re-scores candidates.

    Does NOT bypass any coherence/gender/occasion gate — those still run in
    compose_outfit's _find_best_candidate.  Only adjusts the retrieval score
    so a different set of items floats to the top.

    Live-proven bug: for bias_mode="ethnic_shift" this re-scoring alone was not
    enough. get_fill_slots() (src/agents/outfit/slots.py) derives each slot's
    search_query purely from the seed item's anchor_class, never from occasion
    or bias mode — so a western-anchored look (e.g. a blouse) always retrieves
    a western-only candidate POOL (trousers/jeans, blazers, sneakers, handbags),
    even when "make this look more ethnic" is asking for an ethnic re-score.
    Re-scoring can only promote what already exists in the pool; an all-western
    pool always yields an all-western result regardless of bias. `search()`
    below augments the query text itself for ethnic_shift so ethnic candidates
    (kurta/palazzo/dupatta/jutti/...) actually exist in-pool to be promoted.
    """

    # Colours preferred for "alternate colour story" variant.  "dark turquoise",
    # "dark purple" and "indigo" were removed (Phase B Part 1 catalogue colour
    # audit — data/processed/unified/catalogue.parquet has no rows with those
    # colour_group_name values, so they could never match anything) and replaced
    # with colours confirmed present in the real catalogue.
    _ALTERNATE_PALETTE: frozenset[str] = frozenset({
        "gold", "turquoise", "purple", "coral", "teal", "olive", "maroon",
        "dark orange", "dark pink", "violet",
        "rust", "navy blue", "mustard", "burgundy",
    })

    # Embellishment keywords that indicate a dressier item
    _DRESSY_KEYWORDS: frozenset[str] = frozenset({
        "embroidered", "embellished", "sequin", "zari", "bridal",
        "mirror work", "beaded", "resham", "gota", "formal",
    })

    # Lightweight/casual keywords for the "lighter" direction
    _CASUAL_KEYWORDS: frozenset[str] = frozenset({
        "cotton", "floral", "casual", "printed", "lightweight", "linen",
        "everyday", "relaxed",
    })

    def __init__(
        self,
        retriever: HybridRetriever,
        bias_mode: str,
        base_complement_colours: set[str],
        occasion_slug: str,
        gender: str = "women",
        _seed: int = 42,
    ) -> None:
        self._retriever = retriever
        self._bias_mode = bias_mode
        self._base_colours = base_complement_colours
        self._occasion_slug = occasion_slug
        self._gender = gender
        self._seed = _seed

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[dict]:
        """Retrieve candidates and apply bias scoring."""
        effective_query = query
        if self._bias_mode == "ethnic_shift":
            effective_query = _ethnic_shift_query(query, self._gender)
        candidates = self._retriever.search(effective_query, top_k=top_k * 2, filters=filters)
        rescored: list[tuple[float, dict]] = []
        for item in candidates:
            base_score = item.get("score") or 0.5
            bias = self._bias_score(item)
            rescored.append((base_score + bias, item))

        # Stable sort: deterministic because we add a tiny position-based tiebreaker
        rescored.sort(key=lambda t: t[0], reverse=True)
        return [item for _, item in rescored[:top_k]]

    def _bias_score(self, item: dict) -> float:
        """Return a score adjustment (positive = prefer, negative = deprioritise)."""
        colour = (item.get("colour") or "").lower()
        combined = (
            (item.get("prod_name") or "") + " " + (item.get("detail_desc") or "")
        ).lower()

        if self._bias_mode == "alternate_colour":
            # Prefer alternate palette colours; deprioritise colours already in base
            if colour in self._base_colours:
                return -0.15
            if colour in self._ALTERNATE_PALETTE:
                return 0.15
            return 0.0

        if self._bias_mode == "formality_shift":
            from src.agents.outfit.occasions import ETHNIC_HEAVY, ETHNIC_ONLY, get_occasion
            occ = get_occasion(self._occasion_slug)
            is_formal = occ.ethnic_lean in (ETHNIC_HEAVY, ETHNIC_ONLY) or occ.formality >= 4
            if is_formal:
                # Shift toward lighter/casual complements
                if any(kw in combined for kw in self._CASUAL_KEYWORDS):
                    return 0.12
                if any(kw in combined for kw in self._DRESSY_KEYWORDS):
                    return -0.08
            else:
                # Shift toward dressier complements
                if any(kw in combined for kw in self._DRESSY_KEYWORDS):
                    return 0.12
                if any(kw in combined for kw in self._CASUAL_KEYWORDS):
                    return -0.08
            return 0.0

        if self._bias_mode == "ethnic_shift":
            # Prefer ethnic garment types/keywords; deprioritise purely western
            # items. Gender/occasion/coherence gates still run unchanged in
            # _find_best_candidate — this only re-scores within those gates.
            product_type = item.get("product_type") or ""
            prod_name = item.get("prod_name") or ""
            if is_ethnic_item(product_type, prod_name):
                return 0.15
            if is_western_item(product_type, prod_name):
                return -0.1
            return 0.0

        return 0.0


# Phase B pool-underflow fallback: the single most unambiguous western
# tailored bottom PRODUCT_TYPE_NAME value in this catalogue's F1 vocabulary
# (src/catalogue/normalizer.py) — used as a catalogue-FACET filter (not a
# semantic query) so the retry retrieval pass is immune to whatever embedding-
# ranking dilution caused the first pass to come back empty/fully-gated.
_WESTERN_BOTTOM_FALLBACK_PRODUCT_TYPE: str = "trousers"


def _score_candidates(
    candidates: list[dict],
    *,
    query: str,
    slot_name: str,
    occasion_slug: str,
    gender: str,
    anchor_colour: str,
    seen_ids: set[str],
    seen_prod_colour: set[tuple[str, str]],
    budget_remaining: float | None,
    pairing_stats: dict | None,
    anchor_class: str,
    seen_stores: set[str] | None,
    neutral_fallback_ids: set[str],
    body_type: str | None = None,
    body_modifiers: list[str] | None = None,
) -> list[tuple[float, dict]]:
    """Run every hard gate + score every surviving candidate.

    Extracted from _find_best_candidate so the SAME gate/scoring logic can be
    re-run, unchanged, against a second (narrowed) retrieval pass — see the
    pool-underflow fallback in _find_best_candidate below.  Returns the same
    (score, item) tuple list _find_best_candidate previously built inline.

    body_type/body_modifiers: P3 — passed straight to body_type_score_delta,
    added to the score alongside fabric_score_delta (never a gate — every
    candidate that survives the hard gates above is still scored and
    returned; body type only nudges which one wins).
    """
    scored: list[tuple[float, dict]] = []
    for item in candidates:
        if item["article_id"] in seen_ids:
            continue
        item_key = (
            normalize_prod_name(item.get("prod_name") or item.get("display_name") or ""),
            (item.get("colour") or "").lower(),
        )
        if item_key in seen_prod_colour:
            continue
        # Per-item gender hard gate (belt-and-suspenders; coherence.py also checks
        # this; the retriever-level filter above already enforces it for the
        # common case). The narrow gender-neutral-accessory fallback above is the
        # ONLY way an "unknown"-gender item reaches this point.
        is_neutral_fallback_item = item["article_id"] in neutral_fallback_ids
        if not is_neutral_fallback_item and not gender_allowed(
            (item.get("gender") or "unknown").lower(), gender
        ):
            continue
        # Slot-type hard gate: reject candidates whose classified item-type
        # doesn't belong to this slot (e.g. bottom-classified trousers can never
        # fill an "accessory" slot — the live-proven "paperbag pants badged
        # Accessory" bug).
        item_pt = item.get("product_type") or ""
        item_name = item.get("prod_name") or item.get("display_name") or ""
        if not is_slot_type_allowed(slot_name, item_pt, item_name):
            continue
        # Multi-piece SET gate: a "Set" listing (e.g. "Anarkali Sharara Set",
        # a "Co-Ord Set") is a WHOLE OUTFIT, not a single garment — it must
        # never fill a single complement slot. Checked right after the
        # slot-type gate since it's a type-level rejection (product_type
        # alone can under-classify a set as a single ethnic/western bottom —
        # see is_multi_piece_set's docstring for the live-proven root cause).
        if is_multi_piece_set(item_pt, item_name):
            continue
        # Accessory vocabulary gate: within the "accessory" slot class, a
        # dupatta-seeking query must not accept a handbag and vice versa.
        if slot_name == "accessory" and not accessory_query_matches(query, item_pt, item_name):
            continue
        # Novelty/quality guard: conservative denylist rejects costume/novelty
        # items (e.g. "Luxury Piano Shape Statement Handbag") from any slot.
        if is_novelty_item(item_name):
            continue
        # S5 fix: reject juniors/girls/boys/kids items from ADULT look slots —
        # the catalogue's gender column mislabels many of these as "women"/
        # "men" (see is_kids_item docstring), so gender_allowed() alone is not
        # enough (live-proven: an office look's bottom slot filled with a
        # "M&H Juniors Girls ... Denim Skirts" item).
        if is_kids_item(item_name):
            continue
        # Phase B: hard formality gate — for occasion.formality >= 3 (office,
        # haldi, mehendi, party_evening, festive_puja, wedding_guest, engagement,
        # sangeet, traditional_ethnic, reception), reject any candidate carrying a casual/denim-
        # register marker regardless of slot_name. _default_bottom_query()
        # already drops "jeans"/"skirt" from the QUERY text for western
        # formal occasions, but that only shapes ranking — it does not stop a
        # casual item surfaced via other query terms from being ACCEPTED as
        # the best-scoring candidate (live-proven: office look's bottom slot
        # filled with "ONLY Women Blue Solid Denim Mini Skirts").
        if get_occasion(occasion_slug).formality >= 3 and is_casual_marker_item(item_name):
            continue
        if not is_coherent_candidate(
            item, occasion_slug, gender, slot_name, skip_gender_gate=is_neutral_fallback_item
        ):
            continue
        if budget_remaining is not None:
            price = item.get("price_inr") or 0.0
            if price > budget_remaining:
                continue

        base_score = item.get("score") or 0.5
        c_score = colour_score(item.get("colour") or "", anchor_colour, occasion_slug)
        fab_delta = fabric_score_delta(item, occasion_slug)
        bt_delta = body_type_score_delta(item, body_type, body_modifiers)
        fw_boost = _flywheel_boost(anchor_class, slot_name, occasion_slug, pairing_stats)

        final_score = (
            (base_score * 0.5 + c_score * 0.3 + 0.2) * (1.0 + fw_boost) + fab_delta + bt_delta
        )

        # Soft store-diversity preference: penalise (not exclude) candidates whose store is
        # already represented in the current look, so a near-tied item from a new store wins.
        item_store = str(item.get("store") or "").lower()
        if seen_stores and item_store and item_store in seen_stores:
            final_score *= STORE_DIVERSITY_PENALTY

        scored.append((final_score, item))
    return scored


def _find_best_candidate(
    *,
    query: str,
    slot_name: str,
    occasion_slug: str,
    gender: str,
    anchor_colour: str,
    seen_ids: set[str],
    seen_prod_colour: set[tuple[str, str]],
    retriever: HybridRetriever,
    budget_remaining: float | None,
    pairing_stats: dict | None,
    anchor_class: str,
    seen_stores: set[str] | None = None,
    body_type: str | None = None,
    body_modifiers: list[str] | None = None,
) -> dict | None:
    # Hard gender filter AT RETRIEVAL TIME (Phase B Part 1).  Previously this call
    # was unfiltered top_k=20, and gender was ONLY a post-hoc score gate below —
    # for a thin slot (e.g. women's footwear) an unfiltered top-20 window could be
    # 20/20 wrong-gender items, emptying the pool instead of ever widening it.
    # HybridRetriever's gender filter (src/retrieval/hybrid_search.py) ALSO
    # excludes "unknown"-gender rows when a gender filter is set — the same
    # conservative "never guess gender in" rule as gender_allowed() below, now
    # enforced on the retrieval window itself, not just on whatever slipped
    # through unfiltered.
    candidates = retriever.search(query, top_k=40, filters={"gender": gender})

    # Gender-neutral accessory fallback: fires ONLY when the gendered search
    # above returned nothing for an accessory slot.  Widens to an unfiltered
    # search and accepts ONLY unknown-gender candidates whose product_type/name
    # is a genuinely unisex accessory sub-type (sunglasses, belt, watch, cap) —
    # never a garment.  A women's footwear or a men's kurta slot returning empty
    # is NEVER filled this way; it falls through to honest suppression instead.
    neutral_fallback_ids: set[str] = set()
    if not candidates and slot_name == "accessory":
        widened = retriever.search(query, top_k=40)
        for item in widened:
            item_gender = (item.get("gender") or "unknown").lower()
            if item_gender == "unknown" and is_gender_neutral_accessory(
                item.get("product_type") or "", item.get("prod_name") or ""
            ):
                candidates.append(item)
                neutral_fallback_ids.add(item["article_id"])

    scored = _score_candidates(
        candidates,
        query=query,
        slot_name=slot_name,
        occasion_slug=occasion_slug,
        gender=gender,
        anchor_colour=anchor_colour,
        seen_ids=seen_ids,
        seen_prod_colour=seen_prod_colour,
        budget_remaining=budget_remaining,
        pairing_stats=pairing_stats,
        anchor_class=anchor_class,
        seen_stores=seen_stores,
        neutral_fallback_ids=neutral_fallback_ids,
        body_type=body_type,
        body_modifiers=body_modifiers,
    )

    # Phase B pool-underflow fallback (live-proven: "office look for women"
    # boarded with an EMPTY bottom slot on the deployed index — the real
    # retrieval pool for a western-register bottom slot can be dominated by
    # ethnic bottoms/sets/denim in some catalogue compositions even with the
    # register-shaped query text, and the western-register/set/casual-marker
    # gates above then correctly reject every candidate). Retry ONCE with an
    # explicit product_type_name FACET filter narrowed to the single most
    # unambiguous western tailored bottom type — this bypasses embedding/BM25
    # ranking dilution entirely (a catalogue-facet filter, not a semantic
    # query), so it finds genuine western trousers even when the first pass's
    # ranked top-40 window never included one. Scoped narrowly: bottom slot,
    # western-register occasion, AND only when the first pass produced ZERO
    # survivors — never fires to override a gate that correctly rejected a
    # non-empty pool for cause, and never relaxes any gate (the exact same
    # _score_candidates gates run again on the narrowed pool).
    if (
        not scored
        and slot_name == "bottom"
        and is_western_register_occasion(occasion_slug)
    ):
        narrowed = retriever.search(
            query,
            top_k=40,
            filters={"gender": gender, "product_type_name": _WESTERN_BOTTOM_FALLBACK_PRODUCT_TYPE},
        )
        if narrowed:
            logger.info(
                "[composer/pool-underflow] slot=%s occasion=%s first pass empty — "
                "retrying with product_type_name=%s (%d candidates)",
                slot_name, occasion_slug, _WESTERN_BOTTOM_FALLBACK_PRODUCT_TYPE, len(narrowed),
            )
            scored = _score_candidates(
                narrowed,
                query=query,
                slot_name=slot_name,
                occasion_slug=occasion_slug,
                gender=gender,
                anchor_colour=anchor_colour,
                seen_ids=seen_ids,
                seen_prod_colour=seen_prod_colour,
                budget_remaining=budget_remaining,
                pairing_stats=pairing_stats,
                anchor_class=anchor_class,
                seen_stores=seen_stores,
                neutral_fallback_ids=set(),
                body_type=body_type,
                body_modifiers=body_modifiers,
            )

    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def swap_slot_in_look(
    retriever: HybridRetriever,
    *,
    seed_item: dict,
    complements: list[dict],
    slot_name: str,
    occasion_slug: str,
    gender: str,
    exclude_article_ids: set[str] | None = None,
    budget_inr: float | None = None,
    pairing_stats: dict | None = None,
    body_type: str | None = None,
    body_modifiers: list[str] | None = None,
) -> dict | None:
    """Replace ONLY the complement occupying ``slot_name``, keeping the seed and
    every other complement in the current session look unchanged.

    Used by the "swap the {slot}" refinement turn (graph.py outfit_node) so a
    request like "swap the footwear in this look" never triggers a full
    re-compose — only the named slot's item changes.

    Args:
        retriever: HybridRetriever used to find alternative candidates.
        seed_item: the current look's seed item dict (kept as-is, including any
            ``_owned`` flag — never re-picked or re-tagged).
        complements: the current look's complement item dicts; each is expected
            to carry a ``_slot`` key (as stamped by ``compose_outfit``).
        slot_name: the slot to replace, e.g. "bottom", "top", "footwear",
            "outerwear", "accessory".
        exclude_article_ids: extra article_ids to never re-pick for the new slot
            item (the currently-shown item in that slot is always excluded too).
        budget_inr: optional total look budget; remaining budget for the new slot
            item is computed from the seed + all OTHER (unchanged) complements.
        pairing_stats: optional flywheel pairing stats (see ``compose_outfit``).

    Returns:
        A new look dict (same shape as ``compose_outfit``'s return) with only the
        named slot's complement replaced, or ``None`` when ``slot_name`` is not
        present in the current look, or no alternative candidate can be found —
        callers should respond honestly in that case rather than recomposing the
        whole look.
    """
    current = next((c for c in complements if c.get("_slot") == slot_name), None)
    if current is None:
        return None

    anchor_product_type = seed_item.get("product_type") or ""
    anchor_name = seed_item.get("prod_name") or seed_item.get("display_name") or ""
    anchor_class = classify_anchor(anchor_product_type, anchor_name)
    anchor_colour = (seed_item.get("colour") or "").lower()

    fill_slots = get_fill_slots(anchor_class, gender, occasion_slug, body_type, body_modifiers)
    slot_spec = next((s for s in fill_slots if s.slot_name == slot_name), None)
    if slot_spec is None:
        return None

    others = [c for c in complements if c is not current]

    seen_ids: set[str] = {seed_item["article_id"], current["article_id"]}
    seen_ids |= {c["article_id"] for c in others}
    seen_ids |= exclude_article_ids or set()

    seen_prod_colour: set[tuple[str, str]] = {(normalize_prod_name(anchor_name), anchor_colour)}
    for c in others:
        seen_prod_colour.add(
            (
                normalize_prod_name(c.get("prod_name") or c.get("display_name") or ""),
                (c.get("colour") or "").lower(),
            )
        )

    seen_stores: set[str] = set()
    seed_store = seed_item.get("store")
    if seed_store:
        seen_stores.add(str(seed_store).lower())
    for c in others:
        store = c.get("store")
        if store:
            seen_stores.add(str(store).lower())

    budget_remaining: float | None = None
    if budget_inr is not None:
        # Owned seeds never count toward the budget (mirrors compose_outfit).
        seed_price = 0.0 if seed_item.get("_owned") else (seed_item.get("price_inr") or 0.0)
        others_total = seed_price + sum(c.get("price_inr") or 0.0 for c in others)
        budget_remaining = budget_inr - others_total

    new_candidate = _find_best_candidate(
        query=slot_spec.search_query,
        slot_name=slot_spec.slot_name,
        occasion_slug=occasion_slug,
        gender=gender,
        anchor_colour=anchor_colour,
        seen_ids=seen_ids,
        seen_prod_colour=seen_prod_colour,
        retriever=retriever,
        budget_remaining=budget_remaining,
        pairing_stats=pairing_stats,
        anchor_class=anchor_class,
        seen_stores=seen_stores,
        body_type=body_type,
        body_modifiers=body_modifiers,
    )
    if new_candidate is None:
        return None

    new_candidate["_slot"] = slot_name
    new_candidate["_role"] = "complement"
    new_complements = [new_candidate if c is current else c for c in complements]

    running_total = 0.0 if seed_item.get("_owned") else (seed_item.get("price_inr") or 0.0)
    running_total += sum(c.get("price_inr") or 0.0 for c in new_complements)

    comp_names = [c.get("display_name") or c.get("prod_name") for c in new_complements]
    seed_display = seed_item.get("display_name") or seed_item.get("prod_name") or "item"
    rationale = (
        f"**{occasion_slug.replace('_', ' ').title()} look** — "
        f"Styled **{seed_display}** with {', '.join(comp_names)}."
    )

    return {
        "look_id": str(uuid.uuid4()),
        "seed_item": seed_item,
        "complements": new_complements,
        "outfit_rationale": rationale,
        "empty_slots": [],
        "occasion": occasion_slug,
        "gender": gender,
        "budget_total_inr": running_total if running_total > 0 else None,
    }


def _flywheel_boost(
    anchor_class: str,
    fill_slot: str,
    occasion_slug: str,
    pairing_stats: dict | None,
) -> float:
    """Return the transparent flywheel ranking boost for this (anchor_class, fill_slot, occasion).

    Cold-start (< FLYWHEEL_MIN_SIGNALS): returns 0.0.
    Warm: FLYWHEEL_ALPHA × positive_rate.
    total_signals = add_the_look + thumbs_up + thumbs_down + add_single_only.
    """
    if not pairing_stats:
        return 0.0
    stat = pairing_stats.get((anchor_class, fill_slot, occasion_slug))
    if stat is None:
        return 0.0
    total = stat.add_the_look + stat.thumbs_up + stat.thumbs_down + stat.add_single_only
    if total < FLYWHEEL_MIN_SIGNALS:
        return 0.0
    positive_rate = (stat.add_the_look + stat.thumbs_up) / total
    boost = FLYWHEEL_ALPHA * positive_rate
    logger.debug(
        "[flywheel] anchor=%s slot=%s occ=%s positive_rate=%.2f boost=%.3f signals=%d",
        anchor_class,
        fill_slot,
        occasion_slug,
        positive_rate,
        boost,
        total,
    )
    return boost


def _anchor_query_for_occasion(occasion_slug: str, gender: str) -> str:
    """Build a retrieval query to find an appropriate anchor for the occasion."""
    is_men = gender.lower() == "men"
    queries: dict[str, str] = {
        "sangeet": (
            "sherwani kurta embellished festive"
            if is_men
            else "sherwani kurta lehenga anarkali festive embellished"
        ),
        "haldi": "bright yellow marigold cotton kurta anarkali sharara lightweight floral daytime",
        "mehendi": "green mint floral lehenga sharara kurta set playful daytime",
        "reception": (
            "embellished glam lehenga gown sherwani bandhgala indo-western evening jewel tone"
        ),
        "engagement": "elegant pastel anarkali saree gown kurta bandhgala semi-formal",
        "festive_puja": (
            "kurta festive ethnic nehru jacket" if is_men else "kurta kurti anarkali festive ethnic"
        ),
        "wedding_guest": (
            "sherwani kurta wedding formal ethnic"
            if is_men
            else "lehenga anarkali saree wedding ethnic formal"
        ),
        "traditional_ethnic": "saree lehenga traditional ethnic",
        "party_evening": ("shirt formal party" if is_men else "dress evening party top formal"),
        "office": ("shirt formal office" if is_men else "top blouse formal shirt"),
        "smart_casual": ("shirt casual" if is_men else "top casual blouse"),
        "casual": ("shirt casual tshirt" if is_men else "blouse top casual women"),
    }
    return queries.get(occasion_slug, "top casual")


def _anchor_matches_occasion(item: dict, occasion_slug: str) -> bool:
    """Check if an item retrieved as anchor is coherent with the occasion."""
    from src.agents.outfit.occasions import OCCASIONS

    occ = OCCASIONS.get(occasion_slug)
    if occ is None:
        return True
    pt = item.get("product_type") or ""
    name = item.get("prod_name") or ""
    if occ.ethnic_lean == ETHNIC_ONLY and not is_ethnic_item(pt, name):
        return False
    # Prefer ethnic anchor for ethnic_heavy occasions too
    if occ.ethnic_lean == ETHNIC_HEAVY and not is_ethnic_item(pt, name):
        return False
    return True


def _safe_str(val: object) -> str:
    """Return str(val) unless val is None, float NaN, or the sentinel string 'nan'."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    s = str(val)
    return "" if s.lower() == "nan" else s


def _row_to_item(article_id: str, row: pd.Series, facets: dict, role: str) -> dict:
    return {
        "article_id": article_id,
        "display_name": _safe_str(row.get("display_name") or row.get("prod_name")),
        "prod_name": _safe_str(row.get("prod_name")),
        "colour": _safe_str(facets.get("colour_group_name") or row.get("colour_group_name")),
        "product_type": _safe_str(
            facets.get("product_type_name") or row.get("product_type_name")
        ),
        "department": _safe_str(facets.get("department_name") or row.get("department_name")),
        "index_group_name": _safe_str(row.get("index_group_name")),
        "detail_desc": _safe_str(row.get("detail_desc")),
        "image_url": row.get("image_url"),
        "score": 1.0,
        "price_inr": row.get("price_inr"),
        "pdp_handle": row.get("pdp_handle"),
        # Phase E: extract store so the seed item carries store context for deep-link
        # building (build_pdp_url) and cross-store cart detection.  Mirror the pattern
        # used by hybrid_search.py when populating complement candidates.
        "store": (
            str(row["store"])
            if "store" in row.index and row["store"] is not None
            else None
        ),
        "gender": str(row.get("gender") or "unknown").lower(),
        "_role": role,
    }


def _empty_result(look_id: str, occasion_slug: str, gender: str, reason: str) -> dict:
    return {
        "look_id": look_id,
        "seed_item": None,
        "complements": [],
        "outfit_rationale": reason,
        "empty_slots": [],
        "suppressed_slots": [],
        "occasion": occasion_slug,
        "gender": gender,
        "budget_total_inr": None,
    }
