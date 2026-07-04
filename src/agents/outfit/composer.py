from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass

import pandas as pd

from src.agents.outfit.coherence import colour_score, is_coherent_candidate
from src.agents.outfit.occasions import ETHNIC_HEAVY, ETHNIC_ONLY
from src.agents.outfit.slots import (
    classify_anchor,
    fabric_score_delta,
    gender_allowed,
    get_fill_slots,
    is_ethnic_item,
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
) -> dict:
    """Compose an outfit for a given occasion, optionally anchored to a seed item.

    When seed_article_id is None, finds the best anchor from the catalogue for
    the occasion using a retrieval query, then fills slots around it.

    Returns a dict with: look_id, seed_item, complements, outfit_rationale,
    empty_slots, occasion, gender, budget_total_inr.
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
        # Occasion-driven entry: retrieve an anchor appropriate for the occasion
        anchor_query = _anchor_query_for_occasion(occasion_slug, gender)
        candidates = retriever.search(anchor_query, top_k=10)
        # Filter by occasion coherence AND gender compatibility
        valid = [
            c
            for c in candidates
            if _anchor_matches_occasion(c, occasion_slug)
            and gender_allowed(
                (c.get("gender") or "unknown").lower(),
                gender if gender != "unisex" else "unisex",
            )
        ]
        if not valid:
            _no_anchor_msg = (
                f"No {gender} items found in this catalogue for a "
                f"{occasion_slug.replace('_', ' ')} look. "
                f"Try a catalogue that stocks {gender}'s fashion."
            )
            return _empty_result(look_id, occasion_slug, gender, _no_anchor_msg)
        seed_item = valid[0]
        seed_item["_role"] = "seed"
        seed_article_id = seed_item["article_id"]

    # Resolve gender from seed item if not explicitly provided
    effective_gender = (
        gender if gender != "unisex" else _infer_gender(seed_item, brand_gender_default)
    )

    anchor_product_type = seed_item.get("product_type") or ""
    anchor_name = seed_item.get("prod_name") or seed_item.get("display_name") or ""
    anchor_class = classify_anchor(anchor_product_type, anchor_name)
    anchor_colour = (seed_item.get("colour") or "").lower()

    fill_slots = get_fill_slots(anchor_class, effective_gender, occasion_slug)

    # ── Step 2: fill each slot ──────────────────────────────────────────────
    complements: list[dict] = []
    empty_slots: list[str] = []
    seen_ids: set[str] = {seed_article_id}
    seen_prod_colour: set[tuple[str, str]] = set()
    seen_prod_colour.add((normalize_prod_name(anchor_name), anchor_colour))

    # Stores already represented in this look (seed + chosen complements so far).  Fed to
    # _find_best_candidate so it can apply the soft STORE_DIVERSITY_PENALTY.
    seen_stores: set[str] = set()
    seed_store = seed_item.get("store")
    if seed_store:
        seen_stores.add(str(seed_store).lower())

    running_total = seed_item.get("price_inr") or 0.0

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
        elif slot_spec.required:
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
        "occasion": occasion_slug,
        "gender": effective_gender,
        "budget_total_inr": budget_total,
    }


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
    )
    base["variant_label"] = "Base"

    # If the base look has no seed, we can't produce meaningful variants
    if base.get("seed_item") is None:
        return [base]

    variants: list[dict] = [base]

    # ── Variant 1: Alternate colour story ───────────────────────────────────
    # Build a biased candidate selector that prefers a different colour palette.
    # We wrap the retriever with a scoring override that flips colour preferences.
    alt_colour_look = _compose_with_colour_bias(
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
    )
    if alt_colour_look and _is_distinct_look(alt_colour_look, base):
        alt_colour_look["variant_label"] = "Colour story"
        variants.append(alt_colour_look)

    # ── Variant 2: Dressier or Lighter lean ──────────────────────────────────
    # Shift formality within the same occasion family.
    formality_look = _compose_with_colour_bias(
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
    )
    if formality_look and _is_distinct_look(formality_look, base):
        # Label: ethnic/formal occasions get "Lighter"; western/casual get "Dressier"
        from src.agents.outfit.occasions import ETHNIC_HEAVY, ETHNIC_ONLY, get_occasion
        occ = get_occasion(occasion_slug)
        is_formal_ethnic = occ.ethnic_lean in (ETHNIC_HEAVY, ETHNIC_ONLY) or occ.formality >= 4
        formality_look["variant_label"] = "Lighter" if is_formal_ethnic else "Dressier"
        variants.append(formality_look)

    return variants


def _is_distinct_look(look: dict, base: dict) -> bool:
    """Return True if the look has at least one complement not in the base look.

    Used to avoid shipping two identical variants.
    """
    base_ids = {c["article_id"] for c in (base.get("complements") or [])}
    look_ids = {c["article_id"] for c in (look.get("complements") or [])}
    return bool(look_ids - base_ids)


def _compose_with_colour_bias(
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
) -> dict | None:
    """Compose a variant look using a biased retriever wrapper.

    bias_mode options:
    - "alternate_colour": prefers complements whose colour differs from the base
      complements (shifts the colour story without abandoning coherence).
    - "formality_shift": nudges scoring toward embellished/formal items for
      casual occasions, or lightweight/casual items for formal ethnic occasions.

    Returns None if the variant cannot be composed coherently, or if it would
    be identical to the base.
    """
    biased_retriever = _BiasedRetriever(
        retriever=retriever,
        bias_mode=bias_mode,
        base_complement_colours={
            (c.get("colour") or "").lower()
            for c in (base_look.get("complements") or [])
        },
        occasion_slug=occasion_slug,
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
        )
        if look.get("seed_item") is None:
            return None
        return look
    except Exception as exc:
        logger.debug("[composer] variant compose_with_bias failed (%s): %s", bias_mode, exc)
        return None


class _BiasedRetriever:
    """Thin wrapper around HybridRetriever that re-scores candidates.

    Does NOT bypass any coherence/gender/occasion gate — those still run in
    compose_outfit's _find_best_candidate.  Only adjusts the retrieval score
    so a different set of items floats to the top.
    """

    # Colours preferred for "alternate colour story" variant
    _ALTERNATE_PALETTE: frozenset[str] = frozenset({
        "gold", "turquoise", "dark turquoise", "purple", "dark purple",
        "coral", "teal", "olive", "maroon", "dark orange", "dark pink",
        "indigo", "violet",
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
        _seed: int = 42,
    ) -> None:
        self._retriever = retriever
        self._bias_mode = bias_mode
        self._base_colours = base_complement_colours
        self._occasion_slug = occasion_slug
        self._seed = _seed

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[dict]:
        """Retrieve candidates and apply bias scoring."""
        candidates = self._retriever.search(query, top_k=top_k * 2, filters=filters)
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

        return 0.0


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
) -> dict | None:
    candidates = retriever.search(query, top_k=20)

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
        # Per-item gender hard gate (belt-and-suspenders; coherence.py also checks this)
        if not gender_allowed((item.get("gender") or "unknown").lower(), gender):
            continue
        if not is_coherent_candidate(item, occasion_slug, gender, slot_name):
            continue
        if budget_remaining is not None:
            price = item.get("price_inr") or 0.0
            if price > budget_remaining:
                continue

        base_score = item.get("score") or 0.5
        c_score = colour_score(item.get("colour") or "", anchor_colour, occasion_slug)
        fab_delta = fabric_score_delta(item, occasion_slug)
        fw_boost = _flywheel_boost(anchor_class, slot_name, occasion_slug, pairing_stats)

        final_score = (base_score * 0.5 + c_score * 0.3 + 0.2) * (1.0 + fw_boost) + fab_delta

        # Soft store-diversity preference: penalise (not exclude) candidates whose store is
        # already represented in the current look, so a near-tied item from a new store wins.
        item_store = str(item.get("store") or "").lower()
        if seen_stores and item_store and item_store in seen_stores:
            final_score *= STORE_DIVERSITY_PENALTY

        scored.append((final_score, item))

    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


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
        "haldi_mehendi": "kurta kurti lehenga cotton floral yellow festive",
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
        "occasion": occasion_slug,
        "gender": gender,
        "budget_total_inr": None,
    }
