from __future__ import annotations

import functools
import logging
import math
import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.agents.outfit.body_type import body_type_score_delta
from src.agents.outfit.body_type import query_tokens as body_type_query_tokens
from src.agents.outfit.coherence import (
    colour_score,
    is_coherent_candidate,
    is_western_register_occasion,
)
from src.agents.outfit.joint import (
    AnchorPlan,
    JointTimeout,
    Option,
    SlotChoices,
    select_joint,
)
from src.agents.outfit.occasions import get_occasion
from src.agents.outfit.slots import (
    _FORMAL_ETHNIC_OCCASIONS,
    ETHNIC_BOTTOM_KEYWORDS,
    FOOTWEAR_KEYWORDS,
    NON_OUTFIT_ITEM_CLASSES,
    SLOT_ALLOWED_CLASSES,
    WESTERN_BOTTOM_KEYWORDS,
    accessory_query_matches,
    classify_anchor,
    classify_item,
    fabric_score_delta,
    gender_allowed,
    get_fill_slots,
    is_casual_marker_item,
    is_ethnic_item,
    is_gender_neutral_accessory,
    is_multi_piece_set,
    is_novelty_item,
    is_rugged_footwear_item,
    is_slot_type_allowed,
    is_western_item,
    split_accessory_query_by_family,
)
from src.catalogue.cleaning import (
    has_gender_text_conflict,
    is_fabric_bolt_text,
    is_kids_item,
    is_loungewear_text,
    is_occasion_merchandise_name,
    is_occasion_merchandise_type,
)
from src.config.stores import build_pdp_url
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

# Absurd price-outlier guard (2026-07-23): an UNBUDGETED look must never
# include a slot item priced far above the look's own hero (anchor) price.
# Live-proven bug: an "eid outfit for men" anchored on a ₹12,632 "Pastel
# Seafoam Embroidered Kurta Pajama" picked "THE SHEHENSHAH TRADITIONAL NEHRU
# WAISTCOAT" (store=rathore) at ₹3,19,999 for the outerwear slot — 25.3x the
# anchor price, pushing the look total to ₹3,33,979 for a demo user who gave
# no budget signal at all.
#
# A per-product-type percentile cap (the other option the spec considered)
# does NOT catch this: the offending row's product_type_name is mislabeled
# "Fashion" (not "waistcoat"), and that catalogue-wide bucket (7,628 rows,
# mixing ₹300 earrings with ₹12+ lakh luxury pieces) has its own p99 of
# ₹3,76,729 — the item sits INSIDE that bucket's "normal" range, so a
# type-relative cap is defeated by the same mislabeling that let the item
# through in the first place. Anchoring the cap to THIS LOOK's own hero price
# instead is robust to that mislabeling and mirrors _CHEAP_OUTLIER_FACTOR's
# established shape (graph.py) — a multiplier of the look's own price signal,
# never a hardcoded INR threshold.
#
# Calibrated well below the real repro's 25.3x ratio so ordinary pricier
# complements (a sherwani/bandhgala accent piece) still survive: verified
# against data/processed/unified/catalogue.parquet medians — sherwani
# ₹18,999 (~9.5x a ₹1,999-median kurta), footwear ₹2,400, dupatta ₹1,999,
# jewellery ₹12,370 — all comfortably under an 8x cap relative to a
# typical anchor. Skipped entirely (see compose_outfit) when the user gave
# an explicit budget — budget_remaining's own gate already bounds price
# there, and an explicit high budget is an intentional request for
# expensive items.
_PRICE_OUTLIER_FACTOR: float = 8.0

# Budget-aware anchor cap (2026-11): with a stated budget the seed/anchor may cost at most this
# fraction of it, so the required accessory + footwear slots have money left. Chosen from the
# 36-cell budget grid (scripts/outfit_budget_grid.py, 2026-11-20, women+men x sangeet/wedding
# guest/office x Rs 1000..8000): cells reaching >=3 cards were 30 @0.3, 29 @0.4, 24 @0.5,
# 25 @0.6 (9 before any change). 0.4 gives up one cell vs 0.3 but keeps pricier, more
# representative anchors, so it is the default.
ANCHOR_BUDGET_FRACTION: float = 0.4

# Joint seed + slot selection (see _joint_selection and src/agents/outfit/joint.py): with a stated
# budget, the anchor and every slot item are chosen together to fill the most required slots, then
# the most cards, then the most relevant set, within the budget. Switch off to get the greedy
# path (the pre-joint behaviour, kept as the fallback and as the comparison baseline in
# evals/outfit/joint_selection_gate_rule.md).
JOINT_SELECTION_ENABLED: bool = True
# Wall-clock budget for the joint stage of ONE compose_outfit call, searches included. On timeout
# the greedy path runs instead (logged). A turn composes up to three looks (probe/base + two
# variants), so the worst case is three of these on top of the greedy fallback.
JOINT_WALL_BUDGET_S: float = 2.0
_JOINT_TOP_N: int = 20  # options kept per slot
# Slot fetch window. One search costs the same for any top_k (the fused window is fixed by
# retrieval.top_k); a wide top_k just stops the diversified top 40 from hiding the cheap options a
# tight budget needs.
_JOINT_FETCH_K: int = 40
_JOINT_MAX_ANCHORS: int = 8  # anchor candidates considered
_JOINT_MAX_ANCHOR_CLASSES: int = 1  # distinct anchor classes (each needs its own slot searches)

# Window for the one-shot retry when a budget-capped slot search yields no survivor (see
# _find_best_candidate); the normal window stays 40.
_BUDGET_RETRY_TOP_K: int = 200

# The footwear-floor failure is logged once per process, not once per request.
_FLOOR_FAILURE_LOGGED: bool = False


def warm_footwear_floors(retriever: HybridRetriever) -> None:
    """Build the footwear prefilter once, at index load, so the first budgeted outfit request
    doesn't pay for it. Per-(gender, occasion) floors are then derived lazily, cheapest row
    first. Safe on retrievers without a catalogue (no-op)."""
    footwear_price_floor(retriever, "women", "casual")


def footwear_price_floor(
    retriever: HybridRetriever, gender: str, occasion_slug: str
) -> float | None:
    """Cheapest footwear price that can actually pass the footwear slot's gates for this
    gender and occasion, in the retriever's (public) catalogue.

    Derived from the data, never hardcoded: every footwear row (classified by the slot gate's
    own ``classify_item``) is sorted by price per gender once per retriever, and the floor is
    the first row, cheapest first, that survives the SAME hard gates the slot applies
    (``_score_candidates``: gender, slot type, occasion register/coherence, rugged-footwear
    gate...). A plain catalogue minimum would be too low: e.g. a Rs 599 pair of slides is the
    cheapest women's footwear but never fits a sangeet look. Cached per (gender, occasion) on
    the underlying retriever (index lifetime); None when no eligible footwear exists or the
    retriever has no catalogue (test stubs). Used to reserve footwear money before accessories
    spend (see compose_outfit).
    """
    # _BiasedRetriever wraps the real retriever: resolve through it so every variant shares
    # one cache on the underlying (index-lifetime) object.
    base: Any = getattr(retriever, "_retriever", retriever)
    catalogue_df = getattr(base, "catalogue_df", None)
    if not isinstance(catalogue_df, pd.DataFrame):
        return None  # stub/fake retrievers in tests: no catalogue, no reservation
    state = getattr(base, "_footwear_floor_state", None)
    if not isinstance(state, dict):
        state = _footwear_floor_state(catalogue_df)
        base._footwear_floor_state = state  # type: ignore[attr-defined]
    key = (gender, occasion_slug)
    if key not in state["floors"]:
        floor: float | None = None
        for item in _ascending_footwear_items(state, gender):  # first survivor wins
            if _score_candidates(
                [item],
                query="",
                slot_name="footwear",
                occasion_slug=occasion_slug,
                gender=gender,
                anchor_colour="",
                seen_ids=set(),
                seen_prod_colour=set(),
                budget_remaining=None,
                pairing_stats=None,
                anchor_class="",
                seen_stores=None,
                neutral_fallback_ids=set(),
            ):
                floor = float(item["price_inr"])
                break
        state["floors"][key] = floor
    return state["floors"][key]


def _footwear_floor_state(catalogue_df: pd.DataFrame) -> dict:
    """Cheap, vectorised prefilter for the floor scan: footwear-looking, priced, exact-gender
    rows, per gender, as positions into ``catalogue_df`` sorted by price (stable, so ties keep
    catalogue order). The expensive per-row work (``classify_item`` + building the item dict)
    is deferred to ``_ascending_footwear_items``, which stops at the first gate survivor."""
    state: dict = {"df": catalogue_df, "order": {}, "items": {}, "floors": {}}
    needed = {"product_type_name", "prod_name", "gender", "price_inr"}
    if not needed <= set(catalogue_df.columns):
        return state
    text = (
        catalogue_df["product_type_name"].fillna("").astype(str)
        + " "
        + catalogue_df["prod_name"].fillna("").astype(str)
    ).str.lower()
    # Substring prefilter (a superset of classify_item's word-boundary match) so the Python
    # classifier only ever sees footwear-looking rows, never the whole 100k+ catalogue.
    pattern = "|".join(re.escape(k) for k in sorted(FOOTWEAR_KEYWORDS))
    hit = text.str.contains(pattern, regex=True).to_numpy()
    price = pd.to_numeric(catalogue_df["price_inr"], errors="coerce").to_numpy()
    gender = catalogue_df["gender"].fillna("").astype(str).str.lower().to_numpy()
    for g in ("men", "women"):
        pos = np.flatnonzero(hit & (gender == g) & (price > 0))  # NaN > 0 is False
        state["order"][g] = pos[np.argsort(price[pos], kind="stable")]
    return state


def _footwear_item(state: dict, pos: int) -> dict | None:
    """Retriever-shaped item dict for catalogue row ``pos`` if it is footwear-class, else None
    (memoised: the same rows are re-walked for every (gender, occasion) floor)."""
    items: dict[int, dict | None] = state["items"]
    if pos in items:
        return items[pos]
    df: pd.DataFrame = state["df"]
    row = df.iloc[pos]
    ptv = str(row["product_type_name"]) if pd.notna(row["product_type_name"]) else ""
    nmv = str(row["prod_name"]) if pd.notna(row["prod_name"]) else ""
    item: dict | None = None
    if classify_item(ptv, nmv) == "footwear":
        facets = row["facets"] if "facets" in df.columns else None
        facets = facets if isinstance(facets, dict) else {}
        display = row["display_name"] if "display_name" in df.columns else nmv
        item = {
            "article_id": str(df.index[pos]),
            "prod_name": nmv,
            "display_name": display,
            "colour": facets.get("colour_group_name", ""),
            "product_type": ptv,
            "detail_desc": row["detail_desc"] if "detail_desc" in df.columns else None,
            "gender": str(row["gender"]).lower(),
            "price_inr": float(row["price_inr"]),
            "store": row["store"] if "store" in df.columns else None,
        }
    items[pos] = item
    return item


def _ascending_footwear_items(state: dict, gender: str) -> Iterator[dict]:
    """Footwear-class items for ``gender``, cheapest first, built lazily."""
    for pos in state["order"].get(gender, []):
        item = _footwear_item(state, int(pos))
        if item is not None:
            yield item


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
    formality_override: str | None = None,
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
        formality_override: Batch 2 (2026-07-13) — "minimalist"/"comfortable",
            the sibling intent-parser signal a query like "something comfortable
            for sangeet dancing" or "not too flashy" surfaces (see
            slots.fabric_score_delta's docstring). Opt-in bias only (never a
            filter), same treatment as body_type above: nudges complement
            scoring via fabric_score_delta, threaded straight through to every
            _find_best_candidate call. None (default) is a full no-op.

    Returns a dict with: look_id, seed_item, complements, outfit_rationale,
    empty_slots, suppressed_slots, occasion, gender, budget_total_inr.
    """
    look_id = str(uuid.uuid4())
    # Gate-passing, in-budget anchor candidates of the occasion-driven anchor stage, kept (no
    # extra retrieval) so the caller can offer alternates. None unless that stage ran with a budget.
    anchor_alternates: list[dict] | None = None
    # Set only by the budgeted occasion-driven anchor stage when joint selection succeeded.
    joint: _JointChoice | None = None
    # A joint look that left a slot empty: only used if it beats the greedy look (see Step 2).
    joint_alt: _JointChoice | None = None

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
        _bt_tokens = body_type_query_tokens(body_type, body_modifiers, gender)
        if _bt_tokens:
            anchor_query = f"{anchor_query} {_bt_tokens}"
        anchor_filters = {"gender": gender} if gender in ("men", "women") else None
        # 2026-07-19 fix (39-store catalogue regression): the default top_k=10
        # anchor window is budget-BLIND — it is filled by relevance score alone,
        # so a handful of newly-added premium-tier brands (much higher-priced,
        # equally or more relevant for occasion terms like "reception") can
        # monopolize the top 10 and push every genuinely-affordable anchor
        # candidate out of the window before the budget gate below ever sees
        # them (live-proven: "reception outfit under ₹15000" — 0/10 anchors
        # were within budget at top_k=10, but 2/50 were once the window
        # widened). Mirrors the same truncate-before-filter defect graph.py's
        # search_node fixed for price_qualifier/formality_softener (commit
        # 2c78af9) — widen the pre-filter pool whenever a budget constraint is
        # present so the budget gate has a meaningfully larger pool to draw
        # from. No-op (top_k stays 10) when no budget is stated.
        _anchor_top_k = 30 if budget_inr is not None else 10
        # Budget-aware anchor window (2026-11 outfit budget fix): the anchor is the one
        # piece the look is built around, and a seed priced near the whole budget leaves
        # nothing for the required accessory + footwear (live: "sangeet outfit under
        # ₹3000" seeded a ₹2,399 lehenga and shipped 2 cards). With a budget, push a price
        # cap of ANCHOR_BUDGET_FRACTION x budget into the retrieval itself so affordable
        # anchors outside the top-k window become visible, not merely post-filtered.
        if budget_inr is None:
            candidates = retriever.search(
                anchor_query, top_k=_anchor_top_k, filters=anchor_filters
            )
        else:
            candidates = retriever.search(
                anchor_query,
                top_k=_anchor_top_k,
                filters={**(anchor_filters or {}), "price_max": budget_inr * ANCHOR_BUDGET_FRACTION},
            )
        # Filter by occasion coherence AND gender compatibility. S5 fix: also
        # reject juniors/girls/boys/kids items as a look ANCHOR — the same
        # catalogue mislabeling that lets them fill complement slots (see
        # is_kids_item) would otherwise let one become the seed itself.
        _anchor_look_gender = gender if gender != "unisex" else "unisex"

        def _valid_anchors(cands: list[dict]) -> list[dict]:
            return [
                c
                for c in cands
                if _anchor_matches_occasion(c, occasion_slug, _anchor_look_gender)
                and gender_allowed((c.get("gender") or "unknown").lower(), _anchor_look_gender)
                and not is_kids_item(
                    c.get("prod_name") or c.get("display_name") or "", c.get("detail_desc")
                )
                # 2026-08-06 cross-gender leak fix: the catalogue's own gender
                # column is unreliable for a real, audited slice of rows (385,
                # concentrated in vastramay/voylla/jompers) -- e.g. "Men's
                # Yellow - Dupatta" carries gender="women". Name text is the
                # more trustworthy signal here, same shape as is_kids_item()
                # above. See has_gender_text_conflict's own docstring.
                and not has_gender_text_conflict(
                    c.get("prod_name") or c.get("display_name") or "", _anchor_look_gender
                )
                # Garment gate (joint-selection PR): the seed must be a real garment. A rakhi,
                # belt or other accessory used to reach the seed when the price cap left only
                # cheap non-garments (men's budget Rs 800); the more-options row already applied
                # the same _is_outfit_garment predicate. Last in the chain: it is the costliest.
                # Budgeted turns only: the no-budget path is held byte-identical (a bridal
                # "Kalira" still seeds the no-budget reception look; a follow-up, with a fixture
                # re-record, can extend the gate there).
                and (
                    budget_inr is None
                    or _is_outfit_garment(
                        c.get("product_type") or "",
                        c.get("prod_name") or c.get("display_name") or "",
                    )
                )
            ]

        valid = _valid_anchors(candidates)
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
                # Nothing relevant fits under the anchor cap: fall back to the cheapest
                # relevant anchor that fits the budget at all (it may leave little for the
                # complements, but a look beats an empty answer).
                widened = _valid_anchors(
                    retriever.search(
                        anchor_query,
                        top_k=_anchor_top_k,
                        filters={**(anchor_filters or {}), "price_max": budget_inr},
                    )
                )
                # price re-checked here: the pushdown is an optimisation, the budget gate is
                # the guarantee (a retriever that ignores price_max must not over-budget a look).
                valid = sorted(
                    (c for c in widened if (c.get("price_inr") or 0.0) <= budget_inr),
                    key=lambda c: c.get("price_inr") or 0.0,
                )
                if not valid:
                    # Distinguish "all relevant anchors are over budget" (honest budget
                    # message below) from "none exist": re-run without the price pushdown.
                    _pre_budget_valid = _valid_anchors(
                        retriever.search(anchor_query, top_k=_anchor_top_k, filters=anchor_filters)
                    )
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
        if budget_inr is not None and JOINT_SELECTION_ENABLED and not owned_anchor:
            joint = _joint_selection(
                retriever=retriever,
                valid=valid,
                budget=budget_inr,
                occasion_slug=occasion_slug,
                gender=gender,
                brand_gender_default=brand_gender_default,
                exclude_ids=exclude_ids or set(),
                pairing_stats=pairing_stats,
                body_type=body_type,
                body_modifiers=body_modifiers,
                formality_override=formality_override,
            )
        if joint is not None and joint.suppressed_slots:
            # The joint pools come from one shared fetch, so cheap options a tight budget needs
            # can be missing from them. When the joint look leaves ANY slot empty, also run the
            # greedy path (its per-slot price caps see those options) and keep whichever fills
            # more required slots, then more cards (see the end of Step 2).
            joint_alt, joint = joint, None
        seed_item = joint.seed if joint else valid[0]
        seed_item["_role"] = "seed"
        seed_article_id = seed_item["article_id"]
        if budget_inr is not None:
            anchor_alternates = list(valid)

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

    # Absurd price-outlier guard (2026-07-23) — see _PRICE_OUTLIER_FACTOR's
    # module-level docstring for the full rationale. Skipped when the user
    # gave an explicit budget (budget_remaining already bounds price there,
    # and a stated budget is an intentional request) or when the anchor has
    # no catalogue price to anchor against (owned/uploaded seed item).
    _anchor_price_for_cap = seed_item.get("price_inr") or 0.0
    price_outlier_cap: float | None = None
    if budget_inr is None and _anchor_price_for_cap > 0:
        price_outlier_cap = _anchor_price_for_cap * _PRICE_OUTLIER_FACTOR

    # Footwear floor (2026-11 budget fix): when footwear is a required slot that has not been
    # filled yet, accessories and other OPTIONAL slots (e.g. outerwear) that come before it may
    # not spend the money footwear needs. The floor is the cheapest eligible footwear in the
    # catalogue for this gender and occasion (derived, not hardcoded).
    _footwear_idx = next(
        (i for i, s in enumerate(fill_slots) if s.slot_name == "footwear" and s.required), None
    )
    _footwear_floor: float | None = None
    if budget_inr is not None and _footwear_idx is not None and joint is None:
        try:
            _footwear_floor = footwear_price_floor(retriever, effective_gender, occasion_slug)
        except Exception:  # noqa: BLE001 - floor only improves packing; never block a look
            global _FLOOR_FAILURE_LOGGED
            if not _FLOOR_FAILURE_LOGGED:
                _FLOOR_FAILURE_LOGGED = True
                logger.warning(
                    "footwear price floor failed; composing without a footwear reservation",
                    exc_info=True,
                )
            _footwear_floor = None

    # Joint selection already chose every slot item together with the seed: skip the greedy loop.
    for slot_idx, slot_spec in enumerate([] if joint is not None else fill_slots):
        slot_budget: float | None = budget_inr - running_total if budget_inr is not None else None
        if (
            slot_budget is not None
            and _footwear_floor is not None
            and _footwear_idx is not None
            and slot_idx < _footwear_idx
            and (slot_spec.slot_name == "accessory" or not slot_spec.required)
            and slot_budget >= _footwear_floor  # if footwear cannot fit anyway, don't starve this
        ):
            slot_budget -= _footwear_floor
        candidate = _find_best_candidate(
            query=slot_spec.search_query,
            slot_name=slot_spec.slot_name,
            occasion_slug=occasion_slug,
            gender=effective_gender,
            anchor_colour=anchor_colour,
            seen_ids=seen_ids,
            seen_prod_colour=seen_prod_colour,
            retriever=retriever,
            budget_remaining=slot_budget,
            pairing_stats=pairing_stats,
            anchor_class=anchor_class,
            seen_stores=seen_stores,
            body_type=body_type,
            body_modifiers=body_modifiers,
            formality_override=formality_override,
            price_outlier_cap=price_outlier_cap,
        )
        if candidate is None and slot_spec.slot_name == "bottom" and effective_gender == "men":
            # Live defect 2026-07-10: the men's ethnic bottom query (churidar/
            # pyjama/dhoti) has almost no clean survivors in this catalogue, so
            # every men's ethnic board shipped bottomless with a false "no men's
            # bottoms" note — while 4,668 men's trousers/jeans exist. Tailored
            # trousers are a legitimate bottom under a kurta or sherwani; retry
            # once with a western-formal query before suppressing the slot.
            candidate = _find_best_candidate(
                query="tailored trousers formal dark trousers",
                slot_name=slot_spec.slot_name,
                occasion_slug=occasion_slug,
                gender=effective_gender,
                anchor_colour=anchor_colour,
                seen_ids=seen_ids,
                seen_prod_colour=seen_prod_colour,
                retriever=retriever,
                budget_remaining=slot_budget,
                pairing_stats=pairing_stats,
                anchor_class=anchor_class,
                seen_stores=seen_stores,
                body_type=body_type,
                body_modifiers=body_modifiers,
                formality_override=formality_override,
                price_outlier_cap=price_outlier_cap,
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

    if joint_alt is not None:
        greedy_required = sum(1 for s in fill_slots if s.required) - len(empty_slots)
        if (joint_alt.required_filled, joint_alt.cards) > (greedy_required, 1 + len(complements)):
            joint = joint_alt  # strictly better than greedy; a tie keeps the greedy look
            seed_item = joint.seed
            seed_item["_role"] = "seed"
            seed_article_id = seed_item["article_id"]
            anchor_class = classify_anchor(seed_item.get("product_type") or "", _name(seed_item))
    if joint is not None:
        complements = list(joint.complements)
        empty_slots = list(joint.empty_slots)
        suppressed_slots = list(joint.suppressed_slots)
        running_total = joint.total_price

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

    look: dict = {
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
    if anchor_alternates is not None:
        look["anchor_alternates"] = anchor_alternates
    return look


@dataclass
class _JointChoice:
    """Outcome of the joint stage: the seed and the finished slot results."""

    seed: dict
    complements: list[dict]
    empty_slots: list[str]
    suppressed_slots: list[dict]
    total_price: float
    required_filled: int
    cards: int


class _SearchMemo:
    """Per-compose memo of retriever.search (same args -> one search).

    The joint stage asks for the same slot pool for several anchors; retrieval dominates latency.
    Returns a fresh list each call (callers append to it) over shared item dicts (the scoring
    gates never mutate them). Lives for one compose_outfit call, so it cannot go stale.
    """

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever
        self._cache: dict[tuple, list[dict]] = {}
        self.searches = 0

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None
    ) -> list[dict]:
        key = (query, top_k, tuple(sorted((k, repr(v)) for k, v in (filters or {}).items())))
        if key not in self._cache:
            self.searches += 1
            self._cache[key] = self._retriever.search(query, top_k=top_k, filters=filters)
        return list(self._cache[key])


def _joint_selection(
    *,
    retriever: Any,
    valid: list[dict],
    budget: float,
    occasion_slug: str,
    gender: str,
    brand_gender_default: str,
    exclude_ids: set[str],
    pairing_stats: dict | None,
    body_type: str | None,
    body_modifiers: list[str] | None,
    formality_override: str | None,
) -> _JointChoice | None:
    """Choose the seed and every slot item together (see src/agents/outfit/joint.py).

    Candidate anchors: the gate-passing anchors the greedy stage already found (``valid``, price
    cap ANCHOR_BUDGET_FRACTION x budget; no extra anchor search, retrieval is the latency cost),
    best retrieval score first, at most ``_JOINT_MAX_ANCHORS`` of at most
    ``_JOINT_MAX_ANCHOR_CLASSES`` anchor class(es) (each class needs its own slot searches); the
    greedy seed (``valid[0]``) is always among them, so its class is the one searched. One slot
    fetch per slot query serves every anchor (price cap = budget minus the cheapest anchor, the
    most any anchor leaves), and each anchor then gates the shared pool with its own remaining
    budget. Every slot option passes the SAME hard gates as the greedy path (``_score_candidates``:
    gender, slot type, kids, formality/register, colour/coherence, budget, no repeats) because
    options come from ``_ranked_candidates``. Soft store diversity is applied against the anchor's
    store only (it is sequential in the greedy path). Returns None (caller runs the greedy path)
    when nothing fits, or on timeout, with the reason logged.
    """
    deadline = time.monotonic() + JOINT_WALL_BUDGET_S
    memo = _SearchMemo(retriever)
    try:
        pool = list(valid)
        greedy_seed = valid[0]
        max_score = max((c.get("score") or 0.5) for c in pool)
        ordered = sorted(pool, key=lambda c: (-(c.get("score") or 0.5), str(c["article_id"])))
        anchors: list[dict] = [greedy_seed]
        classes = {classify_anchor(greedy_seed.get("product_type") or "", _name(greedy_seed))}
        for cand in ordered:
            if len(anchors) >= _JOINT_MAX_ANCHORS:
                break
            if cand is greedy_seed or any(cand is a for a in anchors):
                continue
            cls = classify_anchor(cand.get("product_type") or "", _name(cand))
            if cls not in classes and len(classes) >= _JOINT_MAX_ANCHOR_CLASSES:
                continue
            classes.add(cls)
            anchors.append(cand)

        # One slot fetch serves every anchor: its price cap is the most any anchor leaves.
        fetch_cap = budget - min(float(a.get("price_inr") or 0.0) for a in anchors)
        gate_cache: dict = {}
        raw: list[tuple[dict, float, list[tuple[Any, list[tuple[float, dict]]]]]] = []
        slot_best: dict[str, float] = {}
        for anchor in anchors:
            price = float(anchor.get("price_inr") or 0.0)
            remaining = budget - price
            eff_gender = gender if gender != "unisex" else _infer_gender(anchor, brand_gender_default)
            a_class = classify_anchor(anchor.get("product_type") or "", _name(anchor))
            a_colour = (anchor.get("colour") or "").lower()
            a_store = anchor.get("store")
            per_slot: list[tuple[Any, list[tuple[float, dict]]]] = []
            for spec in get_fill_slots(a_class, eff_gender, occasion_slug, body_type, body_modifiers):
                if time.monotonic() > deadline:
                    raise JointTimeout("wall budget exhausted while building slot options")
                ranked = _ranked_slot_options(
                    spec,
                    memo=memo,
                    occasion_slug=occasion_slug,
                    gender=eff_gender,
                    anchor=anchor,
                    anchor_class=a_class,
                    anchor_colour=a_colour,
                    anchor_store=a_store,
                    remaining=remaining,
                    fetch_cap=fetch_cap,
                    gate_cache=gate_cache,
                    exclude_ids=exclude_ids,
                    pairing_stats=pairing_stats,
                    body_type=body_type,
                    body_modifiers=body_modifiers,
                    formality_override=formality_override,
                )[:_JOINT_TOP_N]
                if ranked:
                    slot_best[spec.slot_name] = max(slot_best.get(spec.slot_name, 0.0), ranked[0][0])
                per_slot.append((spec, ranked))
            raw.append((anchor, price, per_slot))

        plans: list[AnchorPlan] = []
        for anchor, price, per_slot in raw:
            slots = tuple(
                SlotChoices(
                    spec.slot_name,
                    spec.required,
                    tuple(
                        Option(
                            item,
                            score / slot_best[spec.slot_name] if slot_best[spec.slot_name] > 0 else 1.0,
                            float(item.get("price_inr") or 0.0),
                            (normalize_prod_name(_name(item)), (item.get("colour") or "").lower()),
                        )
                        for score, item in ranked
                    ),
                )
                for spec, ranked in per_slot
            )
            plans.append(
                AnchorPlan(
                    Option(
                        anchor,
                        (anchor.get("score") or 0.5) / max_score,
                        price,
                        (normalize_prod_name(_name(anchor)), (anchor.get("colour") or "").lower()),
                    ),
                    slots,
                )
            )
        solution = select_joint(plans, budget, deadline=deadline)
    except JointTimeout as exc:
        logger.warning("[composer/joint] falling back to greedy: %s", exc)
        return None
    except Exception:  # noqa: BLE001 - the joint stage only improves packing; never block a look
        logger.warning("[composer/joint] falling back to greedy: unexpected error", exc_info=True)
        return None
    if solution is None:
        logger.info("[composer/joint] falling back to greedy: no anchor fits the budget")
        return None

    plan = plans[solution.plan_index]
    complements: list[dict] = []
    empty: list[str] = []
    suppressed: list[dict] = []
    eff_gender = gender if gender != "unisex" else _infer_gender(
        plan.anchor.item, brand_gender_default
    )
    for slot, pick in zip(plan.slots, solution.picks, strict=True):
        if pick is None:
            suppressed.append(
                {"slot": slot.name, "reason": _suppression_reason(slot.name, eff_gender)}
            )
            if slot.required:
                empty.append(slot.name)
            continue
        item = pick.item
        item["_slot"] = slot.name
        item["_role"] = "complement"
        complements.append(item)
    logger.debug(
        "[composer/joint] anchors=%d searches=%d nodes=%d cards=%d total=%.0f/%.0f",
        len(plans), memo.searches, solution.nodes, solution.cards, solution.total_price, budget,
    )
    return _JointChoice(
        plan.anchor.item,
        complements,
        empty,
        suppressed,
        solution.total_price,
        solution.required_filled,
        solution.cards,
    )


def _name(item: dict) -> str:
    return item.get("prod_name") or item.get("display_name") or ""


def _ranked_slot_options(
    spec: Any,
    *,
    memo: _SearchMemo,
    occasion_slug: str,
    gender: str,
    anchor: dict,
    anchor_class: str,
    anchor_colour: str,
    anchor_store: Any,
    remaining: float,
    fetch_cap: float,
    gate_cache: dict,
    exclude_ids: set[str],
    pairing_stats: dict | None,
    body_type: str | None,
    body_modifiers: list[str] | None,
    formality_override: str | None,
) -> list[tuple[float, dict]]:
    """Gate-passing, scored options for one slot of one anchor (best first), budget ``remaining``."""
    if remaining <= 0:
        return []
    common: dict[str, Any] = dict(
        slot_name=spec.slot_name,
        occasion_slug=occasion_slug,
        gender=gender,
        anchor_colour=anchor_colour,
        seen_ids={str(anchor["article_id"])} | exclude_ids,
        seen_prod_colour={(normalize_prod_name(_name(anchor)), anchor_colour)},
        retriever=memo,
        budget_remaining=remaining,
        pairing_stats=pairing_stats,
        anchor_class=anchor_class,
        seen_stores={str(anchor_store).lower()} if anchor_store else set(),
        body_type=body_type,
        body_modifiers=body_modifiers,
        formality_override=formality_override,
        fetch_price_max=fetch_cap,
        fetch_k=_JOINT_FETCH_K,
        gate_cache=gate_cache,
    )
    ranked = _ranked_candidates(query=spec.search_query, **common)
    if not ranked and spec.slot_name == "bottom" and gender == "men":
        # Same retry as the greedy path (live defect 2026-07-10): men's ethnic bottoms are thin.
        ranked = _ranked_candidates(query="tailored trousers formal dark trousers", **common)
    return ranked


MORE_OPTIONS_MAX: int = 6
MORE_OPTIONS_MAX_PER_STORE: int = 2


# Structured product_type facet values (lowercased) that are never an outfit anchor. Grounded in
# the public catalogue's own vocabulary (inner/swim/night/foot types and counts: nightwear 382,
# night suit 88, innerwear 61, vest 50, men loungers 21, sports_bra 21, lounge wear 17, w briefs 8,
# swimwear 4, footwear 3,985).
_NON_ANCHOR_TYPES: frozenset[str] = frozenset(
    {
        "footwear",
        "swimwear", "bikini", "innerwear", "lingerie", "vest", "w briefs", "briefs", "bra",
        "sports_bra", "sports bra", "boxers", "socks",
        "nightwear", "night suit", "men loungers", "lounge wear", "loungewear", "sleepwear",
    }
)
# The sleep/swim/inner types are mislabelled for a handful of real ethnic sets ("Kurta & Pyjama
# Set" typed nightwear): for THOSE types only, an ethnic garment word in the name keeps the item.
_SLEEP_SWIM_INNER_TYPES: frozenset[str] = _NON_ANCHOR_TYPES - {"footwear", "socks"}
_ETHNIC_GARMENT_NAME_RE = re.compile(
    r"\b(kurta|kurti|sherwani|bandhgala|achkan|lehenga|saree|sari|anarkali|sharara)\b", re.I
)
# Name-level terms that are never an anchor whatever the facet says (the facet is primary; this
# is the tie-break for rows typed generically, e.g. "Fashion").
_NON_ANCHOR_NAME_RE = re.compile(
    r"\b(bikini|swimsuit|swimwear|panty|panties|lingerie|briefs?|bra|bras|boxers?|innerwear|"
    r"night\s?suit|nightsuit|nightwear|nighty|night\s?dress|sleepwear|loungewear)\b",
    re.I,
)


@functools.lru_cache(maxsize=8192)  # pure on two strings; ~2 ms per cold call, hot in every turn
def _is_outfit_garment(product_type: str, prod_name: str) -> bool:
    """True when the item can anchor a look: a garment by its STRUCTURED product_type first.

    Order matters (a men's "Sherwani Jutti", product_type footwear, classifies as men_formalwear
    when the name is scanned, which is how footwear reached the row):
      1. the facet alone (name blanked) resolving to footwear/accessory is final;
      2. facet in the footwear/inner/swim/night deny set is final, except sleep/swim/inner-typed
         ethnic sets (name has kurta/sherwani/...), which stay;
      3. the name only breaks ties: non-anchor words reject; the full classification must be a
         known garment class.
    """
    pt = product_type.strip().lower()
    if classify_item(product_type, "") in NON_OUTFIT_ITEM_CLASSES:
        return False
    if pt in _NON_ANCHOR_TYPES:
        if pt in _SLEEP_SWIM_INNER_TYPES and _ETHNIC_GARMENT_NAME_RE.search(prod_name):
            pass  # falls through to the name / class checks below
        else:
            return False
    if _NON_ANCHOR_NAME_RE.search(prod_name) and not (
        _ETHNIC_GARMENT_NAME_RE.search(prod_name) and pt in _SLEEP_SWIM_INNER_TYPES
    ):
        return False
    item_class = classify_item(product_type, prod_name)
    return item_class not in NON_OUTFIT_ITEM_CLASSES and item_class != "unknown"


_BOTTOM_PREFILTER_KEYWORDS: tuple[str, ...] = tuple(
    sorted(ETHNIC_BOTTOM_KEYWORDS | WESTERN_BOTTOM_KEYWORDS)
)


def _is_standalone_bottom(product_type: str, prod_name: str) -> bool:
    """True for a bottom sold alone (pant, palazzo, churidar, pyjama, jeans, skirt...).

    A bottom that is part of a multi-piece set ("Kurta & Pyjama Set", "Anarkali Sharara Set")
    is a whole outfit and stays, via the same ``is_multi_piece_set`` the complement slots use.
    """
    text = f"{product_type} {prod_name}".lower()
    # Cheap substring prefilter (a superset of what classify_item resolves to a bottom class: it
    # matches the same keyword sets, plus the "lower" facet): the full classification is ~1 ms per
    # item and a row is scanned per candidate, so most items must never reach it.
    if product_type.strip().lower() not in ("lower", "lowers") and not any(
        kw in text for kw in _BOTTOM_PREFILTER_KEYWORDS
    ):
        return False
    return classify_item(product_type, prod_name) in SLOT_ALLOWED_CLASSES[
        "bottom"
    ] and not is_multi_piece_set(product_type, prod_name)


def build_more_options(
    looks: list[dict],
    budget_inr: float | None,
    *,
    limit: int = MORE_OPTIONS_MAX,
) -> list[dict] | None:
    """Alternate anchors ("more options") for a budgeted, occasion-driven outfit turn.

    Built purely from the anchor candidates the anchor stage ALREADY retrieved and gated
    (gender, kids, gender-name conflict, occasion coherence, price <= budget; see
    ``compose_outfit``) - no retrieval, no LLM. Returns None when no budget was given or the
    anchor stage did not run (explicit seed, owned anchor): the field is then absent. Returns
    a possibly empty list otherwise.

    Rules, in order:
      * garments only (accessory/footwear/unclassifiable are never an anchor), no fabric bolts or
        occasion merchandise, priced, within budget, with a buy link (server-built PDP URL);
      * never the seed nor any item already shown in ANY look variant;
      * one entry per normalised product name (colourways of one product count once);
      * ordering is the anchor stage's own relevance order (the retrieval rank), diversified:
        first pass takes items of a product type not yet shown, second pass fills the rest;
        at most ``MORE_OPTIONS_MAX_PER_STORE`` per store in both passes; at most ``limit``.
    """
    if budget_inr is None or not looks:
        return None
    base = looks[0]
    pool = base.get("anchor_alternates")
    if pool is None:
        return None
    shown: set[str] = set()
    for look in looks:
        seed = look.get("seed_item") or {}
        if seed.get("article_id"):
            shown.add(str(seed["article_id"]))
        shown |= {str(c["article_id"]) for c in (look.get("complements") or [])}

    eligible: list[dict] = []
    seen_names: set[str] = set()
    for item in pool:
        aid = str(item.get("article_id"))
        price = item.get("price_inr")
        name = item.get("prod_name") or item.get("display_name") or ""
        if aid in shown or price is None or not 0 < price <= budget_inr:
            continue
        if not _is_outfit_garment(item.get("product_type") or "", name):
            continue  # an anchor is a garment: no footwear/accessories/innerwear/swim/nightwear
        if _is_standalone_bottom(item.get("product_type") or "", name):
            continue  # a pant/palazzo/churidar alone is half an outfit, not an alternate look
        if (
            is_fabric_bolt_text(name)
            or is_occasion_merchandise_type(item.get("product_type"))
            or is_occasion_merchandise_name(
                name, item.get("product_type"), item.get("detail_desc")
            )
        ):
            continue
        if not build_pdp_url(item.get("store"), item):
            continue
        key = normalize_prod_name(name)
        if key in seen_names:
            continue
        seen_names.add(key)
        eligible.append(item)

    picked: list[dict] = []
    per_store: dict[str, int] = {}
    types_seen: set[str] = set()

    def _take(item: dict) -> None:
        picked.append(item)
        store = str(item.get("store") or "").lower()
        per_store[store] = per_store.get(store, 0) + 1
        types_seen.add(str(item.get("product_type") or "").lower())

    for distinct_pass in (True, False):
        for item in eligible:
            if len(picked) >= limit:
                break
            if item in picked:
                continue
            store = str(item.get("store") or "").lower()
            if per_store.get(store, 0) >= MORE_OPTIONS_MAX_PER_STORE:
                continue
            if distinct_pass and str(item.get("product_type") or "").lower() in types_seen:
                continue
            _take(item)
    return picked


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
    # 2026-07-24 sweep (same failure class as rationale._display_noun's
    # sports_bra leak fix): every slot_name in use today is covered by
    # `labels` above, but the .get() fallback previously returned the raw
    # slot_name UNSANITIZED if a future slot type were ever added without
    # also updating this dict — sanitize defensively so a hypothetical raw
    # "some_new_slot" can never leak an underscore into user-facing prose.
    label = labels.get(slot_name, slot_name.replace("_", " "))
    # "that match this look" — the honest claim. A bare "No men's bottoms in our
    # partner stores yet" was live-proven FALSE (4,668 exist); suppression only
    # means no candidate survived THIS look's gates, never an inventory absence.
    return f"No {gender}'s {label} that match this look in our partner stores yet"


def probe_reusable_as_base(gender: str, brand_gender_default: str | None) -> bool:
    """True when ``compose_outfit_tool``'s probe look equals variant 0 of
    ``compose_outfit_variants`` for the same inputs.

    The tool wrapper fixes ``pairing_stats=None`` and leaves ``brand_gender_default`` at
    compose_outfit's "women" default, and the variants call passes ``pairing_stats=None`` too.
    ``brand_gender_default`` is only read to resolve a "unisex" gender, so the two compositions
    can differ only for gender == "unisex" with a non-"women" brand default.
    """
    return gender != "unisex" or (brand_gender_default or "women") == "women"


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
    formality_override: str | None = None,
    base_look: dict | None = None,
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
        formality_override:  Batch 2 — see ``compose_outfit`` docstring. Applied
            to every variant (base + biased).

        base_look:           An already-composed base look to use as variant 0 instead of
            recomposing it (the outfit node's viability probe is exactly this call). The
            caller guarantees it came from compose_outfit with arguments identical to the
            ones passed here (see ``probe_reusable_as_base``); it is labelled "Base" in place.

    Returns:
        List of 1–3 look dicts.  Always non-empty (falls back to base-only).
    """
    # Variant 0: base look — standard behaviour (or the caller's identical, already-composed one)
    base = (
        base_look
        if base_look is not None
        else compose_outfit(
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
            formality_override=formality_override,
        )
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
        formality_override=formality_override,
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
        formality_override=formality_override,
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
    formality_override: str | None = None,
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
            formality_override=formality_override,
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
    formality_override: str | None = None,
    price_outlier_cap: float | None = None,
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

    formality_override: Batch 2 (2026-07-13) — passed straight to
    fabric_score_delta (see that function's docstring). Same "nudge, never a
    gate" treatment as body_type above.

    price_outlier_cap: 2026-07-23 — see _PRICE_OUTLIER_FACTOR's module-level
    docstring. A hard gate (unlike body_type/formality_override above): any
    candidate priced above this cap is rejected outright, never merely
    nudged, mirroring _apply_loungewear_gate's "never an acceptable
    substitute" discipline in graph.py. None (the default, and always the
    value passed when the user gave an explicit budget) is a full no-op.
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
        # 2026-08-06 cross-gender leak fix: applies to EVERY slot (top/
        # bottom/footwear/outerwear/accessory), not just accessory -- the
        # live-proven leak was "Men's Yellow - Dupatta" in a women's haldi
        # look's accessory slot, but the catalogue's gender-column
        # unreliability (385 audited rows) is not accessory-specific.
        # Unconditional, including neutral-fallback items -- an item chosen
        # via the unisex-accessory fallback for its UNKNOWN gender column
        # must still be rejected if its own name explicitly states the
        # wrong gender. See has_gender_text_conflict's own docstring.
        if has_gender_text_conflict(
            item.get("prod_name") or item.get("display_name") or "", gender
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
        if is_kids_item(item_name, item.get("detail_desc")):
            continue
        # Loungewear hard gate (2026-07-31, "baraat outfit for men" fix):
        # formal_ethnic occasions must never accept a loungewear/nightwear
        # item, regardless of which class classify_anchor/classify_item
        # resolved it to. Live-proven: classify_anchor's ETHNIC_BOTTOM_
        # KEYWORDS matches a bare "pyjama" (added for genuine "kurta pajama"
        # ethnic sets), which ALSO misclassifies loungewear "Top & Pyjama
        # Set" rows (product_type_name="nightwear") as ethnic_bottom, so
        # is_coherent_candidate's ethnic gates waved them straight through
        # as if legitimate ethnic wear — "baraat outfit for men" (occasion=
        # wedding_guest) composed "Men Solid Multicolor Top & Pyjama Set"
        # into the bottom slot. Applied here (never a soft nudge, same
        # "never an acceptable substitute" discipline as the price-outlier
        # gate below and graph.py's own _apply_loungewear_gate) rather than
        # as a coherence.is_coherent_candidate gate — that function is ALSO
        # called from graph.py's search_node inside a POOL-UNDERFLOW-
        # PROTECTED coherence check, where rejecting a genuine "Kaftan Night
        # Dress" this early would skip the whole protected filter instead of
        # falling through to the later, correctly-unprotected
        # _apply_loungewear_gate (see is_coherent_candidate's own docstring
        # note for the exact regression this caused when first tried there).
        # _FORMAL_ETHNIC_OCCASIONS (not graph.py's private _LOUNGEWEAR_GATE_
        # OCCASIONS, which also adds "gym") — a gym look's ethnic-item
        # rejection (gate 5 above) already covers this item class, since
        # ETHNIC_BOTTOM_KEYWORDS' bare "pyjama" match also makes
        # is_ethnic_item true for it.
        if occasion_slug in _FORMAL_ETHNIC_OCCASIONS and is_loungewear_text(item_name):
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
        # Footwear register gate (sweep 2026-07-10, relevance-adjacent): the
        # casual-marker vocabulary above covers garments only — ₹759 combat
        # boots passed it into a sangeet look. Rugged/athletic footwear never
        # belongs in a formality >= 3 occasion's footwear slot.
        if (
            slot_name == "footwear"
            and get_occasion(occasion_slug).formality >= 3
            and is_rugged_footwear_item(item_name)
        ):
            continue
        if not is_coherent_candidate(
            item, occasion_slug, gender, slot_name, skip_gender_gate=is_neutral_fallback_item
        ):
            continue
        if budget_remaining is not None:
            price = item.get("price_inr") or 0.0
            if price > budget_remaining:
                continue
        # Absurd price-outlier guard — see _PRICE_OUTLIER_FACTOR docstring.
        if price_outlier_cap is not None:
            price = item.get("price_inr") or 0.0
            if price > price_outlier_cap:
                continue

        base_score = item.get("score") or 0.5
        c_score = colour_score(item.get("colour") or "", anchor_colour, occasion_slug)
        fab_delta = fabric_score_delta(item, occasion_slug, formality_override=formality_override)
        bt_delta = body_type_score_delta(item, body_type, body_modifiers, gender)
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


def _find_best_candidate(**kwargs: Any) -> dict | None:
    """Best-scoring gate-passing candidate for one slot, or None (see ``_ranked_candidates``)."""
    ranked = _ranked_candidates(**kwargs)
    return ranked[0][1] if ranked else None


def _ranked_candidates(
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
    formality_override: str | None = None,
    price_outlier_cap: float | None = None,
    fetch_price_max: float | None = None,
    fetch_k: int = 40,
    gate_cache: dict | None = None,
) -> list[tuple[float, dict]]:
    """Every gate-passing candidate for one slot as (score, item), best first.

    Joint selection only: ``fetch_price_max`` overrides the price cap pushed into retrieval so
    one fetch can serve several anchors with different remaining budgets (``budget_remaining``
    still gates every candidate individually; None, the greedy path, pushes ``budget_remaining``);
    ``fetch_k`` widens the window so cheap options below the top 40 are visible; ``gate_cache``
    shares the anchor-independent gate pass across anchors. Defaults are the greedy behaviour.
    """
    # Hard gender filter AT RETRIEVAL TIME (Phase B Part 1).  Previously this call
    # was unfiltered top_k=20, and gender was ONLY a post-hoc score gate below —
    # for a thin slot (e.g. women's footwear) an unfiltered top-20 window could be
    # 20/20 wrong-gender items, emptying the pool instead of ever widening it.
    # HybridRetriever's gender filter (src/retrieval/hybrid_search.py) ALSO
    # excludes "unknown"-gender rows when a gender filter is set — the same
    # conservative "never guess gender in" rule as gender_allowed() below, now
    # enforced on the retrieval window itself, not just on whatever slipped
    # through unfiltered.
    #
    # Multi-family accessory split (2026-07-19 bridal jewellery fix): when the
    # accessory slot's own query spans more than one accessory family (e.g.
    # the bridal "dupatta jewellery clutch ethnic accessory" query spans
    # DUPATTA + BAG + JEWELLERY), retrieving it as ONE combined query lets
    # whichever family dominates lexical/semantic overlap crowd every other
    # family out of the retrieval window entirely (live-proven: jewellery
    # never appeared even in a top-500 window for that combined query — see
    # split_accessory_query_by_family's docstring). Issue one retrieval call
    # per matched family instead and merge the pools (dedup by article_id)
    # so every family gets a genuine, undiluted shot at being scored below —
    # a no-op (single retriever.search call, identical to before) for every
    # single-family accessory query and every non-accessory slot.
    # Price pushdown (2026-11 budget fix): with a budget, push price_max into the fetch (the
    # hybrid retriever applies it to both the dense and BM25 candidate masks) so affordable items
    # that rank below the top-40 by relevance alone become visible, instead of being crowded out
    # and then rejected by the post-fetch budget gate. No-op without a budget.
    _search_filters: dict = {"gender": gender}
    if fetch_price_max is not None:
        _search_filters["price_max"] = fetch_price_max
    elif budget_remaining is not None:
        _search_filters["price_max"] = budget_remaining
    family_queries = (
        split_accessory_query_by_family(query) if slot_name == "accessory" else []
    )

    def _fetch(top_k: int, text: str | None = None) -> list[dict]:
        if not family_queries:
            return retriever.search(text or query, top_k=top_k, filters=_search_filters)
        pool: list[dict] = []
        seen_pool_ids: set[str] = set()
        for family_query in family_queries:
            for item in retriever.search(family_query, top_k=top_k, filters=_search_filters):
                if item["article_id"] not in seen_pool_ids:
                    pool.append(item)
                    seen_pool_ids.add(item["article_id"])
        return pool

    candidates = _fetch(fetch_k)

    # Gender-neutral accessory fallback: fires ONLY when the gendered search
    # above returned nothing for an accessory slot.  Widens to an unfiltered
    # search and accepts ONLY unknown-gender candidates whose product_type/name
    # is a genuinely unisex accessory sub-type (sunglasses, belt, watch, cap) —
    # never a garment.  A women's footwear or a men's kurta slot returning empty
    # is NEVER filled this way; it falls through to honest suppression instead.
    neutral_fallback_ids: set[str] = set()
    if not candidates and slot_name == "accessory":
        widened = retriever.search(
            query,
            top_k=40,
            filters={"price_max": budget_remaining} if budget_remaining is not None else None,
        )
        for item in widened:
            item_gender = (item.get("gender") or "unknown").lower()
            if item_gender == "unknown" and is_gender_neutral_accessory(
                item.get("product_type") or "", item.get("prod_name") or ""
            ):
                candidates.append(item)
                neutral_fallback_ids.add(item["article_id"])

    to_score = candidates
    if gate_cache is not None and not neutral_fallback_ids:
        # Joint selection scores ONE shared pool once per anchor. Every hard gate is independent
        # of the anchor, budget and repeats, so run them once (budget off, nothing seen) and let
        # each anchor re-score only the survivors; the survivors of any anchor are a subset.
        gkey = (slot_name, query, gender, occasion_slug, fetch_price_max, fetch_k, anchor_class)
        passed = gate_cache.get(gkey)
        if passed is None:
            passed = {
                it["article_id"]
                for _, it in _score_candidates(
                    candidates,
                    query=query,
                    slot_name=slot_name,
                    occasion_slug=occasion_slug,
                    gender=gender,
                    anchor_colour="",
                    seen_ids=set(),
                    seen_prod_colour=set(),
                    budget_remaining=None,
                    pairing_stats=None,
                    anchor_class=anchor_class,
                    seen_stores=None,
                    neutral_fallback_ids=set(),
                )
            }
            gate_cache[gkey] = passed
        to_score = [c for c in candidates if c["article_id"] in passed]

    scored = _score_candidates(
        to_score,
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
        formality_override=formality_override,
        price_outlier_cap=price_outlier_cap,
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
            filters={
                **_search_filters,
                "product_type_name": _WESTERN_BOTTOM_FALLBACK_PRODUCT_TYPE,
            },
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
                formality_override=formality_override,
                price_outlier_cap=price_outlier_cap,
            )

    # Tight-budget window widening (2026-11 budget fix): a price cap removes the pricey,
    # on-topic items from the pool, so the cheap items that remain can be dominated by
    # off-slot ones that merely share the query's occasion words (live: a 32-candidate
    # footwear pool under Rs 752 held zero footwear that passed the gates, while two
    # sangeet-coherent pairs sat just below the top-40 cut). Retry ONCE with a wider
    # window, same filters and same hard gates, only when the first pass found nothing.
    if not scored and budget_remaining is not None and budget_remaining > 0:
        # For footwear, also drop the register/embellishment words the slot query carries
        # ("... festive embroidered"): under a tight cap they pull cheap embroidered garments
        # above the cheap footwear. The footwear nouns alone name the slot.
        core = None
        if slot_name == "footwear":
            core_words = [w for w in query.split() if w.lower() in FOOTWEAR_KEYWORDS]
            core = " ".join(core_words) or None
        wide = _fetch(_BUDGET_RETRY_TOP_K, core)
        if len(wide) > len(candidates):
            scored = _score_candidates(
                wide,
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
                formality_override=formality_override,
                price_outlier_cap=price_outlier_cap,
            )

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


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
    formality_override: str | None = None,
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

    # Absurd price-outlier guard (2026-07-25 — closes a gap flagged since
    # 2026-07-23): compose_outfit's own per-slot loop always passes
    # price_outlier_cap into _find_best_candidate (see its computation right
    # before that loop), but this function never did, so "swap the
    # footwear" (etc.) had no protection against an absurdly-priced outlier
    # replacing a single slot — the exact same guard, just never wired into
    # the swap path. Same logic as compose_outfit: anchor against the SEED
    # item's own price (never one of the OTHER unchanged complements, and
    # never the slot being replaced), skipped when the user gave an explicit
    # budget (budget_remaining already bounds price there) or the seed has
    # no catalogue price (owned/uploaded anchor).
    _anchor_price_for_cap = seed_item.get("price_inr") or 0.0
    price_outlier_cap: float | None = None
    if budget_inr is None and _anchor_price_for_cap > 0:
        price_outlier_cap = _anchor_price_for_cap * _PRICE_OUTLIER_FACTOR

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
        price_outlier_cap=price_outlier_cap,
        body_type=body_type,
        body_modifiers=body_modifiers,
        formality_override=formality_override,
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
        "diwali": (
            "kurta bandhgala festive gold embellished"
            if is_men
            else "lehenga saree anarkali gold embellished festive glam"
        ),
        "navratri": (
            "kurta bright colourful festive dance"
            if is_men
            else "chaniya choli lehenga bright colourful dance garba"
        ),
        "karva_chauth": (
            "kurta red traditional ethnic"
            if is_men
            else "red saree lehenga traditional bridal ethnic"
        ),
        "raksha_bandhan": ("kurta casual festive light" if is_men else "kurti casual festive light"),
        "eid": (
            "kurta pajama pastel elegant festive"
            if is_men
            else "anarkali kurta set pastel elegant festive"
        ),
        "party_evening": ("shirt formal party" if is_men else "dress evening party top formal"),
        "office": ("shirt formal office" if is_men else "top blouse formal shirt"),
        "smart_casual": ("shirt casual" if is_men else "top casual blouse"),
        "casual": ("shirt casual tshirt" if is_men else "blouse top casual women"),
        "gym": (
            "t-shirt shorts joggers gym athletic"
            if is_men
            else "sports bra leggings gym athletic activewear"
        ),
    }
    return queries.get(occasion_slug, "top casual")


def _anchor_matches_occasion(item: dict, occasion_slug: str, gender: str) -> bool:
    """Check if an item retrieved as anchor is coherent with the occasion.

    2026-08-11 fix (anchor-vs-complement gate unification): previously ran a
    narrower, hand-duplicated check here (ethnic_lean vs is_ethnic_item only)
    instead of the same is_coherent_candidate gate every COMPLEMENT is
    checked against. That narrower check is a structural no-op for every
    occasion whose ethnic_lean is EITHER -- which includes exactly the 3
    occasions that carry a register gate in is_coherent_candidate specifically
    BECAUSE ethnic_lean alone isn't enough protection for them (office=gate 4,
    gym=gate 5, party_evening=gate 6). An ethnic/festive/casual-marker anchor
    could therefore anchor a look whose OWN complement gate would reject the
    identical item. Confirmed empirically (not just structurally) via
    eval/anchor_complement_gate_class_probe.py and the fixture-driven
    eval/gate_leak_audit.py sweep: office leaks reproduced (4/293 real
    composed looks), same shape confirmed reachable for gym/party_evening
    once a body-type/modifier-varied anchor query surfaces one. Now routes
    through the SAME gate function complements use, so a future gate 4/5/6
    change never needs a second, hand-synced edit here again.

    skip_gender_gate=True: the caller (compose_outfit's anchor resolution,
    and partner.py's mirror of it) already applies gender_allowed() and
    has_gender_text_conflict() explicitly alongside this call — avoids
    running that gate twice, not a correctness change.
    """
    pt = item.get("product_type") or ""
    name = item.get("prod_name") or ""
    item_class = classify_item(pt, name)
    slot_name = next(
        (slot for slot, classes in SLOT_ALLOWED_CLASSES.items() if item_class in classes),
        "top",
    )
    return is_coherent_candidate(
        item, occasion_slug, gender, slot_name, skip_gender_gate=True
    )


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
