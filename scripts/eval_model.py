#!/usr/bin/env python
"""Pre-launch model-evaluation harness — honest per-stage scorecard.

Usage:
    python scripts/eval_model.py --help
    python scripts/eval_model.py --stages all --fixtures eval/fixtures/model_eval_queries.yaml
    python scripts/eval_model.py --stages intent,r1 --fixtures eval/fixtures/model_eval_queries.yaml
    python scripts/eval_model.py --stages judge --judge-sample 40 --seed 42

Stages (see module docstring sections below for the exact contract of each):
    intent  — src.agents.intent_parser.parse_intent, zero LLM, single-turn only.
    r1      — HybridRetriever.search pure-retrieval precision/NDCG/recall, zero LLM.
    gates   — compose_outfit / compose_couple_look deterministic gate checks, zero LLM.
    judge   — Ollama LLM-as-judge (occasion-fit + coherence) on a sample of stage-3 looks.
    e2e     — full multi-turn graph runs (Ollama) for latency/error/empty-result/precision.

JUDGE RUBRIC (verbatim — do not edit without updating any eval report that cites it):
    Given a composed outfit look (seed item + complement items, each with product name,
    colour and product type) and the occasion the user asked to be styled for, rate:
      (a) occasion_score (1-5): "Would a stylist say this outfit suits the stated
          occasion's register/palette?" 1 = completely wrong (e.g. a mixed-gender
          board, or eveningwear at a daytime haldi), 5 = excellent fit.
      (b) coherence_score (1-5): "Do these items work together as one look (colour,
          formality, style)?" 1 = incoherent items thrown together, 5 = a cohesive,
          well-styled outfit.
    Output STRICT JSON only: {"occasion_score": <1-5 int>, "coherence_score": <1-5 int>,
    "reason": "<one-sentence explanation>"}.
    Calibration anchors: 4 synthetic looks (2 good, 2 deliberately bad — mixed-gender,
    and a haldi look built entirely of black velvet eveningwear) are always injected
    into the judged sample. If the judge scores an anchor outside its expected bucket
    (bad > 2, good < 4 on either axis), the whole stage is marked low-confidence and a
    "JUDGE UNRELIABLE" warning is printed — per LLM-judge-as-proxy best practice, this
    stage is never treated as ground truth.

DEVIATION FROM eval_harness.py (documented, not hidden — see final task report):
    scripts.eval_harness.build_components() returns only (agent, llm, config) — it does
    not expose the HybridRetriever or catalogue_df that compose_outfit/compose_couple_look
    and HybridRetriever.search() need directly (the retriever is captured in the compiled
    LangGraph's node closures, unreachable from the returned graph object). Rather than
    modify eval_harness.py (out of this task's scope), _build_components() below mirrors
    its retriever-assembly lines and reuses its _load_catalogue_df() helper directly.
    It ALSO defaults to data/processed/unified (the live, deployed, gender/store-aware
    61,883-item catalogue — see api/main.py's ensure_index_dir default) rather than
    eval_harness.py's hardcoded data/processed root, which is a stale 20,000-item,
    H&M-only catalogue with no gender/store columns at all (confirmed by inspection:
    root catalogue.parquet dated 2026-06-07, no `gender`/`store` columns; unified
    catalogue.parquet dated 2026-07-09, has both). Every gender-filtered R1/GATES query
    would silently return zero results against the stale root catalogue since
    HybridRetriever's gender filter excludes rows lacking a gender column entirely —
    that would make this harness dishonest by construction. --data-dir overrides this.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# ── path setup (mirrors scripts/eval_harness.py) ────────────────────────────
_SCRIPTS_DIR = Path(__file__).parent
_ROOT = _SCRIPTS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_DEFAULT_FIXTURES = _ROOT / "eval" / "fixtures" / "model_eval_queries.yaml"
_DEFAULT_REPORTS_DIR = _ROOT / "reports"
# See the DEVIATION note in the module docstring above for why this differs from
# eval_harness.py's hardcoded `data/processed` root.
_DEFAULT_DATA_DIR = _ROOT / "data" / "processed" / "unified"

_ALL_STAGES: tuple[str, ...] = ("intent", "r1", "gates", "judge", "e2e")

# Novelty/costume denylist — copied verbatim from scripts/browser_proof.py's
# PB_NOVELTY_RE (credit: that module, ~line 638) rather than imported, because
# browser_proof.py imports playwright, an unwanted dependency for this harness.
NOVELTY_RE = re.compile(r"\b(piano|guitar|novelty|quirky|costume)\b", re.IGNORECASE)

# Fields compared for exact-match intent-parser accuracy (Stage 1). body_modifiers
# and store_filter are intentionally excluded — the task spec lists these 7 only.
_INTENT_FIELDS: tuple[str, ...] = (
    "garment_type",
    "gender",
    "colour",
    "occasion",
    "budget_max_inr",
    "body_type",
    "is_product_query",
)


# ============================================================================
# Fixture / schema loading
# ============================================================================


def load_fixture_queries(path: Path) -> list[dict[str, Any]]:
    """Load and return the `queries` list from a model_eval_queries.yaml fixture file.

    Raises KeyError if the file is missing the top-level `queries` key, and any
    yaml.YAMLError on malformed YAML — both fail loudly rather than silently
    returning an empty list, per the "errors are part of the API" convention.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["queries"]


# ============================================================================
# Pure metric functions — no project/heavy imports; safe to unit-test in
# isolation without loading the retrieval index, catalogue, or an LLM client.
# ============================================================================


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of `values`, or 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def item_matches_must(item: dict[str, Any], must: dict[str, Any]) -> bool:
    """Return True iff `item` satisfies every constraint in `must`.

    Supported keys (per the eval fixture schema):
      product_type_contains: list[str] — at least one term must appear as a
          substring of item["product_type"] OR item["prod_name"] (lowercased).
      gender_in: list[str] — item["gender"] (lowercased) must be one of these.
    An empty/missing `must` is vacuously satisfied (returns True).
    """
    product_type = str(item.get("product_type") or "").lower()
    prod_name = str(item.get("prod_name") or "").lower()
    combined = f"{product_type} {prod_name}"

    terms = must.get("product_type_contains") or []
    if terms and not any(str(t).lower() in combined for t in terms):
        return False

    gender_in = must.get("gender_in") or []
    if gender_in:
        item_gender = str(item.get("gender") or "unknown").lower()
        if item_gender not in {str(g).lower() for g in gender_in}:
            return False

    return True


def item_matches_graded(item: dict[str, Any], graded: dict[str, Any]) -> bool:
    """Return True iff `item` satisfies the (optional) graded bonus criteria.

    Currently supports `colour_in: list[str]` — item["colour"] (lowercased) must
    be one of these. An empty/missing `graded` block never matches (there is
    nothing to grade on top of `must`).
    """
    colours = graded.get("colour_in") or []
    if not colours:
        return False
    item_colour = str(item.get("colour") or "").lower()
    return item_colour in {str(c).lower() for c in colours}


def item_gain(item: dict[str, Any], relevance: dict[str, Any]) -> int:
    """Return the NDCG relevance gain for `item` under `relevance`.

    0 — fails `must`.
    1 — satisfies `must` only.
    2 — satisfies `must` AND the `graded` bonus criteria.
    """
    must = relevance.get("must") or {}
    if not item_matches_must(item, must):
        return 0
    graded = relevance.get("graded") or {}
    if graded and item_matches_graded(item, graded):
        return 2
    return 1


def precision_at_k(items: list[dict[str, Any]], relevance: dict[str, Any], k: int) -> float:
    """Fraction of the top-k retrieved `items` satisfying `relevance["must"]`.

    Denominator is the number of items actually available in the top-k window
    (min(k, len(items))), not k itself — so a query that only retrieved 3 items
    for a k=5 precision is scored over those 3, not diluted by 2 phantom misses.
    Returns 0.0 when `items` is empty.
    """
    top = items[:k]
    if not top:
        return 0.0
    must = relevance.get("must") or {}
    n_match = sum(1 for it in top if item_matches_must(it, must))
    return n_match / len(top)


def dcg_at_k(gains: list[int], k: int) -> float:
    """Standard discounted cumulative gain: sum((2^gain - 1) / log2(rank + 1))."""
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(items: list[dict[str, Any]], relevance: dict[str, Any], k: int) -> float:
    """Normalized DCG@k against the IDEAL ordering of the SAME retrieved list.

    Deliberately not normalized against a full-catalogue ideal (we don't have
    complete relevance judgments over the universe) — the ideal is the same
    gain multiset, re-sorted descending. Returns 0.0 when the ideal DCG is 0
    (no retrieved item has any gain).

    Hand-computed example (also asserted in tests/test_eval_model_metrics.py):
    gains = [2, 0, 1], k=3.
      DCG   = (2^2-1)/log2(2) + (2^0-1)/log2(3) + (2^1-1)/log2(4)
            =        3/1      +        0/1.585  +        1/2
            = 3 + 0 + 0.5 = 3.5
      ideal sorted gains = [2, 1, 0]
      idealDCG = 3/1 + 1/1.58496 + 0/2 = 3 + 0.63093 + 0 = 3.63093
      NDCG@3 = 3.5 / 3.63093 ≈ 0.96394
    """
    gains = [item_gain(it, relevance) for it in items[:k]]
    actual_dcg = dcg_at_k(gains, k)
    ideal_dcg = dcg_at_k(sorted(gains, reverse=True), k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def recall_at_k(retrieved_ids: list[str], universe_ids: set[str], k: int) -> float:
    """Fraction of `universe_ids` present among the top-k `retrieved_ids`.

    Returns 0.0 when `universe_ids` is empty (nothing to recall). Callers are
    responsible for only reporting this when len(universe_ids) <= 100 (see
    catalogue_universe_ids' docstring and the markdown report legend) — recall
    against a several-thousand-item universe at k=50 is not meaningful.
    """
    if not universe_ids:
        return 0.0
    top_ids = set(retrieved_ids[:k])
    return len(top_ids & universe_ids) / len(universe_ids)


def catalogue_universe_ids(catalogue_df: Any, must: dict[str, Any]) -> set[str]:
    """Return the set of article_ids in `catalogue_df` satisfying `must`, via one
    vectorized pandas pass (no python-level row loop).

    Mirrors item_matches_must's semantics using the catalogue's own flat
    `product_type_name`/`prod_name`/`gender` columns (the same underlying values
    HybridRetriever.search() copies into each retrieved item's `product_type`/
    `prod_name`/`gender` keys). `catalogue_df` is expected to be indexed by
    article_id (HybridRetriever.__init__ does this via .set_index("article_id")).
    """
    import pandas as pd  # local import — this is the only function needing pandas

    mask = pd.Series(True, index=catalogue_df.index)

    terms = [str(t).lower() for t in (must.get("product_type_contains") or [])]
    if terms:
        pt_col = catalogue_df.get("product_type_name")
        name_col = catalogue_df.get("prod_name")
        pt_lower = (
            pt_col.fillna("").astype(str).str.lower()
            if pt_col is not None
            else pd.Series("", index=catalogue_df.index)
        )
        name_lower = (
            name_col.fillna("").astype(str).str.lower()
            if name_col is not None
            else pd.Series("", index=catalogue_df.index)
        )
        term_mask = pd.Series(False, index=catalogue_df.index)
        for term in terms:
            term_mask = term_mask | pt_lower.str.contains(term, regex=False)
            term_mask = term_mask | name_lower.str.contains(term, regex=False)
        mask &= term_mask

    gender_in = [str(g).lower() for g in (must.get("gender_in") or [])]
    if gender_in:
        gender_col = catalogue_df.get("gender")
        if gender_col is None:
            mask &= False  # no gender column at all -> a gender-scoped universe is empty
        else:
            mask &= gender_col.fillna("unknown").astype(str).str.lower().isin(gender_in)

    return set(catalogue_df.index[mask].astype(str))


# ============================================================================
# GATES stage — pure gate-check logic (dict-based, no heavy imports)
# ============================================================================


def _items_of_look(look: dict[str, Any]) -> list[dict[str, Any]]:
    """Return [seed_item] + complements for one compose_outfit()-shaped look dict
    (seed_item omitted when None, e.g. an empty/failed composition)."""
    items: list[dict[str, Any]] = []
    seed = look.get("seed_item")
    if seed:
        items.append(seed)
    items.extend(look.get("complements") or [])
    return items


def _gates_slots_summary(
    look_bundle: dict[str, Any] | tuple[dict[str, Any], dict[str, Any]],
    couple: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (empty_slots, suppressed_slots) for a look, or for BOTH boards
    concatenated when `couple` is True (compose_couple_look returns a 2-tuple)."""
    if couple:
        primary, partner = look_bundle
        return (
            (primary.get("empty_slots") or []) + (partner.get("empty_slots") or []),
            (primary.get("suppressed_slots") or []) + (partner.get("suppressed_slots") or []),
        )
    return look_bundle.get("empty_slots") or [], look_bundle.get("suppressed_slots") or []


def check_gender_pure(
    look_bundle: dict[str, Any] | tuple[dict[str, Any], dict[str, Any]],
    expected_gender: str | None,
    couple: bool,
) -> bool:
    """True iff every item in the look (or, for a couple pair, every item on
    EACH board) carries the gender that board itself claims to be composed for.

    Each board's own reported "gender" field is used as the expected value for
    that board (falling back to `expected_gender` for the single-look case when
    the look dict doesn't carry one) — this matches compose_outfit's own
    gender_allowed() gate contract rather than assuming a single shared gender
    across a mixed-gender couple pair.
    """
    boards = list(look_bundle) if couple else [look_bundle]
    for board in boards:
        board_gender = board.get("gender") or expected_gender
        if board_gender is None:
            continue
        items = _items_of_look(board)
        if any((it.get("gender") or "").lower() != board_gender.lower() for it in items):
            return False
    return True


def check_budget_respected(
    look_bundle: dict[str, Any] | tuple[dict[str, Any], dict[str, Any]],
    budget_inr: float | None,
    couple: bool,
) -> bool:
    """True iff every board's budget_total_inr respects `budget_inr` INDEPENDENTLY.

    For a couple pair this is a PER-PERSON cap (each board's own total must be
    <= budget_inr), matching src.agents.outfit.partner.compose_couple_look's
    documented "per-person, not a 50/50 split" contract. None budget_inr is a
    full no-op (always True).
    """
    if budget_inr is None:
        return True
    boards = list(look_bundle) if couple else [look_bundle]
    for board in boards:
        total = board.get("budget_total_inr")
        if total is not None and total > budget_inr:
            return False
    return True


def check_no_novelty(
    look_bundle: dict[str, Any] | tuple[dict[str, Any], dict[str, Any]],
    couple: bool,
) -> bool:
    """True iff no item's prod_name/display_name matches NOVELTY_RE, across the
    look (or both boards, for a couple pair)."""
    boards = list(look_bundle) if couple else [look_bundle]
    for board in boards:
        for it in _items_of_look(board):
            name = it.get("prod_name") or it.get("display_name") or ""
            if NOVELTY_RE.search(name):
                return False
    return True


def evaluate_suppression_honest(
    suppressed_slots: list[dict[str, Any]], empty_slots: list[str]
) -> bool:
    """True iff every slot in `empty_slots` has a non-empty reason recorded in
    `suppressed_slots`. Vacuously True when there are no empty slots at all —
    there is nothing to be dishonest about.
    """
    if not empty_slots:
        return True
    reasons_by_slot = {s.get("slot"): s.get("reason") for s in suppressed_slots}
    return all(bool(reasons_by_slot.get(slot)) for slot in empty_slots)


def evaluate_gate_checks(
    look_bundle: dict[str, Any] | tuple[dict[str, Any], dict[str, Any]],
    *,
    checks_wanted: list[str],
    expected_gender: str | None,
    budget_inr: float | None,
    data_ceiling_tags: list[str],
    couple: bool,
) -> dict[str, Any]:
    """Run every named check in `checks_wanted` against `look_bundle`.

    Returns {"checks": {name: bool, ...}, "data_ceiling": bool}.

    DATA-CEILING PASS-CONVERSION (hard reporting requirement — see module
    docstring's legend note): an honestly-suppressed slot (suppressed_slots
    carries a real reason) on a query tagged with `data_ceiling_tags` is the
    CORRECT, expected behaviour for a genuinely thin catalogue category — not a
    model/code defect. `suppression_honest` is already True whenever the
    suppression was honest (independent of data_ceiling_tags); this function
    additionally flags the whole result `data_ceiling=True` whenever
    data_ceiling_tags is set AND at least one slot was actually empty/suppressed,
    so callers/report-builders can break these rows out of the MODEL/CODE
    failure tally entirely rather than conflating "expected thin inventory" with
    "the composer is broken". A DISHONEST suppression (empty slot, no reason
    recorded) is never converted — that is a real code defect regardless of
    data_ceiling_tags.
    """
    empty_slots, suppressed_slots = _gates_slots_summary(look_bundle, couple)
    checks: dict[str, bool] = {}
    if "gender_pure" in checks_wanted:
        checks["gender_pure"] = check_gender_pure(look_bundle, expected_gender, couple)
    if "budget_respected" in checks_wanted:
        checks["budget_respected"] = check_budget_respected(look_bundle, budget_inr, couple)
    if "no_novelty" in checks_wanted:
        checks["no_novelty"] = check_no_novelty(look_bundle, couple)
    if "suppression_honest" in checks_wanted:
        checks["suppression_honest"] = evaluate_suppression_honest(suppressed_slots, empty_slots)

    is_data_ceiling = bool(data_ceiling_tags) and bool(empty_slots)
    return {"checks": checks, "data_ceiling": is_data_ceiling}


# ============================================================================
# JUDGE stage — prompt building + defensive JSON parsing (pure)
# ============================================================================

JUDGE_SYSTEM_PROMPT = (
    "You are an expert fashion stylist evaluating a composed outfit for a specific "
    "occasion. Respond ONLY with strict JSON — no prose, no markdown code fences."
)


def build_judge_prompt(occasion_slug: str, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build the {role, content} chat messages for one judge call. See the module
    docstring's JUDGE RUBRIC section for the exact wording this mirrors."""
    lines = [
        f"- {it.get('prod_name') or it.get('display_name') or '?'} "
        f"(colour: {it.get('colour') or '?'}, type: {it.get('product_type') or '?'})"
        for it in items
    ]
    item_block = "\n".join(lines) if lines else "(no items)"
    user = (
        f"Occasion: {occasion_slug}\n"
        f"Outfit items:\n{item_block}\n\n"
        "Rate this outfit 1-5 on each axis:\n"
        "(a) occasion_score — would a stylist say this outfit suits the stated "
        "occasion's register/palette?\n"
        "(b) coherence_score — do these items work together as one cohesive look?\n\n"
        'Return STRICT JSON only: {"occasion_score": <1-5 int>, "coherence_score": '
        '<1-5 int>, "reason": "<one sentence>"}'
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_judge_json(text: str | None) -> dict[str, Any] | None:
    """Defensively extract {"occasion_score": int, "coherence_score": int, "reason":
    str} from raw LLM output. Tolerates surrounding prose/markdown fences around the
    JSON object. Returns None when no valid object with both required in-range
    (1-5) integer fields can be parsed — callers should retry once, then score null.
    """
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    occasion_score = obj.get("occasion_score")
    coherence_score = obj.get("coherence_score")
    if isinstance(occasion_score, bool) or isinstance(coherence_score, bool):
        return None  # bool is an int subclass in Python — explicitly reject it
    if not isinstance(occasion_score, int) or not isinstance(coherence_score, int):
        return None
    if not (1 <= occasion_score <= 5) or not (1 <= coherence_score <= 5):
        return None
    return {
        "occasion_score": occasion_score,
        "coherence_score": coherence_score,
        "reason": str(obj.get("reason", "")),
    }


# Calibration anchors (Stage 4, mandatory) — hand-written, never drawn from the
# catalogue. 2 deliberately GOOD (occasion-matched, palette-coherent), 2
# deliberately BAD (mixed-gender board; a haldi look built entirely from
# black-velvet eveningwear). See run_judge_stage's calibration check.
_ANCHOR_GOOD_OFFICE: dict[str, Any] = {
    "occasion_slug": "office",
    "items": [
        {"prod_name": "Tailored Navy Blazer", "colour": "Dark Blue", "product_type": "blazer",
         "gender": "women"},
        {"prod_name": "White Cotton Shirt", "colour": "White", "product_type": "shirt",
         "gender": "women"},
        {"prod_name": "Charcoal Tailored Trousers", "colour": "Dark Grey",
         "product_type": "trousers", "gender": "women"},
        {"prod_name": "Black Leather Pumps", "colour": "Black", "product_type": "footwear",
         "gender": "women"},
    ],
}
_ANCHOR_GOOD_HALDI: dict[str, Any] = {
    "occasion_slug": "haldi",
    "items": [
        {"prod_name": "Yellow Cotton Anarkali", "colour": "Yellow", "product_type": "anarkali",
         "gender": "women"},
        {"prod_name": "Mustard Cotton Dupatta", "colour": "Yellow", "product_type": "dupatta",
         "gender": "women"},
        {"prod_name": "Marigold Jhumka Earrings", "colour": "Yellow", "product_type": "jewellery",
         "gender": "women"},
        {"prod_name": "Tan Flat Sandals", "colour": "Brown", "product_type": "footwear",
         "gender": "women"},
    ],
}
_ANCHOR_BAD_MIXED_GENDER: dict[str, Any] = {
    "occasion_slug": "party_evening",
    "items": [
        {"prod_name": "Black Party Dress", "colour": "Black", "product_type": "dress",
         "gender": "women"},
        {"prod_name": "Men's Formal Necktie", "colour": "Black", "product_type": "tie",
         "gender": "men"},
        {"prod_name": "Men's Leather Oxford Shoes", "colour": "Brown", "product_type": "footwear",
         "gender": "men"},
    ],
}
_ANCHOR_BAD_HALDI_VELVET: dict[str, Any] = {
    "occasion_slug": "haldi",
    "items": [
        {"prod_name": "Black Velvet Evening Gown", "colour": "Black", "product_type": "gown",
         "gender": "women"},
        {"prod_name": "Black Sequin Clutch", "colour": "Black", "product_type": "bag",
         "gender": "women"},
        {"prod_name": "Black Stiletto Heels", "colour": "Black", "product_type": "footwear",
         "gender": "women"},
    ],
}
JUDGE_CALIBRATION_ANCHORS: tuple[dict[str, Any], ...] = (
    {"label": "good_office", "expected_bucket": "good", **_ANCHOR_GOOD_OFFICE},
    {"label": "good_haldi", "expected_bucket": "good", **_ANCHOR_GOOD_HALDI},
    {"label": "bad_mixed_gender", "expected_bucket": "bad", **_ANCHOR_BAD_MIXED_GENDER},
    {"label": "bad_haldi_velvet", "expected_bucket": "bad", **_ANCHOR_BAD_HALDI_VELVET},
)


def anchor_passes_calibration(scored: dict[str, Any] | None, expected_bucket: str) -> bool:
    """True iff a scored anchor result falls in its expected bucket.

    "good"  -> occasion_score >= 4 AND coherence_score >= 4.
    "bad"   -> occasion_score <= 2 AND coherence_score <= 2.
    A null score (judge failed to parse twice) is treated as a calibration
    failure too — an unreliable judge that can't even produce valid JSON for an
    unambiguous anchor is not trustworthy on the harder sampled cases either.
    """
    if scored is None:
        return False
    if expected_bucket == "good":
        return scored["occasion_score"] >= 4 and scored["coherence_score"] >= 4
    return scored["occasion_score"] <= 2 and scored["coherence_score"] <= 2


# ============================================================================
# Sampling helpers (deterministic, seed=42, no global seeding)
# ============================================================================


def stratified_sample(
    items: list[dict[str, Any]],
    n: int,
    seed: int,
    key_fn: Any,
) -> list[dict[str, Any]]:
    """Return up to `n` items from `items`, spread as evenly as possible across
    the groups induced by `key_fn`.

    Deterministic: shuffles a copy of `items` with random.Random(seed), buckets
    by key_fn(item) preserving shuffle order, then round-robins across buckets
    (taking each bucket's next unclaimed item in turn) until `n` are selected or
    every bucket is exhausted. Returns all of `items` unchanged (order-shuffled
    is NOT applied) when n >= len(items).
    """
    if n <= 0:
        return []
    if n >= len(items):
        return list(items)

    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    buckets: dict[Any, list[dict[str, Any]]] = {}
    order: list[Any] = []
    for it in shuffled:
        key = key_fn(it)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(it)

    selected: list[dict[str, Any]] = []
    round_idx = 0
    while len(selected) < n:
        progressed = False
        for key in order:
            bucket = buckets[key]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                progressed = True
                if len(selected) == n:
                    break
        round_idx += 1
        if not progressed:
            break
    return selected


# ============================================================================
# STAGE 1 — INTENT (no LLM, single-turn categories only)
# ============================================================================


def score_intent_fields(expected: dict[str, Any], actual: Any) -> dict[str, bool]:
    """Compare `actual` (an IntentV1) against `expected` (expected_intent dict)
    field-by-field for every name in _INTENT_FIELDS. None == null counts as a match.
    """
    return {f: expected.get(f) == getattr(actual, f) for f in _INTENT_FIELDS}


def _mean_field_accuracy(records: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in _INTENT_FIELDS:
        vals = [1.0 if r["field_matches"].get(f) else 0.0 for r in records]
        out[f] = _mean(vals)
    return out


def run_intent_stage(queries: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage 1: parse_intent per-field accuracy. Skips category="refinement" —
    parse_intent is single-turn and refinement queries are only meaningful across
    a multi-turn conversation (see the e2e stage for that coverage)."""
    from src.agents.intent_parser import parse_intent

    per_query: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for q in queries:
        if q.get("category") == "refinement":
            continue
        last_turn = q["turns"][-1]
        expected = q.get("expected_intent") or {}
        actual = parse_intent(last_turn)
        field_matches = score_intent_fields(expected, actual)
        rec = {
            "id": q["id"],
            "category": q.get("category"),
            "field_matches": field_matches,
            "all_exact": all(field_matches.values()),
        }
        per_query.append(rec)
        by_category[q.get("category", "?")].append(rec)

    by_category_out = {
        cat: {
            "n": len(recs),
            "field_accuracy": _mean_field_accuracy(recs),
            "all_fields_exact_pct": _mean([1.0 if r["all_exact"] else 0.0 for r in recs]),
        }
        for cat, recs in by_category.items()
    }
    return {
        "stage": "intent",
        "n_queries": len(per_query),
        "field_accuracy": _mean_field_accuracy(per_query),
        "all_fields_exact_pct": _mean([1.0 if r["all_exact"] else 0.0 for r in per_query]),
        "by_category": by_category_out,
        "per_query": per_query,
    }


# ============================================================================
# STAGE 2 — R1 PURE RETRIEVAL (no LLM)
# ============================================================================


def run_r1_stage(queries: list[dict[str, Any]], retriever: Any) -> dict[str, Any]:
    """Stage 2: HybridRetriever.search precision@5/@10, NDCG@10, recall@50.

    Skips category="refinement" and any query with no `relevance` block.
    Pushes expected_intent.gender into the retriever filter exactly the way the
    live graph does (see src/agents/graph.py's search_node gender-filter push).
    """
    per_query: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for q in queries:
        if q.get("category") == "refinement":
            continue
        relevance = q.get("relevance")
        if not relevance:
            continue

        last_turn = q["turns"][-1]
        expected = q.get("expected_intent") or {}
        gender = expected.get("gender")
        filters = {"gender": gender} if gender else None
        items = retriever.search(last_turn, top_k=50, filters=filters)

        must = relevance.get("must") or {}
        universe_ids = catalogue_universe_ids(retriever.catalogue_df, must)
        universe_size = len(universe_ids)
        retrieved_ids = [it["article_id"] for it in items]

        if 0 < universe_size <= 100:
            recall_50: float | None = recall_at_k(retrieved_ids, universe_ids, 50)
            recall_na_reason = None
        elif universe_size == 0:
            recall_50 = None
            recall_na_reason = "universe size 0 (must-filter matched no catalogue items)"
        else:
            recall_50 = None
            recall_na_reason = f"universe size {universe_size} > 100 — recall@50 not meaningful"

        rec = {
            "id": q["id"],
            "category": q.get("category"),
            "precision_at_5": precision_at_k(items, relevance, 5),
            "precision_at_10": precision_at_k(items, relevance, 10),
            "ndcg_at_10": ndcg_at_k(items, relevance, 10),
            "universe_size": universe_size,
            "recall_at_50": recall_50,
            "recall_na_reason": recall_na_reason,
        }
        per_query.append(rec)
        by_category[q.get("category", "?")].append(rec)

    def _agg(records: list[dict[str, Any]]) -> dict[str, Any]:
        recall_vals = [r["recall_at_50"] for r in records if r["recall_at_50"] is not None]
        return {
            "n": len(records),
            "precision_at_5": _mean([r["precision_at_5"] for r in records]),
            "precision_at_10": _mean([r["precision_at_10"] for r in records]),
            "ndcg_at_10": _mean([r["ndcg_at_10"] for r in records]),
            "recall_at_50": {
                "mean": _mean(recall_vals) if recall_vals else None,
                "n_reported": len(recall_vals),
                "n_na": len(records) - len(recall_vals),
            },
        }

    return {
        "stage": "r1",
        "n_queries": len(per_query),
        "overall": _agg(per_query),
        "by_category": {cat: _agg(recs) for cat, recs in by_category.items()},
        "per_query": per_query,
    }


# ============================================================================
# STAGE 3 — GATES (compose_outfit / compose_couple_look, no LLM)
# ============================================================================


def _aggregate_gate_checks(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    names: set[str] = set()
    for r in rows:
        names |= set(r.get("checks", {}).keys())
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        vals = [1.0 if r["checks"].get(name) else 0.0 for r in rows if name in r.get("checks", {})]
        out[name] = {"pass_rate": _mean(vals), "n": len(vals)}
    return out


def _gate_summaries(per_query: list[dict[str, Any]]) -> tuple[dict, dict]:
    """Split per-query gate rows into MODEL/CODE vs DATA-CEILING buckets and
    aggregate pass-rate-per-check separately for each — the hard reporting
    requirement from evaluate_gate_checks' docstring."""
    model_rows = [r for r in per_query if r.get("checks") and not r.get("data_ceiling")]
    ceiling_rows = [r for r in per_query if r.get("checks") and r.get("data_ceiling")]
    return _aggregate_gate_checks(model_rows), _aggregate_gate_checks(ceiling_rows)


def run_gates_stage(
    queries: list[dict[str, Any]],
    catalogue_df: Any,
    retriever: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Stage 3: deterministic compose_outfit/compose_couple_look gate checks.

    Returns (stage_report, looks_by_id) — looks_by_id feeds Stage 4 (JUDGE)
    without recomposing the same looks twice.
    """
    from src.agents.outfit.composer import compose_outfit
    from src.agents.outfit.partner import compose_couple_look

    per_query: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    looks_by_id: dict[str, dict[str, Any]] = {}

    for q in queries:
        gates = q.get("gates")
        if not gates or not gates.get("compose"):
            continue

        occasion_slug = gates.get("occasion_slug")
        # Some fixture rows (mainly "his and hers"/"couple" phrasing with no explicit
        # gender named, and 2 adversarial occasion-only rows) carry gates.gender=None.
        # compose_outfit/compose_couple_look both require a concrete "men"/"women" —
        # mirror graph.py's own fallback for exactly this case (outfit_node's
        # `_partner_gender = plan.get("gender") or "women"`, ~line 2170) rather than
        # porting the full session/anchor-aware resolve_look_gender chain, which does
        # not apply to a synthetic single-shot GATES probe anyway (no prior session).
        gender = gates.get("gender") or "women"
        budget_inr = gates.get("budget_inr")
        couple = bool(gates.get("couple"))
        body_type = gates.get("body_type")
        expected = q.get("expected_intent") or {}
        body_modifiers = expected.get("body_modifiers") or []
        data_ceiling_tags = q.get("data_ceiling_tags") or []
        checks_wanted = gates.get("checks") or []

        error: str | None = None
        try:
            if couple:
                look_bundle: Any = compose_couple_look(
                    catalogue_df,
                    retriever,
                    occasion_slug=occasion_slug,
                    partner_gender=gender,
                    budget_inr=budget_inr,
                    brand_gender_default="women",
                )
            else:
                look_bundle = compose_outfit(
                    catalogue_df,
                    retriever,
                    occasion_slug=occasion_slug,
                    gender=gender,
                    budget_inr=budget_inr,
                    body_type=body_type,
                    body_modifiers=body_modifiers,
                )
        except Exception as exc:  # noqa: BLE001 — honest per-query failure, never crash the stage
            look_bundle = None
            error = repr(exc)

        if look_bundle is None:
            rec = {
                "id": q["id"], "category": q.get("category"), "error": error,
                "checks": {}, "data_ceiling": False, "empty_slots": [], "suppressed_slots": [],
            }
            per_query.append(rec)
            by_category[q.get("category", "?")].append(rec)
            continue

        gate_result = evaluate_gate_checks(
            look_bundle,
            checks_wanted=checks_wanted,
            expected_gender=gender,
            budget_inr=budget_inr,
            data_ceiling_tags=data_ceiling_tags,
            couple=couple,
        )
        empty_slots, suppressed_slots = _gates_slots_summary(look_bundle, couple)
        looks_by_id[q["id"]] = {
            "look": look_bundle,
            "occasion_slug": occasion_slug,
            "couple": couple,
            "category": q.get("category"),
            "body_type": body_type,
        }
        rec = {
            "id": q["id"],
            "category": q.get("category"),
            "error": None,
            "checks": gate_result["checks"],
            "data_ceiling": gate_result["data_ceiling"],
            "empty_slots": empty_slots,
            "suppressed_slots": suppressed_slots,
        }
        per_query.append(rec)
        by_category[q.get("category", "?")].append(rec)

    checks_summary, data_ceiling_summary = _gate_summaries(per_query)
    by_category_out = {}
    for cat, recs in by_category.items():
        cat_checks, cat_ceiling = _gate_summaries(recs)
        by_category_out[cat] = {
            "n": len(recs),
            "checks_summary": cat_checks,
            "data_ceiling_summary": cat_ceiling,
        }

    n_errors = sum(1 for r in per_query if r.get("error"))
    report = {
        "stage": "gates",
        "n_queries": len(per_query),
        "n_errors": n_errors,
        "checks_summary": checks_summary,
        "data_ceiling_summary": data_ceiling_summary,
        "by_category": by_category_out,
        "per_query": per_query,
    }
    return report, looks_by_id


# ============================================================================
# STAGE 4 — JUDGE (Ollama LLM-as-judge, sampled)
# ============================================================================


def _judge_one(llm: Any, occasion_slug: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Call the judge LLM once, retrying exactly once on a parse failure, per
    the module docstring's JUDGE RUBRIC contract."""
    messages = build_judge_prompt(occasion_slug, items)
    for _ in range(2):
        try:
            raw = llm.chat(messages)
        except Exception:  # noqa: BLE001 — treat a transport error as a parse failure (retry once)
            raw = None
        parsed = parse_judge_json(raw)
        if parsed is not None:
            return parsed
    return None


def run_judge_stage(
    looks_by_id: dict[str, dict[str, Any]],
    llm: Any,
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    """Stage 4: sample composed looks from Stage 3, judge occasion-fit + coherence.

    Stratified across (occasion_slug, couple, body_type) via stratified_sample.
    Always additionally scores the 4 JUDGE_CALIBRATION_ANCHORS — see
    anchor_passes_calibration for the pass bucket and the "JUDGE UNRELIABLE"
    behaviour when an anchor fails.
    """

    def _look_has_seed(info: dict[str, Any]) -> bool:
        look = info["look"]
        if info["couple"]:
            return bool(look[0].get("seed_item"))
        return bool(look.get("seed_item"))

    def _judge_items_for(info: dict[str, Any]) -> list[dict[str, Any]]:
        look = info["look"]
        if info["couple"]:
            return _items_of_look(look[0]) + _items_of_look(look[1])
        return _items_of_look(look)

    candidates = [
        {
            "id": qid,
            "occasion_slug": info["occasion_slug"],
            "category": info["category"],
            "items": _judge_items_for(info),
        }
        for qid, info in looks_by_id.items()
        if _look_has_seed(info)
    ]

    def _key(c: dict[str, Any]) -> Any:
        return (c["occasion_slug"], c["category"])

    sampled = stratified_sample(candidates, min(sample_size, len(candidates)), seed, _key)

    per_query: list[dict[str, Any]] = []
    for c in sampled:
        scored = _judge_one(llm, c["occasion_slug"], c["items"])
        per_query.append(
            {
                "id": c["id"],
                "category": c["category"],
                "occasion_score": scored["occasion_score"] if scored else None,
                "coherence_score": scored["coherence_score"] if scored else None,
                "reason": scored["reason"] if scored else None,
                "null": scored is None,
            }
        )

    anchor_results: list[dict[str, Any]] = []
    for anchor in JUDGE_CALIBRATION_ANCHORS:
        scored = _judge_one(llm, anchor["occasion_slug"], anchor["items"])
        passed = anchor_passes_calibration(scored, anchor["expected_bucket"])
        anchor_results.append(
            {
                "label": anchor["label"],
                "expected_bucket": anchor["expected_bucket"],
                "occasion_score": scored["occasion_score"] if scored else None,
                "coherence_score": scored["coherence_score"] if scored else None,
                "calibration_passed": passed,
            }
        )

    judge_reliable = all(a["calibration_passed"] for a in anchor_results)
    if not judge_reliable:
        print(
            "\n*** JUDGE UNRELIABLE — one or more calibration anchors scored outside "
            "their expected bucket. Stage 4 (JUDGE) results below are LOW-CONFIDENCE. ***\n"
        )

    occ_scores = [r["occasion_score"] for r in per_query if not r["null"]]
    coh_scores = [r["coherence_score"] for r in per_query if not r["null"]]

    def _distribution(scores: list[int]) -> dict[str, int]:
        dist = {str(i): 0 for i in range(1, 6)}
        for s in scores:
            dist[str(s)] += 1
        return dist

    return {
        "stage": "judge",
        "n_sampled": len(sampled),
        "n_scored": len(occ_scores),
        "n_null": len(sampled) - len(occ_scores),
        "occasion_score": {"mean": _mean(occ_scores), "distribution": _distribution(occ_scores)},
        "coherence_score": {"mean": _mean(coh_scores), "distribution": _distribution(coh_scores)},
        "judge_reliable": judge_reliable,
        "anchor_results": anchor_results,
        "per_query": per_query,
        "note": f"LLM-judge proxy, not ground truth (n={len(sampled)}, local llama3.1:8b)",
    }


# ============================================================================
# STAGE 5 — E2E / RESILIENCE (full multi-turn graph runs, Ollama, sampled)
# ============================================================================


def run_e2e_stage(
    queries: list[dict[str, Any]],
    agent: Any,
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    """Stage 5: full multi-turn graph runs via eval_harness._make_state/_invoke.

    Every category="refinement" query is ALWAYS included (highest-risk,
    cross-turn-context class); the remainder is a stratified-by-category sample
    filling the rest of `sample_size`.
    """
    from scripts.eval_harness import _invoke, _make_state

    mandatory = [q for q in queries if q.get("category") == "refinement"]
    others = [q for q in queries if q.get("category") != "refinement"]
    remainder_budget = max(0, sample_size - len(mandatory))
    remainder = stratified_sample(
        others, min(remainder_budget, len(others)), seed, lambda q: q.get("category")
    )
    selected = mandatory + remainder

    per_query: list[dict[str, Any]] = []
    for q in selected:
        messages: list[dict[str, Any]] = []
        retrieved_items: list[dict[str, Any]] = []
        filters: dict[str, Any] = {}
        total_latency = 0.0
        error: str | None = None
        final_result: dict[str, Any] = {}

        try:
            for turn_text in q["turns"]:
                state = _make_state(
                    messages + [{"role": "user", "content": turn_text}],
                    turn_text,
                    retrieved_items,
                    filters,
                )
                result, lat = _invoke(agent, state)
                total_latency += lat
                messages = result.get("messages", messages)
                retrieved_items = result.get("retrieved_items", retrieved_items)
                filters = result.get("filters", filters)
                final_result = result
        except Exception as exc:  # noqa: BLE001 — honest per-query failure, never crash the stage
            error = repr(exc)

        n_items = len(final_result.get("retrieved_items", []) if final_result else [])
        expected = q.get("expected_intent") or {}
        is_product_query = bool(expected.get("is_product_query"))
        relevance = q.get("relevance")

        rec: dict[str, Any] = {
            "id": q["id"],
            "category": q.get("category"),
            "error": error,
            "latency_s": round(total_latency, 3),
            "n_items": n_items,
            "is_product_query": is_product_query,
            "empty_result": is_product_query and n_items == 0 and error is None,
        }
        if relevance and error is None:
            items = final_result.get("retrieved_items", [])
            rec["post_pipeline_precision_at_5"] = precision_at_k(items, relevance, 5)
        per_query.append(rec)

    n = len(per_query)
    n_errors = sum(1 for r in per_query if r["error"])
    lat_values = sorted(r["latency_s"] for r in per_query if r["error"] is None)
    product_rows = [r for r in per_query if r["is_product_query"] and r["error"] is None]
    n_empty = sum(1 for r in product_rows if r["empty_result"])
    precision_vals = [
        r["post_pipeline_precision_at_5"] for r in per_query if "post_pipeline_precision_at_5" in r
    ]

    def _p95(vals: list[float]) -> float | None:
        if not vals:
            return None
        idx = max(0, min(len(vals) - 1, int(len(vals) * 0.95) - 1))
        return vals[idx]

    return {
        "stage": "e2e",
        "n_sampled": n,
        "error_rate": (n_errors / n) if n else 0.0,
        "empty_result_rate": (n_empty / len(product_rows)) if product_rows else None,
        "latency_s": {
            "median": statistics.median(lat_values) if lat_values else None,
            "p95": _p95(lat_values),
            "max": lat_values[-1] if lat_values else None,
            "note": "local Ollama latency is a PROXY only — production uses Groq "
            "(different hardware and model serving stack).",
        },
        "post_pipeline_precision_at_5": {
            "mean": _mean(precision_vals) if precision_vals else None,
            "n": len(precision_vals),
        },
        "per_query": per_query,
    }


# ============================================================================
# Report building
# ============================================================================


def collect_metric_cells(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every per-category metric cell (across intent/r1/gates) into a
    single list of {stage, category, metric, value, n} dicts, for the
    "Weakest areas" report section. GATES data-ceiling rows are intentionally
    excluded here — they are not weaknesses, they are known thin-inventory
    categories (see the module docstring's MODEL/CODE vs DATA-CEILING split).
    """
    cells: list[dict[str, Any]] = []

    intent = payload.get("intent")
    if intent:
        for cat, agg in intent.get("by_category", {}).items():
            for field, val in agg.get("field_accuracy", {}).items():
                cells.append(
                    {"stage": "intent", "category": cat, "metric": f"field_accuracy.{field}",
                     "value": val, "n": agg["n"]}
                )
            cells.append(
                {"stage": "intent", "category": cat, "metric": "all_fields_exact_pct",
                 "value": agg["all_fields_exact_pct"], "n": agg["n"]}
            )

    r1 = payload.get("r1")
    if r1:
        for cat, agg in r1.get("by_category", {}).items():
            cells.append({"stage": "r1", "category": cat, "metric": "precision_at_5",
                          "value": agg["precision_at_5"], "n": agg["n"]})
            cells.append({"stage": "r1", "category": cat, "metric": "precision_at_10",
                          "value": agg["precision_at_10"], "n": agg["n"]})
            cells.append({"stage": "r1", "category": cat, "metric": "ndcg_at_10",
                          "value": agg["ndcg_at_10"], "n": agg["n"]})
            recall = agg["recall_at_50"]
            if recall["mean"] is not None:
                cells.append({"stage": "r1", "category": cat, "metric": "recall_at_50",
                              "value": recall["mean"], "n": recall["n_reported"]})

    gates = payload.get("gates")
    if gates:
        for cat, agg in gates.get("by_category", {}).items():
            for check, stat in agg.get("checks_summary", {}).items():
                cells.append(
                    {"stage": "gates", "category": cat, "metric": f"checks.{check} (MODEL/CODE)",
                     "value": stat["pass_rate"], "n": stat["n"]}
                )

    return cells


def weakest_areas(cells: list[dict[str, Any]], n: int = 3) -> list[dict[str, Any]]:
    """Return the `n` lowest-value cells with n > 0 observations, ascending."""
    scored = [c for c in cells if c["n"] > 0]
    return sorted(scored, key=lambda c: c["value"])[:n]


_LEGEND = """## Legend

- **precision@k** — fraction of the top-k retrieved items that satisfy the query's
  `relevance.must` property constraints. This is a FLOOR on true relevance, not a
  ceiling: an item can satisfy every `must` property (right garment type, right
  gender) and still be a poor stylistic match — property-match precision only
  catches gross retrieval failures, it does not replace human/stylist judgment.
- **NDCG@10** — normalized discounted cumulative gain against the ideal ordering of
  the SAME retrieved list (gains: 0 = fails must, 1 = must-only, 2 = must+graded).
  Not normalized against a full-catalogue ideal — we don't have complete relevance
  judgments over the universe.
- **recall@50** — fraction of ALL must-matching catalogue items appearing in the
  top-50 retrieved. Reported ONLY when the catalogue universe for that query's
  `must` filter is <= 100 items — recall against a several-thousand-item universe
  at k=50 is not a meaningful number (nearly every query would recall ~1-2%
  regardless of retrieval quality). Universe size is always shown; "N/A" rows are
  not failures, they are out of scope for this metric.
- **MODEL/CODE vs DATA-CEILING (GATES stage)** — a query tagged with
  `data_ceiling_tags` (e.g. `womens_footwear`, `jewellery`, `mens_occasionwear`)
  that produces an honestly-suppressed slot (a real reason recorded, not a silent
  drop) is NOT a model or code defect — it reflects genuinely thin catalogue
  inventory for that category. These rows are reported separately
  ("data_ceiling_summary") from true MODEL/CODE gate failures ("checks_summary")
  so a thin-inventory category is never conflated with a composer bug.
- **JUDGE stage** — LLM-as-judge proxy, NOT ground truth. Local llama3.1:8b,
  calibrated against 4 hand-written anchor looks (2 good, 2 deliberately bad). If
  any anchor scores outside its expected bucket, the whole stage is flagged
  "JUDGE UNRELIABLE" and should be treated as low-confidence.
- **E2E latency** — local Ollama wall time is a PROXY; production serves via Groq
  on different hardware with a different model-serving stack. Compare relative
  trends across runs, not absolute numbers, against a production SLO.
"""


def build_markdown_report(payload: dict[str, Any]) -> str:
    """Render the full markdown scorecard from a JSON-shaped `payload` (see main()
    for its assembly). One section per stage that ran, a legend, and a
    "Weakest areas" auto-list of the bottom-3 non-data-ceiling metric cells.
    """
    lines: list[str] = []
    lines.append(f"# Model Evaluation Report — {payload.get('run_timestamp_utc', '')}")
    lines.append("")
    lines.append(f"Stages run: {', '.join(payload.get('stages_run', []))}")
    if payload.get("catalogue_warning"):
        lines.append("")
        lines.append(f"> WARNING: {payload['catalogue_warning']}")
    lines.append("")

    intent = payload.get("intent")
    if intent:
        lines.append("## Stage 1 — INTENT (parse_intent, no LLM)")
        lines.append("")
        all_exact = intent["all_fields_exact_pct"]
        lines.append(f"n={intent['n_queries']}  all-fields-exact={all_exact:.1%}")
        lines.append("")
        lines.append("| category | n | " + " | ".join(_INTENT_FIELDS) + " | all_exact |")
        lines.append("|---|---|" + "---|" * (len(_INTENT_FIELDS) + 1))
        for cat, agg in sorted(intent["by_category"].items()):
            fa = agg["field_accuracy"]
            row = " | ".join(f"{fa[f]:.0%}" for f in _INTENT_FIELDS)
            lines.append(f"| {cat} | {agg['n']} | {row} | {agg['all_fields_exact_pct']:.0%} |")
        lines.append("")

    r1 = payload.get("r1")
    if r1:
        lines.append("## Stage 2 — R1 PURE RETRIEVAL (HybridRetriever, no LLM)")
        lines.append("")
        lines.append(f"n={r1['n_queries']}")
        lines.append("")
        lines.append("| category | n | P@5 | P@10 | NDCG@10 | recall@50 (reported/NA) |")
        lines.append("|---|---|---|---|---|---|")
        for cat, agg in sorted(r1["by_category"].items()):
            recall = agg["recall_at_50"]
            n_total = recall["n_reported"] + recall["n_na"]
            recall_str = (
                f"{recall['mean']:.0%} ({recall['n_reported']}/{n_total})"
                if recall["mean"] is not None
                else f"N/A (0/{recall['n_na']})"
            )
            lines.append(
                f"| {cat} | {agg['n']} | {agg['precision_at_5']:.0%} | "
                f"{agg['precision_at_10']:.0%} | {agg['ndcg_at_10']:.3f} | {recall_str} |"
            )
        lines.append("")

    gates = payload.get("gates")
    if gates:
        lines.append("## Stage 3 — GATES (compose_outfit / compose_couple_look, no LLM)")
        lines.append("")
        lines.append(f"n={gates['n_queries']}  errors={gates['n_errors']}")
        lines.append("")
        lines.append("### MODEL/CODE checks (excludes data-ceiling-tagged rows)")
        lines.append("")
        check_names = sorted(
            {k for agg in gates["by_category"].values() for k in agg["checks_summary"]}
        )
        lines.append("| category | n | " + " | ".join(check_names) + " |")
        lines.append("|---|---|" + "---|" * len(check_names))
        for cat, agg in sorted(gates["by_category"].items()):
            row = " | ".join(
                f"{agg['checks_summary'][c]['pass_rate']:.0%} (n={agg['checks_summary'][c]['n']})"
                if c in agg["checks_summary"] else "—"
                for c in check_names
            )
            lines.append(f"| {cat} | {agg['n']} | {row} |")
        lines.append("")
        lines.append("### DATA-CEILING rows (thin-inventory categories — not model/code failures)")
        lines.append("")
        for cat, agg in sorted(gates["by_category"].items()):
            if agg["data_ceiling_summary"]:
                lines.append(f"- **{cat}**: {agg['data_ceiling_summary']}")
        lines.append("")

    judge = payload.get("judge")
    if judge:
        lines.append("## Stage 4 — JUDGE (Ollama LLM-as-judge, sampled)")
        lines.append("")
        lines.append(f"> {judge['note']}")
        lines.append("")
        reliability = (
            "RELIABLE" if judge["judge_reliable"] else "**UNRELIABLE — see anchor_results**"
        )
        lines.append(f"n_sampled={judge['n_sampled']}  n_scored={judge['n_scored']}  "
                      f"n_null={judge['n_null']}  calibration={reliability}")
        lines.append("")
        lines.append(f"occasion_score mean={judge['occasion_score']['mean']:.2f}  "
                      f"distribution={judge['occasion_score']['distribution']}")
        lines.append(f"coherence_score mean={judge['coherence_score']['mean']:.2f}  "
                      f"distribution={judge['coherence_score']['distribution']}")
        lines.append("")
        lines.append("Calibration anchors:")
        for a in judge["anchor_results"]:
            status = "PASS" if a["calibration_passed"] else "FAIL"
            lines.append(
                f"- {a['label']} (expected {a['expected_bucket']}): "
                f"occasion={a['occasion_score']} coherence={a['coherence_score']} [{status}]"
            )
        lines.append("")

    e2e = payload.get("e2e")
    if e2e:
        lines.append("## Stage 5 — E2E / RESILIENCE (full multi-turn graph, Ollama, sampled)")
        lines.append("")
        lines.append(f"> {e2e['latency_s']['note']}")
        lines.append("")
        lat = e2e["latency_s"]
        empty_rate_str = (
            f"{e2e['empty_result_rate']:.1%}" if e2e["empty_result_rate"] is not None else "N/A"
        )
        lines.append(
            f"n_sampled={e2e['n_sampled']}  error_rate={e2e['error_rate']:.1%}  "
            f"empty_result_rate={empty_rate_str}"
        )
        lines.append(
            f"latency median={lat['median']}s  p95={lat['p95']}s  max={lat['max']}s"
        )
        pp = e2e["post_pipeline_precision_at_5"]
        if pp["mean"] is not None:
            lines.append(f"post-pipeline precision@5: {pp['mean']:.0%} (n={pp['n']})")
        lines.append("")

    cells = collect_metric_cells(payload)
    weak = weakest_areas(cells, 3)
    if weak:
        lines.append("## Weakest areas (bottom 3 non-data-ceiling metric cells)")
        lines.append("")
        for w in weak:
            lines.append(f"- [{w['stage']}] {w['category']} / {w['metric']}: "
                         f"{w['value']:.1%} (n={w['n']})")
        lines.append("")

    lines.append(_LEGEND)
    return "\n".join(lines)


def build_console_summary(payload: dict[str, Any]) -> str:
    """Compact one-screen summary printed at the end of main()."""
    lines = [f"Stages run: {', '.join(payload.get('stages_run', []))}"]
    intent = payload.get("intent")
    if intent:
        lines.append(f"  intent  n={intent['n_queries']:<4} all_fields_exact="
                     f"{intent['all_fields_exact_pct']:.0%}")
    r1 = payload.get("r1")
    if r1:
        o = r1["overall"]
        lines.append(f"  r1      n={r1['n_queries']:<4} P@5={o['precision_at_5']:.0%} "
                     f"P@10={o['precision_at_10']:.0%} NDCG@10={o['ndcg_at_10']:.3f}")
    gates = payload.get("gates")
    if gates:
        lines.append(f"  gates   n={gates['n_queries']:<4} errors={gates['n_errors']} "
                     f"checks={gates['checks_summary']}")
    judge = payload.get("judge")
    if judge:
        reliability = "reliable" if judge["judge_reliable"] else "UNRELIABLE"
        lines.append(f"  judge   n={judge['n_sampled']:<4} "
                     f"occasion_mean={judge['occasion_score']['mean']:.2f} "
                     f"coherence_mean={judge['coherence_score']['mean']:.2f} [{reliability}]")
    e2e = payload.get("e2e")
    if e2e:
        lines.append(f"  e2e     n={e2e['n_sampled']:<4} error_rate={e2e['error_rate']:.0%} "
                     f"latency_median={e2e['latency_s']['median']}s")
    return "\n".join(lines)


# ============================================================================
# Component assembly (heavy imports — only touched by live stages)
# ============================================================================


def _build_components(need_agent: bool, data_dir: Path) -> dict[str, Any]:
    """Assemble retriever/catalogue_df (+ agent/llm when needed) for live stages.

    See the module docstring's DEVIATION note for why this does not simply call
    scripts.eval_harness.build_components() directly (its return signature does
    not expose the retriever/catalogue_df the R1/GATES stages need).
    """
    import os

    from scripts.eval_harness import _CONFIG_PATH, _load_catalogue_df
    from src.catalogue.loader import load_config
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    config = load_config(_CONFIG_PATH)
    config["llm"]["provider"] = "ollama"
    os.environ["LLM_PROVIDER"] = "ollama"

    print(f"Loading retrieval indices from {data_dir}...")
    catalogue_df = _load_catalogue_df(data_dir)
    dense = DenseRetriever.load(config, data_dir)
    sparse = SparseRetriever.load(config, data_dir)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)

    catalogue_warning = None
    missing = [c for c in ("gender", "store") if c not in retriever.catalogue_df.columns]
    if missing:
        catalogue_warning = (
            f"catalogue at {data_dir} is missing column(s) {missing} — gender-filtered "
            "R1/GATES stages will be unreliable against this index (gender filters "
            "exclude any row without a verified gender, so a missing gender column "
            "silently zeroes every gendered result)."
        )
        print(f"\n*** WARNING: {catalogue_warning} ***\n")

    agent = llm = None
    if need_agent:
        from src.agents.graph import build_graph
        from src.llm.client import get_llm_client

        llm = get_llm_client(config)
        agent = build_graph(
            retriever, catalogue_df, llm, config, streaming_mode=False, router_backend=None
        )

    return {
        "retriever": retriever,
        "catalogue_df": catalogue_df,
        "agent": agent,
        "llm": llm,
        "config": config,
        "catalogue_warning": catalogue_warning,
    }


# ============================================================================
# CLI
# ============================================================================


def _parse_stages(raw: str) -> list[str]:
    if raw.strip() == "all":
        return list(_ALL_STAGES)
    return [s.strip() for s in raw.split(",") if s.strip()]


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-launch model-evaluation harness — honest per-stage scorecard.",
    )
    parser.add_argument(
        "--stages", default="all",
        help="Comma-separated stage list or 'all' (choices: intent,r1,gates,judge,e2e)",
    )
    parser.add_argument(
        "--fixtures", default=str(_DEFAULT_FIXTURES), help="Path to model_eval_queries.yaml",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    parser.add_argument(
        "--judge-sample", type=int, default=40, dest="judge_sample",
        help="Number of stage-3 looks to sample for the JUDGE stage",
    )
    parser.add_argument(
        "--e2e-sample", type=int, default=60, dest="e2e_sample",
        help="Number of queries to sample for the E2E stage (refinement queries always included)",
    )
    parser.add_argument(
        "--out", default=str(_DEFAULT_REPORTS_DIR), help="Output directory for the JSON/MD report",
    )
    parser.add_argument(
        "--data-dir", default=str(_DEFAULT_DATA_DIR),
        help="Catalogue/index directory (default: the live unified cross-store index — "
        "see the module docstring's DEVIATION note)",
    )
    args = parser.parse_args()

    stages = _parse_stages(args.stages)
    unknown = set(stages) - set(_ALL_STAGES)
    if unknown:
        parser.error(f"Unknown stage(s): {sorted(unknown)}. Choices: {_ALL_STAGES}")

    fixtures_path = Path(args.fixtures)
    queries = load_fixture_queries(fixtures_path)
    print(f"Loaded {len(queries)} queries from {fixtures_path}")

    payload: dict[str, Any] = {
        "run_timestamp_utc": _utc_stamp(),
        "fixtures_path": str(fixtures_path),
        "seed": args.seed,
        "stages_run": stages,
    }

    if "intent" in stages:
        print("Stage 1/5: INTENT (no LLM)...")
        payload["intent"] = run_intent_stage(queries)

    need_retriever = any(s in stages for s in ("r1", "gates"))
    need_agent = any(s in stages for s in ("judge", "e2e"))
    components: dict[str, Any] = {}
    if need_retriever or need_agent:
        components = _build_components(need_agent=need_agent, data_dir=Path(args.data_dir))
        payload["catalogue_warning"] = components["catalogue_warning"]

    looks_by_id: dict[str, dict[str, Any]] = {}

    if "r1" in stages:
        print("Stage 2/5: R1 PURE RETRIEVAL (no LLM)...")
        payload["r1"] = run_r1_stage(queries, components["retriever"])

    if "gates" in stages:
        print("Stage 3/5: GATES (compose_outfit/compose_couple_look, no LLM)...")
        gates_report, looks_by_id = run_gates_stage(
            queries, components["catalogue_df"], components["retriever"]
        )
        payload["gates"] = gates_report

    if "judge" in stages:
        print(f"Stage 4/5: JUDGE (Ollama, sample={args.judge_sample})...")
        if not looks_by_id:
            # "judge" requested without "gates" in the same --stages list — components
            # were still built above (need_agent covers "judge"), so retriever/catalogue_df
            # are already available; just compose the looks to sample from.
            print("  (gates stage not requested this run — composing looks for judge sampling)")
            _, looks_by_id = run_gates_stage(
                queries, components["catalogue_df"], components["retriever"]
            )
        payload["judge"] = run_judge_stage(
            looks_by_id, components["llm"], args.seed, args.judge_sample
        )

    if "e2e" in stages:
        print(f"Stage 5/5: E2E / RESILIENCE (Ollama, sample={args.e2e_sample})...")
        payload["e2e"] = run_e2e_stage(queries, components["agent"], args.seed, args.e2e_sample)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = payload["run_timestamp_utc"]
    json_path = out_dir / f"model_eval_{stamp}.json"
    md_path = out_dir / f"model_eval_{stamp}.md"

    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(build_markdown_report(payload), encoding="utf-8")

    print(f"\nJSON saved -> {json_path}")
    print(f"MD   saved -> {md_path}")
    print()
    print(build_console_summary(payload))


if __name__ == "__main__":
    main()
