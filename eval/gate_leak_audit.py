#!/usr/bin/env python
"""Gate-leak audit — DIAGNOSIS ONLY, not wired into any gate.

Generates real composed looks the SAME zero-LLM-cost way `scripts/eval_model.py`'s
"gates" stage does (`run_gates_stage`: `compose_outfit`/`compose_couple_look` called
directly against `eval/fixtures/model_eval_queries.yaml`'s `gates.compose` rows — no
LLM, no live router). For every item that ships in every composed look, re-runs
`src.agents.outfit.coherence.is_coherent_candidate` post-hoc and reports how often a
shipped item would fail its own coherence gate if re-checked.

WHY THIS IS NOT A FALSE-POSITIVE-RATE MEASUREMENT (stated up front, not buried):
this is a deterministic RE-APPLICATION of a function this codebase already trusts
(it gates candidate admission during composition) — not a new judgment call being
validated against ground truth. There is no independent "was this item actually
wrong" label to compute precision/recall against. What IS measurable and reported:
how many leaks, where they cluster (occasion/slot), and a best-effort coarse
classification of WHY each leak survived (see `_classify_leak` — genuine documented
carve-out vs. an unintended gap), because those two classes call for different
follow-up (carve-out: no action; gap: a real bug to fix).

KEY STRUCTURAL FINDING (verified by reading composer.py, not assumed): the anchor/
seed item in `compose_outfit` is NEVER passed through `is_coherent_candidate` at
all — it only passes the much narrower `_anchor_matches_occasion` (ethnic_lean vs.
`is_ethnic_item` only; no gender-text-conflict check, no casual-marker/festive-
marker/Jodhpuri/ethnic-footwear-pairing check, no athletic-footwear check). Every
COMPLEMENT (`_score_candidates`, composer.py:1066) DOES run through
`is_coherent_candidate` unconditionally — no pool-underflow protection exists on
this path (that carve-out lives in `graph.py`'s plain-search `search_node`, a
DIFFERENT code path from `compose_outfit`, per direct code inspection). This script
reports seed-item leaks and complement-item leaks separately for exactly this
reason — they are structurally different findings, not the same leak class.

Usage:
    python -m eval.gate_leak_audit
    python -m eval.gate_leak_audit --data-dir data/processed/unified
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agents.outfit.coherence import is_coherent_candidate  # noqa: E402
from src.agents.outfit.slots import (  # noqa: E402
    _FORMAL_ETHNIC_OCCASIONS,
    gender_allowed,
    is_gender_neutral_accessory,
)
from src.catalogue.cleaning import has_gender_text_conflict, is_loungewear_text  # noqa: E402

_FIXTURE_PATH = _ROOT / "eval" / "fixtures" / "model_eval_queries.yaml"
_DEFAULT_DATA_DIR = _ROOT / "data" / "processed" / "unified"


@dataclass
class LeakRecord:
    query_id: str
    category: str | None
    occasion_slug: str
    gender: str
    slot_name: str
    is_seed: bool
    prod_name: str
    classification: str


@dataclass
class AuditResult:
    n_looks_composed: int = 0
    n_looks_with_error: int = 0
    n_items_checked: int = 0
    n_leaks: int = 0
    leaks: list[LeakRecord] = field(default_factory=list)

    @property
    def n_looks_with_leak(self) -> int:
        return len({(leak.query_id) for leak in self.leaks})


def load_queries() -> list[dict[str, Any]]:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["queries"]


def _classify_leak(
    item: dict[str, Any], occasion_slug: str, gender: str, slot_name: str, is_seed: bool
) -> str:
    """Best-effort, coarse classification of WHY an item fails is_coherent_candidate
    on re-check. Not gate-number-exact (is_coherent_candidate's own True/False
    result is the ground truth for PASS/FAIL; this only explains likely cause for
    triage). Checked in the same priority order a human would reason through it.
    """
    if is_seed:
        # Structural, not a carve-out inside is_coherent_candidate itself — the
        # seed simply never reaches this function in compose_outfit at all (see
        # module docstring's KEY STRUCTURAL FINDING).
        return "seed_item_never_gated_by_composer"

    name = item.get("prod_name") or item.get("display_name") or ""
    pt = item.get("product_type") or ""
    item_gender = (item.get("gender") or "unknown").lower()

    if has_gender_text_conflict(name, gender) or not gender_allowed(item_gender, gender):
        if (
            slot_name == "accessory"
            and item_gender == "unknown"
            and is_gender_neutral_accessory(pt, name)
        ):
            return "neutral_fallback_accessory_gender_carveout"
        return "gender_mismatch"

    if occasion_slug in _FORMAL_ETHNIC_OCCASIONS and is_loungewear_text(name):
        # This is the EXACT carve-out documented in is_coherent_candidate's own
        # docstring (loungewear/nightwear deliberately not gated THERE so the
        # graph.py pool-underflow-protected caller doesn't break) — but per the
        # KEY STRUCTURAL FINDING above, compose_outfit's own _score_candidates
        # has its OWN separate hard loungewear gate (composer.py:1043) that DOES
        # run unconditionally, so this should be structurally unreachable for a
        # composer.py-sourced complement. Flagged distinctly so a hit here is
        # investigated as a possible real gap in that second gate, not assumed
        # intentional.
        return "loungewear_pattern_present_investigate"

    return "occasion_register_or_marker_gate"


def run_audit(
    catalogue_df: Any, retriever: Any, queries: list[dict[str, Any]], *, limit: int | None = None
) -> AuditResult:
    from src.agents.outfit.composer import compose_outfit
    from src.agents.outfit.partner import compose_couple_look

    result = AuditResult()
    n_composed = 0
    for q in queries:
        gates = q.get("gates")
        if not gates or not gates.get("compose"):
            continue
        if limit is not None and n_composed >= limit:
            break
        n_composed += 1
        if n_composed % 25 == 0:
            print(f"  ...composed {n_composed} looks so far", flush=True)

        occasion_slug = gates.get("occasion_slug")
        gender = gates.get("gender") or "women"
        budget_inr = gates.get("budget_inr")
        couple = bool(gates.get("couple"))
        body_type = gates.get("body_type")
        expected = q.get("expected_intent") or {}
        body_modifiers = expected.get("body_modifiers") or []

        try:
            if couple:
                boards: list[dict[str, Any]] = list(
                    compose_couple_look(
                        catalogue_df, retriever, occasion_slug=occasion_slug,
                        partner_gender=gender, budget_inr=budget_inr,
                        brand_gender_default="women",
                    )
                )
            else:
                boards = [
                    compose_outfit(
                        catalogue_df, retriever, occasion_slug=occasion_slug, gender=gender,
                        budget_inr=budget_inr, body_type=body_type, body_modifiers=body_modifiers,
                    )
                ]
        except Exception:  # noqa: BLE001 — a compose failure is a data point, not a crash
            result.n_looks_with_error += 1
            continue

        result.n_looks_composed += 1
        for board in boards:
            if board.get("seed_item") is None:
                continue
            board_gender = board.get("gender") or gender
            board_occasion = board.get("occasion") or occasion_slug

            items: list[tuple[dict[str, Any], str, bool]] = [(board["seed_item"], "top", True)]
            for comp in board.get("complements") or []:
                items.append((comp, comp.get("_slot") or "top", False))

            for item, slot_name, is_seed in items:
                result.n_items_checked += 1
                passed = is_coherent_candidate(item, board_occasion, board_gender, slot_name)
                if not passed:
                    result.n_leaks += 1
                    result.leaks.append(
                        LeakRecord(
                            query_id=q["id"],
                            category=q.get("category"),
                            occasion_slug=board_occasion,
                            gender=board_gender,
                            slot_name=slot_name,
                            is_seed=is_seed,
                            prod_name=item.get("prod_name") or item.get("display_name") or "?",
                            classification=_classify_leak(
                                item, board_occasion, board_gender, slot_name, is_seed
                            ),
                        )
                    )
    return result


def render_report(result: AuditResult) -> str:
    lines = ["# Gate-leak audit — raw run (no code touched, diagnosis only)", ""]
    lines.append(
        f"Composed looks: {result.n_looks_composed} (compose errors: {result.n_looks_with_error}). "
        f"Items re-checked: {result.n_items_checked}. Leaks: {result.n_leaks} "
        f"({result.n_leaks / result.n_items_checked:.1%} of items checked) "
        f"across {result.n_looks_with_leak} distinct looks "
        f"({result.n_looks_with_leak / result.n_looks_composed:.1%} of composed looks)."
    )
    lines.append("")

    by_class = Counter(leak.classification for leak in result.leaks)
    lines.append("## By classification")
    lines.append("| Classification | Count |")
    lines.append("|---|---|")
    for cls, count in by_class.most_common():
        lines.append(f"| {cls} | {count} |")
    lines.append("")

    by_occasion = Counter(leak.occasion_slug for leak in result.leaks)
    lines.append("## By occasion")
    lines.append("| Occasion | Count |")
    lines.append("|---|---|")
    for occ, count in by_occasion.most_common():
        lines.append(f"| {occ} | {count} |")
    lines.append("")

    by_slot = Counter(leak.slot_name for leak in result.leaks)
    lines.append("## By slot")
    lines.append("| Slot | Count |")
    lines.append("|---|---|")
    for slot, count in by_slot.most_common():
        lines.append(f"| {slot} | {count} |")
    lines.append("")

    lines.append("## Full leak list")
    lines.append("| Query | Category | Occasion | Gender | Slot | Seed? | Item | Classification |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for leak in result.leaks:
        lines.append(
            f"| {leak.query_id} | {leak.category} | {leak.occasion_slug} | {leak.gender} | "
            f"{leak.slot_name} | {leak.is_seed} | {leak.prod_name[:70]} | {leak.classification} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(_DEFAULT_DATA_DIR))
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of gates.compose queries composed (default: all). "
        "Useful for a fast smoke run; omit for the full-fixture audit.",
    )
    args = parser.parse_args()

    from scripts.eval_harness import _CONFIG_PATH, _load_catalogue_df
    from src.catalogue.loader import load_config
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    data_dir = Path(args.data_dir)
    config = load_config(_CONFIG_PATH)
    print(f"Loading retrieval indices from {data_dir}...")
    catalogue_df = _load_catalogue_df(data_dir)
    dense = DenseRetriever.load(config, data_dir)
    sparse = SparseRetriever.load(config, data_dir)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)

    queries = load_queries()
    print(f"Loaded {len(queries)} queries; running gate-leak audit (limit={args.limit})...")
    result = run_audit(catalogue_df, retriever, queries, limit=args.limit)
    report = render_report(result)
    print(report)


if __name__ == "__main__":
    main()
