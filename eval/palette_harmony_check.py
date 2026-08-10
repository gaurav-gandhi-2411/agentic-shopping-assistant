#!/usr/bin/env python
"""Palette-harmony mechanical check — DIAGNOSIS ONLY, not wired into any gate.

Measures whether thresholding the REAL `src.agents.outfit.coherence.colour_score()`
into a pass/fail gate (`palette_score(look, threshold) -> bool`) is a safe signal,
against the 20-look hand-scored ground truth in
`eval/fixtures/coherence_calibration.yaml`'s `palette_ok` flag.

Per `eval/README.md`'s standing rule ("Proxy metrics may gate exploration; only the
hand-labeled strict eval gates shipping"), this script's own output is NOT itself a
shipping decision — it is the measurement that INFORMS one. See
`reports/mechanical_coherence_checks_<ts>.md` for the recommendation.

METHOD:
  - Each look's items[0] is treated as the anchor (matches this fixture's own
    convention — the hero garment is always listed first: blazer/kurta/lehenga-
    skirt/jeans/tank-top/dress; verified by inspection of all 20 looks, not
    assumed). Every other item in the look is scored against the anchor's colour
    via the REAL colour_score(candidate_colour, anchor_colour, occasion_slug).
  - palette_score(look, threshold) returns FAIL (False) if ANY non-anchor item's
    colour_score < threshold; PASS (True) otherwise.
  - Swept across threshold in {0.2, 0.3, 0.4, 0.5, 0.6} (the task's requested
    sweep), reporting confusion-matrix counts and FP/FN rate at each value
    against ground truth `palette_ok`.

KNOWN DATA WRINKLE (found during this measurement, reported not silently fixed):
  4 of the 20 looks use occasion_slug="wedding_reception" (good_003, bad_003,
  bad_007, borderline_004). The REAL occasion slug in this codebase is
  "reception" (src/agents/outfit/occasions.py — "wedding_reception" is not a key
  and has no alias in _OCCASION_ALIASES). `get_occasion("wedding_reception")`
  therefore silently falls back to "casual" (formality 1, EITHER), AND
  colour_score's dedicated reception-palette override block (its `if
  occasion_slug == "reception":` literal string check) never fires for these 4
  looks either, since colour_score compares the RAW parameter, not the resolved
  Occasion.slug. This is a fixture labeling bug, not a colour_score bug — this
  script does not silently correct eval/fixtures/coherence_calibration.yaml
  (out of scope for this diagnosis pass), but DOES run the sweep twice: once
  against the raw occasion_slug as literally stored (the faithful, "what colour_
  score would really see today" reading) and once with wedding_reception ->
  reception corrected, to show how much this data issue moves the numbers.

Usage:
    python -m eval.palette_harmony_check
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agents.outfit.coherence import colour_score  # noqa: E402

_FIXTURE_PATH = _ROOT / "eval" / "fixtures" / "coherence_calibration.yaml"

THRESHOLDS: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)

# See module docstring's KNOWN DATA WRINKLE section.
_OCCASION_SLUG_FIX: dict[str, str] = {"wedding_reception": "reception"}


@dataclass
class ItemScore:
    prod_name: str
    colour: str
    score: float
    item_pass: bool


@dataclass
class LookVerdict:
    look_id: str
    bucket: str
    occasion_slug: str
    palette_ok_expected: bool
    check_verdict_pass: bool  # True = check says PASS (no palette fail)
    per_item: list[ItemScore]

    @property
    def outcome(self) -> str:
        """Confusion-matrix bucket name relative to palette_ok ground truth."""
        if self.check_verdict_pass and self.palette_ok_expected:
            return "TN"  # check says fine, ground truth says fine
        if not self.check_verdict_pass and not self.palette_ok_expected:
            return "TP"  # check flags it, ground truth says it's really bad
        if not self.check_verdict_pass and self.palette_ok_expected:
            return "FP"  # check flags it, ground truth says it's actually fine
        return "FN"  # check says fine, ground truth says it's really bad


def load_fixture() -> list[dict[str, Any]]:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["looks"]


def palette_score(
    look: dict[str, Any], threshold: float, *, fix_occasion_slug: bool = False
) -> tuple[bool, list[ItemScore]]:
    """Return (check_passes, per_item_scores).

    check_passes is True (look PASSES the palette check) iff every non-anchor
    item's colour_score against the anchor (items[0]) colour is >= threshold.
    A look with 0 or 1 items vacuously passes (nothing to clash with).
    """
    items = look["items"]
    occasion_slug = look["occasion_slug"]
    if fix_occasion_slug:
        occasion_slug = _OCCASION_SLUG_FIX.get(occasion_slug, occasion_slug)
    if len(items) < 2:
        return True, []
    anchor = items[0]
    anchor_colour = anchor.get("colour") or ""
    per_item: list[ItemScore] = []
    all_pass = True
    for item in items[1:]:
        c = item.get("colour") or ""
        score = colour_score(c, anchor_colour, occasion_slug)
        item_pass = score >= threshold
        per_item.append(
            ItemScore(prod_name=item.get("prod_name", "?"), colour=c, score=score, item_pass=item_pass)
        )
        if not item_pass:
            all_pass = False
    return all_pass, per_item


def run_sweep(
    looks: list[dict[str, Any]], *, fix_occasion_slug: bool
) -> dict[float, list[LookVerdict]]:
    results: dict[float, list[LookVerdict]] = {}
    for threshold in THRESHOLDS:
        verdicts: list[LookVerdict] = []
        for look in looks:
            check_pass, per_item = palette_score(look, threshold, fix_occasion_slug=fix_occasion_slug)
            expected = look["hand_score"]["palette_ok"]
            verdicts.append(
                LookVerdict(
                    look_id=look["id"],
                    bucket=look["bucket"],
                    occasion_slug=look["occasion_slug"],
                    palette_ok_expected=expected,
                    check_verdict_pass=check_pass,
                    per_item=per_item,
                )
            )
        results[threshold] = verdicts
    return results


def confusion_counts(verdicts: list[LookVerdict]) -> dict[str, int]:
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for v in verdicts:
        counts[v.outcome] += 1
    return counts


def render_report(
    looks: list[dict[str, Any]],
    raw_results: dict[float, list[LookVerdict]],
    fixed_results: dict[float, list[LookVerdict]],
) -> str:
    lines = ["# Palette-harmony threshold sweep — raw run (no code touched, diagnosis only)", ""]
    lines.append(f"Fixture: `eval/fixtures/coherence_calibration.yaml` ({len(looks)} looks).")
    lines.append("Anchor convention: items[0] per look (verified by inspection, not assumed).")
    lines.append("")

    for label, results in (("RAW occasion_slug (as stored in fixture)", raw_results),
                            ("FIXED occasion_slug (wedding_reception -> reception)", fixed_results)):
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Threshold | TP | TN | FP | FN | FP rate (of 19 palette_ok=true) | "
                      "FN rate (of 1 palette_ok=false) |")
        lines.append("|---|---|---|---|---|---|---|")
        for threshold in THRESHOLDS:
            verdicts = results[threshold]
            counts = confusion_counts(verdicts)
            n_true = counts["TN"] + counts["FP"]
            n_false = counts["TP"] + counts["FN"]
            fp_rate = counts["FP"] / n_true if n_true else float("nan")
            fn_rate = counts["FN"] / n_false if n_false else float("nan")
            lines.append(
                f"| {threshold} | {counts['TP']} | {counts['TN']} | {counts['FP']} | "
                f"{counts['FN']} | {fp_rate:.1%} (n={n_true}) | {fn_rate:.1%} (n={n_false}) |"
            )
        lines.append("")
        for threshold in THRESHOLDS:
            lines.append(f"### threshold={threshold}")
            lines.append("| Look | Bucket | Expected palette_ok | Check verdict | Outcome | Worst item score |")
            lines.append("|---|---|---|---|---|---|")
            for v in results[threshold]:
                worst = min((it.score for it in v.per_item), default=float("nan"))
                lines.append(
                    f"| {v.look_id} | {v.bucket} | {v.palette_ok_expected} | "
                    f"{'PASS' if v.check_verdict_pass else 'FAIL'} | {v.outcome} | "
                    f"{worst:.2f} |" if v.per_item else
                    f"| {v.look_id} | {v.bucket} | {v.palette_ok_expected} | PASS | {v.outcome} | n/a |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Print the sweep to stdout only — this is a diagnosis tool, not a report
    generator. `reports/mechanical_coherence_checks_<ts>.md` (written separately,
    by hand, from this script's output) is the artifact of record."""
    looks = load_fixture()
    raw_results = run_sweep(looks, fix_occasion_slug=False)
    fixed_results = run_sweep(looks, fix_occasion_slug=True)
    report = render_report(looks, raw_results, fixed_results)
    print(report)


if __name__ == "__main__":
    main()
