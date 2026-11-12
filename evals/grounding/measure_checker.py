"""Measure the grounding checker's own precision/recall on the hand-labelled set.

    python -m evals.grounding.measure_checker [--json out.json]

Violation-level: an expected (kind, text) is matched by a reported violation of the same kind whose
span contains the expected text (case-insensitive). Case-level: "flagged at all" vs "expected to be
flagged". Disagreements are printed so a label error is distinguishable from a checker error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.grounding.checker import check

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LABELS = FIXTURES / "grounding_checker_labels.json"
HOLDOUT = FIXTURES / "grounding_checker_labels_holdout.json"


def measure(path: Path = LABELS) -> dict[str, Any]:
    """Run the checker over every labelled case; return P/R and the disagreements."""
    data = json.loads(path.read_text(encoding="utf-8"))
    pool = data["items"]
    tp = fp = fn = 0
    case_tp = case_fp = case_fn = case_tn = 0
    problems: list[dict[str, Any]] = []
    for case in data["cases"]:
        items = [pool[k] for k in case["items"]]
        got = check(case["text"], items, user_text=case.get("user", ""))
        exp = [tuple(e) for e in case["expected"]]
        unmatched = list(got)
        missed = []
        for kind, text in exp:
            hit = next((v for v in unmatched if v.kind == kind and text.lower() in v.text.lower()), None)
            if hit is None:
                missed.append((kind, text))
            else:
                unmatched.remove(hit)
        tp += len(exp) - len(missed)
        fn += len(missed)
        fp += len(unmatched)
        flagged, expected_flag = bool(got), bool(exp)
        case_tp += flagged and expected_flag
        case_fp += flagged and not expected_flag
        case_fn += (not flagged) and expected_flag
        case_tn += (not flagged) and not expected_flag
        if missed or unmatched:
            problems.append({
                "id": case["id"], "text": case["text"], "missed": missed,
                "spurious": [(v.kind, v.text) for v in unmatched],
            })

    def ratio(a: int, b: int) -> float:
        return round(a / b, 4) if b else 1.0

    return {
        "n_cases": len(data["cases"]),
        "n_expected_violations": tp + fn,
        "violation_level": {"tp": tp, "fp": fp, "fn": fn,
                            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn)},
        "case_level": {"tp": case_tp, "fp": case_fp, "fn": case_fn, "tn": case_tn,
                       "precision": ratio(case_tp, case_tp + case_fp),
                       "recall": ratio(case_tp, case_tp + case_fn)},
        "disagreements": problems,
    }


def main() -> None:
    """CLI entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None, help="default: both label files")
    args = ap.parse_args()
    results = {}
    for name, path in ([(args.labels.name, args.labels)] if args.labels
                       else [("tuned", LABELS), ("holdout", HOLDOUT)]):
        res = results[name] = measure(path)
        print(name, json.dumps({k: v for k, v in res.items() if k != "disagreements"}, indent=2))
        for p in res["disagreements"]:
            print("DISAGREE", name, json.dumps(p))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
