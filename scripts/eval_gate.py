#!/usr/bin/env python
"""Relevance regression gate — fails (exit 1) if retrieval/intent/gates drop below thresholds.

Runs the deterministic eval stages (intent, r1, gates — zero LLM calls, free, ~2-4 min
dominated by index load) PLUS the hand-audited strict gold eval (scripts/eval_strict.py,
--mode pipeline — mirrors production's real filter+occasion-gate), and enforces:

    r1 overall precision@5   >= --min-p5        (default 0.80)   [property-based, FLOOR only]
    r1 overall NDCG@10       >= --min-ndcg       (default 0.85)
    intent all-fields-exact  >= --min-intent     (default 0.88, fraction)
    gates                    == 0 errors and every check pass_rate == 1.0
    strict gold precision@5  >= --min-strict-p5  (default 0.70)  [hand-audited, honest]

Thresholds sit deliberately below the 2026-07-11 baselines (property P@5 0.889, NDCG 0.914,
intent 92.4%, gates 100%, strict pipeline P@5 0.764 — reports/strict_eval_POSTFIX_pipeline.txt)
so fixture growth and small-n noise don't false-alarm, while a real ranking/parser/composer
regression trips the gate. The strict check is the one that matters most: property P@5 is a
documented FLOOR (a maxi dress self-grades as a hit for "bodycon dress"), so a code change
that raises property P@5 while quietly lowering strict P@5 is exactly the failure mode this
gate exists to catch.

Usage:
    python scripts/eval_gate.py              # run eval stages, then check
    python scripts/eval_gate.py --no-run     # check the newest existing reports only

Mandatory before any backend deploy — see DEPLOY.md "Relevance regression gate".
Not wired into ci.yml: the retrieval index (data/processed/unified) is gitignored and
CI has no GCS credentials; this gate is a local pre-deploy step by design.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_REPORTS_DIR = _ROOT / "reports"


def newest_report() -> Path:
    reports = sorted(_REPORTS_DIR.glob("model_eval_*.json"))
    if not reports:
        sys.exit("GATE ERROR: no reports/model_eval_*.json found — run scripts/eval_model.py")
    return reports[-1]


def run_eval_stages() -> None:
    cmd = [
        sys.executable, str(_ROOT / "scripts" / "eval_model.py"),
        "--stages", "intent,r1,gates", "--seed", "42",
    ]
    print(f"gate: running {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, cwd=_ROOT)
    if result.returncode != 0:
        sys.exit(f"GATE FAIL: eval_model.py exited {result.returncode}")


def run_strict_eval() -> dict:
    with tempfile.TemporaryDirectory() as d:
        json_path = Path(d) / "strict_gate.json"
        cmd = [
            sys.executable, str(_ROOT / "scripts" / "eval_strict.py"),
            "--mode", "pipeline", "--json-out", str(json_path),
        ]
        print(f"gate: running {' '.join(cmd[1:])}")
        result = subprocess.run(cmd, cwd=_ROOT)
        if result.returncode != 0:
            sys.exit(f"GATE FAIL: eval_strict.py exited {result.returncode}")
        return json.loads(json_path.read_text(encoding="utf-8"))


def check(
    report_path: Path, min_p5: float, min_ndcg: float, min_intent: float,
    strict_result: dict, min_strict_p5: float,
) -> int:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    r1 = payload.get("r1")
    intent = payload.get("intent")
    gates = payload.get("gates")
    for name, stage in (("r1", r1), ("intent", intent), ("gates", gates)):
        if stage is None:
            failures.append(f"{name}: stage missing from report (was it run?)")
    if failures:
        print(f"gate: {report_path.name}")
        for f in failures:
            print(f"  FAIL {f}")
        return 1

    p5 = r1["overall"]["precision_at_5"]
    ndcg = r1["overall"]["ndcg_at_10"]
    # Despite the `_pct` suffix, eval_model.py stores this as a 0-1 fraction (0.924 = 92.4%).
    intent_frac = intent["all_fields_exact_pct"]
    n_errors = gates["n_errors"]
    checks = gates["checks_summary"]

    rows = [
        (f"r1 precision@5      {p5:.3f}  (min {min_p5:.2f}, n={r1['overall']['n']})",
         p5 >= min_p5),
        (f"r1 NDCG@10          {ndcg:.3f}  (min {min_ndcg:.2f})", ndcg >= min_ndcg),
        (f"intent all-exact    {intent_frac * 100:.1f}%  (min {min_intent * 100:.1f}%, "
         f"n={intent['n_queries']})", intent_frac >= min_intent),
        (f"gates errors        {n_errors}  (must be 0)", n_errors == 0),
    ]
    for check_name, summary in checks.items():
        rows.append((
            f"gate {check_name:<18} {summary['pass_rate']:.2f} (n={summary['n']}, must be 1.0)",
            summary["pass_rate"] == 1.0,
        ))
    strict_p5 = strict_result["precision_at_5"]
    rows.append((
        f"strict gold P@5     {strict_p5:.3f}  (min {min_strict_p5:.2f}, "
        f"n={strict_result['n_scored']}, unlabeled={strict_result['n_unlabeled']})",
        strict_p5 >= min_strict_p5 and strict_result["n_unlabeled"] == 0,
    ))

    print(f"gate: {report_path.name}")
    ok = True
    for line, passed in rows:
        print(f"  {'PASS' if passed else 'FAIL'} {line}")
        ok = ok and passed
    # ASCII only: this line prints to cp1252 PowerShell consoles where em dashes mangle.
    print(f"gate: {'ALL PASS' if ok else 'REGRESSION - do not deploy'}")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-run", action="store_true",
                        help="Skip running eval_model.py stages; check the newest existing "
                             "report. The strict gold eval always re-runs (fast, ~15s).")
    parser.add_argument("--min-p5", type=float, default=0.80)
    parser.add_argument("--min-ndcg", type=float, default=0.85)
    parser.add_argument("--min-intent", type=float, default=0.88,
                        help="Minimum intent all-fields-exact, as a 0-1 fraction")
    parser.add_argument("--min-strict-p5", type=float, default=0.70,
                        help="Minimum hand-audited strict gold precision@5 (pipeline mode)")
    args = parser.parse_args()

    if not args.no_run:
        run_eval_stages()
    strict_result = run_strict_eval()
    sys.exit(check(
        newest_report(), args.min_p5, args.min_ndcg, args.min_intent,
        strict_result, args.min_strict_p5,
    ))


if __name__ == "__main__":
    main()
