#!/usr/bin/env python3
"""Run a fixed set of query IDs N rounds and report pass/fail per query.

Usage:
  python scripts/probe_queries.py --query-ids ST5 N4 --rounds 5 --provider groq
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def run_round(query_ids: list[str], provider: str, round_n: int) -> dict[str, str]:
    """Run one round via eval_harness.py --query-id; returns {id: 'PASS'|'FAIL'}."""
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "eval_harness.py"),
        "--provider", provider,
        "--query-id", *query_ids,
    ]
    print(f"\n--- Round {round_n} ---", flush=True)
    _result = subprocess.run(cmd, capture_output=False, text=True)

    # Find the most recently written JSON report for this provider
    reports = sorted(
        (_ROOT / "reports").glob(f"eval_results_*_{provider}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not reports:
        print(f"  [warn] no report found for round {round_n}", flush=True)
        return {qid: "ERROR" for qid in query_ids}

    latest = reports[-1]
    with open(latest, encoding="utf-8") as fh:
        data = json.load(fh)

    outcomes: dict[str, str] = {}
    for r in data.get("results", []):
        if r["id"] in query_ids:
            outcomes[r["id"]] = r["status"]
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-ids", nargs="+", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--inter-round-delay", type=float, default=5.0,
                        help="Seconds between rounds (on top of intra-round rate-limit delays)")
    args = parser.parse_args()

    query_ids = args.query_ids
    tally: dict[str, list[str]] = defaultdict(list)

    for i in range(1, args.rounds + 1):
        outcomes = run_round(query_ids, args.provider, i)
        for qid in query_ids:
            status = outcomes.get(qid, "ERROR")
            tally[qid].append(status)
            print(f"  {qid}: {status}", flush=True)
        if i < args.rounds:
            print(f"  [sleeping {args.inter_round_delay}s between rounds]", flush=True)
            time.sleep(args.inter_round_delay)

    print("\n" + "=" * 50, flush=True)
    print(f"PROBE RESULTS  ({args.rounds} rounds, provider={args.provider})", flush=True)
    print("=" * 50, flush=True)
    for qid in query_ids:
        results = tally[qid]
        passes = results.count("PASS")
        fails  = results.count("FAIL")
        errors = results.count("ERROR")
        verdict = "VARIANCE" if passes >= 3 else "INVESTIGATE"
        error_str = f"  errors={errors}" if errors else ""
        print(f"  {qid}: {passes}/{args.rounds} PASS  ({fails} FAIL{error_str})  [{verdict}]",
              flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
