#!/usr/bin/env python3
"""Analyze structured llm_call log lines captured during a telemetry run.

Inputs:
  server_telemetry.log         - uvicorn stdout captured during the run
  scripts/telemetry_requests.json - per-request timing from telemetry_collect.py

Outputs:
  scripts/telemetry_stats.json - raw statistics dict
  prints a formatted summary table
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

LOG_PATH   = Path("server_telemetry.log")
REQ_PATH   = Path("scripts/telemetry_requests.json")
STATS_PATH = Path("scripts/telemetry_stats.json")


def load_llm_calls(log_path: Path) -> list[dict]:
    """Parse every llm_call JSON record from the server log."""
    calls: list[dict] = []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                outer = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if outer.get("logger") != "src.llm.client":
                continue
            msg = outer.get("msg", "")
            if not msg.startswith("{"):
                continue
            try:
                inner = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if inner.get("event") != "llm_call":
                continue
            inner["_ts_str"]         = outer.get("ts", "")
            inner["_conversation_id"] = outer.get("conversation_id", "")
            # Parse ISO ts -> epoch float
            try:
                inner["_ts_epoch"] = datetime.strptime(
                    inner["_ts_str"], "%Y-%m-%dT%H:%M:%S"
                ).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                inner["_ts_epoch"] = 0.0
            calls.append(inner)
    return calls


def load_requests(req_path: Path) -> list[dict]:
    with open(req_path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Match LLM calls to HTTP turns
# ---------------------------------------------------------------------------

def match_calls_to_turns(calls: list[dict], reqs: list[dict]) -> list[dict]:
    """Assign each llm_call to its HTTP request by time window.

    Returns a list of 'turns', each containing:
      seq, label, conversation_id, duration_ms, llm_calls (ordered by _ts_epoch)
    """
    # Sort reqs by sent_at ascending
    sorted_reqs = sorted((r for r in reqs if "error" not in r), key=lambda r: r["sent_at"])

    turns: list[dict] = []
    unmatched: list[dict] = []

    for req in sorted_reqs:
        lo = req["sent_at"] - 1.0     # 1s grace before request
        hi = req["received_at"] + 2.0 # 2s grace after response
        matched = [c for c in calls if lo <= c["_ts_epoch"] <= hi]
        # Sort matched by timestamp
        matched.sort(key=lambda c: c["_ts_epoch"])
        turns.append({
            "seq": req["seq"],
            "label": req["label"],
            "conversation_id": req.get("conversation_id", ""),
            "duration_ms": req["duration_ms"],
            "action": req.get("action", "?"),
            "llm_calls": matched,
        })

    # Report any unmatched log entries
    matched_turn_ids = {c["turn_id"] for t in turns for c in t["llm_calls"]}
    unmatched = [c for c in calls if c.get("turn_id") not in matched_turn_ids]
    if unmatched:
        print(f"  WARNING:  {len(unmatched)} llm_call log entries could not be matched to a request window")

    return turns


# ---------------------------------------------------------------------------
# Classify call type within a turn
# ---------------------------------------------------------------------------

def classify_turn(llm_calls: list[dict]) -> list[tuple[str, dict]]:
    """Return [(call_type, call_dict), ...] for the calls in one turn.

    Classification:
      1-call turn  : router only
      2-call turn  : router, respond  (OOC / clarify / direct-answer path)
      3-call turn  : router, reranker, respond  (standard search path)
      4+-call turn : router, reranker, extra*, respond  (shouldn't happen; extra classified as 'other')
    """
    n = len(llm_calls)
    if n == 0:
        return []
    if n == 1:
        return [("router", llm_calls[0])]
    if n == 2:
        return [("router", llm_calls[0]), ("respond", llm_calls[1])]
    if n == 3:
        return [("router", llm_calls[0]), ("reranker", llm_calls[1]), ("respond", llm_calls[2])]
    # 4+: router, (n-2 extra), respond
    result = [("router", llm_calls[0])]
    for c in llm_calls[1:-1]:
        result.append(("other", c))
    result.append(("respond", llm_calls[-1]))
    return result


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (pct / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    frac = idx - lo
    return round(s[lo] + frac * (s[hi] - s[lo]), 1)


def stats_for(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0, "p50": 0, "p90": 0, "max": 0, "sum": 0}
    return {
        "count":  len(values),
        "mean":   round(sum(values) / len(values), 1),
        "p50":    percentile(values, 50),
        "p90":    percentile(values, 90),
        "max":    round(max(values), 1),
        "sum":    round(sum(values), 6),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(turns: list[dict]) -> dict[str, Any]:
    by_type: dict[str, dict[str, list]] = {
        "router":   {"input": [], "output": [], "cost": []},
        "reranker": {"input": [], "output": [], "cost": []},
        "respond":  {"input": [], "output": [], "cost": []},
        "other":    {"input": [], "output": [], "cost": []},
    }
    per_turn_input:  list[float] = []
    per_turn_output: list[float] = []
    per_turn_cost:   list[float] = []

    for turn in turns:
        classified = classify_turn(turn["llm_calls"])
        t_in = sum(c["input_tokens"] for _, c in classified)
        t_out = sum(c["output_tokens"] for _, c in classified)
        t_cost = sum(c.get("usd_cost", 0.0) for _, c in classified)
        if classified:
            per_turn_input.append(float(t_in))
            per_turn_output.append(float(t_out))
            per_turn_cost.append(float(t_cost))
        for ctype, c in classified:
            by_type[ctype]["input"].append(float(c["input_tokens"]))
            by_type[ctype]["output"].append(float(c["output_tokens"]))
            by_type[ctype]["cost"].append(float(c.get("usd_cost", 0.0)))

    call_type_stats: dict[str, Any] = {}
    for ctype, data in by_type.items():
        if not data["input"]:
            continue
        call_type_stats[ctype] = {
            "input_tokens":  stats_for(data["input"]),
            "output_tokens": stats_for(data["output"]),
            "total_cost_usd": round(sum(data["cost"]), 6),
        }

    total_cost = sum(c.get("usd_cost", 0.0) for t in turns for c in t["llm_calls"])

    return {
        "sample_size_turns": len([t for t in turns if t["llm_calls"]]),
        "total_llm_calls":   sum(len(t["llm_calls"]) for t in turns),
        "total_cost_usd":    round(total_cost, 6),
        "by_call_type":      call_type_stats,
        "per_turn": {
            "input_tokens":  stats_for(per_turn_input),
            "output_tokens": stats_for(per_turn_output),
            "cost_usd":      stats_for(per_turn_cost),
        },
    }


def print_report(stats: dict[str, Any], plan_input_assumption: int = 3523) -> None:
    print("\n" + "=" * 70)
    print("  TOKEN TELEMETRY - LOCAL BASELINE")
    print("=" * 70)
    print(f"  Sample turns: {stats['sample_size_turns']}")
    print(f"  Total LLM calls: {stats['total_llm_calls']}")
    print(f"  Total cost: ${stats['total_cost_usd']:.6f}\n")

    print(f"  {'CALL TYPE':<10}  {'COUNT':>5}  "
          f"{'IN_MEAN':>7}  {'IN_P50':>6}  {'IN_P90':>6}  {'IN_MAX':>6}  "
          f"{'OUT_MEAN':>8}  {'OUT_P50':>7}  {'OUT_P90':>7}  {'OUT_MAX':>7}  "
          f"{'COST_USD':>10}")
    print("  " + "-" * 90)

    col_order = ["router", "reranker", "respond", "other"]
    for ctype in col_order:
        if ctype not in stats["by_call_type"]:
            continue
        d  = stats["by_call_type"][ctype]
        i  = d["input_tokens"]
        o  = d["output_tokens"]
        print(
            f"  {ctype:<10}  {i['count']:>5}  "
            f"{i['mean']:>7.0f}  {i['p50']:>6.0f}  {i['p90']:>6.0f}  {i['max']:>6.0f}  "
            f"{o['mean']:>8.0f}  {o['p50']:>7.0f}  {o['p90']:>7.0f}  {o['max']:>7.0f}  "
            f"${d['total_cost_usd']:>9.6f}"
        )

    pt = stats["per_turn"]
    i  = pt["input_tokens"]
    o  = pt["output_tokens"]
    c  = pt["cost_usd"]
    print("\n  PER-TURN TOTALS")
    print(f"  {'':10}  {'COUNT':>5}  {'MEAN':>7}  {'P50':>6}  {'P90':>6}  {'MAX':>6}")
    print("  " + "-" * 50)
    print(f"  {'input_tok':<10}  {i['count']:>5}  {i['mean']:>7.0f}  {i['p50']:>6.0f}  {i['p90']:>6.0f}  {i['max']:>6.0f}")
    print(f"  {'output_tok':<10}  {o['count']:>5}  {o['mean']:>7.0f}  {o['p50']:>6.0f}  {o['p90']:>6.0f}  {o['max']:>6.0f}")
    print(f"  {'cost_usd':<10}  {c['count']:>5}  {c['mean']:>7.6f}  {c['p50']:>6.6f}  {c['p90']:>6.6f}  {c['max']:>6.6f}")

    delta_pct = (i["mean"] - plan_input_assumption) / plan_input_assumption * 100
    flag = " <- FLAG: >20% delta" if abs(delta_pct) > 20 else ""
    print(f"\n  PRODUCTION_PLAN assumption: {plan_input_assumption:,} input tokens/turn (mean)")
    print(f"  Observed mean:              {i['mean']:,.0f} tokens/turn")
    print(f"  Delta:                      {delta_pct:+.1f}%{flag}")
    print("=" * 70 + "\n")


def main() -> None:
    if not LOG_PATH.exists():
        sys.exit(f"Log file not found: {LOG_PATH}")
    if not REQ_PATH.exists():
        sys.exit(f"Request file not found: {REQ_PATH}")

    print(f"Loading log from {LOG_PATH} ...", flush=True)
    calls = load_llm_calls(LOG_PATH)
    print(f"  {len(calls)} llm_call entries found")

    print(f"Loading requests from {REQ_PATH} ...", flush=True)
    reqs = load_requests(REQ_PATH)
    print(f"  {len(reqs)} requests loaded")

    turns = match_calls_to_turns(calls, reqs)
    stats = analyze(turns)
    print_report(stats)

    with open(STATS_PATH, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Raw stats -> {STATS_PATH}")

    return stats


if __name__ == "__main__":
    main()
