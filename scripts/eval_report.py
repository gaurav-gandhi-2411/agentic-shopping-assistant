#!/usr/bin/env python
"""Generate a markdown report from eval_harness JSON output.

Usage:
    python scripts/eval_report.py reports/eval_results_20260424_v1.json
    python scripts/eval_report.py          # uses most recent file in reports/
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _fmt_lat(seconds: float) -> str:
    """Format latency; shows Xm Ys for values >= 60s (high values indicate rate-limit retries)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


def generate_markdown(payload: dict) -> str:
    results    = payload["results"]
    run_date   = payload.get("run_date", "?")
    total_time = payload.get("total_time_s", 0)
    n          = len(results)

    pass_n = sum(1 for r in results if r["status"] == "PASS")
    fail_n = sum(1 for r in results if r["status"] == "FAIL")
    err_n  = sum(1 for r in results if r["status"] == "ERROR")
    rate   = pass_n / n * 100 if n else 0

    lats = sorted(r["latency_total"] for r in results if r.get("latency_total", 0) > 0)
    lat_median = statistics.median(lats) if lats else 0.0
    lat_p95    = lats[max(0, int(len(lats) * 0.95) - 1)] if lats else 0.0
    lat_max    = lats[-1] if lats else 0.0

    lines = []

    lines += [
        f"# Evaluation Report — {run_date}",
        "",
        f"**{pass_n}/{n} PASS ({rate:.0f}%)** &nbsp;|&nbsp; "
        f"{fail_n} FAIL &nbsp;|&nbsp; {err_n} ERROR &nbsp;|&nbsp; "
        f"Total: {_fmt_lat(total_time)} &nbsp;|&nbsp; "
        f"Latency median {_fmt_lat(lat_median)}  p95 {_fmt_lat(lat_p95)}  max {_fmt_lat(lat_max)}",
        "",
    ]

    # ── Per-category summary table ────────────────────────────────────────────
    cats = defaultdict(list)
    for r in results:
        cats[r.get("category", "?")].append(r)

    lines += ["## Summary by Category", "", "| Category | Pass | Fail | Error | Rate |",
              "|---|---|---|---|---|"]
    for cat in sorted(cats):
        recs = cats[cat]
        p = sum(1 for r in recs if r["status"] == "PASS")
        f = sum(1 for r in recs if r["status"] == "FAIL")
        e = sum(1 for r in recs if r["status"] == "ERROR")
        pct = p / len(recs) * 100 if recs else 0
        lines.append(f"| {cat} | {p} | {f} | {e} | {pct:.0f}% |")
    lines.append("")

    # ── Full results table ────────────────────────────────────────────────────
    lines += ["## All Results", "",
              "| ID | Category | Status | Items | Latency | Failed Checks |",
              "|---|---|---|---|---|---|"]
    for r in results:
        icon = "PASS" if r["status"] == "PASS" else ("FAIL" if r["status"] == "FAIL" else "ERR")
        fail_str = ", ".join(r.get("failed", [])) or "—"
        lat  = _fmt_lat(r.get('latency_total', 0))
        lines.append(
            f"| {r['id']} | {r.get('category', '')} | {icon} {r['status']} "
            f"| {r['n_items']} | {lat} | {fail_str} |"
        )
    lines.append("")

    # ── Detail for failures and errors ────────────────────────────────────────
    problems = [r for r in results if r["status"] in ("FAIL", "ERROR")]
    if problems:
        lines += ["## Failures and Errors", ""]
        for r in problems:
            lines += [
                f"### {r['id']} — {r['query']}",
                f"**Status:** {r['status']}  |  "
                f"**Category:** {r.get('category', '?')}  |  "
                f"**Items:** {r['n_items']}  |  "
                f"**Latency:** {_fmt_lat(r.get('latency_total', 0))}",
                "",
            ]
            if r["status"] == "ERROR":
                lines += [f"**Error:** `{r.get('error', '?')}`", ""]
            else:
                lines += [
                    f"**Failed checks:** `{', '.join(r.get('failed', []))}`",
                    "",
                    f"**Response (first 400 chars):**",
                    f"> {r.get('response_text', '')[:400]}",
                    "",
                ]
                # Check detail (exclude internal _ keys)
                check_detail = {
                    k: v for k, v in r.get("checks", {}).items()
                    if not k.startswith("_")
                }
                lines += [
                    "**Check results:**",
                    f"```",
                    json.dumps(check_detail, indent=2, default=str),
                    "```",
                    "",
                ]
            lines += [
                f"**Tools called:** `{', '.join(r.get('tool_calls', []))}`",
                f"**Filters applied:** `{r.get('filters', {})}`",
                "",
            ]

    # ── Per-query raw response (collapsed) ───────────────────────────────────
    lines += ["## Per-Query Raw Output", ""]
    for r in results:
        icon = "PASS" if r["status"] == "PASS" else ("FAIL" if r["status"] == "FAIL" else "ERR")
        lines += [
            f"<details>",
            f"<summary>{icon} <strong>{r['id']}</strong> — {r['query']}</summary>",
            "",
            f"- **Status:** {r['status']}",
            f"- **Items:** {r['n_items']}",
            f"- **Tools:** {', '.join(r.get('tool_calls', []))}",
            f"- **Filters:** {r.get('filters', {})}",
            f"- **OOC:** {r.get('out_of_catalogue', False)}",
            f"- **Latency:** {_fmt_lat(r.get('latency_total', 0))} "
            f"(setup: {r.get('latency_setup', [])}  main: {_fmt_lat(r.get('latency_main', 0))})",
            "",
            "**Response:**",
            "",
            f"{r.get('response_text', '') or '_(no response)_'}",
            "",
            "**Check results:**",
            "",
            "```json",
            json.dumps(
                {k: v for k, v in r.get("checks", {}).items() if not k.startswith("_")},
                indent=2, default=str
            ),
            "```",
            "",
            "</details>",
            "",
        ]

    return "\n".join(lines)


def main():
    _ROOT = Path(__file__).parent.parent
    reports_dir = _ROOT / "reports"

    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        files = sorted(reports_dir.glob("eval_results_*.json"))
        if not files:
            print("No eval_results_*.json files found in reports/")
            sys.exit(1)
        json_path = files[-1]
        print(f"Using most recent: {json_path.name}")

    payload  = json.loads(json_path.read_text(encoding="utf-8"))
    md       = generate_markdown(payload)
    md_path  = json_path.with_suffix(".md")
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n-> Saved to {md_path}")


if __name__ == "__main__":
    main()
