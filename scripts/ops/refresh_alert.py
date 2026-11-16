"""Open, update or close the prod-alert issue for the daily index refresh (needs ``issues: write``).

    python scripts/ops/refresh_alert.py --repo OWNER/REPO --run-url URL [--summary refresh-summary.json]
        [--outcome failure|success]

Reuses the canary's prod-alert machinery (``scripts/ops/canary.py``): the same ``prod-alert`` label,
the same pure lifecycle (``decide_issue_action``: failure creates or comments, success closes) and
the same ``gh`` wrapper. Only open issues with this job's OWN fixed title are considered, so the
canary's and the uptime monitor's issues are never touched.

A failed refresh changes nothing in prod (publish is the last stage); the issue is the signal that
the index is getting older. ``/canary`` starts failing ``index_stale`` at 36 h on its own.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canary  # noqa: E402

ALERT_TITLE = "StyleMaitri index refresh is failing"
ALERT_LABEL = canary.ALERT_LABEL


def load_summary(path: Path | None) -> dict[str, Any]:
    """The refresh summary, or ``{}`` when it is missing or unreadable (never raises)."""
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def clean_jobs(text: str) -> str:
    """Failed job names for the body: lower-case letters and commas only, else empty."""
    return text if re.fullmatch(r"[a-z,]{1,60}", text or "") else ""


def build_body(
    summary: dict[str, Any], run_url: str, heading: str, failed_jobs: str = ""
) -> str:
    """Markdown body from fixed machine codes plus one redacted, truncated message."""
    failure = summary.get("failure") or {}
    stores = summary.get("stores") or []
    stopped = sorted(
        f"{s.get('slug')}={s.get('status')}" for s in stores if s.get("status") != "ok"
    )
    lines = [
        heading,
        "",
        f"**Stage:** `{failure.get('stage', summary.get('stage', 'unknown'))}`",
        f"**Code:** `{failure.get('code', 'no summary produced')}`",
        f"**Exit code:** `{summary.get('exit_code', 'n/a')}`",
        f"**Build id:** `{summary.get('build_id')}` (previous `{summary.get('previous_build_id')}`)",
    ]
    if clean_jobs(failed_jobs):
        lines.append(f"**Failed job(s):** `{clean_jobs(failed_jobs)}`")
    if failure.get("message"):
        lines.append(f"**Detail:** {canary.redact(str(failure['message']))[:300]}")
    if stopped:
        lines.append(f"**Stores not refreshed ({len(stopped)}):** {', '.join(stopped[:40])}")
    lines += [
        "",
        "Production is untouched: the serving revision still runs its pinned build.",
        f"[Workflow run]({run_url})",
        "",
        "*Managed by the refresh-index workflow; the title is used to deduplicate alerts.*",
    ]
    return "\n".join(lines)


def apply(
    repo: str,
    outcome: str,
    summary: dict[str, Any],
    run_url: str,
    runner: canary.GhRunner = canary.gh,
    failed_jobs: str = "",
) -> canary.IssueAction:
    """Run the lifecycle against GitHub. ``outcome`` is ``failure`` or ``success``."""
    listed = runner(
        [
            "issue", "list", "--repo", repo, "--label", ALERT_LABEL, "--state", "open",
            "--limit", "100", "--json", "number,title", "--jq",
            r".[] | [(.number|tostring), .title] | @tsv",
        ],
        None,
    )  # fmt: skip
    open_issues = []
    for line in listed.splitlines():
        num, _, title = line.partition("\t")
        if title.strip() == ALERT_TITLE and num.isdigit():
            open_issues.append(int(num))
    action = canary.decide_issue_action("fail" if outcome == "failure" else "pass", open_issues)
    if action.kind == "create":
        labels = runner(
            ["label", "list", "--repo", repo, "--limit", "200", "--json", "name", "--jq", ".[].name"],
            None,
        ).split("\n")
        if ALERT_LABEL not in labels:
            runner(
                [
                    "label", "create", ALERT_LABEL, "--repo", repo, "--color", "B60205",
                    "--description", "Production alert opened by monitoring workflows",
                ],
                None,
            )  # fmt: skip
        runner(
            [
                "issue", "create", "--repo", repo, "--title", ALERT_TITLE,
                "--label", ALERT_LABEL, "--body-file", "-",
            ],
            build_body(summary, run_url, "## Index refresh failed\n\nDetected", failed_jobs),
        )  # fmt: skip
    elif action.kind == "comment":
        runner(
            ["issue", "comment", str(action.numbers[0]), "--repo", repo, "--body-file", "-"],
            build_body(summary, run_url, "Still failing", failed_jobs),
        )
    elif action.kind == "close":
        for n in action.numbers:
            runner(
                ["issue", "comment", str(n), "--repo", repo, "--body-file", "-"],
                f"**Recovered**: the index refresh succeeded. [Workflow run]({run_url})",
            )
            runner(["issue", "close", str(n), "--repo", repo, "--reason", "completed"], None)
    return action


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit 1 when the alert itself could not be delivered."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-url", required=True)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--outcome", choices=("failure", "success"), required=True)
    ap.add_argument("--failed-jobs", default="", help="comma-separated job names, e.g. rollout")
    args = ap.parse_args(argv)
    try:
        action = apply(
            args.repo,
            args.outcome,
            load_summary(args.summary),
            args.run_url,
            failed_jobs=args.failed_jobs,
        )
    except canary.CanaryError as exc:
        print(f"::error::could not update the alert issue: {exc}", file=sys.stderr)
        return 1
    print(f"alert issue action: {action.kind} {list(action.numbers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
