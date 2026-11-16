"""scripts/ops/refresh_alert.py lifecycle + static guards on .github/workflows/refresh-index.yml.

Hermetic: a fake ``gh`` runner; the workflow is only parsed (actionlint is run separately, see
DEPLOY.md "Verifying the workflows without running them").
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import refresh_alert as ra  # noqa: E402

RUN = "https://github.com/o/r/actions/runs/1"
WF = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "refresh-index.yml"
FAIL = {
    "stage": "fetch",
    "exit_code": 3,
    "build_id": None,
    "failure": {"stage": "fetch", "code": "too_few_fresh_stores", "message": "5/38 token=abc"},
    "stores": [{"slug": "libas", "status": "challenge"}, {"slug": "fashor", "status": "ok"}],
}


class FakeGh:
    def __init__(self, open_issues: str = "", labels: str = "prod-alert") -> None:
        self.open_issues, self.labels = open_issues, labels
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(self, args: list[str], stdin: str | None = None) -> str:
        self.calls.append((args, stdin))
        if args[:2] == ["issue", "list"]:
            return self.open_issues
        if args[:2] == ["label", "list"]:
            return self.labels
        return ""

    def verbs(self) -> list[str]:
        return [" ".join(a[:2]) for a, _ in self.calls]


def test_failure_with_no_open_issue_creates_one_with_the_prod_alert_label() -> None:
    gh = FakeGh()
    action = ra.apply("o/r", "failure", FAIL, RUN, runner=gh)
    assert action.kind == "create" and "issue create" in gh.verbs()
    create = next(c for c in gh.calls if c[0][:2] == ["issue", "create"])
    assert "prod-alert" in create[0] and ra.ALERT_TITLE in create[0]
    body = create[1] or ""
    assert "too_few_fresh_stores" in body and "libas=challenge" in body and "fashor" not in body


def test_missing_label_is_created_before_the_issue() -> None:
    gh = FakeGh(labels="bug")
    ra.apply("o/r", "failure", FAIL, RUN, runner=gh)
    assert gh.verbs().index("label create") < gh.verbs().index("issue create")


def test_failure_with_an_open_issue_comments_instead_of_duplicating() -> None:
    gh = FakeGh(open_issues=f"7\t{ra.ALERT_TITLE}\n")
    assert ra.apply("o/r", "failure", FAIL, RUN, runner=gh).kind == "comment"
    assert "issue create" not in gh.verbs()


def test_success_closes_the_open_issue_and_otherwise_does_nothing() -> None:
    gh = FakeGh(open_issues=f"7\t{ra.ALERT_TITLE}\n")
    assert ra.apply("o/r", "success", {}, RUN, runner=gh).kind == "close"
    assert "issue close" in gh.verbs()
    gh2 = FakeGh()
    assert ra.apply("o/r", "success", {}, RUN, runner=gh2).kind == "none"
    assert gh2.verbs() == ["issue list"]


def test_other_prod_alert_issues_are_never_touched() -> None:
    gh = FakeGh(open_issues="3\tStyleMaitri prod canary is failing\n9\tsomething else\n")
    assert ra.apply("o/r", "success", {}, RUN, runner=gh).kind == "none"
    assert ra.apply("o/r", "failure", FAIL, RUN, runner=gh).kind == "create"


def test_missing_or_corrupt_summary_still_alerts(tmp_path: Path) -> None:
    assert ra.load_summary(None) == {} and ra.load_summary(tmp_path / "nope.json") == {}
    bad = tmp_path / "s.json"
    bad.write_text("{oops", encoding="utf-8")
    assert ra.load_summary(bad) == {}
    gh = FakeGh()
    ra.apply("o/r", "failure", {}, RUN, runner=gh)
    create = next(c for c in gh.calls if c[0][:2] == ["issue", "create"])
    assert "no summary produced" in (create[1] or "")


def test_failed_jobs_are_listed_but_only_as_plain_job_names() -> None:
    assert "`preflight,rollout`" in ra.build_body({}, RUN, "x", "preflight,rollout")
    assert "Failed job" not in ra.build_body({}, RUN, "x", "rollout`; curl evil")  # not a job list
    assert "Failed job" not in ra.build_body({}, RUN, "x", "")


def test_the_secret_like_text_in_the_detail_is_redacted() -> None:
    body = ra.build_body(
        {"failure": {"stage": "build", "code": "build_failed", "message": "token=" + "x" * 20}},
        RUN,
        "x",
    )
    assert "x" * 20 not in body and "[redacted]" in body


# --- static guards on the workflow ---------------------------------------------------------


@pytest.fixture(scope="module")
def wf() -> dict[str, Any]:
    data = yaml.safe_load(WF.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))  # PyYAML reads the bare key `on` as True
    return data


def test_workflow_has_no_default_permissions_and_least_privilege_per_job(wf: dict[str, Any]) -> None:
    assert wf["permissions"] == {}
    refresh, alert = wf["jobs"]["refresh"], wf["jobs"]["alert"]
    assert refresh["permissions"] == {"contents": "read", "id-token": "write"}
    assert alert["permissions"] == {"contents": "read", "issues": "write"}
    assert "id-token" not in alert["permissions"]


def test_workflow_never_cancels_a_running_refresh(wf: dict[str, Any]) -> None:
    assert wf["concurrency"] == {"group": "refresh-index", "cancel-in-progress": False}


def test_dry_run_is_the_manual_default_and_schedule_needs_the_pause_switch(
    wf: dict[str, Any],
) -> None:
    dispatch = wf["on"]["workflow_dispatch"]["inputs"]["dry_run"]
    assert dispatch["default"] is True and dispatch["type"] == "boolean"
    cond = " ".join(wf["jobs"]["refresh"]["if"].split())
    assert "github.ref == 'refs/heads/master'" in cond
    assert "vars.REFRESH_ENABLED == 'true'" in cond
    assert "github.repository ==" in cond
    assert wf["on"]["schedule"][0]["cron"].split()[1] in {"20", "21", "22", "23", "0", "1"}


def test_no_workflow_secret_is_read_and_publish_is_flagged_only_when_not_dry(
    wf: dict[str, Any],
) -> None:
    text = WF.read_text(encoding="utf-8")
    assert "secrets." not in text.replace("secrets.GITHUB_TOKEN", "")
    run = next(s for s in wf["jobs"]["refresh"]["steps"] if s.get("id") == "run")["run"]
    assert '"$DRY_RUN" != "true"' in run and "--publish" in run
    assert run.index("--publish") < run.index("refresh_index.py")  # flags are built beforehand
