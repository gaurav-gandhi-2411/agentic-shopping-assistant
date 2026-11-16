"""Static + executed checks for the index-rollout wiring in deploy.yml and refresh-index.yml.

Workflows cannot be run locally, so this does what can be done: parse them, assert the structural
properties the design depends on, and EXECUTE the pure-shell "Resolve commit to deploy" step with
bash against hostile ids. Hermetic: no network, no gcloud.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / ".github" / "workflows" / "deploy.yml"
REFRESH = ROOT / ".github" / "workflows" / "refresh-index.yml"
SHA = "a" * 40


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))  # PyYAML reads the bare key `on` as True
    return data


@pytest.fixture(scope="module")
def deploy() -> dict[str, Any]:
    return _load(DEPLOY)


@pytest.fixture(scope="module")
def steps(deploy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    (job,) = deploy["jobs"].values()
    return {s.get("name") or s.get("id"): s for s in job["steps"]}


def _order(deploy: dict[str, Any]) -> list[str]:
    (job,) = deploy["jobs"].values()
    return [s.get("name", "") for s in job["steps"]]


# --- deploy.yml ----------------------------------------------------------------------------


def test_deploy_accepts_index_build_id_from_dispatch_and_call(deploy: dict[str, Any]) -> None:
    for trigger in ("workflow_dispatch", "workflow_call"):
        inp = deploy["on"][trigger]["inputs"]["index_build_id"]
        assert inp["type"] == "string" and inp["default"] == "" and inp["required"] is False
    assert "workflow_run" in deploy["on"]  # ordinary deploys still follow CI


def test_update_env_vars_appears_only_in_rollout_branches(deploy: dict[str, Any]) -> None:
    (job,) = deploy["jobs"].values()
    users = [s.get("name") for s in job["steps"] if "--update-env-vars" in s.get("run", "")]
    assert users == [
        "Deploy no-traffic tagged revision",
        "Restore the service template pin after a failed rollout (best effort)",
    ]
    deploy_run = next(s for s in job["steps"] if s.get("name") == users[0])["run"]
    guard = deploy_run.index('if [ "$MODE" = "rollout" ]')
    assert guard < deploy_run.index("--update-env-vars") < deploy_run.index("gcloud run deploy")
    text = DEPLOY.read_text(encoding="utf-8")
    for flag in ("--set-env-vars", "--clear-env-vars", "--set-secrets", "--update-secrets"):
        assert flag not in text


def test_rollout_ships_no_new_code_but_a_code_deploy_still_requires_ci(
    steps: dict[str, dict[str, Any]],
) -> None:
    assert "mode == 'code'" in steps["Require successful CI for the target SHA (fail closed)"]["if"]
    assert "mode == 'code'" in steps["Build and push image (tag = full commit SHA)"]["if"]
    assert "mode == 'rollout'" in steps["Reuse the serving image (index rollout)"]["if"]
    # the freshness step must always run (every later step keys off its output)
    assert "if" not in steps["Skip if a newer commit is on master"]
    assert 'echo "stale=false"' in steps["Skip if a newer commit is on master"]["run"]


def test_parity_is_the_tested_script_with_the_expected_pin_in_rollout_mode(
    steps: dict[str, dict[str, Any]],
) -> None:
    run = steps["Assert new revision kept prod config"]["run"]
    assert "scripts/ops/check_parity.py rev-prev.json rev-new.json" in run
    assert '--expect-index-build-id "$INDEX_ID"' in run
    assert "diff -u" not in run and "view()" not in run  # the inline jq view is gone


def test_both_smokes_assert_the_pinned_build_in_rollout_mode(
    steps: dict[str, dict[str, Any]],
) -> None:
    tag = steps["Smoke test tag URL (liveness, /readyz, 1 /chat = 2 measured LLM calls)"]["run"]
    main = steps["Smoke test main URL (liveness + /readyz, no chat)"]["run"]
    assert '--expect-build-id "$INDEX_ID"' in tag and '--expect-build-id "$INDEX_ID"' in main
    assert "--no-chat" in main and "--no-chat" not in tag  # chat only on the tag URL, as before


def test_rollback_and_restore_and_cleanup_ordering(deploy: dict[str, Any]) -> None:
    names = _order(deploy)
    i = {n: names.index(n) for n in names if n}
    rollback = i["Roll back to previous revision"]
    pin = i["Pin live and prev image tags (after a fully successful deploy)"]
    restore = i["Restore the service template pin after a failed rollout (best effort)"]
    cleanup = i["Keep the last 5 revisions (best effort)"]
    assert i["Shift 100% traffic to new revision"] < rollback < pin < restore < cleanup


def test_restore_runs_only_for_a_failed_rollout_and_is_best_effort(
    steps: dict[str, dict[str, Any]],
) -> None:
    s = steps["Restore the service template pin after a failed rollout (best effort)"]
    assert s["continue-on-error"] is True
    assert "job.status != 'success'" in s["if"] and "mode == 'rollout'" in s["if"]
    assert "--no-traffic" in s["run"]  # must never route traffic


def test_cleanup_is_best_effort_protects_new_and_prev_and_only_after_full_success(
    steps: dict[str, dict[str, Any]],
) -> None:
    s = steps["Keep the last 5 revisions (best effort)"]
    assert s["continue-on-error"] is True
    assert "success()" in s["if"] and "pin_images.outcome == 'success'" in s["if"]
    assert '--protect "$NEW"' in s["run"] and '--protect "$PREV"' in s["run"]
    assert "--keep 5" in s["run"] and "revisions delete" in s["run"] and "--quiet" in s["run"]
    assert "run.revisions" not in s["run"]


def test_job_if_lets_a_scheduled_caller_roll_out_only_with_an_id_on_master(
    deploy: dict[str, Any],
) -> None:
    (job,) = deploy["jobs"].values()
    cond = " ".join(job["if"].split())
    assert "inputs.index_build_id != '' && github.event_name == 'schedule'" in cond
    assert "github.ref == 'refs/heads/master'" in cond


def _resolve_script(steps: dict[str, dict[str, Any]]) -> str:
    return steps["Resolve commit to deploy"]["run"]


def _run_resolve(steps: dict[str, dict[str, Any]], tmp_path: Path, index_id: str):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    out = tmp_path / "out.txt"
    out.write_text("", encoding="utf-8")
    env = {
        **os.environ,
        "EVENT_NAME": "workflow_dispatch",
        "RUN_SHA": "",
        "DISPATCH_SHA": SHA,
        "INDEX_ID": index_id,
        "GITHUB_OUTPUT": str(out),
    }
    done = subprocess.run(  # noqa: S603
        [bash, "-c", _resolve_script(steps)], env=env, capture_output=True, text=True, check=False
    )
    return done.returncode, dict(
        line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines() if "=" in line
    )


def test_resolve_step_executed_code_mode(steps: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    rc, out = _run_resolve(steps, tmp_path, "")
    assert rc == 0 and out["mode"] == "code" and out["tag"] == "c-aaaaaaa" and out["index_id"] == ""


def test_resolve_step_executed_rollout_mode(steps: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    rc, out = _run_resolve(steps, tmp_path, "20261116T020000Z")
    assert rc == 0 and out["mode"] == "rollout"
    assert out["tag"] == "ix-20261116t020000z"  # lower-case: Cloud Run tags must be
    assert out["index_id"] == "20261116T020000Z"


@pytest.mark.parametrize(
    "bad",
    [
        "../x",
        "20261116T020000z",
        "20261116",
        "20261116T020000Z; rm -rf /",
        "$(id)",
        "a b",
        "2026\n",
        "20261116T020000Z\nmode=code",  # valid first line + injected GITHUB_OUTPUT line
    ],
)
def test_resolve_step_executed_rejects_hostile_ids(
    steps: dict[str, dict[str, Any]], tmp_path: Path, bad: str
) -> None:
    rc, out = _run_resolve(steps, tmp_path, bad)
    assert rc != 0 and "mode" not in out


# --- refresh-index.yml ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def refresh() -> dict[str, Any]:
    return _load(REFRESH)


def test_rollout_job_is_deploy_yml_called_with_the_build_id_and_least_privilege(
    refresh: dict[str, Any],
) -> None:
    job = refresh["jobs"]["rollout"]
    assert job["uses"] == "./.github/workflows/deploy.yml"
    assert job["with"] == {"index_build_id": "${{ needs.refresh.outputs.build_id }}"}
    assert job["permissions"] == {"contents": "read", "actions": "read", "id-token": "write"}
    assert set(job["needs"]) == {"refresh", "preflight"}
    assert "preflight.result == 'success'" in job["if"]


def test_retention_only_after_a_successful_rollout_and_never_deletes(
    refresh: dict[str, Any],
) -> None:
    job = refresh["jobs"]["retain"]
    assert job["if"] == "needs.rollout.result == 'success'"
    runs = " ".join(s.get("run", "") for s in job["steps"])
    assert "expire_builds.py" in runs and "--apply" in runs and "--keep 3" in runs
    assert "delete" not in runs.replace("deletes", "")  # no delete verb anywhere in the job
    assert "rm " not in runs


def test_preflight_gates_the_rollout_on_the_previous_builds_existence(
    refresh: dict[str, Any],
) -> None:
    job = refresh["jobs"]["preflight"]
    runs = " ".join(s.get("run", "") for s in job["steps"])
    assert "rollout_checks.py preflight" in runs and "rev-prev.json" in runs
    # deployer first (Cloud Run read), publisher second (bucket read)
    auths = [s["with"]["service_account"] for s in job["steps"] if "auth@" in s.get("uses", "")]
    assert auths == ["${{ env.DEPLOYER_SA }}", "${{ env.PUBLISHER_SA }}"]


def test_alert_covers_every_stage_and_only_alert_holds_issues_write(refresh: dict[str, Any]) -> None:
    assert set(refresh["jobs"]["alert"]["needs"]) == {"refresh", "preflight", "rollout", "retain"}
    holders = [n for n, j in refresh["jobs"].items() if "issues" in (j.get("permissions") or {})]
    assert holders == ["alert"]
    assert refresh["permissions"] == {}
