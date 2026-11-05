"""Static checks: AR cleanup policy pins live/prev, and the pin step cannot trigger a rollback."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
# Env overrides let a mutation check point these tests at a mutated copy.
_POLICY = Path(os.environ.get("AR_POLICY_PATH") or _ROOT / "scripts/ops/ar_cleanup_policy.json")
_WORKFLOW = Path(os.environ.get("DEPLOY_YML_PATH") or _ROOT / ".github/workflows/deploy.yml")

ASSERT_NAME = "Pin min-instances 0 at service level and assert none serving"
PIN_NAME = "Pin live and prev image tags (after a fully successful deploy)"
ROLLBACK_NAME = "Roll back to previous revision"


@pytest.fixture(scope="module")
def policy() -> list[dict]:
    return json.loads(_POLICY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def steps() -> list[dict]:
    wf = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    (job,) = wf["jobs"].values()
    return job["steps"]


def _idx(steps: list[dict], name: str) -> int:
    return [s.get("name") for s in steps].index(name)


def test_policy_has_exactly_one_delete_rule_with_olderthan(policy) -> None:
    deletes = [r for r in policy if r["action"]["type"] == "Delete"]
    assert len(deletes) == 1
    assert deletes[0]["condition"]["olderThan"]


def test_policy_keeps_newest_two(policy) -> None:
    keeps = [r for r in policy if r["action"]["type"] == "Keep" and "mostRecentVersions" in r]
    assert [r["mostRecentVersions"]["keepCount"] for r in keeps] == [2]


def test_policy_pins_live_and_prev_tags(policy) -> None:
    pins = [
        r
        for r in policy
        if r["action"]["type"] == "Keep"
        and {"live", "prev"} <= set(r.get("condition", {}).get("tagPrefixes", []))
    ]
    assert len(pins) == 1
    assert pins[0]["condition"]["tagState"] == "tagged"


def test_pin_step_is_after_assertion_and_rollback(steps) -> None:
    pin = _idx(steps, PIN_NAME)
    assert pin > _idx(steps, ASSERT_NAME)
    assert pin > _idx(steps, ROLLBACK_NAME)


def test_pin_step_runs_only_on_full_success(steps) -> None:
    cond = steps[_idx(steps, PIN_NAME)]["if"]
    assert "success()" in cond
    assert "steps.minscale.outcome == 'success'" in cond
    assert steps[_idx(steps, ASSERT_NAME)]["id"] == "minscale"


def test_rollback_trigger_does_not_depend_on_pin_step(steps) -> None:
    cond = steps[_idx(steps, ROLLBACK_NAME)]["if"]
    pin_id = steps[_idx(steps, PIN_NAME)]["id"]
    assert pin_id not in cond
    assert "job.status != 'success'" in cond
