"""Offline tests for the guards in scripts/ops/setup_deploy_iam.sh (stub gcloud, no network, no GCP).

The script is sourced with SETUP_DEPLOY_IAM_SOURCE_ONLY=1 so only its functions load; `gcloud` is a
stub earlier on PATH that prints canned output. Nothing here can reach a real project.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "setup_deploy_iam.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")

GOOD = (
    "principalSet://iam.googleapis.com/projects/631709154646/locations/global/"
    "workloadIdentityPools/github-deploy-pool/attribute.repository/"
    "gaurav-gandhi-2411/agentic-shopping-assistant"
)
LEGACY = GOOD.replace("github-deploy-pool", "github-pool")
WRONG_PROJECT = GOOD.replace("631709154646", "999999999999")
WIU = "roles/iam.workloadIdentityUser"

STUB = """#!/usr/bin/env bash
case "$*" in
  *get-iam-policy*)
    [ "${STUB_POLICY_RC:-0}" = "0" ] || exit "$STUB_POLICY_RC"
    printf '%s' "$STUB_POLICY_ROWS" ;;
  *"--format=value(state)"*)
    printf '%s\\n' "${STUB_STATE:-}" ;;
  *) echo "stub gcloud: unexpected call: $*" >&2; exit 99 ;;
esac
"""


def _rows(*pairs: tuple[str, str]) -> str:
    return "".join(f"{role}\t{member}\n" for role, member in pairs)


@pytest.fixture()
def stub_env(tmp_path: Path) -> dict[str, str]:
    d = tmp_path / "bin"
    d.mkdir()
    g = d / "gcloud"
    g.write_text(STUB, encoding="utf-8", newline="\n")
    g.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = str(d) + os.pathsep + env["PATH"]
    env["SCRIPT"] = SCRIPT.as_posix()
    return env


def _call(env: dict[str, str], body: str) -> subprocess.CompletedProcess[str]:
    cmd = f'set -euo pipefail; SETUP_DEPLOY_IAM_SOURCE_ONLY=1 source "$SCRIPT"; {body}'
    return subprocess.run(
        [BASH, "-c", cmd], env=env, capture_output=True, text=True, timeout=60, check=False
    )


def _verify(env: dict[str, str], rows: str, rc: str = "0") -> subprocess.CompletedProcess[str]:
    env = {**env, "STUB_POLICY_ROWS": rows, "STUB_POLICY_RC": rc}
    return _call(env, "verify_deployer_policy")


def test_policy_correct_passes(stub_env) -> None:
    r = _verify(stub_env, _rows((WIU, GOOD)))
    assert r.returncode == 0, r.stderr
    assert "ERROR" not in r.stderr


@pytest.mark.parametrize(
    ("name", "rows"),
    [
        ("legacy_pool_only", _rows((WIU, LEGACY))),
        ("mixed_good_and_legacy", _rows((WIU, GOOD), (WIU, LEGACY))),
        ("wrong_project_number", _rows((WIU, WRONG_PROJECT))),
        ("allUsers_added", _rows((WIU, GOOD), (WIU, "allUsers"))),
        ("user_added", _rows((WIU, GOOD), (WIU, "user:someone@example.com"))),
        ("serviceAccount_added", _rows((WIU, GOOD), (WIU, "serviceAccount:x@y.iam.gserviceaccount.com"))),
        ("other_principal_added", _rows((WIU, GOOD), (WIU, GOOD.replace("principalSet", "principal")))),
        ("pool_member_under_other_role", _rows((WIU, GOOD), ("roles/owner", GOOD))),
        ("empty_policy", ""),
        ("blank_output", "  \n\t\n"),
        ("duplicate_good_is_not_exactly_one", _rows((WIU, GOOD), (WIU, GOOD))),
    ],
)
def test_policy_bad_fails_with_error(stub_env, name: str, rows: str) -> None:
    r = _verify(stub_env, rows)
    assert r.returncode != 0, f"{name} must fail"
    assert "ERROR" in r.stderr, f"{name} must print ERROR, got: {r.stderr!r}"


def test_policy_read_failure_fails_with_error(stub_env) -> None:
    r = _verify(stub_env, _rows((WIU, GOOD)), rc="1")
    assert r.returncode != 0 and "ERROR" in r.stderr


@pytest.mark.parametrize("state", ["", "ACTIVE"])
def test_not_deleted_passes(stub_env, state: str) -> None:
    env = {**stub_env, "STUB_STATE": state}
    r = _call(env, 'guard_not_deleted "pool p" "gcloud undelete p" gcloud describe p')
    assert r.returncode == 0, r.stderr


def test_deleted_pool_prints_undelete_and_fails(stub_env) -> None:
    env = {**stub_env, "STUB_STATE": "DELETED"}
    r = _call(
        env,
        'guard_not_deleted "pool github-deploy-pool" "gcloud iam workload-identity-pools undelete '
        'github-deploy-pool --location=global" gcloud describe p',
    )
    assert r.returncode != 0
    assert "ERROR" in r.stderr and "DELETED" in r.stderr
    assert "gcloud iam workload-identity-pools undelete github-deploy-pool --location=global" in r.stderr
