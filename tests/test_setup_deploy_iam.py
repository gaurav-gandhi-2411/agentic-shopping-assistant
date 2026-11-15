"""Offline tests for the guards in scripts/ops/setup_deploy_iam.sh (stub gcloud, no network, no GCP).

The script is sourced with SETUP_DEPLOY_IAM_SOURCE_ONLY=1 so only its functions load; `gcloud` is a
stub earlier on PATH that prints canned output. Nothing here can reach a real project.
"""
from __future__ import annotations

import os
import re
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


def test_policy_correct_passes_with_crlf_rows(stub_env) -> None:
    # Real gcloud on Windows emits CRLF; the good row first, then another role, must still pass.
    rows = f"{WIU}\t{GOOD}\r\nroles/zzz.other\tuser:someone@example.com\r\n"
    r = _verify(stub_env, rows)
    assert r.returncode == 0, r.stderr


def test_policy_bad_still_fails_with_crlf_rows(stub_env) -> None:
    rows = f"{WIU}\t{GOOD}\r\n{WIU}\tallUsers\r\n"
    r = _verify(stub_env, rows)
    assert r.returncode != 0 and "ERROR" in r.stderr


def test_deleted_state_with_crlf_still_blocks(stub_env) -> None:
    env = {**stub_env, "STUB_STATE": "DELETED\r"}
    r = _call(env, 'guard_not_deleted "pool p" "gcloud undelete p" gcloud describe p')
    assert r.returncode != 0 and "DELETED" in r.stderr


def test_policy_read_failure_fails_with_error(stub_env) -> None:
    r = _verify(stub_env, _rows((WIU, GOOD)), rc="1")
    assert r.returncode != 0 and "ERROR" in r.stderr


@pytest.mark.parametrize("state", ["", "ACTIVE"])
def test_not_deleted_passes(stub_env, state: str) -> None:
    env = {**stub_env, "STUB_STATE": state}
    r = _call(env, 'guard_not_deleted "pool p" "gcloud undelete p" gcloud describe p')
    assert r.returncode == 0, r.stderr


def _display_names() -> dict[str, str]:
    """Map the gcloud sub-command that carries each --display-name to its value."""
    found: dict[str, str] = {}
    command = ""
    for line in SCRIPT.read_text(encoding="utf-8").splitlines():
        if "gcloud iam" in line and "create" in line:
            command = line.strip()
        m = re.search(r'--display-name="([^"]*)"', line)
        if m:
            found[command] = m.group(1)
    return found


def test_display_names_respect_gcp_length_limits() -> None:
    """GCP rejects WIF pool/provider display names > 32 chars (the 33-char provider name failed
    the first real apply); service-account display names may be up to 100."""
    names = _display_names()
    sa_names = [v for cmd, v in names.items() if "service-accounts create" in cmd]
    by_kind = {
        kind: value
        for cmd, value in names.items()
        for kind in ("pools create", "providers create-oidc")
        if kind in cmd
    }
    assert set(by_kind) == {"pools create", "providers create-oidc"}, names
    # deployer SA + the new runtime SA
    assert len(sa_names) == 2, names
    assert len(by_kind["pools create"]) <= 32, by_kind["pools create"]
    assert len(by_kind["providers create-oidc"]) <= 32, by_kind["providers create-oidc"]
    assert all(len(n) <= 100 for n in sa_names), sa_names
    assert by_kind["providers create-oidc"] == "GitHub OIDC (deploy, master)"


def test_custom_role_title_and_description_respect_gcp_limits() -> None:
    """GCP custom roles: title <= 100 chars, description <= 300 chars."""
    text = SCRIPT.read_text(encoding="utf-8")
    title = re.search(r'iam roles create .*?--title="([^"]*)"', text, re.S)
    desc = re.search(r'--description="([^"]*)"', text)
    assert title and desc
    assert len(title.group(1)) <= 100
    assert len(desc.group(1)) <= 300


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


# ---------------------------------------------------------------------------------------------
# Whole-script runs against a recording stub gcloud (runtime SA, custom role, opt-in removal).
# The stub logs every call's argv and answers describe/get-iam-policy from env; mutating verbs
# just exit 0. Nothing here can reach a real project.
# ---------------------------------------------------------------------------------------------
PROJECT = "stylemaitri-prod-260813"
DEPLOYER = f"stylemaitri-deployer@{PROJECT}.iam.gserviceaccount.com"
RUNTIME = f"stylemaitri-runtime@{PROJECT}.iam.gserviceaccount.com"
COMPUTE = "631709154646-compute@developer.gserviceaccount.com"
TAG_ROLE = f"projects/{PROJECT}/roles/artifactRegistryTagMover"
SECRETS = [
    "asa-database-url",
    "asa-demo-jwt-secret",
    "asa-groq-api-key",
    "asa-supabase-url",
    "CANARY_TOKEN",
]
BUCKET = "gs://asa-demo-indices-stylemaitri-prod"
FIVE_PERMS_GCLOUD = ";".join(
    f"artifactregistry.tags.{p}" for p in ("update", "list", "get", "delete", "create")
)
FIVE_PERMS_SCRIPT = ",".join(
    f"artifactregistry.tags.{p}" for p in ("create", "delete", "get", "list", "update")
)
MUTATING = re.compile(
    r"(^| )(create|create-oidc|update|update-oidc|add-iam-policy-binding|"
    r"remove-iam-policy-binding|set-cleanup-policies)( |$)"
)

FULL_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"iam roles describe"*"includedPermissions"*)
    [ "${STUB_EXISTS:-none}" = "all" ] || exit 1
    printf '%s' "$STUB_ROLE_PERMS" ;;
  *"run services describe"*) printf '%s' "${STUB_SERVICE_SA:-}" ;;
  *"repositories get-iam-policy"*) printf '%s' "$STUB_REPO_ROWS" ;;
  *"projects get-iam-policy"*) printf '%s' "$STUB_PROJECT_ROWS" ;;
  *"get-iam-policy"*) printf '%s' "$STUB_POLICY_ROWS" ;;
  *"--format=value(state)"*) printf '%s\\n' "${STUB_STATE:-}" ;;
  *"repositories describe"*)
    [ "${STUB_REPO_RC:-0}" = "0" ] || exit "$STUB_REPO_RC"
    printf '%s' "$STUB_REPO_JSON" ;;
  *" describe "*) [ "${STUB_EXISTS:-none}" = "all" ] ;;
  *create*|*update*|*add-iam-policy-binding*|*remove-iam-policy-binding*|*set-cleanup-policies*)
    exit 0 ;;
  *) echo "stub gcloud: unexpected call: $*" >&2; exit 99 ;;
esac
"""

REPO_OK = _rows(
    ("roles/artifactregistry.writer", f"serviceAccount:{DEPLOYER}"),
    (TAG_ROLE, f"serviceAccount:{DEPLOYER}"),
)
PROJECT_OK = _rows(("roles/editor", f"serviceAccount:{COMPUTE}"))

# `gcloud artifacts repositories describe --format=json` shapes (pretty-printed like gcloud).
_POLICIES = (
    '"cleanupPolicies": {\n    "keep-recent": {"id": "keep-recent", "action": "KEEP",'
    ' "mostRecentVersions": {"keepCount": 2}}\n  }'
)
REPO_JSON_NO_POLICY = (
    '{\n  "format": "DOCKER",\n  "name": "projects/p/locations/l/repositories/r"\n}\n'
)
REPO_JSON_EMPTY_POLICIES = '{\n  "cleanupPolicies": {},\n  "format": "DOCKER"\n}\n'
REPO_JSON_DRYRUN = f'{{\n  {_POLICIES},\n  "cleanupPolicyDryRun": true,\n  "format": "DOCKER"\n}}\n'
REPO_JSON_LIVE = f'{{\n  {_POLICIES},\n  "format": "DOCKER"\n}}\n'
REPO_JSON_LIVE_EXPLICIT_FALSE = (
    f'{{\n  {_POLICIES},\n  "cleanupPolicyDryRun": false,\n  "format": "DOCKER"\n}}\n'
)


@pytest.fixture()
def full_env(tmp_path: Path) -> dict[str, str]:
    d = tmp_path / "fbin"
    d.mkdir()
    g = d / "gcloud"
    g.write_text(FULL_STUB, encoding="utf-8", newline="\n")
    g.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = str(d) + os.pathsep + env["PATH"]
    env["STUB_LOG"] = (tmp_path / "calls.log").as_posix()
    env["STUB_POLICY_ROWS"] = _rows((WIU, GOOD))
    env["STUB_REPO_ROWS"] = REPO_OK
    env["STUB_PROJECT_ROWS"] = PROJECT_OK
    env["STUB_ROLE_PERMS"] = FIVE_PERMS_GCLOUD
    env["SA_PROPAGATION_WAIT"] = "0"
    env["STUB_REPO_JSON"] = REPO_JSON_NO_POLICY
    env.pop("STUB_REPO_RC", None)
    env.pop("FORCE_AR_POLICY_DRYRUN", None)
    env.pop("DRY_RUN", None)
    env.pop("REMOVE_COMPUTE_ACTAS", None)
    return env


def _run_script(
    env: dict[str, str], **extra: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    env = {**env, **extra}
    r = subprocess.run(
        [BASH, SCRIPT.as_posix()], env=env, capture_output=True, text=True, timeout=120, check=False
    )
    log = Path(env["STUB_LOG"])
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    # normalise: drop the common --account/--project suffixes so assertions read cleanly
    calls = [re.sub(r" --(account|project)=\S+", "", c).strip() for c in calls]
    return r, calls


def _mutations(calls: list[str]) -> list[str]:
    return [c for c in calls if MUTATING.search(c)]


def test_role_created_when_missing(full_env) -> None:
    r, calls = _run_script(full_env, STUB_EXISTS="none")
    assert r.returncode == 0, r.stderr
    created = [c for c in calls if c.startswith("iam roles create artifactRegistryTagMover")]
    assert len(created) == 1, calls
    assert f"--permissions={FIVE_PERMS_SCRIPT}" in created[0]
    assert not [c for c in calls if c.startswith("iam roles update")]


def test_role_converged_when_permissions_drift(full_env) -> None:
    drift = "artifactregistry.tags.create;artifactregistry.repositories.delete"
    r, calls = _run_script(full_env, STUB_EXISTS="all", STUB_ROLE_PERMS=drift)
    assert r.returncode == 0, r.stderr
    updated = [c for c in calls if c.startswith("iam roles update artifactRegistryTagMover")]
    assert len(updated) == 1, calls
    assert f"--permissions={FIVE_PERMS_SCRIPT}" in updated[0]
    assert not [c for c in calls if c.startswith("iam roles create")]


def test_role_untouched_when_permissions_exact(full_env) -> None:
    r, calls = _run_script(full_env, STUB_EXISTS="all")
    assert r.returncode == 0, r.stderr
    assert not [c for c in calls if c.startswith(("iam roles create", "iam roles update"))]


def test_tag_mover_bound_to_deployer_on_the_one_repo(full_env) -> None:
    r, calls = _run_script(full_env)
    assert r.returncode == 0, r.stderr
    want = (
        "artifacts repositories add-iam-policy-binding cloud-run-source-deploy "
        f"--location=asia-south1 --member=serviceAccount:{DEPLOYER} --role={TAG_ROLE}"
    )
    assert want in calls


def test_secret_bindings_are_secret_level_with_exact_member(full_env) -> None:
    r, calls = _run_script(full_env)
    assert r.returncode == 0, r.stderr
    bound = [c for c in calls if c.startswith("secrets add-iam-policy-binding")]
    expected = [
        f"secrets add-iam-policy-binding {s} --member=serviceAccount:{RUNTIME} "
        "--role=roles/secretmanager.secretAccessor"
        for s in SECRETS
    ]
    assert sorted(bound) == sorted(expected)


def test_bucket_binding_is_bucket_level_read_only(full_env) -> None:
    r, calls = _run_script(full_env)
    assert r.returncode == 0, r.stderr
    want = (
        f"storage buckets add-iam-policy-binding {BUCKET} "
        f"--member=serviceAccount:{RUNTIME} --role=roles/storage.objectViewer"
    )
    assert calls.count(want) == 1
    # a stray object-write/admin role for the runtime SA would be a regression
    assert not [
        c for c in calls if RUNTIME in c and re.search(r"storage\.(object(Admin|Creator)|admin)", c)
    ]


def test_runtime_sa_gets_only_secret_and_bucket_level_bindings(full_env) -> None:
    r, calls = _run_script(full_env)
    assert r.returncode == 0, r.stderr
    assert not [c for c in calls if c.startswith("projects add-iam-policy-binding")], calls
    runtime_grants = [c for c in calls if f"--member=serviceAccount:{RUNTIME}" in c]
    assert len(runtime_grants) == len(SECRETS) + 1
    assert all(
        c.startswith(("secrets add-iam-policy-binding ", "storage buckets add-iam-policy-binding "))
        for c in runtime_grants
    ), runtime_grants
    assert not [c for c in runtime_grants if "roles/editor" in c or "logging" in c]


def test_deployer_gets_actas_on_runtime_sa_and_compute_sa_is_not_touched(full_env) -> None:
    r, calls = _run_script(full_env)
    assert r.returncode == 0, r.stderr
    want = (
        f"iam service-accounts add-iam-policy-binding {RUNTIME} "
        f"--member=serviceAccount:{DEPLOYER} --role=roles/iam.serviceAccountUser"
    )
    assert want in calls
    # the only call naming the compute SA is the pre-existing step-3 actAs add: no removal, no roles
    compute_calls = [c for c in calls if COMPUTE in c]
    assert compute_calls == [
        f"iam service-accounts add-iam-policy-binding {COMPUTE} "
        f"--member=serviceAccount:{DEPLOYER} --role=roles/iam.serviceAccountUser"
    ]
    assert not [c for c in calls if "remove-iam-policy-binding" in c]


def test_opt_in_removal_runs_only_with_flag_and_only_after_the_switch(full_env) -> None:
    removal = (
        f"iam service-accounts remove-iam-policy-binding {COMPUTE} "
        f"--member=serviceAccount:{DEPLOYER} --role=roles/iam.serviceAccountUser"
    )
    r, calls = _run_script(full_env, STUB_SERVICE_SA=RUNTIME)
    assert r.returncode == 0 and removal not in calls  # flag absent -> never
    r, calls = _run_script(full_env, REMOVE_COMPUTE_ACTAS="0", STUB_SERVICE_SA=RUNTIME)
    assert r.returncode == 0 and removal not in calls
    r, calls = _run_script(full_env, REMOVE_COMPUTE_ACTAS="1", STUB_SERVICE_SA=RUNTIME)
    assert r.returncode == 0, r.stderr
    assert calls.count(removal) == 1


@pytest.mark.parametrize("service_sa", [COMPUTE, ""])
def test_opt_in_removal_refuses_while_service_not_on_runtime_sa(full_env, service_sa: str) -> None:
    r, calls = _run_script(full_env, REMOVE_COMPUTE_ACTAS="1", STUB_SERVICE_SA=service_sa)
    assert r.returncode != 0
    assert "ERROR" in r.stderr and "REMOVE_COMPUTE_ACTAS" in r.stderr
    assert not [c for c in calls if "remove-iam-policy-binding" in c]


def test_dry_run_mutates_nothing(full_env) -> None:
    r, calls = _run_script(
        full_env, DRY_RUN="1", STUB_EXISTS="none", REMOVE_COMPUTE_ACTAS="1", STUB_SERVICE_SA=RUNTIME
    )
    assert r.returncode == 0, r.stderr
    assert _mutations(calls) == [], calls
    assert "DRY_RUN: gcloud secrets add-iam-policy-binding" in r.stdout
    assert "DRY_RUN: gcloud iam roles create artifactRegistryTagMover" in r.stdout


def test_second_run_when_everything_exists_issues_no_create(full_env) -> None:
    r, calls = _run_script(full_env, STUB_EXISTS="all")
    assert r.returncode == 0, r.stderr
    creates = [c for c in calls if re.search(r"(^| )create(-oidc)?( |$)", c)]
    assert creates == [], creates
    assert "sleep" not in " ".join(calls)


@pytest.mark.parametrize(
    ("name", "rows", "ok"),
    [
        ("exact", REPO_OK, True),
        (
            "other_member_may_hold_admin",
            REPO_OK + _rows(("roles/artifactregistry.admin", "user:x@y.z")),
            True,
        ),
        (
            "extra_role",
            REPO_OK + _rows(("roles/artifactregistry.admin", f"serviceAccount:{DEPLOYER}")),
            False,
        ),
        (
            "missing_tag_role",
            _rows(("roles/artifactregistry.writer", f"serviceAccount:{DEPLOYER}")),
            False,
        ),
        ("missing_writer", _rows((TAG_ROLE, f"serviceAccount:{DEPLOYER}")), False),
        (
            "duplicate_writer",
            REPO_OK + _rows(("roles/artifactregistry.writer", f"serviceAccount:{DEPLOYER}")),
            False,
        ),
        ("empty", "", False),
    ],
)
def test_verify_deployer_repo_roles(stub_env, name: str, rows: str, ok: bool) -> None:
    env = {**stub_env, "STUB_POLICY_ROWS": rows, "STUB_POLICY_RC": "0"}
    r = _call(env, "verify_deployer_repo_roles")
    assert (r.returncode == 0) is ok, f"{name}: {r.stderr!r}"
    if not ok:
        assert "ERROR" in r.stderr


def test_verify_deployer_repo_roles_read_failure_fails(stub_env) -> None:
    env = {**stub_env, "STUB_POLICY_ROWS": REPO_OK, "STUB_POLICY_RC": "1"}
    r = _call(env, "verify_deployer_repo_roles")
    assert r.returncode != 0 and "ERROR" in r.stderr


@pytest.mark.parametrize(
    ("name", "rows", "ok"),
    [
        ("none_for_runtime", PROJECT_OK, True),
        (
            "runtime_has_project_role",
            PROJECT_OK + _rows(("roles/viewer", f"serviceAccount:{RUNTIME}")),
            False,
        ),
        ("empty", "", False),
    ],
)
def test_verify_runtime_sa_has_no_project_roles(stub_env, name: str, rows: str, ok: bool) -> None:
    env = {**stub_env, "STUB_POLICY_ROWS": rows, "STUB_POLICY_RC": "0"}
    r = _call(env, "verify_runtime_sa_no_project_roles")
    assert (r.returncode == 0) is ok, f"{name}: {r.stderr!r}"


# ---------------------------------------------------------------------------------------------
# Step 7 guard: never switch a LIVE Artifact Registry cleanup policy back to --dry-run.
# ---------------------------------------------------------------------------------------------
def _set_policy_calls(calls: list[str]) -> list[str]:
    return [c for c in calls if "set-cleanup-policies" in c]


@pytest.mark.parametrize(
    ("name", "repo_json", "state"),
    [
        ("no_policies_key", REPO_JSON_NO_POLICY, "none"),
        ("empty_policies", REPO_JSON_EMPTY_POLICIES, "none"),
        ("dry_run_policy", REPO_JSON_DRYRUN, "dryrun"),
        ("live_policy_flag_absent", REPO_JSON_LIVE, "live"),
        ("live_policy_flag_false", REPO_JSON_LIVE_EXPLICIT_FALSE, "live"),
        ("live_policy_crlf", REPO_JSON_LIVE.replace("\n", "\r\n"), "live"),
    ],
)
def test_ar_cleanup_policy_state(stub_env, name: str, repo_json: str, state: str) -> None:
    stub = Path(stub_env["PATH"].split(os.pathsep)[0]) / "gcloud"
    stub.write_text("#!/usr/bin/env bash\nprintf '%s' \"$STUB_REPO_JSON\"\n", newline="\n")
    r = _call({**stub_env, "STUB_REPO_JSON": repo_json}, "ar_cleanup_policy_state")
    assert r.returncode == 0, f"{name}: {r.stderr!r}"
    assert r.stdout.strip() == state, name


def test_step7_applies_dry_run_when_no_policies(full_env) -> None:
    r, calls = _run_script(full_env, STUB_REPO_JSON=REPO_JSON_NO_POLICY)
    assert r.returncode == 0, r.stderr
    applied = _set_policy_calls(calls)
    assert len(applied) == 1 and "--dry-run" in applied[0], calls
    assert "SKIPPED step 7" not in r.stdout


def test_step7_reapplies_when_existing_policy_is_already_dry_run(full_env) -> None:
    r, calls = _run_script(full_env, STUB_REPO_JSON=REPO_JSON_DRYRUN)
    assert r.returncode == 0, r.stderr
    applied = _set_policy_calls(calls)
    assert len(applied) == 1 and "--dry-run" in applied[0], calls


@pytest.mark.parametrize(
    "live_json",
    [REPO_JSON_LIVE, REPO_JSON_LIVE_EXPLICIT_FALSE],
    ids=["flag_absent", "flag_false"],
)
def test_step7_skips_live_policy_with_message(full_env, live_json: str) -> None:
    r, calls = _run_script(full_env, STUB_REPO_JSON=live_json)
    assert r.returncode == 0, r.stderr
    assert _set_policy_calls(calls) == [], calls
    assert "SKIPPED step 7" in r.stdout and "LIVE" in r.stdout
    assert "FORCE_AR_POLICY_DRYRUN=1" in r.stdout


def test_step7_force_flag_applies_dry_run_over_live_policy(full_env) -> None:
    r, calls = _run_script(full_env, STUB_REPO_JSON=REPO_JSON_LIVE, FORCE_AR_POLICY_DRYRUN="1")
    assert r.returncode == 0, r.stderr
    applied = _set_policy_calls(calls)
    assert len(applied) == 1 and "--dry-run" in applied[0], calls
    assert "WARNING" in r.stdout


def test_step7_force_flag_other_values_do_not_override(full_env) -> None:
    r, calls = _run_script(full_env, STUB_REPO_JSON=REPO_JSON_LIVE, FORCE_AR_POLICY_DRYRUN="0")
    assert r.returncode == 0, r.stderr
    assert _set_policy_calls(calls) == []


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("describe_fails", {"STUB_REPO_RC": "1", "STUB_REPO_JSON": REPO_JSON_NO_POLICY}),
        ("empty_output", {"STUB_REPO_JSON": ""}),
        ("not_json", {"STUB_REPO_JSON": "ERROR: (gcloud) boom"}),
    ],
)
def test_step7_fails_closed_when_repo_state_unreadable(full_env, name: str, extra: dict) -> None:
    r, calls = _run_script(full_env, **extra)
    assert r.returncode != 0, name
    assert "ERROR" in r.stderr and "cleanup policy" in r.stderr, r.stderr
    assert _set_policy_calls(calls) == [], calls


def test_step7_unreadable_state_not_overridden_by_force_flag(full_env) -> None:
    r, calls = _run_script(full_env, STUB_REPO_RC="1", FORCE_AR_POLICY_DRYRUN="1")
    assert r.returncode != 0 and _set_policy_calls(calls) == []


def test_step7_dry_run_mode_prints_instead_of_applying(full_env) -> None:
    r, calls = _run_script(full_env, DRY_RUN="1", STUB_REPO_JSON=REPO_JSON_LIVE)
    assert r.returncode == 0, r.stderr
    assert _set_policy_calls(calls) == []
    assert "DRY_RUN: gcloud artifacts repositories set-cleanup-policies" not in r.stdout
