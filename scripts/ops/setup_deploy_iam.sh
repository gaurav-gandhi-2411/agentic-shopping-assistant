#!/usr/bin/env bash
# One-time (idempotent) IAM setup for the CI deploy pipeline (.github/workflows/deploy.yml).
#
# PREPARED, NOT RUN. GG runs this once, as gaurav.gandhi1129@gmail.com (the sole project owner).
# Every gcloud call below carries --account/--project explicitly (CLAUDE.md: shared gcloud config).
#
#   bash scripts/ops/setup_deploy_iam.sh            # apply
#   DRY_RUN=1 bash scripts/ops/setup_deploy_iam.sh  # print every command, change nothing
#
# Creates / binds, least privilege:
#   - SA stylemaitri-deployer
#   - roles/run.developer          on the Cloud Run SERVICE asa-stylist-api only (not project-wide)
#   - roles/iam.serviceAccountUser on the RUNTIME SA only (needed to deploy a revision running as it)
#   - roles/artifactregistry.writer on the ONE repo cloud-run-source-deploy only
#   - NO Cloud Build roles: images are built on the GitHub runner, not in Cloud Build.
#   - WIF: a NEW, SEPARATE pool (github-deploy-pool) holding ONE provider (github-deploy-provider)
#     whose attribute condition only accepts repo gaurav-gandhi-2411/agentic-shopping-assistant at
#     refs/heads/master. The existing github-pool / github-provider (owner-wide condition, used by
#     ci.yml on PRs) is NOT modified and is NOT referenced by any deployer binding.
#   Why a separate pool and not just a new provider in github-pool: a principalSet binding names a
#   POOL plus an attribute (principalSet://.../workloadIdentityPools/<pool>/attribute.<k>/<v>), so
#   it matches the identities of EVERY provider in that pool
#   (https://cloud.google.com/iam/docs/workload-identity-federation: custom attributes are used "in
#   IAM principalSet:// role bindings to grant access to all identities with a certain attribute").
#   A deployer bound on github-pool would be impersonable by any PR/branch token that the old
#   owner-wide provider issues. Google also advises "use a single provider per workload identity
#   pool" (https://cloud.google.com/iam/docs/best-practices-for-using-workload-identity-federation).
#   The script ends with a check that the deployer SA's policy references ONLY github-deploy-pool.
# It does not grant anything to gaurav.gandhi2411@ or gaurav.gandhi1249@ (they must hold zero).
#
# Afterwards, add two repo secrets (values are printed at the end; they are identifiers, not keys):
#   GCP_DEPLOY_WIF_PROVIDER, GCP_DEPLOY_SERVICE_ACCOUNT
set -euo pipefail

ACCOUNT="gaurav.gandhi1129@gmail.com"
PROJECT="stylemaitri-prod-260813"
PROJECT_NUMBER="631709154646"
REGION="asia-south1"
SERVICE="asa-stylist-api"
AR_REPO="cloud-run-source-deploy"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
DEPLOYER_NAME="stylemaitri-deployer"
DEPLOYER_SA="${DEPLOYER_NAME}@${PROJECT}.iam.gserviceaccount.com"
POOL="github-deploy-pool"        # dedicated to the deployer; never reuse the CI pool (github-pool)
LEGACY_POOL="github-pool"        # CI pool; must NOT appear in the deployer SA policy
PROVIDER="github-deploy-provider"
GH_REPO="gaurav-gandhi-2411/agentic-shopping-assistant"
POOL_PATH="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run (or just print, under DRY_RUN=1) a command. Describe-style existence probes always run.
run() {
  if [ "${DRY_RUN:-0}" = "1" ]; then printf 'DRY_RUN:'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}
exists() { "$@" >/dev/null 2>&1; }

# The one principalSet that may impersonate the deployer: full path incl. project number and pool.
EXPECTED_MEMBER="principalSet://iam.googleapis.com/${POOL_PATH}/attribute.repository/${GH_REPO}"

# Read-only. A soft-deleted pool/provider still answers `describe` (so `exists` would say "yes" and
# the script would "update" something that is scheduled for deletion). Stop and print the undelete
# command instead. usage: guard_not_deleted <label> <undelete command text> <describe cmd...>
guard_not_deleted() {
  local label="$1" undelete="$2" state
  shift 2
  state="$("$@" --format='value(state)' 2>/dev/null || true)"
  if [ "$state" = "DELETED" ]; then
    {
      echo "ERROR: ${label} is in state DELETED (soft-deleted; not usable). Undelete it, then re-run:"
      echo "  ${undelete}"
    } >&2
    return 1
  fi
}

# Fails (return 1, message on stderr) unless the deployer SA policy grants
# roles/iam.workloadIdentityUser to EXACTLY one member, EXPECTED_MEMBER, and no role at all is held
# by any other workloadIdentityPools principal. Any other member (allUsers, user:, serviceAccount:,
# another principal:// or principalSet://, legacy pool, wrong project number) is an ERROR.
# Reads flattened "role<TAB>member" rows so there is no grep/pipeline that can exit silently under
# `set -o pipefail`; empty or blank output is itself an ERROR, never a pass.
verify_deployer_policy() {
  local rows role member n_ok=0 bad=0
  if ! rows="$(gcloud iam service-accounts get-iam-policy "$DEPLOYER_SA" \
      --account="$ACCOUNT" --project="$PROJECT" \
      --flatten='bindings[].members' --format='value(bindings.role,bindings.members)')"; then
    echo "ERROR: could not read the IAM policy of ${DEPLOYER_SA}" >&2
    return 1
  fi
  if [ -z "$(printf '%s' "$rows" | tr -d '[:space:]')" ]; then
    echo "ERROR: deployer SA policy is empty or unreadable; expected ${EXPECTED_MEMBER}" >&2
    return 1
  fi
  while IFS=$'\t' read -r role member; do
    [ -z "${role}${member}" ] && continue
    if [ "$role" = "roles/iam.workloadIdentityUser" ]; then
      if [ "$member" = "$EXPECTED_MEMBER" ]; then
        n_ok=$((n_ok + 1))
      else
        echo "ERROR: unexpected workloadIdentityUser member: ${member}" >&2
        bad=1
      fi
    elif [[ "$member" == *workloadIdentityPools/* ]]; then
      echo "ERROR: ${member} holds ${role} on the deployer SA; only ${POOL} may appear, and only as workloadIdentityUser" >&2
      bad=1
    fi
  done <<< "$rows"
  if [ "$n_ok" -ne 1 ]; then
    echo "ERROR: expected exactly one workloadIdentityUser binding for ${EXPECTED_MEMBER}, found ${n_ok}" >&2
    return 1
  fi
  [ "$bad" -eq 0 ]
}

# Test hook: `SETUP_DEPLOY_IAM_SOURCE_ONLY=1 source scripts/ops/setup_deploy_iam.sh` loads the
# functions above without running any step (used by tests/ with a stub gcloud).
if [ "${SETUP_DEPLOY_IAM_SOURCE_ONLY:-0}" = "1" ]; then
  return 0 2>/dev/null || exit 0
fi

echo "== 1. deployer service account"
if exists gcloud iam service-accounts describe "$DEPLOYER_SA" --account="$ACCOUNT" --project="$PROJECT"; then
  echo "exists: $DEPLOYER_SA"
else
  run gcloud iam service-accounts create "$DEPLOYER_NAME" \
    --display-name="StyleMaitri CI deployer (Cloud Run + AR, master only)" \
    --account="$ACCOUNT" --project="$PROJECT"
fi

echo "== 2. run.developer on the service only"
# add-iam-policy-binding is idempotent (re-adding an existing member/role is a no-op).
# UNVERIFIED: that a service-scoped run.developer is sufficient for `gcloud run deploy` of an
# existing service (operations.get is project-scoped). If the first CI deploy fails with a
# run.operations.* permission error, fall back to the project-level binding on the line below.
run gcloud run services add-iam-policy-binding "$SERVICE" \
  --region="$REGION" \
  --member="serviceAccount:${DEPLOYER_SA}" --role="roles/run.developer" \
  --account="$ACCOUNT" --project="$PROJECT"
# FALLBACK (only if needed, project-wide, broader):
#   gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:${DEPLOYER_SA}" \
#     --role="roles/run.developer" --account="$ACCOUNT"

echo "== 3. actAs on the runtime SA only"
run gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --member="serviceAccount:${DEPLOYER_SA}" --role="roles/iam.serviceAccountUser" \
  --account="$ACCOUNT" --project="$PROJECT"

echo "== 4. Artifact Registry writer on the one repo"
run gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" \
  --location="$REGION" \
  --member="serviceAccount:${DEPLOYER_SA}" --role="roles/artifactregistry.writer" \
  --account="$ACCOUNT" --project="$PROJECT"

echo "== 5a. dedicated WIF pool"
guard_not_deleted "pool ${POOL}" \
  "gcloud iam workload-identity-pools undelete ${POOL} --location=global --account=${ACCOUNT} --project=${PROJECT}" \
  gcloud iam workload-identity-pools describe "$POOL" \
    --location=global --account="$ACCOUNT" --project="$PROJECT" || exit 1
if exists gcloud iam workload-identity-pools describe "$POOL" \
    --location=global --account="$ACCOUNT" --project="$PROJECT"; then
  echo "exists: $POOL"
else
  run gcloud iam workload-identity-pools create "$POOL" \
    --location=global --display-name="GitHub deploy (master only)" \
    --account="$ACCOUNT" --project="$PROJECT"
fi

echo "== 5b. WIF provider restricted to repo + master ref (the pool's only provider)"
CONDITION="assertion.repository=='${GH_REPO}' && assertion.ref=='refs/heads/master'"
MAPPING="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref"
guard_not_deleted "provider ${PROVIDER}" \
  "gcloud iam workload-identity-pools providers undelete ${PROVIDER} --workload-identity-pool=${POOL} --location=global --account=${ACCOUNT} --project=${PROJECT}" \
  gcloud iam workload-identity-pools providers describe "$PROVIDER" \
    --workload-identity-pool="$POOL" --location=global --account="$ACCOUNT" --project="$PROJECT" || exit 1
if exists gcloud iam workload-identity-pools providers describe "$PROVIDER" \
    --workload-identity-pool="$POOL" --location=global --account="$ACCOUNT" --project="$PROJECT"; then
  # Re-assert the condition so a hand-edited provider converges back to the intended one.
  run gcloud iam workload-identity-pools providers update-oidc "$PROVIDER" \
    --workload-identity-pool="$POOL" --location=global \
    --attribute-mapping="$MAPPING" --attribute-condition="$CONDITION" \
    --account="$ACCOUNT" --project="$PROJECT"
else
  run gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --workload-identity-pool="$POOL" --location=global \
    --display-name="GitHub OIDC (deploy, master only)" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="$MAPPING" --attribute-condition="$CONDITION" \
    --account="$ACCOUNT" --project="$PROJECT"
fi

echo "== 6. let identities of github-deploy-pool (filtered to master by its only provider) impersonate the deployer"
run gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" \
  --member="$EXPECTED_MEMBER" \
  --role="roles/iam.workloadIdentityUser" \
  --account="$ACCOUNT" --project="$PROJECT"

echo "== 6b. verify: ONLY ${POOL} may impersonate the deployer (exact principalSet, nothing else)"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN: skipped (the deployer SA may not exist yet); this check runs on a real apply"
else
  verify_deployer_policy || exit 1
  echo "OK: exactly ${EXPECTED_MEMBER} can impersonate the deployer"
fi

echo "== 7. Artifact Registry cleanup policy: DRY RUN (deletes nothing)"
# --dry-run only LOGS what would be deleted (Cloud Audit Logs); review that before step 8.
# keepCount 3 = newest three versions of the package; older than 1 day and not in that set is
# deleted. Docker images referenced by a parent manifest are never deleted by a policy
# (https://cloud.google.com/artifact-registry/docs/repositories/cleanup-policy).
run gcloud artifacts repositories set-cleanup-policies "$AR_REPO" \
  --location="$REGION" --policy="${HERE}/ar_cleanup_policy.json" --dry-run \
  --account="$ACCOUNT" --project="$PROJECT"

echo "== 8. to ACTIVATE deletion (run by hand after reviewing the dry-run log):"
echo "gcloud artifacts repositories set-cleanup-policies ${AR_REPO} --location=${REGION} \\"
echo "  --policy=scripts/ops/ar_cleanup_policy.json --no-dry-run \\"
echo "  --account=${ACCOUNT} --project=${PROJECT}"
echo "# review dry-run results first, e.g.:"
echo "gcloud logging read 'protoPayload.serviceName=\"artifactregistry.googleapis.com\" AND protoPayload.methodName:\"DeleteVersion\"' \\"
echo "  --freshness=3d --account=${ACCOUNT} --project=${PROJECT}"

echo
echo "== Done. Add these two GitHub repo secrets (Settings > Secrets and variables > Actions):"
echo "GCP_DEPLOY_WIF_PROVIDER      = ${POOL_PATH}/providers/${PROVIDER}"
echo "GCP_DEPLOY_SERVICE_ACCOUNT   = ${DEPLOYER_SA}"
echo
echo "== Verify no retired identity holds any binding (expect no output):"
echo "gcloud projects get-iam-policy ${PROJECT} --account=${ACCOUNT} --format=json | grep -E 'gandhi2411|gandhi1249'"
