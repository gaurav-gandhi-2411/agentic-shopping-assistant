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
  --member="principalSet://iam.googleapis.com/${POOL_PATH}/attribute.repository/${GH_REPO}" \
  --role="roles/iam.workloadIdentityUser" \
  --account="$ACCOUNT" --project="$PROJECT"

echo "== 6b. verify: deployer SA policy references ONLY ${POOL} (never ${LEGACY_POOL})"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN: skipped (the deployer SA may not exist yet); this check runs on a real apply"
else
  POLICY_JSON="$(gcloud iam service-accounts get-iam-policy "$DEPLOYER_SA" \
    --account="$ACCOUNT" --project="$PROJECT" --format=json)"
  POOLS_SEEN="$(printf '%s' "$POLICY_JSON" | grep -o 'workloadIdentityPools/[A-Za-z0-9-]*' | sort -u)"
  echo "pools referenced: ${POOLS_SEEN:-<none>}"
  if [ "$POOLS_SEEN" != "workloadIdentityPools/${POOL}" ]; then
    echo "ERROR: deployer SA policy must reference exactly workloadIdentityPools/${POOL}; got: ${POOLS_SEEN:-<none>}" >&2
    exit 1
  fi
  echo "OK: only ${POOL} can impersonate the deployer"
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
