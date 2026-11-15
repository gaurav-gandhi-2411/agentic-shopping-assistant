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
#   - roles/iam.serviceAccountUser on the CURRENT runtime SA (the default compute SA) only (needed to
#     deploy a revision running as it) AND on the NEW runtime SA stylemaitri-runtime (step 4f)
#   - roles/artifactregistry.writer on the ONE repo cloud-run-source-deploy only
#   - the custom role artifactRegistryTagMover (tags.create/update/delete/get/list) on that same repo
#     only: deploy.yml's pin_images step moves the live/prev tags and the writer role lacks
#     tags.delete. (It was first created by hand on 2026-11-13; this script now converges to it.)
#   - a NEW dedicated runtime SA stylemaitri-runtime, least privilege (steps 4c-4e): secretAccessor
#     on each of the 5 secrets the service reads (secret-level), objectViewer on the index bucket
#     (bucket-level), nothing else. Nothing is switched: the service keeps running as the compute SA
#     until the DEPLOY.md runbook ("Switch Cloud Run to the dedicated runtime service account") is
#     followed. This script never touches the compute SA's own roles.
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
# The service CURRENTLY runs as the default compute SA (holds roles/editor: too broad; see DEPLOY.md).
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
# The dedicated least-privilege runtime SA the service will be switched to (runbook in DEPLOY.md).
RUNTIME_NAME="stylemaitri-runtime"
RUNTIME_SA="${RUNTIME_NAME}@${PROJECT}.iam.gserviceaccount.com"
# The 5 secrets the service reads (each mounted as an env var at :latest). Secret-level bindings only.
SECRETS=(asa-database-url asa-demo-jwt-secret asa-groq-api-key asa-supabase-url CANARY_TOKEN)
INDEX_BUCKET="asa-demo-indices-stylemaitri-prod"
# Custom role created by hand on 2026-11-13; this script converges to exactly these five.
TAG_ROLE_ID="artifactRegistryTagMover"
TAG_ROLE="projects/${PROJECT}/roles/${TAG_ROLE_ID}"
TAG_ROLE_PERMS="artifactregistry.tags.create,artifactregistry.tags.delete,artifactregistry.tags.get,artifactregistry.tags.list,artifactregistry.tags.update"
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
  # gcloud on Windows emits CRLF; a stray \r would make "DELETED\r" != "DELETED" (fail open).
  state="${state//$'\r'/}"
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
  # gcloud on Windows emits CRLF rows; strip \r so members compare exactly.
  rows="${rows//$'\r'/}"
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

# Fails unless the deployer SA holds EXACTLY {roles/artifactregistry.writer, TAG_ROLE} on the AR
# repo (nothing broader, nothing missing). Same fail-closed row handling as verify_deployer_policy.
verify_deployer_repo_roles() {
  local rows role member n_writer=0 n_tag=0 bad=0
  if ! rows="$(gcloud artifacts repositories get-iam-policy "$AR_REPO" --location="$REGION" \
      --account="$ACCOUNT" --project="$PROJECT" \
      --flatten='bindings[].members' --format='value(bindings.role,bindings.members)')"; then
    echo "ERROR: could not read the IAM policy of AR repo ${AR_REPO}" >&2
    return 1
  fi
  rows="${rows//$'\r'/}"
  if [ -z "$(printf '%s' "$rows" | tr -d '[:space:]')" ]; then
    echo "ERROR: AR repo policy is empty or unreadable; expected the deployer's two roles" >&2
    return 1
  fi
  while IFS=$'\t' read -r role member; do
    [ "$member" = "serviceAccount:${DEPLOYER_SA}" ] || continue
    case "$role" in
      roles/artifactregistry.writer) n_writer=$((n_writer + 1)) ;;
      "$TAG_ROLE") n_tag=$((n_tag + 1)) ;;
      *) echo "ERROR: unexpected role ${role} for the deployer on repo ${AR_REPO}" >&2; bad=1 ;;
    esac
  done <<< "$rows"
  if [ "$n_writer" -ne 1 ] || [ "$n_tag" -ne 1 ]; then
    echo "ERROR: deployer must hold exactly roles/artifactregistry.writer and ${TAG_ROLE} on ${AR_REPO} (found writer=${n_writer}, tagMover=${n_tag})" >&2
    return 1
  fi
  [ "$bad" -eq 0 ]
}

# Fails unless the PROJECT-level policy has NO binding at all for the runtime SA: its least
# privilege lives only in secret-level and bucket-level bindings.
verify_runtime_sa_no_project_roles() {
  local rows
  if ! rows="$(gcloud projects get-iam-policy "$PROJECT" --account="$ACCOUNT" \
      --flatten='bindings[].members' --format='value(bindings.role,bindings.members)')"; then
    echo "ERROR: could not read the project IAM policy" >&2
    return 1
  fi
  rows="${rows//$'\r'/}"
  if [ -z "$(printf '%s' "$rows" | tr -d '[:space:]')" ]; then
    echo "ERROR: project IAM policy is empty or unreadable" >&2
    return 1
  fi
  if printf '%s\n' "$rows" | grep -F -q "serviceAccount:${RUNTIME_SA}"; then
    echo "ERROR: ${RUNTIME_SA} holds a PROJECT-level role; it must hold none (see rows below)" >&2
    printf '%s\n' "$rows" | grep -F "serviceAccount:${RUNTIME_SA}" >&2 || true
    return 1
  fi
}

# Prints the AR repo's cleanup-policy state, one of: none (no policies), dryrun (policies, dry-run
# mode), live (policies, deletion ACTIVE). Fails closed (return 1, message on stderr) if the repo
# cannot be read or the answer is not a JSON object: an unreadable state must never be treated as
# "no policies", because step 7 would then re-apply --dry-run over a live policy.
# Whitespace is stripped so the match does not depend on gcloud's pretty-printing or CRLF output.
ar_cleanup_policy_state() {
  local json
  if ! json="$(gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" \
      --account="$ACCOUNT" --project="$PROJECT" --format=json)"; then
    echo "ERROR: could not read AR repo ${AR_REPO}; refusing to touch its cleanup policy" >&2
    return 1
  fi
  json="$(printf '%s' "$json" | tr -d '[:space:]')"
  case "$json" in
    "{"*"}") ;;
    *)
      echo "ERROR: AR repo ${AR_REPO} describe output is empty or not a JSON object; refusing to touch its cleanup policy" >&2
      return 1
      ;;
  esac
  if [[ "$json" != *'"cleanupPolicies":{"'* ]]; then
    echo none
  elif [[ "$json" == *'"cleanupPolicyDryRun":true'* ]]; then
    echo dryrun
  else
    echo live
  fi
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

echo "== 3. actAs on the current runtime SA (default compute SA) only"
run gcloud iam service-accounts add-iam-policy-binding "$COMPUTE_SA" \
  --member="serviceAccount:${DEPLOYER_SA}" --role="roles/iam.serviceAccountUser" \
  --account="$ACCOUNT" --project="$PROJECT"

echo "== 4. Artifact Registry writer on the one repo"
run gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" \
  --location="$REGION" \
  --member="serviceAccount:${DEPLOYER_SA}" --role="roles/artifactregistry.writer" \
  --account="$ACCOUNT" --project="$PROJECT"

echo "== 4b. custom role ${TAG_ROLE_ID} (tag create/update/delete/get/list) on the one repo"
# roles/artifactregistry.writer lacks artifactregistry.tags.delete, which deploy.yml's pin_images
# step needs to move the live/prev tags. Create the role if missing; if present, converge its
# permission list to exactly the five (a describe-and-compare, so an unchanged role is not rewritten).
if exists gcloud iam roles describe "$TAG_ROLE_ID" --account="$ACCOUNT" --project="$PROJECT"; then
  CURRENT_PERMS="$(gcloud iam roles describe "$TAG_ROLE_ID" --account="$ACCOUNT" --project="$PROJECT" \
    --format='value(includedPermissions)' | tr -d '\r' | tr ';' '\n' | sort | paste -sd, -)"
  if [ "$CURRENT_PERMS" = "$TAG_ROLE_PERMS" ]; then
    echo "exists, permissions already exact: $TAG_ROLE"
  else
    run gcloud iam roles update "$TAG_ROLE_ID" --permissions="$TAG_ROLE_PERMS" \
      --account="$ACCOUNT" --project="$PROJECT"
  fi
else
  run gcloud iam roles create "$TAG_ROLE_ID" --title="Artifact Registry tag mover" \
    --description="Create/update/delete/get/list Artifact Registry tags only (deploy.yml pin_images)" \
    --permissions="$TAG_ROLE_PERMS" --stage=GA \
    --account="$ACCOUNT" --project="$PROJECT"
fi
run gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" \
  --location="$REGION" \
  --member="serviceAccount:${DEPLOYER_SA}" --role="$TAG_ROLE" \
  --account="$ACCOUNT" --project="$PROJECT"

echo "== 4c. dedicated runtime service account ${RUNTIME_NAME} (NOT yet attached to the service)"
if exists gcloud iam service-accounts describe "$RUNTIME_SA" --account="$ACCOUNT" --project="$PROJECT"; then
  echo "exists: $RUNTIME_SA"
else
  run gcloud iam service-accounts create "$RUNTIME_NAME" \
    --display-name="StyleMaitri Cloud Run runtime (least privilege)" \
    --account="$ACCOUNT" --project="$PROJECT"
  # A just-created SA is not always visible to IAM binding calls for a few seconds.
  run sleep "${SA_PROPAGATION_WAIT:-10}"
fi

echo "== 4d. secretAccessor on each of the 5 secrets (secret-level, never project-level)"
for secret in "${SECRETS[@]}"; do
  run gcloud secrets add-iam-policy-binding "$secret" \
    --member="serviceAccount:${RUNTIME_SA}" --role="roles/secretmanager.secretAccessor" \
    --account="$ACCOUNT" --project="$PROJECT"
done

echo "== 4e. objectViewer on the index bucket only (bucket-level, never project-level)"
# READ-ONLY on purpose. The app only downloads from this bucket: src/retrieval/index_store.py
# uses storage.Client() + bucket.list_blobs() + blob.download_to_filename() and nothing else
# (needs storage.objects.list + storage.objects.get, both in objectViewer). No upload/delete/
# create call exists in api/ or src/. Index uploads stay a manual step done by GG.
run gcloud storage buckets add-iam-policy-binding "gs://${INDEX_BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/storage.objectViewer" \
  --account="$ACCOUNT" --project="$PROJECT"
# roles/logging.logWriter is DELIBERATELY NOT granted. Cloud Run's platform ships container
# stdout/stderr to Cloud Logging itself, without the runtime SA holding any logging role. The app
# never calls the Cloud Logging API (no google-cloud-logging dependency; it uses stdlib logging and
# structured stdout). logWriter could only be bound at project level (no narrower scope), so it is
# left out until something provably needs it.
# Also deliberately absent: roles/editor, Cloud SQL (the DB is external, reached by URL secret), run.*.

echo "== 4f. actAs for the deployer on the NEW runtime SA (SA-level), so deploys can run revisions as it"
run gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --member="serviceAccount:${DEPLOYER_SA}" --role="roles/iam.serviceAccountUser" \
  --account="$ACCOUNT" --project="$PROJECT"

echo "== 4g. OPT-IN: remove the deployer's actAs on the compute SA (REMOVE_COMPUTE_ACTAS=1 only)"
# Run ONLY after: the service was switched to ${RUNTIME_NAME}, a CI deploy has succeeded with it,
# and the rollback revision no longer runs as the compute SA (a revision running as the compute SA
# cannot be redeployed by the deployer once this binding is gone). Fails closed: refuses unless the
# service's template currently names the new runtime SA.
if [ "${REMOVE_COMPUTE_ACTAS:-0}" = "1" ]; then
  CURRENT_SERVICE_SA="$(gcloud run services describe "$SERVICE" --region="$REGION" \
    --account="$ACCOUNT" --project="$PROJECT" \
    --format='value(spec.template.spec.serviceAccountName)' | tr -d '\r')"
  if [ "$CURRENT_SERVICE_SA" != "$RUNTIME_SA" ]; then
    echo "ERROR: REMOVE_COMPUTE_ACTAS=1 but ${SERVICE} runs as '${CURRENT_SERVICE_SA}', not ${RUNTIME_SA}. Switch the service first (DEPLOY.md)." >&2
    exit 1
  fi
  run gcloud iam service-accounts remove-iam-policy-binding "$COMPUTE_SA" \
    --member="serviceAccount:${DEPLOYER_SA}" --role="roles/iam.serviceAccountUser" \
    --account="$ACCOUNT" --project="$PROJECT"
else
  echo "skipped (default). Set REMOVE_COMPUTE_ACTAS=1 only after the switch; see DEPLOY.md."
fi

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
    --display-name="GitHub OIDC (deploy, master)" \
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

echo "== 6c. verify: deployer holds exactly {artifactregistry.writer, ${TAG_ROLE_ID}} on the repo; runtime SA holds no project-level role"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN: skipped (the SAs and role may not exist yet); these checks run on a real apply"
else
  verify_deployer_repo_roles || exit 1
  echo "OK: deployer bindings on ${AR_REPO} are exactly writer + ${TAG_ROLE_ID}"
  verify_runtime_sa_no_project_roles || exit 1
  echo "OK: ${RUNTIME_SA} holds no project-level role"
fi

echo "== 7. Artifact Registry cleanup policy: DRY RUN (deletes nothing)"
# --dry-run only LOGS what would be deleted (Cloud Audit Logs); review that before step 8.
# keepCount 2 = newest two versions (serving image + one rollback target, all the workflow's
# rollback needs). keepCount alone cannot pin the serving digest: a pushed image that fails its
# gate (never serves) plus a second push could push the serving image out of the newest two. So a
# second Keep rule pins tags live/prev, which the deploy workflow moves to the serving and previous
# images after each fully successful deploy. Older than 1 day and not in either keep set is
# deleted. Docker images referenced by a parent manifest are never deleted by a policy
# (https://cloud.google.com/artifact-registry/docs/repositories/cleanup-policy).
# GUARD: set-cleanup-policies --dry-run REPLACES the repo's mode, so re-running this script after
# step 8 was done by hand would silently switch a LIVE policy back to dry-run (happened once,
# caught by hand). Apply only when there is no policy yet or the existing one is already dry-run.
# FORCE_AR_POLICY_DRYRUN=1 is the explicit override (deliberately re-arm dry-run on a live repo).
if ! AR_POLICY_STATE="$(ar_cleanup_policy_state)"; then
  exit 1
fi
if [ "$AR_POLICY_STATE" = "live" ] && [ "${FORCE_AR_POLICY_DRYRUN:-0}" != "1" ]; then
  echo "SKIPPED step 7: ${AR_REPO} already has a LIVE (non-dry-run) cleanup policy; not switching it back to dry-run."
  echo "  To deliberately re-arm dry-run, re-run with FORCE_AR_POLICY_DRYRUN=1."
else
  if [ "$AR_POLICY_STATE" = "live" ]; then
    echo "WARNING: FORCE_AR_POLICY_DRYRUN=1: switching the LIVE cleanup policy on ${AR_REPO} back to dry-run."
  fi
  run gcloud artifacts repositories set-cleanup-policies "$AR_REPO" \
    --location="$REGION" --policy="${HERE}/ar_cleanup_policy.json" --dry-run \
    --account="$ACCOUNT" --project="$PROJECT"
fi

echo "== 8. to ACTIVATE deletion (run by hand after reviewing the dry-run log):"
echo "gcloud artifacts repositories set-cleanup-policies ${AR_REPO} --location=${REGION} \\"
echo "  --policy=scripts/ops/ar_cleanup_policy.json --no-dry-run \\"
echo "  --account=${ACCOUNT} --project=${PROJECT}"
echo "# ORDER: deploy first (the first deploy after merge creates the live/prev tags), apply this policy second."
echo "# FIRST confirm the live tag exists and points at the serving digest (compare with the image below):"
echo "gcloud artifacts docker tags list ${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/<package> --account=${ACCOUNT} --project=${PROJECT}"
echo "gcloud run services describe <service> --region=${REGION} --format='value(spec.template.spec.containers[0].image)' --account=${ACCOUNT} --project=${PROJECT}"
echo "gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO} --sort-by=~UPDATE_TIME --limit=3 --account=${ACCOUNT} --project=${PROJECT}"
echo "# review dry-run results first, e.g.:"
echo "gcloud logging read 'protoPayload.serviceName=\"artifactregistry.googleapis.com\" AND protoPayload.methodName:\"DeleteVersion\"' \\"
echo "  --freshness=3d --account=${ACCOUNT} --project=${PROJECT}"

echo
echo "== 9. MANUAL follow-ups, NOT automated here (each needs explicit approval; see DEPLOY.md):"
echo "# a) Switch the service to ${RUNTIME_SA}: runbook 'Switch Cloud Run to the dedicated runtime"
echo "#    service account' (no-traffic tagged revision, smoke the tag URL, then shift traffic)."
echo "# b) After a later successful deploy and once the rollback revision has been replaced:"
echo "#    REMOVE_COMPUTE_ACTAS=1 bash scripts/ops/setup_deploy_iam.sh"
echo "# c) Much later, separately approved: drop the compute SA's per-secret bindings and its"
echo "#    project-level roles/editor (exact commands in DEPLOY.md)."

echo
echo "== Done. Add these two GitHub repo secrets (Settings > Secrets and variables > Actions):"
echo "GCP_DEPLOY_WIF_PROVIDER      = ${POOL_PATH}/providers/${PROVIDER}"
echo "GCP_DEPLOY_SERVICE_ACCOUNT   = ${DEPLOYER_SA}"
echo
echo "== Verify no retired identity holds any binding (expect no output):"
echo "gcloud projects get-iam-policy ${PROJECT} --account=${ACCOUNT} --format=json | grep -E 'gandhi2411|gandhi1249'"
