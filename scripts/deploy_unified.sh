#!/usr/bin/env bash
# deploy_unified.sh — upload index to GCS, build + push Docker image, then deploy asa-stylist-api.
#
# Steps:
#   1. Upload rebuilt unified index to GCS (overwrite — NEVER skip; Cloud Run loads from GCS at startup).
#   2–3. Configure Docker auth and build image tagged with the short git SHA.
#   4–5. Push image to Artifact Registry.
#   6. Create new Cloud Run revision with --no-traffic.
#   7. Wait until the revision is Ready.
#   8. Migrate 100% of traffic to the new revision explicitly by name.
#   9. Delete the previous revision so 'latestCreated' stays clean.
#  10. Redeploy Vercel frontend.
#
# QA rule: smoke-test MUST hit the live Cloud Run URL, never localhost.
# A local index/server can diverge from GCS; local results prove nothing about production.
#
# Usage:
#   bash scripts/deploy_unified.sh
#
# Prerequisites:
#   gcloud auth login && gcloud auth configure-docker asia-south1-docker.pkg.dev
#   Docker daemon running locally (or build via Cloud Build — see comment in Step 2)
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GCP_PROJECT="iconic-reactor-496423-m4"
GAR_REGION="asia-south1"
GAR_REPO="shopping-assistant"
SERVICE="asa-stylist-api"
IMAGE_NAME="asa-api"
IMAGE_TAG="$(git rev-parse --short HEAD)"
IMAGE="${GAR_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

# GCS paths for the unified search index.  The container loads these at startup
# via INDEX_STORE_URI=gs://asa-demo-indices/unified/ — the code resolves to
# gs://asa-demo-indices/unified/unified/ for main index files and
# gs://asa-demo-indices/unified/clip/unified/ for CLIP vectors.
GCS_INDEX_BUCKET="gs://asa-demo-indices/unified"
LOCAL_INDEX_DIR="$(git rev-parse --show-toplevel)/data/processed/unified"
LOCAL_CLIP_DIR="$(git rev-parse --show-toplevel)/data/processed/clip/unified"

# Secret Manager references — these secrets exist in the project under these names.
# Never change the secret *names* here without first verifying they exist:
#   gcloud secrets list --project="${GCP_PROJECT}"
SECRETS="GROQ_API_KEY=asa-groq-api-key:2,DEMO_JWT_SECRET=asa-demo-jwt-secret:2,DATABASE_URL=asa-database-url:2,SUPABASE_URL=asa-supabase-url:2"

# ---------------------------------------------------------------------------
# Step 1: Upload rebuilt index to GCS (ALWAYS overwrite — never use --no-clobber)
#
# The Docker image does NOT bundle index files; Cloud Run downloads them from GCS
# at container startup.  Skipping this step means Cloud Run serves stale index
# data even when the code is up-to-date.  --no-clobber would silently skip files
# that already exist and leave the old index in place.
#
# This step must run every time a new index is built, regardless of whether the
# Docker image changed.  If you only changed LLM/search code and not the index,
# you can skip Step 2–4 and just run gcloud run deploy --no-traffic (Step 5)
# to force a cold restart that re-downloads the current GCS index.
# ---------------------------------------------------------------------------
echo "=== Step 1: Upload unified index to GCS (overwrite) ==="

if [[ ! -f "${LOCAL_INDEX_DIR}/catalogue.parquet" ]]; then
  echo "ERROR: ${LOCAL_INDEX_DIR}/catalogue.parquet not found."
  echo "Run the index-build script first: python scripts/build_unified_index.py"
  exit 1
fi

gcloud storage cp \
  "${LOCAL_INDEX_DIR}/catalogue.parquet" \
  "${LOCAL_INDEX_DIR}/dense.faiss" \
  "${LOCAL_INDEX_DIR}/dense_article_ids.npy" \
  "${LOCAL_INDEX_DIR}/bm25.pkl" \
  "${LOCAL_INDEX_DIR}/bm25_article_ids.npy" \
  "${GCS_INDEX_BUCKET}/unified/"
echo "  Main index uploaded to ${GCS_INDEX_BUCKET}/unified/"

if [[ -f "${LOCAL_CLIP_DIR}/clip.faiss" ]]; then
  gcloud storage cp \
    "${LOCAL_CLIP_DIR}/clip.faiss" \
    "${LOCAL_CLIP_DIR}/clip_article_ids.npy" \
    "${GCS_INDEX_BUCKET}/clip/unified/"
  echo "  CLIP index uploaded to ${GCS_INDEX_BUCKET}/clip/unified/"
else
  echo "  WARN: no CLIP index found at ${LOCAL_CLIP_DIR} — skipping (image-search unavailable)"
fi

echo "  Verifying GCS timestamps (must show today):"
gcloud storage ls --long "${GCS_INDEX_BUCKET}/unified/" | grep -v "^TOTAL"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Build Docker image
# ---------------------------------------------------------------------------
echo "=== Step 4: Build Docker image (${IMAGE_TAG}) ==="
docker build -t "${IMAGE}" .

# ---------------------------------------------------------------------------
# Step 5: Push to Artifact Registry
# ---------------------------------------------------------------------------
echo "=== Step 5: Push ${IMAGE} ==="
docker push "${IMAGE}"

# ---------------------------------------------------------------------------
# Step 6: Deploy new revision with --no-traffic
#   The revision is created but receives 0% of traffic until Step 8.
#   --set-secrets wires Secret Manager refs; the format is ENV_VAR=SECRET_NAME:VERSION.
# ---------------------------------------------------------------------------
echo "=== Step 6: Create new revision (no traffic) ==="
# max-instances=1: api/rate_limit.py's IP rate limiter and chat.py's _DEMO_SESSIONS
# are both in-memory and assume a single running instance. >1 instance would let
# traffic silently bypass the demo rate/cost caps and would fragment in-memory
# session state across instances.
# TEMPORARY: DEMO_PER_IP_HOUR_LIMIT and DEMO_DAILY_REQUEST_CAP are raised above
# their defaults (10/IP/hour, 200/brand/day) for GG's testing window. --set-env-vars
# replaces the full env-var set, so omitting these here would silently clobber the
# raised limits currently live on the service. Remove this override in the
# pre-public hardening pass (see project_prepublic_hardening memory).
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}" \
  --region="${GAR_REGION}" \
  --no-traffic \
  --set-env-vars="DEMO_MODE=true,LLM_PROVIDER=groq,CORS_ORIGINS=https://stylemaitri.vercel.app,INDEX_STORE_URI=gs://asa-demo-indices/unified/,DEMO_PER_IP_HOUR_LIMIT=1000,DEMO_DAILY_REQUEST_CAP=5000" \
  --set-secrets="${SECRETS}" \
  --memory=4Gi \
  --cpu=1 \
  --concurrency=4 \
  --timeout=300 \
  --max-instances=1

# ---------------------------------------------------------------------------
# Step 7: Capture the new revision name and wait for Ready
# ---------------------------------------------------------------------------
echo "=== Step 7: Wait for new revision to be Ready ==="
NEW_REVISION=$(gcloud run services describe "${SERVICE}" \
  --region="${GAR_REGION}" \
  --format="value(status.latestCreatedRevisionName)")
echo "  New revision: ${NEW_REVISION}"

ATTEMPTS=0
until gcloud run revisions describe "${NEW_REVISION}" \
    --region="${GAR_REGION}" \
    --format="value(status.conditions[0].status)" 2>/dev/null | grep -q "^True$"; do
  ATTEMPTS=$((ATTEMPTS + 1))
  if [[ ${ATTEMPTS} -ge 20 ]]; then
    echo "ERROR: ${NEW_REVISION} did not become Ready after ~100 s"
    gcloud run revisions describe "${NEW_REVISION}" \
      --region="${GAR_REGION}" \
      --format="value(status.conditions[0].message)"
    exit 1
  fi
  echo "  waiting... (${ATTEMPTS}/20)"
  sleep 5
done
echo "  ${NEW_REVISION} is Ready"

# ---------------------------------------------------------------------------
# Step 8: Migrate 100% of traffic to the new revision (explicit, not latestRevision)
# ---------------------------------------------------------------------------
echo "=== Step 8: Migrate traffic → ${NEW_REVISION} ==="
PREV_REVISION=$(gcloud run services describe "${SERVICE}" \
  --region="${GAR_REGION}" \
  --format="value(status.traffic[0].revisionName)")

gcloud run services update-traffic "${SERVICE}" \
  --to-revisions "${NEW_REVISION}=100" \
  --region="${GAR_REGION}"

echo "  Traffic is now 100% on ${NEW_REVISION}"

# ---------------------------------------------------------------------------
# Step 9: Delete the previous revision (opt-in only)
#   Project rule: keep the previous revision at 0% traffic as rollback until GG
#   confirms the new revision in the browser. Deletion only runs when explicitly
#   requested via DELETE_PREV_REVISION=1; default behavior is to keep it.
# ---------------------------------------------------------------------------
if [[ -n "${PREV_REVISION}" && "${PREV_REVISION}" != "${NEW_REVISION}" ]]; then
  if [[ "${DELETE_PREV_REVISION:-0}" == "1" ]]; then
    echo "=== Step 9: Delete previous revision ${PREV_REVISION} ==="
    # Cloud Run blocks deletion of latestCreatedRevisionName; only attempt if it differs.
    LATEST_CREATED=$(gcloud run services describe "${SERVICE}" \
      --region="${GAR_REGION}" \
      --format="value(status.latestCreatedRevisionName)")
    if [[ "${PREV_REVISION}" != "${LATEST_CREATED}" ]]; then
      gcloud run revisions delete "${PREV_REVISION}" --region="${GAR_REGION}" --quiet \
        && echo "  Deleted ${PREV_REVISION}" \
        || echo "  WARN: could not delete ${PREV_REVISION} (non-fatal)"
    else
      echo "  Skipping delete — ${PREV_REVISION} is still latestCreated (non-fatal)"
    fi
  else
    echo "=== Step 9: Keeping previous revision ${PREV_REVISION} as rollback (0% traffic) ==="
    echo "  Set DELETE_PREV_REVISION=1 to delete it. Rollback revision kept: ${PREV_REVISION}"
  fi
fi

# ---------------------------------------------------------------------------
# Step 10: Redeploy Vercel frontend so stylemaitri.vercel.app alias stays live.
#   Without this step, the alias can point to a deleted/orphaned deployment
#   after a backend redeploy that forces a project re-creation on Vercel.
# ---------------------------------------------------------------------------
echo "=== Step 8: Redeploy Vercel frontend ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="${SCRIPT_DIR}/../frontend"

if command -v vercel &>/dev/null && [[ -f "${FRONTEND_DIR}/.vercel/project.json" ]]; then
  (cd "${FRONTEND_DIR}" && vercel --prod --yes)
  echo "  Frontend redeployed and stylemaitri.vercel.app re-aliased"
else
  echo "  WARN: vercel CLI not found or frontend not linked — skipping frontend redeploy"
  echo "  Run manually: cd frontend && vercel --prod --yes"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== Deploy complete ==="
gcloud run services describe "${SERVICE}" \
  --region="${GAR_REGION}" \
  --format="table(status.traffic[0].revisionName,status.traffic[0].percent)"
echo ""
echo "QA — PROOF RULES:"
echo "  1. Test MUST hit the live Cloud Run URL below, never localhost or a local server."
echo "     A local index can differ from GCS; local results prove nothing about production."
echo "  2. Send 'black dress for women' and 'in blue' — confirm dresses / blue dresses."
LIVE_URL=$(gcloud run services describe "${SERVICE}" --region="${GAR_REGION}" --format="value(status.url)")
echo "  Live URL: ${LIVE_URL}"
echo ""
echo "Smoke test: python scripts/live_smoke_test.py"
