#!/usr/bin/env bash
# deploy_demo.sh — build + push Docker image, then deploy 3 brand services to Cloud Run.
# Usage: fill in the variables below, then: bash scripts/deploy_demo.sh
set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURATION — fill these in before running
# ---------------------------------------------------------------------------
GCP_PROJECT="aetherart-497918"
GAR_REGION="asia-south1"           # Artifact Registry region (also Cloud Run region)
GAR_REPO="shopping-assistant"       # Artifact Registry repository name
IMAGE_NAME="asa-api"
IMAGE_TAG="$(git rev-parse --short HEAD)"
IMAGE="${GAR_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

GCS_BUCKET="asa-demo-indices-aetherart"

# Secrets — read from env so they are never baked into this file.
# Export them before running: export GROQ_API_KEY=... DEMO_JWT_SECRET=... DATABASE_URL=...
GROQ_API_KEY="${GROQ_API_KEY:-}"
DEMO_JWT_SECRET="${DEMO_JWT_SECRET:-}"
DATABASE_URL="${DATABASE_URL:-}"
SUPABASE_URL="https://gtvbficptamgtfyyitth.supabase.co"
VERCEL_URL="${VERCEL_URL:-}"       # Set after first Vercel deploy (or export before running)
SENTRY_DSN="${SENTRY_DSN:-}"      # Optional

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$GROQ_API_KEY" || -z "$DEMO_JWT_SECRET" || -z "$DATABASE_URL" || -z "$SUPABASE_URL" ]]; then
  echo "ERROR: fill in GROQ_API_KEY, DEMO_JWT_SECRET, DATABASE_URL, and SUPABASE_URL above"
  exit 1
fi

CORS_ORIGINS="${VERCEL_URL},http://localhost:3000"

echo "=== Step 1: Configure Docker for Artifact Registry ==="
gcloud auth configure-docker "${GAR_REGION}-docker.pkg.dev" --quiet

echo "=== Step 2: Build Docker image ==="
docker build -t "${IMAGE}" .

echo "=== Step 3: Push to Artifact Registry ==="
docker push "${IMAGE}"

echo "=== Step 4: Run database migration ==="
echo "  Run: DATABASE_URL='${DATABASE_URL}' alembic upgrade head"
echo "  (skipping automatic migration — run it manually if not yet done)"

echo "=== Step 5: Upload indices to GCS ==="
for brand in snitch myntra flipkart; do
  echo "  Uploading ${brand} index..."
  gcloud storage cp -r "data/processed/${brand}" "gs://${GCS_BUCKET}/${brand}/"
done

# Shared env vars (excluding BRAND and INDEX_STORE_URI which are per-service)
COMMON_ENV="DEMO_MODE=true,LLM_PROVIDER=groq,GROQ_API_KEY=${GROQ_API_KEY},DEMO_JWT_SECRET=${DEMO_JWT_SECRET},DATABASE_URL=${DATABASE_URL},SUPABASE_URL=${SUPABASE_URL},CORS_ORIGINS=${CORS_ORIGINS},SENTRY_DSN=${SENTRY_DSN}"

# Common Cloud Run flags (no --set-env-vars here — each deploy call builds a single merged string)
CR_FLAGS=(
  --image="${IMAGE}"
  --region="${GAR_REGION}"
  --platform=managed
  --allow-unauthenticated
  --memory=2Gi
  --cpu=1
  --concurrency=4
  --timeout=300
)

echo "=== Step 6: Deploy snitch (min-instances=1, flagship) ==="
gcloud run deploy asa-snitch \
  "${CR_FLAGS[@]}" \
  --min-instances=1 \
  --set-env-vars="BRAND=snitch,INDEX_STORE_URI=gs://${GCS_BUCKET}/snitch/,${COMMON_ENV}"

echo "=== Step 7: Deploy myntra (min-instances=0, scale-to-zero) ==="
gcloud run deploy asa-myntra \
  "${CR_FLAGS[@]}" \
  --min-instances=0 \
  --set-env-vars="BRAND=myntra,INDEX_STORE_URI=gs://${GCS_BUCKET}/myntra/,${COMMON_ENV}"

echo "=== Step 8: Deploy flipkart (min-instances=0, scale-to-zero) ==="
gcloud run deploy asa-flipkart \
  "${CR_FLAGS[@]}" \
  --min-instances=0 \
  --set-env-vars="BRAND=flipkart,INDEX_STORE_URI=gs://${GCS_BUCKET}/flipkart/,${COMMON_ENV}"

echo ""
echo "=== Deployment complete ==="
echo "Get service URLs:"
echo "  gcloud run services describe asa-snitch   --region=${GAR_REGION} --format='value(status.url)'"
echo "  gcloud run services describe asa-myntra   --region=${GAR_REGION} --format='value(status.url)'"
echo "  gcloud run services describe asa-flipkart --region=${GAR_REGION} --format='value(status.url)'"
echo ""
echo "Next: deploy Vercel frontend and set NEXT_PUBLIC_*_BACKEND_URL env vars to the URLs above."
