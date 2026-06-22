# Deployment Guide — Cloud Run + Vercel

## Architecture

Three Cloud Run services (one per brand: `asa-snitch`, `asa-myntra`, `asa-flipkart`) run in
`asia-south1` from a single Docker image, differentiated by the `BRAND` env var. All services
scale-to-zero (min-instances=0). Retrieval indices are loaded
from a shared GCS bucket at container startup, so index refreshes don't require a Docker rebuild.
The Next.js frontend is deployed to Vercel and talks to each backend service directly.

---

## Prerequisites

- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- Docker installed locally
- Vercel CLI installed (`npm i -g vercel`)
- A GCP project with the following APIs enabled:
  - Cloud Run (`run.googleapis.com`)
  - Artifact Registry (`artifactregistry.googleapis.com`)
  - Cloud Storage (`storage.googleapis.com`)
- An Artifact Registry Docker repository (create once):
  ```bash
  gcloud artifacts repositories create shopping-assistant \
    --repository-format=docker \
    --location=asia-south1 \
    --description="Shopping Assistant API images"
  ```
- A GCS bucket for retrieval indices (create once):
  ```bash
  gcloud storage buckets create gs://$GCS_BUCKET --location=asia-south1
  ```
- A Supabase project with the demo schema migration applied (see "Database migration" below)

---

## Environment variables

### Backend (set on each Cloud Run service)

| Variable | Required | Value / Notes |
|---|---|---|
| `BRAND` | Yes | `snitch`, `myntra`, or `flipkart` — selects brand config at startup |
| `DEMO_MODE` | Yes | `true` — enables demo rate-limiting and session endpoints |
| `GROQ_API_KEY` | Yes | Groq inference API key |
| `DEMO_JWT_SECRET` | Yes | Random 32-byte hex secret for demo session tokens |
| `DATABASE_URL` | Yes | Supabase Postgres connection string (`postgresql://...`) |
| `SUPABASE_URL` | Yes | `https://<ref>.supabase.co` |
| `CORS_ORIGINS` | Yes | Single comma-free allowed origin (the Vercel URL). gcloud `--set-env-vars` treats every comma as a separator — a multi-origin value silently corrupts the env block. Use `https://asa-stylist.vercel.app`. |
| `INDEX_STORE_URI` | Yes | `gs://<bucket>/<brand>/` — GCS path to FAISS index + catalogue parquet |
| `LLM_PROVIDER` | Yes | `groq` |
| `SENTRY_DSN` | No | Sentry project DSN; omit to disable error reporting |

### Vercel (set in Vercel project settings or via `vercel env add`)

| Variable | Value |
|---|---|
| `NEXT_PUBLIC_SNITCH_BACKEND_URL` | Cloud Run URL for `asa-snitch` |
| `NEXT_PUBLIC_MYNTRA_BACKEND_URL` | Cloud Run URL for `asa-myntra` |
| `NEXT_PUBLIC_FLIPKART_BACKEND_URL` | Cloud Run URL for `asa-flipkart` |
| `NEXT_PUBLIC_SUPABASE_URL` | Your Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase anon/public key |
| `NEXT_PUBLIC_BACKEND_URL` | Same as `NEXT_PUBLIC_SNITCH_BACKEND_URL` (fallback for authenticated path) |

---

## Database migration

Run once before first deploy (or after any schema change):

```bash
DATABASE_URL=<url> alembic upgrade head
```

This creates the `demo_rate_limits` and `demo_daily_stats` tables required by the demo
rate-limiting middleware.

---

## Build & push Docker image

```bash
GCP_PROJECT="your-gcp-project-id"
GAR_REGION="asia-south1"
GAR_REPO="shopping-assistant"
IMAGE_NAME="asa-api"
IMAGE_TAG="$(git rev-parse --short HEAD)"
IMAGE="${GAR_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GAR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

# Authenticate Docker to Artifact Registry
gcloud auth configure-docker "${GAR_REGION}-docker.pkg.dev" --quiet

# Build
docker build -t "${IMAGE}" .

# Push
docker push "${IMAGE}"
```

---

## Deploy Cloud Run services

All three services use the same image and scale to zero (min-instances=0). Cold starts show
the "warming up…" spinner in the frontend.

```bash
# CORS_ORIGINS must be a single comma-free origin. gcloud --set-env-vars treats every comma
# as an env-var separator; adding ",http://localhost:3000" silently corrupts the env block.
CORS_ORIGINS="https://asa-stylist.vercel.app"
GCS_BUCKET="your-gcs-bucket-name"

# Shared flags
COMMON="--image=${IMAGE} --region=${GAR_REGION} --platform=managed --allow-unauthenticated \
  --memory=2Gi --cpu=1 --concurrency=4 --timeout=300"

# snitch
gcloud run deploy asa-snitch \
  ${COMMON} \
  --min-instances=0 \
  --set-env-vars="BRAND=snitch,DEMO_MODE=true,LLM_PROVIDER=groq,\
GROQ_API_KEY=${GROQ_API_KEY},DEMO_JWT_SECRET=${DEMO_JWT_SECRET},\
DATABASE_URL=${DATABASE_URL},SUPABASE_URL=${SUPABASE_URL},\
CORS_ORIGINS=${CORS_ORIGINS},INDEX_STORE_URI=gs://${GCS_BUCKET}/snitch/"

# myntra — scale-to-zero
gcloud run deploy asa-myntra \
  ${COMMON} \
  --min-instances=0 \
  --set-env-vars="BRAND=myntra,DEMO_MODE=true,LLM_PROVIDER=groq,\
GROQ_API_KEY=${GROQ_API_KEY},DEMO_JWT_SECRET=${DEMO_JWT_SECRET},\
DATABASE_URL=${DATABASE_URL},SUPABASE_URL=${SUPABASE_URL},\
CORS_ORIGINS=${CORS_ORIGINS},INDEX_STORE_URI=gs://${GCS_BUCKET}/myntra/"

# flipkart — scale-to-zero
gcloud run deploy asa-flipkart \
  ${COMMON} \
  --min-instances=0 \
  --set-env-vars="BRAND=flipkart,DEMO_MODE=true,LLM_PROVIDER=groq,\
GROQ_API_KEY=${GROQ_API_KEY},DEMO_JWT_SECRET=${DEMO_JWT_SECRET},\
DATABASE_URL=${DATABASE_URL},SUPABASE_URL=${SUPABASE_URL},\
CORS_ORIGINS=${CORS_ORIGINS},INDEX_STORE_URI=gs://${GCS_BUCKET}/flipkart/"
```

---

## Upload indices to GCS

**CRITICAL: always upload after rebuilding the index.** The Docker image does NOT bundle index
files — Cloud Run downloads them from GCS at container startup. If you test locally with a
rebuilt index and skip the GCS upload, production continues to serve the old (stale) index
and local results prove nothing about what the live service returns.

**Never use `--no-clobber`** on an index upload. Stale files already exist in GCS; `--no-clobber`
silently skips them and leaves the old index live.

### Unified index (current deployment — `asa-stylist-api`)

Run after every `scripts/build_unified_index.py` run, or use `scripts/deploy_unified.sh`
which includes this step automatically:

```bash
# Main index files
gcloud storage cp \
  data/processed/unified/catalogue.parquet \
  data/processed/unified/dense.faiss \
  data/processed/unified/dense_article_ids.npy \
  data/processed/unified/bm25.pkl \
  data/processed/unified/bm25_article_ids.npy \
  gs://asa-demo-indices/unified/unified/

# CLIP vectors (image search)
gcloud storage cp \
  data/processed/clip/unified/clip.faiss \
  data/processed/clip/unified/clip_article_ids.npy \
  gs://asa-demo-indices/unified/clip/unified/

# Verify timestamps show today
gcloud storage ls --long gs://asa-demo-indices/unified/unified/
```

Then force a new Cloud Run revision so it cold-starts and re-downloads the fresh index:

```bash
gcloud run deploy asa-stylist-api \
  --region=asia-south1 \
  --image=$(gcloud run services describe asa-stylist-api --region=asia-south1 --format="value(spec.template.spec.containers[0].image)")
```

### Per-brand indices (legacy — `asa-snitch`, `asa-myntra`, `asa-flipkart`)

```bash
GCS_BUCKET="your-gcs-bucket-name"

gcloud storage cp -r data/processed/snitch    gs://${GCS_BUCKET}/snitch/
gcloud storage cp -r data/processed/myntra    gs://${GCS_BUCKET}/myntra/
gcloud storage cp -r data/processed/flipkart  gs://${GCS_BUCKET}/flipkart/
```

### QA rule — proof must come from the live Cloud Run URL

After any GCS upload + service restart, verify by hitting the **deployed Cloud Run URL**, not
`localhost` and not a local server. A local index can diverge from GCS; local results prove
nothing about what production returns.

```bash
# Get a demo session and send a test query to the live service
TOKEN=$(curl -s -X POST https://asa-stylist-api-<hash>.a.run.app/demo/session \
  -H "Content-Length: 0" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_token'])")

curl -s -X POST https://asa-stylist-api-<hash>.a.run.app/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"message":"black dress for women","conversation_id":null}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); [print(i['prod_name'],'-',i['product_type']) for i in d['items']]"
```

Expected: items whose `product_type` is `"dress"` and whose names contain "Dress"/"Gown"/etc.
If shorts, sweatshirts, or jackets appear, the index is stale — re-upload and restart.

---

## Deploy Vercel frontend

```bash
cd frontend
vercel --prod
```

Set the six `NEXT_PUBLIC_*` environment variables (see table above) in the Vercel project
settings dashboard or via CLI before the final production deploy:

```bash
vercel env add NEXT_PUBLIC_SNITCH_BACKEND_URL   production
vercel env add NEXT_PUBLIC_MYNTRA_BACKEND_URL   production
vercel env add NEXT_PUBLIC_FLIPKART_BACKEND_URL production
vercel env add NEXT_PUBLIC_SUPABASE_URL         production
vercel env add NEXT_PUBLIC_SUPABASE_ANON_KEY    production
vercel env add NEXT_PUBLIC_BACKEND_URL          production
```

---

## CORS configuration

Each Cloud Run service's `CORS_ORIGINS` env var must be set to the canonical Vercel URL:
`https://asa-stylist.vercel.app`. **Keep the value comma-free** — gcloud's `--set-env-vars`
treats every comma as an env-var separator, so a multi-origin string (e.g.
`https://asa-stylist.vercel.app,http://localhost:3000`) silently splits into two env vars
and corrupts the env block.

```bash
VERCEL_URL="https://asa-stylist.vercel.app"

for svc in snitch myntra flipkart; do
  gcloud run services update asa-${svc} \
    --region=asia-south1 \
    --update-env-vars="CORS_ORIGINS=${VERCEL_URL}"
done
```

Also update the `DEMO_CORS_ORIGINS` GitHub secret to the same single-origin value so future
CI deploys use the correct origin (see "CI/CD" below).

---

## CI/CD

The workflow at `.github/workflows/deploy-demo.yml` automates the full build-push-deploy
cycle. It triggers on manual dispatch (`workflow_dispatch`) and accepts optional inputs:
`brands` (space-separated, default `snitch myntra flipkart`) and `image_tag`.

Authentication uses Workload Identity Federation (WIF) — no service account key JSON is
stored as a secret. Required GitHub secrets:

| Secret | Description |
|---|---|
| `GCP_PROJECT_ID` | GCP project ID |
| `WIF_PROVIDER` | WIF provider resource name |
| `WIF_SERVICE_ACCOUNT` | Deploy service account email |
| `GROQ_API_KEY` | Groq API key |
| `DEMO_JWT_SECRET` | Demo session token secret |
| `DATABASE_URL` | Supabase Postgres connection string |
| `SUPABASE_URL` | Supabase project URL |
| `GCS_BUCKET` | GCS bucket name (index storage) |
| `DEMO_CORS_ORIGINS` | Single comma-free allowed origin (`https://asa-stylist.vercel.app`); see CORS configuration note above |

To trigger a deploy from the CLI:

```bash
gh workflow run deploy-demo.yml --field brands="snitch"
```
