# Demo Deployment Runbook

End-to-end steps to go from code to a live public URL.

---

## Prerequisites

All of the following must exist before starting:

- GCP project with billing enabled and these APIs active: Cloud Run, Artifact Registry, Cloud Storage
- Artifact Registry repository `shopping-assistant` in `asia-south1`
- GCS bucket for indices (e.g. `asa-demo-indices`)
- Supabase project created; database migration not yet run (step 1 covers this)
- Vercel account connected to the GitHub repo
- GitHub repository secrets configured (see table below)

---

## GitHub Secrets

Set these at GitHub → Settings → Secrets and variables → Actions → Repository secrets.

| Secret | Description |
|---|---|
| `GCP_PROJECT_ID` | GCP project ID (e.g. `my-project-123`) |
| `WIF_PROVIDER` | WIF provider resource name: `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL/providers/PROVIDER` |
| `WIF_SERVICE_ACCOUNT` | Email of the deploy service account (e.g. `deploy-sa@my-project-123.iam.gserviceaccount.com`) |
| `GROQ_API_KEY` | Groq API key |
| `DEMO_JWT_SECRET` | Random 32-byte hex string; generate with: `openssl rand -hex 32` |
| `DATABASE_URL` | Supabase Postgres connection string (`postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres`) |
| `SUPABASE_URL` | `https://<ref>.supabase.co` |
| `GCS_BUCKET` | GCS bucket name used for index storage (no `gs://` prefix) |
| `DEMO_CORS_ORIGINS` | Comma-separated allowed origins — set after step 8 when the Vercel URL is known; placeholder: `http://localhost:3000` |

---

## Step-by-step runbook

### 1. Run database migration (one-time)

```bash
DATABASE_URL=<url> alembic upgrade head
```

Adds `demo_rate_limits` and `demo_daily_stats` tables. Safe to re-run; Alembic is idempotent.

---

### 2. Upload brand indices to GCS (one-time per brand)

Build indices locally first if not already built:

```bash
python scripts/01_build_retrieval.py --brand snitch
python scripts/01_build_retrieval.py --brand myntra
python scripts/01_build_retrieval.py --brand flipkart
```

Then upload:

```bash
GCS_BUCKET="your-gcs-bucket-name"

gcloud storage cp -r data/processed/snitch/    gs://$GCS_BUCKET/snitch/
gcloud storage cp -r data/processed/myntra/    gs://$GCS_BUCKET/myntra/
gcloud storage cp -r data/processed/flipkart/  gs://$GCS_BUCKET/flipkart/
```

---

### 3. Build, push, and deploy to Cloud Run

**Option A — local script** (fill in variables first):

```bash
bash scripts/deploy_demo.sh
```

**Option B — GitHub Actions** (preferred for CI-tracked deploys):

```bash
gh workflow run deploy-demo.yml --field brands="snitch myntra flipkart"
```

Or via the GitHub UI: Actions → "Deploy Demo (Cloud Run + Vercel)" → Run workflow.

---

### 4. Get Cloud Run service URLs

```bash
for svc in snitch myntra flipkart; do
  gcloud run services describe asa-$svc \
    --region asia-south1 \
    --format='value(status.url)'
done
```

Note the three URLs — you will need them in step 6.

---

### 5. Smoke-test each backend

```bash
# Replace with actual URLs from step 4
SNITCH_URL="https://asa-snitch-xxx-uc.a.run.app"

curl -s ${SNITCH_URL}/readyz | jq .
# Expected: {"status":"ready",...}

curl -s -X POST ${SNITCH_URL}/demo/session | jq .
# Expected: {"session_token":"...","expires_in":...}
```

Both calls must return HTTP 200 before proceeding.

---

### 6. Deploy Vercel frontend

```bash
cd frontend
vercel --prod
```

Set these environment variables in the Vercel project settings (dashboard or CLI) before
triggering the production build. Use the URLs from step 4.

```bash
vercel env add NEXT_PUBLIC_SNITCH_BACKEND_URL   production   # URL from step 4
vercel env add NEXT_PUBLIC_MYNTRA_BACKEND_URL   production   # URL from step 4
vercel env add NEXT_PUBLIC_FLIPKART_BACKEND_URL production   # URL from step 4
vercel env add NEXT_PUBLIC_SUPABASE_URL         production   # https://<ref>.supabase.co
vercel env add NEXT_PUBLIC_SUPABASE_ANON_KEY    production   # Supabase anon key
vercel env add NEXT_PUBLIC_BACKEND_URL          production   # same as SNITCH URL (auth fallback)
```

After adding env vars, redeploy:

```bash
vercel --prod
```

---

### 7. Update CORS_ORIGINS on Cloud Run services

After Vercel assigns a permanent URL (e.g. `https://asa-demo.vercel.app`):

```bash
VERCEL_URL="https://asa-demo.vercel.app"

for svc in snitch myntra flipkart; do
  gcloud run services update asa-$svc \
    --region asia-south1 \
    --update-env-vars CORS_ORIGINS="${VERCEL_URL},http://localhost:3000"
done
```

Also update the `DEMO_CORS_ORIGINS` GitHub secret to the same value so future CI deploys
carry the correct origins without a manual patch.

---

### 8. Smoke-test the public URL

Open `https://asa-demo.vercel.app/demo` in an incognito browser window. Run through this
checklist:

- [ ] Pick the **Snitch** brand from the brand selector
- [ ] Send: "Show me party shirts under ₹1500"
- [ ] Products appear with images and INR prices
- [ ] "Buy on Snitch" link opens a real product page on snitch.in
- [ ] Network tab shows `X-RateLimit-Remaining` and `X-RateLimit-Limit` response headers
- [ ] Demo banner is visible at the top of the chat interface

---

### 9. Done

Hand the URL to recruiters. The demo is live.

---

## Rollback

To roll back a Cloud Run service to a previous image:

```bash
gcloud run services update asa-<brand> \
  --region asia-south1 \
  --image=<previous-image-uri>
```

Find previous image URIs in Artifact Registry:

```bash
gcloud artifacts docker images list \
  asia-south1-docker.pkg.dev/<GCP_PROJECT>/shopping-assistant/asa-api \
  --sort-by=~CREATE_TIME \
  --limit=10
```
