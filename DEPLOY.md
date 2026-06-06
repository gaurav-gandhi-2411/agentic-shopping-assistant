# Deployment Guide — Google Cloud Run

This guide deploys the Shopping Assistant **API** container to Google Cloud Run
(region: `asia-south1`, ≥2 GB RAM, min-instances=1).

The Streamlit UI (`spaces/`) is a separate deployment and is not covered here.

---

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
  (`gcloud auth login`, `gcloud auth configure-docker`)
- Docker installed locally
- A GCP project with the following APIs enabled:
  - Cloud Run (`run.googleapis.com`)
  - Artifact Registry (`artifactregistry.googleapis.com`)
- An Artifact Registry Docker repository (create once):
  ```bash
  gcloud artifacts repositories create shopping-assistant \
    --repository-format=docker \
    --location=asia-south1 \
    --description="Shopping Assistant API images"
  ```

---

## Environment variables

### Required at runtime

| Variable | Purpose |
|---|---|
| `BRAND` | Selects the brand config shipped in the image (`hm` for H&M demo, `sample_in` for India sample) |
| `GROQ_API_KEY` | LLM inference (Groq) |
| `DATABASE_URL` | Postgres connection string |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anon/public key |
| `SUPABASE_SERVICE_KEY` | Supabase service-role key (bypasses RLS for server-side ops) |

### Optional

| Variable | Purpose |
|---|---|
| `INDEX_STORE_URI` | GCS URI (`gs://bucket/path/`) from which retrieval indices are loaded at boot. When unset, the container falls back to the indices baked into the image. |
| `JWT_VERIFICATION_DISABLED` | Set to `true` to skip JWT verification (dev/demo only — never in production). |
| `SENTRY_DSN` | Sentry project DSN for error reporting. Omit to disable. |

### BRAND= and multi-brand images

All `brands/*.yaml` config files ship inside every image — no rebuild is needed
to switch brands.  Set `BRAND=<brand_slug>` and the container loads the matching
config at startup.

| `BRAND` value | Description |
|---|---|
| `hm` | H&M fashion demo catalogue |
| `sample_in` | India sample catalogue (default for local dev) |

### INDEX_STORE_URI (future A5)

When `INDEX_STORE_URI` is set (e.g. `gs://my-bucket/indices/hm/`), the container
downloads the FAISS index and catalogue parquet from GCS at startup rather than
using the baked-in files.  This allows index updates without a Docker rebuild.
Leave unset during local development; the image-embedded indices are used as the
fallback.

---

## Health probes

Cloud Run requires probes to determine container health:

| Probe | Endpoint | Expected response |
|---|---|---|
| **Startup** | `GET /healthz` | `200 {"status": "ok"}` |
| **Readiness / Liveness** | `GET /readyz` | `200 {"status": "ready", ...}` when indices are loaded, `503` until ready |

`/healthz` returns immediately with no dependency checks — suitable as the
startup probe.  `/readyz` checks the retriever, catalogue, and LLM client;
use it as the liveness probe once the container has started.

**Recommended startup probe timeout:** 120 s.  The sentence-transformers model
load takes ~30 s; the FAISS index load adds more time on cold start.

---

## Quick deploy steps

### 1 — Build the image

```bash
# Set your GCP project and image tag
PROJECT_ID=<your-gcp-project-id>
REGION=asia-south1
REPO=shopping-assistant
IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/api

docker build -t $IMAGE:latest .
```

### 2 — Push to Artifact Registry

```bash
docker push $IMAGE:latest
```

### 3 — Deploy to Cloud Run

```bash
gcloud run deploy shopping-assistant-api \
  --image $IMAGE:latest \
  --region $REGION \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 5 \
  --concurrency 1 \
  --timeout 300 \
  --set-env-vars "BRAND=hm" \
  --set-secrets "GROQ_API_KEY=groq-api-key:latest,DATABASE_URL=database-url:latest,SUPABASE_URL=supabase-url:latest,SUPABASE_ANON_KEY=supabase-anon-key:latest,SUPABASE_SERVICE_KEY=supabase-service-key:latest" \
  --port 8080 \
  --no-allow-unauthenticated
```

> **Why `--concurrency 1`?**  The API maintains in-memory session state (conversation
> history, agent loop).  Multiple concurrent requests on the same instance would
> race on that state.  Use concurrency=1 and let Cloud Run autoscale instances
> instead.

> **Secrets:** The `--set-secrets` flag maps Google Secret Manager secrets to
> environment variables.  Create each secret once with
> `gcloud secrets create <name> --data-file=-`.

### 4 — Configure health probes (Cloud Run v2 YAML — optional but recommended)

After the initial deploy, apply probe configuration via the Cloud Run service
YAML (`gcloud run services replace`) or through the GCP console:

```yaml
# Startup probe — fast check, long timeout
startupProbe:
  httpGet:
    path: /healthz
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 12   # 12 × 10 s = 120 s total window
  timeoutSeconds: 5

# Liveness probe — deeper check after startup
livenessProbe:
  httpGet:
    path: /readyz
  periodSeconds: 30
  failureThreshold: 3
  timeoutSeconds: 10
```

---

## Smoke tests

Replace `<SERVICE_URL>` with the URL printed by `gcloud run deploy`.

```bash
SERVICE_URL=$(gcloud run services describe shopping-assistant-api \
  --region asia-south1 --format "value(status.url)")

# Startup probe
curl -s $SERVICE_URL/healthz
# → {"status":"ok"}

# Readiness probe (200 when indices loaded, 503 while loading)
curl -s $SERVICE_URL/readyz
# → {"status":"ready","checks":{"retriever":"ok (20,000 vectors)","catalogue":"ok (20,000 items)","llm":"ok"},"auth_enabled":true}
```

WebSocket smoke test with [wscat](https://github.com/websockets/wscat):

```bash
wscat -c wss://<service-url>/chat/stream
> {"type":"user_message","message":"show me blue jackets","conversation_id":null}
< {"type":"session","conversation_id":"<uuid>"}
< {"type":"routing","decision":{"action":"search","query":"blue jackets"}}
< {"type":"tool_start","tool":"search"}
< {"type":"items","items":[...]}
< {"type":"token","text":"Here "}
...
< {"type":"done","final_state":{...}}
```

---

## Per-brand deployments

The same Docker image handles all brands.  Deploy one Cloud Run service per
brand, differing only in `BRAND=` and (when A5 is live) `INDEX_STORE_URI=`.

| Service name | `BRAND` | `INDEX_STORE_URI` |
|---|---|---|
| `shopping-assistant-hm` | `hm` | `gs://my-bucket/indices/hm/` |
| `shopping-assistant-sample-in` | `sample_in` | `gs://my-bucket/indices/sample_in/` |

```bash
# Deploy H&M brand service (reuse the same image tag)
gcloud run deploy shopping-assistant-hm \
  --image $IMAGE:latest \
  --region $REGION \
  --memory 2Gi \
  --min-instances 1 \
  --concurrency 1 \
  --set-env-vars "BRAND=hm,INDEX_STORE_URI=gs://my-bucket/indices/hm/" \
  --set-secrets "GROQ_API_KEY=groq-api-key:latest,..." \
  --no-allow-unauthenticated

# Deploy India sample brand service
gcloud run deploy shopping-assistant-sample-in \
  --image $IMAGE:latest \
  --region $REGION \
  --memory 2Gi \
  --min-instances 1 \
  --concurrency 1 \
  --set-env-vars "BRAND=sample_in,INDEX_STORE_URI=gs://my-bucket/indices/sample_in/" \
  --set-secrets "GROQ_API_KEY=groq-api-key:latest,..." \
  --no-allow-unauthenticated
```

---

## Subsequent deploys

```bash
docker build -t $IMAGE:latest . && docker push $IMAGE:latest

gcloud run deploy shopping-assistant-hm \
  --image $IMAGE:latest \
  --region $REGION
# Cloud Run performs a zero-downtime rollout; old revision handles traffic
# until the new revision passes health probes.
```

---

## Scaling notes

- **Memory:** 2 GB is the minimum for sentence-transformers + FAISS. If the
  container OOMs on startup, increase to 4 GB: `--memory 4Gi`.
- **CPU:** 1 vCPU is sufficient for sequential (concurrency=1) requests. For
  lower cold-start latency, set `--cpu-boost` to allocate extra CPU during
  startup.
- **Min instances:** Keep at 1 to avoid cold starts in demo/client environments.
  Set to 0 only if cost matters more than response latency.
