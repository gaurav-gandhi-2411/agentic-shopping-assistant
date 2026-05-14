# Fly.io → Google Cloud Run Migration

**Date:** 2026-05-15

## What was removed

| File / location | Change |
|---|---|
| `fly.toml` | Deleted entirely |
| `api/rate_limit.py` docstring | "Fly.io" replaced with "Cloud Run, Docker" |
| `PROJECT_AUDIT.md` — architecture diagram | "Fly.io → force_https=true" → "Cloud Run → HTTPS enforced at load balancer" |
| `PROJECT_AUDIT.md` — BUG-1 heading | Reworded to remove Fly machine/memory spec |
| `PROJECT_AUDIT.md` — WEAK-7, WEAK-8, WEAK-10 | "Fly.io" references replaced with platform-agnostic equivalents |
| `PROJECT_AUDIT.md` — Architecture Assessment | Fly machine reference replaced with Cloud Run |
| `PROJECT_AUDIT.md` — Appendix B | Status updated to RESOLVED; Fly deployment history removed |
| `PRODUCTION_PLAN.md` — Frontend section | "Fly.io (no CDN)" → "container" |
| `PRODUCTION_PLAN.md` — Observability P2 | "fly dashboard" → Cloud Console Metrics |
| `PRODUCTION_PLAN.md` — Infra & Deployment, Backend | Entire Fly.io machine spec replaced with Cloud Run |
| `PRODUCTION_PLAN.md` — Frontend section | "CORS with the Fly.io API" → "CORS with the Cloud Run API" |
| `PRODUCTION_PLAN.md` — CI/CD | "push to Fly.io registry" → "push to Artifact Registry"; `FLY_API_TOKEN` → `GCP_SA_KEY`; health check wording updated |
| `PRODUCTION_PLAN.md` — Cost tables (3×) | "Fly + Vercel" infra column replaced with "Cloud Run + Vercel" |
| `PRODUCTION_PLAN.md` — P0 roadmap | "Upsize Fly.io machine" → "Confirm Cloud Run ≥ 2 GB" |
| `PRODUCTION_PLAN.md` — P1 roadmap | "Fly volume / R2" → "GCS or R2" |
| `PRODUCTION_PLAN.md` — Explicit Assumptions | "Fly `bom`" → "Cloud Run `asia-south1`" |

## Why

Fly.io requires a credit card on file before allowing any deployment, even to free-tier resources. The project has existing Google Cloud Platform credits. Cloud Run is a better fit:

- No credit card required to deploy (covered by GCP credits)
- Scale-to-zero billing — cost is zero when idle
- 2 GB memory is straightforward to configure
- `asia-south1` (Mumbai) region matches the original `bom` intent
- Docker image deploys directly from Artifact Registry with no new toolchain

## Where deployment now lives

Cloud Run setup will be done in a separate PR. The Dockerfile and application code are unchanged — the same `EXPOSE 8080` / uvicorn entrypoint works on Cloud Run without modification.

Deployment state lives outside this repo (Cloud Run service config, Artifact Registry, Secret Manager). See the next PR for the `cloudbuild.yaml` / `gcloud run deploy` command and secrets list.
