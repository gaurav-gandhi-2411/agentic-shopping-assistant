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
| `CORS_ORIGINS` | Yes | Single comma-free allowed origin (the Vercel URL). gcloud `--set-env-vars` treats every comma as a separator — a multi-origin value silently corrupts the env block. Use `https://stylemaitri.vercel.app`. |
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

## Deploy backend (unified service)

`pwsh scripts/deploy_backend.ps1` is THE deploy path for the backend (`asa-stylist-api`,
the current unified single-service deployment). There is no CI deploy path — the GitHub
Actions workflow that used to cover this (`.github/workflows/deploy-demo.yml`) was removed
2026-07-09 because it never had its required secrets (`WIF_PROVIDER`,
`WIF_SERVICE_ACCOUNT`, `GCP_PROJECT_ID`, `GROQ_API_KEY`, `DEMO_JWT_SECRET`, `DATABASE_URL`,
`SUPABASE_URL`, `GCS_BUCKET`) configured on the repo and never ran successfully. Every real
deploy has always happened locally; the script codifies that path instead of a workflow
that only pretended to cover it.

```powershell
pwsh scripts/deploy_backend.ps1
```

Defaults: project `iconic-reactor-496423-m4`, region `asia-south1`, service
`asa-stylist-api`, Artifact Registry repo `shopping-assistant`, image `asa-api`, tag
`wave-deploy-<git short sha>`. Pass `-SkipBuild -Tag <tag>` to deploy an already-pushed
image without rebuilding.

The script builds, pushes, deploys, and moves traffic in one run, verifying two gotchas
that have previously produced a "deployed but not live" or "config silently reverted"
outcome:

1. **Traffic pinning.** `gcloud run deploy` creates a new revision but does not move
   traffic onto it if traffic is pinned to a named revision (this service's traffic
   history has done that before) — the deploy succeeds while the old revision keeps
   serving 100% of requests, silently. The script always runs
   `gcloud run services update-traffic asa-stylist-api --to-latest` after deploying, then
   verifies `latestReadyRevisionName` equals the revision holding 100% traffic before
   declaring success. `--to-latest` reassigns only the untagged 100%-traffic slot; it does
   not remove or reassign existing revision tags (this service intentionally keeps a
   0%-traffic tagged revision, tag `w1test` on `asa-stylist-api-00056-fum`, as a rollback
   reference).
2. **Env-block replacement.** `gcloud run deploy --set-env-vars=...` REPLACES the entire
   env block rather than merging into it — it has previously wiped manually-raised vars
   (`DEMO_PER_IP_HOUR_LIMIT`, `DEMO_DAILY_REQUEST_CAP`) that existed only on the running
   revision. The script deploys with `--image` only, which inherits the previous
   revision's env block untouched. **Never redeploy with `--set-env-vars`.** To change env
   vars, run this as its own separate step:
   ```bash
   gcloud run services update asa-stylist-api --region=asia-south1 \
     --update-env-vars="SOME_VAR=value"
   ```

If a deploy needs to be reverted, the script prints the exact rollback command at the end
of every run:

```bash
gcloud run services update-traffic asa-stylist-api --to-revisions <anchor-revision>=100 \
  --region=asia-south1 --project=iconic-reactor-496423-m4
```

A future CI path (if one is built) should authenticate via Workload Identity Federation
(as the removed workflow attempted) and pass secrets to Cloud Run via `--set-secrets`
against Secret Manager references (see `scripts/deploy_unified.sh`'s `SECRETS` variable
for the reference names already in use), not via plain env-var GitHub secrets — this
avoids both the CI-secrets bootstrap problem that stalled the old workflow and the
env-block-replacement gotcha above.

### LLM provider secrets and switching providers

All LLM provider API keys on `asa-stylist-api` are Secret Manager references, never plain
env vars: `GROQ_API_KEY` → `asa-groq-api-key`, `OPENROUTER_API_KEY` → `asa-openrouter-api-key`
(added 2026-07-10 — Groq's free-tier daily token quota has been exhausted by heavy testing
more than once).

**As of 2026-07-31, `GroqClient` falls over to a 3-tier chain automatically** on a TPD
(tokens-per-day) 429: Groq → OpenRouter → Gemini — see `GroqClient._iter_fallback_tiers()`
in `src/llm/client.py`. No manual `LLM_PROVIDER` env-var swap is needed for a quota outage;
`LLM_PROVIDER=groq` in production now carries an implicit 2-tier fallback for both `chat()`
and `chat_stream()`. Each tier is tried in order; a tier that raises OR returns falsy/empty
content (OpenRouter's free model is a reasoning model that can burn its whole token budget
on hidden reasoning and return `None`; Gemini can return empty output on safety-filtered
responses) is treated as failed and the chain advances to the next tier — a wired-but-silent
empty response is not treated as success. The manual switch below is still the right tool
for a *deliberate* provider change (e.g. cost or model-quality reasons), and the whole chain
falls back to the legacy wait-and-retry-on-Groq behaviour if literally no tier can construct
(all keys unset/invalid) rather than hanging on a broken fallback.

Every fallthrough is logged as a structured event — `{"event": "llm_fallback_served",
"from_provider": "groq", "to_provider": "<openrouter|gemini>", "reason": "tpd_exhausted"}` —
so degraded-mode traffic is visible in Cloud Logging (`jsonPayload.msg:"llm_fallback_served"`)
instead of only discoverable via user complaints. Full chain exhaustion logs
`llm_fallback_exhausted` with the list of tiers attempted. Every `llm_call` log line (from
any of the four client classes) also now carries an explicit `"provider"` field.

**Resolved 2026-07-31 (was "Known gap 2026-07-30"):** `asa-openrouter-api-key` was rotated
to a new version (v2) with a valid key, live-verified against `openrouter.ai/api/v1/auth/key`
and a real `chat/completions` call. The *model* it was pointed at also needed fixing
separately — `google/gemma-3-27b-it:free` was deprecated by OpenRouter entirely (`404`, not
a key problem) — swapped to `openai/gpt-oss-20b:free`. `GEMINI_API_KEY` (secret
`asa-gemini-api-key`) was added as the third tier the same day, after finding the
previously-configured `gemini-2.0-flash-lite` also had zero free-tier quota (`limit: 0`,
confirmed via the API's own error) — swapped to `gemini-2.5-flash`, live-verified working.

**To switch providers deliberately** (e.g. for cost or model-quality reasons, not quota outages):

```bash
gcloud run services update asa-stylist-api --region=asia-south1 \
  --project=iconic-reactor-496423-m4 \
  --update-env-vars="LLM_PROVIDER=openrouter"
# ... and back, once Groq's quota resets:
gcloud run services update asa-stylist-api --region=asia-south1 \
  --project=iconic-reactor-496423-m4 \
  --update-env-vars="LLM_PROVIDER=groq"
```

**Critical gotcha, learned the hard way (2026-07-10): `--set-secrets` REPLACES the entire
secrets block, exactly like `--set-env-vars` replaces the entire env block** — running it
with only one secret named wiped `GROQ_API_KEY`/`DEMO_JWT_SECRET`/`DATABASE_URL`/
`SUPABASE_URL` from the revision, which then crashed on startup (`ValueError: GROQ_API_KEY
environment variable is not set`) because the container couldn't construct its LLM client.
Worse: `gcloud run services update`'s next incremental call bases its diff on the **last
requested revision template, not the last successfully serving one** — so a second
`--update-secrets` call issued on top of the broken state still didn't recover the missing
secrets, because it was merging into an already-incomplete base. Recovery required reading
every secret's exact `name:key` pair off the **actually-serving** revision (`gcloud run
revisions describe <serving-revision> --format=json | grep -A2 secretKeyRef`, not the
latest/failed one) and re-specifying the full set explicitly in one `--update-secrets` call.
**Always use `--update-secrets`, never `--set-secrets`, for anything short of an intentional
full reset** — and if a revision ever fails to start, immediately confirm which revision is
actually serving traffic (`status.traffic`, not `status.latestReadyRevisionName`) before
touching anything further; Cloud Run does not cut traffic to a revision that fails its
startup health check, so the live service stays safe, but the *next* incremental update's
base can still be silently wrong.

### Free-tier quota ceiling — 3-tier chain (measured, 2026-07-30/31, Gemini 2026-08-06)

Real production traffic on `asa-stylist-api` averages **~2,922 tokens/conversation** and
**~2.07 LLM calls/conversation** (source: 31 real `llm_call` log events across 15 real
conversations pulled from Cloud Run logs, 2026-07-30/31 — this is Groq-tier traffic; the
fallback tiers weren't live yet when this sample was taken).

**Tier 1 — Groq** (`llama-3.1-8b-instant`, source: console.groq.com/docs/rate-limits):
RPM 30, RPD 14,400, TPM 6,000, TPD 500,000. Dividing the daily token budget by the measured
per-conversation cost: **500,000 TPD / ~2,922 tokens ≈ ~170 real user conversations/day**
before this tier exhausts. TPM=6,000/min is a tighter bottleneck than it looks — roughly
**2 conversations/minute** of burst capacity before requests start hitting TPM 429s (short
1s/3s backoff, not a hard failure, but adds latency under bursty concurrent traffic).

**Tier 2 — OpenRouter** (`openai/gpt-oss-20b:free`, source:
openrouter.ai/docs/api-reference/limits): a hard **50 requests/day** cap (not token-based),
or **1,000/day** if the account has ever purchased ≥$10 in credits (this account hasn't, per
`is_free_tier: true` confirmed live 2026-07-31) — adds a flat **+50 conversations/day** once
Groq's tier exhausts, regardless of how token-heavy each conversation is.

**Tier 3 — Gemini** (`gemini-2.5-flash`): Google no longer publishes a static RPD/TPM table
(`ai.google.dev/gemini-api/docs/rate-limits` redirects to the account's own dashboard at
`aistudio.google.com/rate-limit`, which requires login no session here has). **2026-08-06:
measured by controlled empirical probing instead of reading the dashboard** — real,
non-mocked `google-genai` calls against the live API, request shape matched to this app's
own measured production average (system+user prompt sized to land the model's reported
`prompt_token_count` near the ~925-1,739 token range, `max_output_tokens=400` matching
`config.yaml`), sent rapid-fire with no artificial delay until the API's own 429 fired, one
phase at a time, stopping immediately on first failure rather than exhausting the account
(9 total calls: 8 succeeded, 1 hit the limit; ~9,150 tokens spent probing).

- **RPM = 5, authoritative** — read directly off the 429's own structured error body, not
  inferred: `quotaId: 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', quotaValue: '5'`.
  6 requests landed before the 7th tripped it (window-boundary effect); the quota system's own
  reported value (5) is the number to plan against. In conversations/minute terms (÷ the
  measured ~2.07 calls/conversation): **~2.4 conversations/minute of burst capacity**, the
  same shape as Groq's own TPM-derived "~2 conversations/minute" figure above.
- **TPM: no separate binding ceiling found at realistic sizes** — a second phase sent 2
  requests at ~1,739 input tokens each (1.9x the phase-1 size, still comfortably above this
  app's real per-call average) inside one RPM window; both succeeded with no token-based
  violation. The quotaId itself confirms this is a request-COUNT quota ("...Requests",
  "...PerMinute"), not token-gated, so TPM (if any exists) sits above what this app's real
  conversations would ever approach.
- **RPD / TPD: not measured, deliberately** — discovering these would mean sustaining calls
  past RPM=5 for a full day, i.e. actually exhausting the tier's real daily quota to find its
  edge. That's the opposite of "probe enough to bound the number, not exhaust it," and would
  burn the same quota real fallback traffic might need that day. Do not treat the absence of a
  measured RPD as "unlimited" — it is genuinely unknown, not zero-risk.

**Combined measured floor: ~170 (Groq) + 50 (OpenRouter) = ~220 conversations/day, PLUS
Gemini's own measured ~2.4 conversations/minute of burst capacity once tiers 1+2 exhaust, for
as long as Gemini's (unmeasured) daily quota lasts.** Do not collapse this into a single
combined "conversations/day" number — Groq/OpenRouter's 220 is a real full-day figure, while
Gemini's contribution is a *rate* (minute-level), not yet a *volume* (day-level); multiplying
5 RPM × 1,440 minutes to get a theoretical 7,200/day would be exactly the kind of unverified
extrapolation this section exists to avoid (free-tier RPD caps are routinely far tighter than
RPM×1440 would suggest). All three tiers verified live end-to-end 2026-07-31 (real, non-mocked
calls forcing Groq+OpenRouter failures and confirming Gemini serves a real response, for both
`chat()` and `chat_stream()`); Gemini's RPM specifically re-verified live 2026-08-06.

`DEMO_DAILY_REQUEST_CAP` was lowered from 700 to 150 on 2026-07-31, before the OpenRouter/
Gemini tiers were fixed — it sits below the Groq-only ~170 ceiling so real users hit the
app's own honest "Demo limit reached for today — try again tomorrow" message (see
`api/routes/chat.py` / `api/routes/demo.py`) instead of a raw provider 429 or a multi-minute
hang. Now that tiers 2+3 add real headroom above 170, this cap is conservative rather than
tight — tighter than necessary fails safe (an honest cap message), tighter is not a
production risk. Gemini's RPD remains genuinely unknown (see above), so there still isn't a
safe combined-conversations/day number to raise the cap against.

What it would take to raise the ceiling further, each option labeled with what's actually
verified vs. not:

1. **Check Gemini's real RPD** in the account's AI Studio dashboard (still login-walled) —
   free, no code change; this is the one number empirical probing deliberately declined to
   chase (see above) since finding it would mean exhausting it.
2. **Groq's paid "Developer" tier** raises limits above the free tier, but the exact TPD/TPM
   numbers and pricing are NOT published on console.groq.com/docs/rate-limits without logging
   into the account's own billing page (console.groq.com/settings/billing/plans) — do not
   assume a number here; this is a paid-tier decision and needs explicit sign-off before
   adoption (project's $0-cost-by-default policy, see CLAUDE.md).
3. **Reduce tokens/conversation** — already relatively lean at ~2.07 LLM calls/turn (the
   router already fast-paths and skips an LLM call for simple product searches, e.g.
   `[router] fast-path: search (items=4) -> respond (LLM skipped)` in the logs); further
   reduction would need prompt-level auditing, not assumed here.

---

## Deploy Cloud Run services (legacy — per-brand)

All three services use the same image and scale to zero (min-instances=0). Cold starts show
the "warming up…" spinner in the frontend.

```bash
# CORS_ORIGINS must be a single comma-free origin. gcloud --set-env-vars treats every comma
# as an env-var separator; adding ",http://localhost:3000" silently corrupts the env block.
CORS_ORIGINS="https://stylemaitri.vercel.app"
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

Then force a new Cloud Run revision so it cold-starts and re-downloads the fresh index. Use
`scripts/deploy_backend.ps1 -SkipBuild` for this rather than a bare `gcloud run deploy` —
the same image is redeployed unchanged, but the script still runs the
`update-traffic --to-latest` + verify steps, so this cold-restart-only redeploy can't
silently leave traffic pinned to the old revision (see "Deploy backend (unified service)"
above):

```powershell
$currentTag = (gcloud run services describe asa-stylist-api --region=asia-south1 `
  --format="value(spec.template.spec.containers[0].image)") -replace '^.*:', ''
pwsh scripts/deploy_backend.ps1 -SkipBuild -Tag $currentTag
```

### Per-brand indices (legacy — `asa-snitch`, `asa-myntra`, `asa-flipkart`)

```bash
GCS_BUCKET="your-gcs-bucket-name"

gcloud storage cp -r data/processed/snitch    gs://${GCS_BUCKET}/snitch/
gcloud storage cp -r data/processed/myntra    gs://${GCS_BUCKET}/myntra/
gcloud storage cp -r data/processed/flipkart  gs://${GCS_BUCKET}/flipkart/
```

### Relevance regression gate — mandatory before any backend deploy

Run the deterministic eval gate (zero LLM calls, ~2-4 min) and deploy only on `ALL PASS`:

```bash
python scripts/eval_gate.py
```

Thresholds (baseline 2026-07-10: P@5 0.889, NDCG@10 0.914, intent 92.4%, gates 100%):
precision@5 >= 0.80, NDCG@10 >= 0.85, intent all-exact >= 88%, correctness gates at 100%.
A `REGRESSION - do not deploy` line means ranking/parser/composer quality silently dropped —
fix before shipping. Not in ci.yml: `data/processed/unified` is gitignored and CI has no GCS
credentials, so this gate runs locally as part of the deploy ritual.

### Two-gold-set practice — dev set vs. cold holdout

`eval/fixtures/strict_gold_queries.yaml` + `strict_gold_labels.yaml` is the **inspectable
development set**: read it, reason about misses, add regression queries for every fix — this is
normal and expected.

A **separate, never-inspected holdout** is required whenever a generalization claim is being
made ("does this class of fix hold beyond the exact queries used to develop it?"). The holdout
must be:
1. Written and hand-labeled cold, before any fix informed by it.
2. Scored exactly once as a genuine out-of-sample check.
3. Retired to development data the moment its results are read — see
   `eval/fixtures/holdout_gold_queries.yaml`'s header for the canonical example (2026-07-25:
   designed as a holdout, scored once at a true cold 0.862, then 3 of its 24 queries drove real
   code fixes the same session — after that point it is honestly DEVELOPMENT DATA, not a holdout,
   and citing its post-fix 0.881 as a "generalization" number would be fabrication by omission).

**Never reuse an opened holdout as if it were still cold.** Once a holdout's results have been
read (even just to hand-label it), it has informed the session and any responsive fix — it must
be relabeled DEVELOPMENT DATA in its own file header, and a genuine next generalization check
needs a **new** query set this session has never inspected. This is why validation must never be
allowed to burn its own instrument: the whole value of a holdout number is that reading it comes
before, not after, the fixes it's meant to validate.

```bash
# Cold-checking generalization: write a NEW eval/fixtures/<name>_queries.yaml +
# <name>_labels.yaml that no fix this session has looked at, then:
python scripts/eval_strict.py --mode pipeline \
  --queries-path eval/fixtures/<name>_queries.yaml \
  --labels-path eval/fixtures/<name>_labels.yaml
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
`https://stylemaitri.vercel.app`. **Keep the value comma-free** — gcloud's `--set-env-vars`
treats every comma as an env-var separator, so a multi-origin string (e.g.
`https://stylemaitri.vercel.app,http://localhost:3000`) silently splits into two env vars
and corrupts the env block.

```bash
VERCEL_URL="https://stylemaitri.vercel.app"

for svc in snitch myntra flipkart; do
  gcloud run services update asa-${svc} \
    --region=asia-south1 \
    --update-env-vars="CORS_ORIGINS=${VERCEL_URL}"
done
```

There is no CI deploy path today (see "CI/CD" below) — `CORS_ORIGINS` is set directly on
each Cloud Run service via the `update-env-vars` command above.

---

## CI/CD

There is no CI deploy path. `.github/workflows/deploy-demo.yml` was removed 2026-07-09:
it triggered on manual dispatch and looked correct on paper (WIF auth, no service account
key JSON), but the required GitHub secrets (`GCP_PROJECT_ID`, `WIF_PROVIDER`,
`WIF_SERVICE_ACCOUNT`, `GROQ_API_KEY`, `DEMO_JWT_SECRET`, `DATABASE_URL`, `SUPABASE_URL`,
`GCS_BUCKET`, `DEMO_CORS_ORIGINS`) were never actually added to the repo, so the workflow
never ran successfully — every deploy to date has been local, via
`pwsh scripts/deploy_backend.ps1` (see "Deploy backend (unified service)" above).

If a CI deploy path is built in the future, it should:

- Authenticate via Workload Identity Federation (WIF), as the removed workflow attempted —
  no service account key JSON as a GitHub secret.
- Pass secrets to Cloud Run via `--set-secrets` against Secret Manager references (see
  `scripts/deploy_unified.sh`'s `SECRETS` variable for the reference names already in use
  for this project), not via plain env-var GitHub secrets rendered into
  `--set-env-vars` — that pattern both requires re-adding all the secrets above to GitHub
  and is exactly the mechanism that produces the env-block-replacement gotcha documented
  above.
- Include the same post-deploy `update-traffic --to-latest` + verification step as
  `scripts/deploy_backend.ps1`, so CI can't reproduce the traffic-pinning "deployed but
  not live" failure either.
