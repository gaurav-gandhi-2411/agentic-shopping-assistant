# W8 part A: deploy path, cost inventory (E11), cold-start baseline (E8)

Date: 2026-11-03. Project `stylemaitri-prod-260813`, region `asia-south1`. Base commit: master `8a53f29`.
Everything here is READ-ONLY output. Nothing was applied to GCP. Every gcloud call ran with
`--account=gaurav.gandhi1129@gmail.com --project=stylemaitri-prod-260813`. No chat call, no probe
traffic and no secret payload read. Scratch files live outside the repo and are not committed.

Shorthand: `A="--account=gaurav.gandhi1129@gmail.com --project=stylemaitri-prod-260813"`.

## 1. Pricing sources fetched (2026-11-03, via `curl -sL`, HTML stripped)

| Page | Used for |
|---|---|
| https://cloud.google.com/run/pricing | request-based rates, free tier, billable-time rules (Mumbai is in the Tier 1 list for request-based billing) |
| https://cloud.google.com/artifact-registry/pricing | storage free tier and rate |
| https://cloud.google.com/secret-manager/pricing | secret version / access free tier |
| https://cloud.google.com/build/pricing | Cloud Build free minutes |
| https://cloud.google.com/storage/pricing | GCS always-free scope; us-central1 standard rate |
| https://cloud.google.com/logging/pricing | log ingestion free allotment |
| https://cloud.google.com/scheduler/pricing | Cloud Scheduler free jobs |
| https://cloud.google.com/run/docs/configuring/min-instances | service- vs revision-level min instances, tagged-revision billing |
| https://cloud.google.com/sdk/gcloud/reference/run/services/update | `--min` (service level) vs `--min-instances` (revision level) |
| https://cloud.google.com/artifact-registry/docs/repositories/cleanup-policy | policy JSON, "docker images referenced by a parent manifest are not deleted" |

Values read from those pages: Cloud Run request-based free tier 180,000 vCPU-s, 360,000 GiB-s, 2M requests
per month per billing account; Tier 1 rates $0.000024 /vCPU-s, $0.0000025 /GiB-s, $0.40 per 1M requests;
instance start-up time is billable. Artifact Registry: first 0.5 GiB-month free per billing account, then
$0.000136986 /GiB-hour. Secret Manager: 6 active versions and 10,000 access ops free; extra version
$0.000082192 /hour. Cloud Build: 2,500 free build-minutes per billing account, e2-standard-2 $0.006/min.
Cloud Logging: first 50 GiB/project/month free. Cloud Scheduler: 3 jobs free per billing account, then
$0.10/job/month. GCS Always Free: 5 GB-months standard, applies ONLY in us-west1/us-central1/us-east1.

NOT fetched or not resolvable from static HTML: Mumbai (asia-south1) GCS standard $/GiB (page renders it by
JS; only the us-central1 list rate $0.000027397 /GiB-hour = $0.02/GiB-month was readable), Cloud Trace,
Pub/Sub, BigQuery pricing. Those rows are marked UNVERIFIED.

## 2. Cloud Run

Commands: `gcloud run services describe asa-stylist-api --region=asia-south1 $A --format=json` and
`gcloud run revisions list --service=asa-stylist-api --region=asia-south1 $A --format=json`.

| Item | Value |
|---|---|
| URL | https://asa-stylist-api-ec3mwppesa-el.a.run.app |
| Serving | `asa-stylist-api-00004-c7x` 100% (created 2026-08-15). Tag `candidate` -> `00003-wgm`, 0% |
| Service-level maxScale | 10 (annotation `run.googleapis.com/maxScale`), no service-level minScale |
| Template (what a plain deploy inherits) | `minScale: 1`, `maxScale: 1`, `startup-cpu-boost: true` |
| Runtime SA (all revisions) | `631709154646-compute@developer.gserviceaccount.com` |
| Resources (all revisions) | cpu 1, memory 4Gi, concurrency 4, timeout 300 s, startup TCP probe 240 s |
| CPU throttling annotation | unset on every revision = default request-based billing (docs: instance billed only while starting, shutting down, or serving a request) |
| Env | DEMO_MODE, LLM_PROVIDER=groq, CORS_ORIGINS, INDEX_STORE_URI + secret refs GROQ_API_KEY, DEMO_JWT_SECRET, DATABASE_URL, SUPABASE_URL, GEMINI_API_KEY, OPENROUTER_API_KEY (names only) |
| `DEMO_DAILY_REQUEST_CAP` | NOT set on the live revision (deploy/service.yaml says 150; code default 700/day) |

| Revision | Created | min / max (revision annotations) | boost |
|---|---|---|---|
| 00005-cbg | 2026-08-17 | **1** / 1 | true |
| 00004-c7x (serving) | 2026-08-15 | none / 1 | true |
| 00003-wgm (tag candidate) | 2026-08-15 | none / 1 | true |
| 00002-cfw | 2026-08-13 | none / 1 | true |
| 00001-bv5 | 2026-08-13 | none / 1 | true |

Finding: `00005-cbg` is still listed (Ready) with minScale 1. It receives no traffic and carries no tag, so per the
min-instances doc it is not started; but it is why the template still says minScale 1. A deploy MUST pass
`--min-instances=0` (deploy.yml does, and asserts it).

Free tier: 180,000 vCPU-s / 360,000 GiB-s / 2M requests. Measured use below (section 4/5): about 12.7k vCPU-s
and 50.7k GiB-s of start-up time per month (7.0% and 14.1% of the free tier) plus 495 requests (0.025% of 2M).
Headroom: large. UNVERIFIED: real billed instance time (no billing export; see section 9).

## 3. Everything else billable or enabled

| Resource | Measured | Command | Free allowance (source) | Headroom / cost |
|---|---|---|---|---|
| Artifact Registry `cloud-run-source-deploy` (asia-south1) | 4,563.552 MiB = 4.457 GiB (the `repositories list` column is MiB; `gcloud artifacts repositories describe` reports the same repo as "4785.231MB" decimal = 4.785e9 bytes = 4.457 GiB), 5 versions, ~1.61 GB per image, **no cleanup policy** | `gcloud artifacts repositories list $A`; `gcloud artifacts docker images list asia-south1-docker.pkg.dev/stylemaitri-prod-260813/cloud-run-source-deploy $A --include-tags --format=json` | 0.5 GiB-month per BILLING ACCOUNT (shared with 4 projects) | **Over free tier**: about (4.457 - 0.5) x $0.10 = **$0.40/month** gross. Steady state with keep-3 is about 4.5 GiB = $0.40/month; keep-2 = $0.25; keep-1 = $0.10. No image fits in 0.5 GiB, so AR is never $0. |
| GCS `asa-demo-indices-stylemaitri-prod` (ASIA-SOUTH1, STANDARD) | 862.07 MiB | `gcloud storage du -s -r gs://asa-demo-indices-stylemaitri-prod $A` | Always Free is US regions only, so 0 here | about 0.84 GiB x $0.02 (us-central1 list rate; Mumbai rate UNVERIFIED) = about $0.017/month |
| GCS `run-sources-stylemaitri-prod-260813-asia-south1` | 136.18 MiB (Cloud Build source zips) | same with that bucket | none (non-US) | about $0.003/month; deletable once builds move to the runner (not done) |
| Secret Manager | 6 secrets x 1 version each (asa-database-url, asa-demo-jwt-secret, asa-gemini-api-key, asa-groq-api-key, asa-openrouter-api-key, asa-supabase-url) | `gcloud secrets list $A` then `gcloud secrets versions list <name> $A` | 6 active versions, 10,000 access ops (per billing account) | **0 headroom on versions**: a 7th costs about $0.06/month. Access ops: about 6 x 179 starts = about 1,074/month estimated (10.7% of free), but the billing account is shared |
| Cloud Build | 2 builds total, both 2026-08-13 (6m17s and 7m40s, status SUCCESS); **0 in the last 30 days** | `gcloud builds list $A --region=asia-south1 --filter="createTime>-P30D"` -> 0 rows; `--region=global` -> 0 | 2,500 min/month per billing account | 100% headroom |
| Cloud Scheduler | API not enabled; no jobs possible | `gcloud scheduler jobs list --location=asia-south1 $A` -> "API not enabled" | 3 jobs free per billing account | n/a; enabling is a change, not done |
| Cloud Logging | not measured | n/a | 50 GiB/project/month | NOT MEASURED; 30-day read returned 358 system + 495 request entries plus stdout, tiny |
| Pub/Sub | no topics | `gcloud pubsub topics list $A` -> empty | n/a | $0 (API enabled only) |
| BigQuery, Dataform, Dataplex, Analytics Hub, Datastore, Cloud Trace, Telemetry, Cloud SQL component | APIs enabled, no resources found where checked (BigQuery/Datastore/Trace contents NOT checked: `bq` CLI unusable here) | `gcloud services list --enabled $A` | varies | UNVERIFIED, expected $0 (no resources created by this app) |
| Compute/VPC/NAT/LB/static IP | not checked (compute API not enabled in the list) | | | $0 expected |

Enabled APIs (31): analyticshub, artifactregistry, bigquery (+connection, datapolicy, datatransfer, migration,
reservation, storage), cloudapis, cloudbilling, cloudbuild, cloudtrace, containerregistry, dataform, dataplex, datastore,
iam, iamcredentials, logging, monitoring, pubsub, run, secretmanager, servicemanagement, serviceusage,
sql-component, storage-api, storage-component, storage, telemetry.

## 4. E11 verdict: is the project at INR 0?

**No, not strictly.** Artifact Registry alone is above its free allowance: about $0.40 per month gross ((4.457 - 0.5) x $0.10 = $0.396)
(computation: `python cost.py` in scratch; inputs 4,563.552 MiB = 4.457 GiB, $0.10/GiB-month, 0.5 GiB free). That is the one
confirmed billing line. Everything else (Cloud Run compute and requests, Cloud Build, Secret Manager,
Cloud Scheduler) is inside its free tier on measured usage, with Secret Manager versions at exactly the cap
(6/6) and GCS about $0.02/month outside any free tier. Estimated total about $0.42/month (about INR 35 to 40 at a
rate I did not fetch; UNVERIFIED). Actual invoiced spend is UNVERIFIED: see section 9.
Caveat: the 0.5 GiB AR free tier and the Cloud Run/Build/Secret free tiers are per BILLING ACCOUNT
(`01285B-91E4CB-70AD7E`, which also carries triageiq, reviewiq, dealhunter), so other projects may already have
consumed them.

## 5. E8 baseline: cold-start time of the serving revision (00004-c7x)

Window: logs retained from 2026-10-04T08:47:54Z to 2026-11-03T04:18Z (the 30-day retention floor; all entries are
revision `asa-stylist-api-00004-c7x`). Generated no traffic.

Filters (exact):

```
# instance start + startup-probe events
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="asa-stylist-api" AND logName="projects/stylemaitri-prod-260813/logs/run.googleapis.com%2Fvarlog%2Fsystem"' $A --freshness=30d --limit=20000 --order=asc --format=json
# request log
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="asa-stylist-api" AND logName="projects/stylemaitri-prod-260813/logs/run.googleapis.com%2Frequests"' $A --freshness=30d --limit=50000 --order=asc --format=json
```

Events paired per `labels.instanceId`: "Starting new instance. Reason: AUTOSCALING ..." and "Default STARTUP TCP probe
succeeded after 1 attempt ...". Script: scratch `e8.py` (percentiles by linear interpolation).

| Metric | n | min | p50 | p90 | p95 | max |
|---|---|---|---|---|---|---|
| **Start -> startup probe succeeded (headline)** | **179** | 35.9 s | **71.5 s** | **78.4 s** | 80.8 s | 88.1 s |
| Start -> first 200 response complete (excl. 12 instances whose first logged request predates their start event, unexplained) | 155 | 32.8 s | 71.0 s | 78.8 s | 82.9 s | 91.5 s |
| Latency of the first 200 request on a fresh instance (all 167 that had one) | 167 | 0.0 s | 69.9 s | 77.6 s | 80.1 s | 87.3 s |

Reading: 179 cold starts in 30 days (about 6 per day, max 9 on one day). The probe passes only after the app has
loaded model and index (eager lifespan); once it passes, requests are answered in milliseconds (warm
`/api/brand` p50 7.8 ms, p90 35.9 ms, n=152). So the user-visible cold start is about the probe time.
The probe in use is a TCP probe on port 8080 (failureThreshold 1, 240 s), and CPU boost was already on for 00004.
Traffic mix over the 30 days, 495 requests: 225 `/readyz`, 224 `/api/brand`, 12 `/auth/ws-ticket`, 12 `/chat/stream`,
3 `/demo/session`, rest misc; i.e. about 91% is the two monitors, only 12 real chat sessions in 30 days.
Statuses: 458 x 200, 12 x 429, 12 x 404, 12 x 101, 1 x 405.
Not measured: the "after" number (needs the W8 deploy applied).

## 6. Scheduler reality check (affects the keep-warm design)

`gh run list --repo gaurav-gandhi-2411/agentic-shopping-assistant --workflow "Uptime Alert" --limit 100 --json createdAt,event`
(script `gaps.py`): the workflow is scheduled `*/15 * * * *` but 100 scheduled runs span 2026-10-16T11:21Z to
2026-11-03T01:48Z, i.e. 5.7 runs/day instead of 96. Gap between runs: min 100 min, **median 246 min**, p90 376 min,
max 613 min. A 10-minute GitHub cron will behave the same way, so it will cut cold starts but cannot hold the
instance warm. The same data explains why the cold-start count (179) is far below the nominal ping count.
Options if a real warm guarantee is wanted: Cloud Scheduler (3 jobs free per billing account, needs the API enabled;
not done here), or min-instances 1 (rejected by the brief: idle billing).

## 7. keep-warm cost (request-based billing)

`GET /api/brand` every 10 min = 6 x 24 x 30 = 4,320 requests/month (cap; real runs are about 40 x fewer, section 6).
Billed time is rounded up to 100 ms per request: 4,320 x 0.1 s = 432 s => 432 vCPU-s, 1,728 GiB-s (4 GiB).
Gross: 432 x $0.000024 + 1,728 x $0.0000025 + 4,320/1M x $0.40 = $0.0104 + $0.0043 + $0.0017 = **$0.0164/month**
(`python cost.py`). Against the free tier: 0.24% of vCPU-s, 0.48% of GiB-s, 0.22% of requests, so **$0 net**.
If it does keep the instance alive it also removes cold starts: each costs about 71 s x (1 vCPU + 4 GiB) of billed
start-up, 179 starts = 12,673 vCPU-s + 50,693 GiB-s (gross $0.43/month, still inside the free tier).
Idle instances are not billed under request-based billing (min instances 0).

## 8. Build path decision (numbers)

| | A. build on GitHub runner, push to AR (chosen) | B. Cloud Build (`gcloud run deploy --source`) |
|---|---|---|
| Compute cost | $0 (standard runners are free on this public repo: `gh repo view` -> PUBLIC) | 2 historical builds took 6m17s and 7m40s; 2,500 free min/month => about 325 to 397 builds/month free, so $0 too |
| Free tier consumed | none | billing-account-wide minutes shared with 3 other projects |
| Extra IAM for the deployer | none beyond AR writer | Cloud Build editor, source bucket write, actAs on the build SA |
| Extra storage | none | `run-sources-...` bucket (136 MiB, about $0.003/month) keeps accumulating zips |
| Risk | runner disk (14 GB) vs a torch CPU image: UNVERIFIED, 1.6 GB compressed image | slower to diagnose, wider permissions |

Both are INR 0; A wins on least privilege and on not drawing from a shared free pool. Image tag = full commit SHA.
AR storage is the same either way (about 1.61 GB per image), so the only cost lever is how many images are kept.

## 9. IAM and billing visibility

Project IAM policy (`gcloud projects get-iam-policy stylemaitri-prod-260813 $A --format=json`), every binding:

| Role | Member |
|---|---|
| roles/owner | user:gaurav.gandhi1129@gmail.com |
| roles/editor | serviceAccount:631709154646-compute@developer.gserviceaccount.com (runtime SA; broad, flagged) |
| roles/cloudbuild.builds.builder | serviceAccount:631709154646@cloudbuild.gserviceaccount.com |
| roles/cloudbuild.serviceAgent | service-631709154646@gcp-sa-cloudbuild |
| roles/artifactregistry.serviceAgent | service-631709154646@gcp-sa-artifactregistry |
| roles/containerregistry.ServiceAgent | service-631709154646@containerregistry |
| roles/pubsub.serviceAgent | service-631709154646@gcp-sa-pubsub |
| roles/run.serviceAgent | service-631709154646@serverless-robot-prod |

**gaurav.gandhi2411@ and gaurav.gandhi1249@: ZERO bindings** in the project policy. (Not checked: SA-level policies
other than `stylemaitri-ci-index-reader`, bucket policy of the indices bucket, secret-level policies. The two
inspected, SA policy of ci-index-reader and indices bucket policy, contain neither identity.)
Cloud Run service metadata still records `serving.knative.dev/creator: gaurav.gandhi2411@gmail.com` (creation
history, not a binding).

Service accounts: `631709154646-compute@developer.gserviceaccount.com` (default compute, runtime),
`stylemaitri-ci-index-reader@...` (CI, `roles/storage.objectViewer` on the indices bucket only).

WIF: pool `github-pool` (ACTIVE), provider `github-provider`, issuer `https://token.actions.githubusercontent.com`,
attribute condition `assertion.repository_owner=='gaurav-gandhi-2411'`, mapping google.subject=assertion.sub,
attribute.repository, attribute.repository_owner (NO attribute.ref mapping, which is why the master-only deploy
binding needs a separate POOL, since principalSet bindings are pool-scoped). `stylemaitri-ci-index-reader` allows `roles/iam.workloadIdentityUser` to
`principalSet://.../github-pool/attribute.repository/gaurav-gandhi-2411/agentic-shopping-assistant`.

Repo secrets present (names only, `gh secret list`): DATABASE_URL, DEMO_CORS_ORIGINS, WIF_PROVIDER,
WIF_SERVICE_ACCOUNT. A workflow named `.github/workflows/deploy-demo.yml` is still listed as active by
`gh workflow list --all` although the file is not in master.

Billing spend: billing account `01285B-91E4CB-70AD7E` (INR, open) is readable; `gcloud billing budgets list` failed
because the Budget API is not enabled on the project, and no BigQuery billing export was found. **Actual invoiced spend
is NOT available**, hence every cost above is a computation from usage and list prices, not an invoice.

## 10. Deploy pipeline delivered (not applied)

`.github/workflows/deploy.yml`, `.github/workflows/keep-warm.yml`, `scripts/ops/deploy_smoke.py` (+ `tests/test_deploy_smoke.py`),
`scripts/ops/setup_deploy_iam.sh`, `scripts/ops/ar_cleanup_policy.json`, `DEPLOY.md` CI/CD section. Revised after independent verification (second commit): dedicated WIF pool,
chat smoke on the tag URL only, tested `main()`, `workflow_dispatch` requires green CI for the SHA, rollback also on
`cancelled`. actionlint
1.7.12 (checksum-verified binary in scratch) reports no findings for both workflows (shellcheck is not installed, so the inline
shell inside the workflows was NOT linted; `setup_deploy_iam.sh` passed `bash -n` and a `DRY_RUN=1` run).

## 10b. LLM calls in one `/chat` "black kurta set for haldi" (measured)

Method: script `scripts/ops/count_llm_calls.py` (committed; rerun with `--scratch <dir>`). In-process FastAPI `TestClient` (no lifespan) on `api.main.app`,
real `HybridRetriever` built from the local `data/processed/unified` index (read-only), a counting stub LLM
installed via `deps._init`, in-memory session store, no DATABASE_URL, `HF_HUB_OFFLINE=1`, LLM host pointed at a dead port.
Hermetic roots printed and asserted: TEMP, USERPROFILE, LOCALAPPDATA, APPDATA all under the scratch dir; HF_HOME points at
the real model cache, read-only (offline mode never writes). Each call is attributed to its call site via the stack frame.

| Scenario (router stub) | HTTP | items | LLM calls | by call site |
|---|---|---|---|---|
| router returns `search` | 200 | 5 | **2** | `reranker.py:170` x1, `graph.py:3993` x1 |
| router returns `outfit` (occasion haldi) | 200 | 5 | **2** | same; `router.py:58` was never called |

The router LLM is bypassed for this query (the code logs a deterministic fast-path, see DEPLOY.md "LLM calls per deploy"),
so the forced-`outfit` stub had no effect: the observed route was `search` with filters product_type=kurta, colour=Black in both runs.
Not exercised: LLM-routed turns, outfit turns (`outfit/rationale.py:446`, one call per look) and the memory summariser
(`memory/conversation.py:54`, only after 12 turns). Static upper bound for a fresh single turn on those paths: router
1 per agent iteration + reranker 1 + respond 1 + 1 rationale per look. Real provider clients may add retry/failover attempts
that this logical-call count does not include. `/readyz` currently makes 0 LLM calls (`deps.get_llm()` only fetches the
client); with W4 (#77) it makes 1. Per deploy: 2 measured now (tag-URL chat), 4 after #77.
Earlier claim "2 LLM-backed calls per deploy" was an unmeasured assumption of one chat per smoke run; it happened to equal
the real figure for a different reason (one chat = 2 calls, and the main-URL chat is now removed).

## 10c. Verifier fixes (second commit) in one place

1. WIF: a `principalSet://.../workloadIdentityPools/github-pool/attribute.repository/<repo>` binding is scoped to the whole pool, so it would
   match tokens from the existing owner-wide `github-provider` (used by ci.yml on PRs). Now a dedicated pool `github-deploy-pool` with one provider
   (condition: this repo AND `refs/heads/master`) and the deployer is bound only to that pool; the script verifies the deployer SA policy
   references only `github-deploy-pool` and exits 1 otherwise. Docs: https://cloud.google.com/iam/docs/workload-identity-federation
   (principalSet bindings grant access to all identities with an attribute) and
   https://cloud.google.com/iam/docs/best-practices-for-using-workload-identity-federation ("Use a single provider per workload identity pool").
2. Unit fix: AR size is 4,563.552 MiB = 4.457 GiB (not "MB"/range); cost about $0.40/month.
3. Follow-up, NOT implemented: the runtime SA is the default compute SA (`631709154646-compute@developer.gserviceaccount.com`) holding
   project `roles/editor`. Recommend a dedicated least-privilege runtime SA (Secret Manager accessor on the six `asa-*` secrets, objectViewer on the
   indices bucket, log writer) in a separate PR, then drop Editor from the compute SA and re-point the deployer's `serviceAccountUser`.
4. `roles/run.developer` is kept service-scoped with the commented project-level fallback; UNVERIFIED until the first real deploy.

## 11. UNVERIFIED list

1. Whether `/healthz` really 404s on Cloud Run (could not test without sending a request; deploy/keep-warm use `/api/brand`, verified 200 in logs).
2. Actual invoiced spend; Mumbai GCS price; Cloud Trace/BigQuery/Datastore contents.
3. That a service-scoped `roles/run.developer` is enough for `gcloud run deploy` (fallback documented in the IAM script).
4. That `gcloud run deploy --image` keeps env/secrets from the docs (no explicit statement found in fetched pages); the workflow asserts parity instead of trusting it.
5. That `gcloud run services update --min=0` writes the annotation `run.googleapis.com/minScale` the workflow reads (absent defaults to 0, so the per-revision check is the stronger guard).
6. Runner disk sufficiency for the image build; time the build takes.
7. Whether AR cleanup keepCount counts versions as assumed; dry-run logs will show it. The 3 newest versions today are all the c223bea family (prod image), so the two Aug-13 images (about 3.2 GB) are the delete candidates.
8. Why prod images dated 2026-08-15/17 have no Cloud Build record (only two builds, 2026-08-13, are listed).
