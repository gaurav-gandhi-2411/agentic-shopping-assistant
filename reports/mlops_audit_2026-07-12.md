# Report 8 — ML System Design / MLOps Audit

Compiled 2026-07-12. Sourced to files/commands run this session — no invented claims.
Ordered: what's solid → what's missing, each with a rough priority.

## What's solid

**Index build pipeline is deterministic and well-documented.** `numpy` seed=42 fixed
in build scripts (6 files confirmed via grep). `build_unified_index.py`'s own
docstring explains every design decision (why dense is re-embedded vs. CLIP
concatenated, why BM25 is corpus-global and rebuilt not merged, dedup policy, excluded
stores) — this is unusually good self-documentation for a solo project.

**The build pipeline has a real internal integrity check.** `build_unified_index.py`
asserts CLIP vector count == catalogue row count before completing — this actually
fired and caught a real mistake THIS SESSION (I rebuilt 4 brands' catalogues without
re-running their CLIP indices first; the assertion stopped the build rather than
silently shipping a misaligned index). That's a working safety net, not theoretical.

**Deploy ritual has a real rollback path.** `scripts/deploy_backend.ps1` captures the
currently-serving revision as a named "rollback anchor" before every deploy and prints
the exact `gcloud run services update-traffic` command to revert. Used successfully
multiple times this session.

**Eval discipline (once invoked) is genuinely rigorous.** `eval_gate.py` refuses to
report a comparison with ANY unlabeled item (`n_unlabeled == 0` is a hard gate
condition, not a warning) — this fired for real this session (catalogue rebuild
changed the labeled pool) and correctly blocked a deploy until labels were completed
rather than silently comparing apples to oranges.

**pip dependencies are currently vulnerability-free** (`pip-audit`, this session).

## What's missing or weak, prioritized

### 1. HIGH — The rigorous eval harness is 100% local-only; CI runs a different, stale one
`.github/workflows/ci.yml`'s `eval` job runs `scripts/eval_harness.py` (32 queries,
`continue-on-error: true`, "informational only") — a completely different, older
harness than `eval_gate.py`/`eval_strict.py`/`eval_model.py` (54 hand-labeled + 251
property-matched queries), which have driven every quality decision this session.
**None of that real infrastructure runs automatically, anywhere.** Every gate result
in this session's commits was produced by me running it manually. If I hadn't been
disciplined about running it before every commit, nothing would have caught a
regression. **This is the single most consequential MLOps gap in the repo** — it's
also exactly the gap Report 4 above proposes closing (wire `eval_gate.py` into CI,
same idea as wiring in the `requires_index` pytest tier).

### 2. HIGH — Index/data has no version history; only code does
Cloud Run revisions ARE versioned and rollback-able (confirmed, used this session).
The GCS index bucket is NOT: `gs://asa-demo-indices/unified/unified/*` is a flat,
unversioned path (bucket versioning confirmed OFF via `gcloud storage buckets
describe`) that every rebuild overwrites in place. **If a bad index gets uploaded and
deployed, there is no "roll back the data" button** — the only recovery path is
re-running the entire build pipeline from raw per-brand data again, hoping that raw
data hasn't itself changed or been lost. Cheap fix: timestamp or git-sha-suffix the
GCS index path (e.g. `unified/unified-<sha>/`) and update `INDEX_STORE_URI` per
deploy, or simply turn on GCS object versioning as an immediate stopgap.

### 3. HIGH — Zero production observability beyond error tracking
Grep for `/metrics`, `prometheus`, `structlog` across `api/`: zero matches. There is
Sentry (error tracking) and a basic per-request access-log middleware
(`api/main.py`), but **no latency histograms, no request-rate counters, no
LLM-token/cost metrics exported anywhere**, despite this session's own working
CLAUDE.md conventions calling for exactly that (Prometheus `/metrics`, structured
`latency_ms`/`provider`/`tokens_in`/`usd_cost` fields). Concretely: there is currently
no way to know, without SSHing into logs, whether p95 latency is degrading, whether
the demo cost cap is being approached, or whether a specific brand's index is failing
silently. This session's own latency numbers (Section E of the eval scorecard) are
ALL from local Ollama proxy runs — **there is no fresh production latency number
because there is no instrumentation producing one.**

### 4. MEDIUM — No retraining/refresh cadence for the catalogue at all
Nothing in the repo schedules a re-sync. Every re-sync this session (including Fix 1's
sold-out-item removal) was manually triggered by a human/me running
`download_shopify.py` by hand. For the 4 Shopify stores, that means inventory drift
resumes immediately after this fix — a product that sells out tomorrow will show as
"in stock" again until someone manually re-runs the pipeline. For the 4 non-Shopify
snapshot stores (myntra, flipkart, globalrepublic, libas), there is no stock signal
AT ALL, ever (a data-partnership gap, not a code gap — noted in Fix 1's report). A
scheduled GitHub Actions cron (`[skip ci]`-tagged, per this project's own house
convention for automated commits) re-running the Shopify sync weekly-or-more would
close most of this.

### 5. MEDIUM — Single-instance-only in-memory state (cross-reference: Report 7)
The rate limiter and WS ticket store are both explicitly single-instance-only
in-memory structures. Not a correctness bug today (min-instances/traffic don't
require multi-instance), but it means this service cannot horizontally scale without
a design change (Redis-backed state) — worth knowing before any real-traffic launch
that might need more than one instance.

### 6. LOW — Cold start is real but bounded and monitored at deploy time
`deploy_backend.ps1` allows up to 300s for a cold-start probe after every deploy
(observed this session: actual cold starts landed well under that, but CLIP/embedding
model loading alone took ~17s in local benchmarking this session, plausibly more on a
smaller Cloud Run CPU allocation). This is accounted for at deploy time but not
continuously monitored in production (see gap #3 — no metrics mean no visibility into
whether cold starts are getting worse as the index grows).

## Summary judgment

This is a genuinely well-engineered solo-project pipeline for what it is: deterministic
builds, a real internal integrity check, a real rollback path for code, and (when
manually invoked) a rigorous eval harness. The gap is uniformly the same shape across
findings #1, #2, and #3: **strong artifacts, no automation wrapping them.** The build
script asserts correctly but nothing schedules it to run. The eval harness gates
correctly but nothing runs it except a human. The index is well-structured but nothing
versions it. Closing #1 (wire eval_gate.py + the requires_index pytest tier into CI)
is the highest-leverage single fix — it would have caught neither of this session's
two production bugs directly (they were found via live-URL proof, a different
mechanism), but it converts "a disciplined human ran the gate every time" into "the
gate always runs," which is the actual point of having one.
