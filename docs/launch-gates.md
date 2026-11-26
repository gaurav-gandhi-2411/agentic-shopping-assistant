# Launch gates (single checklist)

One list of what must be true before StyleMaitri is announced publicly. Status vocabulary:
**done** (evidence in the repo or a linked artifact), **open** (known not done), **status
unverified** (the repo has no artifact that proves either way; do not read it as done). Evidence
is limited to files in this repository at the commit that last touched this file; nothing was
re-measured against production when writing it. Where a gate depends on a number that is not in the
repo, it says so instead of quoting the number.

## A. Operational gates

| Gate | Owner | Status | Evidence / what exists |
|---|---|---|---|
| Dedicated Groq organisation and key for prod chat | GG | **open (not done)** | Prod chat, the daily WS canary and every eval/gate/judge run share ONE org key (project `CLAUDE.md`, section "SHARED GROQ KEY", GG decision 2026-11-21; that file is untracked in the primary checkout, so this is not in the repo tree). Observed consequence: the 2026-11-21 02:29 UTC canary and the 08:46 UTC deploy smoke fell back to `gpt-oss-120b`/`qwen` after stage-2 evals had used the day's `gpt-oss-20b` tokens (recorded in the same CLAUDE.md section; Cloud Run logs are not in the repo). Budget arithmetic: `docs/ops/canary-budget.md` |
| Catalogue freshness: < 2% dead links | architect / executor | **open** (V3e baseline pending) | Target from the Wave 1b spec (B3). The 60-query measurement (V3e) has not produced an artifact: no `reports/baselines/` directory exists in this tree. Index age is checked (`/readyz` `index` block, `index_stale` canary assertion at 36 h: `DEPLOY.md` "Index age", `api/routes/canary.py`), which is freshness of the build, not link validity |
| Uptime alerting | GG | **partly done** | UptimeRobot was declined by GG. What exists: `.github/workflows/uptime-alert.yml` (cron `*/15`, probes `/api/brand` and, in a second job, `/readyz`; opens/updates a `prod-alert` issue, deduplicated by title); `canary.yml` (daily WS-path canary, opens `prod-alert` issue `StyleMaitri prod canary is failing`); the Cloud Scheduler job `asa-keepwarm-healthz` (GET every 4 min, keeps an instance warm; `docs/ops/api-docs-switch.md`). Limits stated honestly: GitHub's cron is best-effort (measured median gap 246 min for the `*/15` job, `canary-budget.md`), and nothing external polls `/canary/status` |
| Affiliate network signup | GG | **open (not started)** | Wave 1b spec section 4 item 3: after V2 because reviewers look at the live site. The code is ready for it and unconfigured: affiliate tags are read from `ASA_AFFILIATE_<STORE>` env vars (`src/config/stores.py`, tests in `tests/test_affiliate_links.py`), no real IDs are committed, and without those variables links are unchanged. Whether any `ASA_AFFILIATE_*` variable is set on the live service is not in the repo: status unverified, but signup itself has not started |
| Public-claims PR approved | GG | **open** (V6 PR pending) | Wave 1b V6 / B8: every public number traced to an artifact; text is GG-approved before it changes. Not merged |
| Lighthouse mobile targets | GG | targets documented, **measurement not in the repo** | Documented in `DEPLOY.md` ("Measure Lighthouse (B2) locally"): Performance >= 85, LCP < 2.5 s, CLS < 0.1, mobile, throttled, 3 runs, median. The only committed Lighthouse report (`reports/lighthouse-stylemaitri-baseline.report.json`, fetched 2026-07-12) is the **desktop** preset and predates the landing-page change, so it is not evidence for this gate. A current production mobile measurement is a to-do |

## B. Wave 1b exit criteria (B1 to B9)

Statements are from `docs/wave1b-spec.md` section 0 (a file that is untracked in the primary
checkout at the time of writing, so it is not in this PR's tree; the statements and targets below
are copied from it). Status is given only where a repo file evidences it.

| # | Criterion (target) | Status | Evidence in the repo, or why unverified |
|---|---|---|---|
| B1 | `/` serves the working assistant with no sign-in (yes; first answer in <= 2 taps) | **implemented; production check not in the repo** | Merged as `a53fbfd` "Wave 1b V2: landing page = the working assistant (B1, B2) (#104)"; `frontend/app/page.tsx`, `frontend/components/landing/AssistantLanding.tsx`. No committed artifact of a production check, so the "measured on production" bar is **status unverified** |
| B2 | Lighthouse mobile on `/` (Performance >= 85, LCP < 2.5 s, CLS < 0.1; throttled, 3 runs, median) | **status unverified** | Targets and the measurement procedure are in `DEPLOY.md`; no mobile result is committed (see the Lighthouse row above) |
| B3 | Shown-card gone/sold-out rate over 60 queries (< 2% point estimate, Wilson 95% interval reported) | **open / status unverified** | Needs the V3e measurement CSV (`reports/baselines/`, absent). Public-index filtering of non-refreshable stores exists (`281179b`, #106) and catalogue hygiene landed (`54ee822`, #130), but neither is a measurement of this rate |
| B4 | Shown-card price mismatch vs live store (< 1%, Wilson 95% interval) | **open / status unverified** | Same sample and artifact as B3; none in the repo. The canary's PRICE INVARIANT (`scripts/ops/canary.py`) checks that a price is present and consistent across the response; it does not compare with the live store |
| B5 | Assistant text with an unrendered price/discount/size/delivery claim, 150 live prompts (0) | **status unverified** | No 150-prompt live evaluation artifact in the repo. CI runs an offline grounding smoke (`.github/workflows/ci.yml`, `python -m evals.grounding.run --llm mock --agent stub --smoke`), which is not this measurement |
| B6 | Tokens per chat turn, p50 over >= 30 live turns (<= 2,200; stretch <= 1,800) | **status unverified** | Not measured as specified. Indicative only, from canary turns (`docs/ops/canary-budget.md`): search turn about 3,345 tokens today, about 2,150 with reranker variant C (PR #136, not merged), outfit turn about 1,390. These are not the B6 sample and carry no cached/uncached split |
| B7 | Daily capacity statement (turns/day and turns/minute the free tier supports) | **status unverified** | `docs/ops/canary-budget.md` computes the canary's share against the Groq limits (RPD 1,000 and TPM 8,000 verified; TPD 200,000 **unverified**); the published capacity statement of B7 is not in the repo |
| B8 | Public claims each traced to an artifact (100%, GG-approved) | **open** | V6 PR pending (see section A) |
| B9 | Catalogue refresh job runs daily, green 3 consecutive days, index age < 36 h in `/readyz` | **status unverified** | The age plumbing exists (`/readyz` `index` block, `index_stale` canary assertion, `DEPLOY.md` "Index age"; refresh library code under `src/` with `DomainThrottle`), but `.github/workflows/` contains no scheduled refresh workflow (`canary.yml`, `ci.yml`, `deploy.yml`, `migration-integrity.yml`, `uptime-alert.yml` only), and there is no run history in the repo |

## C. Related items this PR touches

- `/ping` is the liveness/keep-warm route; `/healthz` is intercepted by Cloud Run's front end
  (404) and works only in-container. `DISABLE_API_DOCS` stays OFF until the keep-warm job is
  repointed: `docs/ops/api-docs-switch.md`.
- The daily WS canary asserts every landing chip returns >= 4 cards and a warm turn < 8 s:
  `scripts/ops/canary.py` (module docstring).
