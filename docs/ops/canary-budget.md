# Canary LLM budget vs Groq limits

The tables below are produced by `scripts/ops/canary_budget.py` (committed, tested); its
inputs are listed here. Everything labelled ESTIMATE, UNVERIFIED or NOT MEASURED is not measured.

## Limits compared against

| Limit (per model; the chain is `openai/gpt-oss-20b` primary, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`) | Value | Status |
|---|---|---|
| Requests per day (RPD) | 1,000 | VERIFIED for this org on 2026-11-13 from the live API response header `x-ratelimit-limit-requests`, one call per chain model, all three read 1000 |
| Tokens per minute (TPM) | 8,000 | VERIFIED the same way, header `x-ratelimit-limit-tokens`, all three read 8000 |
| Tokens per day (TPD) | 200,000 | UNVERIFIED. It comes from the table at https://console.groq.com/docs/rate-limits (fetched 2026-11-13), which did not label itself Free or Developer; the headers do not expose a daily token limit and the plan name was not determinable from them. Confirm at https://console.groq.com/settings/limits |
| Requests per minute (RPM) | 30 | UNVERIFIED (docs table only, same caveat; not in the headers) |

Each model has its own quota; the primary takes the traffic and a fallback only sees traffic after
the primary 429s. **Real user chat shares the same org limits** (and `DEMO_DAILY_REQUEST_CAP` is
150 conversations/day).

## Inputs

| Input | Value | Source |
|---|---|---|
| Outfit turn | about 1,390 tokens, 1 LLM call | MEASURED: n=11 Sangeet canary turns, Cloud Logging `llm_call` records 2026-11-18..21 (supplied to this change; not re-derived here). Supersedes the earlier "up to 6 calls" ESTIMATE |
| Search turn, today | about 3,345 tokens = reranker about 2,000 + respond about 1,350 | MEASURED, same records |
| Search turn with reranker variant C | about 2,150 tokens (752 per rerank) | PR #136, **not merged yet**: a projection until it is deployed |
| Earlier measurement | p50 1,219 in + 124 out = 1,343 per call, 2 calls, 3,262 per turn (22 records, 2026-11-12..13) | kept for the provenance of the old tables; replaced by the two rows above |
| Reasoning tokens | NOT MEASURED in prod turns: the app logs `output_tokens` only. One direct probe (`max_tokens` 64, `gpt-oss-20b`) returned `reasoning_tokens` 17 of 35 completion tokens: a single tiny sample, not a p50 | direct probe 2026-11-13 |
| `/readyz` LLM probe | 1 call per uncached hit, cache 60 s (`READYZ_LLM_CACHE_SECONDS`), about 100 tokens ESTIMATE (prompt "ping", `max_tokens=1`; only the call count is from code) | `api/routes/health.py`, `src/llm/client.py` `probe()` |
| Callers of `/readyz` | `uptime-alert.yml` (cron `*/15` = 96/day nominal; measured GitHub median gap 246 min, about 6/day; the worst case uses 96) and 2 per deploy smoke (tag URL + main URL). Cloud Run's startupProbe is TCP; the Dockerfile HEALTHCHECK is in-container only. Liveness/keep-warm uses `/ping`, which makes no LLM call | `.github/workflows/`, `deploy/service.yaml` |
| `/canary/status` | public and unauthenticated; pass TTL 90 min (`CACHE_TTL_S`). **Nothing calls it on a schedule today** (not the deploy smokes, not any workflow or script; UptimeRobot was declined). `MAX_RUNS_PER_DAY = 3` bounds the worst case if someone hammers it: 3 runs x 2,150 = 6,450 tokens/day. After the 3rd run the last result is served as is for one TTL (90 min after a pass, 5 min after a fail), flagged `X-Canary-Stale: true` until 2 x TTL (3 h / 10 min), then `503 {"status":"stale"}` until UTC midnight (per instance; a restart resets the counter). So it is **not usable as a continuous monitor** in this configuration | `api/routes/canary.py` |
| WS canary | `canary.yml`, once a day, 11 turns = 2 outfit (the "sangeet outfit" query and the Sangeet chip) + 9 search; failed queries retried once, at most `MAX_RETRIES_PER_RUN` per run | `scripts/ops/canary.py` |
| Deploy smoke | 1 chat (a search-type turn) + `/readyz` on the tag and the main URL (2 probes x 100) | DEPLOY.md |

## Rule and the worst case

**All SCHEDULED checks combined must stay <= 25% of the primary model's tokens per day.** The
200,000 TPD figure is **UNVERIFIED** (see above), so the ceiling is 50,000 tokens/day *if* that
figure holds. Worst case = 3 deploys/day, `/readyz` cron at its nominal 96/day, the WS canary's
single run per day, which the daily cap enforces (manual re-runs the same day are skipped).

The table is produced by a committed script, not by hand, and gated by a test:

```
python scripts/ops/canary_budget.py          # prints the tables below
pytest tests/test_canary_budget.py           # worst case with MAX_RETRIES_PER_RUN <= 50,000
```

Output, with `MAX_RETRIES_PER_RUN = 2` (search turn 2,150 = variant C):

| Check | Tokens/day | Assumption |
|---|---:|---|
| WS canary, no retry | 22,130 | 2 outfit x 1,390 + 9 search x 2,150 |
| deploy smokes | 7,050 | 3/day x (1 chat 2,150 + 2 readyz x 100) |
| `/canary/status` | 6,450 | 3 runs/day (`MAX_RUNS_PER_DAY`) x 2,150; worst case if hammered, no scheduled caller |
| `/readyz` uptime-alert | 9,600 | 96/day x 100 |
| **Total, no retry** | **45,230** (22.6% of TPD) | |
| Total, <= 2 failed queries retried | 49,530 (24.8%) | 2 x 2,150; fits 50,000 |
| Total, whole run retried (all 11 fail) | 67,360 (33.7%) | 2 x 22,130; **EXCEEDS** 50,000 |

Same arithmetic with today's reranker (search turn 3,345): WS canary 32,885, deploy smokes
10,635, status 10,035, readyz 9,600 = **63,155 (31.6%) with zero retries, already over 25%**; 2
retries 69,845 (34.9%); whole run retried 96,040 (48.0%). So the 25% rule is met only **after
reranker variant C (PR #136) is deployed**; until then the scheduled stack is over the rule even
without retries (`tests/test_canary_budget.py` pins that fact so it is not forgotten).

### Why the retry is capped at 2 queries per run

A retry of the WHOLE run when everything fails would cost 2 x 22,130 = 44,260 for the canary alone
and 67,360 in total: over the limit. So retries are per query, failed queries only, once each, and
capped: 22,130 + 23,100 (everything else) leaves 4,770 tokens, which fits floor(4,770 / 2,150) = 2
retried queries (49,530); a third (51,680) would exceed it. A retried query is priced as a search
turn because the cap cannot know which query will fail and a search turn is the more expensive
kind. Alternative bound, stated for completeness: priced at the cheaper outfit turn (1,390) three
retries would fit (49,400), but a failing query is not known to be an outfit one, so the search
price is used. The cap is `MAX_RETRIES_PER_RUN = 2` in `scripts/ops/canary.py`; the daily cap
charges each run `11 + 2` queries (`RUN_QUERY_BUDGET`), and `DAILY_QUERY_CAP` equals one run (13), which is what this budget assumes: a manual dispatch or re-run on the same UTC day is skipped (exit 0, status `skip`, not a failure) and costs nothing. A `force_fail` dispatch (alert-path proof, no retries) may exceed the cap by one run. `tests/test_canary_budget.py` ties `DAILY_QUERY_CAP` to `WS_RUNS_PER_DAY x (11 + MAX_RETRIES_PER_RUN)`.

## Per minute, demo limit, time

- Per minute (8,000 TPM, from the 2026-11-13 header measurement): the canary's gpt-oss-20b calls run
  in about 5-8 s and use about 2,680 tokens per query (n=3 queries, about 8,000 tokens in one run,
  read-only prod `llm_call` report 2026-11-26) = about 100% of the 8,000 TPM limit when queries run
  back to back. Target <= 50% = 4,000 tokens/min, so at most one query per 60 s window:
  `CANARY_QUERY_SPACING_S = 65` s between query starts (was 30 s = 2 turns/min = 5,360 tokens/min =
  67%). At 65 s: 1 turn/min = about 2,680 tokens = 33.5% of the limit (about 2,150 = 27% with
  variant C). A real chat in the same minute can still push a minute over it.
- Demo per-IP limit (35/hour, VERIFIED in `api/demo/guards.py`, one count per WS connection): at most
  13 per run (11 + 2 retries) and one run a day, so it cannot be reached by the canary alone.
- Time: pacing alone is 12 x 65 s = 780 s = 13 min for 13 queries (11 + 2 retries), plus turn time
  (warm turns < 8 s, a cold first turn up to 120 s): 780 + 13 x 8 + 120 = 1,004 s worst nominal, under
  the 1,080 s point where the script stops starting queries (fail closed) and the 30 min job
  timeout. The sleep is outside every latency and `cold_start_ms` measurement. Spacing is defence
  in depth: the canary-day fallbacks measured since 2026-11-21 come from the old 150-token reranker
  cap (replaced by `RERANK_MAX_TOKENS` 700 in #136), not from TPM.

## Cloud Scheduler (the only billable line seen)

| Item | Value | Status |
|---|---|---|
| Job | `asa-keepwarm-healthz`, `GET` every 4 minutes (360 invocations/day); target `/openapi.json` today, `/ping` after the steps in `docs/ops/api-docs-switch.md` | |
| Cost | about $0.10/month | Google's Cloud Scheduler pricing page: free allowance 3 jobs per billing account, then $0.10 per job per month. **UNCONFIRMED** here (not re-read in this change) |
| Why it is billable | the billing account already has 4 jobs, so this one is above the free 3 | as stated by GG, not re-counted here |
| Approval | GG-approved exception, 2026-11-21 | |
| Other billable lines | none seen: the only billable line seen in the cost review this budget was built from | not independently verified in this change |

## Not included

Real user chat, the cost guard's own accounting, retries on 429/5xx inside one turn (a failing run
retries each model once, so HTTP attempts exceed logical calls; whether 429s count against RPD is
not documented), and the output p50 possibly already containing reasoning tokens (NOT MEASURED).
To refine: sum `llm_call` log events (`tokens_in`/`tokens_out`, user `anon:canary-internal`) per day
from Cloud Logging after a week of operation, and read the real TPD at
https://console.groq.com/settings/limits.
