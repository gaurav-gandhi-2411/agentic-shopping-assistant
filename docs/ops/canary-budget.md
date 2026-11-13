# Canary LLM budget vs Groq limits

Arithmetic reproduced by a throwaway script (not committed); inputs are listed so it can be
redone. Everything labelled ESTIMATE or UNVERIFIED is not measured.

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
| Tokens per LLM call | p50 input 1,219 + p50 output 124 = 1,343 (p90: 1,905 in, 153 out) | MEASURED: 22 `llm_call` records in the prod Cloud Run logs, 2026-11-12 18:47Z to 2026-11-13 08:47Z, from the canary, WS canary and deploy-smoke chat turns. Models: `gpt-oss-20b` 21 calls, `gpt-oss-120b` 1 call |
| LLM calls per chat turn | 2 (max 2 in the 8 conversations seen) | same 22 records |
| Tokens per chat turn | p50 3,262, p90 3,399, max 3,402 (used below as the per-turn cost) | same 22 records |
| Reasoning tokens | NOT MEASURED in prod turns: the app logs `output_tokens` only, not a reasoning split. One direct probe with `max_tokens` 64 on `gpt-oss-20b` returned `reasoning_tokens` 17 of 35 completion tokens: a single tiny sample, not a p50 | direct probe 2026-11-13 |
| `/readyz` LLM probe | 1 call per uncached hit, cache 60 s (`READYZ_LLM_CACHE_SECONDS`), tokens ESTIMATE 100 per probe (prompt "ping", `max_tokens=1`; only the call count is from code) | `api/routes/health.py`, `src/llm/client.py` `probe()` |
| Callers of `/readyz` in this repo | `uptime-alert.yml` (cron `*/15`: nominal 96/day, measured GitHub median gap 246 min = about 6/day) and 2 per deploy smoke (tag URL + main URL). Dockerfile HEALTHCHECK uses `/healthz`; Cloud Run startupProbe is a TCP probe. The external monitor must NOT hit `/readyz` | `.github/workflows/`, `Dockerfile`, `deploy/service.yaml` |
| `/canary/status` runs per day | 24 nominal: pass TTL is 60 min, one query per run, 2 calls and 3,262 tokens per turn. Cap-bound worst case 60/day (`MAX_RUNS_PER_DAY`) | `api/routes/canary.py` |
| Canary query cost | q2/q3-style search is 2 calls (measured above). q1 "sangeet outfit" is an outfit turn: router + reranker + respond + one rationale per look, ESTIMATE up to 6 calls (static ceiling from DEPLOY.md, not measured); priced in the pessimistic rows at 6 x 1,343 = 8,058 tokens | DEPLOY.md |
| WS canary (`canary.yml`) | once a day x 3 queries (12 queries in the cap-bound rows) | `.github/workflows/canary.yml` |
| Deploy smoke | 1 chat (2 calls, 3,262 tokens) + `/readyz` on tag and main URL (2 probes, 200 tokens) = 4 calls, 3,462 tokens | DEPLOY.md |

## Daily load on the primary model

TPD column is against the UNVERIFIED 200,000; the RPD column is against the VERIFIED 1,000.

| Scenario | Calls/day | RPD (of 1,000) | Tokens/day | TPD (of 200,000, UNVERIFIED) | Real-user headroom (conversations/day) |
|---|---|---|---|---|---|
| A nominal: 24 status runs x 2 calls, 96 probes, WS canary 3 queries, 1 deploy | 154 | 15.4% | 101,136 | 50.6% | 30 |
| B as A but GitHub's measured cadence (6 probes) | 64 | 6.4% | 92,136 | 46.1% | 33 |
| C pessimistic: 24 runs x 6 calls (ESTIMATE ceiling, 8,058 tokens per run) | 250 | 25.0% | 216,240 | 108.1% | 0 (overflows to the fallbacks) |
| D cap-bound worst: 60 runs (the daily cap) x 2 calls, WS 12 queries, 3 deploys | 252 | 25.2% | 254,850 | 127.4% | 0 (overflows) |
| E cap-bound worst, pessimistic: 60 runs x 6 calls | 492 | 49.2% | 542,610 | 271.3% | 0 (overflows) |

Headroom = (200,000 - tokens/day) / 3,262 tokens per conversation, rounded down, floored at 0.

Breakdown (calls / tokens): A status runs 48 / 78,288, probes 96 / 9,600, WS canary 6 / 9,786,
deploy 4 / 3,462. B differs only in probes 6 / 600. C status runs 144 / 193,392. D status runs
120 / 195,720, WS 24 / 39,144, deploys 12 / 10,386. E status runs 360 / 483,480 with D's other
lines.

## Reading it

- Requests are not the constraint (15% of the verified 1,000 RPD nominal, 49% worst). **Tokens are,
  if the 200K TPD figure holds** (UNVERIFIED): nominal, the 24 runs/day use about 39% of the
  primary's TPD and the whole canary stack about 51%, leaving roughly 30 real conversations/day.
  C, D and E exceed 100%, so the chain would overflow to `gpt-oss-120b`/`qwen3.8-27b` (their own
  quotas) and real users share that overflow.
- Pass TTL 30 -> 60 min halves the status runs (48 -> 24/day: 156,576 -> 78,288 tokens, about 78%
  -> 39% of a 200K TPD) and lifts real-user headroom from about 6 (whole stack 179,424 tokens/day,
  89.7% of TPD) to about 30 conversations/day. The 30 min figures are recomputed with the same
  measured 3,262 tokens per turn, not taken from the previous version of this document.
- Per minute (VERIFIED 8,000 TPM): one 2-call turn is 3,262 tokens, 41% of the limit; a real chat
  in the same minute adds another 3,262. A 6-call outfit run (about 8,058 tokens, ESTIMATE) alone
  reaches the 8,000 TPM limit, so q1 may 429-retry.
- Failures: a fail TTL of 5 min would allow 288 runs/day, so the 60-run/day cap (D/E) is what
  binds. Staleness of `/canary/status`: while the last result is at most 2 x its TTL old (2 h after
  a pass, 10 min after a fail) and no refresh can run, it is served with `X-Canary-Stale: true`; older
  than that, or with no result on a fresh instance and no run possible, it answers
  `503 {"status":"stale"}`. A refresh is always attempted first under the daily cap, so a stale
  state costs at most the cap, never more. A failing run retries each model once, so HTTP attempts
  exceed logical calls (whether 429s count against RPD is not documented; UNVERIFIED).
- Not included: real user chat, the cost guard's own accounting, retries on 429/5xx, the 124-token
  output p50 possibly already containing reasoning tokens (NOT MEASURED; only the one probe above).
- To refine: sum `llm_call` log events (`tokens_in`/`tokens_out`, user `anon:canary-internal`) per
  day from Cloud Logging after a week of operation, and read the real TPD at
  https://console.groq.com/settings/limits.
