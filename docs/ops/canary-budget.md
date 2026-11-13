# Canary LLM budget vs Groq free tier

Arithmetic reproduced by a throwaway script (not committed); inputs are listed so it can be
redone. Everything labelled ESTIMATE is not measured.

## Limits compared against (UNVERIFIED for the free plan)

Source: https://console.groq.com/docs/rate-limits, fetched 2026-11-13. The page did not say whether
its table is the Free or Developer tab, so treat as UNVERIFIED for the free plan. For each of
`openai/gpt-oss-20b` (primary), `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` (the chain in
`config.yaml`): 30 RPM, 1,000 RPD, 8,000 TPM, 200,000 TPD. Each model has its own quota; the
primary takes the traffic and a fallback only sees traffic after the primary 429s. **Real user chat
shares the same org limits** (and `DEMO_DAILY_REQUEST_CAP` is 150 conversations/day).

## Inputs

| Input | Value | Source |
|---|---|---|
| LLM calls per `/chat` turn ("black kurta set for haldi") | 2 (measured) | DEPLOY.md "LLM calls per deploy", `scripts/ops/count_llm_calls.py` |
| Tokens per conversation / calls per conversation | 2,922 / 2.07 = 1,412 tokens per call | DEPLOY.md quota section: 31 real `llm_call` events, 15 conversations, 2026-07-30/31; this was `llama-3.1-8b-instant` traffic, gpt-oss tokens per call may differ (reasoning) |
| `/readyz` LLM probe | 1 call per uncached hit, cache 60 s (`READYZ_LLM_CACHE_SECONDS`), tokens ESTIMATE 100 (prompt "ping", `max_tokens=1`; only the call count is from code) | `api/routes/health.py`, `src/llm/client.py` `probe()` |
| Callers of `/readyz` in this repo | `uptime-alert.yml` (cron `*/15`: nominal 96/day, measured GitHub median gap 246 min = about 6/day) and 2 per deploy smoke (tag URL + main URL). Dockerfile HEALTHCHECK uses `/healthz`; Cloud Run startupProbe is a TCP probe. The external monitor must NOT hit `/readyz` | `.github/workflows/`, `Dockerfile`, `deploy/service.yaml` |
| Canary query cost | q2/q3-style search about 2 calls (q2 measured, q3 assumed alike). q1 "sangeet outfit" is an outfit turn: router + reranker + respond + one rationale per look, ESTIMATE up to 6 calls (static ceiling from DEPLOY.md, not measured) | DEPLOY.md |
| Deploy smoke | 1 chat (2 calls) + `/readyz` on tag and main URL (2 probes) = 4 calls (DEPLOY.md says 4 once W4 landed; the task brief said ~3) | DEPLOY.md |

## Daily load on the primary model

| Scenario | Calls/day | RPD (of 1,000) | Tokens/day | TPD (of 200,000) |
|---|---|---|---|---|
| A nominal: 48 fresh `/canary/status` runs x 2 calls, 96 probes, WS canary 3 queries, 1 deploy | 202 | 20.2% | 156,606 | 78.3% |
| B as A but GitHub's measured cadence (6 probes) | 112 | 11.2% | 147,606 | 73.8% |
| C pessimistic: 48 runs x 6 calls (ESTIMATE ceiling) | 394 | 39.4% | 427,632 | 213.8% |
| D cap-bound worst: 60 runs (the daily cap) x 2 calls, WS cap 12 queries, 3 deploys | 252 | 25.2% | 221,939 | 111.0% |
| E cap-bound worst, pessimistic: 60 runs x 6 calls | 492 | 49.2% | 560,722 | 280.4% |

Breakdown of A: status runs 96 calls / 135,513 tokens; probes 96 / 9,600; WS canary 6 / 8,470;
deploy 4 / 3,023.

## Reading it

- Requests are not the constraint (20% of RPD nominal, 49% worst). **Tokens are**: at the measured
  1,412 tokens per call the 48 runs/day alone use about 68% of the primary's 200K TPD, and the
  whole canary stack about 78%, leaving roughly 14 real conversations/day of primary TPD
  ((200,000 - 156,606) / 2,922). Scenarios C, D and E exceed 100% of the primary's TPD, so the
  chain would overflow to `gpt-oss-120b`/`qwen3.8-27b` (their own 200K TPD each) and real users
  share that overflow.
- Per minute: one 2-call run is about 2.8K tokens (of 8K TPM). A 6-call outfit run (about 8.5K
  tokens, ESTIMATE) can by itself reach the 8K TPM limit, so q1 may 429-retry; a real chat in the
  same minute adds about 2.9K.
- The pass TTL of 30 min is the lever: 60 min halves A's status-run cost (about 68 -> 34% of TPD).
  Not changed here; the architect fixed 30 min.
- Failures: fail TTL 5 min would allow 288 runs/day, so the 60-run/day cap (D/E) is what binds;
  past it `/canary/status` serves the last known result with `X-Canary-Stale: true` and spends
  nothing. A failing run retries each model once, so HTTP attempts exceed logical calls (whether
  429s count against RPD is not documented; UNVERIFIED).
- Not included: real user chat, the cost guard's own accounting, retries on 429/5xx.
- To measure instead of estimate: sum `llm_call` log events (`tokens_in`/`tokens_out`, user
  `anon:canary-internal`) per day from Cloud Logging after a week of operation.
