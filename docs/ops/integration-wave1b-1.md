# Integration log: release/wave1b-1

Base: `origin/master` ad17f47. Five component branches merged with `git merge --no-ff`, in the
architect's order. Every merge completed with **zero textual conflicts** (git's ort strategy, no
conflict markers in any merge), so there were no hunks to resolve by hand.

| # | Merge commit | PR | Head merged |
|---|---|---|---|
| 1 | 3b422d3 | #136 reranker variant C | 6f6fb44 |
| 2 | ba56150 | #141 joint outfit slot selection | f4458b5 |
| 3 | 67b9445 | #138 canary + /ping v2 | d7a524b |
| 4 | fe50b06 | #142 fetcher pagination + 429 | 9bec606 |
| 5 | f5e756b | #143 incremental index builder (stacked on #142) | 1979e72 |

## Conflict log

| File | Hunk | Resolution | Why |
|---|---|---|---|
| (none) | - | - | no file was edited by two of the branches except the stacked pair below |

Files touched by more than one branch, found with `git diff --name-only ad17f47...origin/<branch>`
for each branch: only #142 and #143 overlap, and only because #143 is stacked on #142
(`scripts/ops/fetch_pass.py`, `scripts/ops/refresh_feed.py`, `scripts/ops/feed_delta_funnel.py`,
`tests/test_fetch_pass.py`, `tests/test_refresh_feed.py`, `tests/test_feed_delta_funnel.py`,
`reports/fetch/fetch_pass_2026-11-26.md`, `reports/fetch/delta_explanation.md`). Merging #142 first
made #143's copy of those files an identical ancestor, so git took them without a conflict.

## Interactions reviewed (semantic, not textual)

| Pair | Surface | Finding |
|---|---|---|
| #136 x #141 | `src/agents/reranker.py`, `rerank_filters.py` (#136) vs `src/agents/outfit/joint.py`, `composer.py` (#141) | disjoint files; `joint.py` does not import the reranker, so `RERANK_MAX_TOKENS` 700 does not change joint selection |
| #136 x #138 | reranker call vs canary / `canary_budget.py` | `canary-budget.md` still labels variant C a projection ("not merged yet"); it is merged on this branch but not deployed, so the label stays accurate until deploy |
| #138 x #141 | canary Sangeet outfit queries vs joint selection | no shared code; the canary asserts card counts on the live API, which only a deploy can exercise (NOT verified here) |
| #138 internal | `/ping` (`api/routes/health.py`) and the access-log skip in `api/main.py` | one branch only (#138), no overlap |
| README / docs rows | none of the five branches edited the root README.md (only `docs/launch-gates.md`, new in #138) | nothing to reconcile |
| #142 x #143 | stacked | clean, see above |

## Extra commit after the merges

`fix(canary): space query starts 65 s apart ...` sets `CANARY_QUERY_SPACING_S = 65` (was
`QUERY_SPACING_S = 30`) in `scripts/ops/canary.py`. Premise correction recorded here: after the
#138 merge the canary runs 11 turns (3 core + 8 card checks, plus up to 2 retries), not 3. The
arithmetic in `docs/ops/canary-budget.md` states both (about 2,680 tokens/query measured on the 3
core queries; 8,000 TPM from the 2026-11-13 header measurement). Nominal worst run time is
12 x 65 s + 12 x 8 s + 120 s = 996 s, below `RUN_TIME_BUDGET_S` 1,080 s and the 30 min job timeout.
