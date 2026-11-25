# Plain-search quality PR: gate rule (pre-registered 2026-11-25, before any result was computed)

Stacked on PR #136 (production reranker = stage-2 variant C). This file is committed before the
post-processing code runs on any eval pool or any live chip. Numbers that exist when it is
written: only catalogue-structure counts used to choose the de-duplication key (see the PR body).

## What is being gated

`search_node` post-processing applied to the reranker's ranked list (model picks first, then the
rest of the candidate pool in retrieval order):

1. de-duplicate (same store + same product; key defined in `src/agents/search_postprocess.py`),
2. at most 2 cards per store,
3. budget guard (`price <= X` for an "under X" query; items with no price are not shown),
4. gender-unspecified queries only: women/men interleave, each gender in its own ranked order.

Selection is greedy in ranked order and backfills from deeper candidates, so a result that has no
duplicate, no third card from a store, no over-budget item and a gendered query is returned
unchanged (no-op regression, unit-tested).

## (i) Offline eval guard

- Pools: the 53 stage-2 query pools (`evals/retrieval/candidates_master/q*.json` on
  feat/v4-retrieval-eval), consensus labels = the gate's `load_consensus` (strict majority of
  gpt-oss-120b / qwen3.8-27b / claude, else median), `scripts/rerank_gate.py` metric definitions.
- Reranker replies: the stage-2 eval's variant C rep-1 cache replayed through the production
  `rerank()` (`ReplayEvalCache`; byte-identical prompts), so the run is deterministic and makes
  zero live calls.
- Current arm: the production `rerank()` top-5, unprocessed. New arm: the post-processed top-5
  from the same reply. The new arm uses only the pool and the query text (parse_intent for the
  budget, the production gender-extraction for "gender specified?").
- Metric: nDCG@5 against the ideal ordering of all labelled candidates of the query. A shown item
  without a consensus label scores 0 in BOTH arms (conservative; the gate's convention of dropping
  it would flatter shorter lists); the count of unlabelled shown items per arm is reported.
- Queries: every one of the 53 with a defined nDCG@5 (ideal DCG > 0) for both arms. n and the
  excluded query ids are reported (expected about 49).
- Statistic: paired bootstrap over queries, 10,000 resamples, seed 42, percentile 95% CI of
  mean(new - current).
- PASS if the lower bound of (new - current) >= -0.03. Otherwise the guard FAILS and is reported
  as failed: the PR does not weaken the threshold, the metric or the pools after the fact.
- Interleave note: interleave changes the result only for gender-unspecified queries, so the
  overall mean dilutes it. A secondary line reports (new - current) on the gender-unspecified
  subset alone (n reported, same bootstrap); it is informational, the gate is the overall line.
- Also reported (not gated): duplicate rate (share of queries with >= 1 duplicate pair, and
  duplicate pairs per query) in the unprocessed vs processed top-5 on the 53 pools and on the 8
  chip pools (`evals/fixtures/rerank_chip_pools`); max cards per store; price violations on every
  query with a parsed budget; mean shown count (post-processing may show fewer than 5).

## (ii) Live gate (local backend, real index, live LLM within the shared-key caps)

Run after the day's WS canary, gpt-oss-20b only (no 120b/qwen path), at most 60 live chat calls
for the task, prod logs checked for fallback/429 before and during.

The 5 landing chips (`OCCASION_CHIPS`), each run twice (10 turns), every card printed
(title, store, price, gender). Per turn:

- zero duplicate products (same-store pairs under the module's key);
- at most 2 cards per store;
- both genders present on the gender-neutral chips (a chip whose text and session carry no
  gender: 'Haldi kurta set', 'Office wear under ₹1,500', 'Sangeet outfit under ₹3,000' if it
  reaches plain search; a chip that routes to the outfit composer is reported as such, not
  counted);
- price <= budget for the budget chip(s) that reach plain search.

The criteria are reported per turn, pass/fail, and any miss is reported as a miss (no rerun
until green). A shown count below 5 is expected where the pool is narrow (documented edge) and
is not a criterion miss by itself; it is reported with the logged reason.
