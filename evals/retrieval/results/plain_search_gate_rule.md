# Plain-search quality PR: gate rule v2 (pre-registered 2026-11-26, before any cap-3 run)

This file REPLACES the 2026-11-25 rule committed with the first draft (PR #139). Honest history:
the first rule (per-store cap 2, bar = lower bound of the paired 95% interval of
mean(new - unprocessed reranker top-5) >= -0.03) was run and NOT met (lower bound -0.112, see the
history of PR #139). The architect then decided the shipping rule below. This file is committed
before the guard is run with `MAX_PER_STORE = 3` on anything. Numbers that exist when it is
written: the cap-2 results above, and catalogue-structure counts used to choose the de-duplication
key. Nothing has been computed with cap 3.

## Architect decision (2026-11-26)

(a) de-duplicate by store + product key ALWAYS; (b) per-store cap 3 (`MAX_PER_STORE = 3`);
(c) gender interleave when the query fixes no gender, plus the approved Women / Men refine control
('Women' / 'Men'); (d) the ship rule below.

## The ship rule (a point-estimate rule)

SHIP if and only if

    mean over queries of ( nDCG@5[no-cap arm] - nDCG@5[cap-3 arm] )  <=  0.03

as a POINT ESTIMATE (nDCG@5 drop of the cap-3 arm vs the no-cap arm, not larger than 0.03).
This supersedes the earlier "lower bound of the CI >= -0.03" bar, which is withdrawn. The paired
bootstrap 95% interval of the same difference is REPORTED (not gated). If the rule is not met it is
reported as not met: the rule, labels, pools, arms and seed are not changed afterwards.

## The exact comparison

- no-cap arm: `search_postprocess.select_results` with de-duplication ON, gender interleave ON for
  gender-unspecified queries (`mix_window = ASK_N`, the production setting), per-store cap OFF
  (`max_per_store = 99`), no budget guard.
- cap-3 arm: the same call with `max_per_store = 3` (the shipped constant). Nothing else differs.
- Both arms start from the same production `rerank_ranked()` output (model picks, then the rest of
  the pool in retrieval order) and select the top 5.
- Budget guard is a hard filter, not a ranking choice; it is outside this comparison. Its effect is
  reported separately (informational ablation, not gated).
- Informational lines (not gated): each arm vs the unprocessed reranker top-5; the gender-unspecified
  subset alone; duplicate / store-cap / over-budget counts; mean shown count; ablation arms.

## Labels, pools, statistic

- Pools: the stage-2 query pools (`evals/retrieval/candidates_master/q*.json` of the stage-2 retrieval
  eval); 8 chip pools (`evals/fixtures/rerank_chip_pools`) are reported structurally only.
- Labels: stage-2 consensus labels (`scripts/rerank_gate.py` `load_consensus`: strict majority of
  gpt-oss-120b / qwen3.8-27b / claude, else median), RELEVANCE ONLY. No label is revised.
- Reranker replies: the stage-2 variant C rep-1 cache replayed through the production `rerank()`
  (`ReplayEvalCache`). Deterministic, zero network, zero LLM calls.
- Metric: nDCG@5 against the ideal ordering of all labelled candidates of the query; an unlabelled
  shown item scores 0 in both arms (its count is reported per arm).
- Queries: every query with a defined nDCG@5 (ideal DCG > 0) in both arms. n and the excluded
  query ids are reported.
- Interval: paired bootstrap over queries, 10,000 resamples, seed 42, percentile 95% CI of
  mean(cap-3 - no-cap). The point estimate is the gate.

## Live 5-chip check (informational, not part of the ship rule)

The five landing-chip queries through the real graph, gpt-oss-20b only (`allow_fallback=False`, no
120b / qwen), at most 10 LLM calls and 15K tokens in total, run after the UTC day's canary and with
no fallback / 429 in the prod logs of the previous 15 minutes. Per chip: query, top-5 titles, store,
gender, price, and counts of de-duplication drops, per-store-cap drops and refill. Reported as
observed; a miss is a miss.
