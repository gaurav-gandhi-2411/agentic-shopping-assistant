# Reranker completion cap: decision rule (pre-registered 2026-11-25, before the change)

Context: live gate 2026-11-25 (reports/rerank_gate/gate_live.json, head 42e29d6) FAILED the
nDCG bar: production 0.8612 vs eval C 0.8644, diff -0.0032, 95% CI [-0.0380, +0.0305], bar is
lower bound >= -0.03. Failure rate 1/53 = 1.9% (bar 5%) passed. The one failure is q15, outcome
`length` (671 prompt + 400 completion tokens, 385 reasoning), which fell back to retrieval order.
The cap 400 was set from the eval's 53 completions (max 316).

## Offline diagnosis (n = 49 queries with defined nDCG@5, consensus labels, gate's own functions)

- 11 of 49 queries differ between production and eval C; per-query diffs sum to -0.1576.
- q15 alone: production 0.5478 vs eval C 1.0000, diff -0.4522 (outcome `length`).
- Counterfactual (q15 set to eval C's value, paired bootstrap 10,000 resamples, seed 42):
  mean +0.0060, 95% CI [-0.0234, +0.0355].
- Caveat: other queries also moved with outcome `ok` (q50 -0.281, q51 -0.369, q25 -0.152,
  against q53 +0.387, q26/q55/q04 about +0.21), i.e. run-to-run noise in both directions that the
  CI already accounts for. The counterfactual is evidence about q15's weight, not a re-run.

## Rule (fixed before the change)

1. If the counterfactual lower bound >= -0.03, the cap was too small: raise `RERANK_MAX_TOKENS`
   from 400 to 700. Reasoning tokens consume the completion cap under reasoning_effort=low; 700
   is a worst-case bound (typical cost unchanged, mean completion about 140) and covers the
   reasoning tail seen in stage 2 (variant A2 reasoning max 598).
2. If it still fails, do NOT change the cap; report the per-query diffs and stop.
3. Nothing else changes (filters, prompt text, parser).

## Outcome

Counterfactual lower bound -0.0234 >= -0.03: rule 1 applies. Cap 400 -> 700. This is not a
passing gate: the live re-gate (gpt-oss-20b only, after the 2026-11-26 canary) decides.
