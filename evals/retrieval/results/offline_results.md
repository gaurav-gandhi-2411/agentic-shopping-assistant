# Stage-2 offline results (no LLM calls)

Labels: LLM-consensus (3 families), not human ground truth; 961 items, 11 tied (median-resolved). Metrics are bootstrap means with 95% CIs (10,000 resamples, seed 42) over queries.
Excluded, no candidates: q46, q56, q59. Excluded from the primary comparison, pool <= 5 candidates: q11, q13, q16, q23.

## primary_pool_gt_5 (n = 53 queries)

n per metric: P@5 53; nDCG@5 49 (a query whose pool holds no label >= 1 has an ideal DCG of 0, so nDCG@5 is undefined and dropped from means and pairs).

| variant | nDCG@5 | P@5 (>=1) | P@5 (>=1.5) | P@5 (>=2) | nDCG@5 excl. ties |
|---|---|---|---|---|---|
| A | 0.661 [0.559, 0.758] | 0.642 [0.532, 0.747] | 0.540 [0.430, 0.645] | 0.540 [0.430, 0.645] | 0.648 [0.543, 0.748] |
| B | 0.631 [0.533, 0.726] | 0.615 [0.509, 0.717] | 0.506 [0.404, 0.608] | 0.506 [0.404, 0.608] | 0.618 [0.518, 0.716] |

Paired A - B (same queries):

- ndcg5: +0.030 [+0.008, +0.055] (n=49, sign p=0.065, wilcoxon p=0.023)
- p5_ge1: +0.026 [+0.008, +0.049] (n=53, sign p=0.031, wilcoxon p=0.026)
- p5_ge2: +0.034 [+0.004, +0.072] (n=53, sign p=0.289, wilcoxon p=0.062)
- p5_ge1_5: +0.034 [+0.004, +0.072] (n=53, sign p=0.289, wilcoxon p=0.062)

A usage (live calls n=53, skipped 0): mean prompt 1724, completion 138, latency p50 1098 ms, p95 1595 ms.
A outcomes: {'ok': 17, 'partial': 1, 'length': 31, 'empty_selection': 4}; no_exact_match 4; hit cap 31.

## sensitivity_all_with_candidates (n = 57 queries)

n per metric: P@5 57; nDCG@5 53 (a query whose pool holds no label >= 1 has an ideal DCG of 0, so nDCG@5 is undefined and dropped from means and pairs).

| variant | nDCG@5 | P@5 (>=1) | P@5 (>=1.5) | P@5 (>=2) | nDCG@5 excl. ties |
|---|---|---|---|---|---|
| A | 0.672 [0.578, 0.762] | 0.628 [0.526, 0.730] | 0.530 [0.428, 0.632] | 0.530 [0.428, 0.632] | 0.659 [0.563, 0.753] |
| B | 0.644 [0.555, 0.732] | 0.604 [0.505, 0.698] | 0.498 [0.404, 0.593] | 0.498 [0.404, 0.593] | 0.632 [0.539, 0.723] |

Paired A - B (same queries):

- ndcg5: +0.028 [+0.008, +0.051] (n=53, sign p=0.065, wilcoxon p=0.023)
- p5_ge1: +0.025 [+0.007, +0.046] (n=57, sign p=0.031, wilcoxon p=0.026)
- p5_ge2: +0.032 [+0.004, +0.067] (n=57, sign p=0.289, wilcoxon p=0.062)
- p5_ge1_5: +0.032 [+0.004, +0.067] (n=57, sign p=0.289, wilcoxon p=0.062)

A usage (live calls n=53, skipped 4): mean prompt 1724, completion 138, latency p50 1098 ms, p95 1595 ms.
A outcomes: {'ok': 17, 'partial': 1, 'length': 31, 'empty_selection': 4}; no_exact_match 4; hit cap 31.

## Factors, variant B only (LLM arms need live calls)

F1 sets demoted vs not, flagged ethnic-wear queries (single-garment named on an ethnic occasion): q01 "grey indo-westen anarkali gown sangeet women <3000" (pool changed); q02 "kuta haldi men"; q07 "men's white kurta for diwali under ₹3000" (pool changed); q13 "Need women's blue fusion lehenga for sangeet night yaar, budget 5000 max"; q16 "Can you suggest lehenga for a woman to wear to a wedding? I want it to look elegant but stay comfortable. Please keep it under ₹1500." (pool changed); q27 "men's yellow kurta for mehendi" (pool changed); q28 "Can you suggest saree for a woman to wear to a wedding? I want it to look elegant but stay comfortable. Please keep it under ₹3000." (pool changed); q43 "diwali ke liye blue anarkali chahiye ladkiyon ke liye, 5000 se kam mein" (pool changed).

### F1 treated (master, sets demoted) - base (frozen, not demoted), n = 8 queries (6 with a different top 5)

| arm | nDCG@5 | P@5 (>=1) |
|---|---|---|
| base | 0.840 [0.745, 0.930] | 0.775 [0.550, 0.950] |
| treated | 0.814 [0.632, 0.963] | 0.750 [0.500, 0.950] |

- paired ndcg5: -0.026 [-0.142, +0.071] (n=8, sign p=1.000, wilcoxon p=1.000)
- paired p5_ge1: -0.025 [-0.075, +0.000] (n=8, sign p=1.000, wilcoxon p=1.000)
- pool loss: {'n_before': 95, 'n_after': 76, 'n_removed': 19, 'n_removed_relevant': 19}

### F2 treated (gender-interleaved) - base (current order), n = 12 queries (9 with a different top 5)

| arm | nDCG@5 | P@5 (>=1) |
|---|---|---|
| base | 0.494 [0.300, 0.685] | 0.500 [0.300, 0.700] |
| treated | 0.544 [0.420, 0.681] | 0.550 [0.383, 0.717] |

- paired ndcg5: +0.050 [-0.030, +0.145] (n=11, sign p=0.453, wilcoxon p=0.272)
- paired p5_ge1: +0.050 [-0.017, +0.133] (n=12, sign p=0.625, wilcoxon p=0.269)

