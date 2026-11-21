# Stage-2 decision rule applied (pre-registered, README)

Primary population: 53 queries (pool > 5 candidates), rep 1, consensus labels (LLM labels, not human ground truth). Admissible: 95% lower bound of (variant - best) >= -0.03. Cheapest admissible wins (tokens per turn, then latency p50).

best by nDCG@5 point estimate: **C**; chosen: **C**

| variant | nDCG@5 | n | diff vs best [95% CI] | admissible | tokens/turn | latency p50 ms |
|---|---|---|---|---|---|---|
| A | 0.661 | 49 | -0.203 [-0.305, -0.110] | False | 1862 | 1098 |
| A2 | 0.847 | 49 | -0.017 [-0.072, +0.034] | False | 1947 | 1120 |
| B | 0.631 | 49 | -0.233 [-0.328, -0.145] | False | 0 | 0 |
| C | 0.864 | 49 | (best) | True | 752 | 951 |
