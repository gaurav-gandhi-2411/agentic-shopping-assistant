# StyleMaitri (stylemaitri.vercel.app) — Performance Baseline

Measured 2026-07-12, Lighthouse 13.4.0 CLI, desktop preset, headless Chrome, cold load.
Raw artifacts: `lighthouse-stylemaitri-baseline.report.json`, `lighthouse-stylemaitri-baseline.report.html` (same directory).

## Core Web Vitals (observed, from trace — not Lantern-simulated)

| Metric | Value | Rating |
|---|---|---|
| Performance score | 0.94 | — |
| LCP | 812 ms | Good (<2.5s) |
| TBT | 2.5 ms | Good (<200ms) |
| CLS | 0 | Good (<0.1) |
| FCP | 632 ms | Good (<1.8s) |
| Speed Index | 2386 ms | Good (<3.4s) |

## Payload

| | |
|---|---|
| Total page weight | 369.1 KB |
| Total JS transferred | 269.5 KB |

Top scripts:
- `48-0146606416833393.js` — 117.4 KB
- `456-ee8f26eeae652a6b.js` — 56.9 KB
- `4bd1b696-d40b6e984aaba1a0.js` — 54.6 KB

## Proposed rule-15e budgets (pending ratification)

| Budget | Proposed | Basis |
|---|---|---|
| Initial JS bundle | ≤ 350 KB | current 269.5 KB + ~30% headroom |
| LCP | ≤ 1500 ms | current 812 ms; headroom for real-world/mobile variance |
| TBT | ≤ 200 ms | Lighthouse "good" threshold; current 2.5 ms has large headroom already |
| CLS | ≤ 0.05 | tighter than the 0.1 "good" threshold; current is 0 |
