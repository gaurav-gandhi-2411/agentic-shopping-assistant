# Public claims audit

Base: `origin/master` ad17f47 (file:line refer to that commit, i.e. before this PR's edits).
Scope swept: `README.md`, `BRANDS.md`, `spaces/README.md`, `.portfolio/metrics.json`,
`frontend/lib/copy.ts`, `frontend/lib/site.ts`, `frontend/app/layout.tsx` metadata,
`frontend/app/opengraph-image.tsx`, `frontend/app/pdp-demo/**` (greps for digits, `%`, `K`,
"free", "accura", "stores", "F1", "fastest/best/instant" in `frontend/app` and
`frontend/components`; `DEMO.md` greps found no public metric claims). Not swept: git history,
GitHub Settings, Vercel project settings, live-site runtime strings produced by the API.
No model or eval was re-run; every verdict is from tracked artifacts and code.

Source commit SHAs are the last commit touching the artifact (`git log -1 -- <path>`):
`db4a6ed` (reports/final_scorecard_2026-07-12.txt, reports/soldout_filter_fix_2026-07-12.txt),
`b7bb2c2` (reports/performance-baseline-2026-07-12.md), `51f57f2` (config.yaml),
`a53fbfd` (frontend/package.json), `02ed107` (.portfolio/metrics.json).

Verdict counts: SUPPORTED 16, STALE 5, UNSUPPORTED 12, CONTRADICTED 5 (total 38).

| # | Claim | File:line | Source artifact (path @ SHA) | Verdict | n / interval | Action |
|---|---|---|---|---|---|---|
| 1 | Hybrid FAISS + BM25 fused via RRF | README:17 | `src/retrieval/hybrid_search.py:627` | SUPPORTED | n/a | none |
| 2 | Six tools (search, compare, filter, outfit-bundle, clarify, respond) | README:18 | `src/agents/tools.py` + graph nodes | SUPPORTED | n/a | none |
| 3 | "the agent cannot hallucinate a price, size, or fabric" | README:19 | NO SOURCE (absolute claim; only a post-hoc validator exists) | UNSUPPORTED | none | softened to "a guard, not a guarantee" |
| 4 | H&M ~20 K items | README:37, BRANDS.md | NO SOURCE (untracked dataset) | UNSUPPORTED | none | not changed, flagged |
| 5 | Sample IN 500 items | README:38, BRANDS.md | `scripts/_gen_in_feed.py:1` ("200 rows"); `data/samples/in_feed.csv` 201 lines incl. header | CONTRADICTED | 200 | edited to 200 |
| 6 | Myntra 14 K, ₹169-₹47,999 | README:39, BRANDS.md | NO SOURCE (no tracked post-2026-07-12 count) | UNSUPPORTED | none | not changed, flagged |
| 7 | Flipkart 15 K, ₹99-₹7,799 | README:40, BRANDS.md | NO SOURCE | UNSUPPORTED | none | not changed, flagged |
| 8 | Snitch 15 K | README:41, BRANDS.md | reports/soldout_filter_fix_2026-07-12.txt :16-20 @ db4a6ed: 7,040 kept | STALE | 7,040 | edited |
| 9 | Powerlook 927 | README:42, BRANDS.md | same, 963 kept | STALE | 963 | edited |
| 10 | Fashor 3.6 K | README:43, BRANDS.md | same, 2,343 kept | STALE | 2,343 | edited |
| 11 | Virgio 1.8 K | README:44, BRANDS.md | same, 1,615 kept | STALE | 1,615 | edited |
| 12 | Price ranges of Snitch/Powerlook/Fashor/Virgio | README:41-44, BRANDS.md | NO SOURCE after the sold-out re-sync (min/max may have moved) | UNSUPPORTED | none | not changed, flagged |
| 13 | "free Groq API key ... free, no credit card" | README:53,72 | config.yaml:73 @ 51f57f2 says "free-tier quota"; "no credit card" has no source | UNSUPPORTED | none | "no credit card" removed, "free tier" kept |
| 14 | "~30 s" download, "~4 min on CPU" build | README:79-80 | NO SOURCE (and stale: Snitch catalogue halved) | UNSUPPORTED | none | removed |
| 15 | "Quick Start (under 10 minutes)" | README:51 | NO SOURCE | UNSUPPORTED | none | removed |
| 16 | "Adding a Shopify brand takes under 10 minutes" | README:130 | NO SOURCE | UNSUPPORTED | none | replaced with "config-driven" |
| 17 | Next.js 15 | README:27,147,177 | frontend/package.json:23 @ a53fbfd ("15.5.28") | SUPPORTED | n/a | none |
| 18 | RRF k=60 | README:174 | config.yaml:12 @ 51f57f2 | SUPPORTED | n/a | none |
| 19 | 6-turn rolling window | README:160 | config.yaml:96 `recent_turns: 6` | SUPPORTED | n/a | none |
| 20 | all-MiniLM-L6-v2, 384-dim | README:172 | config.yaml:8 (model; 384 is that model's published dimension) | SUPPORTED | n/a | none |
| 21 | Diagram: LLM llama-3.1-8b-instant | README:158 | config.yaml:65-68: Groq deprecated it 2026-08-16; `groq_model: openai/gpt-oss-20b` | CONTRADICTED | n/a | edited |
| 22 | Tech stack: Groq llama-3.1-8b-instant | README:175 | same | CONTRADICTED | n/a | edited |
| 23 | "Groq (Llama 3.1 8B)" | spaces/README.md:25 | same | CONTRADICTED | n/a | edited |
| 24 | "106 passed, ~50 skipped, 0 errors" | README:190 | reports/soldout_filter_fix_2026-07-12.txt:33 @ db4a6ed: 1,325 passed, 0 failed, 31 skipped (with built index) | STALE | 1,356 collected | replaced with dated measured figure |
| 25 | "Eval suite (57 queries)" | README:196 | reports/eval_baseline_groq.md @ last-touch: 32 queries, 31/32 pass (April 2026); no 57-query artifact | CONTRADICTED | n=32 | replaced |
| 26 | Initial JS 269.5 KB | README:212 | reports/performance-baseline-2026-07-12.md:22 @ b7bb2c2 | SUPPORTED | single cold load, no interval | none |
| 27 | LCP 812 ms | README:213 | same:11 | SUPPORTED | single run | none |
| 28 | TBT 2.5 ms | README:214 | same:12 | SUPPORTED | single run | none |
| 29 | CLS 0 | README:215 | same:13 | SUPPORTED | single run | none |
| 30 | Python 3.11 badge | README:3 | pyproject.toml:4, Dockerfile:10 | SUPPORTED | n/a | none |
| 31 | License MIT badge/section | README:4,239 | no `LICENSE` file in repo (badge link 404s); spaces/README.md says `license: mit` | UNSUPPORTED | none | not changed, flagged (legal) |
| 32 | Cloud Run `asa-stylist-api`, project `iconic-reactor-496423-m4` | README:6 | DEPLOY.md:161,176 | SUPPORTED | n/a | none |
| 33 | "Intent-parsing accuracy 93.8% (n=211)" | .portfolio/metrics.json:9 | reports/final_scorecard_2026-07-12.txt:109 @ db4a6ed (scorecard cites bdf66a4): rule-based `parse_intent` all-exact on own fixtures, min gate 88.0% | SUPPORTED | n=211, no interval | label clarified; NOT the DistilBERT router (F1 0.9035 in-dist / OOD 0.25 per PROJECT_AUDIT.md:387) |
| 34 | "~52K items across 8 stores" | .portfolio/metrics.json:17 | reports/soldout_filter_fix_2026-07-12.txt:28,62 @ db4a6ed: 52,494; 4 Shopify + 4 static stores | SUPPORTED | 52,494 | none |
| 35 | "can take about a minute" (cold start), busy "about a minute" | frontend/lib/copy.ts:41,52 | no measured cold-start number; reports/mlops_audit_2026-07-12.md:91 only says well under 300 s | UNSUPPORTED | none | not changed (GG-reviewed copy file), flagged |
| 36 | Taglines, "No sign-in needed", chips (no numbers) | frontend/lib/copy.ts:10-36 | code behaviour; copy.ts header forbids counts/claims | SUPPORTED | n/a | none |
| 37 | Site/OG description "your AI stylist for fashion discovery / weddings, sangeets" | frontend/app/layout.tsx:25-36, opengraph-image.tsx:13 | qualitative, no figures | SUPPORTED | n/a | none |
| 38 | "42% off", "52% off", "Free shipping on orders over ₹999", "loads instantly" | frontend/app/pdp-demo/**:50,71,90,109,318, page.tsx:103 | NO SOURCE; mock merchant page, publicly reachable (`publicPaths.ts:22`) | UNSUPPORTED | none | not changed, flagged |

Note on the known item: the string "93.8% intent accuracy" is not in README.md at ad17f47; it
exists only as the `.portfolio/metrics.json` entry (row 33), which feeds the portfolio site. The
README contains no DistilBERT F1 (0.9858 appears nowhere in tracked files).

## Items needing GG (not changed here)

- GitHub repo description and topics: Settings item, not editable in files. Check they do not
  repeat any figure from rows 3-16, 21-25, 33.
- Missing `LICENSE` file (row 31): add MIT text or remove the badge.
- Myntra/Flipkart/H&M counts and all Shopify price ranges (rows 4, 6, 7, 12): need a fresh
  count/range from the built index (no tracked artifact since the 2026-07-12 re-sync).
- README brand table omits Berrylush, Global Republic, Libas, which are in `src/config/stores.py`
  and `frontend/lib/stores.ts`; whether H&M is still in the live unified index is not evidenced
  by a tracked artifact.
- `/pdp-demo` mock discounts and shipping banner (row 38); cold-start wording in copy.ts (row 35).
- Whether the `.portfolio/metrics.json` label change (row 33) is acceptable to the portfolio
  refresher (id unchanged).
