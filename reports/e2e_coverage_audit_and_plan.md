# Report 4 — E2E Test Coverage Audit + Proposed Suite Plan

Compiled 2026-07-12. Every claim below is sourced to a file path or a live command run
this session — no invented numbers.

## Part 1 — What exists today

### Backend tests (`tests/*.py`, pytest)
1,325 tests total (fresh count, this session). Three tiers:
- **Pure unit tests** — no index, no LLM, no network. The bulk of the suite.
- **`@pytest.mark.requires_index` tests** (e.g. `test_occasion_search_augmentation.py`,
  `test_single_garment_set_exclusion.py`, `test_ws_multiturn_gender.py`,
  `test_ws_body_type_sequential_turns.py`, `test_ws_session_history.py`) — run against
  the REAL unified FAISS/BM25 index, through the real `build_graph()` agent, over
  FastAPI's in-process `TestClient` (including real WebSocket frames for the WS
  tests). LLM calls are mocked (`_MockLLM`). This is the closest thing to an E2E test
  that exists — but "in-process" means no real browser, no real HTTP over the wire, no
  real deployed service.
- **`@pytest.mark.requires_ollama` tests** — need a local Ollama daemon.

**CRITICAL GAP: `.github/workflows/ci.yml` runs `pytest -m "not requires_ollama and
not requires_index"`** — every single test in the middle tier above, including every
regression test written THIS SESSION for the gender-heuristic and occasion-pollution
bugs, **never runs in CI**. They only run when a human (or I) invokes `pytest tests/`
locally without the marker filter, which is what happened all session. This is not a
theoretical gap — it directly explains why both production bugs shipped and stayed
live until a live-URL proof (not a test) caught them.

### Live-API smoke scripts (`scripts/*.py`, plain `requests`/websocket-client)
Real network calls against the deployed Cloud Run URL, asserting on response JSON
content (article IDs, slot roles, gender, price sums, PDP URL shapes) — not mocked,
not in-process. None of these run in CI or on any schedule; each is a manually-invoked,
point-in-time proof script written for a specific past bug/feature:
- `product_checks.py` — 6 checks (owned-anchor composition, budget-sum, grounding)
- `wave5_smoke.py`, `wave6_smoke.py` — 5 checks each (image-anchor upload, footwear
  slot, buy-similar, cross-turn memory, slot-swap)
- `b2c_smoke_test.py` — cross-store search/outfit/image-search/PDP-resolve checks
- `smoke_test.py` — NOT actually E2E: imports `compose_outfit`/retrievers directly, no
  network at all, closer to an integration test

**Gap**: these hit the `/chat` REST+WS API directly. None of them load the actual
rendered Next.js frontend in a browser. A bug that only manifests in the React
rendering layer (wrong field read from the API response, a broken button, a share-page
that doesn't hydrate) is invisible to every one of these scripts.

### The CI "eval" job
`.github/workflows/ci.yml`'s `eval` job runs `python -m eval.run --provider ollama`,
which is `scripts/eval_harness.py` against `scripts/eval_queries.yaml` (32 queries) —
an **older, separate, smaller eval harness**, unrelated to this session's
`eval_gate.py` / `eval_strict.py` / `eval_model.py` (54/246 and 251-query harnesses).
The job is `continue-on-error: true`, explicitly "informational only." **None of this
session's actual eval infrastructure is wired into CI at all.** (Full detail in
Report 8.)

### Frontend
`grep`/`find` for Playwright, Cypress, or any `.spec.ts`/`.spec.tsx` file: **zero
results.** No browser-automation infrastructure exists anywhere in this repo today.

## Part 2 — The gap, stated plainly

There is currently no automated check, anywhere, that:
1. Drives the REAL rendered frontend (React hydration, client-side routing, actual
   button clicks) through a REAL browser.
2. Runs on every PR/push (everything above is either excluded from CI or is a
   manually-invoked local script).
3. Would have caught either of this session's two production bugs before a human
   ran an ad-hoc live-URL script.

The `requires_index` pytest tier is real, valuable, in-process backend coverage — it
should be wired into CI (a cheap, mechanical fix, not part of the "build a new suite"
scope below) — but it is not, and cannot become, a substitute for a real browser
E2E suite, because it never touches the frontend.

## Part 3 — Proposed E2E suite (PLAN ONLY — not built this turn, per your instruction)

### Tooling
**Python + `pytest-playwright`**, not `@playwright/test` in the frontend. Rationale:
one test runner (pytest, already the house convention), one CI job, and — most
importantly — a single test can assert across both layers (drive the browser AND query
the backend API/DB directly to confirm the state the browser shows actually matches
reality, e.g. "the item shown is not sold out" needs a live Shopify-availability check
that has nothing to do with the browser). A pure `@playwright/test` suite would need a
second language/runner to do that cross-checking.

### Structure
```
tests_e2e/
  conftest.py          # Playwright browser/page fixtures, backend-URL + frontend-URL config
  journeys/
    test_search_refine.py       # search -> refine (colour/budget) -> result content asserted
    test_outfit_compose.py      # "complete the look" -> outfit -> slot content + budget sum
    test_partner_styling.py     # couple/partner look -> both boards populated, gender-pure
    test_body_type.py           # body-type turn -> occasion turn -> composed look
    test_save_and_share.py      # save a look -> /look/[id] share page renders saved content
  assertions/
    no_sold_out.py       # shared helper: for each shown item, verify pdp_url resolves AND
                          # (for Shopify stores) cross-check live bulk /products.json
                          # availability — this is the check that would have caught Fix 1
    routing.py            # shared helper: assert the query landed on the intent-appropriate
                          # route/filters (would have caught both bugs fixed this session --
                          # e.g. "lehenga for sangeet" must show lehengas, "kurta for men"
                          # must show men's items)
```

### What each journey asserts (content, not just "no 500")
- **search → refine**: initial search returns items of the right gender/type; a colour
  refinement ("in blue") changes colour while preserving type; **every shown item's
  PDP link is verified not-sold-out** via `assertions/no_sold_out.py`.
- **outfit compose**: "complete the look" returns ≥2 complementary slots, budget sum
  matches the displayed total (mirrors `product_checks.py`'s existing check 3, but
  through the real browser this time), no gender-mixing.
- **partner/couple styling**: both partner boards populate, each gender-pure.
- **body-type**: a body-type turn followed by an occasion turn composes a full look
  (mirrors `test_ws_body_type_sequential_turns.py`'s existing logic, but through the
  real browser).
- **save → share**: save a look, then load `/look/[id]` fresh (new browser context, no
  session) and assert the saved items render with correct content — this is the one
  journey with literally zero existing coverage of any kind today.
- **routing correctness** (cross-cutting, asserted in every journey): the specific
  regression class from this session — an occasion word or a gender word in the query
  must route to the correct facet filter, not a heuristic default.

### CI wiring
New `e2e` job in `ci.yml`, gated to run against a **preview deploy** (Vercel preview
URL + a staging/preview Cloud Run revision), not production — so a flaky or failing
E2E run never blocks on prod and never hits the demo rate limits meant for real users.
Runs on PR, not on every push, given real backend/LLM calls cost money and time (a
budget note: this suite makes real Groq calls per journey — needs a small explicit
$/run budget line once built, per the cost-awareness default).

### Phased build (once you approve)
1. **Phase 1**: wire the existing `requires_index` pytest tier into CI (cheap,
   mechanical, closes the most dangerous silent gap immediately — this could ship
   independently of the browser suite and is worth doing regardless).
2. **Phase 2**: `search_refine` + `no_sold_out` journey only — proves the pattern,
   directly targets Fix 1's bug class.
3. **Phase 3**: remaining journeys (outfit, partner, body-type, save/share).
4. **Phase 4**: CI wiring against preview deploys.

### Explicit non-goals for this suite
- Not a replacement for `eval_gate.py`/`eval_strict.py` (those measure retrieval
  *quality*; this suite measures user-journey *correctness* — different failure
  modes, both needed).
- Not testing visual/pixel regressions (no screenshot-diffing) — out of scope unless
  requested separately.
