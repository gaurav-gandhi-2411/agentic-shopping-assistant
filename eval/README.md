# Eval methodology — standing rules

This file holds cross-cutting eval conventions that apply across scripts, not
just one fixture or one run. Individual scripts' own docstrings cover their
specific mechanics; this is for rules that keep recurring.

## Proxy metrics may gate exploration; only the hand-labeled strict eval gates shipping

A cheap, self-graded, or synthetic precision proxy (property-floor P@5,
"literal P@5" against a filter-defined universe, an LLM judge, etc.) is fine
for **exploring** a change quickly — sweeping a parameter, comparing several
candidates, deciding what's worth building. It is **not** sufficient evidence
that a change is safe to ship. Only `scripts/eval_strict.py`'s hand-labeled
precision (`eval/fixtures/strict_gold_labels.yaml`) — audited item-by-item
against the rubric in `strict_gold_queries.yaml`'s header, never self-graded —
gates an actual ship decision.

**This has now cost a real regression once, concretely, not hypothetically.**
`eval/per_store_cap_sweep.py`'s "literal P@5" proxy (top-5 items' membership
in a filter-defined universe — hand-label-free by construction, useful for a
fast parameter sweep) read `per_store_cap=4→8` as flat, ~0.721→0.714 (a ~1pp
move, described in the sweep report as "essentially flat"). Re-running the
REAL hand-labeled strict eval at the same config value showed strict P@5
actually moved **0.874→0.866, a real −0.8pp regression, −2.4pp on the
`occasion` category specifically** (`reports/pushdown_fix_20260806.md`,
2026-08-06). The proxy wasn't wrong on its own terms — it measured exactly
what it claimed to measure — but it understated the real cost because it
can't see the same nuance a rubric-following human/Claude label can (garment
type vs. type-adjacent set, silhouette contradictions, occasion-register
formality). The config change was reverted once the real number came in.

**Practical rule**: before changing a value that survived only a proxy-metric
sweep, run the real strict eval and treat its number as authoritative — not
the proxy's. If a proxy and the strict eval disagree, the strict eval wins,
full stop; investigate the proxy's blind spot afterward, don't average the two
or split the difference.

## Know what a control's surface actually covers before citing it as evidence

Related, and caught in the same investigation: `scripts/eval_strict.py`
(`--mode pipeline`) and `scripts/eval_model.py`'s `intent`/`r1`/`gates` stages
all deliberately bypass the live LLM router (`src/agents/graph.py`'s
`ROUTER_PROMPT` / `LLMRouterBackend`) — they call `retriever.search()`
directly with a hand-set `filters={"gender": ...}`, or mirror the
post-router pipeline deterministically. That's why they exist (no LLM cost,
reproducible) — but it also means a change to `ROUTER_PROMPT` itself (e.g.
the 2026-08-06 facet-vocabulary fix, `src/agents/graph.py`) will **never**
move any of these numbers, regardless of whether the change is correct or
badly broken. A flat strict-P@5/property-floor/coherence-gate result after a
router-prompt change is not evidence the change is safe — it's evidence
those gates don't reach that code path at all. The only things that actually
exercise `ROUTER_PROMPT` are the `e2e` stage (`scripts/eval_model.py
--stages e2e`, full live graph turns) and a direct probe of the router LLM
itself (see `reports/router_stale_type_probe.json` /
`router_general_type_probe.json` for the pattern — build the exact prompt
`LLMRouterBackend.decide()` would, call the LLM, parse the response, check
the emitted filter against the catalogue). Before citing "the regression
suite is green" as evidence for a change, name which stages the change
actually touches and confirm at least one of them exercises the changed
code path — a green gate on an untouched path proves nothing.

## A precision gain from fewer/zero results is not a gain — check result COUNT, not just score

`precision@k = n_relevant / n_scored`. Any change that shrinks `n_scored` —
by making a query return fewer than `k` items, or zero — can move the ratio
up even though nothing got better, because a bad query that returns nothing
exits the denominator entirely instead of counting as a miss. A rising
score and a shrinking denominator look identical in the headline number; only
checking the count catches the difference.

**Caught concretely in the compose-logic wave's Cluster F (2026-08-07,
`reports/compose_wave_final_20260807.md`).** Two strict-eval misses were
"wrong colour entirely": `parse_intent` had no `"gold"` entry in its colour
vocabulary at all, so "gold embellished lehenga for eid" resolved
`colour=None` (no filter, an unrelated red item ranked top-5) and "gold
jewellery to go with red lehenga" resolved `colour="Red"` (the only
recognised colour word, wrongly applied to the item being searched for).
Adding `"gold"→"Gold"` root-caused correctly and looked like it worked — the
strict-eval precision figure rose. But the catalogue has exactly 1
"Gold"-tagged lehenga (a kids item, filtered out separately) and **zero**
"Gold"-tagged jewellery rows (jewellery's colour facet tracks stone colour,
not metal finish) — so the fix made both target queries retrieve **zero
items**, not a wrong one. Both queries silently dropped out of the strict
eval's `n_scored` denominator instead of counting as continued misses,
which is why the headline number rose. Caught only by testing the two
target queries directly against the retrieval pipeline before trusting the
eval delta (`_retrieve_pipeline(...) -> []` for both) — the eval run alone
would not have shown this; it doesn't distinguish "5 items, 4 relevant"
from "0 items, 0 scored" in the top-line score. Reverted before shipping.

**Practical rule**: whenever a change is expected to move precision/recall,
also report `n_scored` (or raw item count) before and after, per query if
the change is narrowly targeted. A metric that improved alongside a stable
or growing `n_scored` is real. A metric that improved alongside a shrinking
`n_scored` needs the raw retrieved items checked directly, not just the
score — the same discipline as rule 98a's "fail closed, never open" for
guards: an empty/degraded result must never look identical to a good one in
the number that gates the decision.
