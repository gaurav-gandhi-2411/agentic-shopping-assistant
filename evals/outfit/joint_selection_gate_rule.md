# Joint slot selection: pre-registered gate rule

Registered BEFORE any result of the joint selector is computed (this file is committed ahead of
the implementation; no number was measured before it was committed). "master" below
means code at `ad17f47` (#135) with the joint path absent; it is re-measured by the author on the
same index and scripts, not taken from #135's description (which says 29/36 and 23 of 358).

Index: the real local public-view unified index (`INCLUDE_NON_PUBLIC_STORES` unset), the same
build the snapshot fixture was recorded on (fingerprint-checked by the snapshot script).
No LLM and no network calls are involved anywhere: composition is retrieval-only.

## What is compared

* master = the greedy path (seed chosen first, slots filled one by one).
* branch = joint selection for budgeted, occasion-driven, non-owned turns (no explicit seed);
  everything else, including every no-budget turn, takes the unchanged greedy path.
* "greedy" in the relevance comparison = the same branch code with the joint path switched off
  (`composer.JOINT_SELECTION_ENABLED = False`), which is the master code path.

## Joint objective (what the selector maximises)

Per anchor candidate a (top ~20 gate-passing, garment-gated, in-budget anchors) and per slot s
of `get_fill_slots(class(a))`, the candidate list is the gate-passing items of the slot's
retrieval (every gate of `_score_candidates` unchanged), top 20 by the existing `final_score`.
Each item's value is `final_score / max final_score of its own list` in (0, 1]. The selector
maximises, lexicographically: (1) number of required slots filled, (2) number of cards,
(3) sum of item values, subject to total price <= budget, no item twice, no repeated
(product-name, colour). Ties broken by article id ascending. Bounded branch-and-bound search
with a wall-time budget; on timeout the greedy path runs and the reason is logged.

## Gates (every one must hold, else the PR is opened as DRAFT and says which failed)

1. Grid: on the 36-cell grid (`scripts/outfit_budget_grid.py`, real index) the number of cells
   with >= 3 cards is >= the master number measured by the author, and the number of over-budget
   cells is 0.
2. `casual outfit under ₹2,500 for men` returns >= 3 cards (seed + complements; snapshot entry
   Base variant, and the compose_turn Base look).
3. No-budget parity: `scripts/outfit_compose_snapshot.py --check` must show every no-budget entry
   of the 30 recorded queries (15 of them) byte-identical. Budgeted entries (15) may change; if
   they do the fixture is re-recorded in a SEPARATE commit whose message carries an old/new table,
   as #135 did.
4. Relevance must not drop. Independent evaluator (`scripts/outfit_joint_eval.py`, deliberately
   not the selector's own objective): for each chosen item, take the top-20 retrieval for that
   item's slot query (gender filter, price_max = the cell budget; seed uses the occasion anchor
   query; complement slots use `get_fill_slots(class(chosen seed))`), and let
   `rel = score / top score` of that top-20 list, 0 if the item is not in the list. Per cell,
   `per_card_rel` = mean rel over the cards. Over the grid cells where branch and greedy return the
   SAME card count (n reported), the mean of `per_card_rel` for joint must be >= greedy's mean
   minus 0.02 (absolute). Cells where card counts differ are reported separately, not averaged in.
5. Latency: `sangeet outfit under ₹3000 for women` full turn (`compose_turn`: probe + variants)
   median of n=10 warm local runs (one discarded warm-up run first, same process, same index)
   must be <= master's median x 1.10. Both are measured; master is measured at the commit that
   adds this file (code identical to master).
6. Kids / non-fashion cards = 0 and gender violations = 0 on the 72-cell row grid
   (`scripts/outfit_more_options_grid.py`: its independent audit's footwear/swimwear/innerwear/
   nightwear/jewellery_bag leaks, plus a kids-name and cross-gender-name check added to the
   audit as a measurement-only change in the same script).

## Item-specific acceptance (not gates on the numbers above, but stated up front)

* B (more options): standalone bottoms in the 72-cell row grid: master count (expected 23 of
  358 cards) -> 0 on the branch, with real "Kurta & Pyjama Set" style sets kept.
* C (garment gate at the anchor stage): at budget ₹800, for men and for women, the seed of the
  occasion-driven look is `_is_outfit_garment` true, or the result is the honest empty path
  (no seed + the budget message); which of the two happens is reported per case.
* D (grid floor): the unit-test floor is replaced by (CI-measured count of cells with >= 3 cards
  on CI's own index) - 1, read from the CI log, not guessed. Both local and CI numbers reported.

## Complexity and safety

Per anchor: slots <= 4, candidates <= 20 per slot, worst case 20^4 nodes, pruned by an upper bound
(current + sum of remaining per-slot maxima, lexicographic) and by the remaining budget. Anchors
<= 20. Retrieval is cached per (query, filters) inside one compose call. Wall budget constant
`JOINT_WALL_BUDGET_S`; deadline checked at every search node and between anchors.

## Amendments registered before any joint-selector result was computed

Master was measured first (code at a99b4c1, identical to ad17f47 outside this doc), then these
two clarifications were made, still before the joint selector had ever been run:

* Item value normalisation: `final_score / max final_score` is taken over the slot's options
  across all anchor plans of one compose call (one reference per slot name), not per (anchor,
  slot) list; a per-list reference would give an anchor with poor slot options the same value
  as one with excellent options. The anchor's value is its retrieval score over the best
  retrieval score in the anchor pool.
* Gate 6 reading: on master the 72-cell audit already reports 2 hits, both verified by hand as
  audit false positives ("Black Printed Maxi Dress With Belt" matches `belt`; "Men's Yellow
  Lehariya Cotton Kurta & Pyjama Set" is typed nightwear but is a real set). Gate 6 holds when
  kids = 0, gender violations = 0 and the audit's hits are a subset of those two.

## Master baseline (measured, index = recorded local build, venv Python 3.13)

Commands: `outfit_compose_snapshot.py --check`, `outfit_joint_eval.py grid|rows|latency` run from a
`git archive` export of a99b4c1 (plus this branch's two measurement-only scripts).

| measure | master |
|---|---|
| snapshot --check | 30/30 identical (index fingerprint matched) |
| grid cells with >= 3 cards | 29/36, over-budget cells 0 |
| `casual outfit under ₹2500 for men`, Base look cards | 2 |
| 72-cell row grid: cards / standalone bottoms | 335 / 25 (the #135 text said 23 of 358) |
| 72-cell audit hits | 2 (the two false positives above), kids 0, gender violations 0 |
| Sangeet women ₹3,000 turn, n=10 warm | median 2202 ms (min 2115, max 2312) |

Latency limit (gate 5) = 2202 x 1.10 = 2422 ms.
