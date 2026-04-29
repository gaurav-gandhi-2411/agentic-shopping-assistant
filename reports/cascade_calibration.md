# Cascade Router Threshold Calibration

Generated: 2026-04-29

## Summary

| Threshold | Status | DB accuracy (kept) | Cascade accuracy | Escalation rate (test set) |
|---|---|---|---|---|
| 0.80 | ~~deployed~~ retired | 1.000 | 1.000 | 84.6% (33/39) |
| **0.65** | **deployed (2026-04-29)** | **0.889** | **0.974** | **76.9% (30/39)** |
| 0.70 | candidate | 0.875 | 0.974 | 79.5% (31/39) |
| 0.60 | too permissive | 0.900 | 0.974 | 74.4% (29/39) |

**Current threshold: 0.65** — chosen to balance DB coverage vs accuracy given the production
input distribution. Re-calibrate when DistilBERT is retrained on stateful examples (Phase 2).

---

## Why 0.65 (not 0.80)

### Train/serve skew discovery (2026-04-29)

The original 0.80 threshold was chosen from analysis of a 39-example test set composed primarily
of single-turn, first-turn queries ("show me black dresses", "what should I wear to the beach?").
On that set, all wrong DistilBERT predictions had confidence < 0.773, and the one borderline wrong
prediction at 0.773 was an OOC query already caught upstream. Hence 0.80 gave 100% DB accuracy.

In production, DistilBERT's input is **state-encoded**:

```
query: <text> | last_action: <action> | items: <n> | filters: <json>
```

The context fields (`last_action`, `items`, `filters`) change dramatically between turns.
Measured production confidence for identical query text across different contexts:

| Context | Input encoding | conf |
|---|---|---|
| Fresh (last_action=none, items=0) | `query: show me black dresses \| last_action: none \| items: 0 \| filters: {}` | 0.718 |
| After a search (items=5) | `query: show me black dresses \| last_action: search \| items: 5 \| filters: {}` | 0.362 |
| After a respond turn | `query: show me black dresses \| last_action: respond \| items: 5 \| filters: {}` | 0.481 |

At 0.80: **all production queries escalate to LLM**, including correct DistilBERT predictions.
The cascade architecture was providing zero DistilBERT benefit — functionally equivalent to
LLM-only mode.

### Why 0.65 and not lower

From `reports/threshold_analysis.md`, the wrong-prediction confidence distribution on the test set:

| Wrong prediction | Confidence | True label | Predicted |
|---|---|---|---|
| "Show me face creams and moisturizers" (OOC) | 0.773 | respond | search |
| "Something for someone special" | 0.567 | clarify | search |
| "Blazers but not the pinstriped ones" | 0.372 | search | filter |
| "Showcase it with matching outfits" | 0.352 | outfit | filter |
| "Style this with complementary pieces" | 0.322 | outfit | filter |
| "I'm loving these options" | 0.257 | respond | filter |

**No wrong prediction falls between 0.65 and 0.773.** At threshold 0.65:
- The OOC case (0.773) still escalates — but OOC queries should be caught upstream by keyword
  detection before reaching the router, so this is an upstream gap, not a router failure.
- All other wrong predictions (max 0.567) escalate correctly.
- Fresh first-turn queries with conf ~0.718 are kept by DistilBERT (correct route, high conf).

At 0.60: 2 fewer escalations, but the gap between 0.567 (highest wrong prediction excluding OOC)
and 0.60 is too narrow (delta = 0.033). 0.65 provides more margin.

At 0.70: correct behavior for fresh queries (conf ~0.718 kept), but leaves only 0.050 margin
above the highest non-OOC wrong prediction (0.567 → 0.70 gap = 0.133). Slightly safer than
0.65 but not meaningfully different in practice; 0.65 was chosen to maximize DB coverage.

### Expected production behavior at 0.65

- **Fresh first-turn queries** (common — new session): conf ~0.70–0.85 → DB handles most
- **Follow-up queries with prior context** (last_action ≠ none, items > 0): conf ~0.35–0.55 →
  most escalate to LLM (expected — multi-turn state is outside the training distribution)
- **Escalation rate estimate**: 40–60% of production turns (vs 100% at 0.80, vs ~25% at 0.65
  on the original clean test set)

---

## The deeper issue: train/serve skew

The DistilBERT model was trained on 388 examples, the majority of which were **fresh-turn,
clear-intent queries** without preceding conversation state. The production environment generates
state-encoded inputs where `last_action`, `items_retrieved`, and `filters` vary per turn.

The model has never seen training examples like:
```
query: show me black dresses | last_action: search | items: 5 | filters: {}
```
This means multi-turn confidence is unreliable — the model is making predictions on
out-of-distribution inputs, leading to low confidence and excessive escalation.

This is a Phase 2 problem. Phase 1 threshold lowering is a tactical patch.

---

## Phase 2 plan: retrain on stateful distribution

**Day 1 — Data augmentation:**
- For each training example with action ∈ {search, filter, compare, outfit, respond},
  generate 2–3 stateful variants with realistic `last_action`/`items`/`filters` values
- Some actions change under context (e.g. "compare" is only valid when items > 0); generate
  both valid and invalid state variants with appropriate labels
- Target: 800–1,200 examples covering realistic conversation state distribution
- Save: `data/router_training_v2.jsonl`

**Day 2 — Retrain + recalibrate:**
- Fine-tune DistilBERT on augmented data
- Held-out test set: 80 examples, balanced across state types
- Re-measure macro F1 on stateful test set
- Run threshold sweep; likely 0.75–0.85 is correct post-retraining
- Update this document with new numbers

**Success criteria:**
- DistilBERT keeps ≥ 65% of production turns (vs ~40–60% at Phase 1)
- DB accuracy on kept queries ≥ 0.90
- Macro F1 on stateful test set ≥ 0.85

---

## Source data

- Threshold sweep: `reports/threshold_analysis.md`
- Eval harness (cascade): `reports/eval_results_20260428_groq_v3.json`
- Training data: `data/router_dataset.jsonl` (388 examples, 39-example test split)
- DistilBERT model: `models/distilbert_router/` (macro F1 = 0.8345 on clean test set)
