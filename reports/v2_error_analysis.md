# V2 Router Error Analysis

Generated: 2026-04-30

## Summary

V2 model has 2 errors on the 123-example stateful test set (accuracy 0.9837, macro F1 0.9858).
Both errors involve the `respond` class being over-predicted on ambiguous first-turn queries.

---

## Misclassifications

### Error 1: clarify → respond
```
query: "Get something nice for my friend's birthday"
state: la=none  items=0  filters={}
true: clarify
pred: respond  (conf=0.5336)
```
**Why it's wrong:** The model sees a gift/occasion query with no item type and routes it as `respond`
(conversational reply) rather than `clarify` (ask what type of item). Low confidence (0.53) signals
model uncertainty — this would escalate to LLM at any threshold ≥ 0.54.

**Root cause:** Gift-intent queries ("something nice for X") are borderline: they could warrant
clarification ("what kind of item?") or a search ("birthday gift clothing"). The training data
likely has similar phrasing labeled as `search` or `respond`.

**Production impact:** Minimal. At threshold=0.70, this escalates to LLM (conf=0.53 < 0.70).

---

### Error 2: search → respond
```
query: "What are my top fashion shopping options?"
state: la=none  items=0  filters={}
true: search
pred: respond  (conf=0.6827)
```
**Why it's wrong:** "What are my top... options?" reads as a question that could be answered
conversationally, but it's actually a discovery/search intent.

**Root cause:** "What are my options?" framing triggers respond-class associations. Training data
probably has similar phrasing as `clarify` or `respond`. The correct behavior (search for popular
fashion items) requires understanding shopping context.

**Production impact:** At threshold=0.70, this also escalates to LLM (conf=0.68 < 0.70).

---

## Low-Confidence Correct Predictions

| Query | la | items | true | pred | conf |
|---|---|---|---|---|---|
| "Show me face creams and moisturizers" | none | 0 | respond | respond | 0.484 |
| "i wanna see sum sweterz" | none | 0 | search | search | 0.632 |
| "See the latest products filtered by these options" | filter | 0 | search | search | 0.680 |

The lowest-confidence correct prediction (OOC query, conf=0.484) correctly routes to `respond`
but would escalate to LLM at any threshold ≥ 0.49. This is the right behavior — OOC queries
should be handled by the LLM.

---

## Confusion Pattern Summary

Only one systematic pattern:

**respond over-prediction on ambiguous first-turn queries** (2/2 errors):
- Triggered by: gift-intent language, meta-shopping questions
- Confidence range: 0.53–0.68 (both below threshold=0.70)
- Production impact: Both escalate to LLM, so cascade handles them correctly

No examples of:
- compare/outfit confusion (was the main v1 weakness after encoding fix)
- filter/search confusion (0 errors)
- clarify being over-predicted (0 errors)

---

## Recommendation

Both errors are caught by the cascade at threshold ≥ 0.70. No additional training data
is needed before deployment. If `respond` over-prediction on gift queries becomes a
production pattern, add 10-15 gift-intent examples labeled `clarify` or `search` in v3.

---

## Source
- Model: `models/distilbert_router_v2/`
- Test set: `data/router_dataset_v2_test.jsonl` (123 examples)
- Results: `reports/v2_test_results.json`
