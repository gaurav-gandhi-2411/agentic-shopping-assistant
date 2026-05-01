# V2 Router — Clarify/Search Boundary Failure Analysis

Generated: 2026-05-02

---

## Failure Cases (OOD Set)

### Search → Clarify Misclassifications (4 cases, all high-confidence)

| ID | Query | True | Pred | Conf |
|---|---|---|---|---|
| ood_00 | "i need new clothes for my new job" | search | clarify | 0.956 |
| ood_02 | "what's trending right now in fashion" | search | clarify | 0.899 |
| ood_03 | "show me what other people are wearing" | search | clarify | 0.890 |
| ood_06 | "i want to look professional but not boring" | search | clarify | 0.693 |

### Clarify → Search Misclassifications (2 cases)

| ID | Query | True | Pred | Conf |
|---|---|---|---|---|
| ood_01 | "shopping for my mom's anniversary" | clarify | search | 0.760 |
| ood_09 | "what should i wear tomorrow" | clarify | search | 0.680 |

---

## Surface-Form Analysis

### What the 4 search→clarify failures have in common

All four are **first-turn queries with no explicit item noun in standard catalog form** but carry enough signal to issue a meaningful product search:

- `ood_00`: "new clothes" is a category; "new job" = professional/work context → search `professional work clothes`
- `ood_02`: "trending in fashion" = popularity browse → search `trending items`
- `ood_03`: "show me what other people are wearing" = social popularity browse → search `popular styles`
- `ood_06`: "professional but not boring" = style attribute → search `smart casual professional`

None contain explicit uncertainty language. None are gifts. None are underspecified in a way that blocks search.

### Why the model got them wrong: training data bias

The clarify training set (79 examples) is dominated by two template clusters:

| Pattern | Count | % |
|---|---|---|
| Gift-intent ("gift for X", "birthday", "something for Y") | ~32 | 41% |
| Explicit uncertainty ("not sure what", "no idea", "overwhelmed") | ~20 | 25% |
| Vague style ("something nice", "something fashionable") | ~16 | 20% |
| Other | ~11 | 14% |

The model learned: **clarify ≈ gift intent OR explicit uncertainty markers**.

The search→clarify failures don't fit either cluster, but share surface similarity:
- `ood_00`: "i need + context" → resembles clarify "I need help with fashion" / "I need fashion advice"
- `ood_02/03`: open-ended question form → resembles "what should I get?" clarify patterns
- `ood_06`: "I want to look X" → resembles vague style desires in clarify training

Search training (478 examples) is dominated by **explicit item mentions** ("black dresses", "winter coat", "jeans"), which the OOD failures lack. The model has no examples of search via occasion, trend, or style attribute without a named item.

### Why the 2 clarify→search failures happened

- `ood_01`: "shopping for my mom's anniversary" — The model has seen "shopping for [item]" as search. Without "not sure what to get" or "no idea", the uncertainty-less phrasing triggers search.
- `ood_09`: "what should i wear tomorrow" — "wear" is a clothing word; "tomorrow" provides timing. Model pattern-matches to "find me something to wear to X" → search. But without items retrieved and occasion context, this requires clarification.

---

## Root Cause

**The model learned the wrong proxy for clarify/search.**

True boundary:
> **SEARCH** = query contains enough information to formulate a meaningful product search string, even if broad (item category, style attribute, occasion with implicit category, trend intent).
> **CLARIFY** = query is so underspecified that a search would be a random guess — we must ask a follow-up first.

Learned proxy:
> **CLARIFY** ≈ contains "not sure", "gift for", "something nice", "help me decide"
> **SEARCH** ≈ contains a named item type

This proxy breaks on:
- Occasion queries without explicit uncertainty ("new job clothes") → should be search, predicted clarify
- Gift-intent queries without uncertainty markers ("shopping for mom's anniversary") → should be clarify, predicted search

---

## Fix: Contrastive Pair Generation

The training data needs **same-surface-form pairs** where the label flips based on specificity:

```
"I need help with fashion"         → CLARIFY  (no item, no direction)
"I need help finding a winter coat" → SEARCH   (item named)
"I need something for my new job"   → SEARCH   (context sufficient for professional clothes search)
"I need something for my friend"    → CLARIFY  (no item, no style)

"Looking for something nice"                    → CLARIFY
"Looking for something smart for job interviews" → SEARCH
"What's trending right now"                     → SEARCH  (trend browse)
"What should I get her"                         → CLARIFY (no item, no style signal)
```

Target: 80–120 contrastive pairs across 4 failure mode patterns.

---

## Files

- Contrastive training data: `data/router_training_v2_contrastive.jsonl`
- V3 train/val/test splits: `data/router_dataset_v3_{train,val,test}.jsonl`
- V3 model: `models/distilbert_router_v3/`
