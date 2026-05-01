# V3 vs V3.1 — Final Evaluation Report

Generated: 2026-05-02 | Phase 2 Day 2.5 (V3.1 iteration)

---

## Pass/Fail Decision

**FAIL. Stopping cascade iteration. Deploying V1 + Phase 1 prompt fixes.**

---

## Results vs Criteria

| Criterion | Target | V3 | V3.1 | Pass? |
|---|---|---|---|---|
| OOD-20 macro F1 | ≥ 0.65 | 0.60 | **0.63** | FAIL |
| OOD-10 target pattern F1 | ≥ 0.80 | 0.44* | **0.44*** | FAIL |
| OOD-10 accuracy | — | 80% | 80% | — |
| OOD-30 macro F1 | ≥ 0.85 | 0.89 | **0.86** | PASS |

*OOD-10 macro F1 is low because all 10 examples are `search`; macro averages over all 6 classes including 5 with zero support. 2/10 target-pattern queries still fail at high confidence (conf=0.864, 0.748). Not "genuinely fixed."

---

## Full Comparison Table

| Metric | V2 | V3 | V3.1 | Target |
|---|---|---|---|---|
| In-dist macro F1 (own test) | 0.9035 | 0.9933 | 0.9304 | — |
| OOD-20 macro F1 | 0.25 | 0.60 | **0.63** | ≥ 0.65 |
| OOD-20 accuracy | 50% | 75% | 80% | — |
| OOD-30 macro F1 | 0.88 | 0.89 | **0.86** | ≥ 0.85 |
| OOD-30 accuracy | 83% | 83% | 87% | — |
| OOD-10 accuracy | — | 80% | 80% | ≥ 80% |
| Combined OOD-60 macro F1 | — | — | **0.82** | — |
| Combined OOD-60 accuracy | — | — | 83% (50/60) | — |
| Clarify F1, OOD-20 | 0.50 | 0.80 | 0.73 | — |
| Training examples | 1006 | 1111 | 1106 | — |

---

## V3.1 OOD-20 Prediction Detail

| ID | Query | True | V3 | V3.1 | Note |
|---|---|---|---|---|---|
| ood_00 | "i need new clothes for my new job" | search | search ✓ | search ✓ | — |
| ood_01 | "shopping for my mom's anniversary" | clarify | clarify ✓ | **search** ✗ | V3.1 regression |
| ood_02 | "what's trending right now in fashion" | search | search ✓ | search ✓ | — |
| ood_03 | "show me what other people are wearing" | search | search ✓ | search ✓ | — |
| ood_04 | "i'm tired of my current style" | clarify | search ✗ | **clarify** ✓ | Fixed |
| ood_05 | "what should i wear" | clarify | clarify ✓ | clarify ✓ | — |
| ood_06 | "i want to look professional but not boring" | search | search ✓ | search ✓ | — |
| ood_07–11 | various search | search | correct ✓ | correct ✓ | — |
| ood_12 | "show me sustainable brands" | respond* | search ✗ | search ✗ | *label debatable |
| ood_13–15 | various search | search | correct ✓ | correct ✓ | — |
| ood_14 | "athleisure recommendations" | search | clarify ✗ | **search** ✓ | Fixed |
| ood_16 | compare intent | compare | compare ✓ | compare ✓ | — |
| ood_17 | "filter to under $50" (stateful) | respond* | filter ✗ | filter ✗ | *label debatable |
| ood_18 | "i changed my mind, show me something else" | search | respond ✗ | respond ✗ | Unchanged |
| ood_19 | "are these on sale" | respond | respond ✓ | respond ✓ | Conf improved: 0.365→0.909 |

**V3.1 net: fixed 2 (ood_04, ood_14), regressed 1 (ood_01). 16/20 correct.**

---

## High-Confidence Wrong Predictions (V3.1, OOD-60)

These 8 predictions will NOT be caught by any cascade threshold:

| ID | Query | True | Pred | Conf |
|---|---|---|---|---|
| ood_12 | "show me sustainable brands" | respond | search | 0.990 |
| ood_18 | "i changed my mind, show me something else" | search | respond | 0.985 |
| ood30_07 | "office appropriate but not stuffy" | search | clarify | 0.970 |
| ood_01 | "shopping for my mom's anniversary" | clarify | search | 0.956 |
| ood30_09 | "I'm rebuilding my wardrobe from scratch…" | search | clarify | 0.949 |
| ood10_03 | "looking to impress at a first date" | search | clarify | 0.864 |
| ood30_24 | "only show me things under forty pounds" | filter | respond | 0.846 |
| ood10_05 | "I'm attending a conference and want to look sharp" | search | clarify | 0.748 |

---

## Why Stopping is the Right Call

Three rounds of contrastive augmentation produced diminishing and unstable returns:

| Iteration | Contrastive pairs added | OOD-20 F1 | Regressions introduced |
|---|---|---|---|
| V2 (baseline) | 0 | 0.25 | — |
| V3 | +110 (clarify/search boundary) | 0.60 | ood_04 (new failure) |
| V3.1 | +20 (event-verb pattern) | 0.63 | ood_01 (regression: clarify→search) |

Each iteration fixes the patterns it was trained on while breaking neighboring patterns. The model is memorizing surface forms, not learning the semantic decision rule.

**The core limit:** Distinguishing search from clarify requires understanding whether a query contains *enough shopping intent to retrieve relevant products* — a judgment that depends on broad world knowledge about occasions, fashion categories, and user intent. A 67M-parameter classifier trained on 1106 examples cannot learn this reliably.

---

## Threshold Sweep: V3.1 on Combined OOD-60

| Threshold | Kept | Kept Acc | Escalation | Effective Acc (LLM perfect) |
|---|---|---|---|---|
| 0.50 | 60/60 | 83.3% | 0% | 83.3% |
| 0.70 | 57/60 | 86.0% | 5% | 86.7% |
| 0.80 | 54/60 | 87.0% | 10% | 88.3% |
| 0.90 | 49/60 | 89.8% | 18% | 91.7% |
| 0.95 | 44/60 | 90.9% | 27% | 93.3% |

The 8 high-confidence wrong predictions (conf 0.748–0.990) survive all thresholds below 0.75–0.99. To catch all of them you'd need threshold ≥ 0.99, which escalates almost everything. The cascade value proposition collapses.

---

## Decision: Ship V1 + Phase 1 Prompt Fixes

**What this means in production:**
- Router: Groq LLM (`llama-3.1-8b-instant`) with improved Phase 1 prompt
- No DistilBERT cascade in production
- Latency: ~300–600ms LLM round-trip vs 5–30ms DistilBERT
- Cost: ~$0.05–0.10/1k requests vs $0 for DistilBERT

**What to document in README:**

> *Cascade routing explored (Phase 2): Fine-tuned DistilBERT classifier (6-class, 67M params) on augmented stateful training data. Achieved 0.93 macro F1 on in-distribution held-out test. OOD evaluation (60 natural user queries) revealed systematic failures: the model memorizes surface patterns (gift-intent markers, specific phrasing) rather than learning the semantic decision rule (whether a query contains enough shopping intent to retrieve relevant products). After 3 rounds of contrastive augmentation (130 pairs total), OOD macro F1 reached 0.63 but stalled due to whack-a-mole regressions on neighboring patterns. Production deployment requires either (a) a larger base model, (b) a fundamentally different approach to the search/clarify boundary, or (c) training data from actual production queries with diverse natural phrasing. Current production: LLM-only routing with prompt engineering.*

---

## What Would Actually Fix This

1. **Production query logging + labeling**: 1000+ real user queries labeled by the routing rule (not synthetic augmentations). This would expose the full distribution of natural phrasings the model needs to generalize to.

2. **Larger base model**: GPT-2, BERT-large, or a small instruction-tuned LLM would handle the semantic generalization better. DistilBERT's 67M parameters and pre-training on generic text isn't well-suited to subtle intent classification.

3. **Different decision boundary**: Instead of trying to teach clarify vs search, use the LLM router for all ambiguous first-turn queries (those without an explicit item noun) and DistilBERT only for stateful routing (filter/compare/outfit/respond) where the signal is structural.

---

## Files Created This Session

| File | Description |
|---|---|
| `data/router_training_v2_contrastive.jsonl` | 110 clarify/search contrastive pairs (V3) |
| `data/router_training_v3_contrastive_v3_1.jsonl` | 20 event-verb pattern pairs (V3.1) |
| `data/router_dataset_v3_{train,val,test}.jsonl` | V3 training splits |
| `data/router_dataset_v3_1_{train,val,test}.jsonl` | V3.1 training splits |
| `data/router_ood30_test.jsonl` | 30-query generalization OOD set |
| `data/router_ood10_event_verb.jsonl` | 10-query target-pattern OOD (held out) |
| `models/distilbert_router_v3/` | V3 model |
| `models/distilbert_router_v3_1/` | V3.1 model |
| `reports/v2_failure_analysis.md` | Root cause analysis |
| `reports/v2_vs_v3_comparison.md` | V2 vs V3 comparison |
| `reports/v3_1_results.md` | This report |
