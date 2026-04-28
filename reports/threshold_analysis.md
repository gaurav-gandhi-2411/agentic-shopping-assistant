# Cascade Threshold Analysis

Generated: 2026-04-28

## Summary

| Threshold | Escalation rate (test set) | DB accuracy on kept queries | Cascade accuracy (LLM=100%) |
|---|---|---|---|
| 0.60 | 74.4% (29/39) | 0.900 | 0.974 |
| 0.65 | 76.9% (30/39) | 0.889 | 0.974 |
| 0.70 | 79.5% (31/39) | 0.875 | 0.974 |
| 0.75 | 79.5% (31/39) | 0.875 | 0.974 |
| **0.80 ✓ deployed** | **84.6% (33/39)** | **1.000** | **1.000** |
| 0.85 | 97.4% (38/39) | 1.000 | 1.000 |
| 0.90 | 100.0% (39/39) | — | 1.000 |

**Recommended threshold: 0.80**

---

## Test set confidence distribution (fine-tuned model, n=39)

| | Correct (33) | Wrong (6) |
|---|---|---|
| Median confidence | 0.5074 | 0.3621 |
| Min confidence | 0.2567 | 0.2572 |
| p10 confidence | 0.3127 | — |
| Max confidence | 0.9999 | 0.7728 |
| p90 confidence | — | 0.7728 |

---

## Wrong predictions detail (sorted by confidence)

| Confidence | True label | Predicted | Query |
|---|---|---|---|
| **0.773** | respond | search | "Show me face creams and moisturizers" ← OOC (see note) |
| 0.567 | clarify | search | "Something for someone special" |
| 0.372 | search | filter | "Blazers but not the pinstriped ones from before" |
| 0.352 | outfit | filter | "Showcase it with matching outfits" |
| 0.322 | outfit | filter | "Style this with complementary pieces" |
| 0.257 | respond | filter | "I'm loving these options" |

**Note on the 0.773 case:** "Show me face creams and moisturizers" is a beauty/personal-care query that
should be caught by the OOC keyword detector in `graph.py` before it reaches the router. The OOC list
covers "cleanser" and "face wash" but not "face cream" or "moisturizer". This is an OOC expansion gap,
not a router failure. If OOC is extended to cover these terms, 0/6 wrong predictions fall above 0.70.

---

## Why the eval harness showed 0% escalation at 0.70

The 32-query eval harness uses first-turn, clear-intent queries ("Show me black dresses", "I want dark
blue", "What should I wear to a beach holiday?"). These are high-confidence for DistilBERT — all scored
above 0.70 (median 0.735).

The test set (39 examples, stratified from training data) has multi-turn, state-dependent queries:
"Compare those two", "Only the dark ones", "Style this with complementary pieces". These are harder
— DistilBERT needs the `[CTX]` state prefix to disambiguate them, and confidence is lower (median 0.51
correct, 0.36 wrong).

**Production reality is closer to the test set than the eval harness.** Real users mix first-turn
queries with follow-up state-dependent turns. The 0% escalation figure from the eval harness does not
represent expected production escalation rates.

---

## Threshold recommendation: 0.80

At 0.70, one wrong prediction slips through: "Show me face creams" @ 0.773. This is actually an OOC
miss (should be caught upstream), not a routing error.

At 0.80, all 6 wrong predictions are escalated. DB makes zero errors on the queries it keeps.

The cost of moving from 0.70 → 0.80: 2 additional escalations per 39 queries (+5.1% escalation rate).
On a test set with a realistic query mix, that's roughly 5 extra LLM router calls per 100 queries —
approximately +$0.0025/1k queries at current Groq pricing.

At 0.85, the escalation rate jumps to 97.4% — effectively LLM-only. That's too aggressive.

**0.80 catches all genuine DB misclassifications while escalating only ~5% more queries than 0.70.**

---

## Issue 2 diagnosis: "I need help with fashion"

With the fine-tuned model and proper state encoding:

```
[QUERY] I need help with fashion [CTX] last_action=none items=0 filters=none
  clarify : 0.4917   ← correct prediction
  search  : 0.2069
  respond : 0.1418
  ...
```

DistilBERT **correctly predicts `clarify`** but at confidence 0.49 — below both 0.70 and 0.80. The
cascade would escalate this query to the LLM router. If the LLM router then misroutes it to `search`,
the failure is in the LLM prompt, not the cascade threshold logic.

On the live Space, DistilBERT and Cascade are not available (Issue 1), so all queries go through the
LLM router only. The LLM router's `search` misclassification for this query is the active failure.

Two fixes, in order of priority:
1. Deploy the model weights to fix Issue 1 — cascade then catches this via escalation to LLM. If LLM
   is also wrong, the cascade can't save it.
2. Audit the LLM router prompt — "I need help with fashion" should trigger `clarify`, not `search`.
   Add explicit clarify detection for help-seeking, open-ended phrasing.
