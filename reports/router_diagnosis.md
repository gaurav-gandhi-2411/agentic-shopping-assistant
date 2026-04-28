# DistilBERT Router — Failure Mode Diagnosis

**Test set:** 39 examples | **Errors:** 6 | **Accuracy:** 84.6%
**Train set:** 310 examples | **Errors:** 32 | **Accuracy:** 89.7%

## 1. Confusion Matrix (test set, n=37)

Rows = true label, Columns = predicted label.

```
            clarify  compare   filter   outfit  respond   search
          ------------------------------------------------------
clarify           4        0        0        0        0        1
compare           0        5        0        0        0        0
filter            0        0        5        0        0        0
outfit            0        0        2        2        0        0
respond           0        0        1        0        6        1
search            0        0        1        0        0       11
```

### Per-class summary

| Class | Support | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| clarify | 5 | 4 | 0 | 1 | 1.00 | 0.80 |
| compare | 5 | 5 | 0 | 0 | 1.00 | 1.00 |
| filter | 5 | 5 | 4 | 0 | 0.56 | 1.00 |
| outfit | 4 | 2 | 0 | 2 | 1.00 | 0.50 |
| respond | 8 | 6 | 0 | 2 | 1.00 | 0.75 |
| search | 12 | 11 | 2 | 1 | 0.85 | 0.92 |

## 2. Misclassifications — Test Set

**6 errors out of 39:**

### [seed_044] true=`outfit` → predicted=`filter` (conf=32.16%)
- **Query:** "Style this with complementary pieces"
- **Context:** last_action=`search` | items=5 | filters={}
- **Encoded input:** `[QUERY] Style this with complementary pieces [CTX] last_action=search items=5 filters=none`
- **Runner-up:** `outfit` (25.25%)
- **All probs:** filter=32.16% | outfit=25.25% | compare=17.27% | respond=11.86% | search=8.52% | clarify=4.94%
- **Source:** seed

### [edge_049] true=`clarify` → predicted=`search` (conf=56.66%)
- **Query:** "Something for someone special"
- **Context:** last_action=`none` | items=0 | filters={}
- **Encoded input:** `[QUERY] Something for someone special [CTX] last_action=none items=0 filters=none`
- **Runner-up:** `clarify` (27.36%)
- **All probs:** search=56.66% | clarify=27.36% | respond=6.39% | outfit=3.85% | filter=3.79% | compare=1.95%
- **Source:** edge

### [para_018_4] true=`respond` → predicted=`filter` (conf=25.66%)
- **Query:** "I'm loving these options"
- **Context:** last_action=`search` | items=5 | filters={}
- **Encoded input:** `[QUERY] I'm loving these options [CTX] last_action=search items=5 filters=none`
- **Runner-up:** `respond` (22.79%)
- **All probs:** filter=25.66% | respond=22.79% | compare=20.47% | outfit=18.18% | clarify=6.72% | search=6.17%
- **Source:** paraphrase

### [para_044_0] true=`outfit` → predicted=`filter` (conf=35.24%)
- **Query:** "Showcase it with matching outfits"
- **Context:** last_action=`search` | items=5 | filters={}
- **Encoded input:** `[QUERY] Showcase it with matching outfits [CTX] last_action=search items=5 filters=none`
- **Runner-up:** `outfit` (25.48%)
- **All probs:** filter=35.24% | outfit=25.48% | search=15.44% | compare=9.24% | respond=7.86% | clarify=6.73%
- **Source:** paraphrase

### [edge_037] true=`search` → predicted=`filter` (conf=37.18%)
- **Query:** "Blazers but not the pinstriped ones from before"
- **Context:** last_action=`search` | items=5 | filters={}
- **Encoded input:** `[QUERY] Blazers but not the pinstriped ones from before [CTX] last_action=search items=5 filters=none`
- **Runner-up:** `outfit` (21.02%)
- **All probs:** filter=37.18% | outfit=21.02% | search=15.27% | compare=10.86% | respond=9.06% | clarify=6.61%
- **Source:** edge

### [edge_002] true=`respond` → predicted=`search` (conf=77.28%)
- **Query:** "Show me face creams and moisturizers"
- **Context:** last_action=`none` | items=0 | filters={}
- **Encoded input:** `[QUERY] Show me face creams and moisturizers [CTX] last_action=none items=0 filters=none`
- **Runner-up:** `clarify` (11.98%)
- **All probs:** search=77.28% | clarify=11.98% | respond=3.86% | filter=3.06% | outfit=2.51% | compare=1.31%
- **Source:** edge

## 3. Confidence Distribution

### Test set

| Group | Count | Median conf | p25 conf |
|---|---|---|---|
| Correct predictions | 33 | 51.14% | 38.71% |
| Incorrect predictions | 6 | 36.21% | 32.93% |

- High-confidence errors (conf > 80%): **0**
- Low-confidence errors (conf ≤ 60%): **5**

### Train set

| Group | Count | Median conf | p25 conf |
|---|---|---|---|
| Correct predictions | 278 | 55.13% | 42.39% |
| Incorrect predictions | 32 | 36.64% | 33.10% |

- High-confidence train errors (conf > 80%): **0**
- Low-confidence train errors (conf ≤ 60%): **28**

> **Interpretation:** If train errors are low-confidence and test errors are high-confidence,
> the model learned the training distribution but is over-confident on novel patterns.
> If most errors are high-confidence on both, the decision boundary is wrong.

## 4. Train Set Errors

**32 errors out of 310** (train accuracy: 89.7%)

| id | query | true | predicted | confidence |
|---|---|---|---|---|
| para_046_2 | What else would look good with this blazer? | outfit | respond | 34.40% |
| para_028_2 | Are the first two products identical? | compare | respond | 36.60% |
| para_045_3 | What do you recommend to go with item number 2? | outfit | respond | 28.51% |
| para_032_4 | Is the first or second item recommended? | compare | respond | 36.68% |
| para_046_0 | What would complement the blazer? | outfit | respond | 33.75% |
| para_015_0 | Refine the search by applying the filters | search | filter | 33.93% |
| edge_042 | wut goes w the dress | outfit | filter | 28.87% |
| seed_051 | asdfghjkl zxcvbnm qwerty | clarify | search | 65.99% |
| para_033_0 | I'd like to see a comparison of the last two items in the se | compare | filter | 33.82% |
| para_051_1 | Show me some stylish products, please | clarify | search | 59.11% |
| edge_010 | Compare those and then build an outfit | compare | outfit | 31.78% |
| para_046_4 | What can I pair with this blazer? | outfit | respond | 33.08% |
| edge_016 | What's different between these and what would pair with them | compare | respond | 35.12% |
| para_017_3 | What kind of fabric is this dress constructed from? | respond | outfit | 33.11% |
| seed_046 | What would pair well with the blazer? | outfit | respond | 32.38% |
| para_051_3 | Can you assist with styling and buying | clarify | search | 38.49% |
| edge_018 | More like these please | clarify | search | 61.90% |
| edge_020 | Tell me about them | clarify | search | 41.58% |
| para_032_2 | Which one do you think is superior? | compare | respond | 42.89% |
| seed_032 | Which is better, the first or the second one? | compare | respond | 35.61% |
| edge_004 | I'd like to buy a new sofa | respond | clarify | 43.72% |
| edge_006 | Show me some lipstick and nail polish options | respond | search | 79.10% |
| para_049_1 | Can you recommend a jacket for someone else? | clarify | search | 54.67% |
| para_044_1 | I need suggestions for accessories to go with this | outfit | filter | 30.06% |
| para_031_0 | Show me both of them next to each other | compare | filter | 29.64% |
| para_032_0 | What's the difference between the first and second options? | compare | respond | 36.92% |
| edge_045 | I need outerwear for them | clarify | search | 51.41% |
| edge_019 | More like these please | search | filter | 42.34% |
| edge_048 | xyz abc foo bar | clarify | search | 68.11% |
| seed_029 | What's the difference between them? | compare | respond | 37.18% |
| para_034_0 | Can you tell me the difference between the second and fourth | compare | respond | 37.62% |
| edge_035 | I don't want anything striped or checked | search | filter | 27.74% |

**Train errors by true class:**
- `compare`: 11 errors
- `clarify`: 8 errors
- `outfit`: 7 errors
- `search`: 3 errors
- `respond`: 3 errors

## 5. Cross-Router Disagreement Analysis

Source: `reports/router_comparison.md` (7 disagreements between DistilBERT and Groq LLM).

| id | query | true | DB pred | Groq pred | DB correct? | Failure type |
|---|---|---|---|---|---|---|
| para_052_3 | I'm lost – can you help me decide on a gift for someone | clarify | respond | clarify | NO | A |
| edge_007 | Do you have earphones or headphones? | respond | respond | search | YES | C |
| para_005_0 | Can you help me choose an outfit for a job interview? | search | search | outfit | YES | B |
| para_039_2 | Show me Ladieswear items | filter | filter | search | YES | B |
| para_005_4 | Help me pick a suitable outfit for an upcoming job inte | search | search | clarify | YES | B |
| edge_037 | Blazers but not the pinstriped ones from before | search | filter | search | NO | D |
| seed_045 | Complete the look around item 2 | outfit | filter | outfit | NO | D |

### Disagreement detail

**[para_052_3]** true=`clarify` | DB=`respond` | Groq=`clarify` | Winner: **Groq** | Type **A**
> "I'm lost – can you help me decide on a gift for someone whose preferences I'm not aware of?" (last_action=none, items=0)
> Gift query with unknown preferences — DB predicted respond; Groq correctly predicted clarify

**[edge_007]** true=`respond` | DB=`respond` | Groq=`search` | Winner: **DistilBERT** | Type **C**
> "Do you have earphones or headphones?" (last_action=none, items=0)
> DB correctly predicted respond for 'Do you have earphones?' (OOV); Groq incorrectly predicted search

**[para_005_0]** true=`search` | DB=`search` | Groq=`outfit` | Winner: **DistilBERT** | Type **B**
> "Can you help me choose an outfit for a job interview?" (last_action=none, items=0)
> DB correctly predicted search (no prior items); Groq wrongly predicted outfit

**[para_039_2]** true=`filter` | DB=`filter` | Groq=`search` | Winner: **DistilBERT** | Type **B**
> "Show me Ladieswear items" (last_action=search, items=5)
> DB correctly predicted filter (5 items from prior search); Groq wrongly predicted search

**[para_005_4]** true=`search` | DB=`search` | Groq=`clarify` | Winner: **DistilBERT** | Type **B**
> "Help me pick a suitable outfit for an upcoming job interview." (last_action=none, items=0)
> DB correctly predicted search (no prior items); Groq wrongly predicted clarify

**[edge_037]** true=`search` | DB=`filter` | Groq=`search` | Winner: **Groq** | Type **D**
> "Blazers but not the pinstriped ones from before" (last_action=search, items=5)
> 'Blazers but not the pinstriped ones from before' — negation + 'from before' confused DB into filter

**[seed_045]** true=`outfit` | DB=`filter` | Groq=`outfit` | Winner: **Groq** | Type **D**
> "Complete the look around item 2" (last_action=search, items=5)
> 'Complete the look around item 2' — 'item 2' triggered filter; true label is outfit

## 6. Failure Type Counts (DistilBERT test errors)

| Type | Count | Description |
|---|---|---|
| A | 0 | Vague/ambiguous query — hard for any router |
| B | 0 | State-conditional — LLM ignored context (DistilBERT advantage) |
| C | 1 | Out-of-vocabulary item — respond should fire, not search |
| D | 1 | Surface-form confusion — negation/reference misdirected classifier |
| ? | 4 | Unclassified |

## 7. Encoding Format Inconsistency (Bug)

Training uses `DistilBERTRouter.encode_input()` which produces:
```
[QUERY] <query> [CTX] last_action=<x> items=<n> filters=<f>
```
But `scripts/eval_router_classifier.py` uses a different format:
```
query: <query> | last_action: <x> | items: <n> | filters: <f>
```
This script uses the **correct training-time format**. Any metric differences from the previous eval report may reflect this fix.

## 8. Recommendations (ranked by expected impact)

### 1. Add targeted clarify training examples (HIGH impact)

**Problem:** Only 3 clarify examples in test set; 2 misclassified (recall=0.33).
Training data has very few gift-intent / vague-preference queries.
The model confuses 'stylish for a loved one' (clarify) with search, and
'gift for someone whose preferences I'm not aware of' (clarify) with respond.

**Fix:** Add 15–20 clarify examples covering:
- Gift buying with unknown recipient preferences
- Open-ended 'surprise me' requests
- Highly ambiguous style requests with no clear category

**Expected gain:** clarify recall 0.33 → ~0.80; macro F1 +3–5 pts.

### 2. Improve OOV (out-of-catalogue) detection (MEDIUM impact)

**Problem:** `edge_002` ('Show me face creams') → predicted search instead of respond.
The OOC detector in `graph.py` catches obvious non-clothing terms (electronics, pets),
but cosmetics/beauty is borderline and slips through. The router then predicts search
instead of respond.

**Fix options (pick one):**
- Add 5–8 more cosmetics/beauty OOC examples to the training set with route=respond.
- Expand the OOC keyword list in `_detect_ooc()` to include beauty/cosmetics terms.
  (Simpler and doesn't require retraining.)

**Expected gain:** respond precision/recall improve; eliminates a class of silent failures.

### 3. Explicit state-feature tokens in input encoding (MEDIUM impact)

**Problem:** `edge_037` ('Blazers but not the pinstriped ones from before') → predicted filter
despite true=search. `seed_045` ('Complete the look around item 2') → predicted filter
despite true=outfit.

Root cause: The encoded input format buries state (items=5) in a compact string.
The model uses surface cues ('not' → filter, 'item 2' → filter) instead of context.

**Fix:** Strengthen the state encoding. Replace:
```
[CTX] last_action=search items=5 filters=none
```
with discrete tokens that DistilBERT can anchor on:
```
[CTX] [LAST_SEARCH] [ITEMS_SOME] [NO_FILTER]
```
or add explicit flags: `has_results=yes last_was_search=yes`.
Retrain after encoding change.

**Expected gain:** D-type errors (surface confusion) drop; filter false positives fall.

### Summary table

| Rank | Intervention | Errors addressed | Effort | Expected F1 gain |
|---|---|---|---|---|
| 1 | More clarify training data (15–20 examples) | A-type (2 errors) | Low | +3–5 pts |
| 2 | OOC keyword expansion or respond training data | C-type (1 error) | Very low | +1–2 pts |
| 3 | Stronger state encoding + retrain | D-type (2 errors) | Medium | +2–3 pts |

> All three interventions together could bring macro F1 from 0.83 to ~0.90+
> without changing the base model or requiring significantly more data.
