# Router Classification Dataset Card

## Overview

| Field | Value |
|-------|-------|
| Total examples | 368 |
| Train / Val / Test | 294 / 37 / 37 |
| Classes | search, compare, filter, clarify, outfit, respond |
| Disk size | 133.1 KB |
| Build time | 1442s |

## Class Distribution

| Route | Count | % |
|-------|------:|--:|
| search | 118 | 32.1% |
| compare | 46 | 12.5% |
| filter | 46 | 12.5% |
| clarify | 31 | 8.4% |
| outfit | 44 | 12.0% |
| respond | 83 | 22.6% |

## Generation Methodology

1. **Seed examples** (53): hand-labeled, covering all 6 routes with realistic
   `last_action`, `items_retrieved`, and `active_filters` context.
2. **Paraphrase expansion** (5× per seed via Groq `llama-3.1-8b-instant`):
   intent-preserving rewrites varying formality, length, and phrasing.
3. **Edge cases** (50): hand-crafted hard cases — OOC queries, multi-intent,
   context-dependent pairs, negation patterns, misspellings.

## Known Limitations

- Paraphrases are **synthetic** — real user queries may differ in distribution.
- `clarify` class is intentionally small (~20 examples) reflecting its rarity in production.
- Context features (`last_action`, `items_retrieved`) are **idealised** — the classifier
  must generalise to intermediate states not captured here.
- Vocabulary is H&M-catalogue specific; may not transfer to other fashion retailers.

## Sample Examples

### `search`

- **query**: `Can you help me find something to wear to a summer garden bash?`  last_action=`none` items_retrieved=`0`
- **query**: `Find me a good winter coat for the season please.`  last_action=`none` items_retrieved=`0`
- **query**: `I'm looking for black evening gowns`  last_action=`none` items_retrieved=`0`
- **query**: `Filter products again with the existing criteria`  last_action=`filter` items_retrieved=`0`

### `compare`

- **query**: `I'd like to see how item 1 compares to item 3`  last_action=`search` items_retrieved=`5`
- **query**: `Item 1 vs. Item 3`  last_action=`search` items_retrieved=`5`
- **query**: `Can you compare these two items?`  last_action=`search` items_retrieved=`5`
- **query**: `What are the key differences between these items?`  last_action=`search` items_retrieved=`5`

### `filter`

- **query**: `Filter those results to black please`  last_action=`search` items_retrieved=`5`
- **query**: `Give me the results filtered to Ladieswear only`  last_action=`search` items_retrieved=`5`
- **query**: `Only show dark blue items`  last_action=`search` items_retrieved=`5`
- **query**: `Limit the search results to Divided products`  last_action=`search` items_retrieved=`5`

### `clarify`

- **query**: `I need a gift but have no idea what they like`  last_action=`none` items_retrieved=`0`
- **query**: `Could you help me find some fashionable items?`  last_action=`none` items_retrieved=`0`
- **query**: `I'm looking for a jacket to buy for them`  last_action=`none` items_retrieved=`0`
- **query**: `I need a jacket for them`  last_action=`none` items_retrieved=`0`

### `outfit`

- **query**: `Put together a cohesive look with this item as the starting point.`  last_action=`search` items_retrieved=`5`
- **query**: `Showcase it with matching outfits`  last_action=`search` items_retrieved=`5`
- **query**: `I need suggestions for accessories to go with this`  last_action=`search` items_retrieved=`5`
- **query**: `Assist me in building an ensemble around that jacket`  last_action=`search` items_retrieved=`5`

### `respond`

- **query**: `Do you sell coffee or any beverages?`  last_action=`none` items_retrieved=`0`
- **query**: `How much will I be paying for this?`  last_action=`search` items_retrieved=`5`
- **query**: `That first one looks good, I'll take it`  last_action=`outfit` items_retrieved=`4`
- **query**: `Can I get this tomorrow?`  last_action=`search` items_retrieved=`5`

## Quality Checks

```
============================================================
QUALITY REPORT
============================================================

FULL distribution (n=368):
  search      118
  compare      46
  filter       46
  clarify      31
  outfit       44
  respond      83

TRAIN distribution (n=294):
  search       94
  compare      37
  filter       37
  clarify      25
  outfit       35
  respond      66

VAL distribution (n=37):
  search       12
  compare       5
  filter        5
  clarify       3
  outfit        4
  respond       8

TEST distribution (n=37):
  search       12
  compare       4
  filter        4
  clarify       3
  outfit        5
  respond       9

============================================================
SAMPLES (5 per class, from full dataset)
============================================================

--- SEARCH ---
  [paraphrase] last=none    items=0 | Can you help me find something to wear to a summer garden bash?
  [paraphrase] last=none    items=0 | Find me a good winter coat for the season please.
  [paraphrase] last=none    items=0 | I'm looking for black evening gowns
  [paraphrase] last=filter  items=0 | Filter products again with the existing criteria
  [paraphrase] last=none    items=0 | Help me pick a suitable outfit for an upcoming job interview.

--- COMPARE ---
  [paraphrase] last=search  items=5 | Item 1 vs. Item 3
  [paraphrase] last=search  items=5 | Can you compare these two items?
  [paraphrase] last=search  items=5 | What are the key differences between these items?
  [seed      ] last=search  items=5 | What's the difference between them?
  [edge      ] last=search  items=5 | What's different between these and what would pair with them?

--- FILTER ---
  [paraphrase] last=search  items=5 | Only show dark blue items
  [paraphrase] last=search  items=5 | Limit the search results to Divided products
  [paraphrase] last=search  items=5 | Show only blazers from the results
  [paraphrase] last=search  items=5 | Show me Ladieswear items
  [paraphrase] last=search  items=5 | Filter search results to show Divided items only

--- CLARIFY ---
  [seed      ] last=none    items=0 | I need a jacket for them
  [paraphrase] last=none    items=0 | Can you recommend a jacket for someone else?
  [seed      ] last=none    items=0 | Something nice for my partner, I'm not sure what
  [paraphrase] last=none    items=0 | I'd love to find a gift for my partner, could you help me?
  [paraphrase] last=none    items=0 | Can you assist with styling and buying

--- OUTFIT ---
  [paraphrase] last=search  items=5 | Can you suggest an entire ensemble like the initial dress?
  [paraphrase] last=search  items=5 | Create a full look with the first piece as the centerpiece.
  [paraphrase] last=search  items=5 | Provide some style suggestions for this jacket
  [seed      ] last=search  items=5 | Style this with complementary pieces
  [paraphrase] last=search  items=5 | Put together an outfit that matches the aesthetic of the first dress

--- RESPOND ---
  [paraphrase] last=outfit  items=4 | I'll choose the first item
  [paraphrase] last=compare items=2 | Great, just what I wanted, thanks
  [paraphrase] last=search  items=5 | Can you deliver this to me by tomorrow?
  [paraphrase] last=none    items=0 | Can you tell me about your return procedure?
  [edge      ] last=none    items=0 | Do you carry bedding and pillows?

============================================================
HIGH-JACCARD PAIRS: none flagged (all paraphrases sufficiently diverse).

```