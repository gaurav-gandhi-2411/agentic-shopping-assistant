# Agentic Shopping Assistant

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/gauravgandhi2411/agentic-shopping-assistant)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A multi-turn conversational shopping assistant over the H&M fashion catalogue. Combines hybrid
retrieval (dense + BM25 via Reciprocal Rank Fusion) with a LangGraph agent loop that orchestrates
search, compare, filter, and outfit-bundling tools — all streamed through a single-process Streamlit app.

🔗 **[Live Demo](https://huggingface.co/spaces/gauravgandhi2411/agentic-shopping-assistant)**

---

## Table of Contents

- [Motivation](#motivation)
- [Demo](#demo)
- [Features](#features)
- [Architecture](#architecture)
  - [Components](#components)
  - [Router](#router)
  - [Outfit bundling](#outfit-bundling)
- [Tech Stack](#tech-stack)
- [Evaluation](#evaluation)
- [Corpus](#corpus)
- [Setup](#setup)
  - [Local dev (Ollama)](#local-dev-ollama)
  - [HuggingFace Space deploy](#huggingface-space-deploy)
- [Project Structure](#project-structure)
- [Known limitations](#known-limitations)
- [What I learned](#what-i-learned)
- [License](#license)

---

## Motivation

Modern recommender systems surface items efficiently but offer no conversational affordance — users can't refine results, compare options, or ask follow-up questions without resorting to faceted filters. This project explores whether an LLM-orchestrated agent can make a fashion catalogue feel like a conversation with a knowledgeable shop assistant: understanding vague queries, handling multi-turn refinement, comparing items, and suggesting complementary pieces.

Built entirely on consumer hardware with open-source tooling.

---

## Demo

Screenshots to be added.

| Natural-language search with card UI | Multi-turn refinement + compare | Outfit bundling |
|---|---|---|
| ![](docs/screenshots/01-search.png) | ![](docs/screenshots/02-compare.png) | ![](docs/screenshots/03-outfit.png) |

---

## Features

| Feature | Example |
|---|---|
| Natural-language search | "show me something for a beach wedding, not too formal" |
| Multi-turn refinement | "something more casual" / "in blue instead" |
| Facet filtering | "only black ones" → applies `colour_group_name: Black` |
| Item comparison | "compare the first two you showed me" |
| Outfit bundling | "style this with accessories" / "build an outfit around it" |
| Card UI with images | Thumbnail, metadata caption, expandable description per result |
| One-click card actions | **More like this** and **Style this** buttons per card |
| Prompt chips | Quick-start suggestions on a fresh session |

---

## Architecture

```text
User input
    │
    ▼
┌────────────────────────────────────────────────────────┐
│                    LangGraph Agent                     │
│                                                        │
│  START ──▶ router ──┬──▶ search  ──────────────────┐  │
│       (LLM / DB /  ├──▶ compare ──────────────────┤  │
│        Cascade)    ├──▶ filter  ──────────────────┤  │
│                     ├──▶ outfit  ───────────────▶ END  │
│                     ├──▶ clarify ───────────────▶ END  │
│                     └──▶ respond ───────────────▶ END  │
│                ▲                                    │  │
│                └──────────── (loop) ────────────────┘  │
└────────────────────────────────────────────────────────┘
           │                        │
     HybridRetriever             Groq LLM
   (FAISS + BM25 / RRF)    (llama-3.1-8b-instant)
           │
    ConversationMemory
    (6-turn rolling window)
```

### Components

| Layer | Implementation |
|---|---|
| **LLM** | Groq `llama-3.1-8b-instant` (Space) · Ollama `llama3.1:8b` (local dev) |
| **Dense retrieval** | FAISS `IndexFlatIP` · `all-MiniLM-L6-v2` (384-dim cosine) |
| **Sparse retrieval** | BM25Okapi over cleaned product descriptions |
| **Fusion** | Reciprocal Rank Fusion (k=60) over ranked dense + sparse lists |
| **Agent loop** | LangGraph `StateGraph` — 6-iteration cap + deterministic loop guard |
| **Outfit tool** | Rule-based complement selection with colour-compatibility heuristic |
| **Memory** | Last 6 turns injected into the router prompt each iteration |
| **UI** | Single-process Streamlit · `st.write_stream()` for token-by-token streaming |

### Router

Three interchangeable backends selectable from the sidebar (production uses LLM):

| Provider | Model | Latency | Macro F1 (in-dist) | Cost / 1k |
|---|---|---|---|---|
| **LLM** *(production)* | Groq llama-3.1-8b-instant | ~300–600 ms | — | ~$0.05 |
| Classifier | DistilBERT V2 (6-class, fine-tuned) | ~30 ms (CPU) | 0.93 | $0 |
| Cascade | DistilBERT + Groq fallback | ~30–600 ms | — | ~$0.02 |

The LLM router formats conversation history, last tool used, retrieved item count, and active filters
into a structured prompt and parses one JSON action object. Parameter extraction (search query, filter
key/value, article IDs) is done by the LLM directly.

Control flow has two enforcement layers shared by all routers:

1. **OOC short-circuit** — out-of-catalogue queries bypass the router and force `search → OOC canned response`
2. **Code-level guard in `route_decision()`** — after `search` or `compare` with non-empty `retrieved_items`, forces `respond` regardless of router output (eliminates 6-iteration loops)

### Routing Architecture Exploration

Cascade routing explored in Phase 2: fine-tuned a 6-class DistilBERT classifier (67M params) on augmented stateful training data. Achieved 0.93 macro F1 on in-distribution held-out test. OOD evaluation (60 natural user queries) revealed systematic failures: the model memorizes surface patterns (gift-intent markers, specific phrasing) rather than learning the semantic decision rule (whether a query contains enough shopping intent to retrieve relevant products). After 3 rounds of contrastive augmentation (130 pairs total), OOD macro F1 reached 0.63 but stalled due to whack-a-mole regressions on neighboring patterns. Production deployment requires either (a) a larger base model, (b) a fundamentally different approach to the search/clarify boundary, or (c) training data from actual production queries with diverse natural phrasing. Current production: LLM-only routing with prompt engineering.

| Version | Training examples | In-dist macro F1 | OOD-20 macro F1 | Notes |
|---|---|---|---|---|
| V1 | 388 | 0.8345 | — | Initial fine-tune, no stateful augmentation |
| V2 | 1,006 + augmented | 0.9035 | 0.25 | Stateful augmentation; search/clarify boundary fails OOD |
| V3 | 1,111 | 0.9933 | 0.60 | +110 contrastive pairs; whack-a-mole regressions start |
| V3.1 | 1,106 | 0.9304 | 0.63 | +20 event-verb pairs; stalled, 8 high-confidence wrong predictions |

**What would actually fix it:**

1. **Production query logging + labeling** — 1000+ real queries labeled by the routing rule to expose the full distribution of natural phrasings.
2. **Larger base model** — BERT-large or a small instruction-tuned LLM handles semantic generalization better; DistilBERT's 67M parameters on generic text pretraining aren't suited to subtle intent classification.
3. **Different decision boundary** — use LLM routing for ambiguous first-turn queries (no explicit item noun) and DistilBERT only for stateful routing (filter/compare/outfit/respond) where the signal is structural.

Full exploration: [`reports/v3_1_results.md`](reports/v3_1_results.md).

### Outfit bundling

`suggest_outfit()` classifies the seed item (dress → jacket + accessories; bottom → top + outerwear;
top → bottoms + outerwear), runs two complement searches, then post-filters by colour compatibility:
neutral seeds/complements (black, white, grey, beige) are universally compatible; non-neutral seeds
prefer same-palette complements, falling back to best relevance match if none found.

---

## Tech Stack

| Component | Library |
|---|---|
| Agent framework | LangGraph StateGraph |
| Dense retrieval | FAISS IndexFlatIP + sentence-transformers/all-MiniLM-L6-v2 |
| Sparse retrieval | rank_bm25 (BM25Okapi) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| LLM (production) | Groq API + LLaMA 3.1 8B |
| LLM (local dev) | Ollama + LLaMA 3.1 8B |
| Router classifier | DistilBERT fine-tuned (V1–V3.1, exploration); sidebar-selectable, not production |
| UI | Streamlit (single-process, sidebar router toggle) |
| Deployment | HuggingFace Spaces (CPU basic) |

---

## Evaluation

A 32-query automated test suite covering 6 categories: colour, occasion, season, style, negation, and tool behaviour. Each query has programmatic pass criteria evaluated without manual review.

| Category | Queries | Notes |
|---|---|---|
| Colour | 5 | Exact colour match in retrieval |
| Occasion | 5 | Date night, beach, office, brunch, garden party |
| Season | 5 | Winter outerwear, summer light, autumn layers |
| Style | 5 | Minimalist, smart-casual, feminine, loungewear, basics |
| Negation | 5 | "not black", "no shorts", "other than dresses" |
| Tool behaviour | 7 | Compare, outfit, filter, OOC detection, gender facets |

During evaluation, the style category surfaced a colour-tone matching issue: queries like "minimalist wardrobe in neutral tones" returned items the LLM considered tonally compatible but which fell outside a strict neutral palette. The fix: CIELAB ΔE 2000 colour-distance scoring — items are accepted if their perceptual colour distance from the neutral palette is within a configurable threshold, rather than requiring exact colour name matches. This approach mirrors how human perception works and is more robust to catalogue vocabulary variation.

The harness is reproducible and supports all three router backends:

```bash
python scripts/eval_harness.py --provider groq --router llm
python scripts/eval_harness.py --provider groq --router distilbert
python scripts/eval_harness.py --provider groq --router cascade
```

**Three-router comparison on 32-query eval suite:**

| Category | LLM router *(production)* | DistilBERT router | Cascade (explored)† |
|---|---|---|---|
| Colour | 5/5 | 5/5 | 5/5 |
| Occasion | 5/5 | 4/5 | 5/5 |
| Season | 5/5 | 5/5 | 5/5 |
| Style | 5/5 | 5/5 | 5/5 |
| Negation | 5/5 | 2/5 | 3/5 |
| Tool behaviour | 7/7 | 3/7 | 7/7 |
| **Total** | **32/32 (100%)** | **24/32 (75%)** | **30/32 (94%)** |
| **Median latency** | **43.9s** | **14.6s** | **~31ms** (router) |

†Cascade eval (`eval_results_20260428_groq_v3.json`): 22/32 raw, 2 genuine FAIL (N1, N3), 8 ERROR
from internet connectivity loss; all 8 ERRORs confirmed PASS per LLM baseline → 30/32. Escalation
rate: 0% — DistilBERT confidence ≥ 0.70 on all 32 in-distribution queries. Cascade is not in
production; OOD evaluation (60 queries) showed macro F1 stalling at 0.63. See
[Routing Architecture Exploration](#routing-architecture-exploration) for details.

Detailed results: [`reports/router_comparison.md`](reports/router_comparison.md)

---

## Corpus

**Live Space:** 1,800-item subset of the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
dataset, selected by purchase count. Product images resized to 300 px JPEG-75.

**Local:** Full 20,000-article sample (no images). Rebuild indices with
`python scripts/01_build_retrieval.py`.

---

## Setup

### Local dev (Ollama)

```bash
git clone https://github.com/gaurav-gandhi-2411/agentic-shopping-assistant
cd agentic-shopping-assistant
pip install -r requirements.txt

# Pull local model
ollama pull llama3.1:8b

# Add H&M data: download articles.csv from Kaggle -> data/hm/articles.csv
python scripts/01_build_retrieval.py   # ~2 min on CPU, one-time

# Verify end-to-end
python scripts/02_smoke_test.py

# Launch
streamlit run spaces/app.py
```

### HuggingFace Space deploy

```bash
# 1. Build Space-optimised 1800-item subset + resized images (one-time)
python scripts/03_build_image_subset.py

# 2. Upload data artifacts to the Space repo
python spaces/upload_artifacts.py --repo <user>/<space> --space

# 3. Deploy code (clone-overlay flow)
DEPLOY=$(mktemp -d)
git clone https://huggingface.co/spaces/<user>/<space> "$DEPLOY"
cp spaces/app.py "$DEPLOY/app.py"
cp config.yaml  "$DEPLOY/config.yaml"
cp -r src/. "$DEPLOY/src/"          # note: src/. not src/ -- avoids nested src/src/
git -C "$DEPLOY" add -A
git -C "$DEPLOY" commit -m "deploy"
git -C "$DEPLOY" push
```

Set `GROQ_API_KEY` as a Space secret (`Settings -> Variables and secrets`).

---

## Project Structure

```text
agentic-shopping-assistant/
├── config.yaml
├── requirements.txt
├── models/
│   ├── distilbert_router/          # V1 classifier (0.8345 macro F1)
│   ├── distilbert_router_v2/       # V2 stateful augmentation (0.9035 in-dist, 0.25 OOD)
│   ├── distilbert_router_v3/       # V3 +110 contrastive pairs (OOD-20 F1 0.60)
│   └── distilbert_router_v3_1/     # V3.1 +20 event-verb pairs (OOD-20 F1 0.63; exploration end)
├── src/
│   ├── agents/
│   │   ├── graph.py
│   │   ├── router.py            # LLMRouterBackend + DistilBERTRouterBackend + CascadeRouterBackend
│   │   ├── distilbert_router.py # inference wrapper (LABEL_MAP, NUM_LABELS)
│   │   ├── state.py
│   │   └── tools.py
│   ├── retrieval/
│   │   ├── dense_search.py
│   │   ├── sparse_search.py
│   │   └── hybrid_search.py
│   ├── llm/client.py
│   ├── memory/conversation.py
│   └── catalogue/loader.py
├── scripts/
│   ├── 01_build_retrieval.py
│   ├── 02_smoke_test.py
│   ├── 03_build_image_subset.py
│   ├── train_router_distilbert.py
│   ├── build_router_dataset_v3_1.py   # dataset builder (base-query-level split)
│   ├── generate_contrastive_v3_1.py   # contrastive pair generation for V3.1
│   ├── eval_harness.py                # 32-query automated test suite
│   └── eval_queries.yaml              # query definitions and pass criteria
├── reports/                     # eval results (JSON + Markdown)
├── tests/
└── spaces/
    ├── app.py
    ├── upload_artifacts.py
    └── README.md
```

---

## Known limitations

- Minor UX quirks — on rare occasions, "More like this" and "Style this" buttons require a second click due to Streamlit render timing; follow-up queries with very short prompts can occasionally surface overlapping items with the prior turn.
- **Static catalogue** — the index is pre-built; new items require a full re-index and re-upload.
- **Colour filter precision** — the router maps natural-language colour terms to exact catalogue
  values (`"navy"` → `"Dark Blue"`). If the LLM picks an invalid value, the filter is silently
  dropped and search runs unfiltered.
- **Outfit suggestions are heuristic** — complement categories and colour matching are rule-based,
  not learned from co-purchase data. Results are plausible but not fashion-expert quality.
- **Single-session memory** — conversation history lives in Streamlit session state and resets on
  page refresh or Space restart.
- **1,800-item Space corpus** — the live demo runs on a purchase-count-selected subset for startup
  speed; the full 20,000-item index is available locally.

---

## What I learned

**Deterministic code guards outperform prompt rules for agent control flow.**
The router prompt had a STRICT RULE: after a search that returns results, output `respond`.
The 8B model followed it roughly 70% of the time; on refinement queries it kept re-searching
for all six iterations before hitting the hard cap. One line in `route_decision()` —
`if last_tool in {"search", "compare"} and retrieved_items: return "respond"` — fixed it
completely. For anything structurally important (loop termination, error containment), write a
code-level guard; don't rely on a prompt instruction to hold under all inputs.

**Semantic similarity doesn't model "I've already seen these."**
`"something more casual"` after `"show me summer dresses"` returned the same five dresses, because
`cosine("casual summer dresses", "summer dresses") ≈ 0.99` in MiniLM's embedding space.
The fix required explicit refinement detection (does the new query mention the dominant product
type of prior results?), fetching a 3× candidate pool on refinement turns, then excluding prior
article IDs. Retrieval systems don't intrinsically model user history — that's application logic.

**Two-phase graph + stream is the right Streamlit pattern.**
Running the full LangGraph graph first (all routing and tool calls, no LLM response) then streaming
the final answer with `st.write_stream()` gives predictable progress indicators and avoids
partial-state problems that arise when streaming mid-graph. The cost is that the "Searching..."
spinner blocks until tool calls finish (~1-2 s), but for a shopping assistant that sequence is
natural — the user expects results before commentary.

**`on_click` callbacks, not `pending_query + st.rerun()`, for injecting queries from buttons.**
Setting `session_state.pending_query` then calling `st.rerun()` inside a button handler sets the
value *after* `st.session_state.pop("pending_query", None)` has already run at the top of the same
render pass, requiring two clicks. `on_click=callback` fires the callback *before* Streamlit reruns
the script, so the value is already present when the script reads it — single click, no extra rerun,
no flag needed.

**Small models fabricate plausible-but-wrong catalogue values.**
`llama3.1:8b` routinely output `colour_group_name: "Lightweight"` or `product_type_name:
"Breathable"` — syntactically valid filter JSON, but values absent from the catalogue. Every
downstream filtered search returned zero results with no visible error. The fix: build a set of
valid values per facet at graph-construction time and silently reject out-of-vocabulary values
in the filter node, falling back to an unfiltered search. Defence in depth over prompt instruction.

**Perceptual colour distance (CIELAB ΔE 2000) beats exact string matching for tonal queries.**
The ST1 query ("Minimalist wardrobe pieces in neutral tones") failed with an exact-name check because
the reranker returned items like "Light Pink" (ΔE 24.2 from White) — tonally correct but not in
the palette string list. Switching to CIE ΔE 2000 with a 25-unit threshold correctly accepts
near-neutral colours and rejects non-neutrals (Greenish Khaki, ΔE 46 from any neutral). The
`colour-science` library provides `sRGB_to_XYZ → XYZ_to_Lab → delta_E(method="CIE 2000")` in three
lines; the eval check mirrors the ≥50% threshold used by the existing `colour_match` check.

**Building an evaluation harness is more valuable than fixing bugs ad-hoc.**
Before the harness, every "fix" felt subjective ("does it work better now?"). After: 32 programmatic
criteria with pass/fail signal. The harness caught 5 specific failures with clear root causes
(negation handling, OOC bypass, baby-item leakage, facet vocabulary confusion), and guided targeted
fixes for each. In production, this kind of evaluation pipeline is what separates
"ship it" from "we hope it works."

**HuggingFace Spaces has non-obvious deployment traps.**
`git subtree split --prefix=spaces` pushes `spaces/*` to the Space root but leaves `src/` behind —
the app imports `src.*` and breaks on startup with a generic `ModuleNotFoundError`. The clone-overlay
flow (clone Space repo → copy `src/` alongside `app.py` → push) is more explicit and repeatable.
A second trap: `cp -r src "$DEPLOY/src"` when `$DEPLOY/src/` already exists creates
`$DEPLOY/src/src/`; `cp -r src/. "$DEPLOY/src/"` copies contents only.

---

## License

MIT
