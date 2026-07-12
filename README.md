# Agentic Shopping Assistant

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Product:** [StyleMaitri](https://stylemaitri.vercel.app) — this repo predates the product rebrand. The deployed frontend (`stylemaitri` Vercel project) and Cloud Run backend (`asa-stylist-api`, project `iconic-reactor-496423-m4`) are both built from this repo's `frontend/` and API code respectively.

White-label conversational shopping assistant for Indian D2C fashion brands. Point `BRAND=snitch`
at a catalogue, deploy one Docker container to Cloud Run, and shoppers get a natural-language
assistant that retrieves real products, quotes real INR prices, and links directly to the brand's
own product pages.

---

## What it does

Hybrid retrieval (FAISS dense + BM25 sparse, fused via Reciprocal Rank Fusion) backs a LangGraph
agent loop with six tools: **search**, **compare**, **filter**, **outfit-bundle**, **clarify**, and
**respond**. Every response is grounded — the agent cannot hallucinate a price, size, or fabric
that is not in the retrieved item data. Each product card carries a **Buy** CTA that opens the
real PDP on the brand's own site.

- **INR pricing** — all prices in ₹; `price_min` / `price_max` filtering supported
- **Indian sizing** — `sizing_system: IN` in the brand config shows Indian size labels
- **Brand-isolated** — one `BRAND=` env var selects the config; all brands share a single Docker
  image and each gets its own Cloud Run service
- **Streaming** — FastAPI WebSocket backend streams tokens to a Next.js 15 frontend
- **Grounded** — `validate_response()` strips any sentence that references a price, size, or
  material not present in the retrieved items

---

## Demoable brands

| Brand | `BRAND=` | Source | Items | Price range | Demo query |
|---|---|---|---|---|---|
| **H&M** | `hm` | Kaggle dataset | ~20 K | SEK | `Black slim fit jeans` |
| **Sample IN** | `sample_in` | Bundled sample | 500 | ₹499–₹4,999 | `Casual kurtas for office` |
| **Myntra** | `myntra` | Kaggle dataset | 14 K | ₹169–₹47,999 | `Black kurtas under ₹1500` |
| **Flipkart** | `flipkart` | Kaggle dataset | 15 K | ₹99–₹7,799 | `Winter jackets for men` |
| **Snitch** | `snitch` | Live Shopify | 15 K | ₹219–₹5,699 | `Oversized streetwear shirts` |
| **Powerlook** | `powerlook` | Live Shopify | 927 | ₹399–₹2,599 | `Formal shirts for office` |
| **Fashor** | `fashor` | Live Shopify | 3.6 K | ₹499–₹7,799 | `Ethnic kurta sets for wedding` |
| **Virgio** | `virgio` | Live Shopify | 1.8 K | ₹359–₹6,298 | `Sustainable linen dresses` |

See **[BRANDS.md](BRANDS.md)** for the full reference sheet including Buy-CTA URL verification and
instructions for adding any new Shopify brand.

---

## Quick Start (under 10 minutes)

Runs the **Snitch** demo — no external credentials needed except a free Groq API key.

### 1. Clone and install

```bash
git clone https://github.com/gaurav-gandhi-2411/agentic-shopping-assistant
cd agentic-shopping-assistant
pip install -r requirements.txt
```

### 2. Configure your Groq API key

```bash
cp .env.example .env
```

Open `.env` and make two changes:

```
GROQ_API_KEY=gsk_...       # your key from console.groq.com (free, no credit card)
LLM_PROVIDER=groq          # uncomment this line to use Groq instead of Ollama
```

### 3. Download the Snitch catalogue and build the retrieval index

```bash
python scripts/download_shopify.py --domain snitch.co.in   # ~30 s, live public API
python scripts/01_build_retrieval.py --brand snitch --sample 0  # ~4 min on CPU
```

### 4. Start the backend

```bash
JWT_VERIFICATION_DISABLED=true BRAND=snitch uvicorn api.main:app --reload
# → Uvicorn running on http://127.0.0.1:8080
```

### 5. Start the frontend (new terminal)

```bash
cd frontend
cp .env.local.example .env.local    # no edits needed for local dev
npm install
npm run dev
# → Ready on http://localhost:3000
```

### 6. Open and try it

Open **http://localhost:3000** and type one of the suggestion chips, or try:

> *"Oversized streetwear shirts under ₹1500"*

Every product card shows a real INR price and a **Buy** button that opens the live Snitch product
page. Multi-turn refinement, comparisons, and outfit bundling all work the same way.

**One-command shortcut** (after running `make install` once):

```bash
make demo BRAND=snitch
```

---

## Brand switching

To switch brands, change `BRAND=` and point to that brand's pre-built index:

```bash
BRAND=fashor uvicorn api.main:app --reload
```

Each brand is defined by a single file, `brands/{slug}.yaml`, which sets the display name,
theme colours, tagline, currency, sizing system, PDP URL template, and suggestion chips. The
Docker image ships all brand configs; only the `BRAND=` environment variable and `INDEX_STORE_URI`
differ between Cloud Run deployments.

**Adding a Shopify brand takes under 10 minutes:**

```bash
python scripts/download_shopify.py --domain yourbrand.com
cp brands/snitch.yaml brands/yourbrand.yaml   # edit display_name, colours, pdp_url_template
python scripts/01_build_retrieval.py --brand yourbrand --sample 0
BRAND=yourbrand uvicorn api.main:app --reload
```

Full guide — Shopify auto-ingestion, CSV-based ingestion, Cloud Run deploy:
**[CLIENT_ONBOARDING.md](CLIENT_ONBOARDING.md)**

---

## Architecture

```
Browser (Next.js 15)
    │  WebSocket /chat/stream  +  REST /conversations/*
    ▼
FastAPI (api/)  ── Supabase RS256 JWT auth ── PostgresSessionStore
    │
    ▼
LangGraph StateGraph
  router_node → search / compare / filter / outfit / clarify / respond
       │                                        │
       ▼                                        ▼
HybridRetriever                            Groq LLM
(FAISS IndexFlatIP + BM25Okapi / RRF)  (llama-3.1-8b-instant)
       │
ConversationMemory (6-turn rolling window, Postgres-backed)
```

The router uses a two-layer control flow: an LLM router classifies intent and a code-level guard
in `route_decision()` enforces deterministic transitions after search/compare nodes return results,
eliminating 6-iteration loops without relying on prompt instructions.

### Tech stack

| Layer | Choice |
|---|---|
| Agent framework | LangGraph StateGraph |
| Dense retrieval | FAISS `IndexFlatIP` + `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Sparse retrieval | `rank_bm25` (BM25Okapi) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| LLM | Groq `llama-3.1-8b-instant` (swap via `config.yaml`; Ollama supported for local dev) |
| API | FastAPI + uvicorn, streaming WebSocket |
| Frontend | Next.js 15, TanStack Query, Tailwind CSS |
| Auth | Supabase RS256 JWT (bypassable in dev via `JWT_VERIFICATION_DISABLED=true`) |
| Deployment | Google Cloud Run (API) + Vercel (frontend) |

**Detailed deploy:** [DEPLOY.md](DEPLOY.md)  
**Brand onboarding:** [CLIENT_ONBOARDING.md](CLIENT_ONBOARDING.md)

---

## Tests

```bash
pytest -m "not requires_ollama"
# → 106 passed, ~50 skipped, 0 errors
```

Tests that need pre-built index files skip automatically on a fresh checkout. Run
`python scripts/01_build_retrieval.py --brand snitch --sample 0` first to un-skip them.

**Eval suite (57 queries):** colour, occasion, season, style, negation, tool behaviour, 20 golden-path
queries, and 5 multi-product queries. Run with:

```bash
python -m eval.run --provider groq
```

---

## Performance Budgets (StyleMaitri frontend)

Per-product numeric performance budgets (rule 15e) — measured baseline and artifact in
`reports/performance-baseline-2026-07-12.md` (Lighthouse 13.4.0, desktop preset, 2026-07-12).

| Budget | Value | Measured baseline |
|---|---|---|
| Initial JS bundle | ≤ 350 KB | 269.5 KB |
| LCP | ≤ 1500 ms | 812 ms |
| TBT | ≤ 200 ms | 2.5 ms |
| CLS | ≤ 0.05 | 0 |

"Feels fast" is not a measurement — these numbers are what perf claims about StyleMaitri get checked against going forward, per rule 65b.

---

## Known limitations

- **Static catalogue snapshot** — the index is a point-in-time snapshot of the catalogue.
  New products require a re-download, re-index, and container restart. There is no live stock
  feed or real-time price sync.
- **Link-out checkout, no cart** — the Buy CTA opens the brand's product page in a new tab.
  The assistant does not integrate with a cart, payment processor, or inventory system.
- **Single-session memory without a database** — conversation history resets on page refresh
  unless `DATABASE_URL` is set in `.env`. With Supabase configured, sessions persist across
  restarts.
- **Outfit suggestions are heuristic** — complement categories and colour matching are rule-based
  (see `src/agents/tools.py`), not trained from co-purchase data. Results are plausible but not
  fashion-expert quality.

---

## License

MIT
