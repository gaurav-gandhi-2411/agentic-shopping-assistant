# Production Plan — Agentic Shopping Assistant

**Written:** 2026-05-14
**Assumes:** `PROJECT_AUDIT.md` has been read.
**Scope:** First production release capable of handling real users.

---

## Model Choice

### Recommended configuration

| Role | Model | Rationale |
|---|---|---|
| **Respond** | `claude-sonnet-4-6` via Anthropic SDK | Best reasoning + tool-use at moderate cost. Fashion descriptions are short; latency is acceptable. Beats llama-3.1-8b on nuanced outfit rationale, negative-constraint handling, and grounding compliance. |
| **Reranker** | Same model (prompt-cache the system message) | The reranker system prompt is large and fixed — Anthropic prompt caching eliminates most of the marginal cost. Alternatively, skip the reranker entirely for queries with confident colour/type filters (retrieval order is already correct). |
| **Router** | LLM router (keep current) — DistilBERT deployment abandoned | Three rounds of DistilBERT training (V2→V3→V3.1) stalled at OOD F1=0.63. Train/serve skew (multi-turn state-encoded inputs drop confidence from 0.72 to 0.36) means the cascade still escalates 40-60% of turns to the LLM at the workable threshold of 0.65. Net saving is negligible; the LLM router stays. See PROJECT_AUDIT.md Appendix A. |
| **Conversation summariser** | `claude-haiku-4-5` | Summary is a cheap extraction task. Haiku is ~8× cheaper than Sonnet. |
| **Embeddings** | `all-MiniLM-L6-v2` (current) — keep | 384-dim is fine for 20K items. If the catalogue grows past 200K items, consider `text-embedding-3-small` (OpenAI) for its superior recall. Don't change until you have data showing retrieval quality is the bottleneck. |

### Cost vs. quality tradeoffs

**Current (llama-3.1-8b-instant on Groq):** Free tier (500K TPD), excellent latency (200–400ms/call), but weak on multi-constraint reasoning and prone to hallucinating filter values. The code already has 6+ workaround heuristics to compensate for model limitations (filter remap table, OOC keyword list, code-level loop guard, etc.). Upgrading the respond model eliminates most of this compensatory logic.

**Claude Sonnet 4.6:** ~$3/M input tokens. A typical 10-turn shopping session with search = ~8K tokens total = $0.024/session. At 1K DAU × 3 sessions/day = ~$72/day. This is the right quality/cost tradeoff for a v1 that handles real users.

**Router cost:** The LLM router (llama-3.1-8b-instant / Haiku 4.5) stays. The DistilBERT cascade was abandoned after V3.1 testing; see PROJECT_AUDIT.md Appendix A for the full analysis. With three LLM calls per turn (router + reranker + respond), the actual input token count is ~3,523/turn — not the 2K originally estimated. The revised cost model below accounts for this and for the Haiku/Sonnet split with prompt caching.

---

## Agent Framework

**Recommendation: Keep LangGraph. Fix the graph-per-request bug.**

The current LangGraph StateGraph is the right choice and is already well-designed:
- Conditional edges with code-level guards express the routing logic clearly
- TypedDict state is easy to test
- Streaming mode (bypassing respond_node for token streaming) is a good pattern

What is wrong is not the framework but the usage: `build_graph()` and `builder.compile()` are called per request instead of once at startup. Fix: store the compiled graphs (streaming=True and streaming=False variants) in `api/deps.py` at startup, not in a factory closure.

**Alternatives considered and rejected:**
- **Raw SDK + state machine (no LangGraph):** The current graph is already essentially a hand-written state machine. LangGraph adds clear debugging (graph visualization) and structured node output. No compelling reason to remove it.
- **Claude Agent SDK (Managed Agents):** Useful if you want hosted execution and tracing, but adds vendor lock-in for the agent loop. The current architecture is provider-agnostic. Keep LLM-agnostic orchestration.
- **LangGraph v0.4+ with memory store:** LangGraph has its own persistence layer now. Migrating to it would eliminate the custom `PostgresSessionStore`. Worth evaluating after v1, not before.

---

## Tools & Capabilities to Add

Listed in order of UX impact. Each is independent — add in phases.

### 1. Price range filtering (P1, high impact)
**Gap:** The single most common user question the system can't answer. H&M price data is available via the Kaggle dataset (not included currently) or via their public API.
**Implementation:** Add `price_min` / `price_max` columns to `catalogue.parquet`. Add `price_range` filter in the router prompt vocabulary. Post-filter in `hybrid_search.py`.
**Impact:** Removes the "I don't have pricing information" disclaimer that currently appears ~30% of turns.

### 2. Thumbs-up / thumbs-down feedback per message (P1, high impact)
**Gap:** The `feedback` table is schema-ready with `rating` (+1/-1) and `comment`. The UI has no implementation.
**Implementation:** Add 👍/👎 buttons to `MessageBubble.tsx`. Add `POST /messages/{id}/feedback` route. Wire to the `feedback` table.
**Impact:** Creates the quality signal needed to measure production performance and identify bad turns. Without this, you're flying blind post-launch.

### 3. Wishlist / save item (P1, medium impact)
**Gap:** No way for users to save items between sessions.
**Implementation:** Add a `saved_items` table (`user_id`, `article_id`, `saved_at`). Add a bookmark icon to `ItemCard`. Add `GET /saved` and `POST/DELETE /saved/{article_id}` routes.
**Impact:** Converts the assistant from a single-session tool to something users return to.

### 4. Real-time product search fallback (P2, high impact)
**Gap:** The H&M catalogue is static (20K items, no prices, no stock). Users occasionally ask about items not in the index.
**Implementation:** Wire the [H&M Product Discovery API](https://developer.hm.com) as a live search fallback when local retrieval returns < 3 results. Or use SerpAPI's Google Shopping results as a real-time layer.
**Impact:** Turns the assistant from a demo into something useful for actual shopping decisions.

### 5. "Build a complete outfit from scratch" (P1, medium impact)
**Gap:** The current `suggest_outfit` requires a seed item. Users often want a full outfit without knowing a starting item ("I need something for a garden party wedding").
**Implementation:** Modify the outfit flow: when `items_retrieved=0` and the user asks for an outfit, run a search for a hero item first, then chain into `suggest_outfit`. (The router prompt already has a rule for this; the gap is in the outfit node graceful handling.)
**Impact:** Closes the most common outfit-flow failure path.

### 6. Comparison table UI (P2, medium impact)
**Gap:** `compare_items` returns data but the frontend shows it as an unstructured assistant text bubble. There's no side-by-side comparison view.
**Implementation:** Detect when the WS `routing` frame has `action: "compare"`, then render retrieved items in a 2–3 column comparison table component instead of the standard card list.
**Impact:** Makes the compare feature actually useful; currently the LLM prose comparison is inferior to a visual side-by-side.

### 7. Deal alerts / price drop notifications (P2, low impact)
**Gap:** No async notification path.
**Implementation:** `saved_items` + a daily job that checks prices and sends email via Supabase edge functions or Resend.
**Impact:** Drives return visits. Deferred until real-time price data (item 4) is in place.

### 8. Image-based search (P2, future)
**Gap:** Users can't search by photo ("I saw a jacket like this on Instagram").
**Implementation:** CLIP embeddings stored alongside text embeddings in FAISS. `POST /chat/image` endpoint accepting image upload → embed → dense search.
**Impact:** Significant UX differentiator. Complexity is high — defer to v1.1.

---

## Memory & Personalization

### Short-term (in-session) — current state is adequate
The 6-turn rolling window with LLM summarisation at 12-turn trigger is correct. The session is persisted to Postgres. No changes needed here.

### Long-term (cross-session) — this is entirely missing
There is no user preference model. The agent doesn't know that a returning user always shops Ladieswear, prefers dark colours, or dislikes synthetic fabrics.

**Recommended design (simple, not ML):**
1. After each session, extract preferences from the summary via an LLM call: `{"preferred_groups": ["Ladieswear"], "preferred_colours": ["Black", "Dark Blue"], "avoided_categories": ["Shorts"]}`.
2. Store in a `user_preferences` table (JSONB, one row per user, updated incrementally).
3. On each new session, prepend a `<preferences>` block to the router system prompt and the respond prompt.

**What not to do:** Don't train a collaborative filtering model. The catalogue is static and small. Rule-based preference extraction from conversation summaries is sufficient for v1 and v1.1.

**Session continuity:** When a user returns to the app and opens a past conversation, restore `retrieved_items` and `filters` from Postgres (already done) AND restore the conversation's summary to the `ConversationMemory` object (already done via `restore_summary()`). The gap is that `ConversationMemory._summary_computed_at` is reset on each new session object creation — but `restore_summary()` is called by `PostgresSessionStore.get()` so this is actually fine. No change needed.

---

## Retrieval

### Current state
FAISS `IndexFlatIP` + BM25Okapi, RRF fusion, post-filter by facets. This is good for the current scale (20K items).

### What to improve

**Hybrid search weight tuning:** RRF treats dense and sparse equally. Add a configurable `alpha` parameter (dense weight vs. sparse weight) and tune it against the eval suite. For fashion, semantic similarity (dense) typically outperforms keyword matching (sparse) — alpha=0.7 dense / 0.3 sparse is a reasonable starting point.

**Reranker skip heuristic:** When the router applies an exact facet filter (e.g., `colour_group_name=Black, product_type_name=Dress`) the RRF+filter result set is already well-ranked. Skip the LLM reranker for turns where `len(effective_filters) >= 2`. Saves one LLM call per filtered search.

**Switch to HNSW if catalogue grows:** `IndexFlatIP` is exhaustive scan — O(N) per query. Fine at 20K, slow at 200K. `faiss.IndexHNSWFlat` is approximate but 10–100× faster at scale. Migration is a one-time rebuild: `scripts/01_build_retrieval.py`.

**No RAG needed currently.** The catalogue has structured metadata (name, colour, type, description). RAG is overkill for structured retrieval. If you add unstructured content (styling blog posts, editorial lookbooks, user reviews), add a separate vector store for that content.

**Vector store choice if catalogue scales:** Keep FAISS for self-hosted. If you move to managed infra: Pinecone Serverless is cost-effective up to ~10M vectors; Supabase pgvector works well up to ~1M vectors with proper HNSW index configuration and is already in your stack.

---

## Evals

The existing eval harness is the right foundation. Extend it rather than replace it.

### Gaps in the current 32-query suite

1. **No multi-turn tests for preference persistence.** Add 5 tests that verify the second turn uses information from the first: "Show me blue dresses" → "More formal ones" → check that `colour_group_name=Blue` is still active.
2. **No adversarial prompt injection tests.** Add 3 queries with embedded JSON-like strings. Verify the router still returns a valid action.
3. **No latency regression tests.** Add a `max_latency_seconds` check to the YAML schema. Flag if any turn exceeds 10 seconds.
4. **No streaming-specific tests.** The harness only calls `agent.invoke()` (non-streaming). Add a `test_ws_streaming.py` integration test that opens a WS connection, sends a message, and verifies the frame sequence (session → routing → items → token+ → done).
5. **No feedback-quality tests.** After enabling 👍/👎, add automated LLM-as-judge scoring: for each eval query, ask Claude to rate the assistant response on a 1–5 scale and verify median ≥ 4.

### Golden query corpus
The existing 32 queries are skewed toward edge cases (negation, OOC, tool behaviour). Add 20 "golden path" queries (ordinary search, refinement, outfit) that represent what real users will actually ask. These should never regress.

### CI integration
Add `pytest --provider=ollama` (cheap, no API key) running the 32-query suite on every PR via GitHub Actions. Any regression blocks merge. Accept that Ollama results may differ from Groq but set the bar appropriately.

---

## UI / UX

### Current state
Next.js 15 + minimal Tailwind components + TanStack Query. The architecture is correct. The implementation is incomplete.

### Required before launch (P0/P1)

1. **Add `react-markdown`** to `MessageBubble.tsx`. The agent produces markdown; it should render as formatted text. Use `remarkGfm` for tables (comparison output). Sanitize with `rehype-sanitize` to prevent XSS from any future external content injection.

2. **Replace `<img>` with `next/image`** in `ItemCard` and `SimilarItemRow`. Add `width={64} height={80} sizes="64px"`. This enables lazy loading and WebP serving — important since images are served from Fly.io (no CDN).

3. **Add message input character limit** in `ChatInput.tsx`. Hard limit at 2000 chars with a visible counter at 1500.

4. **Add WS reconnect logic** in `useChatStream`. On `ws.onerror`, retry up to 3 times with 1s/2s/4s backoff before showing the error message.

5. **Product card "Style this" button.** Currently the agent's "Pick one and I can put together a complete look" prompt has no corresponding button in the Next.js UI (only in the Streamlit app). Add a "Style this" action to `ItemCard` that sends a pre-filled message.

### Nice-to-have v1.1 (P2)

- **Skeleton loaders** for product cards and conversation list
- **Comparison view:** 2-column table layout when the WS routing frame is `compare`
- **Filter chips** below the chat input showing active filters (colour, type) with ✕ to remove
- **Empty state** with suggestion chips on fresh conversation (currently only in Streamlit)
- **Mobile nav:** the sidebar pushes the chat area off-screen on mobile — needs a drawer pattern
- **Keyboard shortcuts:** Cmd+K to new conversation, Cmd+/ to focus chat input

### Production UI stack recommendation
The current stack (Next.js 15 + Tailwind + shadcn-lite) is correct. Do not change it. Add the missing shadcn components (`Dialog`, `Sheet` for mobile sidebar, `Tooltip`) rather than building custom.

---

## Observability

### P0 — Required before any prod traffic

**Sentry error tracking:** Add `@sentry/nextjs` to the frontend and `sentry-sdk[fastapi]` to the backend. Capture unhandled exceptions with user_id and conversation_id context. Free tier covers 5K errors/month — more than enough at v1 scale.

```python
# api/main.py — add to lifespan
import sentry_sdk
sentry_sdk.init(dsn=os.environ.get("SENTRY_DSN"), traces_sample_rate=0.1)
```

### P1 — Add within first week of prod traffic

**Structured logging already exists** (`api/logging_config.py`). Extend it to include:
- `user_id` (add to access_log middleware, not the token itself)
- `n_items` and `action` (already present on the chat log line)
- `llm_tokens_in` / `llm_tokens_out` (requires tracking in `GroqClient.chat()`)
- `latency_ms` per node (router, search, reranker, respond separately)

**LangSmith tracing** is already wired (set `LANGSMITH_API_KEY`). Enable it in production — it gives per-node latency and the full prompt/response for debugging. Free tier covers 5K traces/month.

**Cost tracking:** Add a `TurnCost` dataclass in `GroqClient` that accumulates `(model, input_tokens, output_tokens, cost_usd)` per call and logs it on turn completion. Aggregate in a daily cron job.

### P2 — Nice to have

- **Grafana + Prometheus** for latency histograms if you move to a larger deployment
- **Langfuse** as a self-hosted LangSmith alternative with better cost tracking
- **Fly.io machine metrics** (CPU, memory) visible in fly.io dashboard — enable with `fly dashboard`

---

## Safety & Guardrails

### Prompt injection
The current OOC keyword list and JSON-depth-tracking parser are good but not sufficient.

**Mitigate:**
1. Cap injected user content in prompts at 2000 chars (match the input length limit).
2. Wrap user query in a delimited block in the router prompt: `<user_query>{query}</user_query>`. Instruct the model to treat content inside the tags as data, not instructions.
3. In the grounding layer (`grounding.py`), add a check: if the response contains `<` or `>` characters that don't appear in any retrieved item, flag and strip (basic XSS/injection signal).

### PII handling
The conversation content (messages) is stored verbatim in Postgres. Users may share personal details (address, credit card, health info) in the chat.

**Minimum for v1:**
1. Add a Privacy Policy that discloses data storage.
2. Implement `DELETE /conversations/{id}` (already exists) and wire a "Delete conversation" button in the UI.
3. If you use LangSmith tracing: ensure user messages are not sent unmasked. LangSmith supports masking — enable it for any field that matches PII patterns.

### Output validation
`validate_response()` in `grounding.py` already handles the main hallucination vectors (price, size, material). Extend it with one more check: if the response mentions a specific product name that is **not** in `retrieved_items`, that sentence is a hallucination and should be replaced with a generic item reference.

### Rate limits
- Per-user: max 10 chat requests / minute (asyncio in-memory counter; Redis for multi-instance)
- Per-IP: max 20 requests / minute (FastAPI middleware; `slowapi` library)
- Input: max 2000 chars per message

---

## Infra & Deployment

### Backend (API)

**Current:** Fly.io, 512MB RAM, 1 shared CPU — too small (see BUG-1).
**Recommended:** Fly.io `performance-1x` (1 CPU, 2GB RAM). Monthly cost: ~$17. This is the minimum viable size for the loaded process.

**Alternative if you need more control:** Fly.io `performance-2x` (2 CPU, 4GB RAM, $34/mo) allows running 2–4 uvicorn workers, handling bursts without queuing.

**Data indices:** Move `data/processed/` off the Docker image to a Fly.io persistent volume (`fly volumes create`) or Cloudflare R2 bucket. Load at startup. This shrinks the Docker image from ~600MB to ~100MB and decouples index updates from code deploys.

**Docker build:** Pin to `python:3.11-slim-bookworm`. The current `build-essential` purge is correct. Add `--platform linux/amd64` to the build command for M-chip Mac developers.

### Frontend (Next.js)

**Deploy to Vercel.** It's the natural home for Next.js 15. Free tier covers personal projects. If you need custom domains + CORS with the Fly.io API, set `CORS_ORIGINS` in Fly env to include the Vercel preview and production URLs.

**Environment variables:** Manage via Vercel dashboard (not `.env.local` checked in). The three variables needed are: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_BACKEND_URL`.

### CI/CD

1. **GitHub Actions workflow:**
   - On PR: `pytest -m 'not requires_ollama'` (unit tests, no API key needed) + ruff lint + mypy
   - On merge to `master`: build Docker image, push to Fly.io registry, deploy
   - On merge to `master`: `vercel deploy --prod` (or let Vercel's GitHub integration handle it)

2. **Secrets management:** Use GitHub Secrets for `FLY_API_TOKEN`, `GROQ_API_KEY`, `SENTRY_DSN`. Never commit `.env`.

3. **Health check on deploy:** The `fly deploy` command uses the `/healthz` probe (30s grace). Add `/readyz` to the deploy check so a deploy only succeeds if indices are loaded.

### Caching strategy

Currently zero caching. Add:
1. **Query-level cache:** `functools.lru_cache` on `retriever.search()` keyed by `(query_hash, filters_hash)`, TTL = 1 hour, max 500 entries. Repeat queries (e.g., same search term in two sessions) are free.
2. **LLM prompt cache:** Use Anthropic's prompt caching for the reranker system prompt (it's large and fixed). This reduces reranker cost by ~80% after the first call.

---

## Security

### JWT / Auth

1. **WS ticket endpoint:** Add `POST /auth/ws-ticket` that returns a 60-second nonce. The frontend calls this HTTP endpoint first (with the Bearer JWT), gets a nonce, and passes the nonce as `?ticket=<nonce>` instead of the full JWT. The WS handler exchanges the nonce for the user_id via a short-lived in-memory map. Eliminates JWT-in-URL log exposure.

2. **Token refresh:** The current flow passes the Supabase JWT at WS open time. If a session lasts >1 hour, the token expires mid-session. Add `supabase.auth.onAuthStateChange()` listener in the frontend to update the WS URL on token refresh.

3. **RLS end-to-end test:** The migration creates RLS policies but notes they're untested on Supabase. Add an integration test that connects as two different users and verifies User A can't read User B's conversations via the API. Do this before allowing multiple real users.

### API key scoping

- `GROQ_API_KEY`: Groq does not currently support scoped keys. Monitor usage via Groq dashboard daily alerts.
- `SUPABASE_URL` / anon key: The anon key is public (browser). The service role key (for admin operations) must never be in the frontend or API — it goes only in migration scripts.

### Dependency audit

Run `pip-audit` in CI on every PR. Add to `pyproject.toml`:
```toml
[tool.pip-audit]
requirement-files = ["requirements.txt"]
```

Run `npm audit --audit-level=moderate` in the frontend CI step.

---

## Cost Model

**Revised 2026-05-14.** Replaces the earlier estimate that assumed 2K tokens/turn and DistilBERT routing. Neither assumption holds — see below.

### Token math per turn (3 LLM calls, all-LLM routing)

The three LLM calls per typical search turn break down as follows. Token counts are measured from the actual prompt templates in `src/agents/graph.py` and `src/agents/reranker.py`.

| LLM Call | Input tokens | Output tokens | Static cacheable prefix |
|---|---|---|---|
| Router | 1,233 | 108 | 875 (ROUTER_PROMPT template) |
| Reranker | 1,175 | 120 | 850 (_SYSTEM prompt) |
| Respond | 1,115 | 200 | 700 (RESPOND_PROMPT template) |
| **Total** | **3,523** | **428** | **2,425 (69% of input)** |

**Why 2K was too low:** The router prompt template alone is ~875 tokens before any context or user query is appended. Three calls × ~1,175 average each = 3,523, not 2,000.

Dynamic (non-cached) input per turn: 3,523 − 2,425 = **1,098 tokens**.

**DistilBERT routing is NOT assumed.** The cascade router was tested through V3.1 and abandoned — OOD F1 stalled at 0.63 and train/serve skew causes 40-60% LLM escalation at any workable threshold. Full analysis in PROJECT_AUDIT.md Appendix A. All turns use the LLM router.

### Model assignment

- **Haiku 4.5** for all three calls on ~80% of turns (simple search, refine, OOC)
- **Sonnet 4.6** for reranker + respond only (Haiku still handles router) on ~20% of turns (outfit assembly, compare, multi-step clarify chains)
- **Prompt caching** on all three static prefixes — active after the first call per session warms the prefix

Pricing used: Haiku 4.5 = $1.00/M input, $5.00/M output, $0.10/M cache read, $1.25/M cache write (5-min TTL). Sonnet 4.6 = $3.00/M input, $15.00/M output, $0.30/M cache read, $3.75/M cache write.

### Per-turn cost (cache-warm)

| Turn type | Call breakdown | Cost |
|---|---|---|
| **Simple — Haiku × 3** | 1,098 dyn × $1.00/M + 2,425 reads × $0.10/M + 428 out × $5.00/M | **$0.00348** |
| **Complex — Haiku router + Sonnet × 2** | Router (Haiku): 358 dyn × $1.00/M + 875 reads × $0.10/M + 108 out × $5.00/M = **$0.00099** · Reranker (Sonnet): 325 dyn × $3.00/M + 850 reads × $0.30/M + 120 out × $15.00/M = **$0.00303** · Respond (Sonnet): 415 dyn × $3.00/M + 700 reads × $0.30/M + 200 out × $15.00/M = **$0.00446** | **$0.00848** |
| **Weighted avg (80% simple / 20% complex)** | 0.8 × $0.00348 + 0.2 × $0.00848 | **$0.00448** |

**Cache impact:** No-cache weighted baseline = $0.00722/turn. Cache reduces this to $0.00448 — a **~38% reduction**.

### DAU breakdown (15 turns/day/user — 3 sessions × 5 turns)

| Scale | Turns/day | LLM cost/day | LLM/mo | Infra/mo | **Total/mo** |
|---|---|---|---|---|---|
| **100 DAU** | 1,500 | $6.72 | $202 | $17 Fly + $0 Vercel | **~$219** |
| **1K DAU** | 15,000 | $67.20 | $2,016 | $34 Fly + $20 Vercel | **~$2,070** |
| **10K DAU** | 150,000 | $672.00 | $20,160 | $200 Fly cluster + $100 Vercel | **~$20,460** |

### Cost reduction levers

At all scales, LLM cost is ≥ 97% of total spend. The levers in order of impact:

1. **Reranker skip heuristic (P2):** When ≥ 2 hard filters are active, RRF + filter already produces a well-ranked result — skip the LLM reranker. Applies to ~30% of complex turns. Saves one Sonnet call ($0.00446) on ~6% of all turns → ~$0.00027/turn average, ~9% cost reduction.

2. **Query-result cache (P2):** Cache `hybrid_search()` keyed by `(query_hash, filters_hash)` with 1h TTL. With a bounded 20K catalogue, repeat queries are common. Estimated 20% hit rate → eliminates the reranker call on 20% of turns. ~$0.00078/turn average saving.

3. **Tighten Haiku/Sonnet split (P2):** Narrow "complex" to outfit-node turns only (not all compare or clarify). Shifts 90/10 instead of 80/20 → saves ~$0.00011/turn.

4. **Long-term: watch for multi-turn context growth.** The 1,098 dynamic tokens per turn assumes a 6-turn rolling window (~200 tokens of context). At 12 turns (just before summarisation kicks in), dynamic context roughly doubles. The $0.00388 average is a midpoint estimate — early turns are cheaper, late turns are more expensive.

**For v1 launch (100 DAU):** ~$192/mo is well within range for a solo or small-team project. Break-even at 1K DAU ($1,801/mo) requires ~$2/active user/month in revenue.

---

## Roadmap

### P0 — Must fix before any real user traffic

| Item | File(s) | Effort | Impact |
|---|---|---|---|
| Upsize Fly.io machine to 2GB | `fly.toml` | 5 min | Prevents OOM crash |
| Compile graph once at startup | `api/deps.py` | 30 min | Removes per-request overhead |
| Add `max_length=2000` to `ChatRequest` | `api/schemas.py` | 5 min | Prevents token bombing |
| Add `max_length` to `ChatInput.tsx` | `frontend/components/chat/ChatInput.tsx` | 10 min | Consistent with backend limit |
| Replace 500 error detail with generic message | `api/routes/chat.py` | 5 min | Stops error leakage |
| Add `GroqClient.chat_stream()` try/except | `src/llm/client.py` | 20 min | Prevents silent broken WS responses |
| Fix `NEXT_PUBLIC_BACKEND_URL` port | `frontend/.env.local.example` | 2 min | Fixes local dev frontend→API |
| Add Sentry to backend + frontend | `api/main.py`, `frontend/` | 1h | Visible production errors |
| Add per-user rate limit (10 req/min) | `api/routes/chat.py` + slowapi | 2h | Prevents quota exhaustion |

### P1 — Launch blockers (before opening to real users)

| Item | Effort | Impact |
|---|---|---|
| ~~Wire DistilBERT router~~ — **REMOVED** (cascade abandoned; OOD F1=0.63, train/serve skew forces 40-60% LLM escalation; see Appendix A in PROJECT_AUDIT.md) | — | — |
| WS ticket endpoint (stop putting JWT in URL) | 4h | Eliminates JWT log exposure |
| Fix `list_conversations` N+1 queries → single JOIN | 2h | API correctness at scale |
| Fix `list_conversations` sort (use `updated_at`, not message_count) | 30 min | Correct conversation ordering |
| Fix `_is_public` not persisted to Postgres | 1h | Data integrity |
| Add `react-markdown` to `MessageBubble` | 1h | Renders formatted assistant responses |
| Replace `<img>` with `next/image` in item cards | 1h | Lazy loading + image optimization |
| Add WS reconnect with backoff in `useChatStream` | 2h | UX: no silent failures |
| "Style this" button in `ItemCard` | 1h | Closes primary outfit flow gap |
| Make `suggest_outfit` deterministic | 30 min | Consistent UX |
| Add 👍/👎 feedback to `MessageBubble` + `POST /messages/{id}/feedback` route | 4h | Quality signal for production |
| RLS end-to-end integration test | 4h | Multi-user data isolation verified |
| Prompt injection: delimit user query in router prompt | 30 min | Basic injection hardening |
| Move retrieval indices to Fly volume / R2 | 3h | Smaller image, faster deploys |
| `pip-audit` + `npm audit` in CI | 1h | Dependency vulnerability scanning |

### P2 — v1.1 / Nice-to-have

| Item | Effort | Notes |
|---|---|---|
| Price data in catalogue (from H&M API or Kaggle extended) | 2d | Removes #1 missing feature |
| Wishlist / saved items | 1d | Drives return visits |
| User preference extraction + cross-session personalization | 2d | Long-term memory |
| Switch respond model to Claude Sonnet 4.6 | 2h | Quality step-change |
| Reranker skip heuristic (≥2 filters → skip reranker) | 1h | Cost reduction |
| Prompt caching for reranker system prompt | 2h | 80% reranker cost reduction |
| Comparison table UI (2-column layout on compare action) | 4h | Makes compare feature usable |
| Skeleton loaders for cards + sidebar | 2h | Perceived performance |
| Filter chips UI (active filters visible + removable) | 3h | Transparency + control |
| Mobile sidebar → Sheet/drawer pattern | 3h | Usable on mobile |
| Prompt injection: strip JSON metacharacters from user query | 30 min | Defence in depth |
| Add pagination to `GET /conversations` and `GET /conversations/{id}` | 4h | Scalability |
| `InMemorySessionStore` TTL + eviction | 1h | Prevents OOM in local dev |
| HNSW index migration in FAISS (if catalogue > 100K items) | 2h | Search latency at scale |
| Embed the empty-state suggestion chips from Streamlit into Next.js frontend | 2h | Better first-session UX |

---

## Explicit Assumptions

1. **The H&M catalogue remains static** at ≤ 20K items for v1. If it grows to 200K+, retrieval architecture needs revisiting.
2. **Groq free tier is acceptable for early access** (< 50 users). Switch to paid before public launch.
3. **Single-region deployment** (Fly `bom` = Mumbai). If significant US or EU traffic emerges, add a second region.
4. **No payment processing** in v1 — the "buy now" CTA will link out to the H&M website, not integrate a cart API.
5. **DistilBERT routing is out of scope for v1.** The cascade router was explicitly abandoned after V3.1 (OOD F1=0.63 < 0.65 target; high-confidence wrong predictions survive all thresholds; train/serve skew reduces practical gains to negligible). Model weights are gitignored and not re-trainable without the original data pipeline. The LLM router is the permanent v1 choice.

---

## What to Do First

Stop and fix BUG-1 (OOM), BUG-2 (graph per request), and BUG-3 (no input limit) before deploying to any real users. These three changes take < 2 hours combined and eliminate the highest-probability failure modes.

After that: Sentry, rate limiting, and the feedback button. You need observability and a quality signal before you can iterate meaningfully on any of the other items.
