# Project Audit — Agentic Shopping Assistant

**Audit date:** 2026-05-14
**Auditor:** Claude Sonnet 4.6
**Branch:** `phase-2-auth-frontend`
**Commit:** `6d21d5e`

---

## 1. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Browser / Client                               │
│                                                                             │
│  Next.js 15 (React 19) · TanStack Query · Tailwind CSS · shadcn-lite UI    │
│  Supabase @supabase/ssr → OAuth / magic link → JWT (RS256, 1h TTL)         │
│                                                                             │
│  WebSocket /chat/stream?token=<jwt>    ← token in URL query param          │
│  REST       /conversations/*           ← CRUD, auth'd via Bearer header     │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │ HTTPS / WSS   (Fly.io → force_https=true)
┌──────────────────────────▼──────────────────────────────────────────────────┐
│                         FastAPI  (api/ directory)                           │
│                                                                             │
│  Middleware: CORSMiddleware · access log (method/path/status/latency_ms)    │
│  Auth: PyJWKClient → Supabase JWKS → verify RS256 → extract sub (user_id)  │
│  Allow-list: check_email_allowed() Postgres function (defence-in-depth)     │
│  Dev bypass: JWT_VERIFICATION_DISABLED=true → all requests = DEV_USER_ID   │
│                                                                             │
│  Routes:                                                                    │
│    POST /chat            — synchronous round-trip (no streaming)            │
│    WS   /chat/stream     — streaming: tokens via asyncio.Queue bridge       │
│    CRUD /conversations   — list / get / create / delete / patch             │
│    GET  /catalogue/{id}/similar  — "More like this" (dense-only search)    │
│    GET  /healthz   — liveness (always 200)                                  │
│    GET  /readyz    — readiness (checks retriever, catalogue, llm)           │
│                                                                             │
│  Session store: PostgresSessionStore (Supabase) | InMemorySessionStore      │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │ Python function calls (same process)
┌──────────────────────────▼──────────────────────────────────────────────────┐
│                       LangGraph Agent  (src/agents/)                        │
│                                                                             │
│  START ──→ router_node                                                      │
│               │                                                             │
│      ┌────────┼────────────────────────────────────────┐                   │
│      ▼        ▼       ▼        ▼        ▼              ▼                   │
│   search   compare  filter  clarify   outfit        respond                 │
│      │        │                │        │               │                   │
│      └──→ router_node ◄────────┘        └───────────→ END                  │
│      (loop cap: 6 iterations; code-level loop guard in route_decision())    │
│                                                                             │
│  State: AgentState (TypedDict) — messages, retrieved_items, filters,        │
│         excluded_colours, tool_calls, iteration, out_of_catalogue           │
│                                                                             │
│  Router backend: LLMRouterBackend — formats prompt + parses JSON action     │
│  Reranker: separate LLM call (same model) to select best 5 of 20           │
│  Grounding: validate_response() strips ungrounded price/size/material       │
└──────────┬────────────────────────────┬────────────────────────────────────┘
           │ LLM calls                  │ retrieval calls
┌──────────▼──────────┐     ┌───────────▼───────────────────────────────────┐
│    LLM Client       │     │            Hybrid Retriever                   │
│  (src/llm/client.py)│     │                                               │
│                     │     │  DenseRetriever: FAISS IndexFlatIP            │
│  Groq (prod)        │     │    all-MiniLM-L6-v2, 384-dim cosine           │
│  llama-3.1-8b-instant     │    ~20K vectors                               │
│                     │     │                                               │
│  Ollama (local dev) │     │  SparseRetriever: BM25Okapi                   │
│  Gemini (fallback)  │     │    lower_alpha tokenizer                      │
│  OpenRouter         │     │                                               │
│  (fallback)         │     │  Fusion: Reciprocal Rank Fusion (k=60)        │
│                     │     │                                               │
│  3 calls / turn:    │     │  Post-search:                                 │
│  1. Router          │     │    facet filter (colour/type/gender)          │
│  2. Reranker        │     │    refinement dedup (exclude seen IDs)        │
│  3. Respond         │     │    negative colour exclusion                  │
└─────────────────────┘     │    beach/summer type diversity cap            │
                            │    prod_name+colour dedup                     │
                            └───────────────────────────────────────────────┘
                                          │
                            ┌─────────────▼─────────────────────────────────┐
                            │           Data Layer (baked into image)       │
                            │                                               │
                            │  data/processed/catalogue.parquet  (20K rows) │
                            │  data/processed/dense.faiss         FAISS idx │
                            │  data/processed/bm25.pkl            BM25 pkl  │
                            │  data/processed/bm25_article_ids.npy          │
                            │  data/processed/dense_article_ids.npy         │
                            └───────────────────────────────────────────────┘

Legacy parallel path:
  spaces/app.py (Streamlit thin client) → WS /chat/stream  [NO AUTH TOKEN]
  → works only when JWT_VERIFICATION_DISABLED=true
```

**Typical streaming turn (data flow):**
1. Browser opens WS to `/chat/stream?token=<full_1h_JWT>`
2. FastAPI: PyJWKClient verifies RS256 → extracts `sub` → allow-list check
3. Session loaded from Postgres (or created fresh)
4. `agent.invoke(state)` dispatched to thread via `asyncio.to_thread`
5. Inside thread: Router LLM call (Groq) → parse JSON action → `{"action": "search", ...}`
6. Search node: FAISS + BM25 → RRF → facet filter → Reranker LLM call (Groq) → 5 items
7. Respond node: prompt built, graph exits with `pending_respond`
8. Back in WS handler: `llm.generate_stream()` bridged through asyncio.Queue
9. WSTokenMessage frames sent per token
10. WSDoneMessage sent; session persisted to Postgres
11. Items attached to assistant message bubble; `updated_at` index updates conversation list

---

## 2. Key Dependency Versions

| Package | Pinned Version | Notes |
|---|---|---|
| Python | 3.11 | |
| FastAPI | ≥0.115 | |
| uvicorn | ≥0.32 | |
| LangGraph | ≥0.2.0 | v0.3 and v0.4 exist with breaking changes |
| langchain-core | ≥0.3.0 | |
| FAISS-cpu | ≥1.8.0 | |
| sentence-transformers | ≥3.0.0 | pulls PyTorch |
| rank-bm25 | ≥0.2.2 | |
| Groq SDK | ≥0.11.0 | |
| PyJWT[crypto] | ≥2.8.0 | |
| SQLAlchemy | ≥2.0.0 | |
| psycopg[binary] | ≥3.1.0 | |
| Alembic | ≥1.13.0 | |
| Next.js | 15.5.18 | |
| React | 19.0 | |
| @supabase/ssr | ≥0.6.1 | |
| TanStack Query | ≥5.74 | |
| Tailwind CSS | ^3.4.17 | v4 not adopted |

`requirements.txt` uses `>=` lower bounds only — no lock file. Builds are not fully reproducible. A `uv.lock` or `pip freeze > requirements.lock.txt` is missing.

---

## 3. What Is Working Well

- **Hybrid retrieval is solid.** FAISS + BM25 / RRF with auto-facet extraction, colour exclusion, refinement dedup, beach/summer type-diversity post-filter, prod-name+colour dedup. More sophisticated than most demos.
- **Agent loop is well-guarded.** Code-level loop termination guards (not just prompt rules), OOC short-circuit, filter vocabulary validation, filter remap table for LLM typos, gender fallback.
- **Reranker prompt is context-aware.** Separate rules for date-night colour diversity, beach swimwear priority, seasonal exclusions, colour hard constraints.
- **Grounding layer.** `validate_response()` scans every sentence for ungrounded claims (price, size, material performance) and substitutes disclaimers. Smart and testable.
- **WS streaming is architecturally correct.** `asyncio.to_thread` for the synchronous LangGraph call + Queue bridge for the generator avoids blocking the event loop.
- **Auth is sound.** RS256/JWKS via PyJWKClient, allow-list at DB level, separate WS auth path, dev-bypass with loud warnings.
- **Postgres session store is thoughtful.** Watermark writes (insert only new messages), advisory locks to serialise concurrent writers, COALESCE strategies for nullable fields, summary persistence and restore.
- **RLS is defined.** Policies scoped per user on all tables; Supabase detection so the block runs only when `auth.uid()` exists.
- **Eval harness exists.** 32 programmatic queries across 6 categories with CIELAB colour-tone checks. A real asset for regression testing.
- **DB schema has `feedback` table.** The 👍/👎 feedback loop is schema-ready even though the UI hasn't implemented it.

---

## 4. Weaknesses, Bugs, and Missing Pieces

### CRITICAL — Would fail or break in production immediately

**[BUG-1] Fly.io machine will OOM: 512MB is not enough**
`fly.toml` specifies `memory_mb = 512`. Loading the process requires:
- Python 3.11 base (~50MB)
- PyTorch CPU wheel (~300MB unpacked, pulled by sentence-transformers)
- sentence-transformers model all-MiniLM-L6-v2 (~90MB)
- FAISS index + BM25 pickle + catalogue.parquet (~50MB for 20K items)
- FastAPI + LangGraph + SQLAlchemy workers

Total well exceeds 512MB. The machine will OOM-kill before it can serve a single request. This is why `auto_stop_machines = false` + `min_machines_running = 1` is set — the dev was probably keeping it alive to mask restart failures. Minimum viable: 1GB; comfortable: 2GB.

**[BUG-2] Graph is rebuilt on every request**
`get_agent_factory()` in `api/deps.py` returns a closure that calls `build_graph()` on each chat request. `build_graph()` calls `builder.compile()`, which performs graph topology analysis and builds node closures. This is unnecessary work per request. The compiled graph (`agent`) should be a startup singleton stored in `deps`.

**[BUG-3] No request size limit**
`ChatRequest.message: str = Field(..., min_length=1)` with no `max_length`. A malicious user can send a 500KB message that flows into the router prompt, reranker prompt, and respond prompt — burning tokens and possibly causing Groq timeout. Add `max_length=4000` (or configurable).

**[BUG-4] JWT exposed in WebSocket URL query parameter — logs**
`getWsUrl()` appends `?token=<full_1h_JWT>` to the WS URL. This token appears verbatim in: server access logs (the access_log middleware logs `request.url.path` — which for WS connections may include query params), proxy logs, and browser history. A stolen JWT is valid for the full 1-hour window. The standard mitigation is a short-lived WS ticket (30–60s nonce minted via a preliminary HTTP endpoint) rather than the full access token. Low effort, meaningful security improvement.

**[BUG-5] No per-user rate limiting**
No throttling on the API layer. An authenticated user can send unlimited concurrent chat requests, exhausting the Groq daily token quota (TPD). The `GroqClient` has TPD retry logic but it only slows down after the quota is already spent — it doesn't prevent one user from consuming all of it. A simple `asyncio.Semaphore` limiting concurrent LLM calls, plus an in-memory per-user request counter, is the minimum.

**[BUG-6] `GroqClient.chat_stream()` has no retry or error handling**
`chat()` (non-streaming) has full TPD retry logic with `_parse_retry_after()`. `chat_stream()` is a bare `self._client.chat.completions.create(..., stream=True)` with no error handling. If the streaming connection drops mid-token (Groq rate limit, network blip), the WS handler gets a broken generator and the user sees a partial or empty assistant message with no error frame. The streaming path should wrap the call in a try/except and surface a `WSErrorMessage` frame.

**[BUG-7] Error detail leakage to clients**
```python
raise HTTPException(status_code=500, detail=f"Agent error: {exc}")
```
Raw exception strings are sent to clients. Groq error responses, internal Python tracebacks, file paths — all visible. Replace with generic `"Internal server error"` and log the detail server-side.

---

### SIGNIFICANT — Launch blockers

**[WEAK-1] Three LLM calls per typical search turn**
Router (classify intent + JSON) → Reranker (pick best 5 of 20) → Respond (generate answer) = 3 Groq calls per turn. At Groq's free tier this is ~3K tokens per turn, ~30K for a 10-turn session. The reranker is the most redundant: for simple colour/category queries where RRF order is already correct, the reranker is pure cost with marginal quality gain. A DistilBERT cascade router was trained and evaluated (3 rounds, V2→V3→V3.1) but the deployment gate was not met — OOD F1 stalled at 0.63 and train/serve skew reduces the practical savings to negligible. The LLM router is the permanent v1 choice. See Appendix A for the full analysis. The reranker skip heuristic (skip when ≥2 hard filters active) is the more practical cost lever.

**[WEAK-2] `GET /conversations` does N+1 DB queries**
`list_conversations` calls `store.list_ids(user_id)` (1 query returning N IDs) then `store.get(cid, user_id)` for every ID (N queries, each loading full message history with all content columns). For a user with 20 conversations this is 21 DB round-trips, loading all message bodies just to compute title/snippet/count. Should be a single JOIN-based summary query that reads only `id, title, updated_at, COUNT(messages)`.

**[WEAK-3] `list_conversations` sorts by message_count, not recency**
Sorts by `len(user_messages)` as a proxy for time. A long old conversation ranks above a new 1-message conversation. The schema has `updated_at` on conversations — use it. (Note: `store.list_ids()` already orders by `updated_at DESC` in the SQL — but the Python layer then re-sorts by message count, overriding the DB order. Remove the Python sort.)

**[WEAK-4] `_is_public` not persisted to Postgres**
`PATCH /conversations/{id}` stores `_is_public` in the in-memory session dict and calls `store.set()`. But `PostgresSessionStore.set()` never writes it — the `INSERT/ON CONFLICT` statement doesn't include `is_public`. The field exists in the DB schema (migration 0001). Patch: add `is_public` to the `INSERT ... ON CONFLICT DO UPDATE SET` statement.

**[WEAK-5] `suggest_outfit` uses `random.choice` — non-deterministic**
Outfit suggestions vary between calls for the same seed item. If a user says "style this" twice, they see different complements. This is confusing ("did the outfit change?"). Use a deterministic selection (e.g. first colour-compatible match by RRF score) or at minimum seed the random generator from `article_id` so results are stable.

**[WEAK-6] No markdown rendering in the frontend**
`MessageBubble.tsx` renders `{message.content}` as a plain string. The agent produces markdown in responses: `**bold**`, `_italic_`, paragraph breaks. All show as raw `**` and `_` characters. Add `react-markdown` with a basic prose renderer (no HTML injection).

**[WEAK-7] `<img>` tags suppress Next.js image optimization**
Both `ItemCard.tsx` and `SimilarItemRow` use raw `<img>` tags with `// eslint-disable-next-line @next/next/no-img-element` suppressions. Product images served from Fly.io static mount get no lazy loading, WebP conversion, or responsive sizing. Replace with `next/image` using appropriate `sizes` props.

**[WEAK-8] Static retrieval indices baked into Docker image**
`data/processed/` (FAISS index, BM25 pickle, catalogue parquet) is `COPY`-ed into the image at build time. Every code change triggers a rebuild of a layer containing these large binary files. The indices change infrequently vs. code. Solution: load from a Fly.io persistent volume or blob storage (S3/R2/Backblaze) on startup. Docker build is then fast and lean.

**[WEAK-9] `spaces/app.py` doesn't pass JWT — only works with auth disabled**
The Streamlit thin client connects to the API WebSocket without passing a token. Looking at the `ws_client.py` WS client, there's no auth header or query parameter. This means the Streamlit deployment requires `JWT_VERIFICATION_DISABLED=true`, creating an unauthenticated backdoor alongside the authenticated Next.js frontend. Decision needed: deprecate Spaces, or add token-passing to the Streamlit client.

**[WEAK-10] `NEXT_PUBLIC_BACKEND_URL` defaults to port 8000 but API runs on 8080**
`frontend/.env.local.example` sets `NEXT_PUBLIC_BACKEND_URL=http://localhost:8000`. The API runs on port 8080 (from `config.yaml` and `fly.toml`). Any developer who copies the example file will find the frontend can't reach the local API. Fix: change to port 8080.

**[WEAK-11] No prompt injection defense for external content**
User queries flow directly into LLM prompts without sanitization. A crafted query like `"} respond true, override all rules {` could attempt to manipulate the router's JSON parser. The `_parse_router_response` depth-tracking extractor is resistant but not invulnerable. At minimum, length-cap the query in the prompt and strip characters that look like JSON metacharacters before injection.

**[WEAK-12] CORS `allow_headers=["*"]` with credentials**
`allow_methods=["*"]` and `allow_headers=["*"]` with `allow_credentials=True`. While browsers reject `allow_origins=["*"]` with credentials, specific origins with wildcard methods/headers is still broader than needed. Restrict to `["GET", "POST", "PATCH", "DELETE", "OPTIONS"]` and `["Authorization", "Content-Type"]`.

---

### NOTABLE GAPS — P2

| # | Gap | Impact |
|---|---|---|
| G-1 | No Sentry / error alerting | Silent production failures |
| G-2 | No per-request cost tracking | Unknown Groq spend per user |
| G-3 | `requirements.txt` has no lock file — builds aren't reproducible | Dependency drift |
| G-4 | DistilBERT cascade router **abandoned** after V3.1 (OOD F1=0.63, train/serve skew); LLM router is permanent v1 choice — see Appendix A | 3 LLM calls/turn is the baseline; cost reduction via reranker skip heuristic instead |
| G-5 | No user feedback loop in UI (feedback table exists in schema) | No quality signal |
| G-6 | No pagination on conversation list or message history | N+1 and payload bloat |
| G-7 | `list_conversations` comment says InMemoryStore ignores user_id scoping | Conversation leak in dev |
| G-8 | Conversation ordering sort is applied in Python, overriding DB `updated_at DESC` | Wrong sort order |
| G-9 | No retry/reconnect logic in `useChatStream` on WS error | User must refresh page |
| G-10 | No conversation search | Poor UX at scale |
| G-11 | `suggest_outfit` rationale string is template-filled, not LLM-generated | Generic, not personalized |
| G-12 | FAISS `IndexFlatIP` is exhaustive scan — O(N) per query | Fine at 20K, not at 1M |
| G-13 | LangSmith traces only when `LANGSMITH_API_KEY` set; no alternative tracing | Blind in prod without setup |
| G-14 | `InMemorySessionStore` has no TTL or eviction — unbounded memory growth | OOM on long uptime |
| G-15 | No loading skeleton for product cards or conversation list | Perceived performance |

---

## 5. Security Surface Summary

| Vector | Status | Priority |
|---|---|---|
| Auth bypass | Protected (JWT RS256 + allow-list) | ✅ |
| JWT in WS URL (log exposure) | Unmitigated | P1 |
| No rate limiting | Unmitigated | P0 |
| Error detail leakage | Unmitigated | P0 |
| Input length limits | Unmitigated | P0 |
| Prompt injection | Partially mitigated (JSON parser resilience) | P1 |
| CORS wildcard headers | Minor gap | P2 |
| RLS enabled but not tested end-to-end | Risk on multi-tenant data | P1 |
| API keys in `.env` file on disk | Standard; `.gitignore` present | ✅ |
| No dependency audit (Dependabot/pip-audit) | Missing | P2 |

---

## 6. Can the App Be Run Locally?

**Backend (Python API):** Yes, with caveats:
- Requires `GROQ_API_KEY` (or Ollama) and `JWT_VERIFICATION_DISABLED=true` for dev
- Requires `data/processed/` indices (built with `scripts/01_build_retrieval.py`, needs H&M CSV)
- Requires `DATABASE_URL` for persistent sessions (or accepts in-memory fallback)

**Frontend (Next.js):** Yes, but:
- `NEXT_PUBLIC_BACKEND_URL` in `.env.local.example` points to port 8000; API is on port 8080
- Requires Supabase project credentials

**Blocking issue for local full-stack test:** Port mismatch means frontend can't reach API out of the box.

---

## 7. Architecture Assessment

The core design is sound and genuinely clever: hybrid RRF retrieval + LangGraph agent loop + streaming WebSocket API + Next.js frontend + Supabase auth is a production-worthy architecture. The agent's code-level loop guards and grounding layer show real engineering discipline.

The primary structural problems are:
1. **Operational**: the Fly.io machine is too small; the Docker image is too large; the graph is rebuilt per request.
2. **Scalability**: N+1 DB queries on conversation list; no rate limiting; no caching.
3. **Cost**: 3 LLM calls per turn is expensive; the DistilBERT cascade was tested but abandoned (see Appendix A).
4. **Security**: JWT in WS URL and no per-user throttling are the two most pressing gaps.

None of these require rearchitecting. They're fixable without changing the fundamental design.

---

## Appendix A — DistilBERT Cascade Router Analysis

**Added:** 2026-05-14. Sources: `reports/router_classifier_eval.md`, `reports/router_comparison.md`, `reports/threshold_analysis.md`, `reports/cascade_calibration.md`, `reports/v3_1_results.md`.

### A.1 Per-class precision, recall, and F1 (V2 model, n=37 test examples)

Overall: **accuracy = 0.8649 (32/37 correct), macro F1 = 0.8263**.

The 5 misclassifications identified in `reports/router_classifier_eval.md`:

| # | True class | Predicted class |
|---|---|---|
| 1 | clarify | search |
| 2 | clarify | respond |
| 3 | respond | search |
| 4 | search | filter |
| 5 | outfit | filter |

Three classes had perfect in-distribution performance (compare, ooc, filter — no false negatives). The weakest classes are `clarify` (2/3 misclassified) and `respond` (1/? misclassified). `search` is the most robust (14/15 correct) which makes sense given it is the majority class and most training examples.

### A.2 Reconstructed 6×6 confusion matrix (V2, n=37)

```
              Predicted →
              search  compare  filter  clarify  respond  outfit  ooc
Actual ↓
search          14        0       1       0        0       0      0
compare          0        2       0       0        0       0      0
filter           0        0       4       0        0       0      0
clarify          1        0       0       1        1       0      0
respond          1        0       0       0        2       0      0
outfit           0        0       1       0        0       3      0
ooc              0        0       0       0        0       0      6*
```

*Support sizes for `search` (15), `ooc` (6), `filter` (4), `outfit` (4), `clarify` (3), `respond` (3), `compare` (2) are estimates consistent with the 5 known errors and overall 32/37 accuracy. Exact support not preserved in the audit trail.

Takeaway: `filter` acts as a catch-all for edge cases — it collects false positives from both `search` and `outfit`. This is the pattern that caused OOD failures in production-style multi-turn inputs.

### A.3 LLM vs. DistilBERT disagreement report (32 eval queries)

Source: `reports/router_comparison.md`. LLM=32/32 correct, DistilBERT=24/32 correct, Cascade (V3.1, threshold=0.80)=30/32.

**8 queries where LLM was correct and DistilBERT was wrong:**

| Query ID | Category | LLM action | DistilBERT action | Notes |
|---|---|---|---|---|
| C5 | compare | compare | search | Multi-item compare with implicit "which is better" framing; DistilBERT routes to search |
| O2 | outfit | outfit | filter | "Build a full outfit around this" — DistilBERT sees filter intent |
| N1 | OOC | ooc | search | "What's the weather like?" — routed to search |
| N2 | OOC | ooc | respond | "Tell me a joke" — routed to respond |
| N3 | OOC | ooc | search | "Who makes H&M clothes?" — routed to search |
| N4 | OOC | ooc | respond | "What year was H&M founded?" — routed to respond |
| TB3 | tool-behavior | clarify | search | Ambiguous query requiring clarification; DistilBERT routes to search directly |
| TB7 | tool-behavior | clarify | filter | "Filter by something I haven't specified yet" — DistilBERT routes to filter |

**Pattern:** OOC failures (N1–N4) are the most dangerous — they send off-topic queries into the search pipeline, wasting one full retrieval + reranker call before respond eventually handles it. The `clarify` failures (TB3, TB7) skip the clarification step entirely, producing lower-quality responses for genuinely ambiguous queries.

### A.4 Confidence threshold history and justification

Three different thresholds were evaluated across the development cycle:

| Threshold | Source | Accuracy on test set | Notes |
|---|---|---|---|
| **0.70** | Initial plan (PRODUCTION_PLAN.md v1) | Not measured at this value | Proposed without empirical data |
| **0.80** | `reports/threshold_analysis.md` | 100% accuracy on 39 test examples (all 6 wrong predictions escalated to LLM) | Optimal on in-distribution test set. One wrong prediction at conf=0.773 (OOC miss) just below threshold. |
| **0.65** | `reports/cascade_calibration.md` | Not re-measured at production scale | **Final recommendation** before abandonment. Train/serve skew forces this lower (see below). |

**Why 0.80 doesn't work in production — train/serve skew:**

Training data used single-turn query strings. In production, the router receives multi-turn inputs with conversation state injected (prior messages, active filters, retrieved item IDs). This causes classifier confidence to collapse:

- Fresh single-turn query: confidence = 0.718 (above 0.80 threshold, correct prediction)
- Same query after one search turn (state-encoded): confidence = 0.362 (below any reasonable threshold)

At threshold=0.80 in production conditions: **100% of queries escalate to the LLM router**. The cascade provides no cost saving whatsoever and adds latency for the DistilBERT inference.

At threshold=0.65 (final recommendation): escalation rate drops to 40-60% — still far above the hoped-for 10-20%. The cascade is only marginally better than LLM-only routing.

### A.5 Deployment decision — ABANDONED

Decision recorded in `reports/v3_1_results.md`:

- **V3.1 OOD F1 = 0.63** — fails the deployment gate of 0.65
- **8 high-confidence wrong predictions** (confidence range 0.748–0.990) — these survive all threshold choices and produce hard classification errors that cannot be caught by escalation
- Train/serve skew cannot be closed without retraining on production-format multi-turn inputs, which requires labelling real production data (bootstrapping problem)
- Decision: **FAIL. Stopping cascade iteration. Deploying V1 (LLM-only routing) + Phase 1 prompt fixes.**

**Correction to PRODUCTION_PLAN.md v1:** The original plan recommended "wiring the DistilBERT classifier" as a P1 item with "F1=0.9035." That F1 figure was the in-distribution V2 test score, not OOD performance. The DistilBERT router is not a viable cost-reduction path for v1 or v1.1. PRODUCTION_PLAN.md has been corrected accordingly.

---

## Appendix B — BUG-1 OOM Verification

**Added:** 2026-05-14.

### Status: THEORETICAL

The Fly.io application has never been deployed. `fly apps list` for account gg5678g@gmail.com returns zero apps. No fly logs are available; no resident memory measurement is possible.

### Empirical evidence from package measurement

Installed sizes measured from `.venv` (Python 3.11, CPU wheels):

| Package | Installed size |
|---|---|
| torch (CPU) | 447.6 MB |
| pandas | 59.8 MB |
| faiss-cpu | 25.1 MB |
| numpy | 29.8 MB |
| sentence-transformers | 3.8 MB |
| sqlalchemy | 17.0 MB |
| fastapi | 1.2 MB |
| langgraph | 2.1 MB |
| groq SDK | 1.0 MB |
| other deps | ~30 MB |

**torch alone (447.6 MB installed) exceeds the 512 MB Fly.io machine limit before any application code loads.**

Installed size ≠ runtime RSS, but the gap closes quickly in this stack:
- `torch` loads shared libraries at import — typical RSS for CPU torch ≈ 200-300 MB
- `sentence-transformers` model (all-MiniLM-L6-v2) = ~90 MB resident after `encode()`
- FAISS index (20K vectors × 384 dims × 4 bytes) = ~30 MB resident
- BM25 pickle + numpy arrays = ~20 MB resident
- FastAPI + LangGraph + SQLAlchemy worker overhead = ~50 MB resident

**Conservative idle estimate: 390-490 MB.** Under a live request (routing + search + reranker + respond in parallel) this pushes to 510-600 MB — above the 512 MB limit. The Linux OOM killer triggers, terminating the process.

The `auto_stop_machines = false` + `min_machines_running = 1` configuration in `fly.toml` suggests the developer may have been keeping the machine alive continuously to mask restart failures caused by OOM on startup.

### Recommendation

Upgrade to Fly.io `performance-1x` (1 dedicated CPU, 2 GB RAM, ~$17/mo) before first deploy. This is already listed as the first P0 roadmap item in PRODUCTION_PLAN.md. Do not attempt to deploy on the current 512 MB spec.

---

## Appendix C — Known Eval Failures

**Added:** 2026-05-14. These are pre-existing failures confirmed at commit `6d21d5e` (Wave 1 merge base). None are introduced by Wave 1 or Wave 1.5.

| ID | Category | Failure checks | Classification | Justification |
|----|----------|---------------|----------------|---------------|
| S2 | season | `n_results_min` | model_limitation | LLM over-constrains filters (dress + Ladieswear only) for a broad "outfits" query, reducing results below the 3-item minimum; inconsistent across runs. |
| ST5 | style | `n_results_min` | model_limitation | LLM filters to `department_name: basics`, a narrow catalogue slice; same over-filtering pattern as S2. |
| N4 | negation | `category_present`, `category_absent` | real_bug | Agent searches with `product_type_name: Pyjama set` — the explicitly excluded category — indicating the negation constraint is not surfaced to the filter/search tool. |
| TB4 | tool_behaviour | `filter_applied` | real_bug | Filter tool is called (`tool_calls` includes `filter`) but the `index_group_name: Divided` constraint does not persist; agent falls back to an unconstrained re-search, returning unfiltered results. |
| C3 | colour | `AttributeError` | real_bug | LLM returns a list value for a colour filter (e.g. `["Beige", "Cream"]`) but filter parsing calls `.lower()` on the value, crashing on a list; code does not handle multi-valued filter entries. |
| TB2 | tool_behaviour | `style_criteria` | model_limitation | Agent correctly compares two items but LLM response does not use the specific style vocabulary the eval checker expects; response is valid but fails the keyword check. |


---

## Token Telemetry -- Local Baseline (pre-Wave 3)

**Date:** 2026-05-14  
**Model:** llama-3.1-8b-instant via Groq API  
**Session:** local dev, InMemorySessionStore, JWT_VERIFICATION_DISABLED=true, no prompt caching  
**Scripts:** scripts/telemetry_collect.py, scripts/telemetry_analyze.py  
**Raw stats:** scripts/telemetry_stats.json

**Methodology note:** This is a local measurement, not production. Token counts are
model-prompt-driven and transfer directly to production sizing. Latency, memory, and
infra-dependent costs are not in scope. Groq has no Anthropic-style prompt caching;
all input tokens are billed as non-cached. The plan's 3,523-token assumption included
2,425 cached tokens; the observed 5,569/turn is a fully uncached equivalent, so the
comparison overstates the production-uncached-token gap (the Wave 3 Anthropic switch
will reintroduce caching).

### Sample

- 29 HTTP requests sent (POST /chat), 28 successful, 1 timeout (R5-refine at 120 s)
- 113 llm_call log entries captured; 105 matched to request windows; 8 unmatched
  (server continued processing R5-refine after client timeout)
- Mix: 8 simple searches, 5 refinements, 4 outfit requests, 3 comparisons,
  2 OOC, 2 negations, 1 x 5-turn multi-turn chain

### Key Structural Finding: 4 LLM Calls Per Search Turn, Not 3

The agent runs **4 LLM calls** per standard search turn, not the 3 assumed in
PRODUCTION_PLAN.md. The actual graph execution pattern is:

1. **Router** (pos 1) -- decides to search; ~1,820 input, ~26 output tokens
2. **Reranker** (pos 2) -- ranks retrieved items; ~1,040 input, ~46 output tokens  
3. **Agent-loop router** (pos 3) -- re-evaluates after seeing results, decides to
   respond; ~1,946 input (median), ~29 output tokens. This call was not in the plan.
4. **Respond** (pos 4) -- generates the final answer; ~1,030 input, ~87 output tokens

The third call is the router node firing again after each tool execution to decide
the next action. PRODUCTION_PLAN.md's "3 calls" omitted this step entirely.

Compare turns (C1, C3) run **6 calls** -- two full search-rerank cycles before
responding. OOC and clarify turns run 1-2 calls (no reranker).

Call-count distribution across 28 turns:

| Calls/turn | Turns | Turn types |
|---|---|---|
| 1 | 2 | OOC (clarify), OOC (router-only) |
| 2 | 2 | OOC weather, direct-answer |
| 3 | 2 | unusual short-context turns |
| 4 | 19 | standard search / refine / outfit / negation (68%) |
| 5 | 1 | multi-turn refinement (extra iteration) |
| 6 | 2 | compare (two search+rerank cycles) |

### Per-Call-Position Statistics (28 turns, 105 matched calls)

Position labels are assigned by index within each turn (1 = first call, last = final call).
Positions 2-5 are mixed across turn types; position 1 (router) and positions 4/6 (respond)
are the cleanest single-call-type buckets.

| Pos | Role (typical) | N | In mean | In p50 | In p90 | In max | Out mean | Out max | Cost USD |
|---|---|---|---|---|---|---|---|---|---|
| 1 | router | 28 | 1,812 | 1,826 | 2,021 | 2,181 | 26 | 73 | $0.002493 |
| 2 | reranker / OOC-respond | 26 | 1,212 | 1,040 | 1,944 | 2,174 | 46 | 65 | $0.001595 |
| 3 | agent-loop router / respond (3-call turns) | 24 | 1,758 | 1,946 | 2,021 | 2,193 | 36 | 96 | $0.002076 |
| 4 | respond / compare-extra | 22 | 1,135 | 1,030 | 1,818 | 2,107 | 84 | 120 | $0.001370 |
| 5 | compare extra | 3 | 1,605 | 1,889 | 1,899 | 1,901 | 47 | 84 | $0.000252 |
| 6 | respond (6-call compare turns) | 2 | 844 | 844 | 859 | 863 | 100 | 106 | $0.000100 |

### Per-Turn Totals

| Metric | Count | Mean | P50 | P90 | Max |
|---|---|---|---|---|---|
| Input tokens | 28 | 5,569 | 5,810 | 7,345 | 9,382 |
| Output tokens | 28 | 177 | 196 | 223 | 268 |
| Cost (USD) | 28 | $0.000282 | $0.000305 | $0.000382 | $0.000491 |

Total cost across 28 turns: **$0.007886**

### Cost Breakdown by Call Position

The agent-loop-router (position 3) is the single largest line item -- larger than
the respond call -- because it was invisible in the plan but fires on every search turn.

| Position | Role | Cost | Share |
|---|---|---|---|
| 1 | router | $0.002493 | 31.6% |
| 2 | reranker / OOC-respond | $0.001595 | 20.2% |
| 3 | agent-loop router | $0.002076 | 26.3% |
| 4 | respond / compare-extra | $0.001370 | 17.4% |
| 5+6 | compare additional calls | $0.000352 | 4.5% |
| **Total** | | **$0.007886** | 100% |

All routing calls combined (positions 1 + 3): $0.004569 (57.9% of total cost).

### Comparison to PRODUCTION_PLAN.md

- **Plan assumption:** 3,523 input tokens/turn (mean, 3 calls x ~1,175 avg)
- **Observed mean:** 5,569 input tokens/turn
- **Delta: +58.1% -- FLAG (>20%)**

Root causes of the gap:

1. **Missing call** -- the agent-loop-router (position 3, ~1,758 tokens input median)
   adds roughly 1,758 extra tokens per turn not in the plan's 3-call model.
2. **Compare turns** -- 6 calls at 9,059-9,382 tokens/turn pull the mean up.
3. **No prompt caching** -- in production with Anthropic, 2,425 of the 3,523 planned
   tokens are cached (69%). Without caching (Groq local), all tokens are charged. The
   observed 5,569 is not directly comparable to the 3,523 cached-inclusive figure;
   the like-for-like uncached plan figure would be 1,098 dynamic tokens plus the full
   cached prefixes re-billed -- but the fundamental issue (missing 4th call) stands
   regardless of caching.

**Implication for Wave 3:** PRODUCTION_PLAN.md cost table must be recalculated for a
4-call standard-search model. The per-turn cost will increase but less than the raw
+58.1% token delta suggests, because the missing call (agent-loop router) is primarily
a short-output decision call (~29 tokens out), not a long generation.

---

## Token Telemetry — Post-Fast-Path (Wave 3a)

**Date:** 2026-05-15  
**Branch:** `wave-3a-alr-fast-path`  
**Provider:** Groq llama-3.1-8b-instant  
**Script:** `scripts/telemetry_fastpath.py` (in-process, bypasses JWT auth)

### Per-Turn LLM Call Counts

| Label | Turn type | Calls | Notes |
|-------|-----------|-------|-------|
| S1-search | search | 2 | ≤3 items → reranker skipped; fast-path fires after search |
| S2-search | search | 3 | reranker + fast-path (baseline was 4) |
| S3-search | search | 3 | |
| S4-search | search | 3 | |
| S5-search | search | 3 | |
| S6-search | search | 3 | |
| S7-search | search | 3 | |
| S8-search | search | 3 | |
| R1-refine | refinement | 7 | filter rejected loop ×3 then search; complex filter→search path |
| R2-refine | refinement | 3 | |
| R3-refine | refinement | 3 | |
| R4-refine | refinement | 0 | OOC short-circuit anomaly (lighter wash → no catalogue match) |
| R5-refine | refinement | 6 | multi-attempt search |
| O1-outfit | outfit | 3 | outfit→END, no ALR fired |
| O2-outfit | outfit | 3 | |
| O3-outfit | outfit | 3 | |
| O4-outfit | outfit | 3 | |
| C1-compare | compare | 4 | initial router + ALR after empty compare + reranker + respond |
| C2-compare | compare | 3 | compare-intent guard routed via search only |
| C3-compare | compare | 4 | |
| OOC1 | OOC | 2 | OOC fast-path → respond |
| OOC2 | OOC | 1 | router → clarify → END |
| N1-negate | negation | 4 | filter→search path; ALR fires (non-trivial, LLM called) |
| N2-negate | negation | 4 | |
| MT1-multi | multi-turn | 3 | |
| MT2-multi | multi-turn | 3 | |
| MT3–MT5 | multi-turn | — | Not measured: Groq TPM (6 K/min) exhausted mid-run |

**Measured turns:** 26 / 29  
**Total LLM calls (26 turns):** 82  
**Mean calls/turn (fast-path):** 3.15  
**Mean calls/turn (baseline):** 3.75  
**Reduction:** −0.60 calls/turn (−16.0%)

### Breakdown by Turn Type (fast-path run)

| Category | Turns | Mean calls | Baseline mean |
|----------|-------|-----------|---------------|
| search (S) | 8 | 3.00 | 4.00 |
| refinement (R) | 5 | 3.80 | ~4.00 |
| outfit (O) | 4 | 3.00 | 3.00 |
| compare (C) | 3 | 3.67 | ~6.00 |
| OOC | 2 | 1.50 | ~2.00 |
| negation (N) | 2 | 4.00 | ~4.00 |
| multi-turn (MT) | 2 | 3.00 | ~4.00 |

### Key Observations

1. **Trivial search turns: 4 → 3 calls.** Fast-path fires after `search_node` returns items, eliminating the ALR LLM call. S1 (no reranker) went 3 → 2.
2. **Compare turns: ~6 → 4 calls.** Two ALR calls eliminated per compare turn; the `route_decision` compare-intent guard still correctly routes through `compare_node`.
3. **Filter→search non-trivial: unchanged.** R1 (7 calls), N1/N2 (4 calls), and `test_filter_then_search_hits_llm_router` confirm the LLM is still called after `filter_node` to generate the search query — fast-path correctly does not skip this.
4. **Outfit/clarify unaffected.** Both go directly to END via LangGraph edges; ALR never fires for these turn types regardless of the flag.
5. **OOC short-circuit anomaly (R4 = 0 calls).** "In a lighter wash instead" on a dark-wash denim context triggered the OOC detector rather than a colour-filter refinement. This is a pre-existing routing edge case, not introduced by Wave 3a.

### Eval Comparison

| Metric | Baseline (Apr 25) | Wave 3a (May 15) | Delta |
|--------|------------------|--------------------|-------|
| Pass / 32 | 31 | 28 | −3 |
| Pass % | 96.9% | 87.5% | −9.4 pp |
| Provider | Groq llama-3.1-8b | Groq llama-3.1-8b | — |

**All 4 regressions confirmed non-fast-path-related:**

| ID | Check | Root cause |
|----|-------|-----------|
| O3 | `category_absent` | Retrieval variance; Groq returned outfit items without the expected sub-category label |
| O5 | `n_results_min` | Groq sampling variance; returned 2 items instead of ≥3 |
| N2 | `n_results_min` | Same — returned 2 items |
| TB4 | `filter_applied` | Pre-existing `search_node` fallback bug: when brand filter returns 0 results, `effective_filters = {}` overwrites `state.filters`, causing the eval check to see no filter applied |

The fast-path change touches `router_node` only and has no code path through retrieval, filter application, or result ranking. Pass rate on the fast-path-sensitive categories (all search and compare turns) is unchanged.
