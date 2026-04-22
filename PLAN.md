# PLAN.md — Agentic Shopping Assistant with Hybrid RAG and Tool Orchestration

> **Claude Code: read this entire file before writing any code. Follow phases in order. After each phase, run verification and commit to git. Stop and report at each phase boundary.**

---

## Project overview

A conversational shopping assistant that answers multi-turn natural-language questions over a fashion catalogue (H&M articles). Given a query like *"show me black summer dresses similar to what I looked at earlier"* or *"compare these two jackets"*, the system routes the query through a LangGraph agent that uses hybrid retrieval (dense FAISS + sparse BM25, fused via Reciprocal Rank Fusion) and a small set of tools (search, compare, filter, clarify). Conversation state is persisted across turns. Served via streaming FastAPI with a Streamlit chat UI.

**Target user:** an e-commerce customer using natural language instead of faceted search.

**Key design decisions (do not change without asking):**
- Catalogue: H&M articles.csv (same dataset as Project 1, symlinked at `data/hm/`). Text-only — no images in this project.
- Retrieval: hybrid — sentence-transformers `all-MiniLM-L6-v2` (dense) + `rank_bm25` (sparse), combined via Reciprocal Rank Fusion (RRF, k=60)
- Agent framework: LangGraph with explicit state machine (not LangChain's AgentExecutor)
- LLM: `llama3.1:8b` via Ollama locally at `http://localhost:11434`
- Conversation memory: last 6 turns in full, older turns summarised via LLM
- Backend: FastAPI with Server-Sent Events (SSE) streaming
- UI: Streamlit chat via `st.chat_message` and `st.write_stream`
- Deployment: HuggingFace Spaces (Streamlit SDK, Groq for LLM)

---

## Environment setup (FIRST, before Phase 1)

```bash
.venv\Scripts\activate
pip install --upgrade pip setuptools wheel
```

`requirements.txt`:

```
sentence-transformers>=3.0.0
rank-bm25>=0.2.2
faiss-cpu>=1.8.0
pandas>=2.2.0
numpy>=1.26.0
pyarrow>=15.0.0
tqdm>=4.66.0
pyyaml>=6.0.0

langgraph>=0.2.0
langchain-core>=0.3.0
langchain-ollama>=0.2.0
ollama>=0.3.0

fastapi>=0.115.0
uvicorn[standard]>=0.32.0
sse-starlette>=2.1.0
pydantic>=2.8.0

streamlit>=1.38.0
httpx>=0.27.0

pytest>=8.0.0
```

Install:
```bash
pip install -r requirements.txt
```

**No GPU required for this project.** Sentence-transformers runs fine on CPU for the one-time embedding step (~2 min for 20k items). LLM inference goes through Ollama which uses its own GPU management independent of this venv.

**Verify Ollama is reachable:**
```python
import requests
r = requests.get("http://localhost:11434/api/tags")
print(r.status_code, [m["name"] for m in r.json().get("models", [])])
```
Should print `200` and show `llama3.1:8b` in the list. If not, run `ollama serve` in another terminal and/or `ollama pull llama3.1:8b`.

---

## Repository structure (final state)

```
agentic-shopping-assistant/
├── README.md
├── PLAN.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── data/
│   ├── hm/                          (symlink to H&M dataset — already exists)
│   └── processed/                   (FAISS index, BM25 state, catalogue parquet)
├── src/
│   ├── __init__.py
│   ├── catalogue/
│   │   ├── __init__.py
│   │   └── loader.py                (Phase 1)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense_search.py          (Phase 2)
│   │   ├── sparse_search.py         (Phase 2)
│   │   └── hybrid_search.py         (Phase 2)
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py                (Phase 3)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py                 (Phase 4)
│   │   ├── tools.py                 (Phase 4)
│   │   └── graph.py                 (Phase 4)
│   └── memory/
│       ├── __init__.py
│       └── conversation.py          (Phase 4)
├── api/
│   └── main.py                      (Phase 5)
├── app/
│   └── streamlit_chat.py            (Phase 5)
├── scripts/
│   ├── 01_build_retrieval.py        (Phase 2)
│   └── 02_smoke_test.py             (Phase 4)
└── tests/
    ├── test_retrieval.py
    ├── test_tools.py
    └── test_agent.py
```

---

## `.gitignore` (Phase 0)

```
.venv/
__pycache__/
*.pyc
data/hm/
data/processed/
*.faiss
*.pkl
.ipynb_checkpoints/
.DS_Store
*.log
.env
```

---

## `config.yaml` (Phase 0)

```yaml
catalogue:
  articles_csv: "data/hm/articles.csv"
  processed_dir: "data/processed"
  # Sample to keep things fast for a demo. Full 105k works too.
  sample_num_items: 20000
  seed: 42

retrieval:
  dense_model: "sentence-transformers/all-MiniLM-L6-v2"
  dense_batch_size: 128
  dense_dim: 384
  bm25_tokenizer: "lower_alpha"   # lowercase + alphanumeric split
  rrf_k: 60
  top_k: 20                       # retrieved before reranking / filtering
  final_k: 5                      # shown to user

llm:
  provider: "ollama"
  model: "llama3.1:8b"
  host: "http://localhost:11434"
  temperature: 0.2
  max_tokens: 400
  timeout_seconds: 60

agent:
  max_iterations: 6
  enable_clarify_tool: true

memory:
  recent_turns: 6
  summary_trigger_turns: 12       # summarise older turns after this many

api:
  host: "127.0.0.1"
  port: 8080
  cors_origins: ["http://localhost:8501"]
```

---

## PHASE 0 — Scaffolding (15 min)

**Goal:** Empty skeleton with venv, requirements installed, config, stub README, empty module tree.

**Tasks:**
1. Venv active, install requirements
2. Verify Ollama reachable (snippet above — paste output)
3. Create all empty `__init__.py` files
4. Create `config.yaml`, `.gitignore`, stub `README.md`
5. `git init` (already done), commit, push

**Verification:**
```bash
python -c "import langgraph, faiss, rank_bm25, sentence_transformers, fastapi, streamlit; print('All imports OK')"
python -c "import requests; r=requests.get('http://localhost:11434/api/tags'); assert r.status_code==200 and any('llama3.1' in m['name'] for m in r.json().get('models',[])); print('Ollama OK with llama3.1')"
```

**Commit:** `Phase 0: scaffolding, requirements, config, Ollama check`

---

## PHASE 1 — Catalogue loader (30 min)

**Goal:** Load and normalise the H&M articles into a clean, searchable DataFrame.

### Files

**`src/catalogue/loader.py`**

Functions:
- `load_articles(config) -> pd.DataFrame`
  - Load `articles.csv`
  - Keep columns: `article_id`, `prod_name`, `product_type_name`, `product_group_name`, `graphical_appearance_name`, `colour_group_name`, `department_name`, `index_group_name`, `garment_group_name`, `detail_desc`
  - Drop rows where `detail_desc` is NaN
  - Sample `sample_num_items` with config seed
  - Return with `article_id` as string, reset_index

- `build_searchable_text(articles_df) -> pd.DataFrame`
  - Add column `search_text` = "{prod_name}. {product_type_name}. {colour_group_name}. {department_name}. {detail_desc}"
  - Add column `display_name` = "{prod_name} ({colour_group_name} {product_type_name})"
  - Add column `facets` = dict of {colour, product_type, department, index_group, garment_group} for filtering
  - Save to `data/processed/catalogue.parquet`

**`scripts/01_build_retrieval.py`** (the full Phase 2 script; stub it now, orchestrator pattern — just calls loader for this phase)

### Verification

Run:
```bash
python scripts/01_build_retrieval.py
```

Expected output:
```
Loaded catalogue: 20,000 articles
Unique product types: ~130
Unique colour groups: ~50
Saved to data/processed/catalogue.parquet
```

Quick sanity check in Python:
```python
import pandas as pd
df = pd.read_parquet("data/processed/catalogue.parquet")
print(df[["prod_name", "display_name", "search_text"]].head(3))
print(df["facets"].iloc[0])
```

**Commit:** `Phase 1: catalogue loader with searchable text and facets`

---

## PHASE 2 — Hybrid retrieval (1.5 hours)

**Goal:** Dense + sparse retrieval with RRF fusion. Working end-to-end retriever given a free-text query.

### Files

**`src/retrieval/dense_search.py`**

```python
class DenseRetriever:
    def __init__(self, config): ...
    def build_index(self, catalogue_df: pd.DataFrame, save_dir: Path):
        """Encode search_text, build FAISS IndexFlatIP (cosine on L2-normalised),
        save index + parallel article_id array."""
    @classmethod
    def load(cls, config, save_dir: Path): ...
    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Returns [(article_id, score), ...] sorted by score desc."""
```

Implementation notes:
- Use `SentenceTransformer(model_name, device='cpu')` — CPU is fine for query-time encoding
- Normalise embeddings on build AND on query (use `normalize_embeddings=True`)
- Save to `data/processed/dense.faiss` + `data/processed/dense_article_ids.npy`

**`src/retrieval/sparse_search.py`**

```python
class SparseRetriever:
    def __init__(self, config): ...
    def build_index(self, catalogue_df: pd.DataFrame, save_dir: Path):
        """Tokenise search_text, build BM25Okapi, pickle it."""
    @classmethod
    def load(cls, config, save_dir: Path): ...
    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Returns [(article_id, score), ...] sorted by BM25 score desc."""
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split on non-alphanumeric, drop very short tokens."""
```

Implementation notes:
- Use `rank_bm25.BM25Okapi` with the tokenised corpus
- Tokeniser: `re.findall(r"[a-z0-9]+", text.lower())`, drop tokens with length < 2
- Save to `data/processed/bm25.pkl` + `data/processed/bm25_article_ids.npy`

**`src/retrieval/hybrid_search.py`**

```python
class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, catalogue_df: pd.DataFrame, config): ...

    def search(self, query: str, top_k: int = None, filters: dict = None) -> list[dict]:
        """
        1. Get top-k from dense and sparse independently (at 2*top_k each to give RRF room)
        2. Fuse via Reciprocal Rank Fusion:
             rrf_score(item) = sum_over_retrievers(1 / (rrf_k + rank_in_that_retriever))
        3. If filters is non-empty (e.g. {'colour_group_name': 'black'}), apply
           case-insensitive exact match on facets BEFORE returning top_k
        4. Return list of dicts: {article_id, display_name, colour, product_type,
           department, detail_desc, score}
        """
```

Implementation notes:
- `rrf_k=60` is standard; in config
- When an item appears in only one retriever, it still gets a score (just smaller)
- Filter values are exact match, case-insensitive

**`scripts/01_build_retrieval.py`** (full version)

Loads catalogue → builds both indices → saves everything. Prints timing.

### Tests (`tests/test_retrieval.py`)

Pytest tests:
- `test_dense_search_returns_top_k`: query "black dress", expect top-1 to contain "black" or "dress" in display_name
- `test_sparse_search_literal_match`: query with exact product name, expect that product ranked top-1
- `test_hybrid_outperforms_either`: for a semantic query like "comfy wearable for summer", hybrid should surface items that pure lexical search misses
- `test_filter_applied`: query "dress" with filter `{"colour_group_name": "black"}`, every result must have black colour
- `test_retrievers_load_from_disk`: save, then reload from disk, verify consistent output

### Verification

After running the build script, in Python:
```python
from src.retrieval.hybrid_search import HybridRetriever
# (load from disk)
results = retriever.search("lightweight black summer jacket", top_k=5)
for r in results:
    print(r["display_name"], r["score"])
```
Should return coherent results. If results look random (unrelated product types), something is off — stop and report.

**Commit:** `Phase 2: hybrid retrieval with dense FAISS + sparse BM25 + RRF fusion`

---

## PHASE 3 — LLM client (30 min)

**Goal:** Thin, well-tested wrapper around Ollama. Interchangeable with Groq later.

### Files

**`src/llm/client.py`**

```python
class LLMClient(Protocol):
    def generate(self, prompt: str, system: str = None, **kwargs) -> str: ...
    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]: ...
    def chat(self, messages: list[dict], **kwargs) -> str: ...
    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]: ...


class OllamaClient(LLMClient):
    """Wraps ollama.Client. Messages use OpenAI-style {role, content}."""
    def __init__(self, config): ...
    def chat(self, messages, temperature=None, max_tokens=None) -> str: ...
    def chat_stream(self, messages, temperature=None, max_tokens=None) -> Iterator[str]: ...
    # generate is just a wrapper that builds a single-message chat


def get_llm_client(config) -> LLMClient:
    """Factory. Switches on config.llm.provider. Currently only 'ollama';
    'groq' stub raises NotImplementedError until Phase 6."""
```

Implementation notes:
- Use the `ollama` Python package (not raw requests) — cleaner, handles streaming natively
- `chat_stream` yields chunk strings as they arrive
- All methods respect `timeout_seconds` from config
- On any Ollama error (connection, timeout), log clearly and raise — don't silently return empty strings

### Tests (`tests/test_llm.py`)

- `test_ollama_generate_basic`: send "Respond with only the word 'OK'", expect "OK" in response (case-insensitive)
- `test_ollama_chat_with_system`: system="You always say 'penguin'.", user="hi", expect "penguin" in response
- `test_ollama_stream_yields_chunks`: stream "count 1 to 5", collect chunks, verify > 1 chunk received

**Note:** These tests require Ollama to be running. If CI ever runs without Ollama, skip with `@pytest.mark.requires_ollama`.

**Commit:** `Phase 3: Ollama LLM client with streaming support`

---

## PHASE 4 — Agent graph, tools, memory (2.5 hours)

**Goal:** LangGraph state machine with 4 tools, conversation memory, end-to-end agent that answers multi-turn queries.

### Files

**`src/agents/state.py`**

```python
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    # Conversation
    messages: Annotated[list[dict], operator.add]   # role/content dicts, accumulated
    user_query: str                                  # most recent user input

    # Agent internals
    current_plan: str | None                         # what the router decided to do
    tool_calls: list[dict]                           # history within this turn
    retrieved_items: list[dict]                      # latest retrieval results
    filters: dict                                    # accumulated filters (colour, etc.)

    # Output
    final_answer: str | None
    iteration: int
```

**`src/agents/tools.py`**

Four tools, each a plain callable that takes a state-relevant argument and returns a dict.

```python
def search_catalogue(query: str, filters: dict | None, retriever: HybridRetriever, top_k: int) -> dict:
    """Runs hybrid retrieval. Returns {items: [...], query: ..., n_results: int}."""

def compare_items(article_ids: list[str], catalogue_df: pd.DataFrame) -> dict:
    """Given 2-5 article_ids, returns a comparison dict with their attributes side-by-side."""

def apply_filter(current_filters: dict, filter_key: str, filter_value: str) -> dict:
    """Merges a new filter into current_filters. Returns the updated dict.
       filter_key must be one of the facet keys in the catalogue."""

def clarify(question: str) -> dict:
    """Stub tool — just returns {clarification_question: question}.
       The graph uses this signal to route back to the user instead of looping."""
```

Each tool is a pure function — no hidden LLM calls inside. The LLM decides WHICH tool to call; the tool executes a deterministic action.

**`src/memory/conversation.py`**

```python
class ConversationMemory:
    def __init__(self, llm: LLMClient, config): ...

    def get_context(self, messages: list[dict]) -> list[dict]:
        """
        If len(messages) <= recent_turns: return as-is.
        Else: return [summary_system_message, *messages[-recent_turns:]].
        Summary is produced by summarise() once every summary_trigger_turns.
        """

    def summarise(self, messages: list[dict]) -> str:
        """LLM call. Prompt: 'Summarise this shopping conversation in 3 bullets,
        preserving user preferences, constraints, and items already discussed.'"""
```

Cache the summary in memory — recompute only when new trigger is hit.

**`src/agents/graph.py`** — the heart of the project

Nodes:
- `router` — LLM decides next action given state
- `search_node` — calls `search_catalogue`, updates state
- `compare_node` — calls `compare_items`, updates state
- `filter_node` — calls `apply_filter`, updates state
- `clarify_node` — terminates with a clarification question as final_answer
- `respond` — LLM generates final user-facing answer using retrieved items as context

Edges:
- START → `router`
- `router` → conditional edge based on LLM decision (search/compare/filter/clarify/respond)
- Each tool node → `router` (looping; `iteration` increments)
- `clarify_node` → END
- `respond` → END
- Hard cap: if `iteration >= max_iterations`, force `respond`

Router prompt skeleton (Claude Code, implement exactly this contract — structured output):

```
You are a shopping assistant planner. Given the conversation so far and the latest user query,
decide the NEXT action. Respond with ONE of the following JSON objects and nothing else:

{"action": "search", "query": "<search string>", "filters": {<optional facet: value>}}
{"action": "compare", "article_ids": ["<id1>", "<id2>", ...]}
{"action": "filter", "key": "<facet>", "value": "<value>"}
{"action": "clarify", "question": "<clarification for the user>"}
{"action": "respond"}

Use "respond" when you have enough retrieved context to answer the user's question.
Use "clarify" ONLY if the query is truly ambiguous (e.g., unclear gender, price range, occasion).
Use "search" for a new information need. Use "filter" to narrow prior results.

Available facets: colour_group_name, product_type_name, department_name, index_group_name, garment_group_name

Current retrieved items (if any): <brief summary>
Current filters: <json>
Latest user query: <query>
Recent conversation: <context>
```

Parse the JSON robustly — wrap in try/except; if parse fails, fall back to `{"action": "search", "query": user_query}`.

Response prompt (for the `respond` node):

```
You are a friendly, concise shopping assistant. The user asked: "{user_query}"

Here are the items we retrieved to help answer:
{top 5 items with name, colour, type, short description}

Write a helpful 2-4 sentence response that directly answers the user's question
and naturally weaves in the best 2-3 items. If the user asked for a comparison,
highlight the key differences. Do not invent attributes not in the retrieved items.
```

**`scripts/02_smoke_test.py`** — end-to-end non-streaming test

```python
# Build agent, seed state with a test query, run graph to completion, print final_answer.
test_queries = [
    "show me some black jackets",
    "something for summer, light and breathable",
    "compare the first two you showed me",    # requires memory
    "anything in blue instead?",               # requires filter logic
]
# For each, run graph, print user_query and final_answer.
```

### Tests (`tests/test_tools.py`, `tests/test_agent.py`)

- `test_search_tool_returns_items`: direct tool call, check schema
- `test_compare_tool_handles_2_to_5_items`: valid + edge cases (1 item, 6 items)
- `test_apply_filter_merges_correctly`: adds to existing filters
- `test_router_parses_valid_json`: mock LLM to return valid JSON, verify routing
- `test_router_fallback_on_bad_json`: mock LLM returns garbage, verify graceful fallback to search
- `test_agent_end_to_end_search`: real Ollama, simple query, expect non-empty final_answer mentioning retrieved items
- `test_agent_max_iterations_cap`: verify the iteration cap triggers `respond` even if the LLM keeps calling tools

### Verification

Run `python scripts/02_smoke_test.py`. Expected:
- Each query completes without exceptions
- `final_answer` is a non-empty string
- For query 3 (compare), the answer references items from query 1 or 2 (memory working)
- For query 4 (colour swap), new results should be predominantly blue

Paste the smoke-test output in your phase report.

**Commit:** `Phase 4: LangGraph agent with 4 tools, hybrid retrieval wiring, and conversation memory`

---

## PHASE 5 — FastAPI streaming + Streamlit UI (1.5 hours)

**Goal:** Working chat demo at `http://localhost:8501` that streams token-by-token.

### Files

**`api/main.py`** — FastAPI with SSE streaming

Endpoints:
- `GET /health` — returns `{"status": "ok"}` + Ollama reachability check
- `POST /chat/stream` — body `{"message": str, "conversation_id": str}`
  - Streams SSE events with token chunks: `data: {"type": "token", "content": "..."}`
  - Event `{"type": "tool_call", "tool": "search", "args": {...}}` emitted when agent calls a tool (useful for UI "thinking" indicator)
  - Event `{"type": "items", "items": [...]}` emitted when retrieved items are ready
  - Event `{"type": "done"}` at end
  - Errors: `{"type": "error", "message": "..."}`

Implementation notes:
- Use `sse-starlette.EventSourceResponse`
- Conversations stored in-memory: `dict[conversation_id, list[dict]]` — fine for demo, note in README this is non-persistent
- Build the agent graph **once at startup** (lifespan handler); reuse across requests
- CORS: allow `http://localhost:8501` (config-driven)

**`app/streamlit_chat.py`** — Streamlit UI

Layout:
- Title: "Agentic Shopping Assistant"
- Sidebar: short description, tech-stack chips, link to GitHub repo, "Reset conversation" button
- Main: chat history using `st.chat_message` for user / assistant
- Below each assistant turn: if items were retrieved, show a horizontal list of up to 5 items (name, colour, type, short description) — no images, text-only
- Input: `st.chat_input("Ask about anything in the store...")`
- Stream responses via `httpx` to the FastAPI SSE endpoint; display tokens incrementally using `st.write_stream`
- Generate a stable `conversation_id` per browser session via `st.session_state`

Visual cue for tool calls: while waiting for tokens, show a subtle "🔎 Searching…" caption that updates based on the `tool_call` SSE events.

### Verification

Two terminals:
```bash
# Terminal 1
ollama serve   # if not already running
uvicorn api.main:app --host 127.0.0.1 --port 8080 --reload

# Terminal 2
streamlit run app/streamlit_chat.py
```

Open `http://localhost:8501` and test:
- "show me black jackets" → should stream a response, show items
- "compare the first two" → memory + compare tool
- "in blue instead" → filter switch
- Reset button clears state

**Commit:** `Phase 5: FastAPI SSE streaming backend and Streamlit chat UI`

---

## PHASE 6 — README + HuggingFace Spaces deployment (1 hour)

**Goal:** Recruiter-ready README + live demo at HF Spaces.

### `README.md` structure

```markdown
# Agentic Shopping Assistant

> A multi-turn conversational shopping assistant over a fashion catalogue. Combines hybrid retrieval (dense + BM25 with Reciprocal Rank Fusion) with a LangGraph agent that orchestrates search, compare, and filter tools — all powered by a local Llama 3.1 8B.

🔗 **[Live Demo](<link>)** · 🎥 **[Screenshots / GIF](#demo)**

## Motivation
Faceted search is efficient but unnatural. Customers actually ask things like
"show me something for a beach wedding, but not too formal" — a query that needs
both semantic understanding, attribute filtering, and conversation memory. This
project explores a practical agentic approach to that problem.

## Architecture
<ASCII diagram:
  User query → LangGraph router → [search | compare | filter | clarify | respond]
     search → HybridRetriever(dense FAISS ⊕ BM25 with RRF) → top-K items
     compare → side-by-side attribute diff
     filter → facet narrowing on prior results
  All routed back to router until respond → LLM synthesis → streamed to UI
>

### Design decisions
- **Hybrid retrieval (dense + sparse with RRF)** because fashion queries mix semantic
  intent ("comfy summer vibe") with lexical signals ("linen")
- **Explicit LangGraph state machine** over a free-form agent loop — deterministic
  routing, inspectable state, bounded iterations
- **Ollama locally** so the demo is free to run and fully private; Groq on HF Spaces
  for the free hosted demo
- **Streaming via SSE** for low perceived latency — users see tokens within ~200ms

## Dataset
[H&M Personalized Fashion Recommendations](<link>) — 20k-article sample, text-only
(no image embeddings in this project).

## Tech stack
LangGraph · LangChain · Ollama · FAISS · rank-bm25 · FastAPI (SSE) · Streamlit · HuggingFace Hub

## Running locally
[commands from Phase 5 verification]

## Demo
<screenshots or GIF>

## What I learned
<3-5 honest bullets after finishing>

## License
MIT
```

### HuggingFace Spaces

Same pattern as Project 1:
- Space name: `agentic-shopping-assistant`
- SDK: Streamlit, CPU basic (free)
- Replace Ollama with Groq in the Spaces branch (the factory in `llm/client.py` already supports this — just implement `GroqClient` and add `GROQ_API_KEY` secret to Space)
- Precomputed `dense.faiss`, `bm25.pkl`, `catalogue.parquet` checked in via Git LFS or uploaded as release artifacts
- FastAPI runs on the Space via a startup command, Streamlit connects to `127.0.0.1:8080`
- Alternative if FastAPI-on-Spaces is painful: fold the agent directly into the Streamlit app (no separate API). Simpler for free-tier Spaces.

**Commit:** `Phase 6: README with architecture and HuggingFace Spaces deployment`

---

## Final acceptance criteria

- [ ] Repo public at `github.com/gaurav-gandhi-2411/agentic-shopping-assistant`
- [ ] README has a working live demo link
- [ ] Running locally per README produces a working chat experience
- [ ] All tests pass via `pytest`
- [ ] Smoke test script demonstrates 4 multi-turn interactions (search, follow-up, compare, filter)
- [ ] HF Space loads and responds within ~10s on first query, ~3s on subsequent
- [ ] At least one screenshot or GIF in README

---

## Claude Code: guidelines

1. Stop after each phase and summarise. Wait for confirmation before proceeding.
2. Run verification after each phase and paste output.
3. Do not modify `config.yaml` values without asking.
4. On any Ollama connection error, report immediately — don't silently fall back.
5. Prefer simplicity. LangGraph is already more complex than most projects need — don't add complexity beyond the plan.
6. Keep commits atomic and phase-aligned with the exact messages given above.
