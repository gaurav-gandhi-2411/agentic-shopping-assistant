# Deployment Guide — Fly.io

This guide deploys the Shopping Assistant **API** container to Fly.io
(region: `bom`, shared-cpu-1x, 512 MB RAM).

The Streamlit UI (`spaces/`) is a separate deployment and is not covered here.

---

## Prerequisites

- [flyctl](https://fly.io/docs/hands-on/install-flyctl/) installed and authenticated
- Docker installed locally (for the build)
- API keys ready: Groq, LangSmith (optional)

---

## Step 1 — Sanity check

```bash
fly auth whoami
```

---

## Step 2 — Initialise the app (first deploy only)

```bash
fly launch --no-deploy --copy-config --region bom
```

When prompted:
- **App name:** choose a globally unique name (e.g. `shopping-assistant-gg`); this replaces the placeholder in `fly.toml`
- **Set up Postgres?** → No
- **Set up Redis?** → No
- **Set up Tigris?** → No

`fly launch` updates `fly.toml` with your chosen app name.

---

## Step 3 — Set secrets

Secrets are injected as environment variables at runtime; they are never baked
into the image.  See `api/README.md` for the full env-vars reference.

```bash
fly secrets set \
  GROQ_API_KEY=<your-groq-api-key> \
  LANGSMITH_API_KEY=<your-langsmith-api-key> \
  LANGCHAIN_TRACING_V2=true \
  LANGSMITH_PROJECT=shopping-assistant
```

Only `GROQ_API_KEY` is required.  Omit the LangSmith vars to disable tracing.

---

## Step 4 — Deploy

```bash
fly deploy
```

The Docker build runs locally (or on a remote builder) and the resulting image
is pushed to Fly's registry.  First deploy takes ~5–10 minutes (large image
due to FAISS + sentence-transformers + baked indices).

---

## Step 5 — Watch startup logs

```bash
fly logs
```

A successful startup looks like:

```
INFO  Starting up Shopping Assistant API
INFO  Loading retrieval indices from /app/data/processed
INFO  Retrieval ready: 20000 items, 20000 dense vectors
INFO  Initialising LLM client (provider=groq)
INFO  LLM client ready
INFO  LangSmith tracing enabled
INFO  Startup complete
```

---

## Step 6 — Verify

```bash
fly status      # machine state
fly open        # opens https://<app>.fly.dev in a browser
```

Smoke test the liveness probe:
```bash
curl https://<app>.fly.dev/healthz
# → {"status":"ok"}
```

Readiness probe (will 503 until indices are loaded, then 200):
```bash
curl https://<app>.fly.dev/readyz
# → {"status":"ready","retriever_items":20000,"catalogue_items":20000,"llm":"ok"}
```

WebSocket smoke test with [wscat](https://github.com/websockets/wscat):
```bash
wscat -c wss://<app>.fly.dev/chat/stream
> {"type":"user_message","message":"show me blue jackets","conversation_id":null}
< {"type":"session","conversation_id":"<uuid>"}
< {"type":"routing","decision":{"action":"search","query":"blue jackets"}}
< {"type":"tool_start","tool":"search"}
< {"type":"items","items":[...]}
< {"type":"token","text":"Here "}
...
< {"type":"done","final_state":{...}}
```

---

## Subsequent deploys

```bash
fly deploy          # rebuild image + deploy
fly logs            # watch rollout
fly status          # confirm healthy
```

---

## Scaling

To increase memory (recommended if startup OOMs):
```bash
fly scale memory 1024
```

To add a second machine for redundancy:
```bash
fly scale count 2
```
