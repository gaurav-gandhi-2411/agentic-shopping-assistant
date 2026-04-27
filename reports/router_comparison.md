# Router Comparison: DistilBERT vs Groq LLM

## Summary table

| Metric | DistilBERT | Groq LLM (llama-3.1-8b-instant) |
|---|---|---|
| Macro F1 on test | 0.8263 | 0.8462 (on 37/37 non-skipped) |
| Latency p50 | 31.4 ms (CPU) | 2093 ms (API round-trip) |
| Latency p95 | 37.8 ms (CPU) | 5353 ms |
| Cost per 1k requests | $0 (local inference) | ~$0.05–0.10 (API) |
| Rate-limit risk | None | High (TPD quota) |
| Deployment | Bundled in app | External API dependency |
| Cold-start | None (loaded once) | None (stateless API) |
| Groq calls skipped (rate limit) | — | 0 / 37 |

## Methodology

- Test set: 37 held-out examples from `data/router_dataset_test.jsonl`
- LLM router called with same state fields (query, last_action, items_retrieved, active_filters)
- No conversation history passed to LLM (single-turn evaluation; test examples are single-turn)
- DistilBERT latency measured on CPU only (production deployment target)
- Groq latency includes full API round-trip (network + inference)

## Disagreement examples (7 found)

Cases where DistilBERT and Groq LLM chose different routes:

| id | query | true label | DistilBERT | Groq LLM | DB correct? |
|---|---|---|---|---|---|
| para_052_3 | I'm lost – can you help me decide on a gift for someone | clarify | respond | clarify | no |
| edge_007 | Do you have earphones or headphones? | respond | respond | search | yes |
| para_005_0 | Can you help me choose an outfit for a job interview? | search | search | outfit | yes |
| para_039_2 | Show me Ladieswear items | filter | filter | search | yes |
| para_005_4 | Help me pick a suitable outfit for an upcoming job inte | search | search | clarify | yes |
| edge_037 | Blazers but not the pinstriped ones from before | search | filter | search | no |
| seed_045 | Complete the look around item 2 | outfit | filter | outfit | no |

### Notable disagreements

**1. `para_052_3`** — True: `clarify` | DB: `respond` | Groq: `clarify` | Winner: Groq
> Query: "I'm lost – can you help me decide on a gift for someone whose preferences I'm not aware of?" (last_action=none, items=0)

**2. `edge_007`** — True: `respond` | DB: `respond` | Groq: `search` | Winner: DistilBERT
> Query: "Do you have earphones or headphones?" (last_action=none, items=0)

**3. `para_005_0`** — True: `search` | DB: `search` | Groq: `outfit` | Winner: DistilBERT
> Query: "Can you help me choose an outfit for a job interview?" (last_action=none, items=0)

**4. `para_039_2`** — True: `filter` | DB: `filter` | Groq: `search` | Winner: DistilBERT
> Query: "Show me Ladieswear items" (last_action=search, items=5)

**5. `para_005_4`** — True: `search` | DB: `search` | Groq: `clarify` | Winner: DistilBERT
> Query: "Help me pick a suitable outfit for an upcoming job interview." (last_action=none, items=0)
