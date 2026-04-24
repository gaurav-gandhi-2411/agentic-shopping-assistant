import json
import re
import time

from src.llm.client import LLMClient

_SYSTEM = """\
You are a fashion product reranker. Given a user query and a numbered list of candidate \
products, select the best 5 that match the full intent of the query.

Ranking rules:
- Occasion context: "date night" / "evening out" / "wedding" / "cocktail" → formal \
or elegant wear; never sleepwear, underwear, or swimwear
- Date night colour preference: strongly prefer rich, elegant colours — black, red, \
dark red, dark blue, dark green, burgundy, emerald. Deprioritise casual neutrals: \
beige, light beige, brown, camel, light brown. Casual types such as sweater, \
sweatshirt, hoodie, t-shirt should be ranked last unless explicitly requested. \
Pure white is acceptable for summer cocktail context.
- Season context: "winter" / "autumn" / "fall" → warm items (coats, sweaters, boots); \
never shorts, swimwear, or sandals. "summer" / "beach" → light items, swimwear; \
never coats or heavy knitwear
- Professional context: "office" / "work" / "meeting" → tailored or smart-casual; \
never sleepwear or swimwear
- Colour constraints: if the query names a colour ("black", "red", "blue", etc.), only \
select items of that colour — hard constraint, not a preference
- Style words ("minimalist", "casual", "elegant") describe feel, not colour — do not \
infer a colour from style words

Respond with ONLY valid JSON, no other text:
{"selected": [i, j, k, l, m], "reasoning": "one sentence"}
`selected` must contain exactly 5 distinct integers from the product numbers (1-indexed)."""


def _format_candidates(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(
            f"{i}. {it['display_name']}"
            f" ({it.get('colour', '')} {it.get('product_type', '')} | {it.get('department', '')})"
        )
    return "\n".join(lines)


def _parse_selected(text: str, n: int) -> list[int] | None:
    """Extract and validate selected indices from LLM output. Returns 1-based ints or None."""
    candidates = [text.strip()]
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        candidates.append(m.group())

    for blob in candidates:
        try:
            parsed = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        raw = parsed.get("selected")
        if not isinstance(raw, list):
            continue
        try:
            ints = [int(x) for x in raw]
        except (TypeError, ValueError):
            return None
        seen: set[int] = set()
        valid = [x for x in ints if 1 <= x <= n and x not in seen and not seen.add(x)]  # type: ignore[func-returns-value]
        return valid or None

    return None


def rerank(
    query: str,
    items: list[dict],
    llm: LLMClient,
    top_k: int = 5,
) -> list[dict]:
    """Rerank retrieved candidates with the LLM; falls back to retrieval order on any failure."""
    if len(items) <= top_k:
        return items

    retrieved_ids = [it["article_id"] for it in items[:top_k]]
    user_msg = (
        f'Query: "{query}"\n\n'
        f"Products:\n{_format_candidates(items)}\n\n"
        f"Select the best 5 products."
    )

    t0 = time.time()
    fallback = "none"
    llm_indices: list[int] = []

    try:
        raw = llm.generate(user_msg, system=_SYSTEM, max_tokens=150, temperature=0)
    except Exception:
        latency_ms = int((time.time() - t0) * 1000)
        _log(query, retrieved_ids, [], retrieved_ids, latency_ms, "timeout")
        return items[:top_k]

    latency_ms = int((time.time() - t0) * 1000)
    indices = _parse_selected(raw, len(items))

    if indices is None:
        _log(query, retrieved_ids, [], retrieved_ids, latency_ms, "parse_error")
        return items[:top_k]

    llm_indices = indices
    selected = [items[i - 1] for i in indices]

    if len(selected) < top_k:
        fallback = "partial"
        seen = {it["article_id"] for it in selected}
        for it in items:
            if len(selected) >= top_k:
                break
            if it["article_id"] not in seen:
                selected.append(it)
                seen.add(it["article_id"])

    final_ids = [it["article_id"] for it in selected[:top_k]]
    _log(query, retrieved_ids, llm_indices, final_ids, latency_ms, fallback)
    return selected[:top_k]


def _log(
    query: str,
    retrieved_top5: list[str],
    llm_picked: list[int],
    final_top5: list[str],
    latency_ms: int,
    fallback: str,
) -> None:
    print(
        f'[reranker] query="{query}" '
        f"retrieved_top5={retrieved_top5} "
        f"llm_picked={llm_picked} "
        f"final_top5={final_top5} "
        f"latency_ms={latency_ms} "
        f"fallback={fallback}"
    )
