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
- Date night / evening / cocktail / wedding queries — apply ALL of these:
  * Prefer rich, elegant colours: black, red, dark blue, dark red, burgundy, emerald, dark green, wine
  * DIVERSITY REQUIRED: your 5 picks MUST include AT LEAST 2 distinct colour values. \
Do NOT return 5 items in the same colour unless the query explicitly names one colour.
  * Pure white acceptable for summer/cocktail context
  * REJECT casual neutrals: beige, light beige, brown, camel, light brown, pastels — not date-night
  * REJECT casual types: sweater, sweatshirt, hoodie, t-shirt — unless explicitly requested
- Beach / pool / beach vacation queries:
  * STRONGLY PREFER: swimwear, bikini, bikini top, bikini bottom, swimsuit, cover-up, sundress, \
beach dress, light tank, board shorts, sarong, flip-flops, beach bag
  * HARD REJECT — do NOT select even if they appear relevant:
    - Any dress described as bodycon, shift, midi, formal, cocktail, structured, or tailored
    - Formal blouses, blazers, trousers, jeans, coats, knitwear, heavy sweaters
    - Any beige / dark beige / sand-coloured item that is NOT swimwear or a casual sundress
  * Neutral-acceptable: light linen or cotton sundress in white/light colours if clearly casual
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


_DATE_NIGHT_RE = re.compile(
    r"\b(date\s+night|date-night|evening\s+out|evening|cocktail|wedding|gala)\b",
    re.IGNORECASE,
)


def _enforce_colour_diversity(
    selected: list[dict],
    candidates: list[dict],
    query: str,
) -> list[dict]:
    """Post-rerank check: for date-night queries, if all picks share one colour, swap
    the lowest-ranked pick for the best different-colour candidate from the pool."""
    if not _DATE_NIGHT_RE.search(query):
        return selected
    colours = [it.get("colour", "").lower() for it in selected]
    if len(set(colours)) >= 2:
        return selected  # already diverse
    dominant = colours[0] if colours else ""
    seen_ids = {it["article_id"] for it in selected}
    for cand in candidates:
        if cand["article_id"] in seen_ids:
            continue
        if cand.get("colour", "").lower() != dominant:
            swapped = selected[:-1] + [cand]
            print(
                f"[reranker] colour-diversity swap: dropped {selected[-1]['article_id']} "
                f"({selected[-1].get('colour','')}), added {cand['article_id']} ({cand.get('colour','')})"
            )
            return swapped
    return selected  # no alternative found — keep as is


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

    selected = _enforce_colour_diversity(selected[:top_k], items, query)

    final_ids = [it["article_id"] for it in selected]
    _log(query, retrieved_ids, llm_indices, final_ids, latency_ms, fallback)
    return selected


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
