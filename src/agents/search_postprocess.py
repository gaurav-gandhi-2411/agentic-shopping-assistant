"""Plain-search result post-processing: de-duplication, per-store cap, budget guard, gender mix.

`search_node` hands this module the reranker's ranked candidate list (the model's picks first,
then the rest of the pool in retrieval order) and shows what `select_results` returns. Selection is
greedy in ranked order and backfills from deeper candidates, so:

  * a list with no duplicate, no third card from one store, no over-budget item and a gendered
    query comes back unchanged (the first `top_k` of the ranked list);
  * fewer than `top_k` cards are returned only when the whole ranked list cannot supply more
    admissible cards; the reason is reported (never relaxed silently).

Rules (pre-registered in evals/retrieval/results/plain_search_gate_rule.md):

1. De-duplication. Two cards are the same product when they come from the same store AND any of
     - the normalised titles are equal (lower-case, punctuation removed, whitespace collapsed);
     - the image URL is equal;
     - the price is equal and one title's words are a subset of the other's (>= 3 words in the
       shorter): the catalogue lists celebrity/campaign copies of one product under a longer title
       ("Chaitanya Haldi Kurta Set" / "Abhiijeet Saawant in Chaitanya Haldi Kurta Set").
   The legacy cross-store key (normalised product name + colour) is kept so nothing the old
   de-duplication removed comes back.
2. At most `MAX_PER_STORE` cards per store.
3. Budget: with a price cap, only items priced <= cap are shown; an item with no price cannot be
   shown as "within budget" and is not shown.
4. Gender: when the query carries no gender, women's and men's results alternate (the gender of
   the best-ranked card goes first; each gender keeps its own ranked order; unisex/unknown cards
   keep their rank and do not take a turn). With one gender only the order is unchanged. Only
   the reranker's own picks (`mix_window`) take part: a gender with nothing ranked well is not
   padded with whatever retrieval put next.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from src.retrieval.hybrid_search import normalize_prod_name

MAX_PER_STORE = 3  # architect decision 2026-11-26 (first draft: 2); kill switch = set to 99
MAX_GATE_PASSES = 4  # select -> result gates -> refill rounds in search_node before settling
MIN_SUBSET_WORDS = 3  # shorter title must carry >= 3 words before containment counts as a copy
GENDERS = ("women", "men")
# Copy sent when a shopper taps the Women / Men refine control (the query is re-run with the
# gender stated in the query text, the one mechanism the parser already understands).
REFINE_SUFFIX = {"women": "for women", "men": "for men"}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def title_words(title: object) -> tuple[str, ...]:
    """Lower-case alphanumeric words of a product title."""
    return tuple(w for w in _NON_ALNUM.split(str(title or "").lower()) if w)


def _title(it: Mapping[str, Any]) -> str:
    return str(it.get("prod_name") or it.get("display_name") or "")


def _store(it: Mapping[str, Any]) -> str:
    return str(it.get("store") or "")


def _price(it: Mapping[str, Any]) -> float | None:
    p = it.get("price_inr")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or p != p:  # NaN-safe
        return None
    return float(p)


def legacy_key(it: Mapping[str, Any]) -> tuple[str, str]:
    """The pre-existing de-duplication key (normalised name, colour); kept as a floor."""
    return (
        normalize_prod_name(it.get("prod_name") or it.get("display_name") or ""),
        str(it.get("colour") or "").lower(),
    )


def same_product(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """True when two cards are the same product listed twice in one store (see module doc)."""
    if _store(a) != _store(b):
        return False
    wa, wb = title_words(_title(a)), title_words(_title(b))
    if wa and wa == wb:
        return True
    ia, ib = a.get("image_url"), b.get("image_url")
    if ia and ia == ib:
        return True
    pa, pb = _price(a), _price(b)
    if pa is not None and pa == pb:
        small, large = (set(wa), set(wb)) if len(set(wa)) <= len(set(wb)) else (set(wb), set(wa))
        return len(small) >= MIN_SUBSET_WORDS and small <= large
    return False


def gender_of(it: Mapping[str, Any]) -> str:
    """'women' / 'men' / 'other' (unisex, unknown, missing)."""
    g = str(it.get("gender") or "").lower()
    return g if g in GENDERS else "other"


def query_gender(filters: Mapping[str, Any]) -> str | None:
    """The gender the query (or the conversation) already fixed, from the merged filters."""
    g = str(filters.get("gender") or "").lower()
    if g in GENDERS:
        return g
    ign = str(filters.get("index_group_name") or "").lower()
    return {"ladieswear": "women", "menswear": "men"}.get(ign)


def budget_cap(filters: Mapping[str, Any], intent_budget: float | None = None) -> float | None:
    """The price cap: the merged filters' price_max, else the parsed budget of this query."""
    for v in (filters.get("price_max"), intent_budget):
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return float(v)
    return None


def within_budget(it: Mapping[str, Any], price_max: float | None) -> bool:
    """True with no cap; with a cap only a priced item at or under it."""
    if price_max is None:
        return True
    p = _price(it)
    return p is not None and p <= price_max


def enforce_budget(items: Sequence[dict], price_max: float | None) -> list[dict]:
    """Final guard: drop anything over the cap (or unpriced). Never adds items."""
    return [it for it in items if within_budget(it, price_max)]


def gender_refine_options(filters: Mapping[str, Any], ranked: Sequence[Mapping[str, Any]]) -> bool:
    """True when the Women / Men refine control applies: the query fixed no gender AND the
    candidates hold both women's and men's items (a saree or sherwani query never does)."""
    if query_gender(filters) is not None:
        return False
    present = {gender_of(it) for it in ranked}
    return all(g in present for g in GENDERS)


def refine_queries(raw_query: str) -> list[dict[str, str]]:
    """[{gender, query}] for the refine control: the same query constrained to one gender."""
    q = raw_query.strip()
    return [{"gender": g, "query": f"{q} {REFINE_SUFFIX[g]}"} for g in GENDERS]


_REFINE_RE = re.compile(r"(.+?)\s+for\s+(women|men)", re.IGNORECASE)


def is_gender_refine_rerun(raw_query: str, messages: Sequence[Any]) -> bool:
    """True when this turn is the Women / Men control re-running the previous user query
    ("<previous query> for women|men"). Such a turn is a fresh search of the same request, not a
    refinement, so cards the shopper already saw must stay eligible."""
    m = _REFINE_RE.fullmatch(raw_query.strip())
    if not m:
        return False
    users = [
        str(msg.get("content") or "")
        for msg in messages
        if isinstance(msg, Mapping) and msg.get("role") == "user"
    ]
    # the current message is already the last user message of the state
    return len(users) >= 2 and users[-2].strip().lower() == m.group(1).strip().lower()


def select_results(
    ranked: Sequence[dict],
    top_k: int,
    *,
    price_max: float | None = None,
    interleave_genders: bool = False,
    max_per_store: int = MAX_PER_STORE,
    mix_window: int | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Pick up to `top_k` cards from `ranked` (best first). Returns (cards, report).

    `mix_window`: with `interleave_genders`, only the first `mix_window` ranked items take part in
    the alternation (the reranker's own picks); the items behind them are used only to fill slots
    once the window is spent, so a gender with nothing good ranked is never padded with whatever
    retrieval put next. None = the whole list.

    report: n_ranked, n_shown, removed {reason: count} (duplicate, store_cap, over_budget,
    no_price), interleaved (bool), short_reason (None, or why fewer than top_k were shown)."""
    removed: Counter[str] = Counter()
    chosen: list[dict] = []
    legacy: set[tuple[str, str]] = set()
    per_store: Counter[str] = Counter()

    def reject_reason(it: dict) -> str | None:
        if price_max is not None:
            p = _price(it)
            if p is None:
                return "no_price"
            if p > price_max:
                return "over_budget"
        if legacy_key(it) in legacy or any(same_product(it, c) for c in chosen):
            return "duplicate"
        if per_store[_store(it)] >= max_per_store:
            return "store_cap"
        return None

    def take(it: dict) -> None:
        chosen.append(it)
        legacy.add(legacy_key(it))
        per_store[_store(it)] += 1

    interleaved = False
    window = len(ranked) if mix_window is None else mix_window
    if interleave_genders and len({gender_of(it) for it in ranked[:window]} & set(GENDERS)) == 2:
        interleaved = True
        queues: dict[str, list[tuple[int, dict]]] = {"women": [], "men": [], "other": []}
        for idx, it in enumerate(ranked[:window]):
            queues[gender_of(it)].append((idx, it))

        def head(kind: str) -> tuple[int, dict] | None:
            q = queues[kind]
            while q:
                why = reject_reason(q[0][1])
                if why is None:
                    return q[0]
                removed[why] += 1
                q.pop(0)
            return None

        turn: str | None = None
        while len(chosen) < top_k:
            heads = {k: head(k) for k in queues}
            if turn is None:
                avail = [(k, h) for k, h in heads.items() if h]
                if not avail:
                    break
                kind, pick = min(avail, key=lambda kh: kh[1][0])
            else:
                other_turn = "men" if turn == "women" else "women"
                first = heads[turn] or heads[other_turn]
                cands = [(turn if heads[turn] else other_turn, first), ("other", heads["other"])]
                avail = [(k, h) for k, h in cands if h]
                if not avail:
                    break
                kind, pick = min(avail, key=lambda kh: kh[1][0])
            queues[kind].pop(0)
            take(pick[1])
            if kind in GENDERS:
                turn = "men" if kind == "women" else "women"
        for it in ranked[window:]:  # window spent: fill remaining slots in plain ranked order
            if len(chosen) >= top_k:
                break
            why = reject_reason(it)
            if why is None:
                take(it)
            else:
                removed[why] += 1
    else:
        for it in ranked:
            if len(chosen) >= top_k:
                break
            why = reject_reason(it)
            if why is None:
                take(it)
            else:
                removed[why] += 1

    short_reason = None
    if len(chosen) < top_k:
        short_reason = (
            "candidate list exhausted after " + ", ".join(f"{k}={v}" for k, v in sorted(removed.items()))
            if removed
            else "fewer candidates than top_k"
        )
    return chosen, {
        "n_ranked": len(ranked),
        "n_shown": len(chosen),
        "removed": dict(removed),
        "interleaved": interleaved,
        "short_reason": short_reason,
    }
