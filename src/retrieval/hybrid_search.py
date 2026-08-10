from __future__ import annotations

import math
import re

import faiss
import numpy as np
import pandas as pd

from src.catalogue.cleaning import is_fabric_bolt_text, is_kids_item
from src.config.stores import get_inactive_stores

from .dense_search import DenseRetriever
from .sparse_search import SparseRetriever

_CATEGORY_SUFFIXES = frozenset(
    {
        "blouse",
        "shirt",
        "top",
        "tee",
        "t shirt",
        "tshirt",
        "dress",
        "skirt",
        "trousers",
        "trouser",
        "pants",
        "jeans",
        "jacket",
        "coat",
        "blazer",
        "sweater",
        "jumper",
        "cardigan",
        "hoodie",
        "shoe",
        "shoes",
        "bag",
        "shorts",
        "leggings",
        "tights",
        "vest",
        "bodysuit",
        "dungarees",
        "jumpsuit",
        "playsuit",
        "bikini",
    }
)

# Store-redundancy penalty applied per additional item from the same store.
# With penalty=0.5, the 2nd item from store X costs 0.5*(1-λ) in score, the 3rd costs
# 0.25*(1-λ), etc.  Chosen so a second item from the same store is penalised but can still
# beat an irrelevant item from a different store.
_STORE_PENALTY: float = 0.5

# F2 relevance floor — items below this RRF score are excluded as noise.
# Locked at 0.0060 post-F1 rebuild (≈ p5-p10 across 5 calibration queries on the
# clean index). The primary relevance gate is the product_type_name filter; this floor
# is the backstop for queries with genuinely no catalogue matches.
_RELEVANCE_FLOOR: float = 0.0060

# Fabric bolts / unstitched material — not wearable garments. Myntra lists them
# under product_type="dress", so they dominate BM25 scores for dress queries.
# is_fabric_bolt_text (src/catalogue/cleaning.py) is the single source of truth,
# shared with graph.py's _is_material — a "blouse piece" mention alone does NOT
# exclude a row when it is also a finished saree.  Applied in the BM25 pre-filter
# so the retrieval window is filled by real garments, not fabric bolts.


def normalize_prod_name(name: str) -> str:
    """Normalize a product name for dedup.

    Strips punctuation and trailing category-suffix words so that
    'Gyda blouse' and 'Gyda!' both reduce to 'gyda', while
    'Miami Slim' and 'Miami Slim HW' remain distinct.
    """
    if not name:
        return ""
    n = name.lower()
    # Fuse hyphenated garment terms before general punct removal so
    # "t-shirt" → "tshirt" (in suffixes) rather than "t" + "shirt" split.
    n = re.sub(r"\bt-shirt\b", "tshirt", n)
    # Replace remaining non-alphanumeric chars with spaces
    n = re.sub(r"[^\w\s]", " ", n)
    words = n.split()
    # Remove trailing category suffixes (loop handles "slim trousers" → "slim")
    while words and words[-1] in _CATEGORY_SUFFIXES:
        words.pop()
    return " ".join(words)


def store_diversity_rerank(
    candidates: list[dict],
    top_k: int,
    store_diversity: float,
) -> list[dict]:
    """Greedy MMR-style re-rank to spread results across stores.

    Formula
    -------
    At each greedy step, score every remaining candidate as::

        mmr_score(item) = λ * rel_norm(item) - (1-λ) * (1 - penalty ** n_store)

    where:
      - λ = 1 - store_diversity  (so λ=1.0 → pure relevance, λ=0.5 → balanced)
      - rel_norm(item) = item's RRF score / max_rrf_score  (normalised to [0,1])
      - n_store = how many items from item's store are already in selected list
      - penalty = _STORE_PENALTY = 0.5  (geometric decay per extra same-store item)
      - redundancy term = 1 - 0.5**n_store   (0 when n_store=0; →1 as n_store grows)

    Properties:
      - store_diversity=0.0 → redundancy term is zero → pure relevance order preserved
      - store_diversity=1.0 → λ=0 → pure diversity (ignores relevance)
      - The #1 RRF result is always selected first (rel_norm=1, redundancy=0 at step 0),
        preserving the most-relevant result at the top.

    Guards (skip re-rank, return pure-relevance order):
      - store_diversity == 0.0
      - fewer than 2 distinct stores in the candidate window

    Parameters
    ----------
    candidates:
        Full list of filter-passing result dicts, ordered by descending RRF score.
        Each must have 'score' (RRF float) and 'store' (str | None).
    top_k:
        Number of items to return.
    store_diversity:
        Knob in [0.0, 1.0].  0.0 = pure relevance (current behaviour). Default
        in config.yaml is 0.0; owner sets to desired value after reviewing sweep table.

    Returns
    -------
    Re-ranked list of at most top_k items.
    """
    if not candidates:
        return []

    # Guard: no-op when knob is off
    if store_diversity == 0.0:
        return candidates[:top_k]

    # Guard: no-op if fewer than 2 distinct stores in candidate window
    distinct_stores = {c["store"] for c in candidates if c["store"]}
    if len(distinct_stores) < 2:
        return candidates[:top_k]

    lam = 1.0 - store_diversity

    # Normalise RRF scores to [0,1] over the candidate window
    max_score = max(c["score"] for c in candidates)
    if max_score <= 0:
        return candidates[:top_k]

    selected: list[dict] = []
    remaining: list[dict] = list(candidates)
    store_counts: dict[str | None, int] = {}

    for _ in range(min(top_k, len(candidates))):
        best_item: dict | None = None
        best_mmr: float = float("-inf")

        for item in remaining:
            rel_norm = item["score"] / max_score
            n_store = store_counts.get(item["store"], 0)
            redundancy = 1.0 - (_STORE_PENALTY**n_store)
            mmr = lam * rel_norm - (1.0 - lam) * redundancy
            if mmr > best_mmr:
                best_mmr = mmr
                best_item = item

        if best_item is None:
            break

        selected.append(best_item)
        remaining.remove(best_item)
        s = best_item["store"]
        store_counts[s] = store_counts.get(s, 0) + 1

    return selected


def dedup_candidates_keep_cheapest(candidates: list[dict]) -> list[dict]:
    """Dedup a candidate pool by (normalized prod_name, colour), keeping the cheapest.

    Runs at candidate-collection time — BEFORE the store-diversity rerank — so
    near-duplicate listings (same product, same colour, often re-listed across
    stores or in multiple sizes) never crowd the pool handed to the diversity
    rerank and the LLM reranker. The existing post-rerank dedup in
    ``src/agents/graph.py`` remains as a safety net for anything that slips through
    (e.g. duplicates introduced by the anchor-based retrieval path).

    Preserves the relative order of the FIRST-seen item per duplicate group (i.e.
    RRF order is unaffected), but keeps the cheapest item's fields for that group
    when a cheaper duplicate appears later in the pool. Items with unknown/None
    price never displace a priced item within the same group.

    Parameters
    ----------
    candidates:
        Full list of filter-passing result dicts, in descending-RRF-score order.

    Returns
    -------
    Deduped list, same relative order as the input (one entry per group, positioned
    at that group's first occurrence).
    """
    best: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []

    for item in candidates:
        key = (
            normalize_prod_name(item.get("prod_name") or item.get("display_name", "")),
            (item.get("colour") or "").lower(),
        )
        price = item.get("price_inr")
        price_sort = price if isinstance(price, (int, float)) else float("inf")

        if key not in best:
            best[key] = item
            order.append(key)
        else:
            cur_price = best[key].get("price_inr")
            cur_sort = cur_price if isinstance(cur_price, (int, float)) else float("inf")
            if price_sort < cur_sort:
                best[key] = item

    return [best[k] for k in order]


def apply_per_store_cap(
    selected: list[dict],
    full_pool: list[dict],
    cap: int,
    top_k: int,
) -> list[dict]:
    """Enforce a per-store cap on the candidate pool handed to the LLM reranker.

    Walks *selected* (already store-diversity-reranked, in final order) and keeps
    at most *cap* items per store. If the cap drops the pool below *top_k*,
    backfills the shortfall from *full_pool* (the wider RRF-ordered candidate
    window, pre-diversity-rerank) with items from stores still under cap, in RRF
    order — so the reranker still receives up to *top_k* items when the wider pool
    has more diversity to offer than the MMR selection surfaced.

    Guards (no-op, returns *selected* unchanged):
      - ``cap <= 0`` (or None).
      - Fewer than 2 distinct stores in *full_pool* — mirrors store_diversity_rerank's
        own single-store guard. Single-brand indices (e.g. the legacy H&M-only index)
        have no ``store`` column, so every item's store is ``None``; without this
        guard the cap would wrongly collapse ALL of them into one "store" bucket and
        truncate the pool for a purely cross-store feature that doesn't apply here.
        Checked against *full_pool* (not *selected*) so a narrow query where the
        diversity rerank still surfaces only one store's items — while OTHER stores
        exist deeper in the RRF pool — still gets the cap+backfill treatment.

    Parameters
    ----------
    selected:
        Store-diversity-reranked candidate list (output of store_diversity_rerank).
    full_pool:
        The full filter-passing candidate pool in RRF order (pre-diversity-rerank),
        used both as the cross-store-potential check and as a backfill source.
    cap:
        Maximum items per store to keep. Config knob: retrieval.per_store_cap.
    top_k:
        Target pool size to backfill up to.
    """
    if not cap or cap <= 0:
        return selected

    distinct_stores = {item.get("store") for item in full_pool}
    if len(distinct_stores) < 2:
        return selected

    store_counts: dict[str | None, int] = {}
    kept: list[dict] = []
    kept_ids: set[str] = set()

    for item in selected:
        store = item.get("store")
        n = store_counts.get(store, 0)
        if n < cap:
            kept.append(item)
            kept_ids.add(item["article_id"])
            store_counts[store] = n + 1
        # else: dropped — this store already has `cap` items in the pool

    if len(kept) < top_k:
        for item in full_pool:
            if len(kept) >= top_k:
                break
            if item["article_id"] in kept_ids:
                continue
            store = item.get("store")
            n = store_counts.get(store, 0)
            if n < cap:
                kept.append(item)
                kept_ids.add(item["article_id"])
                store_counts[store] = n + 1

    return kept


def compute_adaptive_per_store_cap(full_pool: list[dict], base_cap: int, top_k: int) -> int:
    """Loosen `base_cap` when a query's qualifying pool is itself store-concentrated.

    A single fixed cap mechanically truncates recall on any query whose TRUE
    qualifying universe happens to sit almost entirely in one or two stores
    (e.g. "maroon dupatta, women" is 95% one store in this catalogue) — there
    is nothing genuine to diversify against, so the cap only costs recall
    there. Relaxes proportionally to how many distinct stores actually hold
    qualifying inventory (``full_pool``, the same wide pre-diversity-rerank
    pool ``apply_per_store_cap`` already receives), and is a no-op wherever a
    query's pool is already multi-store the way ``per_store_cap`` was
    designed for.

    Formula: ``max(base_cap, ceil(top_k / n_stores))``, where ``n_stores`` is
    the number of distinct stores in ``full_pool``. Degenerates correctly at
    both ends via that single ``max()`` — no extra branching needed:
      - ``n_stores == 1``: ``ceil(top_k/1) == top_k``, so the cap can no
        longer truncate a single-store universe below `top_k` (in practice
        ``apply_per_store_cap``'s own <2-distinct-stores guard already no-ops
        here too — this is a redundant-but-harmless second safety net).
      - ``n_stores >= top_k / base_cap``: ``ceil(top_k/n_stores) <= base_cap``,
        so ``max(...)`` picks `base_cap` — byte-identical to the fixed-cap
        behaviour this replaces.

    Validated (eval/adaptive_per_store_cap_test.py, 2026-08-10): niche-subset
    recall@50 +3.1pp (0.438->0.469), broad-query P@5/P@10 byte-identical (no
    regression), real hand-labeled strict P@5 +0.5pp overall and +2.2pp on the
    `occasion` category specifically — see
    reports/adaptive_per_store_cap_strict_eval_20260810T122542Z.md. Unlike the
    earlier GLOBAL cap 4->8 raise (reverted: -0.8pp strict P@5, -2.4pp on
    `occasion` — reports/pushdown_fix_20260806.md), this is query-scoped, so
    genuinely multi-store queries (where the cap's diversity purpose still
    applies) are mathematically unaffected.

    Parameters
    ----------
    full_pool:
        The full filter-passing candidate pool (pre-diversity-rerank) —
        `apply_per_store_cap`'s own `full_pool` argument.
    base_cap:
        The configured `retrieval.per_store_cap` value. Returned unchanged
        when <= 0 (cap disabled) or when `full_pool` holds 0-1 stores.
    top_k:
        Target pool size — `apply_per_store_cap`'s own `top_k` argument.
    """
    if not base_cap or base_cap <= 0:
        return base_cap
    n_stores = len({item.get("store") for item in full_pool})
    if n_stores <= 0:
        return base_cap
    return max(base_cap, math.ceil(top_k / n_stores))


def _facet_value_matches(facets: dict, key: str, value: object) -> bool:
    """True if `facets[key]` matches `value`.

    `value` may be a list/tuple/set — matches ANY member (isin semantics),
    e.g. colour_group_name=["Navy Blue", "Dark Blue"] for a "navy" query, so a
    colour-family query isn't silently narrowed to only ONE of several
    catalogue-distinct near-synonym buckets (see intent_parser._COLOUR_FAMILY).
    Plain string values keep exact-equality semantics, unchanged.
    """
    item_val = str(facets.get(key, "")).lower()
    if isinstance(value, (list, tuple, set)):
        return item_val in {str(v).lower() for v in value}
    return item_val == str(value).lower()


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        catalogue_df: pd.DataFrame,
        config: dict,
    ):
        self.dense = dense
        self.sparse = sparse
        self.catalogue_df = catalogue_df.set_index("article_id")
        self.config = config

        # Precomputed once (not per-.search() call): is_kids_item's regex scan
        # over prod_name measured at ~150ms for the full 52k-row unified
        # catalogue (regex work isn't sped up by pandas vectorization the way
        # simple string equality/isin comparisons are — see _not_fabric_mask/
        # _not_inactive_store_mask above, which stay cheap and ARE recomputed
        # per-call). Recomputing this mask on every search() call added ~165ms
        # of latency per request (measured, HybridRetriever.search: ~170ms ->
        # ~335ms) for a value that never changes for the lifetime of this
        # retriever instance (catalogue_df is fixed at construction).
        self._not_kids_mask: np.ndarray | None = None
        if "prod_name" in self.catalogue_df.columns:
            self._not_kids_mask = (
                ~self.catalogue_df["prod_name"].fillna("").apply(is_kids_item)
            ).values

        # article_id -> position-in-FAISS-index lookup, precomputed once (mirrors
        # SparseRetriever's own _id_to_pos pattern) so the gender/colour/price
        # pushdown mask (see search()) can be translated into a faiss.IDSelectorArray
        # without an O(n) scan per call.
        self._dense_id_to_pos: dict[str, int] = {
            str(aid): i for i, aid in enumerate(self.dense.article_ids)
        }

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        if top_k is None:
            top_k = self.config["retrieval"]["final_k"]
        fetch_k = self.config["retrieval"]["top_k"]
        rrf_k = self.config["retrieval"]["rrf_k"]

        # Pre-filter BM25 by product_type_name when that facet filter is present.
        # BM25 already scores all 44k items before argsort; masking here costs nothing
        # extra and guarantees the sparse retrieval window is filled by the right type
        # rather than by unrelated items that happen to mention the type word in text.
        # Also exclude fabric-bolt items (unstitched dress material etc.) from the BM25
        # window: they match the type filter correctly but score very high for garment
        # queries (e.g. "black dress" → 49/52 catalogue entries are fabric bolts that
        # dominate BM25 and crowd out the 3 real dresses).  The same exclusion fires
        # post-retrieval in graph.py; applying it here ensures real garments fill the
        # BM25 window before the graph-level filter runs.
        # Dense (FAISS) uses the wider fetch_k window to compensate for no pre-filter.
        # fabric_material items (unstitched bolts, blouse pieces) must never appear in
        # garment search results.  Exclude them permanently from the BM25 window so they
        # cannot crowd out real garments regardless of whether a type filter is set.
        _not_fabric_mask: np.ndarray | None = None
        if "product_type_name" in self.catalogue_df.columns:
            _not_fabric_mask = (
                self.catalogue_df["product_type_name"].str.lower() != "fabric_material"
            ).values  # boolean array aligned with catalogue_df

        # Inactive stores (e.g. berrylush — password-walled since 2026-07) must never
        # appear in ANY result path.  STORE_CONFIG is the single source of truth; mirror
        # the fabric_material exclusion mechanism so inactive-store rows are stripped
        # out of the BM25 window before scoring, same as fabric bolts.
        _not_inactive_store_mask: np.ndarray | None = None
        if "store" in self.catalogue_df.columns:
            inactive_stores = get_inactive_stores()
            if inactive_stores:
                _not_inactive_store_mask = (
                    ~self.catalogue_df["store"].str.lower().isin(inactive_stores)
                ).values  # boolean array aligned with catalogue_df

        # Juniors/girls/boys/kids items must never appear in ANY search result —
        # the catalogue's gender column mislabels many of them as "women"/"men"
        # (see src.catalogue.cleaning.is_kids_item docstring), so a gender filter
        # alone does not exclude them. Live-proven: "red lehenga bridal" (a
        # non-occasion-keyword query, so the occasion-gated kids check in
        # graph.py never even ran) surfaced girls' lehengas ranked above adult
        # bridal items. Mirror the fabric_material exclusion mechanism so kids
        # rows are stripped out of the BM25 window before scoring, same as
        # fabric bolts and inactive stores. Precomputed once in __init__ (see
        # comment there) rather than recomputed here on every call.
        _not_kids_mask = self._not_kids_mask

        def _combine_masks(*masks: np.ndarray | None) -> np.ndarray | None:
            """AND together any non-None boolean masks aligned with catalogue_df."""
            present = [m for m in masks if m is not None]
            if not present:
                return None
            combined = present[0]
            for m in present[1:]:
                combined = combined & m
            return combined

        _exclusion_mask = _combine_masks(
            _not_fabric_mask, _not_inactive_store_mask, _not_kids_mask
        )

        # Extract optional store + gender filters BEFORE fetching (hoisted up from
        # after RRF fusion — see the pushdown-mask comment below for why).
        # Both live in direct catalogue columns, not in the `facets` dict, so they're
        # handled separately from the generic facet-filter loop below.
        # gender filter replaces index_group_name: Shopify stores (virgio, fashor, etc.)
        # have index_group_name="N/A" but carry an accurate gender="women"/"men" column
        # derived from brand_config.gender_default at ingest time.
        store_filter: str | None = None
        gender_filter: str | None = None
        remaining_filters: dict | None = None
        if filters:
            store_filter = filters.get("store") or None
            gender_filter = filters.get("gender") or None

            # Translate index_group_name (H&M/Myntra-only vocabulary) to the gender column
            # so the gender filter works across all stores (Shopify stores have
            # index_group_name="N/A" but carry an accurate gender column).
            ign = (filters.get("index_group_name") or "").lower()
            if ign == "ladieswear":
                gender_filter = gender_filter or "women"
            elif ign == "menswear":
                gender_filter = gender_filter or "men"

            remaining_filters = {
                k: v
                for k, v in filters.items()
                if k not in ("store", "gender", "index_group_name")
            } or None

        sparse_allowed_ids: np.ndarray | None = None
        type_filter_val = (filters or {}).get("product_type_name")
        type_mask: pd.Series | None = None
        if type_filter_val is not None and "product_type_name" in self.catalogue_df.columns:
            pt_col = self.catalogue_df["product_type_name"].str.lower()
            type_mask = pt_col == type_filter_val.lower()
            if "prod_name" in self.catalogue_df.columns:
                not_material = ~self.catalogue_df["prod_name"].fillna("").apply(
                    is_fabric_bolt_text
                )
                type_mask = type_mask & not_material
            if _exclusion_mask is not None:
                type_mask = type_mask & _exclusion_mask

        # --- Pushdown mask: gender + colour_group_name + price_min/max applied
        # BEFORE fetch, not after. Root-caused (eval/diagnose_pushdown_mechanism.py,
        # 2026-08-06): dense/BM25 ranking carries no colour/price signal, so items
        # satisfying an exact colour+price constraint were getting crowded out of
        # the fetch window by same-gender/same-type items of the WRONG colour/price
        # long before the post-fetch filter loop below ever saw them — confirmed by
        # measuring recall@50 jump from 8-42% (post-fetch filtering, status quo) to
        # 100% (pushed into fetch) on 3 traced queries, at zero extra latency
        # (IndexFlatIP already scores the full index per call regardless of
        # restriction; BM25's allowed_ids mask already existed for product_type_name
        # alone). Gender is included here too even though it turned out NOT to be
        # the dominant loss (the unfiltered pool is already ~81-99% correct-gender
        # from the query text alone) — pushing it is free and modestly helps.
        # A None mask (no gender/colour/price filter set) preserves the exact prior
        # behaviour (unrestricted dense.search, type-only-or-unfiltered sparse).
        _pushdown_mask: pd.Series | None = None
        if gender_filter is not None or remaining_filters:
            _pushdown_mask = pd.Series(True, index=self.catalogue_df.index)
            if gender_filter is not None:
                gender_col = self.catalogue_df.get("gender")
                if gender_col is None:
                    _pushdown_mask &= False
                else:
                    _pushdown_mask &= (
                        gender_col.fillna("unknown").astype(str).str.lower() == gender_filter.lower()
                    )
            if remaining_filters:
                colour_val = remaining_filters.get("colour_group_name")
                if colour_val is not None:
                    colour_col = self.catalogue_df.get("colour_group_name")
                    if colour_col is None:
                        _pushdown_mask &= False
                    else:
                        colour_lower = colour_col.fillna("").astype(str).str.lower()
                        if isinstance(colour_val, (list, tuple, set)):
                            _pushdown_mask &= colour_lower.isin({str(v).lower() for v in colour_val})
                        else:
                            _pushdown_mask &= colour_lower == str(colour_val).lower()
                price_min = remaining_filters.get("price_min")
                price_max = remaining_filters.get("price_max")
                if price_min is not None or price_max is not None:
                    price_col = self.catalogue_df.get("price_inr")
                    if price_col is None:
                        _pushdown_mask &= False
                    else:
                        if price_min is not None:
                            _pushdown_mask &= price_col.fillna(float("-inf")) >= float(price_min)
                        if price_max is not None:
                            _pushdown_mask &= price_col.fillna(float("inf")) <= float(price_max)
            if _exclusion_mask is not None:
                _pushdown_mask &= _exclusion_mask
            if type_mask is not None:
                _pushdown_mask &= type_mask

        _fetch_restriction_mask = _pushdown_mask if _pushdown_mask is not None else type_mask
        if _fetch_restriction_mask is None and _exclusion_mask is not None:
            _fetch_restriction_mask = pd.Series(_exclusion_mask, index=self.catalogue_df.index)

        if _fetch_restriction_mask is not None:
            sparse_allowed_ids = self.catalogue_df.index[_fetch_restriction_mask].values.astype(str)
            _allowed_positions = np.fromiter(
                (
                    self._dense_id_to_pos[aid]
                    for aid in sparse_allowed_ids
                    if aid in self._dense_id_to_pos
                ),
                dtype=np.int64,
            )
            _dense_params = faiss.SearchParameters(sel=faiss.IDSelectorArray(_allowed_positions))
        else:
            _dense_params = None

        dense_hits = self.dense.search(query, top_k=fetch_k * 2, params=_dense_params)
        sparse_hits = self.sparse.search(query, top_k=fetch_k * 2, allowed_ids=sparse_allowed_ids)

        rrf_scores: dict[str, float] = {}
        for rank, (article_id, _) in enumerate(dense_hits, start=1):
            rrf_scores[article_id] = rrf_scores.get(article_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (article_id, _) in enumerate(sparse_hits, start=1):
            rrf_scores[article_id] = rrf_scores.get(article_id, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Collect ALL filter-passing candidates from the full RRF window.
        # We do NOT truncate here — diversity re-rank needs the full candidate pool.
        candidates: list[dict] = []
        for article_id, score in ranked:
            if score < _RELEVANCE_FLOOR:
                continue  # skip noise; ranked is not guaranteed sorted so use continue not break
            if article_id not in self.catalogue_df.index:
                continue
            row = self.catalogue_df.loc[article_id]
            facets = row["facets"] if isinstance(row["facets"], dict) else {}

            # --- Inactive-store exclusion (belt-and-suspenders on top of the BM25 mask) ---
            # The BM25 pre-filter already strips inactive-store rows from its window, but
            # dense (FAISS) hits are never pre-filtered, so this check is the only guard on
            # that path.  STORE_CONFIG is the single source of truth (see get_inactive_stores).
            item_store_raw = (
                str(row["store"]).lower()
                if "store" in row.index and row["store"] is not None
                else ""
            )
            if item_store_raw in get_inactive_stores():
                continue

            # --- Store filter (cross-store unified index only; no-op on per-brand indices) ---
            if store_filter is not None:
                if item_store_raw != store_filter.lower():
                    continue

            # --- Gender filter (column-level; covers Shopify stores with index_group_name="N/A") ---
            item_gender = (
                str(row["gender"]).lower()
                if "gender" in row.index and row["gender"] is not None
                else "unknown"
            )
            if gender_filter is not None:
                # Explicit gender filter: exclude the OPPOSITE gender AND anything outside
                # {men, women} (unknown/null/empty).  Previously "unknown" rows were always
                # kept, which let items with no verified gender (mostly store=globalrepublic)
                # leak into e.g. a "men" search — the men's constraint was silently unenforced
                # against them.  With no gender filter, behaviour is unchanged (unknown rows
                # still pass through normally).
                if item_gender not in ("men", "women") or item_gender != gender_filter.lower():
                    continue

            if remaining_filters:
                price_min = remaining_filters.get("price_min")
                price_max = remaining_filters.get("price_max")
                facet_filters = {
                    k: v
                    for k, v in remaining_filters.items()
                    if k not in ("price_min", "price_max")
                }

                if facet_filters and not all(
                    _facet_value_matches(facets, k, v) for k, v in facet_filters.items()
                ):
                    continue

                if price_min is not None or price_max is not None:
                    item_price = (
                        row.get("price_inr")
                        if hasattr(row, "get")
                        else row["price_inr"]
                        if "price_inr" in row.index
                        else None
                    )
                    if item_price is None or not isinstance(item_price, (int, float)):
                        continue  # skip items without price when price filter is active
                    if price_min is not None and float(item_price) < float(price_min):
                        continue
                    if price_max is not None and float(item_price) > float(price_max):
                        continue

            candidates.append(
                {
                    "article_id": article_id,
                    "prod_name": row.get("prod_name", ""),
                    "display_name": row["display_name"],
                    "colour": facets.get("colour_group_name", ""),
                    "product_type": facets.get("product_type_name", ""),
                    "department": facets.get("department_name", ""),
                    "detail_desc": row["detail_desc"],
                    "image_url": _img
                    if isinstance(_img := row.get("image_url"), str) and _img
                    else None,
                    "score": score,
                    "store": (
                        str(row["store"])
                        if "store" in row.index and row["store"] is not None
                        else None
                    ),
                    "price_inr": (
                        float(row["price_inr"])
                        if "price_inr" in row.index
                        and row["price_inr"] is not None
                        and not pd.isna(row["price_inr"])
                        else None
                    ),
                    "pdp_handle": (
                        str(row["pdp_handle"])
                        if "pdp_handle" in row.index and row["pdp_handle"] is not None
                        else None
                    ),
                    "pdp_live": (
                        bool(row["pdp_live"])
                        if "pdp_live" in row.index
                        and row["pdp_live"] is not None
                        and not pd.isna(row["pdp_live"])
                        else None
                    ),
                    "gender": item_gender,
                }
            )

        # --- Candidate-pool dedup (pre-rerank) ---
        # Dedup by (normalized prod_name, colour) BEFORE the diversity rerank so
        # near-duplicate listings never occupy a slot in the pool handed to the LLM
        # reranker. graph.py's post-rerank dedup remains as a safety net.
        candidates = dedup_candidates_keep_cheapest(candidates)

        # --- Store-diversity re-rank (MMR nudge) ---
        # Guard: skip when a store filter is set (user explicitly narrowed to one store).
        store_diversity: float = self.config["retrieval"].get("store_diversity", 0.0)
        if store_filter is not None:
            results = candidates[:top_k]
        else:
            results = store_diversity_rerank(candidates, top_k, store_diversity)
            per_store_cap: int = self.config["retrieval"].get("per_store_cap", 0)
            if per_store_cap:
                effective_cap = compute_adaptive_per_store_cap(candidates, per_store_cap, top_k)
                results = apply_per_store_cap(results, candidates, effective_cap, top_k)

        # Deprioritize items with known-dead PDP links — move them to end of list
        live = [it for it in results if it.get("pdp_live") is not False]
        dead = [it for it in results if it.get("pdp_live") is False]
        return live + dead
