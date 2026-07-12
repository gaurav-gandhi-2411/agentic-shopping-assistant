from __future__ import annotations

"""Two-tower fused-embedding head-to-head evaluation.

Compares three item embedding schemes on the unified 6-store catalogue:
  (a) CLIP-512 only  -- current image-path embeddings
  (b) SBERT-384 only -- current text-path embeddings (MiniLM-L6-v2)
  (c) Two-tower FUSED-256 -- ItemTower MLP from the H&M-trained sibling model
      applied to concat[CLIP-512 ; SBERT-384] -> 256-d L2-normed

Measurement ONLY.  The fused index is a scratch artifact; the live retrieval
path (hybrid_search / dense / clip) is NOT modified.

== ItemTower forward signature ==
  item_tower(image_emb: Tensor[B,512], text_emb: Tensor[B,384]) -> Tensor[B,256]
  Internally: concat -> Linear(896,512) -> GELU -> Dropout -> Linear(512,256) -> L2-norm
  Source: multimodal-fashion-recommender/src/models/item_tower.py

== Degenerate-input honesty ==
TEXT QUERY path: encodes SBERT(query_text)[384] into fused via
  image_emb=zeros(512), text_emb=SBERT(query)[384]
  The MLP was trained on real CLIP+SBERT item pairs; zero-padded image input is
  OUT-OF-DISTRIBUTION.  Results labeled explicitly as degenerate.

IMAGE QUERY path: encodes CLIP(image)[512] into fused via
  image_emb=CLIP(image)[512], text_emb=zeros(384)
  Same OOD caveat.  Results labeled as degenerate.

== Tasks ==
1. ITEM->ITEM similarity (leave-one-out, n_sample=200, seed=42):
   Metrics: category_consistency@10, cross_store@10 (eligible items only)
   Comparisons: CLIP-512 vs SBERT-384 vs FUSED-256

2. TEXT QUERY (degenerate, zero-padded image):
   Metrics: relevance@10 using the same fixtures as cross_store_retrieval.py
   Comparisons: current HybridRetriever (MiniLM+BM25+RRF) vs FUSED-256[zero+SBERT]

3. IMAGE QUERY (degenerate, zero-padded text):
   Metrics: category_consistency@10, cross_store@10 (eligible only)
   Comparisons: CLIP-512 only vs FUSED-256[CLIP+zero]

Runnable as:
    python -m eval.twotower_compare
    python eval/twotower_compare.py
"""

import importlib.util as _ilu  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import faiss  # type: ignore[import-untyped]  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SIBLING_REPO = Path("C:/Users/gaura/ml-projects/multimodal-fashion-recommender")

# Import this repo's packages FIRST (before any sibling path manipulation)
from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402


def _import_from_file(module_name: str, file_path: Path) -> Any:
    """Import a Python file as a named module, bypassing sys.path shadowing."""
    spec = _ilu.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot find spec for {file_path}")
    mod = _ilu.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Import sibling model classes by file path (avoids 'src' namespace collision).
# two_tower.py imports 'from src.models.item_tower import ItemTower' etc., so we
# must register the sibling modules under the canonical src.models.* keys BEFORE
# importing two_tower.py.  This is safe: this repo does not have a
# src/models/item_tower.py so we are not clobbering an existing module.
_sibling_item_tower = _import_from_file(
    "src.models.item_tower",
    _SIBLING_REPO / "src" / "models" / "item_tower.py",
)
_sibling_user_tower = _import_from_file(
    "src.models.user_tower",
    _SIBLING_REPO / "src" / "models" / "user_tower.py",
)
_sibling_two_tower_mod = _import_from_file(
    "src.models.two_tower",
    _SIBLING_REPO / "src" / "models" / "two_tower.py",
)
TwoTowerModel = _sibling_two_tower_mod.TwoTowerModel  # noqa: N816

from eval.cross_store_retrieval import (  # noqa: E402, I001
    QueryFixture,
    _is_relevant,
    _load_fixtures,
    compute_cross_store_eligible_types,
    is_query_cross_store_eligible,
)

# -- Paths --------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent
_FIXTURE_PATH = _EVAL_DIR / "fixtures" / "cross_store_retrieval.yaml"
_UNIFIED_DIR = _REPO_ROOT / "data" / "processed" / "unified"
_CLIP_DIR = _REPO_ROOT / "data" / "processed" / "clip" / "unified"
_CHECKPOINT_PATH = _SIBLING_REPO / "checkpoints" / "best.pt"

_RNG_SEED: int = 42
_N_SAMPLE: int = 200
_NEIGHBORS_K: int = 10

# -- Data structures ----------------------------------------------------------


@dataclass
class ItemItemResult:
    """Leave-one-out item->item metrics for one sampled item across all schemes."""

    article_id: str
    store: str
    product_type: str
    cross_store_eligible: bool
    # category_consistency@10 for each scheme
    cat_consistency_clip: float
    cat_consistency_sbert: float
    cat_consistency_fused: float
    # cross-store neighbour for each scheme
    cross_store_clip: bool
    cross_store_sbert: bool
    cross_store_fused: bool


@dataclass
class TextQueryResult:
    """Text query relevance@10 for current hybrid vs fused-256 (degenerate)."""

    query_id: str
    query: str
    cross_store_eligible: bool
    relevance_at_10_hybrid: float  # current MiniLM+BM25+RRF
    relevance_at_10_fused: float   # fused-256 with zero CLIP (OOD)


@dataclass
class ImageQueryResult:
    """Image-query (leave-one-out) comparison: CLIP-512 vs FUSED-256[CLIP+zero]."""

    article_id: str
    store: str
    product_type: str
    cross_store_eligible: bool
    cat_consistency_clip: float
    cat_consistency_fused_image: float
    cross_store_clip: bool
    cross_store_fused_image: bool


# -- ItemTower loading --------------------------------------------------------


def load_item_tower(checkpoint_path: Path) -> torch.nn.Module:
    """Load the ItemTower MLP from the two-tower checkpoint (CPU, eval mode).

    The checkpoint contains 'config' which specifies the tower dimensions.
    We instantiate TwoTowerModel (which constructs item_tower internally) and
    extract model.item_tower after loading the state dict.

    Forward: item_tower(image_emb[B,512], text_emb[B,384]) -> [B,256] L2-normed
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = TwoTowerModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    item_tower = model.item_tower
    # Freeze: this is measurement only
    for p in item_tower.parameters():
        p.requires_grad_(False)
    return item_tower


# -- Fused index construction -------------------------------------------------


def build_fused_index(
    clip_index: faiss.Index,
    dense_index: faiss.Index,
    article_ids: np.ndarray,
    item_tower: torch.nn.Module,
    *,
    batch_size: int = 512,
) -> tuple[faiss.Index, np.ndarray]:
    """Build a FAISS IndexFlatIP over 256-d fused embeddings (in-memory scratch).

    Vectors are reconstructed from the existing CLIP and dense FAISS indices --
    no network, no model re-embedding.  The ItemTower MLP fuses them CPU-only.

    Both indices must be aligned (same order, same article_ids array).

    Returns
    -------
    fused_index: faiss.IndexFlatIP with 256-d L2-normed vectors
    article_ids: same ID array (unchanged, passed through for clarity)
    """
    n = clip_index.ntotal
    assert dense_index.ntotal == n, (
        f"Index size mismatch: clip={n}, dense={dense_index.ntotal}"
    )

    fused_vecs = np.empty((n, 256), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Reconstruct batch from stored vectors -- no encoders needed
        img_batch = np.stack([clip_index.reconstruct(i) for i in range(start, end)])
        txt_batch = np.stack([dense_index.reconstruct(i) for i in range(start, end)])

        img_t = torch.from_numpy(img_batch)
        txt_t = torch.from_numpy(txt_batch)

        with torch.no_grad():
            fused_t = item_tower(img_t, txt_t)

        fused_vecs[start:end] = fused_t.numpy()

    fused_index = faiss.IndexFlatIP(256)
    fused_index.add(fused_vecs)
    return fused_index, article_ids


# -- Task 1: Item->Item similarity (leave-one-out) ----------------------------


def _leave_one_out_neighbors(
    index: faiss.Index,
    pos: int,
    id_array: np.ndarray,
    query_id: str,
    k: int,
) -> list[str]:
    """Reconstruct vector at pos, search index, drop self, return k neighbor IDs."""
    vec = index.reconstruct(pos).reshape(1, -1).astype(np.float32)
    _scores, indices = index.search(vec, k + 1)
    return [
        str(id_array[idx]) for idx in indices[0] if idx >= 0 and str(id_array[idx]) != query_id
    ][:k]


def run_item_item_eval(
    cat_df: pd.DataFrame,
    clip_index: faiss.Index,
    clip_ids: np.ndarray,
    dense_index: faiss.Index,
    dense_ids: np.ndarray,
    fused_index: faiss.Index,
    fused_ids: np.ndarray,
    eligible_types: frozenset[str],
    *,
    n_sample: int = _N_SAMPLE,
    neighbors_k: int = _NEIGHBORS_K,
    seed: int = _RNG_SEED,
) -> list[ItemItemResult]:
    """Leave-one-out item->item eval for CLIP, SBERT, and FUSED embeddings.

    Assumes clip_ids, dense_ids, fused_ids are all identical and same order
    (verified by alignment check in main()).  Uses clip position map for all
    three since they share the same article_id -> position mapping.

    Parameters
    ----------
    cat_df : unified catalogue with article_id, store, product_type_name.
    clip_index, dense_index, fused_index : FAISS indices to compare.
    clip_ids, dense_ids, fused_ids : aligned article_id arrays.
    eligible_types : product types eligible for cross-store gate.
    n_sample : number of items to sample (stratified by store).
    neighbors_k : neighbors to evaluate per item.
    seed : RNG seed.
    """
    rng = np.random.default_rng(seed)

    # All three share the same ID->position mapping (verified in main)
    id_str = clip_ids.astype(str)
    pos_map: dict[str, int] = {aid: i for i, aid in enumerate(id_str)}

    cat_indexed = cat_df.set_index("article_id")
    clip_in_cat = cat_df[cat_df["article_id"].astype(str).isin(pos_map)].copy()
    clip_in_cat["article_id"] = clip_in_cat["article_id"].astype(str)

    stores = clip_in_cat["store"].unique()
    sampled_ids: list[str] = []
    for store in sorted(stores):
        store_items = clip_in_cat[clip_in_cat["store"] == store]["article_id"].values
        n_store = max(1, round(n_sample * len(store_items) / len(clip_in_cat)))
        n_store = min(n_store, len(store_items))
        chosen = rng.choice(store_items, size=n_store, replace=False)
        sampled_ids.extend(chosen.tolist())

    all_ids = clip_in_cat["article_id"].values.copy()
    rng.shuffle(all_ids)
    existing = set(sampled_ids)
    for aid in all_ids:
        if len(sampled_ids) >= n_sample:
            break
        if aid not in existing:
            sampled_ids.append(aid)
            existing.add(aid)
    sampled_ids = sampled_ids[:n_sample]

    results: list[ItemItemResult] = []

    for aid in sampled_ids:
        pos = pos_map.get(aid)
        if pos is None:
            continue
        try:
            row = cat_indexed.loc[aid]
        except KeyError:
            continue

        query_type = str(row.get("product_type_name", ""))
        query_store = str(row.get("store", ""))
        is_eligible = query_type in eligible_types

        def eval_neighbors(nbr_ids: list[str]) -> tuple[float, bool]:
            """Compute (category_consistency, has_cross_store) for a neighbor list."""
            type_matches = 0
            cross_found = False
            for nid in nbr_ids:
                try:
                    n_row = cat_indexed.loc[nid]
                except KeyError:
                    continue
                if str(n_row.get("product_type_name", "")) == query_type:
                    type_matches += 1
                if str(n_row.get("store", "")) != query_store:
                    cross_found = True
            consistency = type_matches / max(len(nbr_ids), 1)
            return round(consistency, 4), cross_found

        clip_nbrs = _leave_one_out_neighbors(clip_index, pos, id_str, aid, neighbors_k)
        sbert_nbrs = _leave_one_out_neighbors(dense_index, pos, id_str, aid, neighbors_k)
        fused_nbrs = _leave_one_out_neighbors(fused_index, pos, id_str, aid, neighbors_k)

        if not clip_nbrs and not sbert_nbrs and not fused_nbrs:
            continue

        clip_cons, clip_cross = eval_neighbors(clip_nbrs)
        sbert_cons, sbert_cross = eval_neighbors(sbert_nbrs)
        fused_cons, fused_cross = eval_neighbors(fused_nbrs)

        results.append(
            ItemItemResult(
                article_id=aid,
                store=query_store,
                product_type=query_type,
                cross_store_eligible=is_eligible,
                cat_consistency_clip=clip_cons,
                cat_consistency_sbert=sbert_cons,
                cat_consistency_fused=fused_cons,
                cross_store_clip=clip_cross,
                cross_store_sbert=sbert_cross,
                cross_store_fused=fused_cross,
            )
        )

    return results


# -- Task 2: Text query relevance (degenerate) --------------------------------


def _encode_text_fused(
    query: str,
    item_tower: torch.nn.Module,
    dense_retriever: DenseRetriever,
) -> np.ndarray:
    """Encode a text query as [zeros(512) ; SBERT(query)(384)] -> ItemTower -> 256-d.

    This is a DEGENERATE / OOD input: the ItemTower was trained on real CLIP
    image embeddings, not zeros.  Results are expected to be poor and are
    reported with that caveat.
    """
    # Encode text via the SentenceTransformer model inside DenseRetriever (MiniLM-L6-v2)
    text_vec = dense_retriever.model.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)[0]  # [384]
    text_t = torch.from_numpy(text_vec).unsqueeze(0)   # [1, 384]
    img_t = torch.zeros(1, 512)                        # [1, 512] -- OOD zero padding

    with torch.no_grad():
        fused = item_tower(img_t, text_t)  # [1, 256] L2-normed

    return fused.numpy().astype(np.float32)


def run_text_query_eval(
    fixtures: list[QueryFixture],
    hybrid_retriever: HybridRetriever,
    item_tower: torch.nn.Module,
    dense_retriever: DenseRetriever,
    fused_index: faiss.Index,
    fused_ids: np.ndarray,
    cat_df: pd.DataFrame,
    eligible_types: frozenset[str],
) -> list[TextQueryResult]:
    """Compare HybridRetriever vs degenerate-fused text query retrieval.

    The fused query uses zeros(512) for the image side — clearly OOD.
    """
    cat_indexed = cat_df.set_index("article_id")
    fused_id_str = fused_ids.astype(str)

    results: list[TextQueryResult] = []
    for fixture in fixtures:
        # Current hybrid retrieval
        hybrid_hits = hybrid_retriever.search(fixture.query, top_k=10)
        rel_hybrid = sum(1 for r in hybrid_hits if _is_relevant(r, fixture)) / max(
            len(hybrid_hits), 1
        )

        # Degenerate fused retrieval
        query_vec = _encode_text_fused(fixture.query, item_tower, dense_retriever)
        _scores, indices = fused_index.search(query_vec, 10)
        fused_hit_ids = [fused_id_str[idx] for idx in indices[0] if idx >= 0]

        rel_fused = 0.0
        for fid in fused_hit_ids:
            try:
                row = cat_indexed.loc[fid]
            except KeyError:
                continue
            result_dict = {
                "product_type": str(row.get("product_type_name", "")),
                "colour": str(row.get("colour_group_name", "")),
            }
            if _is_relevant(result_dict, fixture):
                rel_fused += 1.0
        rel_fused /= max(len(fused_hit_ids), 1)

        results.append(
            TextQueryResult(
                query_id=fixture.id,
                query=fixture.query,
                cross_store_eligible=is_query_cross_store_eligible(fixture, eligible_types),
                relevance_at_10_hybrid=round(rel_hybrid, 4),
                relevance_at_10_fused=round(rel_fused, 4),
            )
        )
    return results


# -- Task 3: Image query (degenerate, zero text) ------------------------------


def run_image_query_eval(
    cat_df: pd.DataFrame,
    clip_index: faiss.Index,
    clip_ids: np.ndarray,
    fused_index: faiss.Index,
    fused_ids: np.ndarray,
    item_tower: torch.nn.Module,
    eligible_types: frozenset[str],
    *,
    n_sample: int = _N_SAMPLE,
    neighbors_k: int = _NEIGHBORS_K,
    seed: int = _RNG_SEED,
) -> list[ImageQueryResult]:
    """Compare CLIP-512-only vs FUSED-256[CLIP+zeros(384)] in leave-one-out image retrieval.

    The fused image query uses zeros(384) for the text side -- OOD.
    """
    rng = np.random.default_rng(seed)
    clip_id_str = clip_ids.astype(str)
    fused_id_str = fused_ids.astype(str)

    # Both share the same ordering, so position i refers to the same item
    pos_map: dict[str, int] = {aid: i for i, aid in enumerate(clip_id_str)}

    cat_indexed = cat_df.set_index("article_id")
    clip_in_cat = cat_df[cat_df["article_id"].astype(str).isin(pos_map)].copy()
    clip_in_cat["article_id"] = clip_in_cat["article_id"].astype(str)

    stores = clip_in_cat["store"].unique()
    sampled_ids: list[str] = []
    for store in sorted(stores):
        store_items = clip_in_cat[clip_in_cat["store"] == store]["article_id"].values
        n_store = max(1, round(n_sample * len(store_items) / len(clip_in_cat)))
        n_store = min(n_store, len(store_items))
        chosen = rng.choice(store_items, size=n_store, replace=False)
        sampled_ids.extend(chosen.tolist())

    all_ids = clip_in_cat["article_id"].values.copy()
    rng.shuffle(all_ids)
    existing = set(sampled_ids)
    for aid in all_ids:
        if len(sampled_ids) >= n_sample:
            break
        if aid not in existing:
            sampled_ids.append(aid)
            existing.add(aid)
    sampled_ids = sampled_ids[:n_sample]

    results: list[ImageQueryResult] = []

    for aid in sampled_ids:
        pos = pos_map.get(aid)
        if pos is None:
            continue
        try:
            row = cat_indexed.loc[aid]
        except KeyError:
            continue

        query_type = str(row.get("product_type_name", ""))
        query_store = str(row.get("store", ""))
        is_eligible = query_type in eligible_types

        # CLIP-512-only: standard leave-one-out
        clip_vec = clip_index.reconstruct(pos).reshape(1, -1).astype(np.float32)
        _, clip_idxs = clip_index.search(clip_vec, neighbors_k + 1)
        clip_nbrs = [
            clip_id_str[i] for i in clip_idxs[0] if i >= 0 and clip_id_str[i] != aid
        ][:neighbors_k]

        # FUSED[CLIP+zeros]: build fused query on the fly using the same CLIP vec
        img_t = torch.from_numpy(clip_index.reconstruct(pos)).unsqueeze(0)  # [1,512]
        txt_t = torch.zeros(1, 384)  # OOD zero padding
        with torch.no_grad():
            fused_vec = item_tower(img_t, txt_t).numpy().astype(np.float32)

        _, fused_idxs = fused_index.search(fused_vec, neighbors_k + 1)
        fused_nbrs = [
            fused_id_str[i] for i in fused_idxs[0] if i >= 0 and fused_id_str[i] != aid
        ][:neighbors_k]

        def eval_nbrs(nbr_ids: list[str]) -> tuple[float, bool]:
            """Compute (category_consistency, has_cross_store) for neighbor list."""
            type_matches = 0
            cross_found = False
            for nid in nbr_ids:
                try:
                    n_row = cat_indexed.loc[nid]
                except KeyError:
                    continue
                if str(n_row.get("product_type_name", "")) == query_type:
                    type_matches += 1
                if str(n_row.get("store", "")) != query_store:
                    cross_found = True
            consistency = type_matches / max(len(nbr_ids), 1)
            return round(consistency, 4), cross_found

        clip_cons, clip_cross = eval_nbrs(clip_nbrs)
        fused_cons, fused_cross = eval_nbrs(fused_nbrs)

        results.append(
            ImageQueryResult(
                article_id=aid,
                store=query_store,
                product_type=query_type,
                cross_store_eligible=is_eligible,
                cat_consistency_clip=clip_cons,
                cat_consistency_fused_image=fused_cons,
                cross_store_clip=clip_cross,
                cross_store_fused_image=fused_cross,
            )
        )

    return results


# -- Aggregate helpers --------------------------------------------------------


def _agg_item_item(
    results: list[ItemItemResult],
) -> dict[str, Any]:
    """Aggregate item->item metrics across all results and eligible-only subsets."""
    n = len(results)
    if n == 0:
        return {}

    eligible = [r for r in results if r.cross_store_eligible]
    n_elig = len(eligible)

    def safe_mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def safe_rate(bools: list[bool]) -> float:
        return round(sum(bools) / len(bools), 4) if bools else 0.0

    return {
        "n_items": n,
        "n_eligible": n_elig,
        # All-item category consistency
        "cat_consistency_at10_clip_all": safe_mean(
            [r.cat_consistency_clip for r in results]
        ),
        "cat_consistency_at10_sbert_all": safe_mean(
            [r.cat_consistency_sbert for r in results]
        ),
        "cat_consistency_at10_fused_all": safe_mean(
            [r.cat_consistency_fused for r in results]
        ),
        # Eligible-only cross-store rate
        "cross_store_at10_clip_eligible": safe_rate(
            [r.cross_store_clip for r in eligible]
        ),
        "cross_store_at10_sbert_eligible": safe_rate(
            [r.cross_store_sbert for r in eligible]
        ),
        "cross_store_at10_fused_eligible": safe_rate(
            [r.cross_store_fused for r in eligible]
        ),
    }


def _agg_text_query(results: list[TextQueryResult]) -> dict[str, Any]:
    """Aggregate text query relevance metrics."""
    n = len(results)
    if n == 0:
        return {}

    def safe_mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "n_queries": n,
        "relevance_at10_hybrid": safe_mean([r.relevance_at_10_hybrid for r in results]),
        "relevance_at10_fused_degenerate": safe_mean(
            [r.relevance_at_10_fused for r in results]
        ),
        "degenerate_note": (
            "Fused text query uses zeros(512) for image side -- OUT-OF-DISTRIBUTION. "
            "The ItemTower MLP was trained on real CLIP+SBERT item pairs."
        ),
    }


def _agg_image_query(results: list[ImageQueryResult]) -> dict[str, Any]:
    """Aggregate image query metrics for CLIP vs fused-degenerate."""
    n = len(results)
    if n == 0:
        return {}

    eligible = [r for r in results if r.cross_store_eligible]
    n_elig = len(eligible)

    def safe_mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def safe_rate(bools: list[bool]) -> float:
        return round(sum(bools) / len(bools), 4) if bools else 0.0

    return {
        "n_items": n,
        "n_eligible": n_elig,
        "cat_consistency_at10_clip_all": safe_mean(
            [r.cat_consistency_clip for r in results]
        ),
        "cat_consistency_at10_fused_image_all": safe_mean(
            [r.cat_consistency_fused_image for r in results]
        ),
        "cross_store_at10_clip_eligible": safe_rate(
            [r.cross_store_clip for r in eligible]
        ),
        "cross_store_at10_fused_image_eligible": safe_rate(
            [r.cross_store_fused_image for r in eligible]
        ),
        "degenerate_note": (
            "Fused image query uses zeros(384) for text side -- OUT-OF-DISTRIBUTION."
        ),
    }


# -- Report printing ----------------------------------------------------------

_LINE = "-" * 100


def _print_head_to_head_table(
    item_item: dict[str, Any],
    text_query: dict[str, Any],
    image_query: dict[str, Any],
) -> None:
    """Print the comparison table to stdout."""
    print("\n" + "=" * 100)
    print("TWO-TOWER HEAD-TO-HEAD COMPARISON TABLE")
    print("=" * 100)

    print("\n-- TASK 1: Item->Item Similarity (leave-one-out, n={n}, "
          "eligible={n_elig}) --".format(
              n=item_item.get("n_items", 0), n_elig=item_item.get("n_eligible", 0)
          ))
    print(f"  {'Metric':<40} {'CLIP-512':>12} {'SBERT-384':>12} {'FUSED-256':>12} {'Winner':>12}")
    print("  " + "-" * 90)

    c_clip = item_item.get("cat_consistency_at10_clip_all", 0.0)
    c_sbert = item_item.get("cat_consistency_at10_sbert_all", 0.0)
    c_fused = item_item.get("cat_consistency_at10_fused_all", 0.0)
    cat_best = max(c_clip, c_sbert, c_fused)
    cat_winner = (
        "CLIP" if cat_best == c_clip and cat_best > c_sbert and cat_best > c_fused
        else "SBERT" if cat_best == c_sbert and cat_best > c_fused
        else "FUSED" if cat_best == c_fused
        else "tie"
    )
    print(
        f"  {'category_consistency@10 (all)':<40} {c_clip:>12.4f} {c_sbert:>12.4f} "
        f"{c_fused:>12.4f} {cat_winner:>12}"
    )

    x_clip = item_item.get("cross_store_at10_clip_eligible", 0.0)
    x_sbert = item_item.get("cross_store_at10_sbert_eligible", 0.0)
    x_fused = item_item.get("cross_store_at10_fused_eligible", 0.0)
    xs_best = max(x_clip, x_sbert, x_fused)
    xs_winner = (
        "CLIP" if xs_best == x_clip and xs_best > x_sbert and xs_best > x_fused
        else "SBERT" if xs_best == x_sbert and xs_best > x_fused
        else "FUSED" if xs_best == x_fused
        else "tie"
    )
    print(
        f"  {'cross_store@10 (eligible only)':<40} {x_clip:>12.4f} {x_sbert:>12.4f} "
        f"{x_fused:>12.4f} {xs_winner:>12}"
    )

    print("\n-- TASK 2: Text Query Relevance@10 "
          "(n={n} queries) -- [FUSED is DEGENERATE/OOD] --".format(
              n=text_query.get("n_queries", 0)
          ))
    print(f"  {'Metric':<40} {'Hybrid(cur)':>12} {'FUSED-OOD':>12} {'Winner':>12}")
    print("  " + "-" * 76)

    r_hybrid = text_query.get("relevance_at10_hybrid", 0.0)
    r_fused = text_query.get("relevance_at10_fused_degenerate", 0.0)
    t_winner = "Hybrid" if r_hybrid >= r_fused else "FUSED-OOD"
    print(
        f"  {'relevance@10':<40} {r_hybrid:>12.4f} {r_fused:>12.4f} {t_winner:>12}"
    )
    print(f"  NOTE: {text_query.get('degenerate_note', '')}")

    print("\n-- TASK 3: Image Query (leave-one-out, n={n}, "
          "eligible={n_elig}) -- [FUSED is DEGENERATE/OOD] --".format(
              n=image_query.get("n_items", 0), n_elig=image_query.get("n_eligible", 0)
          ))
    print(f"  {'Metric':<40} {'CLIP-512':>12} {'FUSED-OOD':>12} {'Winner':>12}")
    print("  " + "-" * 76)

    ic_clip = image_query.get("cat_consistency_at10_clip_all", 0.0)
    ic_fused = image_query.get("cat_consistency_at10_fused_image_all", 0.0)
    ic_winner = "CLIP" if ic_clip >= ic_fused else "FUSED-OOD"
    print(
        f"  {'category_consistency@10 (all)':<40} {ic_clip:>12.4f} {ic_fused:>12.4f} "
        f"{ic_winner:>12}"
    )

    ix_clip = image_query.get("cross_store_at10_clip_eligible", 0.0)
    ix_fused = image_query.get("cross_store_at10_fused_image_eligible", 0.0)
    ix_winner = "CLIP" if ix_clip >= ix_fused else "FUSED-OOD"
    print(
        f"  {'cross_store@10 (eligible only)':<40} {ix_clip:>12.4f} {ix_fused:>12.4f} "
        f"{ix_winner:>12}"
    )
    print(f"  NOTE: {image_query.get('degenerate_note', '')}")

    print("\n" + "=" * 100)
    print()


def _print_verdict(
    item_item: dict[str, Any],
    text_query: dict[str, Any],
    image_query: dict[str, Any],
) -> None:
    """Print a plain-English verdict on whether the two-tower is worth integrating."""
    print("VERDICT")
    print(_LINE)

    c_clip = item_item.get("cat_consistency_at10_clip_all", 0.0)
    c_sbert = item_item.get("cat_consistency_at10_sbert_all", 0.0)
    c_fused = item_item.get("cat_consistency_at10_fused_all", 0.0)
    x_clip = item_item.get("cross_store_at10_clip_eligible", 0.0)
    x_sbert = item_item.get("cross_store_at10_sbert_eligible", 0.0)
    x_fused = item_item.get("cross_store_at10_fused_eligible", 0.0)

    fused_wins_cat = c_fused > max(c_clip, c_sbert)
    fused_wins_cross = x_fused > max(x_clip, x_sbert)
    delta_cat = c_fused - max(c_clip, c_sbert)
    delta_cross = x_fused - max(x_clip, x_sbert)

    print()
    if fused_wins_cat or fused_wins_cross:
        print("  Two-tower FUSED-256 WINS on at least one item->item metric:")
        if fused_wins_cat:
            best_baseline = "CLIP" if c_clip > c_sbert else "SBERT"
            print(
                f"    category_consistency@10: FUSED={c_fused:.4f} vs best-baseline="
                f"{max(c_clip,c_sbert):.4f} ({best_baseline})  delta={delta_cat:+.4f}"
            )
        if fused_wins_cross:
            best_baseline = "CLIP" if x_clip > x_sbert else "SBERT"
            print(
                f"    cross_store@10(eligible): FUSED={x_fused:.4f} vs best-baseline="
                f"{max(x_clip,x_sbert):.4f} ({best_baseline})  delta={delta_cross:+.4f}"
            )
    else:
        print(
            "  Two-tower FUSED-256 does NOT beat either baseline on item->item metrics."
        )
        print(
            f"    category_consistency@10:   CLIP={c_clip:.4f}  SBERT={c_sbert:.4f}  "
            f"FUSED={c_fused:.4f}"
        )
        print(
            f"    cross_store@10(eligible):  CLIP={x_clip:.4f}  SBERT={x_sbert:.4f}  "
            f"FUSED={x_fused:.4f}"
        )

    r_hybrid = text_query.get("relevance_at10_hybrid", 0.0)
    r_fused = text_query.get("relevance_at10_fused_degenerate", 0.0)
    print()
    print(
        f"  Text query (degenerate/OOD): Hybrid={r_hybrid:.4f}  FUSED-OOD={r_fused:.4f}"
    )
    if r_fused < r_hybrid:
        print(
            "  -> Fused text is worse (expected: MLP was never trained with zero image input)."
        )
    else:
        print("  -> Fused text unexpectedly matches or beats hybrid (inspect results carefully).")

    print()
    print(
        "  Integration recommendation: see task summary above.  Do NOT integrate "
        "without owner sign-off."
    )
    print()


# -- Unit tests ---------------------------------------------------------------


def test_load_item_tower_shape() -> None:
    """ItemTower loaded from checkpoint produces correct output shape and is L2-normed."""
    if not _CHECKPOINT_PATH.exists():
        return  # skip if checkpoint unavailable
    tower = load_item_tower(_CHECKPOINT_PATH)
    img = torch.zeros(4, 512)
    txt = torch.zeros(4, 384)
    with torch.no_grad():
        out = tower(img, txt)
    assert out.shape == (4, 256), f"Expected (4,256), got {out.shape}"
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"Output not L2-normed: {norms}"


def test_build_fused_index_shape() -> None:
    """build_fused_index produces a 256-d FAISS index of the correct size."""
    if not _CHECKPOINT_PATH.exists():
        return  # skip if checkpoint unavailable
    tower = load_item_tower(_CHECKPOINT_PATH)

    # Build tiny stub indices
    n, d_img, d_txt = 10, 512, 384
    rng = np.random.default_rng(42)

    clip_stub = faiss.IndexFlatIP(d_img)
    clip_vecs = rng.random((n, d_img)).astype(np.float32)
    clip_vecs /= np.linalg.norm(clip_vecs, axis=1, keepdims=True)
    clip_stub.add(clip_vecs)

    dense_stub = faiss.IndexFlatIP(d_txt)
    dense_vecs = rng.random((n, d_txt)).astype(np.float32)
    dense_vecs /= np.linalg.norm(dense_vecs, axis=1, keepdims=True)
    dense_stub.add(dense_vecs)

    ids = np.array([str(i) for i in range(n)])
    fused_idx, out_ids = build_fused_index(clip_stub, dense_stub, ids, tower, batch_size=4)

    assert fused_idx.ntotal == n, f"Expected {n} vectors, got {fused_idx.ntotal}"
    assert fused_idx.d == 256, f"Expected dim=256, got {fused_idx.d}"
    assert len(out_ids) == n


# -- Main entry point ---------------------------------------------------------


def main() -> None:
    """Run the full two-tower head-to-head eval."""
    print("\nTwo-Tower Fused Embedding Head-to-Head Eval")
    print(_LINE)

    # -- Verify checkpoint exists --------------------------------------------
    if not _CHECKPOINT_PATH.exists():
        print(f"BLOCKER: checkpoint not found at {_CHECKPOINT_PATH}")
        sys.exit(2)

    # -- Load models and indices ---------------------------------------------
    print("Loading ItemTower from checkpoint...")
    item_tower = load_item_tower(_CHECKPOINT_PATH)
    print(
        f"  ItemTower loaded: {sum(p.numel() for p in item_tower.parameters()):,} params"
    )
    print(
        "  Forward: item_tower(image_emb[B,512], text_emb[B,384]) -> [B,256] L2-normed"
    )

    config = load_config(str(_REPO_ROOT / "config.yaml"))

    print("Loading CLIP and dense indices...")
    clip_index = faiss.read_index(str(_CLIP_DIR / "clip.faiss"))
    clip_ids = np.load(str(_CLIP_DIR / "clip_article_ids.npy"), allow_pickle=True).astype(str)

    dense_index = faiss.read_index(str(_UNIFIED_DIR / "dense.faiss"))
    dense_ids = np.load(
        str(_UNIFIED_DIR / "dense_article_ids.npy"), allow_pickle=True
    ).astype(str)

    # Verify alignment
    n_common = int(np.sum(clip_ids == dense_ids))
    print(
        f"  CLIP index: {clip_index.ntotal:,} vectors (dim={clip_index.d}), "
        f"{len(clip_ids):,} IDs"
    )
    print(
        f"  Dense index: {dense_index.ntotal:,} vectors (dim={dense_index.d}), "
        f"{len(dense_ids):,} IDs"
    )
    print(
        f"  Aligned items (same ID at same position): {n_common:,} / "
        f"{len(clip_ids):,}"
    )
    if n_common != len(clip_ids):
        print("BLOCKER: CLIP and dense IDs are not fully aligned. Cannot build fused index.")
        sys.exit(2)

    cat_df = pd.read_parquet(_UNIFIED_DIR / "catalogue.parquet")
    eligible_types = compute_cross_store_eligible_types(cat_df)
    print(f"  Cross-store-eligible types: {sorted(eligible_types)}")

    print("\nBuilding fused-256 index (CPU, batch=512)...")
    fused_index, fused_ids = build_fused_index(
        clip_index, dense_index, clip_ids, item_tower, batch_size=512
    )
    print(f"  Fused index: {fused_index.ntotal:,} vectors (dim={fused_index.d})")

    # Load hybrid retriever (for text query task)
    print("\nLoading HybridRetriever for text query baseline...")
    dense_retriever = DenseRetriever.load(config, _UNIFIED_DIR)
    sparse_retriever = SparseRetriever.load(config, _UNIFIED_DIR)
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever, cat_df, config)
    fixtures = _load_fixtures(_FIXTURE_PATH)
    print(f"  Loaded {len(fixtures)} text query fixtures")

    # -- Task 1: Item->Item --------------------------------------------------
    print("\nTask 1: Item->Item similarity (n=200, leave-one-out, seed=42)...")
    item_results = run_item_item_eval(
        cat_df, clip_index, clip_ids, dense_index, dense_ids,
        fused_index, fused_ids, eligible_types,
        n_sample=_N_SAMPLE, neighbors_k=_NEIGHBORS_K, seed=_RNG_SEED,
    )
    item_item_agg = _agg_item_item(item_results)
    print(f"  Evaluated {item_item_agg['n_items']} items "
          f"({item_item_agg['n_eligible']} cross-store-eligible)")

    # -- Task 2: Text query --------------------------------------------------
    print("Task 2: Text query relevance (degenerate/OOD fused, 36 queries)...")
    text_results = run_text_query_eval(
        fixtures, hybrid_retriever, item_tower, dense_retriever,
        fused_index, fused_ids, cat_df, eligible_types,
    )
    text_query_agg = _agg_text_query(text_results)

    # -- Task 3: Image query -------------------------------------------------
    print("Task 3: Image query (leave-one-out, n=200, degenerate/OOD fused text)...")
    image_results = run_image_query_eval(
        cat_df, clip_index, clip_ids, fused_index, fused_ids,
        item_tower, eligible_types,
        n_sample=_N_SAMPLE, neighbors_k=_NEIGHBORS_K, seed=_RNG_SEED,
    )
    image_query_agg = _agg_image_query(image_results)
    print(f"  Evaluated {image_query_agg['n_items']} items "
          f"({image_query_agg['n_eligible']} cross-store-eligible)")

    # -- Print table ---------------------------------------------------------
    _print_head_to_head_table(item_item_agg, text_query_agg, image_query_agg)
    _print_verdict(item_item_agg, text_query_agg, image_query_agg)


if __name__ == "__main__":
    main()
