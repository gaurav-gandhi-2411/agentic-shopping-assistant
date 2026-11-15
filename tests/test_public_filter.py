"""Query-time PUBLIC FILTER (Wave 1b V3): non-public stores never reach users.

Small synthetic catalogues with real FAISS + BM25 (tests/_index_fixtures.py). Non-public stores
here are myntra / flipkart (static dumps), bhasinbrothers / priyaasi (feed not verified); public
ones are libas / fashor. Covers: text search, store-filtered search, recall under oversampling,
the snapshot's public DataFrame (item lookup, compare, brand/store counts), outfit composition
and the CLIP image-anchor path. INCLUDE_NON_PUBLIC_STORES=1 restores the full index.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pandas as pd
import pytest

from src.agents.outfit.composer import compose_outfit_variants
from src.agents.outfit.image_anchor import find_anchor_from_image
from src.agents.tools import compare_items
from src.catalogue.entity_resolution import build_brand_stores_map
from src.config.store_registry import INCLUDE_NON_PUBLIC_ENV
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.snapshot import load_index_parts, public_catalogue
from tests._index_fixtures import CONFIG, FakeModel, make_rows, write_index_files

NON_PUBLIC = ("myntra", "flipkart", "bhasinbrothers", "priyaasi")
PUBLIC = ("libas", "fashor")


@pytest.fixture(autouse=True)
def _default_public_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(INCLUDE_NON_PUBLIC_ENV, raising=False)


@pytest.fixture()
def index_dir(tmp_path: Path) -> Path:
    # Non-public stores are 6x larger than public ones and mention the query word in every
    # row: without a pushdown mask they would fill the whole fetch window (top_k=10 -> 20 hits).
    df = make_rows(
        {
            "myntra": 30,
            "flipkart": 30,
            "bhasinbrothers": 30,
            "priyaasi": 30,
            "libas": 6,
            "fashor": 6,
        }
    )
    write_index_files(df, tmp_path / "idx")
    return tmp_path / "idx"


def _stores(results: list[dict]) -> set[str]:
    return {str(r["store"]).lower() for r in results}


def test_search_over_an_unfiltered_catalogue_never_returns_non_public_items(
    index_dir: Path,
) -> None:
    # Retriever built over the FULL frame (as an eval/dev script would): the mask inside
    # search() is the only guard here.
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.sparse_search import SparseRetriever

    df = pd.read_parquet(index_dir / "catalogue.parquet")
    dense = DenseRetriever.load(CONFIG, index_dir, model=FakeModel())
    retr = HybridRetriever(dense, SparseRetriever.load(CONFIG, index_dir), df, CONFIG)
    results = retr.search("kurta myntra flipkart", top_k=10)
    assert results, "public items must still be found"
    assert _stores(results) <= set(PUBLIC)


def test_search_is_not_starved_of_public_items_by_the_hidden_majority(index_dir: Path) -> None:
    retr, df = load_index_parts(index_dir, CONFIG, model=FakeModel())
    # 12 public items exist; the query favours the hidden stores' tokens.
    results = retr.search("kurta myntra priyaasi", top_k=10)
    assert len(results) == 10
    assert _stores(results) <= set(PUBLIC)
    assert len(df) == 12 and set(df["store"]) == set(PUBLIC)


def test_store_filter_on_a_non_public_store_returns_nothing(index_dir: Path) -> None:
    retr, _ = load_index_parts(index_dir, CONFIG, model=FakeModel())
    for slug in NON_PUBLIC:
        assert retr.search("kurta", top_k=10, filters={"store": slug}) == []
    assert _stores(retr.search("kurta", top_k=10, filters={"store": "libas"})) == {"libas"}


def test_unknown_store_in_the_index_fails_closed(tmp_path: Path) -> None:
    df = make_rows({"libas": 4, "brand-new-store": 4})
    write_index_files(df, tmp_path / "i")
    retr, public_df = load_index_parts(tmp_path / "i", CONFIG, model=FakeModel())
    assert set(public_df["store"]) == {"libas"}
    assert _stores(retr.search("kurta", top_k=10)) == {"libas"}


def test_public_catalogue_drops_non_public_rows_and_is_identity_when_nothing_hidden() -> None:
    df = make_rows({"libas": 3, "myntra": 2, "priyaasi": 1})
    out = public_catalogue(df)
    assert set(out["store"]) == {"libas"} and len(out) == 3
    only_public = make_rows({"libas": 3})
    assert public_catalogue(only_public) is only_public
    no_store_col = only_public.drop(columns=["store"])
    assert public_catalogue(no_store_col) is no_store_col  # legacy per-brand index


def test_include_non_public_switch_restores_everything(
    index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(INCLUDE_NON_PUBLIC_ENV, "1")
    retr, df = load_index_parts(index_dir, CONFIG, model=FakeModel())
    assert len(df) == 132
    assert _stores(retr.search("kurta myntra", top_k=10)) & set(NON_PUBLIC)


def test_item_lookup_compare_and_store_counts_see_only_public_rows(index_dir: Path) -> None:
    retr, df = load_index_parts(index_dir, CONFIG, model=FakeModel())
    out = compare_items(["myntra-0", "libas-0", "priyaasi-3"], df)
    assert [i["article_id"] for i in out["items"]] == ["libas-0"]
    assert "myntra-0" not in set(retr.catalogue_df.index)
    stores_seen = {s for stores in build_brand_stores_map(df).values() for s in stores}
    assert stores_seen <= set(PUBLIC)


def test_outfit_composition_never_picks_non_public_seed_or_complements(index_dir: Path) -> None:
    retr, df = load_index_parts(index_dir, CONFIG, model=FakeModel())
    hidden = compose_outfit_variants(df, retr, seed_article_id="myntra-0", occasion_slug="casual")
    assert all(not v.get("seed_item") for v in hidden)
    shown = compose_outfit_variants(df, retr, seed_article_id="libas-0", occasion_slug="casual")
    assert any(look.get("seed_item") for look in shown)
    for look in shown:
        items = [look.get("seed_item")] + list(look.get("complements") or [])
        assert {str(i["store"]).lower() for i in items if i} <= set(PUBLIC)


def _clip_index(ids: list[str]) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    idx = faiss.IndexFlatIP(4)
    vecs = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(ids), 1))
    vecs[:, 1] = np.linspace(0.0, 0.5, len(ids))  # ids earlier in the list rank higher
    idx.add(vecs / np.linalg.norm(vecs, axis=1, keepdims=True))
    return idx, np.array(ids)


def _encoder() -> MagicMock:
    enc = MagicMock()
    enc.encode_image.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return enc


def test_image_anchor_with_snapshot_allow_list_skips_hidden_nearest_neighbours(
    tmp_path: Path,
) -> None:
    # The 30 nearest CLIP neighbours are non-public; the 5 public ones rank last. Oversampling
    # must still surface the public ones instead of returning [].
    hidden = [f"myntra-{i}" for i in range(30)]
    public_ids = [f"libas-{i}" for i in range(5)]
    entry = _clip_index(hidden + public_ids)
    with patch("src.agents.outfit.image_anchor.get_clip_encoder", return_value=_encoder()):
        got = find_anchor_from_image(
            object(),
            brand="unified",
            top_k=3,
            clip_entry=entry,
            allowed_ids=set(public_ids),
            _catalogue_base=tmp_path,  # no catalogue.parquet there -> only allowed_ids filters
        )
    assert got and set(got) <= set(public_ids) and len(got) == 3


def test_image_anchor_legacy_path_drops_non_public_and_unmapped_ids(tmp_path: Path) -> None:
    ids = ["a-myntra", "b-priyaasi", "c-libas", "d-ghost"]
    entry = _clip_index(ids)
    base = tmp_path / "cat"
    (base / "unified").mkdir(parents=True)
    pd.DataFrame({"article_id": ids[:3], "store": ["myntra", "priyaasi", "libas"]}).to_parquet(
        base / "unified" / "catalogue.parquet"
    )
    with patch("src.agents.outfit.image_anchor.get_clip_encoder", return_value=_encoder()):
        got = find_anchor_from_image(
            object(), brand="unified", top_k=4, clip_entry=entry, _catalogue_base=base
        )
    assert got == ["c-libas"]  # d-ghost has no catalogue row -> fail closed
