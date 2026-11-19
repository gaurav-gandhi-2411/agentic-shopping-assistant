"""IndexSnapshot + api.deps: one immutable reference, taken once per turn (Wave 1b V3(b)).

build_graph is replaced by a recorder (compiling the real LangGraph needs an LLM and ~seconds);
what is under test is WHICH retriever/DataFrame each graph is compiled from and how the single
reference is published, read and swapped.
"""

from __future__ import annotations

import dataclasses
import gc
import weakref
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import api.deps as deps
from src.catalogue.entity_resolution import get_cached_brand_stores_map
from src.config.store_registry import INCLUDE_NON_PUBLIC_ENV
from src.retrieval.snapshot import IndexSnapshot, load_index_parts
from tests._index_fixtures import CONFIG, FakeModel, make_rows, write_index_files


class _Graph:
    """Stand-in for a compiled graph; remembers what it was compiled around."""

    def __init__(self, retriever: Any, df: pd.DataFrame, streaming: bool) -> None:
        self.retriever, self.df, self.streaming = retriever, df, streaming


@pytest.fixture()
def fresh_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(INCLUDE_NON_PUBLIC_ENV, raising=False)
    monkeypatch.setattr(deps, "_snapshot", None)
    monkeypatch.setattr(deps, "_retriever", None)
    monkeypatch.setattr(deps, "_catalogue_df", None)
    monkeypatch.setattr(deps, "_agent_sync", None)
    monkeypatch.setattr(deps, "_agent_streaming", None)

    def fake_build_graph(
        retriever: Any, df: pd.DataFrame, llm: Any, config: dict, streaming_mode: bool = False
    ) -> _Graph:
        return _Graph(retriever, df, streaming_mode)

    monkeypatch.setattr("src.agents.graph.build_graph", fake_build_graph)


def _loaded(tmp_path: Path, name: str, stores: dict[str, int]) -> tuple[Any, pd.DataFrame]:
    write_index_files(make_rows(stores), tmp_path / name)
    return load_index_parts(tmp_path / name, CONFIG, model=FakeModel())


def test_make_snapshot_compiles_both_graphs_from_its_own_retriever_and_public_frame(
    fresh_deps: None, tmp_path: Path
) -> None:
    retr, df = _loaded(tmp_path, "a", {"libas": 3, "myntra": 2})
    full = make_rows({"libas": 3, "myntra": 2})
    snap = deps.make_snapshot(retr, full, llm=object(), config=CONFIG)
    assert set(snap.catalogue_df["store"]) == {"libas"}  # public view
    for g, streaming in ((snap.agent_sync, False), (snap.agent_streaming, True)):
        assert g.retriever is retr and g.df is snap.catalogue_df and g.streaming is streaming
    assert snap.build_id == "legacy" and snap.manifest is None and snap.source == "flat"
    assert snap.public_article_ids == frozenset({"libas-0", "libas-1", "libas-2"})
    assert len(df) == 3


def test_install_swaps_the_single_reference_and_returns_the_old_one(
    fresh_deps: None, tmp_path: Path
) -> None:
    r1, d1 = _loaded(tmp_path, "a", {"libas": 2})
    r2, d2 = _loaded(tmp_path, "b", {"fashor": 2})
    s1 = deps.make_snapshot(r1, d1, None, CONFIG)
    s2 = deps.make_snapshot(r2, d2, None, CONFIG)
    assert deps.install_snapshot(s1) is None
    assert deps.get_snapshot() is s1 and deps.get_retriever() is r1
    assert deps.get_catalogue_df() is s1.catalogue_df
    assert deps.install_snapshot(s2) is s1
    assert deps.get_snapshot() is s2 and deps.get_retriever() is r2


def test_agent_factory_is_bound_to_the_snapshot_current_when_the_turn_started(
    fresh_deps: None, tmp_path: Path
) -> None:
    r1, d1 = _loaded(tmp_path, "a", {"libas": 2})
    r2, d2 = _loaded(tmp_path, "b", {"fashor": 2})
    s1 = deps.make_snapshot(r1, d1, None, CONFIG)
    s2 = deps.make_snapshot(r2, d2, None, CONFIG)
    deps.install_snapshot(s1)
    factory = deps.get_agent_factory()  # a turn starts here
    deps.install_snapshot(s2)  # reload swaps mid-turn
    assert factory(None, streaming=True) is s1.agent_streaming  # in-flight turn: old build
    assert factory(None, streaming=False) is s1.agent_sync
    assert deps.get_agent_factory()(None, streaming=False) is s2.agent_sync  # next turn: new


def test_legacy_fallbacks_apply_only_when_no_snapshot_is_installed(
    fresh_deps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel_r, sentinel_df = object(), pd.DataFrame({"article_id": ["x"]})
    monkeypatch.setattr(deps, "_retriever", sentinel_r)
    monkeypatch.setattr(deps, "_catalogue_df", sentinel_df)
    assert deps.get_retriever() is sentinel_r and deps.get_catalogue_df() is sentinel_df
    with pytest.raises(AssertionError):
        deps.get_snapshot()
    assert deps.get_snapshot_or_none() is None


def test_init_installs_a_snapshot_and_does_not_pin_the_index_in_legacy_globals(
    fresh_deps: None, tmp_path: Path
) -> None:
    retr, df = _loaded(tmp_path, "a", {"libas": 2})
    deps._init(retr, df, llm=object(), config=CONFIG)
    assert deps.get_snapshot().retriever is retr
    assert deps._retriever is None and deps._catalogue_df is None  # would pin the old index


def test_snapshot_is_frozen_and_released_when_unreferenced(
    fresh_deps: None, tmp_path: Path
) -> None:
    retr, df = _loaded(tmp_path, "a", {"libas": 2})
    snap = deps.make_snapshot(retr, df, None, CONFIG)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(snap, "build_id", "x")
    ref = weakref.ref(snap)
    del snap
    gc.collect()
    assert ref() is None


def test_brand_store_cache_follows_the_catalogue_object_not_its_length() -> None:
    """A reload can swap in a different catalogue of the SAME length: the cache must not
    serve the previous build's map."""
    a = pd.DataFrame(
        {
            "article_id": ["1", "2"],
            "store": ["libas", "libas"],
            "prod_name": ["Fabindia Kurta", "Fabindia Top"],
        }
    )
    b = pd.DataFrame(
        {
            "article_id": ["1", "2"],
            "store": ["fashor", "fashor"],
            "prod_name": ["Zara Kurta", "Zara Top"],
        }
    )
    map_a = get_cached_brand_stores_map(a)
    assert get_cached_brand_stores_map(a) is map_a  # same object: cache hit
    map_b = get_cached_brand_stores_map(b)
    assert map_b is not map_a
    assert map_b != map_a


def test_snapshot_dataclass_fields_cover_everything_a_request_reads() -> None:
    names = {f.name for f in dataclasses.fields(IndexSnapshot)}
    assert {
        "retriever",
        "catalogue_df",
        "manifest",
        "build_id",
        "clip",
        "agent_sync",
        "agent_streaming",
    } <= names
