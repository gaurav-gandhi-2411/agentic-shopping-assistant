"""Outfit latency budget: the viability probe is reused as variant 0 (no recomposition).

Hermetic (stubbed compose_outfit; no index, no LLM).  Selection parity against the real index
is covered separately by evals/fixtures/outfit_compose_snapshot.json
(tests/test_outfit_snapshot_parity.py).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.agents import graph as graph_mod
from src.agents.outfit import composer
from src.agents.outfit.composer import compose_outfit_variants, probe_reusable_as_base
from src.memory.conversation import ConversationMemory


def _look(seed: str, comp_ids: list[str]) -> dict:
    return {
        "look_id": f"look-{seed}-{'-'.join(comp_ids)}",
        "seed_item": {"article_id": seed, "display_name": seed, "colour": "red"},
        "complements": [{"article_id": c, "display_name": c, "colour": "blue"} for c in comp_ids],
        "outfit_rationale": "r",
        "empty_slots": [],
        "occasion": "sangeet",
        "gender": "women",
        "budget_total_inr": 100.0,
    }


class _StubCompose:
    """compose_outfit stand-in: records calls; excluded ids are never returned."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._pool = ["c1", "c2", "c3", "c4", "c5", "c6"]

    def __call__(self, catalogue_df: Any, retriever: Any, **kw: Any) -> dict:
        self.calls.append(kw)
        excluded = set(kw.get("exclude_ids") or set())
        picks = [c for c in self._pool if c not in excluded][:2]
        return _look("seed", picks)


@pytest.fixture()
def stub(monkeypatch: pytest.MonkeyPatch) -> _StubCompose:
    s = _StubCompose()
    monkeypatch.setattr(composer, "compose_outfit", s)
    return s


def _ids(variants: list[dict]) -> list[list[str]]:
    return [[c["article_id"] for c in v["complements"]] for v in variants]


def test_variants_without_base_look_composes_base_itself(stub: _StubCompose) -> None:
    variants = compose_outfit_variants(None, None, occasion_slug="sangeet")  # type: ignore[arg-type]
    assert len(stub.calls) == 3  # base + colour story + formality shift
    assert [v["variant_label"] for v in variants][0] == "Base"


def test_base_look_is_reused_not_recomposed(stub: _StubCompose) -> None:
    probe = _look("seed", ["c1", "c2"])
    variants = compose_outfit_variants(
        None, None, occasion_slug="sangeet", base_look=probe  # type: ignore[arg-type]
    )
    assert len(stub.calls) == 2  # only the two biased variants; no base recomposition
    assert variants[0] is probe
    assert probe["variant_label"] == "Base"
    # Biased variants still exclude the reused base's own complements (S6 distinctness).
    assert all(not ({"c1", "c2"} & set(ids)) for ids in _ids(variants[1:]))


def test_reused_base_gives_same_selection_as_recomposed(stub: _StubCompose) -> None:
    recomposed = compose_outfit_variants(None, None, occasion_slug="sangeet")  # type: ignore[arg-type]
    probe = _look("seed", ["c1", "c2"])
    reused = compose_outfit_variants(
        None, None, occasion_slug="sangeet", base_look=probe  # type: ignore[arg-type]
    )
    assert _ids(reused) == _ids(recomposed)
    assert [v["variant_label"] for v in reused] == [v["variant_label"] for v in recomposed]


def test_seedless_base_look_returns_single_variant(stub: _StubCompose) -> None:
    probe = {"seed_item": None, "complements": []}
    variants = compose_outfit_variants(
        None, None, occasion_slug="sangeet", base_look=probe  # type: ignore[arg-type]
    )
    assert variants == [probe]
    assert stub.calls == []


@pytest.mark.parametrize(
    ("gender", "brand_default", "expected"),
    [
        ("women", "women", True),
        ("men", "women", True),
        ("men", "men", True),  # gender explicit: brand default is never read
        ("unisex", "women", True),
        ("unisex", None, True),  # None resolves to "women" exactly like compose_outfit
        ("unisex", "", True),
        ("unisex", "men", False),  # probe would resolve "women", variants "men"
        ("unisex", "mixed", False),
    ],
)
def test_probe_reusable_as_base(gender: str, brand_default: str | None, expected: bool) -> None:
    assert probe_reusable_as_base(gender, brand_default) is expected


# Minimal facet columns: build_graph derives its router vocabulary from these at construction.
_CATALOGUE = pd.DataFrame(
    {"article_id": ["a"], "colour_group_name": ["Red"], "product_type_name": ["Kurta"]}
)


class _NoLLM:
    """Forces generate_rationales onto its template fallback (it catches Exception)."""

    def generate(self, *a: Any, **k: Any) -> str:
        raise RuntimeError("no LLM in this test")

    def generate_stream(self, *a: Any, **k: Any) -> Any:
        raise RuntimeError("no LLM in this test")


_CFG: dict = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
    "retrieval": {"final_k": 10, "top_k": 50, "rrf_k": 60},
}


def test_outfit_node_composes_three_times_not_four(
    stub: _StubCompose, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Graph wiring: one probe + two biased variants = 3 compose passes (was 4)."""
    probe_calls: list[dict] = []

    def probe_stub(**kw: Any) -> dict:
        probe_calls.append(kw)
        return _look("seed", ["c1", "c2"])

    monkeypatch.setattr(graph_mod, "compose_outfit_tool", probe_stub)
    llm = _NoLLM()
    agent = graph_mod.build_graph(  # type: ignore[arg-type]
        None, _CATALOGUE, llm, _CFG, streaming_mode=False
    )
    memory = ConversationMemory(llm, _CFG)  # type: ignore[arg-type]
    query = "sangeet outfit under ₹3000 for women"
    state = {
        "messages": [{"role": "user", "content": query}],
        "user_query": query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": [],
        "filters": {},
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
        "out_of_catalogue": False,
        "excluded_colours": None,
        "anchor_article_id": None,
        "outfit_rationale": None,
        "outfit_variants": None,
        "_memory": memory,
    }
    result = agent.invoke(state)
    assert len(probe_calls) == 1
    assert len(stub.calls) == 2, "base must come from the probe, not a second compose pass"
    variants = result["outfit_variants"]
    assert variants[0]["variant_label"] == "Base"
    assert [c["article_id"] for c in variants[0]["complements"]] == ["c1", "c2"]
