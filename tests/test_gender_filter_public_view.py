"""The synced ``index_group_name`` gender filter survives a catalogue with no H&M-style rows.

Regression (found by CI on the V3b branch, real index): validate-before-trust in
``search_node`` dropped ``index_group_name`` because the graph's facet vocabulary is built from
the catalogue it is given, and the PUBLIC catalogue holds no row whose ``index_group_name`` is
"Menswear"/"Ladieswear" (only the non-public myntra/flipkart/bhasinbrothers rows carry those;
every Shopify store has "N/A"). The fallback branches that rebuild the gender filter from
``index_group_name`` alone then lost gender and could surface the other gender's items.

Hermetic: a tiny synthetic Shopify-style catalogue (index_group_name "N/A" everywhere), real
FAISS + BM25 with the hashing embedder, a stub LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.agents.graph import build_graph
from src.memory.conversation import ConversationMemory
from src.retrieval.snapshot import load_index_parts
from tests._index_fixtures import CONFIG, FakeModel, make_rows, write_index_files

AGENT_CONFIG: dict[str, Any] = {
    **CONFIG,
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
}


class StubLLM:
    def generate(self, prompt: str, system: str | None = None, **_: Any) -> str:
        return "ok"

    def generate_stream(self, prompt: str, system: str | None = None, **_: Any) -> Any:
        yield "ok"

    def chat(self, messages: list[dict], **_: Any) -> str:
        return "ok"

    def chat_stream(self, messages: list[dict], **_: Any) -> Any:
        yield "ok"


def _catalogue() -> pd.DataFrame:
    df = pd.concat(
        [make_rows({"libas": 8}, noun="shirt"), make_rows({"fashor": 8}, noun="dress")],
        ignore_index=True,
    )
    df["gender"] = ["men"] * 8 + ["women"] * 8
    df["index_group_name"] = "N/A"  # what every Shopify store carries
    return df


def _state(query: str, memory: Any) -> dict[str, Any]:
    return {
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


def test_synced_gender_filter_is_kept_when_the_catalogue_has_no_menswear_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_LOOP_FAST_PATH", "true")
    df = _catalogue()
    write_index_files(df, tmp_path / "idx")
    retriever, public_df = load_index_parts(tmp_path / "idx", AGENT_CONFIG, model=FakeModel())
    llm = StubLLM()
    agent = build_graph(retriever, public_df, llm, AGENT_CONFIG)
    memory = ConversationMemory(llm, AGENT_CONFIG)

    result = agent.invoke(_state("shirt men", memory))

    filters = result["filters"]
    assert filters.get("gender") == "men"
    assert filters.get("index_group_name") == "menswear", filters  # was dropped before the fix
    assert result["retrieved_items"], "the search itself must still return items"
    assert all(it.get("gender") == "men" for it in result["retrieved_items"])
