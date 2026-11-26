"""ID-format compliance metric, raw-output capture and the blind dump (V5 live-eval tooling).

All offline: synthetic reports and a fake LLM. No network, no Groq.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from evals.grounding import compliance
from evals.grounding import run as harness
from tests import _grounding_v5_corpus as corpus

ITEMS = corpus.ITEMS
_SPEC = importlib.util.spec_from_file_location(
    "grounding_blind_dump",
    Path(__file__).resolve().parents[1] / "scripts" / "grounding_blind_dump.py",
)
assert _SPEC and _SPEC.loader
dump = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dump)


# ------------------------------------------------------------------ compliance


def test_tokens_only_reply_is_fully_compliant() -> None:
    assert compliance.count_mentions("Try [item:1001] or [item:1002] today.", ITEMS) == (2, 0)


def test_plain_name_mentions_are_counted_and_names_inside_tags_are_not() -> None:
    raw = "The Riviera Cotton Sundress is nice and [item:1002] pairs well with it."
    assert compliance.count_mentions(raw, ITEMS) == (1, 1)
    assert compliance.count_mentions("See [item:1001] now.", ITEMS) == (1, 0)


def test_reply_without_any_mention_counts_nothing() -> None:
    assert compliance.count_mentions("Happy to narrow this down by colour.", ITEMS) == (0, 0)


def test_price_tags_are_not_item_mentions() -> None:
    assert compliance.count_mentions("[item:1001] at [price:1001].", ITEMS) == (1, 0)


def test_summarise_rate_per_path_and_runs_without_mentions() -> None:
    runs = [
        {"path": "ws", "id_tokens": 3, "plain_mentions": 1},
        {"path": "ws", "id_tokens": 0, "plain_mentions": 0},
        {"path": "rest", "id_tokens": 1, "plain_mentions": 3},
    ]
    s = compliance.summarise(runs)
    assert s["all"] == {
        "runs": 3,
        "runs_with_mentions": 2,
        "id_tokens": 4,
        "plain_mentions": 4,
        "rate": 0.5,
    }
    assert s["ws"]["rate"] == 0.75 and s["rest"]["rate"] == 0.25 and s["ws"]["runs"] == 2


def test_summarise_rate_is_none_when_nothing_was_mentioned() -> None:
    assert compliance.summarise([{"path": "ws"}])["all"]["rate"] is None


# ------------------------------------------------------------------ raw capture


class _FakeInner:
    def generate(self, prompt: str, **kw: Any) -> str:
        return "raw [item:1] text"

    def generate_stream(self, prompt: str, **kw: Any) -> Iterator[str]:
        yield from ("raw ", "[item:2] ", "stream")


def test_counting_llm_records_raw_respond_outputs_only() -> None:
    llm = harness.CountingLLM(_FakeInner(), cap=10)
    llm.generate("You are a fashion shopping assistant. Write now.")
    llm.generate("router prompt without the marker")
    assert (
        "".join(llm.generate_stream("You are a fashion shopping assistant."))
        == "raw [item:2] stream"
    )
    list(llm.generate_stream("some other prompt"))
    assert llm.raw_log == ["raw [item:1] text", "raw [item:2] stream"]
    assert llm.n_calls == 4


def test_counting_llm_records_a_partially_consumed_stream_on_close() -> None:
    llm = harness.CountingLLM(_FakeInner(), cap=3)
    gen = llm.generate_stream("You are a fashion shopping assistant.")
    next(gen)
    gen.close()
    assert llm.raw_log == ["raw "]


def test_harness_report_carries_compliance_prompt_raw_and_the_arm_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GROUNDED_RENDERING", raising=False)
    harness.main(
        [
            "--llm",
            "mock",
            "--agent",
            "stub",
            "--mock-mode",
            "grounded",
            "--limit",
            "3",
            "--out-dir",
            str(tmp_path),
        ]
    )
    rep = json.loads(next(tmp_path.glob("grounding_eval_*.json")).read_text(encoding="utf-8"))
    assert rep["grounded_rendering"] == "on"
    assert rep["id_format_compliance"]["all"]["runs"] == 6
    run0 = rep["runs"][0]
    assert run0["prompt"] and isinstance(run0["raw"], list) and run0["raw"]
    assert {"id_tokens", "plain_mentions"} <= set(run0)


# ------------------------------------------------------------------ blind dump


def _report(arm: str, texts: dict[tuple[str, str], str], phash: str = "h1") -> dict[str, Any]:
    return {
        "grounded_rendering": arm,
        "prompt_set_hash": phash,
        "runs": [
            {
                "id": i,
                "path": p,
                "http_ok": True,
                "text": t,
                "prompt": f"prompt {i}",
                "shown": [{"id": "1", "name": "Test Jacket", "price_inr": 1999}],
            }
            for (i, p), t in texts.items()
        ],
    }


def _pair(n: int = 30) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = [(f"r{k:03d}", "ws") for k in range(n)]
    return (
        _report("on", {k: f"ON reply {k[0]}" for k in keys}),
        _report("off", {k: f"OFF reply {k[0]}" for k in keys}),
    )


def test_dump_picks_n_samples_deterministically_and_keys_match() -> None:
    on, off = _pair()
    samples, key = dump.build(on, off, n=20, seed=42)
    again, key2 = dump.build(on, off, n=20, seed=42)
    assert len(samples) == 20 and samples == again and key == key2
    for s in samples:
        arms = key[s["sample"]]
        assert {arms["A"], arms["B"]} == {"on", "off"}
        assert s[f"reply_{'A'}"].startswith(arms["A"].upper())
        assert s["reply_B"].startswith(arms["B"].upper())


def test_dump_randomises_which_arm_is_a() -> None:
    on, off = _pair()
    _, key = dump.build(on, off, n=20, seed=42)
    assert {k["A"] for k in key.values()} == {"on", "off"}


def test_dump_judge_file_has_no_arm_labels() -> None:
    on, off = _pair()
    samples, _ = dump.build(on, off, n=5)
    md = dump.render_markdown(samples)
    assert "Reply A" in md and "Reply B" in md and "Prompt: prompt" in md
    assert '"on"' not in md and "grounded_rendering" not in md


@pytest.mark.parametrize("bad", ["swapped", "hash", "empty"])
def test_dump_rejects_unusable_inputs(bad: str) -> None:
    on, off = _pair(5)
    if bad == "swapped":
        on, off = off, on
    elif bad == "hash":
        off["prompt_set_hash"] = "other"
    else:
        off["runs"] = []
    with pytest.raises(ValueError):
        dump.build(on, off)


def test_dump_skips_pairs_where_an_arm_has_no_text() -> None:
    on, off = _pair(4)
    off["runs"][0]["text"] = "  "
    samples, _ = dump.build(on, off, n=10)
    assert len(samples) == 3 and all(s["case"] != "r000" for s in samples)


def test_dump_main_writes_the_three_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    on, off = _pair(25)
    (tmp_path / "on.json").write_text(json.dumps(on), encoding="utf-8")
    (tmp_path / "off.json").write_text(json.dumps(off), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "x",
            "--on",
            str(tmp_path / "on.json"),
            "--off",
            str(tmp_path / "off.json"),
            "--out-dir",
            str(out),
        ],
    )
    assert dump.main() == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "blind_dump.json",
        "blind_dump.md",
        "blind_key.json",
    ]
    assert len(json.loads((out / "blind_key.json").read_text(encoding="utf-8"))) == 20
