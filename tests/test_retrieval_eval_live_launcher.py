"""Live launcher (key by name only) and the busy-signal stop. Hermetic: tmp files, fake env."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.retrieval import groq_judge as gj
from evals.retrieval import live_run as lr
from evals.retrieval import run

ENV_TEXT = (
    "# comment\n"
    "GROQ_API_KEY=gsk_fake_value\n"
    "GEMINI_API_KEY=g-secret\n"
    "export DATABASE_URL=postgres://u:p@h/db\n"
    "SOME_OTHER=1\n"
)


def test_read_key_returns_only_the_wanted_value_and_other_names(tmp_path: Path) -> None:
    f = tmp_path / "e"
    f.write_text(ENV_TEXT, encoding="utf-8")
    value, others = lr.read_key(f)
    assert value == "gsk_fake_value"
    assert set(others) == {"GEMINI_API_KEY", "DATABASE_URL", "SOME_OTHER"}


def test_prepare_env_pops_other_secrets_and_sets_flags(tmp_path: Path) -> None:
    f = tmp_path / "e"
    f.write_text(ENV_TEXT, encoding="utf-8")
    env = {"GEMINI_API_KEY": "x", "DATABASE_URL": "y", "OPENROUTER_API_KEY": "z", "PATH": "p"}
    removed = lr.prepare_env(f, env)
    assert env["GROQ_API_KEY"] == "gsk_fake_value" and env["STAGE2_LIVE_OK"] == "1"
    assert set(removed) == {"GEMINI_API_KEY", "DATABASE_URL", "OPENROUTER_API_KEY"}
    assert env["PATH"] == "p" and "DATABASE_URL" not in env  # unrelated vars untouched


def test_missing_key_fails_closed(tmp_path: Path) -> None:
    f = tmp_path / "e"
    f.write_text("GEMINI_API_KEY=a\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        lr.prepare_env(f, {})


def _runner(tmp_path: Path, transport: object) -> gj.GroqRunner:
    return gj.GroqRunner(
        transport,  # type: ignore[arg-type]
        gj.Ledger(tmp_path / "l.jsonl"),
        gj.ResultCache(tmp_path / "c.jsonl"),
        min_interval_s=0.0,
        sleep=lambda s: None,
    )


@pytest.mark.parametrize("status", [498, 500, 503])
def test_busy_status_stops_at_once_and_is_ledgered(tmp_path: Path, status: int) -> None:
    calls = []

    def transport(payload: dict) -> gj.HttpResult:
        calls.append(1)
        return gj.HttpResult(status, {}, {})

    spec = gj.ModelSpec("j", "openai/gpt-oss-20b", {}, 100, 10_000)
    with pytest.raises(gj.BusyStopError):
        _runner(tmp_path, transport).call(spec, "q1", "s", "u", "h")
    assert len(calls) == 1  # no retry
    assert '"event": "busy"' in (tmp_path / "l.jsonl").read_text()


def test_transport_exception_is_a_busy_signal(tmp_path: Path) -> None:
    def transport(payload: dict) -> gj.HttpResult:
        raise TimeoutError("read timeout")

    spec = gj.ModelSpec("j", "openai/gpt-oss-20b", {}, 100, 10_000)
    with pytest.raises(gj.BusyStopError, match="TimeoutError"):
        _runner(tmp_path, transport).call(spec, "q1", "s", "u", "h")


def test_busy_stop_aborts_the_variants_stage_not_a_row(tmp_path: Path) -> None:
    from evals.retrieval.common import CANDIDATE_DIR, load_candidates, load_cases

    case = next(c for c in load_cases() if c["id"] == "q02")
    pool = load_candidates("q02", CANDIDATE_DIR)["items"]
    llm = run.GroqLLM(
        _runner(tmp_path, lambda p: gj.HttpResult(503, {}, {})), run.reranker_spec(100_000), "A2"
    )
    with pytest.raises(gj.BusyStopError):
        run._run_arm("A2", case, pool, llm, 1)
