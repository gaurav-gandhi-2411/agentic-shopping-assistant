"""scripts/rerank_gate.py: metrics, pacing/budget/429 policy, resume, live refusal, and a small
end-to-end run on the mock path. Hermetic: every path is tmp_path, no network, no key."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pytest

from scripts import rerank_gate as g
from src.llm.client import LLMCallError, LLMCompletion


def _reply(text: str = "2,1,4,3,6,5,8,7", **kw: Any) -> LLMCompletion:
    return LLMCompletion(text, kw.get("finish", "stop"), 600, 100, 80, 900, "m")


class ScriptedInner:
    """`complete()` that plays a script of replies / errors and records the kwargs it got."""

    def __init__(self, script: list[Any]) -> None:
        self.script, self.calls = list(script), []

    def complete(self, messages: list[dict], **kw: Any) -> LLMCompletion:
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
KW = {"temperature": 0, "max_tokens": 400, "reasoning_effort": "low"}


class Clock:
    def __init__(self) -> None:
        self.t, self.sleeps = 0.0, []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _gate(tmp_path: Path, inner: Any, clock: Clock | None = None, **kw: Any) -> g.GateLLM:
    clock = clock or Clock()
    llm = g.GateLLM(inner, g.JsonlStore(tmp_path / "cache.jsonl"),
                    g.JsonlStore(tmp_path / "ledger.jsonl"), clock=clock.monotonic,
                    sleep=clock.sleep, **kw)
    llm.query_id = "q01"
    return llm


class TestMetrics:
    def test_aggregate_majority_and_median_tie(self) -> None:
        assert g.aggregate_label([2, 2, 0]) == 2.0
        assert g.aggregate_label([0, 1, 2]) == 1.0  # three-way split: median
        assert g.aggregate_label([2, 1]) == 1.5  # two judges disagreeing: median

    def test_ndcg_perfect_and_undefined(self) -> None:
        lab = {"a": 2.0, "b": 1.0, "c": 0.0}
        assert g.query_ndcg(["a", "b", "c"], lab) == pytest.approx(1.0)
        assert g.query_ndcg(["c", "b", "a"], lab) < 1.0
        assert math.isnan(g.query_ndcg(["a"], {"a": 0.0, "b": 0.0}))

    def test_bootstrap_is_seeded_and_zero_diffs_give_a_zero_interval(self) -> None:
        d = [0.1, -0.2, 0.05, 0.0, 0.3]
        assert g.bootstrap_mean_ci(d) == g.bootstrap_mean_ci(d)
        assert g.bootstrap_mean_ci([0.0] * 9) == (0.0, 0.0, 0.0)
        assert math.isnan(g.bootstrap_mean_ci([float("nan")])[0])


class TestGateLLMPolicy:
    def test_throttle_spaces_real_requests_by_eight_seconds(self, tmp_path: Path) -> None:
        clock = Clock()
        llm = _gate(tmp_path, ScriptedInner([_reply(), _reply()]), clock)
        llm.complete(MSGS, **KW)
        llm.query_id = "q02"
        llm.complete(MSGS, **KW)
        assert clock.sleeps == [8.0]

    def test_inner_is_called_with_retry_false(self, tmp_path: Path) -> None:
        inner = ScriptedInner([_reply()])
        _gate(tmp_path, inner).complete(MSGS, **KW)
        assert inner.calls[0]["retry"] is False and inner.calls[0]["max_tokens"] == 400

    def test_cache_resumes_without_a_second_request(self, tmp_path: Path) -> None:
        _gate(tmp_path, ScriptedInner([_reply("9,8,7")])).complete(MSGS, **KW)
        inner = ScriptedInner([])  # would IndexError if asked
        again = _gate(tmp_path, inner)
        out = again.complete(MSGS, **KW)
        assert (out.text, again.last_cached, inner.calls) == ("9,8,7", True, [])

    def test_token_cap_counts_todays_ledger_and_stops_before_the_call(
        self, tmp_path: Path
    ) -> None:
        inner = ScriptedInner([_reply()])
        llm = _gate(tmp_path, inner, token_cap=1000)
        llm.ledger.append({"ts": llm.now().isoformat(), "event": "ok", "total_tokens": 900})
        with pytest.raises(g.GateStop, match="token cap"):
            llm.complete(MSGS, **KW)
        assert inner.calls == []

    def test_yesterdays_tokens_do_not_count(self, tmp_path: Path) -> None:
        llm = _gate(tmp_path, ScriptedInner([_reply()]), token_cap=1000)
        yesterday = (llm.now() - dt.timedelta(days=1)).isoformat()
        llm.ledger.append({"ts": yesterday, "event": "ok", "total_tokens": 99999})
        assert llm.tokens_today() == 0

    def test_two_429s_then_success_third_429_stops(self, tmp_path: Path) -> None:
        e = LLMCallError("rate_limit", 429, retry_after=2.0)
        clock = Clock()
        ok = _gate(tmp_path / "a", ScriptedInner([e, e, _reply()]), clock)
        assert ok.complete(MSGS, **KW).text
        stop = _gate(tmp_path / "b", ScriptedInner([e, e, e, _reply()]))
        with pytest.raises(g.GateStop, match="429"):
            stop.complete(MSGS, **KW)

    def test_a_success_resets_the_429_streak(self, tmp_path: Path) -> None:
        e = LLMCallError("rate_limit", 429)
        llm = _gate(tmp_path, ScriptedInner([e, e, _reply(), e, e, _reply()]))
        llm.complete(MSGS, **KW)
        llm.query_id = "q02"
        assert llm.complete(MSGS, **KW).text

    @pytest.mark.parametrize("kind", ["server", "timeout", "daily_quota", "fatal", "malformed"])
    def test_busy_quota_and_fatal_stop_at_once(self, kind: str, tmp_path: Path) -> None:
        inner = ScriptedInner([LLMCallError(kind, 503), _reply()])
        with pytest.raises(g.GateStop):
            _gate(tmp_path, inner).complete(MSGS, **KW)
        assert len(inner.calls) == 1  # no retry after a busy signal

    def test_failed_calls_are_never_cached(self, tmp_path: Path) -> None:
        llm = _gate(tmp_path, ScriptedInner([LLMCallError("server", 500)]))
        with pytest.raises(g.GateStop):
            llm.complete(MSGS, **KW)
        assert llm.cache.records == []


class TestLiveGuards:
    NOW = dt.datetime(2026, 11, 22, 3, 0, tzinfo=dt.timezone.utc)

    def test_needs_flag_env_and_the_right_utc_day(self) -> None:
        with pytest.raises(SystemExit, match="--live"):
            g.assert_live_allowed(False, None, {}, self.NOW)
        with pytest.raises(SystemExit, match=g.LIVE_ENV_FLAG):
            g.assert_live_allowed(True, None, {}, self.NOW)
        with pytest.raises(SystemExit, match="UTC day"):
            g.assert_live_allowed(True, "2026-11-23", {g.LIVE_ENV_FLAG: "1"}, self.NOW)
        g.assert_live_allowed(True, "2026-11-22", {g.LIVE_ENV_FLAG: "1"}, self.NOW)

    def test_read_named_key_reads_only_that_line(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text('DATABASE_URL=postgres://u:p@h/db\nGROQ_API_KEY="gsk_x"\nOTHER=1\n')
        assert g.read_named_key(env) == "gsk_x"
        assert g.read_named_key(env, "NOPE") is None
        assert g.read_named_key(tmp_path / "missing") is None

    def test_exactly_one_mode(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="exactly one"):
            g.main(["--out", str(tmp_path)])


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    """3 pools (two with 8 candidates, one with 4 = no LLM call), 3 judges, eval-C rows."""
    pools, run = tmp_path / "pools", tmp_path / "run"
    pools.mkdir()
    run.mkdir()
    labels: dict[str, dict[str, int]] = {j: {} for j in g.JUDGES}
    c_rows = []
    for qid, n in (("q01", 8), ("q02", 8), ("q03", 4)):
        items = [{"article_id": f"{qid}x{i}", "display_name": f"Item {i}", "colour": "Black",
                  "price_inr": 900.0} for i in range(1, n + 1)]
        (pools / f"{qid}.json").write_text(json.dumps({"query_id": qid, "text": f"query {qid}",
                                                       "items": items}))
        for it in items:
            for j in g.JUDGES:
                labels[j][f"{qid}:{it['article_id']}"] = 2 if it["article_id"].endswith(("x1", "x2")) else 0
        c_rows.append({"query_id": qid, "rep": 1, "final_ids": [it["article_id"] for it in items[:5]]})
    (pools / "_summary.json").write_text("{}")  # must be ignored
    for j in g.JUDGES:
        (run / f"labels_{j}.json").write_text(json.dumps(labels[j]))
    (run / "variants_partial.json").write_text(json.dumps({"C": c_rows}))
    return pools, run


class TestEndToEndMock:
    def test_mock_run_writes_results_and_applies_the_criteria(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pools, run = _fixture(tmp_path)
        out = tmp_path / "out"
        rc = g.main(["--mock", "--pools", str(pools), "--run-dir", str(run), "--out", str(out)])
        res = json.loads((out / "gate_mock.json").read_text())["summary"]
        assert res["n_llm_queries"] == 2  # q03 has 4 candidates: no LLM call
        assert res["excluded"]["pool_le_top_k_or_empty"] == 1
        assert rc == 0 and res["pass_ndcg"] and res["pass_failure_bar"]
        assert "Reranker gate" in capsys.readouterr().out  # printed the report

    def test_mock_outcomes_and_metric_wiring(self, tmp_path: Path) -> None:
        pools, run = _fixture(tmp_path)
        out = tmp_path / "out"
        g.main(["--mock", "--pools", str(pools), "--run-dir", str(run), "--out", str(out)])
        res = json.loads((out / "gate_mock.json").read_text())["summary"]
        assert res["outcomes"] == {"ok": 2}
        # mock puts x2 then x1 first: both label-2 items are shown, as in eval C's x1,x2 order
        assert res["ndcg5_production"] == pytest.approx(res["ndcg5_eval_c"], abs=0.05)
        assert res["tokens_per_turn_mean"] == 770 and res["length_hits"] == 0
        assert res["diff_prod_minus_c"]["n_boot"] == 10_000 and res["diff_prod_minus_c"]["seed"] == 42
        assert (out / "cache_mock.jsonl").exists() and not (out / "cache_live.jsonl").exists()

    def test_replay_stops_when_the_prompt_is_not_in_the_eval_cache(self, tmp_path: Path) -> None:
        pools, run = _fixture(tmp_path)
        cache = tmp_path / "eval_cache.jsonl"
        cache.write_text("")
        rc = g.main(["--replay-eval-cache", str(cache), "--pools", str(pools),
                     "--run-dir", str(run), "--out", str(tmp_path / "out")])
        assert rc == 3  # GateStop: nothing was scored from a miss

    def test_replay_serves_the_eval_reply_for_a_byte_equal_prompt(self, tmp_path: Path) -> None:
        pools, run = _fixture(tmp_path)
        c = g.load_pools(pools)[0]
        system, user = g.rr.build_prompt(c["text"], c["items"])
        h = g.hashlib.sha256((system + user).encode("utf-8")).hexdigest()[:16]
        cache = tmp_path / "eval_cache.jsonl"
        meta = {"finish_reason": "stop", "latency_ms": 777,
                "usage": {"prompt_tokens": 500, "completion_tokens": 50,
                          "completion_tokens_details": {"reasoning_tokens": 30}}}
        cache.write_text(json.dumps({"key": f"rerank-20b-C-r1|q01|{h}", "text": "", "meta": meta}))
        rep = g.ReplayEvalCache(cache)
        rep.query_id = "q01"
        got = rep.complete([{"role": "system", "content": system}, {"role": "user", "content": user}])
        assert (got.text, got.finish_reason, got.reasoning_tokens, got.latency_ms) == ("", "stop", 30, 777)
