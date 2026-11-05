"""Hermetic tests for scripts/ops/canary.py (loopback server + pure functions; no network)."""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import struct
import sys
import threading
import time
from hashlib import sha1
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

# CANARY_UNDER_TEST lets the mutation check (planted bugs) point these same tests at a mutant copy.
_SPEC = importlib.util.spec_from_file_location(
    "canary",
    os.environ.get("CANARY_UNDER_TEST")
    or Path(__file__).resolve().parents[1] / "scripts" / "ops" / "canary.py",
)
canary = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["canary"] = canary  # dataclasses needs the module registered
_SPEC.loader.exec_module(canary)

GOOD_ITEM = {"article_id": "a1", "price_inr": 1999.0, "display_name": "Kurta"}


def good_frames() -> list[Any]:
    item2 = {"article_id": "a2", "price_inr": 999, "display_name": "Dupatta"}
    return [
        {"type": "session", "conversation_id": "c"},
        {"type": "items", "items": [GOOD_ITEM, item2]},
        {"type": "token", "text": "Here is "},
        {"type": "token", "text": "your look."},
        {"type": "done", "final_state": {"outfit_variants": [
            {"items": [dict(GOOD_ITEM), dict(item2)]}]}},
    ]


# ------------------------------------------------------------------ loopback server


def _ws_frame(text: str) -> bytes:
    data = text.encode()
    if len(data) < 126:
        return bytes([0x81, len(data)]) + data
    return bytes([0x81, 126]) + struct.pack(">H", len(data)) + data


class _Server:
    """Loopback server speaking just enough of the StyleMaitri API + RFC 6455."""

    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self.seen_user_messages: list[dict[str, Any]] = []
        self.client_masked: list[bool] = []
        self.pongs: list[tuple[int, bytes]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _json(self, status: int, body: Any, raw: bool = False) -> None:
                payload = body.encode() if raw else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:  # noqa: N802
                sc = outer.scenario
                if self.path == "/demo/session":
                    if sc.get("session_status", 200) != 200:
                        return self._json(sc["session_status"], {"detail": "x"})
                    if sc.get("session_raw") is not None:
                        return self._json(200, sc["session_raw"], raw=True)
                    return self._json(200, {"session_token": "SECRET-SESSION-TOKEN-123"})
                if self.path == "/auth/ws-ticket":
                    assert self.headers["Authorization"] == "Bearer SECRET-SESSION-TOKEN-123"
                    return self._json(200, {"ticket": "tick123456"})
                self._json(404, {})

            def do_GET(self) -> None:  # noqa: N802
                assert self.path == "/chat/stream?ticket=tick123456"
                key = self.headers["Sec-WebSocket-Key"]
                accept = base64.b64encode(
                    sha1((key + canary.WS_GUID).encode()).digest()  # noqa: S324
                ).decode()
                if outer.scenario.get("bad_accept"):
                    accept = "AAAAAAAAAAAAAAAAAAAAAAAAAAA="
                self.send_response(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                sock = self.connection
                b0, b1 = sock.recv(2)
                outer.client_masked.append(bool(b1 & 0x80))
                n = b1 & 0x7F
                mask = sock.recv(4)
                body = b""
                while len(body) < n:
                    body += sock.recv(n - len(body))
                text = bytes(c ^ mask[i % 4] for i, c in enumerate(body))
                outer.seen_user_messages.append(json.loads(text))
                if outer.scenario.get("raw") is not None:
                    for chunk, delay in outer.scenario["raw"]:
                        sock.sendall(chunk)
                        if delay:
                            time.sleep(delay)
                    if outer.scenario.get("expect_pong"):
                        pb0, pb1 = sock.recv(2)
                        pmask = sock.recv(4)
                        pbody = sock.recv(pb1 & 0x7F)
                        outer.pongs.append(
                            (pb0, bytes(c ^ pmask[i % 4] for i, c in enumerate(pbody)))
                        )
                    return
                for f in outer.scenario["frames"]:
                    sock.sendall(_ws_frame(f if isinstance(f, str) else json.dumps(f)))
                if outer.scenario.get("close", True):
                    sock.sendall(bytes([0x88, 2]) + struct.pack(">H", 1000))
                else:  # hold open silently to provoke the client timeout
                    threading.Event().wait(2)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> _Server:
        self.thread.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def run_full(scenario: dict[str, Any], budget: float = 15000, ws_timeout: float = 5.0):
    with _Server(scenario) as srv:
        token = canary.http_post_json(f"{srv.url}/demo/session")["session_token"]
        res = canary.run_query(srv.url, token, "black kurta set for haldi", budget, ws_timeout)
    return res, srv


# ------------------------------------------------------------------ end-to-end (loopback)


def test_pass_uses_frontend_path_and_payload() -> None:
    res, srv = run_full({"frames": good_frames()})
    assert res.ok, res.failures
    assert res.cards == 2 and res.crosschecked == 2
    assert res.answer == "Here is your look."
    assert srv.seen_user_messages == [
        {"type": "user_message", "conversation_id": None, "message": "black kurta set for haldi"}
    ]


def test_empty_answer_fails() -> None:
    frames = [f for f in good_frames() if f["type"] != "token"]
    res, _ = run_full({"frames": frames})
    assert "answer is empty" in res.failures


def test_zero_cards_fails() -> None:
    frames = [f for f in good_frames() if f["type"] != "items"]
    frames[-1]["final_state"] = {}
    res, _ = run_full({"frames": frames})
    assert "no product cards returned" in res.failures


def test_price_mismatch_vs_item_data_fails() -> None:
    frames = good_frames()
    frames[-1]["final_state"]["outfit_variants"][0]["items"][0]["price_inr"] = 2499.0
    res, _ = run_full({"frames": frames})
    assert any("a1" in f and "!= item data price" in f for f in res.failures)


def test_latency_over_budget_fails_without_extra_queries() -> None:
    res, srv = run_full({"frames": good_frames()}, budget=-1)  # force_fail mode
    assert any("exceeds budget -1 ms" in f for f in res.failures)
    assert len(srv.seen_user_messages) == 1


def test_error_frame_fails() -> None:
    res, _ = run_full({"frames": [{"type": "error", "message": "boom", "code": "x"}]})
    assert any("error frame" in f for f in res.failures)


def test_close_before_done_fails() -> None:
    res, _ = run_full({"frames": good_frames()[:2]})
    assert any("closed before done" in f for f in res.failures)


def test_malformed_ws_json_fails() -> None:
    res, _ = run_full({"frames": ["not json{"]})
    assert "WS frame was not valid JSON" in res.failures


def test_no_done_frame_times_out() -> None:
    res, _ = run_full({"frames": good_frames()[:1], "close": False}, ws_timeout=0.5)
    assert any("no done frame within" in f for f in res.failures)


def test_non_200_session_is_canary_error_and_redacted() -> None:
    with _Server({"frames": [], "session_status": 503}) as srv:
        with pytest.raises(canary.CanaryError, match="HTTP 503"):
            canary.http_post_json(f"{srv.url}/demo/session")


def test_malformed_session_json() -> None:
    with _Server({"frames": [], "session_raw": "<html>"}) as srv:
        with pytest.raises(canary.CanaryError, match="malformed JSON"):
            canary.http_post_json(f"{srv.url}/demo/session")


# ------------------------------------------------------------------ pure assertion logic


def _eval(**kw: Any):
    base = dict(answer="hi", items_frame=[dict(GOOD_ITEM)], final_state={},
                latency_ms=100, budget_ms=15000)
    return canary.evaluate("q", **(base | kw))


def test_evaluate_pass_without_reference_reports_zero_crosschecked() -> None:
    res = _eval()
    assert res.ok and res.crosschecked == 0


@pytest.mark.parametrize("price", [None, 0, -5, "100", True, float("nan"), float("inf")])
def test_invalid_price_fails(price: Any) -> None:
    res = _eval(items_frame=[{"article_id": "a1", "price_inr": price}])
    assert any("price_inr invalid" in f for f in res.failures)


def test_missing_article_id_fails_and_owned_item_exempt() -> None:
    assert any("no article_id" in f for f in _eval(items_frame=[{"price_inr": 5}]).failures)
    owned = {"article_id": "o1", "price_inr": None, "is_owned": True}
    assert _eval(items_frame=[owned, dict(GOOD_ITEM)]).ok


def test_duplicate_card_with_different_price_fails() -> None:
    dup = dict(GOOD_ITEM) | {"price_inr": 1.0}
    assert any("!= item data" in f for f in _eval(items_frame=[GOOD_ITEM, dup]).failures)


def test_cards_fall_back_to_first_outfit_variant() -> None:
    fs = {"outfit_variants": [{"items": [dict(GOOD_ITEM)]}]}
    res = _eval(items_frame=None, final_state=fs)
    assert res.ok and res.cards == 1


# ------------------------------------------------------------------ llm_degraded

_BUSY = canary.LLM_BUSY_MESSAGE
_FALLBACKS = [
    _BUSY,
    "Sorry. " + _BUSY,  # busy message embedded in a longer reply
    canary.GROUNDED_FALLBACK_EMPTY,
    canary.GROUNDED_FALLBACK_PREFIX + "Kurta (\u20b91,299). Tap any item for details.",
]


@pytest.mark.parametrize("answer", _FALLBACKS)
def test_degraded_reply_fails_as_real_failure_not_capped(answer: str) -> None:
    res = _eval(answer=answer)
    assert any(f.startswith("llm_degraded") for f in res.failures)
    assert not res.capped and not res.ok
    assert canary.classify([res]) == "fail"  # fail -> opens/updates the prod-alert issue
    assert canary.decide_issue_action("fail", []).kind == "create"


def test_healthy_reply_still_passes() -> None:
    assert _eval(answer="A lovely black kurta set.").ok


def test_busy_message_over_the_wire_fails_the_run() -> None:
    frames = [f for f in good_frames() if f["type"] != "token"]
    frames.insert(2, {"type": "token", "text": _BUSY})
    res, _ = run_full({"frames": frames})
    assert any(f.startswith("llm_degraded") for f in res.failures)
    assert not res.capped


def test_duplicated_fallback_texts_match_the_app_constants() -> None:
    """scripts/ops/canary.py is stdlib-only and duplicates these strings; keep them pinned."""
    from src.agents import grounding
    from src.llm.resilience import LLM_BUSY_MESSAGE

    assert canary.LLM_BUSY_MESSAGE == LLM_BUSY_MESSAGE
    assert canary.GROUNDED_FALLBACK_PREFIX == grounding.GROUNDED_FALLBACK_PREFIX
    assert canary.GROUNDED_FALLBACK_EMPTY == grounding.GROUNDED_FALLBACK_EMPTY
    assert grounding.grounded_fallback_reply([]) == canary.GROUNDED_FALLBACK_EMPTY
    assert grounding.grounded_fallback_reply(
        [{"display_name": "Kurta", "price_inr": 5}]
    ).startswith(canary.GROUNDED_FALLBACK_PREFIX)


# ------------------------------------------------------------------ redaction


def test_redact_removes_tokens_and_tickets() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbm9uIn0.abcdefghijkl"
    text = f"Authorization: Bearer abc.def-123 ticket=tick123456 {jwt} token: xyz987654"
    out = canary.redact(text)
    for secret in ("abc.def-123", "tick123456", jwt, "xyz987654"):
        assert secret not in out


def test_snippet_truncates_to_500() -> None:
    assert len(canary.snippet("x" * 900)) == 500


def test_http_error_message_never_contains_session_token() -> None:
    err = None
    with _Server({"frames": [], "session_status": 500}) as srv:
        try:
            canary.http_post_json(f"{srv.url}/demo/session", {"Authorization": "Bearer SECRET-SESSION-TOKEN-123"})
        except canary.CanaryError as exc:
            err = str(exc)
    assert err and "SECRET-SESSION-TOKEN-123" not in err


# ------------------------------------------------------------------ daily cap


CURRENT = [("500", "in_progress", "")]  # the running canary itself, always in the listing


def rows(n: int, conclusion: str = "success") -> list[tuple[str, str, str]]:
    return [(str(i), "completed", conclusion) for i in range(1, n + 1)]


def test_cap_counts_prior_runs_excluding_current_and_skipped() -> None:
    r = rows(10) + [("99", "completed", "skipped"), ("500", "in_progress", "")]
    assert canary.count_queries_today(r, "500") == 30


def test_cap_boundary() -> None:
    assert not canary.would_exceed_cap(147)  # 147 + 3 == 150 allowed
    assert canary.would_exceed_cap(148)
    assert canary.would_exceed_cap(150)


def _args(tmp_path: Path, **kw: Any):
    import argparse
    d = dict(base_url="http://127.0.0.1:1", out=str(tmp_path / "r.json"), force_fail=False,
             repo="o/r", workflow="canary.yml", run_id="500")
    return argparse.Namespace(**(d | kw))


def test_run_skips_without_querying_when_over_cap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(canary, "fetch_run_rows", lambda *a, **k: rows(50) + CURRENT)  # 150 spent
    called = []
    monkeypatch.setattr(canary, "http_post_json", lambda *a, **k: called.append(a) or {})
    assert canary.run_canary(_args(tmp_path)) == 0  # skip is not a failure
    out = json.loads((tmp_path / "r.json").read_text())
    assert out["status"] == "skip" and "daily cap" in out["note"]
    assert called == []


def test_run_skips_when_cap_lookup_fails(tmp_path: Path, monkeypatch) -> None:
    def boom(*a: Any, **k: Any):
        raise canary.CanaryError("gh api failed")
    monkeypatch.setattr(canary, "fetch_run_rows", boom)
    monkeypatch.setattr(canary.time, "sleep", lambda s: None)
    assert canary.run_canary(_args(tmp_path)) == 0
    assert json.loads((tmp_path / "r.json").read_text())["status"] == "skip"


def test_fetch_run_rows_parses_gh_tsv() -> None:
    out = canary.fetch_run_rows("o/r", "canary.yml", "2026-11-03",
                                lambda args, stdin: "1\tcompleted\tsuccess\n2\tin_progress\t\n")
    assert out == [("1", "completed", "success"), ("2", "in_progress", "")]


def test_run_canary_end_to_end_force_fail(tmp_path: Path, capsys) -> None:
    with _Server({"frames": good_frames()}) as srv:
        rc = canary.run_canary(_args(tmp_path, base_url=srv.url, repo="", force_fail=True))
    out = json.loads((tmp_path / "r.json").read_text())
    assert rc == 1 and out["status"] == "fail" and out["forced"] is True
    assert len(srv.seen_user_messages) == 3  # exactly the normal 3 queries
    assert "SECRET-SESSION-TOKEN-123" not in json.dumps(out)
    # Windows cp1252 consoles crashed on the rupee sign in a query (found by the live run).
    assert capsys.readouterr().out.isascii()


# ------------------------------------------------------------------ issue lifecycle


@pytest.mark.parametrize(
    ("status", "open_issues", "kind", "numbers"),
    [
        ("fail", [], "create", ()),
        ("fail", [7, 3], "comment", (3,)),
        ("pass", [], "none", ()),
        ("pass", [9, 4], "close", (4, 9)),
        ("skip", [4], "none", ()),
        ("skip", [], "none", ()),
    ],
)
def test_decide_issue_action(status: str, open_issues: list[int], kind: str, numbers: tuple) -> None:
    assert canary.decide_issue_action(status, open_issues) == canary.IssueAction(kind, numbers)


def _fail_result() -> dict[str, Any]:
    return {"status": "fail", "timestamp": "t", "budget_ms": 15000, "forced": False,
            "queries": [{"query": "q1", "failures": ["answer is empty"], "latency_ms": 12.5,
                         "answer": "Bearer abc.def-123 " + "z" * 900}]}


OWN = "5\t" + canary.ALERT_TITLE  # an open issue with the canary's own title


class _FakeGh:
    def __init__(self, open_issues: str, labels: str) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.open_issues, self.labels = open_issues, labels

    def __call__(self, args: list[str], stdin: str | None) -> str:
        self.calls.append((args, stdin))
        if args[:2] == ["issue", "list"]:
            return self.open_issues
        if args[:2] == ["label", "list"]:
            return self.labels
        return ""


def test_apply_create_makes_label_and_body_is_clean() -> None:
    fake = _FakeGh("", "bug\n")
    action = canary.apply_issue_action("o/r", _fail_result(), "https://x/run/1", fake)
    assert action.kind == "create"
    verbs = [c[0][:2] for c in fake.calls]
    assert ["label", "create"] in verbs and ["issue", "create"] in verbs
    body = next(c[1] for c in fake.calls if c[0][:2] == ["issue", "create"])
    assert "q1" in body and "answer is empty" in body and "12.5" in body and "https://x/run/1" in body
    assert "abc.def-123" not in body
    snippet_part = body.split("```")[1]
    assert len(snippet_part.strip()) <= 500


def test_apply_comment_skips_label_creation() -> None:
    fake = _FakeGh(OWN + "\n", "prod-alert\n")
    action = canary.apply_issue_action("o/r", _fail_result(), "u", fake)
    assert action == canary.IssueAction("comment", (5,))
    assert not any(c[0][:2] == ["issue", "create"] for c in fake.calls)


def test_apply_close_comments_then_closes() -> None:
    fake = _FakeGh(OWN + "\n" + OWN.replace("5", "8") + "\n", "prod-alert\n")
    canary.apply_issue_action("o/r", {"status": "pass", "timestamp": "t"}, "u", fake)
    verbs = [c[0][:2] for c in fake.calls if c[0][0] == "issue" and c[0][1] != "list"]
    assert verbs == [["issue", "comment"], ["issue", "close"]] * 2


def test_apply_none_on_skip_only_lists() -> None:
    fake = _FakeGh(OWN + "\n", "")
    assert canary.apply_issue_action("o/r", {"status": "skip"}, "u", fake).kind == "none"
    assert all(c[0][:2] == ["issue", "list"] for c in fake.calls)


# ================================================================== round 2 (verifier fixes)
import socket  # noqa: E402


def _run_with_frames(frames: list[Any]):
    return run_full({"frames": frames})[0]


# ---- M1: hostile frame shapes must become failures, never crashes ------------------------


@pytest.mark.parametrize(
    ("frames", "needle"),
    [
        (["[1, 2]"], "not a JSON object"),
        (["null"], "not a JSON object"),
        ([{"type": "items", "items": 5}], "non-list items"),
        ([{"type": "done", "final_state": ["x"]}], "final_state is not an object"),
        ([{"type": "done", "final_state": {"outfit_variants": 7, "partner_look": 3}}],
         "answer is empty"),
    ],
)
def test_hostile_frame_shapes_are_failures_not_crashes(frames: list[Any], needle: str) -> None:
    res = _run_with_frames(frames)
    assert any(needle in f for f in res.failures), res.failures
    assert not any(f.startswith("unexpected") for f in res.failures)


def test_huge_frame_length_is_rejected_without_allocating() -> None:
    raw = [(bytes([0x81, 127]) + struct.pack(">Q", 2**62), 0)]
    res, _ = run_full({"frames": [], "raw": raw})
    assert any("exceeds" in f for f in res.failures), res.failures


def test_memory_error_inside_query_is_reported_as_unexpected(monkeypatch) -> None:
    def boom(*a: Any, **k: Any):
        raise MemoryError
    monkeypatch.setattr(canary, "ws_recv_text", boom)
    res, _ = run_full({"frames": good_frames()})
    assert "unexpected: MemoryError" in res.failures


def test_evaluate_crash_is_reported_as_unexpected(monkeypatch) -> None:
    def boom(*a: Any, **k: Any):
        raise TypeError
    monkeypatch.setattr(canary, "collect_cards", boom)
    res, _ = run_full({"frames": good_frames()})
    assert "unexpected: TypeError" in res.failures


def test_run_canary_crash_still_writes_fail_result(tmp_path: Path, monkeypatch) -> None:
    def boom(*a: Any, **k: Any):
        raise RuntimeError("secret-ish detail")
    monkeypatch.setattr(canary, "_run_inner", boom)
    assert canary.run_canary(_args(tmp_path, repo="")) == 1
    out = json.loads((tmp_path / "r.json").read_text())
    assert out["status"] == "fail" and out["note"] == "unexpected: RuntimeError"
    assert "secret-ish" not in json.dumps(out)
    assert canary.decide_issue_action(out["status"], []).kind == "create"


# ---- M2: cap lookup ------------------------------------------------------------------------


def _tsv_runner(rows_: list[tuple[str, str, str]]):
    return lambda args, stdin: "".join(f"{i}\t{st}\t{c}\n" for i, st, c in rows_)


def test_lookup_requires_current_run_in_listing(monkeypatch) -> None:
    monkeypatch.setattr(canary.time, "sleep", lambda s: None)
    for rows_ in ([], rows(5)):  # empty and partial (current run id 500 missing)
        with pytest.raises(canary.CanaryError, match="not in workflow-runs listing"):
            canary.lookup_spent("o/r", "canary.yml", "2026-11-03", "500", _tsv_runner(rows_))


def test_lookup_counts_when_current_present_and_retries(monkeypatch) -> None:
    monkeypatch.setattr(canary.time, "sleep", lambda s: None)
    calls = []

    def runner(args: list[str], stdin: str | None) -> str:
        calls.append(1)
        if len(calls) == 1:
            raise canary.CanaryError("gh api failed")
        return "1\tcompleted\tsuccess\n2\tcompleted\tsuccess\n500\tin_progress\t\n"
    assert canary.lookup_spent("o/r", "canary.yml", "2026-11-03", "500", runner) == 6
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("prev", "forced", "expected"),
    [(0, False, ("skip", 1)), (1, False, ("skip", 2)), (2, False, ("fail", 3)),
     (7, False, ("fail", 8)), (5, True, ("error", 5))],
)
def test_decide_lookup_failure(prev: int, forced: bool, expected: tuple[str, int]) -> None:
    assert canary.decide_lookup_failure(prev, forced) == expected


def _fetch_down(*a: Any, **k: Any):
    raise canary.CanaryError("api down")


def test_three_consecutive_lookup_failures_fail_the_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(canary.time, "sleep", lambda s: None)
    monkeypatch.setattr(canary, "fetch_run_rows", _fetch_down)
    state = tmp_path / "state"
    codes = [canary.run_canary(_args(tmp_path, state_file=str(state))) for _ in range(3)]
    out = json.loads((tmp_path / "r.json").read_text())
    assert codes == [0, 0, 1] and out["status"] == "fail" and state.read_text() == "3"
    assert canary.decide_issue_action(out["status"], []).kind == "create"


def test_successful_lookup_resets_failure_count(tmp_path: Path, monkeypatch) -> None:
    state = tmp_path / "state"
    state.write_text("2")
    monkeypatch.setattr(canary, "fetch_run_rows", lambda *a, **k: rows(50) + CURRENT)
    canary.run_canary(_args(tmp_path, state_file=str(state)))  # capped -> skip, lookup was fine
    assert state.read_text() == "0"


def test_forced_run_with_failed_lookup_exits_nonzero(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(canary.time, "sleep", lambda s: None)
    monkeypatch.setattr(canary, "fetch_run_rows", _fetch_down)
    assert canary.run_canary(_args(tmp_path, force_fail=True)) == 1
    assert "::error::" in capsys.readouterr().out
    assert json.loads((tmp_path / "r.json").read_text())["status"] == "skip"


# ---- L1: overall deadline inside a frame --------------------------------------------------


def test_trickling_server_cannot_exceed_overall_deadline() -> None:
    raw = [(bytes([0x81, 100]), 0)] + [(b"x", 0.2)] * 40  # header promises 100 bytes, 1 per 0.2 s
    t0 = time.perf_counter()
    res, _ = run_full({"frames": [], "raw": raw}, ws_timeout=1.0)
    assert time.perf_counter() - t0 < 2.5
    assert any("no done frame within" in f for f in res.failures), res.failures


# ---- L2: WebSocket protocol ------------------------------------------------------------------


def test_client_frames_are_masked() -> None:
    _, srv = run_full({"frames": good_frames()})
    assert srv.client_masked == [True]


def test_bad_sec_websocket_accept_is_rejected() -> None:
    res, _ = run_full({"frames": good_frames(), "bad_accept": True})
    assert any("bad Sec-WebSocket-Accept" in f for f in res.failures), res.failures


def _pair() -> tuple[socket.socket, socket.socket]:
    a, b = socket.socketpair()
    a.settimeout(3)
    b.settimeout(3)
    return a, b


def test_server_frames_with_16bit_and_64bit_lengths() -> None:
    a, b = _pair()
    text16 = "a" * 300
    text64 = "b" * 70000
    b.sendall(bytes([0x81, 126]) + struct.pack(">H", 300) + text16.encode())
    b.sendall(bytes([0x81, 127]) + struct.pack(">Q", 70000) + text64.encode())
    assert canary.ws_recv_text(a) == text16
    assert canary.ws_recv_text(a) == text64
    a.close()
    b.close()


def test_fragmented_message_with_interleaved_ping_gets_masked_pong() -> None:
    a, b = _pair()
    b.sendall(bytes([0x01, 3]) + b"hel")  # text, FIN=0
    b.sendall(bytes([0x89, 2]) + b"pp")  # ping
    b.sendall(bytes([0x80, 2]) + b"lo")  # continuation, FIN=1
    assert canary.ws_recv_text(a) == "hello"
    b0, b1 = b.recv(2)
    mask = b.recv(4)
    body = bytes(c ^ mask[i % 4] for i, c in enumerate(b.recv(b1 & 0x7F)))
    assert b0 == 0x8A and (b1 & 0x80) and body == b"pp"
    a.close()
    b.close()


def test_oversized_ping_is_a_protocol_error_not_a_crash() -> None:
    a, b = _pair()
    b.sendall(bytes([0x89, 126]) + struct.pack(">H", 200) + b"p" * 200)  # ping > 125 bytes
    with pytest.raises(canary.CanaryError, match="control frame"):
        canary.ws_recv_text(a)
    a.close()
    b.close()


def test_close_frame_raises_wsclosed_with_code() -> None:
    a, b = _pair()
    b.sendall(bytes([0x88, 2]) + struct.pack(">H", 1008))
    with pytest.raises(canary.WsClosed, match="1008"):
        canary.ws_recv_text(a)
    a.close()
    b.close()


# ---- L3: demo / rate caps are not outages ----------------------------------------------------


@pytest.mark.parametrize("code", ["demo_limit", "rate_limited"])
def test_cap_error_frames_are_capped_not_failed(code: str) -> None:
    res = _run_with_frames([{"type": "error", "message": "limit", "code": code}])
    assert res.capped and not res.failures and not res.ok


def test_other_error_codes_still_fail() -> None:
    res = _run_with_frames([{"type": "error", "message": "x", "code": "internal_error"}])
    assert res.failures and not res.capped


def test_classify_and_issue_decision_for_capped_runs() -> None:
    ok = canary.QueryResult(query="q")
    cap = canary.QueryResult(query="q", capped=True)
    bad = canary.QueryResult(query="q", failures=["x"])
    assert canary.classify([ok, ok, ok]) == "pass"
    assert canary.classify([ok, cap, ok]) == "capped"
    assert canary.classify([cap, bad, ok]) == "fail"
    for open_issues in ([], [3]):
        assert canary.decide_issue_action("capped", open_issues).kind == "none"


def test_run_with_demo_cap_is_capped_exit_zero(tmp_path: Path, capsys) -> None:
    frames = [{"type": "error", "message": "limit", "code": "demo_limit"}]
    with _Server({"frames": frames}) as srv:
        rc = canary.run_canary(_args(tmp_path, base_url=srv.url, repo=""))
    assert rc == 0
    assert json.loads((tmp_path / "r.json").read_text())["status"] == "capped"
    assert "::warning::" in capsys.readouterr().out


def test_session_429_is_capped(tmp_path: Path) -> None:
    with _Server({"frames": [], "session_status": 429}) as srv:
        rc = canary.run_canary(_args(tmp_path, base_url=srv.url, repo=""))
    assert rc == 0 and json.loads((tmp_path / "r.json").read_text())["status"] == "capped"


# ---- L4: issue scoping + forced cap bypass -----------------------------------------------------


def test_issue_actions_only_touch_the_canarys_own_title() -> None:
    other = "7\tprod-alert from part B: Cloud Run ERROR count"
    fake = _FakeGh(other + "\n" + OWN + "\n", "prod-alert\n")
    canary.apply_issue_action("o/r", {"status": "pass", "timestamp": "t"}, "u", fake)
    touched = {c[0][2] for c in fake.calls if c[0][:2] in (["issue", "comment"], ["issue", "close"])}
    assert touched == {"5"}
    fake2 = _FakeGh(other + "\n", "prod-alert\n")
    assert canary.apply_issue_action("o/r", _fail_result(), "u", fake2).kind == "create"


def test_forced_run_may_exceed_cap_by_one_run() -> None:
    assert not canary.would_exceed_cap(150, forced=True)
    assert canary.would_exceed_cap(151, forced=True)
    assert canary.would_exceed_cap(150, forced=False)


def test_forced_run_at_cap_proceeds_and_beyond_cap_errors(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(canary, "fetch_run_rows", lambda *a, **k: rows(50) + CURRENT)  # 150 spent
    with _Server({"frames": good_frames()}) as srv:
        rc = canary.run_canary(_args(tmp_path, base_url=srv.url, force_fail=True))
    assert rc == 1 and len(srv.seen_user_messages) == 3  # ran (and failed on purpose)
    monkeypatch.setattr(canary, "fetch_run_rows", lambda *a, **k: rows(51) + CURRENT)  # 153 spent
    assert canary.run_canary(_args(tmp_path, force_fail=True)) == 1
    out = json.loads((tmp_path / "r.json").read_text())
    assert out["status"] == "skip" and "::error::" in capsys.readouterr().out


# ---- latency boundary + M3 summary ---------------------------------------------------------------


def test_latency_equal_to_budget_fails_just_under_passes() -> None:
    assert any("exceeds budget" in f for f in _eval(latency_ms=15000, budget_ms=15000).failures)
    assert _eval(latency_ms=14999.9, budget_ms=15000).ok


def test_run_prints_summary_with_crosschecked_total(tmp_path: Path, capsys) -> None:
    with _Server({"frames": good_frames()}) as srv:
        canary.run_canary(_args(tmp_path, base_url=srv.url, repo=""))
    out = capsys.readouterr().out
    assert "SUMMARY: queries=3" in out and "crosschecked_total=6" in out
