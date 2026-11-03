"""Hermetic tests for scripts/ops/canary.py (loopback server + pure functions; no network)."""
from __future__ import annotations

import base64
import importlib.util
import json
import struct
import sys
import threading
from hashlib import sha1
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "canary", Path(__file__).resolve().parents[1] / "scripts" / "ops" / "canary.py"
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
                self.send_response(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                sock = self.connection
                b0, b1 = sock.recv(2)
                n = b1 & 0x7F
                mask = sock.recv(4)
                body = b""
                while len(body) < n:
                    body += sock.recv(n - len(body))
                text = bytes(c ^ mask[i % 4] for i, c in enumerate(body))
                outer.seen_user_messages.append(json.loads(text))
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
    monkeypatch.setattr(canary, "fetch_run_rows", lambda *a, **k: rows(50))  # 150 spent
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
    fake = _FakeGh("5\n", "prod-alert\n")
    action = canary.apply_issue_action("o/r", _fail_result(), "u", fake)
    assert action == canary.IssueAction("comment", (5,))
    assert not any(c[0][:2] == ["issue", "create"] for c in fake.calls)


def test_apply_close_comments_then_closes() -> None:
    fake = _FakeGh("5\n8\n", "prod-alert\n")
    canary.apply_issue_action("o/r", {"status": "pass", "timestamp": "t"}, "u", fake)
    verbs = [c[0][:2] for c in fake.calls if c[0][0] == "issue" and c[0][1] != "list"]
    assert verbs == [["issue", "comment"], ["issue", "close"]] * 2


def test_apply_none_on_skip_only_lists() -> None:
    fake = _FakeGh("5\n", "")
    assert canary.apply_issue_action("o/r", {"status": "skip"}, "u", fake).kind == "none"
    assert all(c[0][:2] == ["issue", "list"] for c in fake.calls)
