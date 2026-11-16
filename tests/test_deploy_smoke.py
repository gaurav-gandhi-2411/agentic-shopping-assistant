"""Unit tests for scripts/ops/deploy_smoke.py (hermetic: loopback server only, no network)."""
from __future__ import annotations

import importlib.util
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

# DEPLOY_SMOKE_PATH lets a mutation check point these tests at a mutated copy of the script.
_PATH = Path(
    os.environ.get("DEPLOY_SMOKE_PATH")
    or Path(__file__).resolve().parents[1] / "scripts" / "ops" / "deploy_smoke.py"
)
_spec = importlib.util.spec_from_file_location("deploy_smoke", _PATH)
assert _spec is not None and _spec.loader is not None
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        self._send(200 if self.path == "/ok" else 404, {"path": self.path})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self._send(200, {"echo": body, "auth": self.headers.get("Authorization")})

    def log_message(self, *args: object) -> None:  # silence test output
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_request_get_ok_and_http_error_is_returned_not_raised(server: str) -> None:
    assert smoke.request("GET", server + "/ok", timeout=5)[0] == 200
    assert smoke.request("GET", server + "/nope", timeout=5)[0] == 404


def test_request_post_sends_json_and_bearer(server: str) -> None:
    status, text = smoke.request("POST", server + "/x", body={"a": 1}, token="tok", timeout=5)
    assert status == 200
    assert json.loads(text) == {"echo": {"a": 1}, "auth": "Bearer tok"}


def test_request_connection_failure_returns_status_zero() -> None:
    status, text = smoke.request("GET", "http://127.0.0.1:9/", timeout=2)
    assert status == 0 and text


def test_fail_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        smoke.fail("boom")
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# main()-level tests: every failure mode must exit 1, the happy path must not.
# ---------------------------------------------------------------------------

_OK_ROUTES: dict[tuple[str, str], tuple[int, str]] = {
    ("GET", "/api/brand"): (200, json.dumps({"brand": "x"})),
    ("GET", "/readyz"): (200, json.dumps({"status": "ready"})),
    ("POST", "/demo/session"): (200, json.dumps({"session_token": "tok"})),
    ("POST", "/chat"): (200, json.dumps({"response": "Try these.", "items": [{"id": 1}]})),
}


class _RecordingServer(HTTPServer):
    seen: list[dict]


def _serve(routes: dict[tuple[str, str], tuple[int, str]]) -> _RecordingServer:
    seen: list[dict] = []  # every request the server received (method, path, auth header, json body)

    class H(BaseHTTPRequestHandler):
        def _go(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            raw_in = self.rfile.read(length) if length else b""
            seen.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "body": json.loads(raw_in) if raw_in else None,
                }
            )
            status, body = routes.get((self.command, self.path), (404, "{}"))
            raw = body.encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = do_POST = _go  # noqa: N815

        def log_message(self, *args: object) -> None:
            pass

    srv = _RecordingServer(("127.0.0.1", 0), H)
    srv.seen = seen
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture()
def run_main(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(smoke, "RETRY_SLEEP_S", 0)
    servers = []

    def _run(overrides: dict | None = None, extra_args: list[str] | None = None) -> list[dict]:
        routes = dict(_OK_ROUTES)
        routes.update(overrides or {})
        srv = _serve(routes)
        servers.append(srv)
        smoke.main(
            ["--base-url", f"http://127.0.0.1:{srv.server_port}", *(extra_args or [])],
            require_https=False,
        )
        return srv.seen

    yield _run
    for s in servers:
        s.shutdown()


def _expect_fail(run, overrides: dict, extra_args: list[str] | None = None) -> None:
    with pytest.raises(SystemExit) as exc:
        run(overrides, extra_args)
    assert exc.value.code == 1


def test_main_happy_path(run_main) -> None:
    run_main()  # returns normally


def test_main_no_chat_skips_llm_endpoints(run_main) -> None:
    # /demo/session and /chat are broken, but --no-chat must not touch them.
    run_main(
        {("POST", "/demo/session"): (500, "{}"), ("POST", "/chat"): (500, "{}")}, ["--no-chat"]
    )


def test_main_rejects_http_base_url_by_default() -> None:
    # A LIVE loopback http server, default require_https: main() must refuse before sending
    # anything. (An unreachable host would also exit 1 and hide a removed https check.)
    srv = _serve(dict(_OK_ROUTES))
    try:
        with pytest.raises(SystemExit) as exc:
            smoke.main(["--base-url", f"http://127.0.0.1:{srv.server_port}"])
        assert exc.value.code == 1
        assert srv.seen == []
    finally:
        srv.shutdown()


def test_main_sends_bearer_token_and_exact_query_on_chat(run_main) -> None:
    seen = run_main()
    chat = [r for r in seen if r["method"] == "POST" and r["path"] == "/chat"]
    assert len(chat) == 1
    assert chat[0]["auth"] == "Bearer tok"  # token from the stubbed /demo/session
    assert chat[0]["body"] == {"message": smoke.QUERY}
    assert smoke.QUERY == "black kurta set for haldi"
    session = [r for r in seen if r["path"] == "/demo/session"]
    assert len(session) == 1 and session[0]["auth"] is None


def test_main_no_chat_sends_no_post_at_all(run_main) -> None:
    seen = run_main(extra_args=["--no-chat"])
    assert [r["method"] for r in seen] == ["GET", "GET"]


def test_main_fails_on_empty_response(run_main) -> None:
    _expect_fail(run_main, {("POST", "/chat"): (200, json.dumps({"response": "  ", "items": [1]}))})


def test_main_fails_on_zero_items(run_main) -> None:
    _expect_fail(run_main, {("POST", "/chat"): (200, json.dumps({"response": "hi", "items": []}))})


def test_main_fails_when_not_ready(run_main) -> None:
    _expect_fail(run_main, {("GET", "/readyz"): (200, json.dumps({"status": "not ready"}))})


@pytest.mark.parametrize(
    "path",
    [("GET", "/api/brand"), ("GET", "/readyz"), ("POST", "/demo/session"), ("POST", "/chat")],
)
def test_main_fails_on_http_500_for_each_path(run_main, path: tuple[str, str]) -> None:
    _expect_fail(run_main, {path: (500, "{}")})


def test_main_fails_on_missing_session_token(run_main) -> None:
    _expect_fail(run_main, {("POST", "/demo/session"): (200, json.dumps({}))})


@pytest.mark.parametrize(
    "path", [("GET", "/readyz"), ("POST", "/demo/session"), ("POST", "/chat")]
)
def test_main_fails_cleanly_on_non_json_body(run_main, path: tuple[str, str]) -> None:
    _expect_fail(run_main, {path: (200, "<html>oops</html>")})


# ---------------------------------------------------------------------------
# Rollout mode: --expect-build-id (Wave 1b V3c). /readyz must show the pinned build.
# ---------------------------------------------------------------------------

BUILD = "20261116T020000Z"


def _ready(**index: object) -> tuple[int, str]:
    block = {
        "source": "build",
        "pinned_build_id": BUILD,
        "loaded_build_id": BUILD,
        "age_hours": 1.5,
    }
    block.update(index)
    return 200, json.dumps({"status": "ready", "index": block})


_EXPECT = ["--no-chat", "--expect-build-id", BUILD]


def test_expect_build_id_passes_when_the_pinned_build_is_loaded_and_fresh(run_main) -> None:
    run_main({("GET", "/readyz"): _ready()}, _EXPECT)


@pytest.mark.parametrize(
    "bad",
    [
        {"source": "legacy"},
        {"source": "legacy-fallback"},
        {"loaded_build_id": "20261115T020000Z"},
        {"pinned_build_id": "20261115T020000Z"},
        {"loaded_build_id": None},
        {"age_hours": 36.0},  # the limit itself is stale: strictly < 36
        {"age_hours": 90},
        {"age_hours": None},
        {"age_hours": "1.5"},
        {"age_hours": True},
    ],
)
def test_expect_build_id_fails_on_each_wrong_index_state(run_main, bad: dict) -> None:
    _expect_fail(run_main, {("GET", "/readyz"): _ready(**bad)}, _EXPECT)


def test_expect_build_id_fails_when_readyz_has_no_index_block(run_main) -> None:
    _expect_fail(run_main, {("GET", "/readyz"): (200, json.dumps({"status": "ready"}))}, _EXPECT)


def test_age_just_under_the_limit_passes_and_the_limit_is_configurable(run_main) -> None:
    run_main({("GET", "/readyz"): _ready(age_hours=35.99)}, _EXPECT)
    _expect_fail(
        run_main, {("GET", "/readyz"): _ready(age_hours=5)}, [*_EXPECT, "--max-index-age-hours", "4"]
    )


def test_without_the_flag_the_index_block_is_not_inspected(run_main) -> None:
    run_main({("GET", "/readyz"): _ready(source="legacy", age_hours=999)}, ["--no-chat"])


def test_index_problem_names_the_reason() -> None:
    ready = json.loads(_ready(age_hours=40)[1])
    assert "age_hours" in (smoke.index_problem(ready, BUILD) or "")
    assert smoke.index_problem(ready, BUILD, max_age_h=48) is None
    assert "expected" in (smoke.index_problem(ready, "20300101T000000Z") or "")
