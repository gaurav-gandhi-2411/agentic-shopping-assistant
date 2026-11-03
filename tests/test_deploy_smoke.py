"""Unit tests for scripts/ops/deploy_smoke.py (hermetic: loopback server only, no network)."""
from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "deploy_smoke.py"
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
