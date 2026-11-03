"""Process-level leak tests: alembic and uvicorn with an unreachable DB (hermetic).

Both run in a real subprocess so their own logging config (alembic.ini handlers,
uvicorn's handlers) and the interpreter's uncaught-traceback printing are covered.
The only network activity is the DNS failure for a ``*.invalid`` host.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HOST = "db.example.invalid"  # RFC 6761: never resolves

# (password, distinctive fragments).  First has every reserved char the directive lists
# ('%' followed by a non-hex so it is read as raw); second is the production shape
# (only '@' inside the password).
CASES = [
    ("Alpha@Bravo:Charlie/Delta#Echo%Fox Golf@Hotel",
     ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Fox", "Golf", "Hotel"]),
    ("Zulu@Yankee", ["Zulu", "Yankee"]),
]
_IDS = ["all-reserved", "prod-shape"]

_SCRUB_ENV = ("SENTRY_DSN", "GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY",
              "DEMO_JWT_SECRET", "LANGSMITH_API_KEY", "ANTHROPIC_API_KEY")


def _env(password: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _SCRUB_ENV}
    env["DATABASE_URL"] = f"postgresql://postgres.projref:{password}@{HOST}:5432/postgres"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _assert_clean(output: str, password: str, fragments: list[str]) -> None:
    assert password not in output
    for frag in fragments:
        assert frag not in output, f"password fragment {frag!r} leaked:\n{output[-1500:]}"


@pytest.mark.parametrize(("password", "fragments"), CASES, ids=_IDS)
def test_alembic_failure_output_has_no_password(password: str, fragments: list[str]) -> None:
    proc = subprocess.run(  # noqa: S603 - fixed argv, test-controlled env
        [sys.executable, "-m", "alembic", "current"],
        cwd=ROOT, env=_env(password), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120, check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert HOST in output  # it really failed on connect, naming the true host
    _assert_clean(output, password, fragments)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - localhost
            return resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.read().decode("utf-8", "replace")


@pytest.mark.parametrize(("password", "fragments"), CASES, ids=_IDS)
def test_uvicorn_startup_and_requests_have_no_password(
    password: str, fragments: list[str], tmp_path: Path
) -> None:
    port = _free_port()
    log_path = tmp_path / "uvicorn.log"
    bodies: list[str] = []
    with log_path.open("wb") as log:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, test-controlled env
            [sys.executable, "-m", "uvicorn", "tests._leak_app:app",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=_env(password), stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    bodies.append(_get(f"http://127.0.0.1:{port}/readyz"))
                    bodies.append(_get(f"http://127.0.0.1:{port}/boom"))
                    bodies.append(_get(f"http://127.0.0.1:{port}/logurl"))
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                    time.sleep(0.5)
            time.sleep(0.5)  # let uvicorn flush the access/exception log lines
        finally:
            if proc.poll() is None:
                proc.terminate()  # only the PID we started
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
    output = log_path.read_text(encoding="utf-8", errors="replace") + "\n".join(bodies)
    # On the pre-fix code the app can fail to even start; that output is still scanned.
    _assert_clean(output, password, fragments)
    # Anti-vacuity: on the fixed code the server answered and named the true host.
    assert bodies, f"server never answered; log:\n{output[-1500:]}"
    assert HOST in output
    # /logurl logged the raw URL: the redaction layer (not URL parsing) must have masked it.
    assert "config url postgresql://postgres.projref:***@" in output


@pytest.mark.parametrize("module", ["api.main", "src.agents.graph"])
@pytest.mark.parametrize(("password", "fragments"), CASES, ids=_IDS)
def test_import_time_output_has_no_credentials(
    module: str, password: str, fragments: list[str]
) -> None:
    """Importing the app / agent graph with DATABASE_URL set must not print the URL.

    -W default surfaces every warning (incl. normally-suppressed ones).
    """
    proc = subprocess.run(  # noqa: S603 - fixed argv, test-controlled env
        [sys.executable, "-W", "default", str(ROOT / "tests" / "_import_probe.py"), module],
        cwd=ROOT, env={**_env(password), "PYTHONPATH": str(ROOT)}, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=180, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    _assert_clean(proc.stdout + proc.stderr, password, fragments)
    assert "postgresql" not in (proc.stdout + proc.stderr)


def test_imports_work_without_database_url() -> None:
    env = _env("x")
    env.pop("DATABASE_URL")
    for module in ("api.main", "src.agents.graph"):
        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(ROOT / "tests" / "_import_probe.py"), module],
            cwd=ROOT, env={**env, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180, check=False,
        )
        assert proc.returncode == 0, proc.stderr[-1500:]
