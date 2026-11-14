"""Static guards on the Dockerfile's non-root runtime user (no image build needed).

These parse the Dockerfile text only. They cannot prove the container actually starts or that
the index download path is writable -- that needs a real build (done in verification).
"""

from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"


def _instructions() -> list[tuple[str, str]]:
    """Return (INSTRUCTION, args) pairs, joining backslash continuations, skipping comments."""
    out: list[tuple[str, str]] = []
    buf = ""
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        word, _, rest = buf.partition(" ")
        out.append((word.upper(), rest.strip()))
        buf = ""
    return out


def _final_stage() -> list[tuple[str, str]]:
    ins = _instructions()
    last_from = max(i for i, (w, _) in enumerate(ins) if w == "FROM")
    return ins[last_from:]


def test_last_user_before_cmd_is_not_root() -> None:
    stage = _final_stage()
    cmd_idx = next(i for i, (w, _) in enumerate(stage) if w in {"CMD", "ENTRYPOINT"})
    users = [a.split(":")[0].strip() for w, a in stage[:cmd_idx] if w == "USER"]
    assert users, "final stage has no USER instruction (container would run as root)"
    assert users[-1].lower() not in {"root", "0"}, f"last USER before CMD is {users[-1]!r}"


def test_user_has_fixed_numeric_uid_and_no_login_shell() -> None:
    text = " ".join(a for w, a in _final_stage() if w == "RUN")
    m = re.search(r"useradd\b[^&]*--uid\s+(\d+)", text)
    assert m, "useradd with an explicit --uid not found"
    assert int(m.group(1)) >= 1000
    assert "nologin" in text or "/bin/false" in text


def test_alembic_still_copied_for_readyz_schema_gate() -> None:
    copies = [a for w, a in _final_stage() if w == "COPY"]
    assert any(re.search(r"\balembic\.ini\b", a) for a in copies)
    assert any(re.search(r"\balembic/\s", a + " ") for a in copies)


def test_writable_cache_env_vars_declared() -> None:
    envs = " ".join(a for w, a in _final_stage() if w == "ENV")
    for var in ("HF_HOME", "SENTENCE_TRANSFORMERS_HOME", "HOME", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        assert re.search(rf"\b{var}=", envs), f"{var} not set in final stage"


def test_download_target_is_chowned_to_app_user() -> None:
    runs = " ".join(a for w, a in _final_stage() if w == "RUN")
    assert re.search(r"chown\s+app:app[^&]*/app/data/processed", runs)


def test_port_unprivileged_and_healthcheck_present() -> None:
    stage = _final_stage()
    ports = [int(p) for w, a in stage if w == "EXPOSE" for p in re.findall(r"\d+", a)]
    assert ports and all(p > 1024 for p in ports)
    assert any(w == "HEALTHCHECK" for w, _ in stage)
