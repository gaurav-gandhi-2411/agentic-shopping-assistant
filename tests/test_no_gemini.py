"""W7: the Gemini provider and the google-genai dependency are gone."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import get_llm_client

_ROOT = Path(__file__).parent.parent
# Source trees that ship or run; tests/ is excluded because this file names the
# module in order to forbid it.
_SCAN_DIRS = ("api", "src", "eval", "evals", "scripts")
_GENAI_IMPORT = re.compile(r"^\s*(?:from\s+google(?:\.genai)?\s+import\s+genai|"
                           r"from\s+google\.genai\b|import\s+google\.genai\b)", re.M)


def test_gemini_provider_is_rejected_with_supported_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(NotImplementedError) as exc:
        get_llm_client({"llm": {"provider": "gemini"}})
    msg = str(exc.value)
    assert "gemini" in msg and "Supported: ollama, groq, openrouter" in msg


def test_gemini_client_module_is_gone() -> None:
    assert not (_ROOT / "src" / "llm" / "gemini_client.py").exists()
    with pytest.raises(ModuleNotFoundError):
        __import__("src.llm.gemini_client")


def test_no_module_imports_google_genai() -> None:
    """Grep-based, so it holds whether or not google-genai happens to be installed."""
    offenders = [
        str(p.relative_to(_ROOT))
        for d in _SCAN_DIRS
        if (_ROOT / d).is_dir()
        for p in (_ROOT / d).rglob("*.py")
        if _GENAI_IMPORT.search(p.read_text(encoding="utf-8", errors="replace"))
    ]
    assert offenders == []


def test_requirements_do_not_list_google_genai() -> None:
    for name in ("requirements.txt", "pyproject.toml"):
        assert "google-genai" not in (_ROOT / name).read_text(encoding="utf-8")
