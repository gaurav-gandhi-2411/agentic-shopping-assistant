"""Prod must not wire OpenRouter (fallback tier removed: `:free` not enforced, key
validity unproven, account charge-ability unknown; see DEPLOY.md "OpenRouter removed").

OpenRouterClient / LLM_PROVIDER=openrouter remain as a local-dev PRIMARY provider, so
this guard covers only deploy/runtime surfaces, not src/llm/openrouter_client.py,
.env.example (documents the local provider) or docs.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_PATTERN = re.compile(r"OPENROUTER_API_KEY|asa-openrouter", re.I)
_DEPLOY_FILES = ("deploy/service.yaml", "scripts/deploy_unified.sh")


def _deploy_surfaces() -> list[Path]:
    workflows = sorted((_ROOT / ".github" / "workflows").glob("*"))
    assert workflows, "no workflow files found: the guard would be vacuous"
    return [_ROOT / n for n in _DEPLOY_FILES] + workflows


def test_deploy_surfaces_are_scanned_and_exist() -> None:
    for p in _deploy_surfaces():
        assert p.is_file(), p


def test_no_openrouter_secret_in_service_deploy_script_or_workflows() -> None:
    offenders = [
        f"{p.relative_to(_ROOT)}:{i}"
        for p in _deploy_surfaces()
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if _PATTERN.search(line)
    ]
    assert offenders == []
