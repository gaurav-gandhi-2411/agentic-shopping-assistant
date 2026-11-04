"""The shipped config.yaml must describe a Groq-only fallback chain that can
actually survive one model's quota running out, and every model in it must be
priced. Reads the real config.yaml; no network."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.cost import CONSTANTS

_ROOT = Path(__file__).parent.parent


def _llm_cfg() -> dict:
    return yaml.safe_load((_ROOT / "config.yaml").read_text(encoding="utf-8"))["llm"]


def _groq_chain(cfg: dict) -> list[str]:
    return [cfg["groq_model"], *cfg.get("groq_fallback_models", [])]


def test_groq_chain_has_at_least_two_distinct_models() -> None:
    # Distinct ids = distinct per-model free-tier quotas; a repeated id adds none.
    chain = _groq_chain(_llm_cfg())
    assert len(set(chain)) >= 2, chain
    assert len(chain) == len(set(chain)), f"duplicate model in chain: {chain}"


def test_every_groq_chain_model_has_a_cost_rate() -> None:
    # CostTracker silently falls back to the $0 ollama rate for an unknown model.
    missing = [m for m in _groq_chain(_llm_cfg()) if m not in CONSTANTS]
    assert missing == []


def test_only_remaining_non_groq_tier_is_openrouter_and_priced() -> None:
    cfg = _llm_cfg()
    assert cfg["openrouter_model"] in CONSTANTS
    assert not [k for k in cfg if "gemini" in k.lower()]
