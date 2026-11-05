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


def test_no_gemini_key_and_no_fallback_tier_budget_in_config() -> None:
    cfg = _llm_cfg()
    assert not [k for k in cfg if "gemini" in k.lower()]
    # The per-fallback-tier budget belonged to the removed OpenRouter tier.
    assert "fallback_tier_budget_seconds" not in cfg


def test_openrouter_model_key_is_local_dev_primary_only_and_priced() -> None:
    # Kept for LLM_PROVIDER=openrouter (local dev); must stay priced so a swap to a
    # paid model cannot silently report $0.
    assert _llm_cfg()["openrouter_model"] in CONSTANTS


def test_chain_budget_is_a_20s_hard_cap_in_shipped_config() -> None:
    # Total retry/fallback wall time (GG requirement); clamped per attempt in GroqClient.
    assert _llm_cfg()["groq_chain_budget_seconds"] == 20
