"""Per-call cost estimation from token counts."""
from __future__ import annotations

from dataclasses import dataclass

# USD per million tokens.  cached_read rates apply to Anthropic prompt-cache
# read hits only (Wave 3+).  Groq and Ollama have no caching tiers.
CONSTANTS: dict[str, dict[str, float]] = {
    # Groq
    "llama-3.1-8b-instant": {
        "input":        0.05,
        "output":       0.08,
        "cached_read":  0.0,
        "cached_write": 0.0,
    },
    # Ollama — local inference, zero marginal cost
    "ollama": {
        "input":        0.0,
        "output":       0.0,
        "cached_read":  0.0,
        "cached_write": 0.0,
    },
    # Anthropic — Wave 3 placeholders (rates as of 2026-05; 5-min prompt-cache TTL)
    "claude-haiku-4-5": {
        "input":        1.00,
        "output":       5.00,
        "cached_read":  0.10,
        "cached_write": 1.25,
    },
    "claude-haiku-4-5-20251001": {
        "input":        1.00,
        "output":       5.00,
        "cached_read":  0.10,
        "cached_write": 1.25,
    },
    "claude-sonnet-4-6": {
        "input":        3.00,
        "output":       15.00,
        "cached_read":  0.30,
        "cached_write": 3.75,
    },
}


@dataclass
class TurnCost:
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0

    @property
    def usd_cost(self) -> float:
        rates = CONSTANTS.get(self.model) or CONSTANTS["ollama"]
        non_cached = max(0, self.input_tokens - self.cached_tokens)
        return (
            non_cached          * rates["input"]       / 1_000_000
            + self.cached_tokens * rates["cached_read"] / 1_000_000
            + self.output_tokens * rates["output"]      / 1_000_000
        )
