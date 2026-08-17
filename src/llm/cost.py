"""Per-call cost estimation from token counts."""
from __future__ import annotations

from dataclasses import dataclass

# USD per million tokens.  cached_read rates apply to Anthropic prompt-cache
# read hits only (Wave 3+).  Groq and Ollama have no caching tiers.
CONSTANTS: dict[str, dict[str, float]] = {
    # Groq
    # llama-3.1-8b-instant deprecated by Groq 2026-08-16; groq_model in
    # config.yaml/src/llm/client.py now defaults to "openai/gpt-oss-20b", but
    # that model has NO entry here yet -- its real $/M-token Groq rate could
    # not be verified from a live source as of this fix (2026-08-18, Groq's
    # pricing page did not return pricing data via automated fetch). Left
    # OUT deliberately rather than guessing a number CLAUDE.md's cost-honesty
    # rules forbid stating as fact: CONSTANTS.get(self.model) or
    # CONSTANTS["ollama"] (see below) means an unrecognized model currently
    # falls through to a SILENT $0 cost -- this is a real, open gap for the
    # new default model, not a solved one. Add a verified "openai/gpt-oss-20b"
    # entry here (check console.groq.com's live pricing page directly, a
    # page that needs JS rendering this fetch couldn't get past) before
    # trusting any cost report that includes calls made under this model.
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
    # OpenRouter free-tier model (config.yaml llm.openrouter_model). $0 by
    # OpenRouter's own pricing for ":free"-suffixed models — kept explicit
    # rather than relying on the CONSTANTS.get() ollama fallback so a future
    # switch to a paid OpenRouter model doesn't silently report $0 cost.
    "openai/gpt-oss-20b:free": {
        "input":        0.0,
        "output":       0.0,
        "cached_read":  0.0,
        "cached_write": 0.0,
    },
    # Dead per 2026-07-31 live test: OpenRouter deprecated the free variant of
    # this model entirely (404 "This model is unavailable for free"). Left in
    # place (harmless) rather than deleted so old log lines with this model
    # string still resolve to a cost instead of falling through to
    # CONSTANTS["ollama"]'s $0 default for an unrelated reason.
    "google/gemma-3-27b-it:free": {
        "input":        0.0,
        "output":       0.0,
        "cached_read":  0.0,
        "cached_write": 0.0,
    },
    # Gemini free tier (config.yaml llm.gemini_model). Confirmed $0 via
    # ai.google.dev/gemini-api/docs/pricing (2026-07-31): "Free Tier" column
    # for gemini-2.5-flash input/output is explicitly "Free of charge" — this
    # app only ever calls Gemini via a non-billing-enabled API key (confirmed
    # live by the 429 RESOURCE_EXHAUSTED/limit:0 quota error, which is a
    # free-tier-only failure mode). If this ever switches to a billed key,
    # the paid-tier rate is $0.30/$2.50 per 1M input/output tokens — update
    # this entry then so cost no longer silently reports $0.
    "gemini-2.5-flash": {
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
