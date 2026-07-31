import json
import logging
import os
import time
import uuid
from typing import Callable, Iterator

from src.llm.context import llm_user_id_var
from src.llm.cost import TurnCost

logger = logging.getLogger(__name__)


def _rate_limit_wait(exc) -> float:
    """Extract reset timestamp from OpenRouter 429 metadata and return seconds to sleep.

    OpenRouter 429 errors carry metadata.headers['X-RateLimit-Reset'] as a Unix
    millisecond timestamp. Falls back to 65s if not present.
    """
    try:
        meta = exc.body.get("error", {}).get("metadata", {})
        headers = meta.get("headers", {})
        reset_ms = headers.get("X-RateLimit-Reset")
        if reset_ms:
            wait = (int(reset_ms) / 1000.0) - time.time()
            return max(wait + 1.0, 1.0)
    except Exception:
        pass
    return 65.0


class OpenRouterClient:
    """Wraps the OpenAI-compatible OpenRouter API.
    Requires: pip install openai  and  OPENROUTER_API_KEY env var."""

    def __init__(self, config: dict):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        llm_cfg = config["llm"]
        self.model = llm_cfg.get("openrouter_model", "openai/gpt-oss-20b:free")
        self.default_temperature = llm_cfg["temperature"]
        self.default_max_tokens = llm_cfg["max_tokens"]
        # Explicit per-request HTTP timeout — see GroqClient's identical comment
        # in src/llm/client.py. The openai SDK's own default (600s) is already
        # bounded but not aligned with this app's timeout_seconds convention.
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=llm_cfg.get("timeout_seconds", 60),
        )
        self.cost_reporter: Callable[[float], None] | None = None

    def chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        # Rate-limit retries: up to 10 waits (parsed from reset header or 65s default)
        # Other errors: short fixed backoff (1s, 3s) then raise.
        ratelimit_retries = 0
        other_delays = iter([1.0, 3.0])
        attempt = 0
        t0 = time.monotonic()
        turn_id = str(uuid.uuid4())
        while True:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.default_temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
                )
                usage = resp.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                latency_ms = round((time.monotonic() - t0) * 1000)
                cost = TurnCost(self.model, input_tokens, output_tokens).usd_cost
                logger.info(
                    json.dumps({
                        "event": "llm_call",
                        "provider": "openrouter",
                        "model": self.model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_tokens": 0,
                        "latency_ms": latency_ms,
                        "usd_cost": round(cost, 8),
                        "user_id": llm_user_id_var.get(""),
                        "turn_id": turn_id,
                    })
                )
                if self.cost_reporter is not None:
                    self.cost_reporter(cost)
                return resp.choices[0].message.content
            except Exception as exc:
                attempt += 1
                status = getattr(exc, "status_code", None)
                if status == 429 and ratelimit_retries < 10:
                    ratelimit_retries += 1
                    wait = _rate_limit_wait(exc)
                    logger.warning("[openrouter] attempt %d rate-limited. Waiting %.0fs…", attempt, wait)
                    time.sleep(wait)
                    continue
                delay = next(other_delays, None)
                if delay is None:
                    raise
                logger.warning("[openrouter] attempt %d failed: %r. Retrying in %.1fs…", attempt, exc, delay)
                time.sleep(delay)

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> Iterator[str]:
        from src.llm.client import STREAM_ERROR_SENTINEL

        t0 = time.monotonic()
        input_tokens = 0
        output_tokens = 0
        turn_id = str(uuid.uuid4())
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.default_temperature,
                max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
        except Exception as exc:
            logger.error("[openrouter] chat_stream error: %s", exc, exc_info=True)
            yield STREAM_ERROR_SENTINEL
        finally:
            latency_ms = round((time.monotonic() - t0) * 1000)
            cost = TurnCost(self.model, input_tokens, output_tokens).usd_cost
            logger.info(
                json.dumps({
                    "event": "llm_call",
                    "provider": "openrouter",
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": 0,
                    "latency_ms": latency_ms,
                    "usd_cost": round(cost, 8),
                    "user_id": llm_user_id_var.get(""),
                    "turn_id": turn_id,
                })
            )
            if self.cost_reporter is not None:
                self.cost_reporter(cost)

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat_stream(messages, **kwargs)
