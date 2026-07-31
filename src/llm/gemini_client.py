import json
import logging
import os
import time
import uuid
from typing import Callable, Iterator

from src.llm.context import llm_user_id_var
from src.llm.cost import TurnCost

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wraps google-genai for Gemini 2.5 Flash.
    Requires: pip install google-genai  and  GEMINI_API_KEY env var."""

    def __init__(self, config: dict):
        try:
            from google import genai
            from google.genai import types as _types
        except ImportError:
            raise ImportError(
                "google-genai package not installed. Run: pip install google-genai"
            )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        llm_cfg = config["llm"]
        self.model_name = llm_cfg.get("gemini_model", "gemini-2.5-flash")
        self.default_temperature = llm_cfg["temperature"]
        self.default_max_tokens = llm_cfg["max_tokens"]
        # Explicit per-request HTTP timeout (milliseconds — see google-genai's
        # HttpOptions.timeout docs) — see GroqClient's identical comment in
        # src/llm/client.py for why this is added even though the SDK default
        # is already bounded.
        _timeout_ms = int(llm_cfg.get("timeout_seconds", 60)) * 1000
        self._client = genai.Client(
            api_key=api_key, http_options=_types.HttpOptions(timeout=_timeout_ms)
        )
        self._types = _types
        self.cost_reporter: Callable[[float], None] | None = None

    def _gen_config(self, temperature: float = None, max_tokens: int = None, system: str = None):
        kwargs = dict(
            temperature=temperature if temperature is not None else self.default_temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
        )
        if system:
            kwargs["system_instruction"] = system
        return self._types.GenerateContentConfig(**kwargs)

    def chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        system, history, prompt = _split_messages(messages)
        config = self._gen_config(temperature, max_tokens, system)
        delays = iter([1.0, 3.0])
        attempt = 0
        t0 = time.monotonic()
        turn_id = str(uuid.uuid4())
        while True:
            try:
                if history:
                    chat_session = self._client.chats.create(
                        model=self.model_name,
                        config=config,
                        history=history,
                    )
                    response = chat_session.send_message(prompt)
                else:
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                # response.usage_metadata field names come from the SDK's
                # camelCase REST fields (promptTokenCount etc.) mapped to
                # snake_case — verified against the installed google-genai
                # package's GenerateContentResponseUsageMetadata, not guessed.
                usage = response.usage_metadata
                input_tokens = (usage.prompt_token_count if usage else 0) or 0
                output_tokens = (usage.candidates_token_count if usage else 0) or 0
                cached_tokens = (usage.cached_content_token_count if usage else 0) or 0
                latency_ms = round((time.monotonic() - t0) * 1000)
                cost = TurnCost(self.model_name, input_tokens, output_tokens, cached_tokens).usd_cost
                logger.info(
                    json.dumps({
                        "event": "llm_call",
                        "provider": "gemini",
                        "model": self.model_name,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_tokens": cached_tokens,
                        "latency_ms": latency_ms,
                        "usd_cost": round(cost, 8),
                        "user_id": llm_user_id_var.get(""),
                        "turn_id": turn_id,
                    })
                )
                if self.cost_reporter is not None:
                    self.cost_reporter(cost)
                # response.text can be None/empty on an empty-candidates or
                # safety-filtered response — the caller (GroqClient's
                # fallback chain) treats a falsy return as "this tier
                # failed" and advances to the next one; don't raise here.
                return response.text
            except Exception as exc:
                attempt += 1
                delay = next(delays, None)
                if delay is None:
                    raise
                logger.warning("[gemini] attempt %d failed: %r. Retrying in %.1fs…", attempt, exc, delay)
                time.sleep(delay)

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> Iterator[str]:
        from src.llm.client import STREAM_ERROR_SENTINEL

        system, history, prompt = _split_messages(messages)
        config = self._gen_config(temperature, max_tokens, system)
        t0 = time.monotonic()
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        turn_id = str(uuid.uuid4())
        try:
            if history:
                chat_session = self._client.chats.create(
                    model=self.model_name,
                    config=config,
                    history=history,
                )
                stream = chat_session.send_message_stream(prompt)
            else:
                stream = self._client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
                # Each chunk's usage_metadata carries the cumulative total so
                # far (verified live) — keep overwriting so the last chunk
                # wins, same pattern as Groq/OpenRouter's usage-chunk handling.
                usage = chunk.usage_metadata
                if usage:
                    input_tokens = usage.prompt_token_count or 0
                    output_tokens = usage.candidates_token_count or 0
                    cached_tokens = usage.cached_content_token_count or 0
        except Exception as exc:
            logger.error("[gemini] chat_stream error: %s", exc, exc_info=True)
            yield STREAM_ERROR_SENTINEL
        finally:
            latency_ms = round((time.monotonic() - t0) * 1000)
            cost = TurnCost(self.model_name, input_tokens, output_tokens, cached_tokens).usd_cost
            logger.info(
                json.dumps({
                    "event": "llm_call",
                    "provider": "gemini",
                    "model": self.model_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
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


def _split_messages(messages: list[dict]) -> tuple[str | None, list, str]:
    """Split OpenAI-style messages into (system_str, history_contents, prompt_str).

    history_contents is a list of google.genai Content dicts for prior turns.
    The last user message becomes the prompt string sent via send_message / generate_content.
    """
    from google.genai import types

    system_parts: list[str] = []
    history: list = []
    prompt = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            history.append(types.Content(role="user", parts=[types.Part(text=content)]))
        elif role == "assistant":
            history.append(types.Content(role="model", parts=[types.Part(text=content)]))

    # Pop the last user turn to use as the live prompt
    if history and history[-1].role == "user":
        prompt = history[-1].parts[0].text
        history = history[:-1]

    system = "\n\n".join(system_parts) if system_parts else None
    return system, history, prompt
