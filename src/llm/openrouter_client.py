import os
import time
from typing import Iterator


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
        self.model = llm_cfg.get("openrouter_model", "google/gemma-3-27b-it:free")
        self.default_temperature = llm_cfg["temperature"]
        self.default_max_tokens = llm_cfg["max_tokens"]
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

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
        while True:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.default_temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as exc:
                attempt += 1
                status = getattr(exc, "status_code", None)
                if status == 429 and ratelimit_retries < 10:
                    ratelimit_retries += 1
                    wait = _rate_limit_wait(exc)
                    print(f"[openrouter] attempt {attempt} rate-limited. Waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                delay = next(other_delays, None)
                if delay is None:
                    raise
                print(f"[openrouter] attempt {attempt} failed: {exc!r}. Retrying in {delay}s...")
                time.sleep(delay)

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

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
