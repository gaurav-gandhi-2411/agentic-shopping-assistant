import os
import re
import time
from typing import Iterator, Protocol


def _parse_retry_after(exc_str: str) -> float:
    """Extract 'Please try again in Xm Ys' from a Groq rate-limit error string."""
    m = re.search(r"try again in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s", exc_str)
    if m:
        minutes = int(m.group(1)) if m.group(1) else 0
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    return 0.0


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str = None, **kwargs) -> str: ...
    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]: ...
    def chat(self, messages: list[dict], **kwargs) -> str: ...
    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]: ...


class OllamaClient:
    """Wraps ollama.Client. Messages use OpenAI-style {role, content}."""

    def __init__(self, config: dict):
        try:
            import ollama as _ollama_lib
        except ImportError:
            raise ImportError("ollama package not installed. Run: pip install ollama")
        self.config = config
        llm_cfg = config["llm"]
        self.model = llm_cfg["model"]
        self.default_temperature = llm_cfg["temperature"]
        self.default_max_tokens = llm_cfg["max_tokens"]
        self.timeout = llm_cfg["timeout_seconds"]
        self._client = _ollama_lib.Client(
            host=llm_cfg["host"],
            timeout=self.timeout,
        )

    def chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        # Collect the stream rather than using stream=False — the non-streaming
        # endpoint triggers a runner crash on some Ollama builds (Windows).
        return "".join(self.chat_stream(messages, temperature=temperature, max_tokens=max_tokens))

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> Iterator[str]:
        options = {
            "temperature": temperature if temperature is not None else self.default_temperature,
            "num_predict": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        stream = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            stream=True,
        )
        for chunk in stream:
            content = chunk.message.content
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


class GroqClient:
    """Wraps the Groq API. Drop-in replacement for OllamaClient on HF Spaces.
    Requires: pip install groq  and  GROQ_API_KEY env var."""

    def __init__(self, config: dict):
        try:
            import groq as _groq_lib
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        llm_cfg = config["llm"]
        self.model = llm_cfg.get("groq_model", "llama-3.1-8b-instant")
        self.default_temperature = llm_cfg["temperature"]
        self.default_max_tokens = llm_cfg["max_tokens"]
        self._client = _groq_lib.Groq(api_key=api_key)

    def chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        # Separate TPD (daily) retries from TPM (per-minute) backoff.
        # TPD: parse "try again in Xm Ys" and wait the full duration (up to 10 times).
        # TPM and other errors: use a short fixed backoff (1s, 3s), then raise.
        tpd_retries = 0
        tpm_delays = iter([1.0, 3.0])
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
                exc_str = str(exc)
                if ("tokens per day" in exc_str or "(TPD)" in exc_str) and tpd_retries < 10:
                    tpd_retries += 1
                    wait = _parse_retry_after(exc_str)
                    wait = max(wait, 60.0)  # minimum 60s for TPD
                    print(f"[groq] attempt {attempt} failed: {exc!r}. TPD limit — waiting {wait:.0f}s...")
                    time.sleep(wait + 2)
                    continue
                delay = next(tpm_delays, None)
                if delay is None:
                    raise
                print(f"[groq] attempt {attempt} failed: {exc!r}. Retrying in {delay}s...")
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


def get_llm_client(config: dict) -> "OllamaClient | GroqClient":
    provider = os.environ.get("LLM_PROVIDER") or config["llm"]["provider"]
    if provider == "ollama":
        return OllamaClient(config)
    if provider == "groq":
        return GroqClient(config)
    if provider == "gemini":
        from src.llm.gemini_client import GeminiClient
        return GeminiClient(config)
    raise NotImplementedError(f"Unknown LLM provider: {provider!r}. Supported: ollama, groq, gemini")
