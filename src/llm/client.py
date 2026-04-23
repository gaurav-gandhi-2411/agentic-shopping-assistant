import os
from typing import Iterator, Protocol

import ollama


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str = None, **kwargs) -> str: ...
    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]: ...
    def chat(self, messages: list[dict], **kwargs) -> str: ...
    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]: ...


class OllamaClient:
    """Wraps ollama.Client. Messages use OpenAI-style {role, content}."""

    def __init__(self, config: dict):
        self.config = config
        llm_cfg = config["llm"]
        self.model = llm_cfg["model"]
        self.default_temperature = llm_cfg["temperature"]
        self.default_max_tokens = llm_cfg["max_tokens"]
        self.timeout = llm_cfg["timeout_seconds"]
        self._client = ollama.Client(
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
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
        )
        return resp.choices[0].message.content

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


def get_llm_client(config: dict) -> OllamaClient | GroqClient:
    provider = config["llm"]["provider"]
    if provider == "ollama":
        return OllamaClient(config)
    if provider == "groq":
        return GroqClient(config)
    raise NotImplementedError(f"Unknown LLM provider: {provider!r}. Supported: ollama, groq")
