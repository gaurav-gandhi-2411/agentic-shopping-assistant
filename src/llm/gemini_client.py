import os
import time
from typing import Iterator


class GeminiClient:
    """Wraps google-genai for Gemini 2.0 Flash.
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
        self.model_name = llm_cfg.get("gemini_model", "gemini-2.0-flash")
        self.default_temperature = llm_cfg["temperature"]
        self.default_max_tokens = llm_cfg["max_tokens"]
        self._client = genai.Client(api_key=api_key)
        self._types = _types

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
                return response.text
            except Exception as exc:
                attempt += 1
                delay = next(delays, None)
                if delay is None:
                    raise
                print(f"[gemini] attempt {attempt} failed: {exc!r}. Retrying in {delay}s...")
                time.sleep(delay)

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> Iterator[str]:
        system, history, prompt = _split_messages(messages)
        config = self._gen_config(temperature, max_tokens, system)
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
