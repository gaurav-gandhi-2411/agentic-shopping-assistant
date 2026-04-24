import os
import time
from typing import Iterator


class GeminiClient:
    """Wraps google-generativeai for Gemini 2.0 Flash.
    Requires: pip install google-generativeai  and  GEMINI_API_KEY env var."""

    def __init__(self, config: dict):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        genai.configure(api_key=api_key)

        llm_cfg = config["llm"]
        self.model_name = llm_cfg.get("gemini_model", "gemini-2.0-flash-exp")
        self.default_temperature = llm_cfg["temperature"]
        self.default_max_tokens = llm_cfg["max_tokens"]
        self._genai = genai

    def _get_model(self, temperature: float = None, max_tokens: int = None):
        return self._genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self._genai.types.GenerationConfig(
                temperature=temperature if temperature is not None else self.default_temperature,
                max_output_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            ),
        )

    def chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        model = self._get_model(temperature, max_tokens)
        gemini_messages = _to_gemini_messages(messages)
        delays = iter([1.0, 3.0])
        attempt = 0
        while True:
            try:
                if gemini_messages["system"] and len(gemini_messages["history"]) == 0:
                    # Single-turn with system prompt: send as combined user message
                    response = model.generate_content(gemini_messages["prompt"])
                else:
                    chat_session = model.start_chat(history=gemini_messages["history"])
                    response = chat_session.send_message(gemini_messages["prompt"])
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
        model = self._get_model(temperature, max_tokens)
        gemini_messages = _to_gemini_messages(messages)
        if gemini_messages["history"]:
            chat_session = model.start_chat(history=gemini_messages["history"])
            stream = chat_session.send_message(gemini_messages["prompt"], stream=True)
        else:
            stream = model.generate_content(gemini_messages["prompt"], stream=True)
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


def _to_gemini_messages(messages: list[dict]) -> dict:
    """Convert OpenAI-style messages to Gemini format.

    Returns dict with keys: system (str|None), history (list), prompt (str).
    Gemini's multi-turn API uses alternating user/model roles; system content
    is prepended to the first user message.
    """
    system_parts = []
    history = []
    prompt = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            history.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            history.append({"role": "model", "parts": [content]})

    if history:
        # Prepend system content into the first user message
        if system_parts and history[0]["role"] == "user":
            system_prefix = "\n\n".join(system_parts) + "\n\n"
            history[0]["parts"][0] = system_prefix + history[0]["parts"][0]

        # Last message is the current user turn (the prompt to send)
        last = history[-1]
        if last["role"] == "user":
            prompt = last["parts"][0]
            history = history[:-1]
        else:
            prompt = ""

    return {"system": "\n\n".join(system_parts) if system_parts else None, "history": history, "prompt": prompt}
