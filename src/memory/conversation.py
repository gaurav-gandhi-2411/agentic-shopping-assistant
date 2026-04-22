from src.llm.client import LLMClient


class ConversationMemory:
    def __init__(self, llm: LLMClient, config: dict):
        self.llm = llm
        self.recent_turns = config["memory"]["recent_turns"]
        self.trigger_turns = config["memory"]["summary_trigger_turns"]
        self._cached_summary: str | None = None
        self._summary_computed_at: int = 0  # len(messages) when summary was last built

    def get_context(self, messages: list[dict]) -> list[dict]:
        """Return messages trimmed to recent_turns, prepended with a summary if history is long."""
        if len(messages) <= self.recent_turns:
            return messages

        older = messages[: -self.recent_turns]
        if (
            self._cached_summary is None
            or len(messages) - self._summary_computed_at >= self.trigger_turns
        ):
            self._cached_summary = self.summarise(older)
            self._summary_computed_at = len(messages)

        summary_msg = {
            "role": "system",
            "content": f"Summary of earlier conversation:\n{self._cached_summary}",
        }
        return [summary_msg] + messages[-self.recent_turns :]

    def summarise(self, messages: list[dict]) -> str:
        """LLM call: summarise older turns into 3 bullets."""
        formatted = "\n".join(
            f"{m['role'].title()}: {m['content'][:200]}" for m in messages
        )
        prompt = (
            "Summarise this shopping conversation in 3 bullets, "
            "preserving user preferences, constraints, and items already discussed.\n\n"
            f"Conversation:\n{formatted}\n\nSummary:"
        )
        return self.llm.generate(prompt)
