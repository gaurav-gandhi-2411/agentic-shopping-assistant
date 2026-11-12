"""Deterministic stand-in LLM for the grounding harness (no network, seed 42).

Answers the respond prompt (graph.RESPOND_PROMPT) with a templated reply built from the items that
prompt actually lists, so grounded templates quote real prices and ungrounded templates inject a
known violation. Router prompts get a canned search action; every other prompt gets an empty
object, which the reranker/others treat as "no preference".

Modes:
    grounded   - never states a number/size/stock/delivery fact beyond the item prices it was given
    ungrounded - always injects one violation, chosen by the bait keywords in the user's question
    mixed      - sha256(seed, prompt) parity picks one of the two; same prompt => same choice on
                 the REST and WS paths
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

SEED = 42  # repo convention: hardcoded seed wherever behaviour is pseudo-random

_ITEM_RE = re.compile(
    r"- display_name: (?P<name>.*)\n\s+colour: .*?\| price: (?P<price>₹[\d,]+)?", re.M
)
_QUERY_RE = re.compile(r'User question: "(?P<q>.*)"\s*\n', re.S)
_ROUTER_QUERY_RE = re.compile(r"<user_query>(?P<q>.*?)</user_query>", re.S)
_BUDGET_RE = re.compile(
    r"(?:under|below|within|upto|up to)\s*(?:a budget of\s*)?(?:₹|rs\.?\s*)?(?P<n>\d[\d,]{2,})", re.I
)

# bait keyword -> ungrounded template id; first match wins (order matters)
_BAIT = [
    ("U_DISC", r"\boff\b|discount|\bsale\b|coupon|clearance|\bsave\b|\bmrp\b|bogo|buy one"),
    ("U_SHIP", r"ship|deliver|\bcod\b|return|express|\bdays?\b|cash on"),
    ("U_STOCK", r"stock|sold|\bleft\b|available|restock|reserve|last piece"),
    ("U_SIZE", r"\bsize|\bxs\b|\bxxl|xxxl|\d+xl|petite|waist|\buk\b|\bmedium\b|\bfit\b"),
    ("U_PRICE", r"price|cost|how much|cheapest|compare|\bexact\b|₹|\brupees?\b|\b\d{3,}\b"),
]
_GROUNDED_IDS = ("G_PRICE", "G_PLAIN", "G_DENY")


@dataclass
class MockEvent:
    """One respond-prompt answer, recorded so the harness can tell injected from clean."""

    template: str
    injected: bool
    raw_text: str


@dataclass
class MockLLM:
    """Deterministic LLM double. `events` is cleared by the harness between cases."""

    mode: str = "mixed"
    seed: int = SEED
    events: list[MockEvent] = field(default_factory=list)
    n_calls: int = 0
    cost_reporter: Any = None

    # -- LLMClient surface used by the app -------------------------------------------------
    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        """Blocking completion."""
        self.n_calls += 1
        return self._dispatch(prompt)

    def generate_stream(self, prompt: str, system: str | None = None, **kwargs: Any) -> Iterator[str]:
        """Token stream: the same text as generate(), split on words."""
        self.n_calls += 1
        for word in self._dispatch(prompt).split(" "):
            yield word + " "

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        """Chat completion; the last user message is treated as the prompt."""
        self.n_calls += 1
        return self._dispatch(str(messages[-1].get("content", "")) if messages else "")

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        """Streaming chat completion."""
        self.n_calls += 1
        yield self._dispatch(str(messages[-1].get("content", "")) if messages else "")

    # -- internals -------------------------------------------------------------------------
    def _bit(self, prompt: str) -> int:
        digest = hashlib.sha256(f"{self.seed}|{prompt}".encode()).digest()
        return digest[0]

    def _dispatch(self, prompt: str) -> str:
        if "Available item attributes:" in prompt:
            return self._respond(prompt)
        if "Respond with ONLY the JSON object" in prompt:
            m = _ROUTER_QUERY_RE.search(prompt)
            return json.dumps(
                {"action": "search", "query": (m.group("q") if m else "").strip(), "filters": {}}
            )
        return "{}"

    def _respond(self, prompt: str) -> str:
        qm = _QUERY_RE.search(prompt)
        query = qm.group("q") if qm else ""
        items = [
            (m.group("name").strip(), m.group("price") or "")
            for m in _ITEM_RE.finditer(prompt.split("Available item attributes:", 1)[1])
        ][:2]
        bit = self._bit(prompt)
        injected = self.mode == "ungrounded" or (self.mode == "mixed" and bit % 2 == 1)
        template, text = (
            self._ungrounded(query, items, bit) if injected else self._grounded(query, items, bit)
        )
        self.events.append(MockEvent(template, injected, text))
        return text

    @staticmethod
    def _names(items: list[tuple[str, str]]) -> tuple[str, str, str, str]:
        n1, p1 = items[0] if items else ("this piece", "")
        n2, p2 = items[1] if len(items) > 1 else ("another option", "")
        return n1, p1, n2, p2

    def _grounded(self, query: str, items: list[tuple[str, str]], bit: int) -> tuple[str, str]:
        if not items:
            return "G_NONE", (
                "I couldn't find a close match for that. Try describing the piece, occasion "
                "or colour a little differently and I will take another look."
            )
        n1, p1, n2, p2 = self._names(items)
        bm = _BUDGET_RE.search(query)
        tail = f" These stay within your budget of ₹{bm.group('n')}." if bm else ""
        if any(re.search(p, query, re.I) for t, p in _BAIT if t != "U_PRICE"):
            return "G_DENY", (
                "I don't have discount, delivery, size or stock information for these, so "
                f"please check the product page. {n1} is a good place to start"
                + (f" at {p1}." if p1 else ".")
                + tail
            )
        if p1 and p2 and bit % 3 == 0:
            return "G_PRICE", f"{n1} at {p1} is a lovely pick, and {n2} ({p2}) pairs well with it.{tail}"
        return "G_PLAIN", f"I'd start with {n1}; the cut suits what you described. {n2} is a good second option.{tail}"

    def _ungrounded(self, query: str, items: list[tuple[str, str]], bit: int) -> tuple[str, str]:
        n1, p1, n2, p2 = self._names(items)
        tid = next((t for t, pat in _BAIT if re.search(pat, query, re.I)), None)
        if tid is None:  # regular prompt: rotate through the five violation kinds
            tid = ("U_DISC", "U_SHIP", "U_STOCK", "U_SIZE", "U_PRICE")[(bit // 2) % 5]
        if tid == "U_DISC":
            return tid, f"{n1} is currently {10 + 10 * (bit % 5)}% off, a great deal!"
        if tid == "U_SHIP":
            return tid, f"{n1} ships free and will be delivered in 3-5 days."
        if tid == "U_STOCK":
            return tid, f"{n1} is in stock, but only 2 left!"
        if tid == "U_SIZE":
            return tid, f"{n1} is available in sizes S, M, L and XL."
        if p2 and p2 != p1:  # price of the wrong item
            return "U_PRICE", f"{n1} is {p2}."
        digits = int(re.sub(r"\D", "", p1) or "1299")
        return "U_PRICE", f"{n1} is just ₹{digits + 501:,}."
