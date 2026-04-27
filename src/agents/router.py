"""
Router backends — LLM-based and DistilBERT-based.

Both implement decide(state) -> dict and return the same format
as router_node in graph.py: {"current_plan": <json str>, "tool_calls": [...]}.

Note: the out-of-catalogue short-circuit is handled in graph.py's router_node
closure (where _detect_ooc lives) and is NOT duplicated here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Protocol

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 6-class label map — must match train_router_distilbert.py
_LABEL_MAP: dict[str, int] = {
    "clarify": 0, "compare": 1, "filter": 2,
    "outfit": 3, "respond": 4, "search": 5,
}
_ID2LABEL: dict[int, str] = {v: k for k, v in _LABEL_MAP.items()}


class RouterBackend(Protocol):
    def decide(self, state: dict) -> dict: ...


# ---------------------------------------------------------------------------
# LLM router backend
# ---------------------------------------------------------------------------

class LLMRouterBackend:
    """Wraps the existing prompt-based routing logic."""

    def __init__(
        self,
        llm,
        memory,
        prompt_template: str,
        format_items_brief: Callable,
        format_messages: Callable,
        parse_response: Callable,
    ):
        self._llm = llm
        self._memory = memory
        self._prompt = prompt_template
        self._format_items_brief = format_items_brief
        self._format_messages = format_messages
        self._parse_response = parse_response

    def decide(self, state: dict) -> dict:
        tool_calls = state.get("tool_calls", [])
        last_action = "none"
        for tc in reversed(tool_calls):
            key = list(tc.keys())[0]
            if key != "router_decision":
                last_action = key
                break

        context = self._memory.get_context(state.get("messages", []))
        prompt = self._prompt.format(
            last_action=last_action,
            items_retrieved=len(state.get("retrieved_items", [])),
            retrieved_summary=self._format_items_brief(state.get("retrieved_items", [])),
            current_filters=json.dumps(state.get("filters", {})),
            user_query=state["user_query"],
            conversation=self._format_messages(context),
        )
        raw = self._llm.generate(prompt)
        parsed = self._parse_response(raw, state["user_query"])
        return {
            "current_plan": json.dumps(parsed),
            "tool_calls": tool_calls + [{"router_decision": parsed}],
        }


# ---------------------------------------------------------------------------
# DistilBERT router backend
# ---------------------------------------------------------------------------

class DistilBERTRouterBackend:
    """Fine-tuned 6-class DistilBERT classifier."""

    _CLARIFY_QUESTION = "I'd love to help! Could you tell me a bit more about what you're looking for?"
    _MAX_LENGTH = 128

    def __init__(self, model_path: str | Path, catalogue_df=None):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"DistilBERT model not found at {path}. Run training first.")
        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

        # Lowercase vocab per facet (for filter param extraction). Longest first so
        # "dark blue" is matched before "blue".
        self._facet_vocab: dict[str, list[str]] = {}
        if catalogue_df is not None:
            for col in [
                "colour_group_name", "product_type_name",
                "index_group_name", "garment_group_name",
            ]:
                if col in catalogue_df.columns:
                    vals = sorted(
                        catalogue_df[col].dropna().str.lower().unique().tolist(),
                        key=len, reverse=True,
                    )
                    self._facet_vocab[col] = vals

    # ------------------------------------------------------------------
    # Encode + predict
    # ------------------------------------------------------------------

    @staticmethod
    def _encode(query: str, last_action: str, items_retrieved: int, active_filters: dict) -> str:
        return (
            f"query: {query} | last_action: {last_action} "
            f"| items: {items_retrieved} | filters: {json.dumps(active_filters)}"
        )

    def _predict(self, text: str) -> tuple[str, float]:
        inputs = self._tokenizer(
            text, return_tensors="pt",
            max_length=self._MAX_LENGTH, truncation=True, padding=True,
        ).to(self._device)
        # DistilBERT has no token_type_ids — drop if tokenizer included them.
        inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(probs.argmax(-1).item())
        return _ID2LABEL[pred_id], float(probs[0, pred_id].item())

    def _extract_filter_params(self, query: str) -> tuple[str, str]:
        """Scan query for known facet values; return (facet_key, value) or ("","")."""
        q = query.lower()
        for facet, vals in self._facet_vocab.items():
            for val in vals:
                if re.search(r"\b" + re.escape(val) + r"\b", q):
                    return facet, val
        return "", ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def decide(self, state: dict) -> dict:
        tool_calls = state.get("tool_calls", [])
        user_query = state["user_query"]

        last_action = "none"
        for tc in reversed(tool_calls):
            key = list(tc.keys())[0]
            if key != "router_decision":
                last_action = key
                break

        items_retrieved = len(state.get("retrieved_items", []))
        active_filters = state.get("filters", {})

        text = self._encode(user_query, last_action, items_retrieved, active_filters)
        route, confidence = self._predict(text)

        if route == "search":
            plan: dict = {"action": "search", "query": user_query, "filters": {}}
        elif route == "filter":
            key, value = self._extract_filter_params(user_query)
            if key and value:
                plan = {"action": "filter", "key": key, "value": value}
            else:
                # Can't extract params — fall back to search and let search_node
                # auto-extract facets from the query text as it normally does.
                plan = {"action": "search", "query": user_query, "filters": {}}
        elif route == "compare":
            plan = {"action": "compare", "article_ids": []}
        elif route == "clarify":
            plan = {"action": "clarify", "question": self._CLARIFY_QUESTION}
        elif route == "outfit":
            first_id = ""
            if state.get("retrieved_items"):
                first_id = state["retrieved_items"][0].get("article_id", "")
            plan = {"action": "outfit", "article_id": first_id}
        else:  # respond
            plan = {"action": "respond"}

        plan["_db_confidence"] = round(confidence, 4)
        return {
            "current_plan": json.dumps(plan),
            "tool_calls": tool_calls + [{"router_decision": plan}],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_router_backend(
    config: dict,
    llm,
    memory,
    catalogue_df=None,
    prompt_template: str = "",
    format_items_brief: Callable = None,
    format_messages: Callable = None,
    parse_response: Callable = None,
) -> LLMRouterBackend | DistilBERTRouterBackend:
    provider = config.get("router", {}).get("provider", "llm")
    if provider == "distilbert":
        raw_path = config["router"].get("distilbert_model_path", "models/distilbert_router")
        model_path = Path(raw_path)
        if not model_path.is_absolute():
            # Resolve relative to repo root (two levels above this file: src/agents/router.py)
            model_path = Path(__file__).parent.parent.parent / raw_path
        if not model_path.exists():
            import warnings
            warnings.warn(
                f"[router] distilbert provider selected but model not found at {model_path}. "
                "Falling back to LLM router.",
                stacklevel=2,
            )
        else:
            return DistilBERTRouterBackend(model_path, catalogue_df)

    return LLMRouterBackend(llm, memory, prompt_template, format_items_brief, format_messages, parse_response)
