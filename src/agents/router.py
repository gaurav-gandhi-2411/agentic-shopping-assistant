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

    _CLARIFY_QUESTIONS: tuple[str, ...] = (
        "What kind of item are you looking for — a dress, top, jacket, or something else?",
        "Happy to help! What style or occasion are you shopping for today?",
        "Could you tell me more about what you have in mind?",
        "What are you hoping to find? I can search by style, occasion, or item type.",
        "I'd love to help narrow it down — what's the occasion or vibe you're going for?",
    )
    _MAX_LENGTH = 128

    def _clarify_question(self, user_query: str) -> str:
        if len(user_query.strip()) <= 5:
            return "I'm not sure I understood that — could you describe what you're looking for?"
        return self._CLARIFY_QUESTIONS[hash(user_query) % len(self._CLARIFY_QUESTIONS)]

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
            plan = {"action": "clarify", "question": self._clarify_question(user_query)}
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
# Cascade router backend
# ---------------------------------------------------------------------------

class CascadeRouterBackend:
    """DistilBERT-primary with LLM escalation for low-confidence predictions.

    For each query, DistilBERT runs first. If its softmax confidence is below
    `threshold`, the LLM router is called instead. Tracks escalation rate via
    the `stats()` method.
    """

    def __init__(
        self,
        distilbert: DistilBERTRouterBackend,
        llm: LLMRouterBackend,
        threshold: float = 0.70,
    ):
        self._db  = distilbert
        self._llm = llm
        self.threshold = threshold
        self.distilbert_count = 0
        self.escalation_count = 0

    def decide(self, state: dict) -> dict:
        result = self._db.decide(state)
        plan   = json.loads(result["current_plan"])
        confidence = plan.get("_db_confidence", 1.0)

        if confidence < self.threshold:
            self.escalation_count += 1
            llm_result = self._llm.decide(state)
            llm_plan   = json.loads(llm_result["current_plan"])
            llm_plan["_cascade_escalated"] = True
            llm_plan["_db_confidence"]     = confidence
            llm_result["current_plan"]     = json.dumps(llm_plan)
            # Patch tool_calls so the visualization can read _db_confidence
            for tc in llm_result.get("tool_calls", []):
                if "router_decision" in tc:
                    tc["router_decision"]["_cascade_escalated"] = True
                    tc["router_decision"]["_db_confidence"]     = confidence
            return llm_result

        self.distilbert_count += 1
        return result

    def stats(self) -> dict:
        total = self.distilbert_count + self.escalation_count
        return {
            "total":            total,
            "distilbert_only":  self.distilbert_count,
            "escalated_to_llm": self.escalation_count,
            "escalation_rate":  self.escalation_count / total if total > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _resolve_model_path(config: dict) -> Path:
    raw = config.get("router", {}).get("distilbert_model_path", "models/distilbert_router")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).parent.parent.parent / raw
    return p


def get_router_backend(
    config: dict,
    llm,
    memory,
    catalogue_df=None,
    prompt_template: str = "",
    format_items_brief: Callable = None,
    format_messages: Callable = None,
    parse_response: Callable = None,
) -> "LLMRouterBackend | DistilBERTRouterBackend | CascadeRouterBackend":
    provider = config.get("router", {}).get("provider", "llm")

    if provider in ("distilbert", "cascade"):
        model_path = _resolve_model_path(config)
        if not model_path.exists():
            import warnings
            warnings.warn(
                f"[router] {provider!r} provider selected but model not found at {model_path}. "
                "Falling back to LLM router.",
                stacklevel=2,
            )
        else:
            db_backend = DistilBERTRouterBackend(model_path, catalogue_df)
            if provider == "distilbert":
                return db_backend
            # cascade: wrap DistilBERT with LLM fallback
            llm_backend = LLMRouterBackend(
                llm, memory, prompt_template,
                format_items_brief, format_messages, parse_response,
            )
            threshold = config.get("router", {}).get("cascade_threshold", 0.70)
            return CascadeRouterBackend(db_backend, llm_backend, threshold)

    return LLMRouterBackend(llm, memory, prompt_template, format_items_brief, format_messages, parse_response)
