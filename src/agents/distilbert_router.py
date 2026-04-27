"""
DistilBERT-based 6-class router classifier.

Replaces the LLM router prompt with a fine-tuned classifier.
Interface matches the LLM router so graph.py can swap it in (Phase 3).

Label map (alphabetical):
    clarify=0  compare=1  filter=2  outfit=3  respond=4  search=5
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Alphabetical ordering — must match train_router_distilbert.py
LABEL_MAP: dict[str, int] = {
    "clarify": 0,
    "compare": 1,
    "filter":  2,
    "outfit":  3,
    "respond": 4,
    "search":  5,
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)


class DistilBERTRouter:
    """6-class router classifier built on distilbert-base-uncased.

    Inputs (encoded into a single string for the tokenizer):
      - User query
      - last_action context  (what the agent last did)
      - items_retrieved count (how many items are in state)
      - active_filters dict   (current catalogue facet filters)

    Output: route name + softmax confidence
    """

    BASE_MODEL = "distilbert-base-uncased"
    MAX_LENGTH = 256
    LABEL_MAP = LABEL_MAP
    ID2LABEL = ID2LABEL
    NUM_LABELS = NUM_LABELS

    def __init__(self, model_path: str | None = None) -> None:
        """
        Args:
            model_path: Path to a fine-tuned checkpoint directory.
                        If None, loads the base model (for training — no classification head yet).
        """
        source = model_path or self.BASE_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=self.NUM_LABELS,
            id2label=self.ID2LABEL,
            label2id=self.LABEL_MAP,
            ignore_mismatched_sizes=True,  # needed when loading base model for training
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def encode_input(
        query: str,
        last_action: str,
        items_retrieved: int,
        active_filters: dict,
    ) -> str:
        """Format all routing inputs into one string for the tokenizer.

        Example output:
          "[QUERY] Show me black dresses [CTX] last_action=none items=0 filters=none"
          "[QUERY] Compare those two [CTX] last_action=search items=5 filters=colour_group_name=Black"
        """
        filters_str = (
            ";".join(f"{k}={v}" for k, v in sorted(active_filters.items()))
            if active_filters
            else "none"
        )
        return (
            f"[QUERY] {query} "
            f"[CTX] last_action={last_action} items={items_retrieved} filters={filters_str}"
        )

    def route(
        self,
        query: str,
        last_action: str,
        items_retrieved: int,
        active_filters: dict,
    ) -> tuple[str, float]:
        """Predict the routing action for a single query + state.

        Returns:
            (route_name, confidence) where confidence is the softmax probability
            of the predicted class. Same return signature as the LLM router output
            for clean drop-in replacement in Phase 3.
        """
        text = self.encode_input(query, last_action, items_retrieved, active_filters)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.MAX_LENGTH,
            truncation=True,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(probs.argmax(dim=-1).item())
        confidence = float(probs[0, pred_id].item())
        return self.ID2LABEL[pred_id], confidence
