#!/usr/bin/env python3
"""
Fine-tune DistilBERT as a 6-class router classifier.

Training data: data/router_dataset_{train,val}.jsonl
Output:        models/distilbert_router/  (best checkpoint by val macro-F1)

Usage:
    python scripts/train_router_distilbert.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.distilbert_router import (
    ID2LABEL,
    LABEL_MAP,
    NUM_LABELS,
    DistilBERTRouter,
)

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "distilbert_router"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def to_hf_dataset(examples: list[dict]) -> Dataset:
    return Dataset.from_list([
        {
            "text": DistilBERTRouter.encode_input(
                ex["query"],
                ex["last_action"],
                ex["items_retrieved"],
                ex["active_filters"],
            ),
            "label": LABEL_MAP[ex["route"]],
        }
        for ex in examples
    ])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"f1_macro": macro_f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading datasets …")
    train_raw = load_jsonl(DATA_DIR / "router_dataset_train.jsonl")
    val_raw   = load_jsonl(DATA_DIR / "router_dataset_val.jsonl")
    test_raw  = load_jsonl(DATA_DIR / "router_dataset_test.jsonl")

    print(f"  train={len(train_raw)}  val={len(val_raw)}  test={len(test_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(DistilBERTRouter.BASE_MODEL)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            max_length=DistilBERTRouter.MAX_LENGTH,
            truncation=True,
            padding=False,
        )

    print("Tokenizing …")
    train_ds = to_hf_dataset(train_raw).map(tokenize, batched=True, remove_columns=["text"])
    val_ds   = to_hf_dataset(val_raw).map(tokenize, batched=True, remove_columns=["text"])
    test_ds  = to_hf_dataset(test_raw).map(tokenize, batched=True, remove_columns=["text"])

    print("Loading base model …")
    model = AutoModelForSequenceClassification.from_pretrained(
        DistilBERTRouter.BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        # Optimisation
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=8,
        weight_decay=0.01,
        warmup_ratio=0.1,
        # Evaluation + checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        # Reproducibility + logging
        seed=42,
        data_seed=42,
        report_to="none",
        logging_steps=10,
        logging_dir=str(MODEL_DIR / "logs"),
    )

    collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("\nTraining …")
    trainer.train()

    # Save best model + tokenizer
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nBest model saved -> {MODEL_DIR}")

    # Full test-set evaluation using trainer (for consistency with training loop)
    print("\nEvaluating on test set …")
    test_result = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(f"  Test macro F1: {test_result['test_f1_macro']:.4f}")

    # Detailed per-class report
    preds_out = trainer.predict(test_ds)
    pred_ids = np.argmax(preds_out.predictions, axis=-1)
    true_ids = preds_out.label_ids
    target_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    print("\nPer-class report (test):")
    print(classification_report(true_ids, pred_ids, target_names=target_names, zero_division=0))

    # Save training log for reference
    log_path = MODEL_DIR / "training_log.json"
    log = {
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "test_f1_macro": test_result["test_f1_macro"],
        "num_train_epochs": training_args.num_train_epochs,
        "train_examples": len(train_raw),
        "val_examples": len(val_raw),
        "test_examples": len(test_raw),
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nTraining log -> {log_path}")


if __name__ == "__main__":
    main()
