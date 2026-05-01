#!/usr/bin/env python3
"""
Phase 2 Day 2.5 — Fine-tune DistilBERT on v3 dataset.

V3 adds 110 clarify/search contrastive pairs targeting the search→clarify
and clarify→search boundary failures identified in OOD diagnostic.

Output: models/distilbert_router_v3/
"""
from __future__ import annotations

import json
import sys
import time
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

from src.agents.distilbert_router import DistilBERTRouter, ID2LABEL, LABEL_MAP, NUM_LABELS

DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models" / "distilbert_router_v3"
REPORTS   = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def to_hf_dataset(examples: list[dict]) -> Dataset:
    return Dataset.from_list([
        {
            "text":  DistilBERTRouter.encode_input(
                ex["query"], ex["last_action"], ex["items_retrieved"], ex["active_filters"]
            ),
            "label": LABEL_MAP[ex["route"]],
        }
        for ex in examples
    ])


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro", zero_division=0)}


def main() -> None:
    t0 = time.time()

    print("Loading v3 datasets …")
    train_raw = load_jsonl(DATA_DIR / "router_dataset_v3_train.jsonl")
    val_raw   = load_jsonl(DATA_DIR / "router_dataset_v3_val.jsonl")
    test_raw  = load_jsonl(DATA_DIR / "router_dataset_v3_test.jsonl")
    print(f"  train={len(train_raw)}  val={len(val_raw)}  test={len(test_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(DistilBERTRouter.BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], max_length=DistilBERTRouter.MAX_LENGTH, truncation=True, padding=False)

    print("Tokenizing …")
    train_ds = to_hf_dataset(train_raw).map(tokenize, batched=True, remove_columns=["text"])
    val_ds   = to_hf_dataset(val_raw).map(tokenize,   batched=True, remove_columns=["text"])
    test_ds  = to_hf_dataset(test_raw).map(tokenize,  batched=True, remove_columns=["text"])

    print("Loading base DistilBERT …")
    model = AutoModelForSequenceClassification.from_pretrained(
        DistilBERTRouter.BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=8,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        seed=42,
        data_seed=42,
        report_to="none",
        logging_steps=10,
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

    print("\nTraining v3 …")
    trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nBest model saved -> {MODEL_DIR}")

    # Test-set evaluation
    print("\nEvaluating on v3 test set …")
    preds_out = trainer.predict(test_ds)
    pred_ids  = np.argmax(preds_out.predictions, axis=-1)
    true_ids  = preds_out.label_ids

    labels_ordered = [ID2LABEL[i] for i in range(NUM_LABELS)]
    acc      = float(np.mean(pred_ids == true_ids))
    macro_f1 = f1_score(true_ids, pred_ids, average="macro", zero_division=0)
    per_class = f1_score(true_ids, pred_ids, average=None, labels=list(range(NUM_LABELS)), zero_division=0)

    print(f"\nV3 test results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"\nPer-class F1:")
    for lbl, f1 in zip(labels_ordered, per_class):
        print(f"  {lbl:10s}: {f1:.4f}")
    print(f"\n{classification_report(true_ids, pred_ids, target_names=labels_ordered, zero_division=0)}")

    # Save training log
    log = {
        "best_val_f1":       round(trainer.state.best_metric, 4),
        "best_checkpoint":   trainer.state.best_model_checkpoint,
        "test_f1_macro":     round(macro_f1, 4),
        "test_accuracy":     round(acc, 4),
        "num_train_epochs":  training_args.num_train_epochs,
        "train_examples":    len(train_raw),
        "val_examples":      len(val_raw),
        "test_examples":     len(test_raw),
        "training_time_sec": round(time.time() - t0, 1),
        "per_class_f1":      {lbl: round(float(f), 4) for lbl, f in zip(labels_ordered, per_class)},
    }
    (MODEL_DIR / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nTraining log saved -> {MODEL_DIR / 'training_log.json'}")


if __name__ == "__main__":
    main()
