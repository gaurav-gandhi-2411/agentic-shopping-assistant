#!/usr/bin/env python3
"""
Phase 2 — Fine-tune DistilBERT on stateful training data (v2).

Training data: data/router_dataset_v2_{train,val,test}.jsonl  (987/123/123)
Output:        models/distilbert_router_v2/

Same architecture and hyperparameters as v1. Difference is the training data:
v2 covers all last_action priors (including respond+clarify), active_filters
in context, and items_retrieved values 1 and 3 — closing the train/serve skew.

Usage:
    python scripts/train_router_distilbert_v2.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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

DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models" / "distilbert_router_v2"
REPORTS   = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


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


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"f1_macro": macro_f1}


def main() -> None:
    t_start = time.time()

    print("Loading v2 datasets …")
    train_raw = load_jsonl(DATA_DIR / "router_dataset_v2_train.jsonl")
    val_raw   = load_jsonl(DATA_DIR / "router_dataset_v2_val.jsonl")
    test_raw  = load_jsonl(DATA_DIR / "router_dataset_v2_test.jsonl")
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

    print("\nTraining v2 …")
    trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nBest model saved -> {MODEL_DIR}")

    # -----------------------------------------------------------------------
    # Test-set evaluation
    # -----------------------------------------------------------------------
    print("\nEvaluating on v2 test set …")
    preds_out = trainer.predict(test_ds)
    pred_ids  = np.argmax(preds_out.predictions, axis=-1)
    true_ids  = preds_out.label_ids

    labels_ordered = [ID2LABEL[i] for i in range(NUM_LABELS)]
    pred_names = [ID2LABEL[i] for i in pred_ids]
    true_names = [ID2LABEL[i] for i in true_ids]

    acc      = float(np.mean(pred_ids == true_ids))
    macro_f1 = f1_score(true_ids, pred_ids, average="macro", zero_division=0)
    per_class = f1_score(true_ids, pred_ids, average=None,
                         labels=list(range(NUM_LABELS)), zero_division=0)
    cm = confusion_matrix(true_ids, pred_ids, labels=list(range(NUM_LABELS)))

    print(f"\nV2 test results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"\nPer-class F1:")
    for lbl, f1 in zip(labels_ordered, per_class):
        print(f"  {lbl:10s}: {f1:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print("          " + " ".join(f"{l[:5]:>7}" for l in labels_ordered))
    for i, row in enumerate(cm):
        print(f"  {labels_ordered[i]:10s}" + " ".join(f"{v:>7}" for v in row))

    print(f"\n{classification_report(true_ids, pred_ids, target_names=labels_ordered, zero_division=0)}")

    # -----------------------------------------------------------------------
    # Confidence distribution (for threshold sweep)
    # -----------------------------------------------------------------------
    import torch
    logits_t = torch.tensor(preds_out.predictions)
    probs    = torch.softmax(logits_t, dim=-1).numpy()
    max_probs = probs.max(axis=1).tolist()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    t_elapsed = time.time() - t_start
    result = {
        "model": "v2",
        "test_set": "v2_stateful",
        "n": len(test_raw),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {lbl: round(float(f1), 4) for lbl, f1 in zip(labels_ordered, per_class)},
        "confusion_matrix": {
            lbl: dict(zip(labels_ordered, [int(v) for v in row]))
            for lbl, row in zip(labels_ordered, cm)
        },
        "train_examples":       len(train_raw),
        "val_examples":         len(val_raw),
        "best_val_f1":          round(trainer.state.best_metric, 4),
        "best_checkpoint":      trainer.state.best_model_checkpoint,
        "training_time_sec":    round(t_elapsed, 1),
    }
    with open(REPORTS / "v2_test_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # -----------------------------------------------------------------------
    # Threshold sweep
    # -----------------------------------------------------------------------
    print("\nThreshold sweep …")
    thresholds = [round(t, 2) for t in np.arange(0.50, 0.95, 0.05)]
    sweep_rows = []
    for thresh in thresholds:
        kept_mask  = [p >= thresh for p in max_probs]
        n_kept     = sum(kept_mask)
        n_esc      = len(kept_mask) - n_kept
        kept_correct = sum(
            1 for i, kept in enumerate(kept_mask) if kept and pred_ids[i] == true_ids[i]
        )
        kept_acc = kept_correct / n_kept if n_kept > 0 else 0.0
        esc_rate = n_esc / len(kept_mask)
        # Effective accuracy: DB correct on kept, LLM perfect on escalated
        eff_acc = (kept_correct + n_esc) / len(kept_mask)
        sweep_rows.append({
            "threshold": thresh,
            "n_kept":    n_kept,
            "n_esc":     n_esc,
            "kept_pct":  round(n_kept / len(kept_mask), 3),
            "esc_rate":  round(esc_rate, 3),
            "kept_acc":  round(kept_acc, 4),
            "eff_acc":   round(eff_acc, 4),
        })
        print(f"  thresh={thresh:.2f}  kept={n_kept:3d}/{len(kept_mask)}  "
              f"kept_acc={kept_acc:.3f}  esc={esc_rate:.1%}  eff_acc={eff_acc:.3f}")

    # Save sweep
    import csv
    with open(REPORTS / "v2_threshold_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
        writer.writeheader()
        writer.writerows(sweep_rows)

    # -----------------------------------------------------------------------
    # V1 vs V2 comparison
    # -----------------------------------------------------------------------
    v1_results_path = REPORTS / "v1_on_stateful_test.json"
    if v1_results_path.exists():
        with open(v1_results_path) as f:
            v1_res = json.load(f)
        comparison = {
            "v1_macro_f1_fresh_test":    0.8345,  # from models/distilbert_router/training_log.json
            "v1_macro_f1_stateful_test": v1_res["macro_f1"],
            "v2_macro_f1_stateful_test": round(macro_f1, 4),
            "v1_accuracy_stateful":      v1_res["accuracy"],
            "v2_accuracy_stateful":      round(acc, 4),
            "per_class_comparison": {
                lbl: {
                    "v1_stateful": v1_res["per_class_f1"].get(lbl, 0),
                    "v2_stateful": round(float(pcf1), 4),
                }
                for lbl, pcf1 in zip(labels_ordered, per_class)
            },
        }
        with open(REPORTS / "v1_vs_v2_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nSaved comparison -> reports/v1_vs_v2_comparison.json")

    # -----------------------------------------------------------------------
    # Training log
    # -----------------------------------------------------------------------
    log = {
        "best_val_f1":         round(trainer.state.best_metric, 4),
        "best_checkpoint":     trainer.state.best_model_checkpoint,
        "test_f1_macro":       round(macro_f1, 4),
        "test_accuracy":       round(acc, 4),
        "num_train_epochs":    training_args.num_train_epochs,
        "train_examples":      len(train_raw),
        "val_examples":        len(val_raw),
        "test_examples":       len(test_raw),
        "training_time_sec":   round(t_elapsed, 1),
    }
    (MODEL_DIR / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\nTraining time: {t_elapsed/60:.1f} min")
    print(f"Saved: reports/v2_test_results.json")
    print(f"Saved: reports/v2_threshold_sweep.csv")
    print(f"Saved: models/distilbert_router_v2/")


if __name__ == "__main__":
    main()
