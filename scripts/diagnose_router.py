#!/usr/bin/env python3
"""
Diagnose DistilBERT router failure modes.

Steps:
  1. Predict on test set (37) and train set (294) with per-example confidence
  2. Full 6x6 confusion matrix with per-error details
  3. Confidence distribution: correct vs incorrect predictions
  4. Cross-router disagreement categorisation (7 cases from router_comparison.md)
  5. Failure type counts (A/B/C/D) and ranked recommendations
  6. Output: reports/router_diagnosis.md

Usage:
    python scripts/diagnose_router.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.distilbert_router import LABEL_MAP, ID2LABEL, NUM_LABELS, DistilBERTRouter

DATA_DIR    = ROOT / "data"
MODEL_DIR   = ROOT / "models" / "distilbert_router"
REPORTS_DIR = ROOT / "reports"

LABELS = [ID2LABEL[i] for i in range(NUM_LABELS)]


# ---------------------------------------------------------------------------
# Data + model helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def predict_with_confidence(
    examples: list[dict],
    router: DistilBERTRouter,
) -> list[tuple[str, float, dict]]:
    """Return (pred_label, confidence, all_softmax_probs) for each example."""
    results = []
    with torch.no_grad():
        for ex in examples:
            text = DistilBERTRouter.encode_input(
                ex["query"],
                ex["last_action"],
                ex["items_retrieved"],
                ex["active_filters"],
            )
            inputs = router.tokenizer(
                text,
                return_tensors="pt",
                max_length=DistilBERTRouter.MAX_LENGTH,
                truncation=True,
                padding=True,
            ).to(router.device)
            inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
            logits = router.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(probs.argmax().item())
            pred_label = ID2LABEL[pred_id]
            confidence = float(probs[pred_id].item())
            all_probs = {ID2LABEL[i]: float(probs[i].item()) for i in range(NUM_LABELS)}
            results.append((pred_label, confidence, all_probs))
    return results


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def build_confusion_matrix(
    examples: list[dict],
    preds_with_conf: list[tuple[str, float, dict]],
) -> np.ndarray:
    cm = np.zeros((NUM_LABELS, NUM_LABELS), dtype=int)
    for ex, (pred, _, _) in zip(examples, preds_with_conf):
        true_id = LABEL_MAP[ex["route"]]
        pred_id = LABEL_MAP[pred]
        cm[true_id][pred_id] += 1
    return cm


def get_errors(
    examples: list[dict],
    preds_with_conf: list[tuple[str, float, dict]],
) -> list[dict]:
    errors = []
    for ex, (pred, conf, all_probs) in zip(examples, preds_with_conf):
        if pred != ex["route"]:
            errors.append({
                "id": ex.get("id", "?"),
                "query": ex["query"],
                "last_action": ex["last_action"],
                "items_retrieved": ex["items_retrieved"],
                "active_filters": ex["active_filters"],
                "true": ex["route"],
                "pred": pred,
                "confidence": conf,
                "all_probs": all_probs,
                "source": ex.get("source", "?"),
            })
    return errors


def confidence_stats(
    examples: list[dict],
    preds_with_conf: list[tuple[str, float, dict]],
) -> dict:
    correct_confs = []
    wrong_confs   = []
    for ex, (pred, conf, _) in zip(examples, preds_with_conf):
        if pred == ex["route"]:
            correct_confs.append(conf)
        else:
            wrong_confs.append(conf)
    return {
        "correct_count":  len(correct_confs),
        "wrong_count":    len(wrong_confs),
        "correct_median": float(np.median(correct_confs)) if correct_confs else 0,
        "correct_p25":    float(np.percentile(correct_confs, 25)) if correct_confs else 0,
        "correct_min":    float(np.min(correct_confs)) if correct_confs else 0,
        "wrong_median":   float(np.median(wrong_confs)) if wrong_confs else 0,
        "wrong_p25":      float(np.percentile(wrong_confs, 25)) if wrong_confs else 0,
        "wrong_max":      float(np.max(wrong_confs)) if wrong_confs else 0,
        "high_conf_errors": sum(1 for c in wrong_confs if c > 0.80),
        "low_conf_errors":  sum(1 for c in wrong_confs if c <= 0.60),
    }


# ---------------------------------------------------------------------------
# Failure type taxonomy
# ---------------------------------------------------------------------------

# Hand-labelled: (id, type, rationale)
# A = vague/ambiguous query (hard for any router)
# B = state-conditional (LLM ignored state, DB advantage)
# C = out-of-vocabulary item (respond should fire, not search)
# D = surface-form confusion (DB misread surface cues)
FAILURE_TAXONOMY: dict[str, tuple[str, str]] = {
    # DB misclassifications on test set
    "para_050_3": ("A", "Gift query ('something stylish for a loved one') looks like search; clarify requires recognising vague intent"),
    "para_052_3": ("A", "Gift query with unknown preferences — DB predicted respond; Groq correctly predicted clarify"),
    "edge_002":   ("C", "'Face creams and moisturizers' = OOV category; DB predicted search instead of respond"),
    "edge_037":   ("D", "'Blazers but not the pinstriped ones from before' — negation + 'from before' confused DB into filter"),
    "seed_045":   ("D", "'Complete the look around item 2' — 'item 2' triggered filter; true label is outfit"),
    # Groq errors where DB was correct (DistilBERT advantage)
    "edge_007":   ("C", "DB correctly predicted respond for 'Do you have earphones?' (OOV); Groq incorrectly predicted search"),
    "para_005_0": ("B", "DB correctly predicted search (no prior items); Groq wrongly predicted outfit"),
    "para_039_2": ("B", "DB correctly predicted filter (5 items from prior search); Groq wrongly predicted search"),
    "para_005_4": ("B", "DB correctly predicted search (no prior items); Groq wrongly predicted clarify"),
}


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def fmt_cm(cm: np.ndarray) -> str:
    """Format confusion matrix as fixed-width text block."""
    col_w  = 9
    hdr    = " " * 10 + "".join(f"{l:>{col_w}}" for l in LABELS)
    rows   = [hdr, " " * 10 + "-" * (col_w * NUM_LABELS)]
    for i, true_label in enumerate(LABELS):
        row = f"{true_label:<10}" + "".join(f"{cm[i][j]:>{col_w}d}" for j in range(NUM_LABELS))
        rows.append(row)
    return "\n".join(rows)


def build_report(
    test_examples:  list[dict],
    test_preds:     list[tuple[str, float, dict]],
    train_examples: list[dict],
    train_preds:    list[tuple[str, float, dict]],
) -> str:
    test_cm    = build_confusion_matrix(test_examples, test_preds)
    test_err   = get_errors(test_examples, test_preds)
    test_stats = confidence_stats(test_examples, test_preds)
    train_err  = get_errors(train_examples, train_preds)
    train_stats = confidence_stats(train_examples, train_preds)

    test_acc  = (len(test_examples) - len(test_err))  / len(test_examples)
    train_acc = (len(train_examples) - len(train_err)) / len(train_examples)

    lines: list[str] = []

    lines += [
        "# DistilBERT Router — Failure Mode Diagnosis",
        "",
        f"**Test set:** {len(test_examples)} examples | "
        f"**Errors:** {len(test_err)} | **Accuracy:** {test_acc:.1%}",
        f"**Train set:** {len(train_examples)} examples | "
        f"**Errors:** {len(train_err)} | **Accuracy:** {train_acc:.1%}",
        "",
    ]

    # ---- Section 1: Confusion matrix ----------------------------------------
    lines += [
        "## 1. Confusion Matrix (test set, n=37)",
        "",
        "Rows = true label, Columns = predicted label.",
        "",
        "```",
        fmt_cm(test_cm),
        "```",
        "",
    ]

    # Per-class precision/recall from CM
    lines += ["### Per-class summary", ""]
    lines += ["| Class | Support | TP | FP | FN | Precision | Recall |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for i, label in enumerate(LABELS):
        support = int(test_cm[i].sum())
        tp = int(test_cm[i][i])
        fp = int(test_cm[:, i].sum()) - tp
        fn = support - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / support if support > 0 else 0.0
        lines.append(f"| {label} | {support} | {tp} | {fp} | {fn} | {prec:.2f} | {rec:.2f} |")
    lines.append("")

    # ---- Section 2: Misclassifications (test) --------------------------------
    lines += [
        "## 2. Misclassifications — Test Set",
        "",
        f"**{len(test_err)} errors out of {len(test_examples)}:**",
        "",
    ]
    for e in test_err:
        runner_up_label = sorted(e["all_probs"], key=e["all_probs"].get, reverse=True)[1]
        runner_up_prob  = e["all_probs"][runner_up_label]
        lines += [
            f"### [{e['id']}] true=`{e['true']}` → predicted=`{e['pred']}` (conf={e['confidence']:.2%})",
            f"- **Query:** \"{e['query']}\"",
            f"- **Context:** last_action=`{e['last_action']}` | items={e['items_retrieved']} | filters={e['active_filters']}",
            f"- **Encoded input:** `{DistilBERTRouter.encode_input(e['query'], e['last_action'], e['items_retrieved'], e['active_filters'])}`",
            f"- **Runner-up:** `{runner_up_label}` ({runner_up_prob:.2%})",
            f"- **All probs:** " + " | ".join(f"{k}={v:.2%}" for k, v in sorted(e['all_probs'].items(), key=lambda x: -x[1])),
            f"- **Source:** {e['source']}",
            "",
        ]

    # ---- Section 3: Confidence distribution ----------------------------------
    lines += [
        "## 3. Confidence Distribution",
        "",
        "### Test set",
        "",
        f"| Group | Count | Median conf | p25 conf |",
        f"|---|---|---|---|",
        f"| Correct predictions | {test_stats['correct_count']} | {test_stats['correct_median']:.2%} | {test_stats['correct_p25']:.2%} |",
        f"| Incorrect predictions | {test_stats['wrong_count']} | {test_stats['wrong_median']:.2%} | {test_stats['wrong_p25']:.2%} |",
        "",
        f"- High-confidence errors (conf > 80%): **{test_stats['high_conf_errors']}**",
        f"- Low-confidence errors (conf ≤ 60%): **{test_stats['low_conf_errors']}**",
        "",
        "### Train set",
        "",
        f"| Group | Count | Median conf | p25 conf |",
        f"|---|---|---|---|",
        f"| Correct predictions | {train_stats['correct_count']} | {train_stats['correct_median']:.2%} | {train_stats['correct_p25']:.2%} |",
        f"| Incorrect predictions | {train_stats['wrong_count']} | {train_stats['wrong_median']:.2%} | {train_stats['wrong_p25']:.2%} |",
        "",
        f"- High-confidence train errors (conf > 80%): **{train_stats['high_conf_errors']}**",
        f"- Low-confidence train errors (conf ≤ 60%): **{train_stats['low_conf_errors']}**",
        "",
        f"> **Interpretation:** If train errors are low-confidence and test errors are high-confidence,",
        f"> the model learned the training distribution but is over-confident on novel patterns.",
        f"> If most errors are high-confidence on both, the decision boundary is wrong.",
        "",
    ]

    # ---- Section 4: Train set errors -----------------------------------------
    lines += [
        "## 4. Train Set Errors",
        "",
        f"**{len(train_err)} errors out of {len(train_examples)}** (train accuracy: {train_acc:.1%})",
        "",
    ]
    if train_err:
        lines += ["| id | query | true | predicted | confidence |"]
        lines += ["|---|---|---|---|---|"]
        for e in train_err:
            q = e["query"][:60].replace("|", "/")
            lines.append(f"| {e['id']} | {q} | {e['true']} | {e['pred']} | {e['confidence']:.2%} |")
        lines.append("")
        # Classes that appear in train errors
        from collections import Counter
        train_err_classes = Counter(e["true"] for e in train_err)
        lines.append("**Train errors by true class:**")
        for cls, cnt in train_err_classes.most_common():
            lines.append(f"- `{cls}`: {cnt} errors")
        lines.append("")
    else:
        lines += ["No train errors — model fits training data perfectly (check for overfitting).", ""]

    # ---- Section 5: Cross-router disagreement analysis -----------------------
    lines += [
        "## 5. Cross-Router Disagreement Analysis",
        "",
        "Source: `reports/router_comparison.md` (7 disagreements between DistilBERT and Groq LLM).",
        "",
        "| id | query | true | DB pred | Groq pred | DB correct? | Failure type |",
        "|---|---|---|---|---|---|---|",
    ]

    # The 7 known disagreements (from router_comparison.md)
    disagreements = [
        {"id": "para_052_3", "query": "I'm lost – can you help me decide on a gift for someone whose preferences I'm not aware of?",
         "last_action": "none", "items": 0, "true": "clarify", "db": "respond", "groq": "clarify"},
        {"id": "edge_007",   "query": "Do you have earphones or headphones?",
         "last_action": "none", "items": 0, "true": "respond", "db": "respond", "groq": "search"},
        {"id": "para_005_0", "query": "Can you help me choose an outfit for a job interview?",
         "last_action": "none", "items": 0, "true": "search",  "db": "search",  "groq": "outfit"},
        {"id": "para_039_2", "query": "Show me Ladieswear items",
         "last_action": "search", "items": 5, "true": "filter", "db": "filter", "groq": "search"},
        {"id": "para_005_4", "query": "Help me pick a suitable outfit for an upcoming job interview.",
         "last_action": "none", "items": 0, "true": "search",  "db": "search",  "groq": "clarify"},
        {"id": "edge_037",   "query": "Blazers but not the pinstriped ones from before",
         "last_action": "search", "items": 5, "true": "search", "db": "filter", "groq": "search"},
        {"id": "seed_045",   "query": "Complete the look around item 2",
         "last_action": "search", "items": 5, "true": "outfit", "db": "filter", "groq": "outfit"},
    ]

    for d in disagreements:
        db_ok  = "YES" if d["db"] == d["true"] else "NO"
        ftype, _ = FAILURE_TAXONOMY.get(d["id"], ("?", ""))
        q = d["query"][:55].replace("|", "/")
        lines.append(
            f"| {d['id']} | {q} | {d['true']} | {d['db']} | {d['groq']} | {db_ok} | {ftype} |"
        )
    lines.append("")

    lines += ["### Disagreement detail", ""]
    for d in disagreements:
        db_ok = d["db"] == d["true"]
        groq_ok = d["groq"] == d["true"]
        winner = ("DistilBERT" if db_ok and not groq_ok
                  else "Groq" if groq_ok and not db_ok
                  else "Both" if db_ok and groq_ok
                  else "Neither")
        ftype, rationale = FAILURE_TAXONOMY.get(d["id"], ("?", ""))
        lines += [
            f"**[{d['id']}]** true=`{d['true']}` | DB=`{d['db']}` | Groq=`{d['groq']}` | Winner: **{winner}** | Type **{ftype}**",
            f"> \"{d['query']}\" (last_action={d['last_action']}, items={d['items']})",
            f"> {rationale}",
            "",
        ]

    # ---- Section 6: Failure type counts -------------------------------------
    from collections import Counter

    # Classify all DB errors (test set errors + DB-wrong disagreements)
    db_error_ids = {e["id"] for e in test_err}
    type_counts: Counter = Counter()
    for eid in db_error_ids:
        ftype, _ = FAILURE_TAXONOMY.get(eid, ("?", "unknown"))
        type_counts[ftype] += 1

    lines += [
        "## 6. Failure Type Counts (DistilBERT test errors)",
        "",
        "| Type | Count | Description |",
        "|---|---|---|",
        f"| A | {type_counts.get('A', 0)} | Vague/ambiguous query — hard for any router |",
        f"| B | {type_counts.get('B', 0)} | State-conditional — LLM ignored context (DistilBERT advantage) |",
        f"| C | {type_counts.get('C', 0)} | Out-of-vocabulary item — respond should fire, not search |",
        f"| D | {type_counts.get('D', 0)} | Surface-form confusion — negation/reference misdirected classifier |",
        f"| ? | {type_counts.get('?', 0)} | Unclassified |",
        "",
    ]

    # ---- Section 7: Encoding inconsistency note -----------------------------
    lines += [
        "## 7. Encoding Format Inconsistency (Bug)",
        "",
        "Training uses `DistilBERTRouter.encode_input()` which produces:",
        "```",
        "[QUERY] <query> [CTX] last_action=<x> items=<n> filters=<f>",
        "```",
        "But `scripts/eval_router_classifier.py` uses a different format:",
        "```",
        "query: <query> | last_action: <x> | items: <n> | filters: <f>",
        "```",
        "This script uses the **correct training-time format**. "
        "Any metric differences from the previous eval report may reflect this fix.",
        "",
    ]

    # ---- Section 8: Recommendations -----------------------------------------
    lines += [
        "## 8. Recommendations (ranked by expected impact)",
        "",
        "### 1. Add targeted clarify training examples (HIGH impact)",
        "",
        "**Problem:** Only 3 clarify examples in test set; 2 misclassified (recall=0.33).",
        "Training data has very few gift-intent / vague-preference queries.",
        "The model confuses 'stylish for a loved one' (clarify) with search, and",
        "'gift for someone whose preferences I'm not aware of' (clarify) with respond.",
        "",
        "**Fix:** Add 15–20 clarify examples covering:",
        "- Gift buying with unknown recipient preferences",
        "- Open-ended 'surprise me' requests",
        "- Highly ambiguous style requests with no clear category",
        "",
        "**Expected gain:** clarify recall 0.33 → ~0.80; macro F1 +3–5 pts.",
        "",
        "### 2. Improve OOV (out-of-catalogue) detection (MEDIUM impact)",
        "",
        "**Problem:** `edge_002` ('Show me face creams') → predicted search instead of respond.",
        "The OOC detector in `graph.py` catches obvious non-clothing terms (electronics, pets),",
        "but cosmetics/beauty is borderline and slips through. The router then predicts search",
        "instead of respond.",
        "",
        "**Fix options (pick one):**",
        "- Add 5–8 more cosmetics/beauty OOC examples to the training set with route=respond.",
        "- Expand the OOC keyword list in `_detect_ooc()` to include beauty/cosmetics terms.",
        "  (Simpler and doesn't require retraining.)",
        "",
        "**Expected gain:** respond precision/recall improve; eliminates a class of silent failures.",
        "",
        "### 3. Explicit state-feature tokens in input encoding (MEDIUM impact)",
        "",
        "**Problem:** `edge_037` ('Blazers but not the pinstriped ones from before') → predicted filter",
        "despite true=search. `seed_045` ('Complete the look around item 2') → predicted filter",
        "despite true=outfit.",
        "",
        "Root cause: The encoded input format buries state (items=5) in a compact string.",
        "The model uses surface cues ('not' → filter, 'item 2' → filter) instead of context.",
        "",
        "**Fix:** Strengthen the state encoding. Replace:",
        "```",
        "[CTX] last_action=search items=5 filters=none",
        "```",
        "with discrete tokens that DistilBERT can anchor on:",
        "```",
        "[CTX] [LAST_SEARCH] [ITEMS_SOME] [NO_FILTER]",
        "```",
        "or add explicit flags: `has_results=yes last_was_search=yes`.",
        "Retrain after encoding change.",
        "",
        "**Expected gain:** D-type errors (surface confusion) drop; filter false positives fall.",
        "",
        "### Summary table",
        "",
        "| Rank | Intervention | Errors addressed | Effort | Expected F1 gain |",
        "|---|---|---|---|---|",
        "| 1 | More clarify training data (15–20 examples) | A-type (2 errors) | Low | +3–5 pts |",
        "| 2 | OOC keyword expansion or respond training data | C-type (1 error) | Very low | +1–2 pts |",
        "| 3 | Stronger state encoding + retrain | D-type (2 errors) | Medium | +2–3 pts |",
        "",
        "> All three interventions together could bring macro F1 from 0.83 to ~0.90+",
        "> without changing the base model or requiring significantly more data.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data ...")
    test_examples  = load_jsonl(DATA_DIR / "router_dataset_test.jsonl")
    train_examples = load_jsonl(DATA_DIR / "router_dataset_train.jsonl")
    print(f"  test={len(test_examples)}  train={len(train_examples)}")

    print("Loading DistilBERT model ...")
    router = DistilBERTRouter(model_path=str(MODEL_DIR))
    print(f"  Device: {router.device}")

    print("Running predictions on test set ...")
    test_preds = predict_with_confidence(test_examples, router)

    print("Running predictions on train set ...")
    train_preds = predict_with_confidence(train_examples, router)

    # --- Console summary ---
    test_errors  = get_errors(test_examples, test_preds)
    train_errors = get_errors(train_examples, train_preds)
    test_stats   = confidence_stats(test_examples, test_preds)

    print(f"\nTest  errors: {len(test_errors)}/{len(test_examples)} "
          f"(acc={1 - len(test_errors)/len(test_examples):.1%})")
    print(f"Train errors: {len(train_errors)}/{len(train_examples)} "
          f"(acc={1 - len(train_errors)/len(train_examples):.1%})")

    print("\nTest confusion matrix (rows=true, cols=pred):")
    cm = build_confusion_matrix(test_examples, test_preds)
    header = " " * 10 + "".join(f"{l:>9}" for l in LABELS)
    print(header)
    for i, label in enumerate(LABELS):
        row = f"{label:<10}" + "".join(f"{cm[i][j]:>9d}" for j in range(NUM_LABELS))
        print(row)

    print("\nTest misclassifications:")
    for e in test_errors:
        print(f"  [{e['id']}] true={e['true']:8s} pred={e['pred']:8s}  conf={e['confidence']:.2%}  '{e['query'][:55]}'")

    print(f"\nConfidence: correct median={test_stats['correct_median']:.2%}  "
          f"wrong median={test_stats['wrong_median']:.2%}  "
          f"high-conf errors={test_stats['high_conf_errors']}")

    print("\nBuilding diagnosis report ...")
    report = build_report(test_examples, test_preds, train_examples, train_preds)

    out_path = REPORTS_DIR / "router_diagnosis.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Report -> {out_path}")


if __name__ == "__main__":
    main()
