#!/usr/bin/env python3
"""
Evaluate the DistilBERT router classifier on the held-out test set and compare
against the Groq LLM router.

Outputs:
  reports/router_classifier_eval.md   — per-class metrics + CPU latency benchmark
  reports/charts/router_confusion.png — confusion matrix heatmap
  reports/router_comparison.md        — DistilBERT vs Groq LLM side-by-side

Usage:
  python scripts/eval_router_classifier.py
  GROQ_API_KEY=<key> python scripts/eval_router_classifier.py   # enable LLM comparison
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.distilbert_router import ID2LABEL, LABEL_MAP, NUM_LABELS

DATA_DIR    = ROOT / "data"
MODEL_DIR   = ROOT / "models" / "distilbert_router"
REPORTS_DIR = ROOT / "reports"
CHARTS_DIR  = REPORTS_DIR / "charts"

LABELS = [ID2LABEL[i] for i in range(NUM_LABELS)]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def encode_input(ex: dict) -> str:
    filters = json.dumps(ex.get("active_filters") or {})
    return (
        f"query: {ex['query']} | last_action: {ex.get('last_action', 'none')} "
        f"| items: {ex.get('items_retrieved', 0)} | filters: {filters}"
    )


# ---------------------------------------------------------------------------
# DistilBERT router
# ---------------------------------------------------------------------------

def load_distilbert(device: str):
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model = model.to(device)
    model.eval()
    return tokenizer, model


def predict_distilbert(examples: list[dict], tokenizer, model, device: str) -> list[str]:
    preds = []
    with torch.no_grad():
        for ex in examples:
            text = encode_input(ex)
            inputs = tokenizer(
                text, return_tensors="pt", max_length=128, truncation=True, padding=True
            ).to(device)
            logits = model(**inputs).logits
            preds.append(ID2LABEL[int(logits.argmax(-1).item())])
    return preds


def benchmark_cpu_latency(examples: list[dict], tokenizer, model, n: int = 100) -> dict:
    """Run n predictions on CPU, return median and p95 latency in ms."""
    model_cpu = model.to("cpu")
    model_cpu.eval()
    pool = examples * (n // len(examples) + 1)
    pool = pool[:n]
    times = []
    with torch.no_grad():
        for ex in pool:
            text = encode_input(ex)
            inputs = tokenizer(
                text, return_tensors="pt", max_length=128, truncation=True, padding=True
            )
            t0 = time.perf_counter()
            model_cpu(**inputs)
            times.append((time.perf_counter() - t0) * 1000)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "n": n,
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


# ---------------------------------------------------------------------------
# Groq LLM router
# ---------------------------------------------------------------------------

ROUTER_PROMPT_TEMPLATE = """\
You are a shopping assistant planner. Given the conversation so far and the latest user query,
decide the NEXT action. Respond with ONE of the following JSON objects and nothing else:

{{"action": "search", "query": "<search string>", "filters": {{}}}}
{{"action": "compare", "article_ids": ["<id1>", "<id2>"]}}
{{"action": "filter", "key": "<facet>", "value": "<value>"}}
{{"action": "outfit", "article_id": "<article_id>"}}
{{"action": "clarify", "question": "<clarification for the user>"}}
{{"action": "respond"}}

STRICT RULES — follow in order:
0. COMPARE PRIORITY (highest): If the user says "compare", "difference between", "vs" AND items_retrieved > 0 — output compare.
1. If last_action is "compare" — output respond.
2. If last_action is "filter" — output search.
3. If items_retrieved > 0 AND last_action is "search" — output respond.
4. Use "search" for a new information need.
5. Use "filter" to narrow results.
6. Use "compare" when user explicitly asks to compare.
7. Use "outfit" when user asks "what goes with", "style this with", "build an outfit".
8. Use "clarify" ONLY if query is completely incomprehensible or missing essential info.
9. NEVER repeat the same action twice in a row.

Last action taken: {last_action}
Items retrieved so far: {items_retrieved}
Current filters: {current_filters}
Latest user query: {user_query}
Recent conversation: (none)

Respond with ONLY the JSON object. No explanation."""


def _parse_action(raw: str) -> str | None:
    """Extract 'action' value from LLM JSON response."""
    import re
    m = re.search(r'"action"\s*:\s*"(\w+)"', raw or "")
    return m.group(1) if m else None


def call_groq_router(ex: dict, groq_client) -> tuple[str | None, float]:
    """Return (predicted_route, latency_ms). Returns (None, latency) on rate limit."""
    prompt = ROUTER_PROMPT_TEMPLATE.format(
        last_action=ex.get("last_action", "none"),
        items_retrieved=ex.get("items_retrieved", 0),
        current_filters=json.dumps(ex.get("active_filters") or {}),
        user_query=ex["query"],
    )
    t0 = time.perf_counter()
    try:
        raw = groq_client.generate(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000
        action = _parse_action(raw)
        return action, latency_ms
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        exc_str = str(exc)
        if "rate" in exc_str.lower() or "limit" in exc_str.lower() or "429" in exc_str:
            return None, latency_ms
        raise


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_classifier_report(
    examples: list[dict],
    preds: list[str],
    latency: dict,
    training_log: dict,
) -> str:
    true_labels = [ex["route"] for ex in examples]
    true_ids = [LABEL_MAP[l] for l in true_labels]
    pred_ids = [LABEL_MAP[p] for p in preds]

    macro_f1 = f1_score(true_ids, pred_ids, average="macro", zero_division=0)
    acc = sum(t == p for t, p in zip(true_ids, pred_ids)) / len(true_ids)
    report = classification_report(true_ids, pred_ids, target_names=LABELS, zero_division=0)

    lines = [
        "# DistilBERT Router Classifier — Evaluation Report",
        "",
        "## Training summary",
        "",
        f"| Item | Value |",
        f"|---|---|",
        f"| Base model | distilbert-base-uncased |",
        f"| Training examples | {training_log.get('train_examples', 294)} |",
        f"| Val examples | {training_log.get('val_examples', 37)} |",
        f"| Test examples | {training_log.get('test_examples', 37)} |",
        f"| Epochs | {training_log.get('num_train_epochs', 8)} |",
        f"| Best val macro F1 | {training_log.get('best_val_f1_macro', 0.846):.4f} (epoch {training_log.get('best_epoch', 7)}) |",
        f"| Training time | {training_log.get('train_runtime_seconds', 23.71):.1f}s on RTX 3070 Laptop GPU |",
        f"| Device | {training_log.get('device', 'cuda')} |",
        "",
        "## Test-set metrics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Accuracy | {acc:.4f} |",
        f"| Macro F1 | {macro_f1:.4f} |",
        "",
        "### Per-class breakdown",
        "",
        "```",
        report.strip(),
        "```",
        "",
        "### Confusion matrix",
        "",
        "![Confusion matrix](charts/router_confusion.png)",
        "",
        "## CPU latency benchmark",
        "",
        f"> n={latency['n']} predictions on CPU (production target)",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Median | {latency['median_ms']:.1f} ms |",
        f"| p95 | {latency['p95_ms']:.1f} ms |",
        f"| Min | {latency['min_ms']:.1f} ms |",
        f"| Max | {latency['max_ms']:.1f} ms |",
        "",
        "## Error analysis",
        "",
    ]

    # Misclassified examples
    errors = [(ex, p) for ex, p, t in zip(examples, preds, true_labels) if p != t]
    if errors:
        lines.append(f"**{len(errors)} misclassified examples out of {len(examples)}:**")
        lines.append("")
        lines.append("| id | query | true | predicted |")
        lines.append("|---|---|---|---|")
        for ex, pred in errors:
            q = ex["query"][:60].replace("|", "/")
            lines.append(f"| {ex.get('id','?')} | {q} | {ex['route']} | {pred} |")
    else:
        lines.append("No misclassified examples.")

    return "\n".join(lines)


def build_comparison_report(
    examples: list[dict],
    db_preds: list[str],
    groq_preds: list[str | None],
    db_latency: dict,
    groq_latencies: list[float],
    groq_skips: int,
) -> str:
    true_labels = [ex["route"] for ex in examples]
    true_ids = [LABEL_MAP[l] for l in true_labels]
    db_ids   = [LABEL_MAP[p] for p in db_preds]

    db_f1 = f1_score(true_ids, db_ids, average="macro", zero_division=0)

    groq_valid = [(p, l) for p, l in zip(groq_preds, groq_latencies) if p is not None]
    groq_covered = [ex for ex, p in zip(examples, groq_preds) if p is not None]
    groq_true_ids  = [LABEL_MAP[ex["route"]] for ex in groq_covered]
    groq_pred_ids  = [LABEL_MAP[p] for _, p in [(ex, p) for ex, p in zip(examples, groq_preds) if p is not None]]

    if groq_pred_ids:
        groq_f1 = f1_score(groq_true_ids, groq_pred_ids, average="macro", zero_division=0)
        groq_p50 = float(np.median([l for l in groq_latencies if l > 0]))
        groq_p95 = float(np.percentile([l for l in groq_latencies if l > 0], 95))
        groq_f1_str = f"{groq_f1:.4f} (on {len(groq_covered)}/{len(examples)} non-skipped)"
        groq_p50_str = f"{groq_p50:.0f} ms (API round-trip)"
        groq_p95_str = f"{groq_p95:.0f} ms"
    else:
        groq_f1_str = "N/A (all skipped)"
        groq_p50_str = "N/A"
        groq_p95_str = "N/A"

    lines = [
        "# Router Comparison: DistilBERT vs Groq LLM",
        "",
        "## Summary table",
        "",
        "| Metric | DistilBERT | Groq LLM (llama-3.1-8b-instant) |",
        "|---|---|---|",
        f"| Macro F1 on test | {db_f1:.4f} | {groq_f1_str} |",
        f"| Latency p50 | {db_latency['median_ms']:.1f} ms (CPU) | {groq_p50_str} |",
        f"| Latency p95 | {db_latency['p95_ms']:.1f} ms (CPU) | {groq_p95_str} |",
        f"| Cost per 1k requests | $0 (local inference) | ~$0.05–0.10 (API) |",
        f"| Rate-limit risk | None | High (TPD quota) |",
        f"| Deployment | Bundled in app | External API dependency |",
        f"| Cold-start | None (loaded once) | None (stateless API) |",
        f"| Groq calls skipped (rate limit) | — | {groq_skips} / {len(examples)} |",
        "",
        "## Methodology",
        "",
        "- Test set: 37 held-out examples from `data/router_dataset_test.jsonl`",
        "- LLM router called with same state fields (query, last_action, items_retrieved, active_filters)",
        "- No conversation history passed to LLM (single-turn evaluation; test examples are single-turn)",
        "- DistilBERT latency measured on CPU only (production deployment target)",
        "- Groq latency includes full API round-trip (network + inference)",
        "",
    ]

    # Disagreements
    disagreements = [
        (ex, db, groq)
        for ex, db, groq in zip(examples, db_preds, groq_preds)
        if groq is not None and db != groq
    ]
    lines.append(f"## Disagreement examples ({len(disagreements)} found)")
    lines.append("")

    if not disagreements:
        lines.append("No disagreements — both routers agree on all evaluated examples.")
    else:
        lines.append("Cases where DistilBERT and Groq LLM chose different routes:")
        lines.append("")
        lines.append("| id | query | true label | DistilBERT | Groq LLM | DB correct? |")
        lines.append("|---|---|---|---|---|---|")
        for ex, db, groq in disagreements[:10]:
            q = ex["query"][:55].replace("|", "/")
            true = ex["route"]
            db_ok = "yes" if db == true else "no"
            lines.append(f"| {ex.get('id','?')} | {q} | {true} | {db} | {groq} | {db_ok} |")

        lines.append("")
        lines.append("### Notable disagreements")
        lines.append("")
        for i, (ex, db, groq) in enumerate(disagreements[:5], 1):
            true = ex["route"]
            db_correct = db == true
            groq_correct = groq == true
            winner = "DistilBERT" if db_correct and not groq_correct else ("Groq" if groq_correct and not db_correct else "Neither" if not db_correct and not groq_correct else "Both")
            lines += [
                f"**{i}. `{ex.get('id','?')}`** — True: `{true}` | DB: `{db}` | Groq: `{groq}` | Winner: {winner}",
                f"> Query: \"{ex['query']}\" (last_action={ex.get('last_action','none')}, items={ex.get('items_retrieved',0)})",
                "",
            ]

    return "\n".join(lines)


def save_confusion_matrix(examples: list[dict], preds: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  matplotlib/seaborn not available — skipping confusion matrix PNG")
        return False

    true_ids = [LABEL_MAP[ex["route"]] for ex in examples]
    pred_ids = [LABEL_MAP[p] for p in preds]
    cm = confusion_matrix(true_ids, pred_ids, labels=list(range(NUM_LABELS)))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("DistilBERT Router — Confusion Matrix (test set, n=37)")
    plt.tight_layout()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = CHARTS_DIR / "router_confusion.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix -> {out}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading test set ...")
    test = load_jsonl(DATA_DIR / "router_dataset_test.jsonl")
    print(f"  {len(test)} examples")

    training_log = json.loads((MODEL_DIR / "training_log.json").read_text(encoding="utf-8"))

    print("Loading DistilBERT model ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_distilbert(device)
    print(f"  Device: {device}")

    # --- DistilBERT test predictions ---
    print("Running DistilBERT predictions ...")
    db_preds = predict_distilbert(test, tokenizer, model, device)

    # --- CPU latency benchmark ---
    print("Benchmarking CPU latency (n=100) ...")
    latency = benchmark_cpu_latency(test, tokenizer, model, n=100)
    print(f"  Median: {latency['median_ms']:.1f}ms  p95: {latency['p95_ms']:.1f}ms")

    # --- Confusion matrix ---
    print("Saving confusion matrix ...")
    save_confusion_matrix(test, db_preds)

    # --- Classifier eval report ---
    print("Writing classifier eval report ...")
    eval_md = build_classifier_report(test, db_preds, latency, training_log)
    eval_path = REPORTS_DIR / "router_classifier_eval.md"
    eval_path.write_text(eval_md, encoding="utf-8")
    print(f"  -> {eval_path}")

    # --- Groq LLM comparison ---
    groq_key = os.environ.get("GROQ_API_KEY")
    groq_preds: list[str | None] = [None] * len(test)
    groq_latencies: list[float] = [0.0] * len(test)
    groq_skips = 0

    if groq_key:
        print("Running Groq LLM router comparison ...")
        try:
            import groq as _groq_lib
            groq_client_raw = _groq_lib.Groq(api_key=groq_key)

            class _SimpleGroqClient:
                def generate(self, prompt):
                    resp = groq_client_raw.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=100,
                    )
                    return resp.choices[0].message.content

            groq_client = _SimpleGroqClient()

            for i, ex in enumerate(test):
                pred, lat = call_groq_router(ex, groq_client)
                groq_preds[i] = pred
                groq_latencies[i] = lat
                if pred is None:
                    groq_skips += 1
                    print(f"  [{i+1}/{len(test)}] SKIPPED (rate limit) — {ex.get('id','?')}")
                else:
                    print(f"  [{i+1}/{len(test)}] {ex.get('id','?')}: true={ex['route']} groq={pred} ({lat:.0f}ms)")
                time.sleep(0.3)  # stay under TPM
        except ImportError:
            print("  groq package not installed — skipping LLM comparison")
        except Exception as exc:
            print(f"  Groq client failed: {exc} — skipping LLM comparison")
    else:
        print("GROQ_API_KEY not set — LLM comparison will show N/A")

    # --- Comparison report ---
    print("Writing comparison report ...")
    cmp_md = build_comparison_report(
        test, db_preds, groq_preds, latency, groq_latencies, groq_skips
    )
    cmp_path = REPORTS_DIR / "router_comparison.md"
    cmp_path.write_text(cmp_md, encoding="utf-8")
    print(f"  -> {cmp_path}")

    # --- Final summary ---
    true_labels = [ex["route"] for ex in test]
    true_ids = [LABEL_MAP[l] for l in true_labels]
    db_ids   = [LABEL_MAP[p] for p in db_preds]
    macro_f1 = f1_score(true_ids, db_ids, average="macro", zero_division=0)

    print("\n" + "="*60)
    print("DONE")
    print(f"  Test macro F1:  {macro_f1:.4f}")
    print(f"  CPU latency:    {latency['median_ms']:.1f}ms median, {latency['p95_ms']:.1f}ms p95")
    print(f"  Groq skips:     {groq_skips}/{len(test)}")
    print(f"  Reports:        {REPORTS_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
