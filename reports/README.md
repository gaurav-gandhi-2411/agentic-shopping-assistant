# Eval Reports

Two permanent reports are committed here:

- `eval_baseline_groq.md` / `.json` — 31/32 reference run (Groq + OpenRouter, April 2026). ST1 was the single failure; serves as the pre-CIELAB baseline.
- `eval_latest_ollama.md` / `.json` — 31/32 verification run (local Ollama, April 2026) confirming the CIELAB fix resolves ST1. N2 failure is a local-quantized-model regression only.

All other `eval_results_*` files are intermediate runs; they are gitignored and stay local only.
To generate a new report: `python scripts/eval_harness.py --provider groq`
