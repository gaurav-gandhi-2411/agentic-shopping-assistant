# V5 grounding by construction: pre-registered acceptance rules

Status: pre-registered BEFORE any implementation code (this file is the first commit of
`feat/grounded-rendering`). Thresholds below are not to be edited after results are seen; a change
needs a new commit that says why and keeps the old text visible.

Goal: the LLM must not be able to state a wrong price, size or stock fact.

## Design under test

1. Id-only references. The respond prompt lists each item with its id and tells the model to refer
   to an item only as `[item:ID]` and to cite its price only as `[price:ID]`. A deterministic
   renderer (`src/agents/grounded_render.py`) substitutes the catalogue record's display name and
   `price_inr` at render time.
2. Post-validator. After substitution every price / size / stock / discount / delivery claim in the
   free text is checked by the existing deterministic checker (`check()`, moved to
   `src/agents/claim_checker.py` so production can import it; the `evals/` tree is not in the
   image). A sentence with an unbacked claim is removed. A sentence that references an id outside
   the shown set (or a malformed token) is removed.
3. Fail closed. If validation raises, or every sentence is removed, the free text is replaced by a
   fixed template sentence built from the record (`grounded_fallback_reply`, or the look's
   template rationale for outfit rationales).
4. Applied at: `respond_node` (REST/non-streaming), the WS `pending_respond` stream (buffered,
   validated, then emitted), `clarify_node` question text, and outfit rationales.
5. Kill switch: env `GROUNDED_RENDERING` (`0`/`false`/`off`/`no` disables; read per call). With the
   switch off the prompt, the streamed frames and every code path are byte-identical to
   `release/wave1b-1`.

## Acceptance rules

(a) Adversarial corpus. >= 60 deterministic synthetic LLM replies built from fixture items (no
    LLM, no network): wrong price, wrong currency, invented size, "in stock" for a sold-out item,
    % off not in the record, id not shown, numeric hallucination in prose, and price formatting
    variants (`1,299`, `Rs. 1299`, `INR 1299`, `1.3k`). Every case lists the exact spans that must
    not survive. PASS = 0 ungrounded price/size/stock claims survive: no listed span in the output
    AND the independent oracle `check()` returns no violation on the output. Survivors are
    counted and listed.
(b) No-regression corpus. >= 30 deterministic clean replies (correct prices, denial echoes such as
    "I don't have size data", user-budget echo, id tokens, plain prose). PASS = every output equals
    the input with only `[item:ID]` / `[price:ID]` tokens substituted (exact string equality).
(c) Existing relevant suites unchanged and green (tests that exercise respond_node, chat/WS
    routes, rationale, the grounding checker, the canary, and the W10 mock harness). No existing
    test is edited except import paths for the moved checker.
(d) Latency. Validator overhead < 5 ms p95 over the union of corpora a+b (time of
    `ground_reply` only, `time.perf_counter`, after one warm-up pass).
(e) Live eval NOT run in this task. Command and metrics are fixed here for a later step:
    - set: the 150 prompts in `evals/fixtures/grounding_regular.json` + `grounding_adversarial.json`
      (prompt-set hash recorded by the harness), WS path, real agent, Groq `openai/gpt-oss-20b`
      only (`--pin-primary` removes the fallback chain: no gpt-oss-120b / qwen).
    - budget: <= 50 prompts per UTC day (3 days), >= 8 s between requests, only after that day's
      WS canary finished (~02:30 UTC), shared-key rules in CLAUDE.md apply; abort on any 429 or
      fallback-tier log line.
    - pre-registered metrics: ungrounded-claim rate (`check()` violations over runs with text)
      must be exactly 0 of 150, `measurement_valid` true; and the existing answer-quality metric
      (strict P@5 from `scripts/eval_strict.py --mode pipeline`, which is retrieval-side and so
      must be unchanged, plus the harness's own `n_runs_with_text` / empty-reply count) must not
      drop by more than 0.03 versus the pre-change run on the same prompts.
    Command: see "Live eval command" below.

## Kill-switch default rule (decided in advance)

Default ON iff (a), (b), (c) and (d) all pass on the offline evidence. Otherwise default OFF with
the switch documented. Passing the offline rules does not verify live model behaviour (the model
may ignore the `[item:ID]` instruction; the renderer then passes names through and the validator
still strips unbacked claims). The live eval in (e) is the pre-deploy gate; failing it means set
`GROUNDED_RENDERING=0` or revert.

## Live eval command (to be run by the orchestrator, not in this task)

```
python -m evals.grounding.run --llm groq --agent real --scratch <empty scratch dir> \
    --path ws --pin-primary --min-interval-s 8 --offset 0 --limit 50 --max-llm-calls 250
```
Repeat with `--offset 50` and `--offset 100` on the next two UTC days. For the baseline arm run the
same three slices with `GROUNDED_RENDERING=0`. The harness writes `reports/grounding_eval_*.json`
with commit SHA, prompt-set hash and counts.
