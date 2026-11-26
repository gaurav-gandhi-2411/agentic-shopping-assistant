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

## AMENDMENT 1 (2026-11-26, architect decision): keep streaming with the switch on

Registered BEFORE the streaming code change. The text above is unchanged and stays in force except
where this amendment says otherwise.

Why: design item 4 buffered the whole WS reply until the model finished, then emitted it. That is a
user-visible regression (no typing effect; first text only after the full generation). The architect
asked to keep streaming with `GROUNDED_RENDERING` on.

### Streaming design (replaces "buffered, then emitted" for the WS `pending_respond` stream)

A deterministic incremental validator (`StreamGrounder`, `src/agents/grounded_stream.py`) sits
between the model's token stream and the WS token frames:

1. Tokens pass through as they arrive, but only up to the last whitespace (the word in progress is
   never emitted), so a keyword or tag split across chunks is never half-emitted.
2. A closed `[item:ID]` span with a known id is replaced by the record's display name and streaming
   continues. An open `[` is held until its `]` arrives.
3. The rest of a sentence is HELD from its first claim-like trigger until the sentence ends, then
   judged with exactly the batch rules (render, `check()`, unattributed-claim rule) and emitted in
   full or not at all. Triggers: any `[` that is not a rendered item span (a `[price:ID]`, an
   unknown id, an unsupported or malformed tag), any digit or currency symbol, the claim-word
   vocabulary (stock, size, sale, discount, price, delivery, pickup, half, free, ...), and any hit
   of `claim_checker._extract_claims` on the sentence so far.
4. A sentence ends where `claim_checker._sentences` ends it (it needs the next sentence's first
   capital, or a newline, or the end of the stream).
5. An unclosed span at the end of the stream is dropped with its sentence and logged as a
   structured event `grounded_unclosed_span` with a `count` field and no reply text.
6. If nothing survives, or the validator raises, the template sentence is appended (fail closed);
   after a validator error no further model text is emitted.
7. Frame types and fields are unchanged. Kill switch off: raw live tokens exactly as before.
8. The empty-stream retry path keeps using the batch `ground_reply` on the retry text.

### Exact guarantee (what can reach the user before validation)

- G1 (claims): no price, size, stock, discount or delivery claim, no raw `[price:..]`, unknown id,
  unsupported or malformed tag, and no text after the first trigger of a sentence reaches the
  user before that sentence has been validated. A sentence that fails validation contributes
  nothing from its first trigger onward.
- G2 (lead-in, NOT eliminated): the claim-free text of a sentence BEFORE its first trigger may be
  emitted before the sentence is validated. If the sentence then fails, the user keeps that
  lead-in fragment (for example "It is") and the rest is dropped. A multi-word claim can also leak
  its leading cue word(s) when the cue is not a trigger (for example "only" before "2 left"). The
  fragment states no checked claim; it is a cosmetic residual, counted by the corpus report.
- G3: what was emitted equals what is persisted as the assistant message.
- Not guaranteed: chunk-invariance of the output for failing sentences (where the lead-in ends can
  depend on chunk boundaries); trailing whitespace after a dropped sentence.

### Amended acceptance rules (added; (a)-(e) above stay)

(f) Streaming safety: over every single split position, and char-by-char, of every adversarial
    case, the concatenated stream output has 0 bad spans and `check()` returns no violation on it;
    every clean case streams to exactly its expected output at every split position; tags split at
    every position render identically.
(g) First frame before the end: with a fake stream that withholds its second half until the client
    has received a token frame, a clean reply delivers its first token frame before the stream ends.
(h) Validator-call latency stays under 5 ms p95 per call (batch call as before; the report adds the
    per-sentence judge call). Corpus report: adversarial survivors 0, clean changed 0.
(i) Each of the tests above is mutation-checked (mutant caught and restored).

### Amended live-eval metrics (not run in this task)

- ID-format compliance rate, from RAW model output (before rendering): item mentions written as
  `[item:ID]` divided by all item mentions (tokens plus plain-name mentions found by the checker's
  name matcher). Diagnostic, no pass/fail threshold registered; reported with n and per path.
- Side-by-side dump: 20 replies, same prompts, switch ON versus OFF, arms randomised per reply with
  a seeded shuffle (seed 42) and a separate key file, for blind judging.

## AMENDMENT 2 (2026-11-26, same day; registered before the code it describes)

First measurement of amendment 1's design on the corpora (7-character chunks): 0 survivors and 0
clean replies changed, but 69 of 73 adversarial replies leave a claim-free lead-in fragment on
screen (guarantee G2). The corpus is adversarial by construction (most put the claim mid-sentence),
so this overstates the live rate, which is unmeasured. Because a fragment such as "Gloss Wrap
Skirt is only" is visible damage, a second mode is added as an option, not a replacement:

- `GROUNDED_STREAM_MODE=sentence` (env, read per turn; default `token`, the amendment 1 behaviour)
  makes `StreamGrounder(hold_sentences=True)` emit a sentence only after it has been judged.
  G2 does not exist in that mode (0 fragments); the cost is sentence-sized frames (first text after
  the first sentence ends instead of after the first word).
- Acceptance (j): in sentence mode, for every adversarial and clean case at every split position,
  the output equals the batch `ground_reply` output (compared on whitespace-split words) and the
  fragment count is 0. A WS test shows token mode leaves the lead-in and sentence mode does not.
- Which mode ships as default is NOT decided here: the orchestrator should decide from the live
  blind dump (fragments visible in token mode versus latency in sentence mode).
