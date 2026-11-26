"""Run the V5 grounding corpora and print the pre-registered acceptance numbers.

    python scripts/grounding_corpus_report.py [--json out.json]

Rules (docs/grounding-v5-acceptance.md): (a) adversarial survivors == 0, (b) clean replies
unchanged apart from id substitution == all, (d) validator p95 < 5 ms. Deterministic, no LLM, no
network. Exit code 1 when any rule fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agents.claim_checker import check  # noqa: E402
from src.agents.grounded_render import ground_reply  # noqa: E402
from tests import _grounding_v5_corpus as corpus  # noqa: E402

P95_BUDGET_MS = 5.0


STREAM_CHUNK_CHARS = 7  # a typical few-token chunk; the tests also split at every position


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def evaluate_stream() -> dict[str, Any]:
    """Amendment 1 rules (f)/(h): StreamGrounder over both corpora, plus per-feed timing."""
    from src.agents.grounded_stream import StreamGrounder

    items = corpus.ITEMS
    survivors: list[str] = []
    with_fragment: list[str] = []
    for c in corpus.adversarial():
        g = StreamGrounder(items, user_text=c["user_text"])
        for ch in _chunks(c["reply"], STREAM_CHUNK_CHARS):
            g.feed(ch)
        g.finish()
        out = g.text
        if (
            [b for b in c["bad_spans"] if b in out]
            or check(out, items, user_text=c["user_text"])
            or [k for k in c["must_keep"] if k not in out]
        ):
            survivors.append(c["id"])
        if g.fragments:
            with_fragment.append(c["id"])
    hold_bad: list[str] = []  # amendment 2: sentence mode must equal the batch output, 0 fragments
    for c in corpus.adversarial() + corpus.clean():
        g = StreamGrounder(items, user_text=c["user_text"], hold_sentences=True)
        for ch in _chunks(c["reply"], STREAM_CHUNK_CHARS):
            g.feed(ch)
        g.finish()
        batch = ground_reply(c["reply"], items, user_text=c["user_text"]).text
        if g.text.split() != batch.split() or g.fragments:
            hold_bad.append(c["id"])
    changed: list[str] = []
    for c in corpus.clean():
        g = StreamGrounder(items, user_text=c["user_text"])
        for ch in _chunks(c["reply"], STREAM_CHUNK_CHARS):
            g.feed(ch)
        g.finish()
        if g.text != c["expected"]:
            changed.append(c["id"])
    times: list[float] = []
    for _ in range(3):
        for c in corpus.adversarial() + corpus.clean():
            g = StreamGrounder(items, user_text=c["user_text"])
            for ch in _chunks(c["reply"], STREAM_CHUNK_CHARS):
                t0 = time.perf_counter()
                g.feed(ch)
                times.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            g.finish()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p95 = times[int(0.95 * len(times)) - 1]
    return {
        "stream_chunk_chars": STREAM_CHUNK_CHARS,
        "stream_adversarial_survivors": len(survivors),
        "stream_survivor_ids": survivors,
        "stream_clean_changed": len(changed),
        "stream_clean_changed_ids": changed,
        "stream_adversarial_with_leadin_fragment": len(with_fragment),
        "stream_sentence_mode_mismatch_or_fragment": len(hold_bad),
        "stream_call_samples": len(times),
        "stream_call_ms_p50": round(times[len(times) // 2], 3),
        "stream_call_ms_p95": round(p95, 3),
        "stream_call_ms_max": round(times[-1], 3),
        "stream_ok": not survivors and not changed and not hold_bad and p95 < P95_BUDGET_MS,
    }


def evaluate() -> dict[str, Any]:
    """Score both corpora and time ground_reply; returns the report dict."""
    items = corpus.ITEMS
    adv, clean = corpus.adversarial(), corpus.clean()
    survivors: list[dict[str, Any]] = []
    for c in adv:
        out = ground_reply(c["reply"], items, user_text=c["user_text"]).text
        left = [b for b in c["bad_spans"] if b in out]
        oracle = [v.kind for v in check(out, items, user_text=c["user_text"])]
        lost = [k for k in c["must_keep"] if k not in out]
        if left or oracle or lost:
            survivors.append(
                {
                    "id": c["id"],
                    "reply": c["reply"],
                    "output": out,
                    "spans_left": left,
                    "oracle": oracle,
                    "keep_lost": lost,
                }
            )
    changed: list[dict[str, str]] = []
    for c in clean:
        out = ground_reply(c["reply"], items, user_text=c["user_text"]).text
        if out != c["expected"]:
            changed.append({"id": c["id"], "expected": c["expected"], "output": out})

    cases = [(c["reply"], c["user_text"]) for c in adv + clean]
    for reply, user in cases:  # warm-up
        ground_reply(reply, items, user_text=user)
    times: list[float] = []
    for reply, user in cases * 5:
        t0 = time.perf_counter()
        ground_reply(reply, items, user_text=user)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p95 = times[int(0.95 * len(times)) - 1]
    stream = evaluate_stream()
    return {
        **stream,
        "adversarial_n": len(adv),
        "adversarial_survivors": len(survivors),
        "survivor_detail": survivors,
        "clean_n": len(clean),
        "clean_changed": len(changed),
        "clean_changed_detail": changed,
        "latency_samples": len(times),
        "latency_ms_p50": round(times[len(times) // 2], 3),
        "latency_ms_p95": round(p95, 3),
        "latency_ms_max": round(times[-1], 3),
        "latency_budget_ms": P95_BUDGET_MS,
        "ok": not survivors and not changed and p95 < P95_BUDGET_MS and stream["stream_ok"],
    }


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    rep = evaluate()
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("_detail")}, indent=2))
    for s in rep["survivor_detail"]:
        print("SURVIVOR", json.dumps(s, ensure_ascii=False))
    for c in rep["clean_changed_detail"]:
        print("CLEAN CHANGED", json.dumps(c, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
