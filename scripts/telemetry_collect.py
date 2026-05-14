#!/usr/bin/env python3
"""Send 25-turn telemetry mix through POST /chat and save per-request timing.

Run with the API already listening on 127.0.0.1:8099.
Outputs scripts/telemetry_requests.json for use by telemetry_analyze.py.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Optional

import requests

API_BASE = "http://127.0.0.1:8099"
OUT_PATH = "scripts/telemetry_requests.json"
INTER_REQUEST_DELAY = 8  # seconds - keeps us at 7.5 req/min, safely under 10/min

records: list[dict] = []
seq = 0


def send(message: str, cid: Optional[str], label: str) -> dict:
    global seq
    seq += 1
    payload: dict = {"message": message}
    if cid:
        payload["conversation_id"] = cid

    sent_at = time.time()
    try:
        resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        received_at = time.time()
        print(f"  [{seq:02d}] {label}: ERROR {exc}", flush=True)
        rec = {
            "seq": seq, "label": label, "message": message,
            "conversation_id": cid, "sent_at": sent_at, "received_at": received_at,
            "duration_ms": round((received_at - sent_at) * 1000),
            "error": str(exc),
        }
        records.append(rec)
        return rec

    received_at = time.time()
    duration_ms = round((received_at - sent_at) * 1000)
    returned_cid = data.get("conversation_id", cid)
    action = data.get("routing", {}).get("action", "?")
    preview = (data.get("response") or "")[:80].replace("\n", " ")

    print(
        f"  [{seq:02d}] {label:<14} cid={returned_cid[:8]}  "
        f"action={action:<15} {duration_ms:>5}ms  {preview!r}",
        flush=True,
    )

    rec = {
        "seq": seq, "label": label, "message": message,
        "conversation_id": returned_cid,
        "sent_at": sent_at, "received_at": received_at,
        "duration_ms": duration_ms,
        "action": action,
        "response_preview": preview,
    }
    records.append(rec)
    return rec


def run(message: str, cid: Optional[str] = None, label: str = "") -> dict:
    rec = send(message, cid, label)
    time.sleep(INTER_REQUEST_DELAY)
    return rec


def main() -> None:
    print(f"Sending telemetry mix to {API_BASE}  (8s inter-request gap)\n", flush=True)

    # -- Group 1: Simple searches (8 messages) --------------------------------
    print("-- Simple searches --", flush=True)
    r_s1 = run("show me blue dresses",                      label="S1-search")
    r_s2 = run("I want a summer top",                       label="S2-search")
    r_s3 = run("looking for a leather jacket",              label="S3-search")
    r_s4 = run("black trousers please",                     label="S4-search")
    r_s5 = run("casual white trainers",                     label="S5-search")
    r_s6 = run("evening gown for a wedding",                label="S6-search")
    r_s7 = run("dark wash denim jeans",                     label="S7-search")
    r_s8 = run("cosy chunky knit jumper",                   label="S8-search")

    # -- Group 2: Refinements (5 messages - continue conversations) ----------─
    print("\n-- Refinements --", flush=True)
    run("make them more formal please",                     cid=r_s1.get("conversation_id"), label="R1-refine")
    run("something cheaper, under thirty pounds",           cid=r_s2.get("conversation_id"), label="R2-refine")
    run("do you have something lighter for summer",         cid=r_s3.get("conversation_id"), label="R3-refine")
    run("in a lighter wash instead",                        cid=r_s7.get("conversation_id"), label="R4-refine")
    run("show me it in an oversized style",                 cid=r_s8.get("conversation_id"), label="R5-refine")

    # -- Group 3: Outfit requests (4 messages) --------------------------------
    print("\n-- Outfit requests --", flush=True)
    run("put together a complete outfit for a job interview",        label="O1-outfit")
    run("full casual beach day outfit please",                       label="O2-outfit")
    run("what would you style with a midi skirt for a date night",   label="O3-outfit")
    run("garden party outfit, smart casual please",                  label="O4-outfit")

    # -- Group 4: Comparisons (3 messages) ------------------------------------
    print("\n-- Comparisons --", flush=True)
    run("compare silk dresses versus cotton dresses for summer",     label="C1-compare")
    run("which is better for a petite frame, midi or maxi skirt",   label="C2-compare")
    run("compare leather jackets versus denim jackets for autumn",   label="C3-compare")

    # -- Group 5: Out-of-catalogue (2 messages) ------------------------------─
    print("\n-- OOC --", flush=True)
    run("what is the weather like today in London",                  label="OOC1")
    run("who are you and what can you help me with",                 label="OOC2")

    # -- Group 6: Negations (2 messages) --------------------------------------
    print("\n-- Negations --", flush=True)
    run("show me dresses but nothing red",                           label="N1-negate")
    run("I want loungewear but absolutely no pyjamas",               label="N2-negate")

    # -- Group 7: Multi-turn chain (5 turns on one conversation_id) ----------─
    print("\n-- Multi-turn chain (5 turns) --", flush=True)
    mt1 = run("show me midi skirts",                                 label="MT1-multi")
    mt2 = run("I like these but need something more formal for the office",
              cid=mt1.get("conversation_id"),                        label="MT2-multi")
    run("do you have these in navy or charcoal grey",
        cid=mt1.get("conversation_id"),                              label="MT3-multi")
    run("which of these has the best quality fabric",
        cid=mt1.get("conversation_id"),                              label="MT4-multi")
    run("now show me blouses that would pair well with these skirts",
        cid=mt1.get("conversation_id"),                              label="MT5-multi")

    # -- Save ----------------------------------------------------------------─
    with open(OUT_PATH, "w") as fh:
        json.dump(records, fh, indent=2)

    total = len(records)
    errors = sum(1 for r in records if "error" in r)
    print(f"\nOK {total} requests sent ({errors} errors).  Saved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
