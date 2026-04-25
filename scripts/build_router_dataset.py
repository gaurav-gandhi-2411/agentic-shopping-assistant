#!/usr/bin/env python3
"""
Build router classification dataset for DistilBERT fine-tuning.

Steps
-----
1. 53 hand-labeled seed examples (6 routes)
2. Groq paraphrase expansion — 5 paraphrases per seed (llama-3.1-8b-instant)
3. 50 hand-crafted edge cases (OOC, multi-intent, context-dep, negation, typos)
4. Stratified 80 / 10 / 10 split
5. Quality checks (distribution, Jaccard similarity, per-class samples)
6. Write reports/router_dataset_card.md

Output files
------------
  data/router_dataset.jsonl
  data/router_dataset_train.jsonl
  data/router_dataset_val.jsonl
  data/router_dataset_test.jsonl
  reports/router_dataset_card.md

Usage
-----
  python scripts/build_router_dataset.py
  python scripts/build_router_dataset.py --dry-run   # skip Groq, use placeholders
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.1-8b-instant"
PARAPHRASES_PER_SEED = 5
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
RANDOM_SEED = 42
INTER_REQUEST_DELAY = 1.2  # seconds between Groq calls (stay within TPM)
HIGH_JACCARD_THRESHOLD = 0.8

ROUTES = ["search", "compare", "filter", "clarify", "outfit", "respond"]

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

# ---------------------------------------------------------------------------
# Step 1 — Seed examples (53 hand-labeled)
# ---------------------------------------------------------------------------

SEED_EXAMPLES: list[dict] = [
    # ── search (16) ─────────────────────────────────────────────────────────
    # last_action="none" + items_retrieved=0 is the canonical new-search state.
    # last_action="filter" → search is required by state-machine rule 2.
    {"query": "Show me black dresses",                          "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "I want something in dark blue",                  "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "Find me a winter coat",                          "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "What swimwear do you have for summer?",          "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "Show me men's jackets please",                   "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "I need an outfit for a job interview",           "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "Looking for casual tops for the weekend",        "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "Find me something elegant for a date night",     "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "Cosy knitwear for autumn layering",              "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "What blazers do you have?",                      "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "Show me flowy summer dresses",                   "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "White trousers please",                          "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "I want minimalist neutral pieces",               "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "Find me something for a summer garden party",    "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    {"query": "Show me grey knitwear",                          "last_action": "none",   "items_retrieved": 0, "active_filters": {},                                "route": "search"},
    # State-machine: last_action=filter → must search next
    {"query": "Find new results with those filters applied",    "last_action": "filter", "items_retrieved": 0, "active_filters": {"index_group_name": "Divided"},   "route": "search"},

    # ── respond (12) ────────────────────────────────────────────────────────
    # Missing-data queries (no price/size/material in catalogue) → respond.
    # Acknowledgements after a successful retrieval → respond.
    # After last_action=compare → must respond (rule 1).
    {"query": "How much do these cost?",                        "last_action": "search",  "items_retrieved": 5, "active_filters": {},  "route": "respond"},
    {"query": "What material is this dress made from?",         "last_action": "search",  "items_retrieved": 5, "active_filters": {},  "route": "respond"},
    {"query": "Thanks, these look great!",                      "last_action": "search",  "items_retrieved": 5, "active_filters": {},  "route": "respond"},
    {"query": "Do you have this in a different size?",          "last_action": "search",  "items_retrieved": 5, "active_filters": {},  "route": "respond"},
    {"query": "Is this available for next-day delivery?",       "last_action": "search",  "items_retrieved": 5, "active_filters": {},  "route": "respond"},
    {"query": "Tell me more about the first item",              "last_action": "search",  "items_retrieved": 5, "active_filters": {},  "route": "respond"},
    {"query": "Which one would you recommend?",                 "last_action": "search",  "items_retrieved": 5, "active_filters": {},  "route": "respond"},
    {"query": "Can you tell me more about that last result?",   "last_action": "compare", "items_retrieved": 2, "active_filters": {},  "route": "respond"},
    {"query": "That's perfect, thank you",                      "last_action": "compare", "items_retrieved": 2, "active_filters": {},  "route": "respond"},
    {"query": "What's your return policy?",                     "last_action": "none",    "items_retrieved": 0, "active_filters": {},  "route": "respond"},
    {"query": "Is the blazer true to size?",                    "last_action": "search",  "items_retrieved": 5, "active_filters": {},  "route": "respond"},
    {"query": "Great, I'll take the first one",                 "last_action": "outfit",  "items_retrieved": 4, "active_filters": {},  "route": "respond"},

    # ── compare (7) ─────────────────────────────────────────────────────────
    # items_retrieved > 0 is a precondition for compare to be meaningful.
    {"query": "Can you compare the first two results?",         "last_action": "search", "items_retrieved": 5, "active_filters": {},                                 "route": "compare"},
    {"query": "What's the difference between them?",            "last_action": "search", "items_retrieved": 5, "active_filters": {},                                 "route": "compare"},
    {"query": "Compare item 1 and item 3 for me",               "last_action": "search", "items_retrieved": 5, "active_filters": {},                                 "route": "compare"},
    {"query": "Put those two side by side",                     "last_action": "search", "items_retrieved": 5, "active_filters": {},                                 "route": "compare"},
    {"query": "Which is better, the first or the second one?",  "last_action": "search", "items_retrieved": 5, "active_filters": {},                                 "route": "compare"},
    {"query": "Compare the last two from those results",        "last_action": "search", "items_retrieved": 5, "active_filters": {"colour_group_name": "Black"},      "route": "compare"},
    {"query": "How do the 2nd and 4th items compare?",          "last_action": "search", "items_retrieved": 5, "active_filters": {},                                 "route": "compare"},

    # ── filter (7) ──────────────────────────────────────────────────────────
    # Narrowing existing results by a catalogue facet.
    {"query": "Show only Divided items from that search",       "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "filter"},
    {"query": "Filter those results to black please",           "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "filter"},
    {"query": "Only show me dresses from those results",        "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "filter"},
    {"query": "Can you narrow that to women's only?",           "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "filter"},
    {"query": "Just show the Ladieswear options please",        "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "filter"},
    {"query": "Filter those results to dark blue",              "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "filter"},
    {"query": "Show only the blazers from what you found",      "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "filter"},

    # ── outfit (7) ──────────────────────────────────────────────────────────
    # Requires items_retrieved > 0 (seed item must exist).
    {"query": "Build me a complete outfit around the first item", "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "outfit"},
    {"query": "What goes with this dress?",                     "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "outfit"},
    {"query": "Style this with complementary pieces",           "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "outfit"},
    {"query": "Complete the look around item 2",                "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "outfit"},
    {"query": "What would pair well with the blazer?",          "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "outfit"},
    {"query": "Help me build an outfit around that jacket",     "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "outfit"},
    {"query": "Suggest a full look based on the first dress",   "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "outfit"},

    # ── clarify (4) ─────────────────────────────────────────────────────────
    # Gender-ambiguous gender-neutral items, completely incoherent input,
    # or gift queries with no product/gender context whatsoever.
    {"query": "I need a jacket for them",                       "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "clarify"},
    {"query": "Something nice for my partner, I'm not sure what", "last_action": "none", "items_retrieved": 0, "active_filters": {}, "route": "clarify"},
    {"query": "asdfghjkl zxcvbnm qwerty",                       "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "clarify"},
    {"query": "I need a gift but have no idea what they like",  "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "clarify"},
]

# ---------------------------------------------------------------------------
# Step 3 — Edge cases (50 hand-crafted)
# ---------------------------------------------------------------------------

EDGE_CASES: list[dict] = [
    # ── Out-of-catalogue queries → respond (8) ──────────────────────────────
    # OOC categories: pet supplies, electronics, beauty, home & furniture, food.
    # Agent short-circuits to respond with "not in catalogue" message.
    {"query": "Do you sell dog food?",                          "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "respond"},
    {"query": "Can I get a laptop here?",                       "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "respond"},
    {"query": "Show me face creams and moisturizers",           "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "respond"},
    {"query": "Do you carry bedding and pillows?",              "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "respond"},
    {"query": "I'd like to buy a new sofa",                     "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "respond"},
    {"query": "Do you sell coffee or any beverages?",           "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "respond"},
    {"query": "Show me some lipstick and nail polish options",  "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "respond"},
    {"query": "Do you have earphones or headphones?",           "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "respond"},

    # ── Multi-intent queries — router picks primary intent (10) ─────────────
    # With no items: always search first regardless of secondary intent.
    # With items: pick the most actionable next step.
    {"query": "Show me black dresses and compare them",                        "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "Find some trousers and filter to dark blue",                    "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "Compare those and then build an outfit",                        "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "compare"},
    {"query": "Show me blazers in both black and white",                       "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "I want dresses but also some casual tops",                      "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "Search for jumpers and style them with jeans",                  "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "Filter to black and compare with white ones",                   "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "filter"},
    {"query": "Find me dresses then build a beach holiday look",               "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "What's different between these and what would pair with them?", "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "compare"},
    {"query": "Give me the results filtered to Ladieswear only",               "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "filter"},

    # ── Context-dependent: identical query, different route based on state (10) ─
    # Pairs illustrate how items_retrieved / last_action changes the correct label.
    {"query": "More like these please",  "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "clarify"},   # no referent
    {"query": "More like these please",  "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "search"},    # refinement
    {"query": "Tell me about them",      "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "clarify"},   # no referent
    {"query": "Tell me about them",      "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "respond"},   # describe shown items
    {"query": "Can you compare them?",   "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "respond"},   # nothing to compare
    {"query": "Can you compare them?",   "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "compare"},   # compare shown items
    {"query": "Make an outfit with it",  "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},    # no seed item, search first
    {"query": "Make an outfit with it",  "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "outfit"},    # use first retrieved item
    {"query": "Only the dark ones",      "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},    # new search for dark items
    {"query": "Only the dark ones",      "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "filter"},    # filter existing results

    # ── Negation / exclusion patterns (10) ──────────────────────────────────
    # All route to search; the exclusion is handled in search_node downstream.
    {"query": "Dresses but not black ones",                       "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "Tops without any floral patterns",                 "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "Casual trousers, not jeans though",                "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "Show me coats except the black ones",              "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "Workwear but nothing too corporate or stiff",      "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "Summer outfits but no shorts",                     "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "Swimwear other than bikinis please",               "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "I don't want anything striped or checked",         "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "search"},
    {"query": "Show me knitwear but nothing too heavy or thick",  "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "Blazers but not the pinstriped ones from before",  "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "search"},

    # ── Misspellings and informal phrasings (7) ──────────────────────────────
    {"query": "shw me blck drssses",                    "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "i wanna see sum sweterz",                "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "somethin comfy 4 wearing at home",       "last_action": "none",   "items_retrieved": 0, "active_filters": {},  "route": "search"},
    {"query": "compre those 2 items pls",               "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "compare"},
    {"query": "wut goes w the dress",                   "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "outfit"},
    {"query": "thx looks gr8!!!",                       "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "respond"},
    {"query": "can u filter 2 jus da blak ones",        "last_action": "search", "items_retrieved": 5, "active_filters": {},  "route": "filter"},

    # ── Ambiguous → clarify (5) ──────────────────────────────────────────────
    # Gender ambiguous + gender-neutral item type with no prior gender context.
    # Completely underspecified or incoherent queries.
    {"query": "I need outerwear for them",                "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "clarify"},
    {"query": "Get something nice for my friend's birthday", "last_action": "none", "items_retrieved": 0, "active_filters": {}, "route": "clarify"},
    {"query": "I need a gift idea, no other constraints",  "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "clarify"},
    {"query": "xyz abc foo bar",                           "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "clarify"},
    {"query": "Something for someone special",             "last_action": "none", "items_retrieved": 0, "active_filters": {},  "route": "clarify"},
]


# ---------------------------------------------------------------------------
# Groq paraphrase generation (Step 2)
# ---------------------------------------------------------------------------

def generate_paraphrases(
    client: Groq | None,
    query: str,
    n: int = PARAPHRASES_PER_SEED,
    dry_run: bool = False,
) -> list[str]:
    """Return n paraphrases of query via Groq.  Falls back to placeholders on error."""
    if dry_run or client is None:
        return [f"{query} [dry-run variant {i + 1}]" for i in range(n)]

    prompt = (
        f"Generate {n} natural paraphrases of this user query for a fashion shopping assistant. "
        f"Keep the intent identical but vary phrasing, formality, and length. "
        f"Return ONLY a JSON array of strings — no explanation, no markdown fences.\n\n"
        f"Original: '{query}'"
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=420,
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[-2] if text.count("```") >= 2 else text.lstrip("`")
            start, end = text.find("["), text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                if isinstance(parsed, list) and parsed:
                    return [str(p).strip() for p in parsed[:n]]
        except Exception as exc:
            msg = str(exc).lower()
            if "rate_limit" in msg or "429" in msg or "too many" in msg:
                wait = 30 * (attempt + 1)
                print(f"\n  [rate-limit] backing off {wait}s …", file=sys.stderr, flush=True)
                time.sleep(wait)
            else:
                print(f"\n  [groq error] {exc}", file=sys.stderr, flush=True)
                break

    # Fallback so the pipeline never stalls
    print(f"\n  [fallback] using placeholder paraphrases for: {query!r}", file=sys.stderr)
    return [f"{query} (paraphrase {i + 1})" for i in range(n)]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def jaccard_sim(s1: str, s2: str) -> float:
    """Token-level Jaccard similarity (lowercased)."""
    t1, t2 = set(s1.lower().split()), set(s2.lower().split())
    if not t1 and not t2:
        return 1.0
    return len(t1 & t2) / len(t1 | t2)


def stratified_split(
    examples: list[dict],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split preserving per-route proportions."""
    rng = random.Random(seed)
    by_route: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_route[ex["route"]].append(ex)

    train, val, test = [], [], []
    for items in by_route.values():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))
        # Ensure test always gets at least 1 sample
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Step 5 — Quality checks
# ---------------------------------------------------------------------------

def run_quality_checks(
    all_examples: list[dict],
    train: list[dict],
    val: list[dict],
    test: list[dict],
    high_sim_pairs: list[tuple[str, str, float]],
) -> str:
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("QUALITY REPORT")
    lines.append("=" * 60)

    # Class distribution per split
    for name, split in [("FULL", all_examples), ("TRAIN", train), ("VAL", val), ("TEST", test)]:
        counts = Counter(ex["route"] for ex in split)
        lines.append(f"\n{name} distribution (n={len(split)}):")
        for route in ROUTES:
            lines.append(f"  {route:<10} {counts.get(route, 0):>4}")

    # Per-class samples (5 per class from full set)
    lines.append("\n" + "=" * 60)
    lines.append("SAMPLES (5 per class, from full dataset)")
    lines.append("=" * 60)
    rng = random.Random(RANDOM_SEED)
    by_route: dict[str, list[dict]] = defaultdict(list)
    for ex in all_examples:
        by_route[ex["route"]].append(ex)
    for route in ROUTES:
        items = by_route.get(route, [])
        sample = rng.sample(items, min(5, len(items)))
        lines.append(f"\n--- {route.upper()} ---")
        for ex in sample:
            lines.append(
                f"  [{ex.get('source', '?'):10}] "
                f"last={ex['last_action']:<7} items={ex['items_retrieved']} "
                f"| {ex['query']}"
            )

    # High-Jaccard pairs
    lines.append(f"\n{'=' * 60}")
    if high_sim_pairs:
        lines.append(
            f"HIGH-JACCARD PAIRS (>{HIGH_JACCARD_THRESHOLD}) — {len(high_sim_pairs)} flagged:"
        )
        for orig, para, sim in high_sim_pairs[:20]:
            lines.append(f"  sim={sim:.2f} | orig: {orig!r}")
            lines.append(f"          | para: {para!r}")
    else:
        lines.append("HIGH-JACCARD PAIRS: none flagged (all paraphrases sufficiently diverse).")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 6 — Dataset card
# ---------------------------------------------------------------------------

def write_dataset_card(
    all_examples: list[dict],
    train: list[dict],
    val: list[dict],
    test: list[dict],
    quality_report: str,
    card_path: Path,
    disk_bytes: int,
    elapsed_sec: float,
) -> None:
    counts = Counter(ex["route"] for ex in all_examples)
    rng = random.Random(RANDOM_SEED)
    by_route: dict[str, list[dict]] = defaultdict(list)
    for ex in all_examples:
        by_route[ex["route"]].append(ex)

    lines: list[str] = [
        "# Router Classification Dataset Card",
        "",
        "## Overview",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Total examples | {len(all_examples)} |",
        f"| Train / Val / Test | {len(train)} / {len(val)} / {len(test)} |",
        f"| Classes | {', '.join(ROUTES)} |",
        f"| Disk size | {disk_bytes / 1024:.1f} KB |",
        f"| Build time | {elapsed_sec:.0f}s |",
        "",
        "## Class Distribution",
        "",
        "| Route | Count | % |",
        "|-------|------:|--:|",
    ]
    for route in ROUTES:
        n = counts.get(route, 0)
        lines.append(f"| {route} | {n} | {100 * n / len(all_examples):.1f}% |")

    lines += [
        "",
        "## Generation Methodology",
        "",
        "1. **Seed examples** (53): hand-labeled, covering all 6 routes with realistic",
        "   `last_action`, `items_retrieved`, and `active_filters` context.",
        f"2. **Paraphrase expansion** ({PARAPHRASES_PER_SEED}× per seed via Groq `{GROQ_MODEL}`):",
        "   intent-preserving rewrites varying formality, length, and phrasing.",
        "3. **Edge cases** (50): hand-crafted hard cases — OOC queries, multi-intent,",
        "   context-dependent pairs, negation patterns, misspellings.",
        "",
        "## Known Limitations",
        "",
        "- Paraphrases are **synthetic** — real user queries may differ in distribution.",
        "- `clarify` class is intentionally small (~20 examples) reflecting its rarity in production.",
        "- Context features (`last_action`, `items_retrieved`) are **idealised** — the classifier",
        "  must generalise to intermediate states not captured here.",
        "- Vocabulary is H&M-catalogue specific; may not transfer to other fashion retailers.",
        "",
        "## Sample Examples",
        "",
    ]
    for route in ROUTES:
        items = by_route.get(route, [])
        sample = rng.sample(items, min(4, len(items)))
        lines.append(f"### `{route}`")
        lines.append("")
        for ex in sample:
            lines.append(
                f"- **query**: `{ex['query']}`  "
                f"last_action=`{ex['last_action']}` items_retrieved=`{ex['items_retrieved']}`"
            )
        lines.append("")

    lines += [
        "## Quality Checks",
        "",
        "```",
        quality_report,
        "```",
    ]

    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Groq API calls; insert placeholder paraphrases instead.",
    )
    args = parser.parse_args()

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key and not args.dry_run:
        print(
            "ERROR: GROQ_API_KEY not set. "
            "Either set the env var or pass --dry-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = Groq(api_key=groq_key) if groq_key else None
    t_start = time.time()

    print(f"Seeds: {len(SEED_EXAMPLES)}  |  Edge cases: {len(EDGE_CASES)}")
    print(f"Paraphrases per seed: {PARAPHRASES_PER_SEED}  (dry-run={args.dry_run})\n")

    # ------------------------------------------------------------------
    # Steps 1 + 2: seed originals + paraphrase expansion
    # ------------------------------------------------------------------
    all_examples: list[dict] = []
    high_sim_pairs: list[tuple[str, str, float]] = []

    for i, seed in enumerate(tqdm(SEED_EXAMPLES, desc="Paraphrasing seeds", unit="seed")):
        original_query = seed["query"]

        # Original seed
        all_examples.append({**seed, "id": f"seed_{i:03d}", "source": "seed"})

        # Paraphrases
        paraphrases = generate_paraphrases(client, original_query, dry_run=args.dry_run)
        for j, para in enumerate(paraphrases):
            sim = jaccard_sim(original_query, para)
            if sim > HIGH_JACCARD_THRESHOLD:
                high_sim_pairs.append((original_query, para, sim))
            all_examples.append({
                "query": para,
                "last_action": seed["last_action"],
                "items_retrieved": seed["items_retrieved"],
                "active_filters": seed["active_filters"],
                "route": seed["route"],
                "id": f"para_{i:03d}_{j}",
                "source": "paraphrase",
            })

        if not args.dry_run:
            time.sleep(INTER_REQUEST_DELAY)

    # ------------------------------------------------------------------
    # Step 3: edge cases
    # ------------------------------------------------------------------
    for i, edge in enumerate(EDGE_CASES):
        all_examples.append({**edge, "id": f"edge_{i:03d}", "source": "edge"})

    elapsed = time.time() - t_start
    print(f"\nTotal examples: {len(all_examples)}  |  Build time: {elapsed:.0f}s")

    # ------------------------------------------------------------------
    # Step 4: stratified split
    # ------------------------------------------------------------------
    train, val, test = stratified_split(all_examples)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    full_path = DATA_DIR / "router_dataset.jsonl"
    write_jsonl(full_path, all_examples)
    write_jsonl(DATA_DIR / "router_dataset_train.jsonl", train)
    write_jsonl(DATA_DIR / "router_dataset_val.jsonl", val)
    write_jsonl(DATA_DIR / "router_dataset_test.jsonl", test)

    disk_bytes = sum(
        (DATA_DIR / f"router_dataset{s}.jsonl").stat().st_size
        for s in ["", "_train", "_val", "_test"]
    )

    # ------------------------------------------------------------------
    # Steps 5 + 6: quality checks + dataset card
    # ------------------------------------------------------------------
    report = run_quality_checks(all_examples, train, val, test, high_sim_pairs)
    print(report)

    card_path = REPORTS_DIR / "router_dataset_card.md"
    write_dataset_card(all_examples, train, val, test, report, card_path, disk_bytes, elapsed)

    print(f"\nFiles written:")
    print(f"  {full_path}  ({len(all_examples)} lines)")
    print(f"  {DATA_DIR / 'router_dataset_train.jsonl'}  ({len(train)} lines)")
    print(f"  {DATA_DIR / 'router_dataset_val.jsonl'}    ({len(val)} lines)")
    print(f"  {DATA_DIR / 'router_dataset_test.jsonl'}   ({len(test)} lines)")
    print(f"  {card_path}")
    print(f"  Total disk: {disk_bytes / 1024:.1f} KB")


if __name__ == "__main__":
    main()
