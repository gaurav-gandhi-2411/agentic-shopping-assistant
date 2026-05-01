#!/usr/bin/env python3
"""
Phase 2 Day 2.5 — Generate clarify/search contrastive training pairs.

These pairs teach the model the actual semantic boundary:
  SEARCH  = enough signal to formulate a product query (item/category, style, occasion, trend)
  CLARIFY = so underspecified that any search would be a random guess

Covers 4 failure mode patterns from OOD diagnostic:
  1. "I need X for Y" — 30 pairs
  2. "Looking for X" — 30 pairs
  3. Gift/occasion intent — 20 pairs
  4. Meta/trend queries  — 20 pairs

Output: data/router_training_v2_contrastive.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "data" / "router_training_v2_contrastive.jsonl"

# All examples: first-turn state (la=none, items=0) since that's where failures occur.
# Format: (query, route)

PAIRS: list[tuple[str, str]] = [
    # ------------------------------------------------------------------ #
    # PATTERN 1 — "I need [X] for [Y]"
    # Search: item/category + context = enough to search
    # Clarify: no item, pure vague goal
    # ------------------------------------------------------------------ #

    # SEARCH — occasion/context that implies a searchable category
    ("I need new clothes for my new job",                      "search"),
    ("I need something for a job interview",                   "search"),
    ("I need a dress for a wedding reception",                 "search"),
    ("I need warm clothes for a ski trip",                     "search"),
    ("I need something casual for the weekend",                "search"),
    ("I need office wear for my internship",                   "search"),
    ("I need gym clothes for my fitness routine",              "search"),
    ("I need comfortable clothes for traveling",               "search"),
    ("I need a jacket for autumn weather",                     "search"),
    ("I need summer clothes for my holiday",                   "search"),
    ("I need smart casual clothes for a dinner out",           "search"),
    ("I need a formal outfit for a black tie event",           "search"),
    ("I need loungewear for working from home",                "search"),
    ("I need running gear for my marathon training",           "search"),
    ("I need evening wear for a charity gala",                 "search"),

    # CLARIFY — no item, no category, just a vague social/relational context
    ("I need something for my sister",                         "clarify"),
    ("I need a gift for my colleague",                         "clarify"),
    ("I need help finding the right thing",                    "clarify"),
    ("I need something for next weekend",                      "clarify"),
    ("I need fashion advice",                                  "clarify"),
    ("I need something for a gift",                            "clarify"),
    ("I need help with what to wear",                          "clarify"),
    ("I need something special but I'm not sure what",         "clarify"),
    ("I need to get something for an occasion coming up",      "clarify"),
    ("I need help choosing something for them",                "clarify"),
    ("I need something but don't know where to start",         "clarify"),
    ("I need to buy something but I don't know what",          "clarify"),
    ("I need something for my partner",                        "clarify"),
    ("I need to get them a treat but have no idea what",       "clarify"),
    ("I need to pick something out but I'm completely lost",   "clarify"),

    # ------------------------------------------------------------------ #
    # PATTERN 2 — "Looking for X"
    # Search: item/style/trend specified
    # Clarify: no item, no style, or fully open-ended
    # ------------------------------------------------------------------ #

    # SEARCH
    ("Looking for a summer dress under knee length",           "search"),
    ("Looking for running shoes for marathon training",        "search"),
    ("Looking for a warm jacket for winter",                   "search"),
    ("Looking for casual jeans for everyday wear",             "search"),
    ("Looking for work attire that's professional but comfortable", "search"),
    ("Looking for something black and elegant for a dinner party", "search"),
    ("Looking for athletic wear for yoga",                     "search"),
    ("Looking for a trench coat under a hundred pounds",       "search"),
    ("Looking for prints and patterns for spring",             "search"),
    ("Looking for minimalist everyday basics",                 "search"),
    ("Looking for streetwear inspired pieces",                 "search"),
    ("Looking for a bold statement piece for a party",         "search"),
    ("Looking for trending items this season",                 "search"),
    ("Looking for what's popular in knitwear right now",       "search"),
    ("Looking for high waisted trousers in dark colors",       "search"),

    # CLARIFY
    ("Looking for something nice",                             "clarify"),
    ("Looking for a gift",                                     "clarify"),
    ("Looking for something for my mum",                       "clarify"),
    ("Looking for something fashionable, not sure where to start", "clarify"),
    ("Looking for something to wear but unsure what",          "clarify"),
    ("Looking for anything that might suit me",                "clarify"),
    ("Looking for something for a special occasion",           "clarify"),
    ("Looking for a gift for someone who has everything",      "clarify"),
    ("Looking for something different",                        "clarify"),
    ("Looking for the right thing but not sure what I want",   "clarify"),
    ("Looking for ideas",                                      "clarify"),
    ("Looking for a surprise gift with no restrictions",       "clarify"),
    ("Looking for something perfect for them",                 "clarify"),
    ("Looking for whatever you'd recommend",                   "clarify"),
    ("Looking for something I'll know when I see it",          "clarify"),

    # ------------------------------------------------------------------ #
    # PATTERN 3 — Gift / occasion intent
    # Search: style or demographic signal present → searchable
    # Clarify: pure occasion/recipient with no item/style signal
    # ------------------------------------------------------------------ #

    # SEARCH — enough signal (style, demographic, category) to issue a query
    ("A gift for my sister who loves boho style",              "search"),
    ("Something for my girlfriend who likes minimalist fashion", "search"),
    ("A birthday gift for a teenage girl into streetwear",     "search"),
    ("Get something smart for a business event",               "search"),
    ("Something summery for my friend's birthday",             "search"),
    ("Holiday gifts in the accessories category",              "search"),
    ("A gift for someone who loves vintage clothing",          "search"),
    ("Something casual and cool as a birthday surprise",       "search"),
    ("Get something trendy for a teenager",                    "search"),
    ("Anniversary gift for someone who loves sustainable fashion", "search"),

    # CLARIFY — pure occasion/recipient, no item or style
    ("Shopping for my mom's anniversary",                      "clarify"),
    ("A gift for my boss",                                     "clarify"),
    ("Something nice for my friend's birthday",                "clarify"),
    ("A present for someone special",                          "clarify"),
    ("Get something for my colleague's farewell",              "clarify"),
    ("A gift for someone I don't know very well",              "clarify"),
    ("Birthday shopping, need ideas",                          "clarify"),
    ("A surprise gift, no hints given",                        "clarify"),
    ("Something my partner will love",                         "clarify"),
    ("A gift without knowing what they'd want",                "clarify"),

    # ------------------------------------------------------------------ #
    # PATTERN 4 — Meta / trend queries
    # Search: trend browse, popularity, style discovery = searchable
    # Clarify: no item, no style, pure advice-seeking or ambiguous intent
    # ------------------------------------------------------------------ #

    # SEARCH — trend/popularity browse is a valid search intent
    ("What's trending right now in fashion",                   "search"),
    ("Show me what other people are wearing",                  "search"),
    ("What's popular this season",                             "search"),
    ("Show me the latest arrivals",                            "search"),
    ("What styles are in right now",                           "search"),
    ("I want to look professional but not boring",             "search"),
    ("What's hot right now for spring",                        "search"),
    ("Show me what's new this week",                           "search"),
    ("What are people wearing to work these days",             "search"),
    ("I want to look stylish and put together",                "search"),
    ("Show me current street style",                           "search"),
    ("What are the most popular items right now",              "search"),
    ("I want to dress like I know fashion",                    "search"),
    ("What styles should I be wearing this year",              "search"),
    ("Show me what's fashionable in casual wear",              "search"),

    # CLARIFY — no searchable signal, genuine advice or opinion needed
    ("What should I wear tomorrow",                            "clarify"),
    ("What are my options",                                    "clarify"),
    ("What do you recommend",                                  "clarify"),
    ("Help me figure out my style",                            "clarify"),
    ("Can you give me fashion advice",                         "clarify"),
    ("Tell me what I should buy",                              "clarify"),
    ("What would look good on me",                             "clarify"),
    ("What's a good choice for me",                            "clarify"),
    ("Can you help me decide what to get",                     "clarify"),
    ("What should I be buying",                                "clarify"),
    ("I don't know what I want",                               "clarify"),
    ("Help me figure out what to wear to the party",           "clarify"),  # no items → need to clarify occasion/style
    ("What's the right thing to get",                          "clarify"),
    ("I need advice on what to wear",                          "clarify"),
    ("Help me choose something",                               "clarify"),
]


def main() -> None:
    records = []
    for i, (query, route) in enumerate(PAIRS):
        records.append({
            "query":           query,
            "last_action":     "none",
            "items_retrieved": 0,
            "active_filters":  {},
            "route":           route,
            "id":              f"contrast_{i:03d}",
            "source":          "contrastive",
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    dist = Counter(r["route"] for r in records)
    print(f"Wrote {len(records)} contrastive pairs -> {OUT}")
    print(f"  search:  {dist['search']}")
    print(f"  clarify: {dist['clarify']}")
    print(f"  search:clarify ratio = {dist['search']/dist['clarify']:.2f}")


if __name__ == "__main__":
    main()
