# ADR 0005 — Flywheel Ranking Blend

## Context
We instrument outfit interactions (look_shown, add_the_look, thumbs_up/down, add_single)
to build a conversion signal for (anchor_category, fill_category, occasion) pairings.

## Decision
Blend conversion signal into outfit candidate scoring as a transparent multiplicative boost:

    total_signals = add_the_look + thumbs_up + thumbs_down + add_single_only
    positive_rate = (add_the_look + thumbs_up) / total_signals
    final_score   = coherence_score × (1 + FLYWHEEL_ALPHA × positive_rate)

Constants: FLYWHEEL_ALPHA = 0.25, FLYWHEEL_MIN_SIGNALS = 10.

**add_single_only** counts events where a user bought the anchor item but did not add
the full look. This drags positive_rate down, penalising pairings the user implicitly
rejected, not just ones they explicitly thumbed down.

Cold-start (< 10 signals): boost = 0.0 (rule-based scoring only).
Warm (>= 10 signals): boost scales from 0% to +25% as positive_rate -> 1.0.

The formula and constants are in source (src/flywheel/stats.py) — not a black box.

## Consequences
- First-mover advantage: brands using this system accumulate proprietary pairing signal
  that compounds over time.
- Distinction is keyword-heuristic, not a trained classifier (see slots.py fabric scoring).
  ADR note: if haldi/sangeet distinction needs improvement, a lightweight text classifier
  can replace the keyword check in a future iteration without changing the blend formula.

## Alternatives
- Collaborative filtering: ruled out — too data-hungry for cold start; would become a
  black box. Transparent re-rank is sufficient and explainable to brands.
- Separate ranking model: same objection. Additive boost is easily auditable.
