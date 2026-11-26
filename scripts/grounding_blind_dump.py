"""Side-by-side dump of grounded rendering ON versus OFF replies for blind judging.

    python scripts/grounding_blind_dump.py --on reports/<on>.json --off reports/<off>.json \
        --out-dir <dir> [--n 20] [--seed 42]

Inputs are two reports written by ``python -m evals.grounding.run`` with the same prompt set (one
run with GROUNDED_RENDERING on, one with it off). Writes, into --out-dir:

    blind_dump.md    prompt, the items shown, and "Reply A" / "Reply B" per sample (judge file)
    blind_dump.json  the same content, machine readable (no arm labels)
    blind_key.json   which arm is A and which is B for each sample (keep away from the judge)

Which arm is A is decided per sample by a seeded shuffle. Selection is a seeded sample of the
(id, path) pairs that have non-empty text in both arms. Deterministic: same inputs, same files.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

SEED = 42  # repo convention


def _runs(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (r["id"], r["path"]): r for r in report["runs"] if r.get("http_ok") and r["text"].strip()
    }


def build(
    on: dict[str, Any], off: dict[str, Any], n: int = 20, seed: int = SEED
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Return (samples without arm labels, key mapping sample id -> {"A": arm, "B": arm})."""
    if on.get("grounded_rendering") != "on" or off.get("grounded_rendering") != "off":
        raise ValueError("expected --on to be a switch-on report and --off a switch-off report")
    if on.get("prompt_set_hash") != off.get("prompt_set_hash"):
        raise ValueError("the two reports were not run on the same prompt set")
    a, b = _runs(on), _runs(off)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no (id, path) pair has text in both arms")
    rng = random.Random(seed)
    picked = sorted(rng.sample(common, min(n, len(common))))
    samples: list[dict[str, Any]] = []
    key: dict[str, dict[str, str]] = {}
    for i, pair in enumerate(picked, 1):
        flip = rng.random() < 0.5
        arms = ("off", "on") if flip else ("on", "off")
        texts = {"on": a[pair]["text"], "off": b[pair]["text"]}
        sid = f"s{i:02d}"
        samples.append(
            {
                "sample": sid,
                "case": pair[0],
                "path": pair[1],
                "prompt": a[pair]["prompt"],
                "shown": a[pair]["shown"],
                "reply_A": texts[arms[0]],
                "reply_B": texts[arms[1]],
            }
        )
        key[sid] = {"A": arms[0], "B": arms[1]}
    return samples, key


def render_markdown(samples: list[dict[str, Any]]) -> str:
    """The judge-facing file: no arm labels, one block per sample."""
    out = ["# Blind comparison: which reply is better? (A or B, or tie)\n"]
    for s in samples:
        shown = "; ".join(f"{i.get('name')} (Rs {i.get('price_inr')})" for i in s["shown"][:5])
        out += [
            f"## {s['sample']}",
            f"Prompt: {s['prompt']}",
            f"Items shown: {shown or 'none'}",
            "",
            f"Reply A:\n> {s['reply_A']}",
            "",
            f"Reply B:\n> {s['reply_B']}",
            "",
            "Verdict: ",
            "",
        ]
    return "\n".join(out)


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--on", type=Path, required=True)
    ap.add_argument("--off", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    on = json.loads(args.on.read_text(encoding="utf-8"))
    off = json.loads(args.off.read_text(encoding="utf-8"))
    try:
        samples, key = build(on, off, args.n, args.seed)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "blind_dump.md").write_text(render_markdown(samples), encoding="utf-8")
    (args.out_dir / "blind_dump.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "blind_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(samples)} samples to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
