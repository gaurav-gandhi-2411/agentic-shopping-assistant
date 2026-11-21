"""Live launcher for stage 2: reads ONLY GROQ_API_KEY (by name) from an env file.

Never `load_dotenv`s the whole file: the file is parsed for the one wanted key, every OTHER name
found in it (and a list of known secrets) is removed from the process environment before any
app code is imported, and no value is printed. STAGE2_LIVE_OK=1 is set for the child run; the
harness still requires `--live` on its own command line.

    python -m evals.retrieval.live_run --env-file <repo>/.env -- --stage variants --live \\
        --arms A2,C --reps 1

Day-2 reranker run (A2 rep 1, then C rep 1; <= 100K tokens/day on gpt-oss-20b, one request per
8 s globally, stops on 3 consecutive 429s or any busy signal, resumable from the cache):
the line above. Tomorrow's rerun continues from the cache.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

WANTED = "GROQ_API_KEY"
KNOWN_SECRETS = (
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "SUPABASE_URL",
    "DEMO_JWT_SECRET",
    "DATABASE_URL",
    "ANTHROPIC_API_KEY",
)


def read_key(env_file: Path, name: str = WANTED) -> tuple[str | None, list[str]]:
    """(value of `name`, names of all other variables in the file). Values are not returned."""
    value: str | None = None
    others: list[str] = []
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key == name:
            value = val.strip().strip('"').strip("'")
        else:
            others.append(key)
    return value, others


def prepare_env(env_file: Path, environ: MutableMapping[str, str]) -> list[str]:
    """Pop every other secret from `environ`, set GROQ_API_KEY and STAGE2_LIVE_OK. Returns the
    names removed (for the log). Raises if the key is missing: fail closed."""
    value, others = read_key(env_file)
    if not value:
        raise SystemExit(f"{WANTED} not found in {env_file.name}")
    removed = []
    for k in {*others, *KNOWN_SECRETS}:
        if k != WANTED and environ.pop(k, None) is not None:
            removed.append(k)
    environ[WANTED] = value
    environ["STAGE2_LIVE_OK"] = "1"
    return sorted(removed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env-file", required=True)
    ap.add_argument("run_args", nargs=argparse.REMAINDER, help="after `--`: run.py arguments")
    args = ap.parse_args(argv)
    run_args = [a for a in args.run_args if a != "--"]
    removed = prepare_env(Path(args.env_file), os.environ)
    print(f"env: {WANTED} set (value hidden), STAGE2_LIVE_OK=1, other secrets removed: {removed}")
    from evals.retrieval import run  # imported AFTER the environment is clean

    return run.main(run_args)


if __name__ == "__main__":
    sys.exit(main())
