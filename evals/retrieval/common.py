"""Shared paths, fixture loading and token counting for the retrieval eval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURE_DIR = REPO / "evals" / "fixtures" / "retrieval"
CANDIDATE_DIR = HERE / "candidates"  # frozen at 4b4fb4e: judged pools, F1 "sets not demoted" arm
# Pools regenerated on current master (#129 demotes sets before rerank): the main A/A2/B/C pools.
MASTER_CANDIDATE_DIR = HERE / "candidates_master"
RUN_DIR = REPO / "reports" / "wave1b" / "v4e"  # untracked outputs (labels, ledgers, packets)
PUBLIC_STORES_PATH = HERE / "public_stores.json"
SEED = 42

# Architect limits for stage 2 (daily token budgets are per model; reranker model shares its
# 200K TPD with prod canary traffic of about 100K/day).
JUDGE_MODELS = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3.8-27b": "qwen/qwen3.8-27b",
}
RERANKER_MODEL = "openai/gpt-oss-20b"
JUDGE_DAILY_TOKEN_BUDGET = 150_000
RERANKER_DAILY_TOKEN_BUDGET = 100_000
DEFAULT_REPS = 2  # runs per query of A and A2: rep 1 plus ONE repeat (stability)
MIN_REQUEST_INTERVAL_S = 8.0  # <= 1 request per 8 s per model


def load_cases(limit: int | None = None, fixture_dir: Path = FIXTURE_DIR) -> list[dict[str, Any]]:
    """Load the query fixtures (one JSON file per case), sorted by id."""
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(fixture_dir.glob("q*.json"))]
    return cases[:limit] if limit else cases


def load_public_stores() -> list[str]:
    """The (temporary) public-store slugs; candidates must come only from these."""
    return list(json.loads(PUBLIC_STORES_PATH.read_text(encoding="utf-8"))["stores"])


def count_tokens(text: str) -> int:
    """Token estimate: tiktoken o200k_base (the gpt-oss tokenizer proxy) when installed,
    else a ~3.6 chars/token heuristic (marked by tests; the README states which was used)."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:  # noqa: BLE001 - tiktoken missing or its data file not downloadable
        return max(1, round(len(text) / 3.6))


def sha(text: str) -> str:
    """Short stable hash used for prompt-cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def resolve_pools_dir(cand_dir: Path, pools: str | None) -> Path:
    """Directory of the pools the variants run on.

    Explicit `--pools` wins. Otherwise, when `cand_dir` is the default frozen directory and the
    master-regenerated pools exist, those are the main-comparison pools; any other `--cand`
    (a test or scratch dir) is its own pool, so a single-directory run keeps working."""
    if pools:
        return Path(pools)
    if cand_dir == CANDIDATE_DIR and MASTER_CANDIDATE_DIR.exists():
        return MASTER_CANDIDATE_DIR
    return cand_dir


def load_candidates(case_id: str, cand_dir: Path = CANDIDATE_DIR) -> dict[str, Any]:
    """Load the candidate file for one case."""
    return json.loads((cand_dir / f"{case_id}.json").read_text(encoding="utf-8"))
