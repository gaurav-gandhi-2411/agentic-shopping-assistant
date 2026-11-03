"""Count LLM calls made by one POST /chat, with a counting stub LLM (no network, no GCP).

Reproduces the "2 LLM calls for 'black kurta set for haldi'" figure quoted in DEPLOY.md and
scripts/ops/deploy_smoke.py. Tool file only: NOT run in CI (it needs the local retrieval index and
the sentence-transformers model cache, neither of which exists on a CI runner).

    python scripts/ops/count_llm_calls.py --scratch <empty-dir-you-own> [--scenario search|outfit]
        [--repo-root <repo>] [--hf-home <model cache>] [--index-dir <dir with dense.faiss>]
        [--message "black kurta set for haldi"]

Hermetic by construction: TEMP/TMP/HOME/USERPROFILE/LOCALAPPDATA/APPDATA are redirected into
--scratch and asserted before anything else runs; HF is forced offline; the real LLM host is a dead
port; there is no DATABASE_URL, so sessions are in memory. HF_HOME points at the real model cache
READ-ONLY (offline mode never writes to it). Each call is attributed to its call site (file:line).

Caveats: counts logical `llm.generate/chat` calls, not provider retries/failovers. The router LLM
is bypassed for queries the deterministic fast-path handles, so `--scenario outfit` only matters
for queries that actually reach the router.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any

ROUTES: dict[str, dict[str, Any]] = {
    "search": {"action": "search", "query": "", "filters": {}},
    "outfit": {
        "action": "outfit",
        "article_id": None,
        "occasion": "haldi",
        "gender": "women",
        "budget_inr": None,
    },
}


class CountingLLM:
    """Stub LLM that records the call site of every call and returns canned, parseable text."""

    def __init__(self, scenario: str, message: str) -> None:
        self.calls: list[str] = []
        self.router_n = 0
        self.cost_reporter = None
        self._route = dict(ROUTES[scenario])
        if scenario == "search":
            self._route["query"] = message

    def _site(self) -> str:
        frame = sys._getframe(2)  # caller of generate()/chat(), not this class
        return f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}"

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        site = self._site()
        self.calls.append(site)
        if site.startswith("router.py"):
            self.router_n += 1
            return json.dumps(self._route if self.router_n == 1 else {"action": "respond"})
        return "Here are some options for you."

    def generate_stream(self, prompt: str, system: str | None = None, **kwargs: Any):
        self.calls.append("STREAM:" + self._site())
        yield "ok"

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        self.calls.append("CHAT:" + self._site())
        return "ok"

    def chat_stream(self, messages: list[dict], **kwargs: Any):
        self.calls.append("CHATSTREAM:" + self._site())
        yield "ok"


def redirect_roots(scratch: Path, hf_home: Path) -> None:
    """Point every writable root at `scratch`, print and assert them (hermetic probe rule)."""
    for sub in ("tmp", "home", "local", "roaming"):
        (scratch / sub).mkdir(parents=True, exist_ok=True)
    env = {
        "TEMP": scratch / "tmp",
        "TMP": scratch / "tmp",
        "TMPDIR": scratch / "tmp",
        "USERPROFILE": scratch / "home",
        "HOME": scratch / "home",
        "LOCALAPPDATA": scratch / "local",
        "APPDATA": scratch / "roaming",
    }
    os.environ.update({k: str(v) for k, v in env.items()})
    os.environ.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TQDM_DISABLE": "1",
            "JWT_VERIFICATION_DISABLED": "true",
            "DEMO_MODE": "",
            "DATABASE_URL": "",
            "LLM_PROVIDER": "ollama",
            "OLLAMA_HOST": "http://127.0.0.1:9",  # dead port: a real call would fail loudly
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    for key in env:
        print("root", key, os.environ[key])
        assert os.path.realpath(os.environ[key]).startswith(os.path.realpath(scratch)), key
    print("root HF_HOME (read-only model cache)", hf_home)


def main(argv: list[str] | None = None) -> None:
    """Build the real retriever, run one /chat through the real graph with a counting stub LLM."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scratch", required=True, type=Path, help="empty dir you own; all roots go here")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"),
    )
    ap.add_argument("--index-dir", type=Path, default=None, help="default: <repo>/data/processed/unified")
    ap.add_argument("--scenario", choices=sorted(ROUTES), default="search")
    ap.add_argument("--message", default="black kurta set for haldi")
    args = ap.parse_args(argv)

    scratch = args.scratch.resolve()
    repo = args.repo_root.resolve()
    index_dir = (args.index_dir or repo / "data" / "processed" / "unified").resolve()
    if not (index_dir / "dense.faiss").exists():
        sys.exit(f"index not found at {index_dir} (needs dense.faiss, catalogue.parquet, bm25.pkl)")
    scratch.mkdir(parents=True, exist_ok=True)
    redirect_roots(scratch, args.hf_home.resolve())
    os.chdir(scratch)
    sys.path.insert(0, str(repo))

    # Imports are deferred until the environment is redirected (they read env at import time).
    import pandas as pd
    from fastapi.testclient import TestClient

    import api.deps as deps
    from api.main import app
    from api.session import InMemorySessionStore
    from src.catalogue.loader import load_config
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    cfg = load_config(str(repo / "config.yaml"))
    df = pd.read_parquet(index_dir / "catalogue.parquet")
    retriever = HybridRetriever(
        DenseRetriever.load(cfg, index_dir), SparseRetriever.load(cfg, index_dir), df, cfg
    )
    llm = CountingLLM(args.scenario, args.message)
    deps._init(retriever, df, llm, cfg, session_store=InMemorySessionStore())
    llm.calls.clear()

    client = TestClient(app)  # not a context manager: lifespan (GCS download, real LLM) is skipped
    resp = client.post("/chat", json={"message": args.message})
    body = resp.json()
    print(
        f"SCENARIO {args.scenario} HTTP {resp.status_code} items {len(body.get('items', []))} "
        f"answer_len {len(body.get('response', ''))}"
    )
    print("routing", body.get("routing"))
    print("LLM CALLS TOTAL", len(llm.calls), dict(collections.Counter(llm.calls)))


if __name__ == "__main__":
    main()
