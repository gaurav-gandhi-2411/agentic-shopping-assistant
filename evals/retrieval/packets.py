"""Claude-judge packets: export the SAME prompts the Groq judges get, import the JSON results.

The orchestrator (not this harness) has a Claude subagent read each packet and write
`<packet_id>.result.json` = {"q07": {"labels": {"1": 2, ...}}, ...}. The permutation of every
query is stored in `manifest.json`, so labels map back to the same candidate ids the other
judges labelled (a different order would break the item alignment).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.retrieval.judge import JudgePrompt, labels_to_ids, load_rubric, parse_labels

PACKET_MAX_QUERIES = 6


def export_packets(
    prompts: Sequence[JudgePrompt],
    out_dir: Path,
    prefix: str = "main",
    max_queries: int = PACKET_MAX_QUERIES,
) -> list[Path]:
    """Write packet text files (<= max_queries queries each) and manifest.json; return paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Merge with an existing manifest: calibration and main packets are exported by separate
    # calls into the same directory and must both stay importable.
    mpath = out_dir / "manifest.json"
    manifest: dict[str, Any] = json.loads(mpath.read_text("utf-8")) if mpath.exists() else {}
    paths = []
    for start in range(0, len(prompts), max_queries):
        chunk = prompts[start : start + max_queries]
        pid = f"{prefix}_{start // max_queries + 1:02d}"
        body = [
            "GRADING PACKET " + pid,
            "Follow the rubric below for EVERY query block. Judge each query independently.",
            "Write one JSON object mapping each query id to its labels, e.g.",
            '{"q07": {"labels": {"1": 2, "2": 0, ...}}, "q08": {...}} and nothing else.',
            "",
            "RUBRIC",
            load_rubric(),
        ]
        for p in chunk:
            body += ["", f"=== QUERY {p.query_id} ===", p.user]
        path = out_dir / f"{pid}.txt"
        path.write_text("\n".join(body) + "\n", encoding="utf-8")
        paths.append(path)
        manifest[pid] = {
            "queries": [p.query_id for p in chunk],
            "orders": {p.query_id: p.order for p in chunk},
            "prompt_hashes": {p.query_id: p.prompt_hash for p in chunk},
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return paths


def import_results(
    out_dir: Path,
    judge: str = "claude",
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Read every `<packet>.result.json` found; return ({judge: {"qid:article_id": label}}, problems).

    Missing/invalid queries are listed in `problems` (never silently filled)."""
    manifest: Mapping[str, Any] = json.loads((out_dir / "manifest.json").read_text("utf-8"))
    labels: dict[str, int] = {}
    problems: list[str] = []
    for pid, meta in manifest.items():
        rpath = out_dir / f"{pid}.result.json"
        if not rpath.exists():
            problems.append(f"{pid}: result file missing")
            continue
        data = json.loads(rpath.read_text("utf-8"))
        for qid in meta["queries"]:
            order = meta["orders"][qid]
            parsed = parse_labels(json.dumps(data.get(qid) or {}), len(order))
            if parsed is None:
                problems.append(f"{pid}/{qid}: missing or invalid labels")
                continue
            for aid, lab in labels_to_ids(parsed, order).items():
                labels[f"{qid}:{aid}"] = lab
    return {judge: labels}, problems
