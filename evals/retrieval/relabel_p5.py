"""Strict P@5 before/after consensus relabelling (no thresholds are changed anywhere).

    # 1. snapshot the strict-gold top-5 of a read-only index copy (CPU embedder, no LLM):
    python -m evals.retrieval.relabel_p5 --capture --data-dir <index copy> --out top5.json
    # 2. strict P@5 with the stored labels vs with the majority labels from proposal.yaml:
    python -m evals.retrieval.relabel_p5 --top5 top5.json --proposal <run dir>/proposal.yaml

P@5 is eval_strict's definition: relevant / (relevant + miss) over LABELLED top-5 items;
unlabelled items are never scored and are counted. "After" applies the proposal's resolved
replacements and new labels (ties are left as they are); strict_gold_labels.yaml is not edited
and eval_gate's thresholds are untouched. `unlabelled_pairs` is also how the hygiene items are
derived: the top-5 items on the merged master that carry no label.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from evals.retrieval.common import REPO
from evals.retrieval.relabel import LABELS_PATH, QUERIES_PATH, split_label_file

SCRIPTS = REPO / "scripts"


def capture(data_dir: Path, queries_path: Path = QUERIES_PATH, top_k: int = 5) -> dict[str, Any]:
    """Strict-gold top-k per query through eval_strict's own pipeline mirror (imported, not
    reimplemented), public-store view ON (default environment). Needs a built index copy."""
    for p in (str(REPO), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from eval_model import _build_components  # type: ignore[import-not-found]
    from eval_strict import _retrieve_pipeline  # type: ignore[import-not-found]

    queries = yaml.safe_load(queries_path.read_text("utf-8"))["queries"]
    retriever = _build_components(need_agent=False, data_dir=data_dir)["retriever"]
    out: dict[str, list[str]] = {}
    for q in queries:
        items = _retrieve_pipeline(retriever, q["query"], q["gender"], occasion_gate=True)
        out[q["id"]] = [str(it["article_id"]) for it in items[:top_k]]
    return {
        "snapshot": {"data_dir": str(data_dir), "mode": "pipeline", "top_k": top_k},
        "queries": out,
    }


def unlabelled_pairs(
    top5: Mapping[str, Any], labels: Mapping[tuple[str, str], Any]
) -> list[tuple[str, str]]:
    """(query_id, article_id) pairs in the snapshot's top-k that have no stored label."""
    return sorted(
        {
            (qid, aid)
            for qid, ids in top5["queries"].items()
            for aid in ids
            if (qid, str(aid)) not in labels
        }
    )


def strict_p5(
    top5: Mapping[str, Any],
    labels: Mapping[tuple[str, str], bool],
    categories: Mapping[str, str],
) -> dict[str, Any]:
    """eval_strict's P@5: relevant / (relevant + miss) over labelled items, per category too."""
    rel = miss = unl = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for qid, ids in top5["queries"].items():
        for aid in ids:
            lab = labels.get((qid, str(aid)))
            slot = 2 if lab is None else (0 if lab else 1)
            by_cat[categories[qid]][slot] += 1
            if lab is None:
                unl += 1
            elif lab:
                rel += 1
            else:
                miss += 1
    scored = rel + miss
    return {
        "p_at_5": rel / scored if scored else float("nan"),
        "n_relevant": rel,
        "n_scored": scored,
        "n_unlabelled": unl,
        "by_category": {
            c: {
                "p_at_5": v[0] / (v[0] + v[1]) if v[0] + v[1] else float("nan"),
                "n_scored": v[0] + v[1],
            }
            for c, v in sorted(by_cat.items())
        },
    }


def apply_proposal(
    labels: Mapping[tuple[str, str], bool], proposal: Mapping[str, Any]
) -> dict[tuple[str, str], bool]:
    """Stored labels with the proposal's resolved replacements and new labels applied."""
    out = dict(labels)
    for e in [*proposal.get("replacements", []), *proposal.get("new_labels", [])]:
        out[(e["query_id"], str(e["article_id"]))] = bool(e["proposed"])
    return out


def before_after(
    top5: Mapping[str, Any],
    labels: Mapping[tuple[str, str], bool],
    proposal: Mapping[str, Any],
    categories: Mapping[str, str],
) -> dict[str, Any]:
    """{"before": strict_p5(stored), "after": strict_p5(majority-applied), "delta_p_at_5"}."""
    before = strict_p5(top5, labels, categories)
    after = strict_p5(top5, apply_proposal(labels, proposal), categories)
    return {
        "before": before,
        "after": after,
        "delta_p_at_5": after["p_at_5"] - before["p_at_5"],
        "note": "thresholds untouched; LLM-consensus labels, not human ground truth",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--top5", default=None)
    ap.add_argument("--proposal", default=None)
    ap.add_argument("--queries", default=str(QUERIES_PATH))
    ap.add_argument("--labels", default=str(LABELS_PATH))
    args = ap.parse_args(argv)
    if args.capture:
        if not args.data_dir or not args.out:
            print("--capture needs --data-dir and --out")
            return 2
        snap = capture(Path(args.data_dir), Path(args.queries))
        Path(args.out).write_text(json.dumps(snap, indent=1), encoding="utf-8")
        print(f"wrote {len(snap['queries'])} queries to {args.out}")
        return 0
    if not args.top5 or not args.proposal:
        print("needs --top5 and --proposal (or --capture)")
        return 2
    queries = {
        q["id"]: q["category"]
        for q in yaml.safe_load(Path(args.queries).read_text("utf-8"))["queries"]
    }
    older, batch = split_label_file(Path(args.labels).read_text("utf-8"))
    labels = {k: bool(v["relevant"]) for k, v in {**older, **batch}.items()}
    res = before_after(
        json.loads(Path(args.top5).read_text("utf-8")),
        labels,
        yaml.safe_load(Path(args.proposal).read_text("utf-8")),
        queries,
    )
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
