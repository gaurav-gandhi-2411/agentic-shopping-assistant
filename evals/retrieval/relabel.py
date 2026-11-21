"""Consensus relabelling of strict-gold labels (3 blind judges, majority vote).

Why: the 127 labels of the ddb4dfd batch in eval/fixtures/strict_gold_labels.yaml carry the
provenance "single LLM judge (2026-11-15), pending consensus". This module re-labels them with
the same three-family blind setup as the retrieval eval (two Groq judges, gpt-oss-120b and
qwen3.8-27b, plus Claude through packets) under the strict-gold rubric (binary relevant /
irrelevant + a miss reason, `relabel_rubric.md`), and also re-labels
  (b) a seed-42 random sample of 60 OLDER labels, stratified by query category, as a drift check
      (do today's judges still agree with the 2026-07-11 labels?), and
  (c) the strict-gold top-5 items that are unlabelled on the merged master (the R1 hygiene items,
      derived from a read-only index copy; see `relabel_p5.py`).

It NEVER edits strict_gold_labels.yaml. Outputs (all in the run dir): per-item majority label,
agreement vs the stored single-judge label (percent, Cohen's kappa, confusion table, per group),
judge-pair kappa and Krippendorff alpha, and `proposal.yaml` listing the replacements. The
strict P@5 before/after script is `relabel_p5.py`.

Blind: judges see the query text and the shopper gender it was filtered with, then title (store
names scrubbed), price, colour, product type and a description snippet; never the store, the
catalogue gender column, the stored label, the rank, the query id or the item's group.
Calibration gate: the first chunk (drift items, with stored irrelevant items over-represented) is
judged by every judge before anything else; any judge pair with kappa < 0.4 (or undefined) stops
the run and prints the 2x2 confusion table.

Live calls need `--live` AND STAGE2_LIVE_OK=1 (same guard as the retrieval eval). `--mock` uses
rule-based fakes; tests never touch the network.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from evals.retrieval import agreement as ag
from evals.retrieval.common import (
    HERE,
    JUDGE_DAILY_TOKEN_BUDGET,
    JUDGE_MODELS,
    REPO,
    RUN_DIR,
    SEED,
    count_tokens,
    sha,
)
from evals.retrieval.groq_judge import (
    GroqRunner,
    Ledger,
    ModelSpec,
    ResultCache,
    assert_live_allowed,
    httpx_transport,
)
from evals.retrieval.scrub import scrub_stores

RUBRIC_PATH = HERE / "relabel_rubric.md"
QUERIES_PATH = REPO / "eval" / "fixtures" / "strict_gold_queries.yaml"
LABELS_PATH = REPO / "eval" / "fixtures" / "strict_gold_labels.yaml"
RELABEL_DIR = RUN_DIR / "relabel"
BATCH_MARKER = "# --- Public-catalogue gold labels (2026-11-16) ---"  # start of the ddb4dfd batch
GROUPS = ("ddb4dfd", "drift", "hygiene")
MISS_REASONS = (
    "type-confusion",
    "set-not-single",
    "kids-leak",
    "budget",
    "occasion-register",
    "attribute-contradiction",
    "data-mislabel",
    "data-ceiling",
)
DRIFT_N = 60
CHUNK_ITEMS = 25  # items per call: ~3.5K prompt tokens, well inside the 8K TPM window
CAL_MIN_STORED_IRRELEVANT = 8  # calibration chunk needs both classes or kappa is undefined
DESC_CHARS = 400  # the original labelers read the full description; 400 chars is the compromise
RELABEL_MAX_TOKENS = 2000  # labels + reasons for 25 items + hidden reasoning
EST_COMPLETION_PER_CALL = 900  # ASSUMED (visible ~25 items x 20 tokens + reasoning); plan only
KAPPA_GATE = ag.KAPPA_GATE
CLAUDE = "claude"
JUDGES = (*JUDGE_MODELS, CLAUDE)


@dataclass(frozen=True)
class Item:
    """One (query, article) pair to label; `stored` is None for unlabelled (hygiene) items."""

    key: str  # "query_id:article_id"
    query_id: str
    query: str
    shopper_gender: str
    category: str
    article_id: str
    title: str
    price_inr: float | None
    colour: str
    product_type: str
    description: str
    group: str
    stored: bool | None
    stored_reason: str | None


# ---------------------------------------------------------------- loading and sampling


def split_label_file(text: str) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    """(older labels, ddb4dfd batch) keyed by (query_id, article_id), split at BATCH_MARKER."""
    if BATCH_MARKER not in text:
        raise ValueError(f"label file has no batch marker {BATCH_MARKER!r}; cannot isolate ddb4dfd")
    head, tail = text.split(BATCH_MARKER, 1)

    def flat(raw: Any) -> dict[tuple[str, str], dict]:
        return {(e["query_id"], str(it["article_id"])): it for e in raw for it in e["items"]}

    older = flat(yaml.safe_load(head)["labels"])
    batch = flat(yaml.safe_load(tail))
    return older, batch


def load_gold(
    queries_path: Path = QUERIES_PATH, labels_path: Path = LABELS_PATH
) -> tuple[dict[str, dict], dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    """(queries by id, older labels, ddb4dfd batch labels)."""
    queries = {q["id"]: q for q in yaml.safe_load(queries_path.read_text("utf-8"))["queries"]}
    older, batch = split_label_file(labels_path.read_text("utf-8"))
    return queries, older, batch


def sample_drift(
    older: Mapping[tuple[str, str], Mapping],
    queries: Mapping[str, Mapping],
    n: int = DRIFT_N,
    seed: int = SEED,
) -> list[tuple[str, str]]:
    """Random n older labels, stratified by query category (largest-remainder allocation,
    `random.Random(f"{seed}:{category}")` inside each stratum). Deterministic; sorted output."""
    strata: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(older):
        strata[queries[key[0]]["category"]].append(key)
    total = sum(len(v) for v in strata.values())
    if n > total:
        raise ValueError(f"asked for {n} drift items but only {total} older labels exist")
    quota = {c: n * len(v) / total for c, v in strata.items()}
    alloc = {c: int(math.floor(q)) for c, q in quota.items()}
    for c in sorted(strata, key=lambda c: (-(quota[c] - alloc[c]), c))[: n - sum(alloc.values())]:
        alloc[c] += 1
    picked: list[tuple[str, str]] = []
    for c in sorted(strata):
        picked += random.Random(f"{seed}:{c}").sample(strata[c], alloc[c])
    return sorted(picked)


def load_articles(catalogue_path: Path, article_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Product fields for `article_ids` from a catalogue parquet (read-only)."""
    import pandas as pd

    cols = [
        "article_id",
        "prod_name",
        "display_name",
        "product_type_name",
        "colour_group_name",
        "price_inr",
        "detail_desc",
    ]
    df = pd.read_parquet(catalogue_path, columns=cols)
    df["article_id"] = df["article_id"].astype(str)
    df = df[df["article_id"].isin({str(a) for a in article_ids})]
    return {str(r["article_id"]): dict(r) for r in df.to_dict("records")}


def build_items(
    queries: Mapping[str, Mapping],
    older: Mapping[tuple[str, str], Mapping],
    batch: Mapping[tuple[str, str], Mapping],
    drift_keys: Sequence[tuple[str, str]],
    hygiene_keys: Sequence[tuple[str, str]],
    articles: Mapping[str, Mapping[str, Any]],
) -> list[Item]:
    """All items to relabel. Missing catalogue rows are an error (never silently skipped)."""
    spec: list[tuple[str, tuple[str, str], Mapping | None]] = (
        [("ddb4dfd", k, batch[k]) for k in sorted(batch)]
        + [("drift", k, older[k]) for k in drift_keys]
        + [("hygiene", k, None) for k in sorted(hygiene_keys)]
    )
    missing = [k for _, k, _ in spec if k[1] not in articles]
    if missing:
        raise KeyError(f"{len(missing)} items not in the catalogue, e.g. {missing[:3]}")
    items = []
    for group, (qid, aid), lab in spec:
        a, q = articles[aid], queries[qid]
        price = a.get("price_inr")
        items.append(
            Item(
                key=f"{qid}:{aid}",
                query_id=qid,
                query=q["query"],
                shopper_gender=str(q.get("gender") or "any"),
                category=q["category"],
                article_id=aid,
                title=str(a.get("display_name") or a.get("prod_name") or ""),
                price_inr=float(price) if price is not None and not math.isnan(price) else None,
                colour=str(a.get("colour_group_name") or "n/a"),
                product_type=str(a.get("product_type_name") or "n/a"),
                description=" ".join(str(a.get("detail_desc") or "").split()),
                group=group,
                stored=None if lab is None else bool(lab["relevant"]),
                stored_reason=None if lab is None else lab.get("reason"),
            )
        )
    return items


# ---------------------------------------------------------------- prompts


@dataclass(frozen=True)
class RelabelPrompt:
    """One judge call: `order[i]` is the item key shown as number i+1."""

    chunk_id: str
    system: str
    user: str
    order: list[str]
    prompt_hash: str
    calibration: bool


def load_rubric() -> str:
    return RUBRIC_PATH.read_text(encoding="utf-8").strip()


def _fmt_item(n: int, it: Item) -> str:
    price = f"₹{it.price_inr:.0f}" if it.price_inr is not None else "price n/a"
    desc = scrub_stores(it.description)[:DESC_CHARS]
    parts = [f"{n}. {scrub_stores(it.title)}", f"price {price}", f"colour {it.colour}"]
    parts.append(f"type {it.product_type}")
    return " | ".join(parts) + (f" | {desc}" if desc else "")


def chunk_items(items: Sequence[Item], size: int = CHUNK_ITEMS) -> list[list[Item]]:
    """Chunks of whole queries (a query's items are never split). Chunk 0 is the calibration
    chunk: drift queries with the most stored-irrelevant items first, until it has >= `size`
    items and >= CAL_MIN_STORED_IRRELEVANT stored-irrelevant ones (as far as the data allows)."""
    by_q: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_q[it.query_id].append(it)
    drift_q = sorted(
        {it.query_id for it in items if it.group == "drift"},
        key=lambda q: (-sum(1 for i in by_q[q] if i.group == "drift" and i.stored is False), q),
    )
    cal: list[Item] = []
    used: set[str] = set()
    for q in drift_q:
        n_irrel = sum(1 for i in cal if i.stored is False)
        if (len(cal) >= size and n_irrel >= CAL_MIN_STORED_IRRELEVANT) or len(cal) >= 2 * size:
            break
        cal += [i for i in by_q[q] if i.group == "drift"]  # only the drift items calibrate
        used.add(q)
    chunks: list[list[Item]] = [cal] if cal else []
    cal_keys = {i.key for i in cal}
    cur: list[Item] = []
    for q in sorted(by_q):
        rest = [i for i in by_q[q] if i.key not in cal_keys]
        if not rest:
            continue
        cur += rest
        if len(cur) >= size:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def build_prompts(
    items: Sequence[Item], seed: int = SEED, size: int = CHUNK_ITEMS
) -> list[RelabelPrompt]:
    """One prompt per chunk; block and item order are a pure function of (seed, chunk, query)."""
    system = load_rubric()
    prompts = []
    for ci, chunk in enumerate(chunk_items(items, size)):
        cid = f"r{ci + 1:02d}"
        by_q: dict[str, list[Item]] = defaultdict(list)
        for it in chunk:
            by_q[it.query_id].append(it)
        qids = sorted(by_q)
        random.Random(f"{seed}:{cid}").shuffle(qids)
        order: list[str] = []
        blocks = []
        n = 0
        for qid in qids:
            its = list(by_q[qid])
            random.Random(f"{seed}:{cid}:{qid}").shuffle(its)
            q = its[0]
            lines = []
            for it in its:
                n += 1
                order.append(it.key)
                lines.append(_fmt_item(n, it))
            blocks.append(
                f'Query: "{q.query}" (shopper gender: {q.shopper_gender})\n' + "\n".join(lines)
            )
        user = "\n\n".join(blocks) + f"\n\nLabel all {n} items. JSON only."
        prompts.append(RelabelPrompt(cid, system, user, order, sha(system + "\n" + user), ci == 0))
    return prompts


def parse_relabel(text: str, n: int) -> tuple[dict[int, int], dict[int, str]] | None:
    """Parse {"labels": {"1": 1|0}, "reasons": {"2": "<reason>"}}; None unless 1..n are all 0/1.
    Reasons outside MISS_REASONS or on a label-1 item are dropped (never fail the labels)."""
    blobs = [text.strip()]
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        blobs.append(m.group())
    for blob in blobs:
        try:
            data = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        raw = data.get("labels") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            continue
        try:
            labels = {int(k): int(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            return None
        if set(labels) != set(range(1, n + 1)) or any(v not in (0, 1) for v in labels.values()):
            return None
        reasons: dict[int, str] = {}
        for k, v in (data.get("reasons") or {}).items():
            try:
                pos = int(k)
            except (TypeError, ValueError):
                continue
            if labels.get(pos) == 0 and v in MISS_REASONS:
                reasons[pos] = v
        return labels, reasons
    return None


# ---------------------------------------------------------------- consensus and statistics


def majority(votes: Mapping[str, int]) -> dict[str, Any]:
    """Strict-majority binary label of one item's judge votes; `label` None on a tie (only
    possible with 2 judges). Fails closed: a tie never becomes a replacement."""
    ones = sum(votes.values())
    n = len(votes)
    if ones * 2 > n:
        label: int | None = 1
    elif ones * 2 < n:
        label = 0
    else:
        label = None
    return {"label": label, "votes": dict(votes), "n_judges": n, "tie": label is None}


def consensus(
    judge_labels: Mapping[str, Mapping[str, int]], items: Sequence[Item]
) -> dict[str, dict[str, Any]]:
    """item key -> majority(votes) over the judges that labelled it (items nobody labelled are
    omitted, never defaulted)."""
    out: dict[str, dict[str, Any]] = {}
    for it in items:
        votes = {j: lab[it.key] for j, lab in judge_labels.items() if it.key in lab}
        if votes:
            out[it.key] = majority(votes)
    return out


def majority_reason(
    reasons: Mapping[str, Mapping[str, str]], key: str, votes: Mapping[str, int]
) -> str | None:
    """Most common miss reason among the judges that voted 0 for `key` (ties: taxonomy order)."""
    found = [reasons[j][key] for j, v in votes.items() if v == 0 and key in reasons.get(j, {})]
    if not found:
        return None
    counts = Counter(found)
    return max(counts, key=lambda r: (counts[r], -MISS_REASONS.index(r)))


def binary_agreement(a: Sequence[int], b: Sequence[int]) -> dict[str, Any]:
    """Percent agreement, Cohen's kappa (NaN if undefined) and the 2x2 confusion (rows = a)."""
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    n = len(a)
    if n == 0:
        return {
            "n": 0,
            "percent": float("nan"),
            "kappa": float("nan"),
            "confusion": [[0, 0], [0, 0]],
        }
    conf = ag.confusion(a, b, (0, 1)).tolist()
    return {
        "n": n,
        "percent": 100.0 * sum(1 for x, y in zip(a, b, strict=True) if x == y) / n,
        "kappa": ag.cohen_kappa(a, b, labels=(0, 1)),
        "confusion": conf,
    }


def agreement_vs_stored(
    cons: Mapping[str, Mapping[str, Any]], items: Sequence[Item]
) -> dict[str, Any]:
    """Majority label vs the stored single-judge label, overall and per group (tied items are
    excluded and counted)."""
    out: dict[str, Any] = {}
    for group in ("ddb4dfd", "drift", "all"):
        sel = [
            it
            for it in items
            if it.stored is not None and it.key in cons and (group == "all" or it.group == group)
        ]
        resolved = [it for it in sel if not cons[it.key]["tie"]]
        stats = binary_agreement(
            [int(it.stored) for it in resolved if it.stored is not None],
            [int(cons[it.key]["label"]) for it in resolved],
        )
        out[group] = {**stats, "n_tied_excluded": len(sel) - len(resolved)}
    return out


def pair_report(judge_labels: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """Pairwise kappa + 2x2 confusion on commonly labelled items and Krippendorff alpha."""
    judges = sorted(j for j, lab in judge_labels.items() if lab)
    pairs: dict[str, Any] = {}
    for ja, jb in itertools.combinations(judges, 2):
        common = sorted(set(judge_labels[ja]) & set(judge_labels[jb]))
        if common:
            pairs[f"{ja}|{jb}"] = binary_agreement(
                [judge_labels[ja][k] for k in common], [judge_labels[jb][k] for k in common]
            )
    keys = sorted({k for j in judges for k in judge_labels[j]})
    units = [[judge_labels[j].get(k) for j in judges] for k in keys]
    alpha = ag.krippendorff_alpha_ordinal(units) if units else float("nan")
    lo, hi = ag.bootstrap_alpha_ci(units) if units else (float("nan"), float("nan"))
    return {"judges": judges, "pairs": pairs, "alpha": alpha, "alpha_ci95": [lo, hi]}


def kappa_gate(judge_labels: Mapping[str, Mapping[str, int]]) -> tuple[bool, str]:
    """Calibration gate on the first chunk: every judge pair needs kappa >= 0.4. Fails closed
    (NaN kappa, fewer than two judges, or no overlap all count as a failure) and prints the 2x2
    confusion table of every pair."""
    rep = pair_report(judge_labels)
    if not rep["pairs"]:
        return False, "relabel calibration gate: fewer than two judges with overlapping labels"
    bad = [
        (p, v["kappa"])
        for p, v in rep["pairs"].items()
        if math.isnan(v["kappa"]) or v["kappa"] < KAPPA_GATE
    ]
    if not bad:
        return True, f"relabel calibration gate passed: all judge-pair kappas >= {KAPPA_GATE:.2f}"
    lines = [
        f"RELABEL CALIBRATION GATE FAILED (kappa < {KAPPA_GATE}): "
        + ", ".join(f"{p} kappa={k:.3f}" for p, k in bad)
    ]
    for p, v in rep["pairs"].items():
        lines.append(
            f"confusion {p} (rows = first judge, cols = second; order 0,1): {v['confusion']}"
        )
    return False, "\n".join(lines)


def build_proposal(
    cons: Mapping[str, Mapping[str, Any]],
    items: Sequence[Item],
    reasons: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Replacement proposal; strict_gold_labels.yaml is never edited by this module."""
    replacements, new_labels, unresolved = [], [], []
    for it in items:
        c = cons.get(it.key)
        if c is None:
            continue
        entry = {
            "query_id": it.query_id,
            "article_id": it.article_id,
            "group": it.group,
            "votes": c["votes"],
        }
        if c["tie"]:
            unresolved.append(entry)
            continue
        proposed = bool(c["label"])
        reason = None if proposed else majority_reason(reasons, it.key, c["votes"])
        if it.stored is None:
            new_labels.append({**entry, "proposed": proposed, "reason": reason})
        elif proposed != it.stored:
            replacements.append(
                {
                    **entry,
                    "stored": it.stored,
                    "stored_reason": it.stored_reason,
                    "proposed": proposed,
                    "reason": reason,
                }
            )
    return {
        "note": "PROPOSAL ONLY; strict_gold_labels.yaml is not edited. LLM-consensus labels "
        "(majority of blind judges), not human ground truth.",
        "n_replacements": len(replacements),
        "n_new_labels": len(new_labels),
        "n_unresolved_ties": len(unresolved),
        "replacements": replacements,
        "new_labels": new_labels,
        "unresolved": unresolved,
    }


def judge_count_warning(
    judge_labels: Mapping[str, Mapping[str, int]], expected: int = 3
) -> list[str]:
    """Warning text when fewer than `expected` judges contributed any label."""
    n = sum(1 for lab in judge_labels.values() if lab)
    if n >= expected:
        return []
    return [
        f"ONLY {n} of {expected} judges contributed labels: with two judges every disagreement is "
        "an unresolved tie (no replacement is proposed for it)"
    ]


# ---------------------------------------------------------------- plan


def plan(
    prompts: Sequence[RelabelPrompt], budget: int = JUDGE_DAILY_TOKEN_BUDGET
) -> dict[str, Any]:
    """Token / day plan for ONE Groq judge model from the real prompts (tiktoken proxy or the
    3.6 chars/token heuristic; completion tokens are ASSUMED, not measured)."""
    prompt_tokens = [count_tokens(p.system + p.user) for p in prompts]
    expected = sum(prompt_tokens) + len(prompts) * EST_COMPLETION_PER_CALL
    worst = sum(prompt_tokens) + len(prompts) * RELABEL_MAX_TOKENS
    return {
        "n_calls_per_judge": len(prompts),
        "items": sum(len(p.order) for p in prompts),
        "prompt_tokens_per_call": prompt_tokens,
        "expected_tokens_per_judge": expected,
        "worst_case_tokens_per_judge": worst,
        "daily_budget": budget,
        "days_expected": math.ceil(expected / budget),
        "days_worst_case": math.ceil(worst / budget),
    }


# ---------------------------------------------------------------- packets (Claude)


def export_packets(prompts: Sequence[RelabelPrompt], out_dir: Path) -> list[Path]:
    """Write one packet text file per chunk (same prompts as the Groq judges) + manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    paths = []
    for p in prompts:
        body = [
            f"RELABEL PACKET {p.chunk_id}",
            "Follow the rubric below. Write one JSON object and nothing else, keyed by the "
            'packet id: {"' + p.chunk_id + '": {"labels": {"1": 1, "2": 0}, "reasons": {"2": '
            '"set-not-single"}}}',
            "",
            "RUBRIC",
            p.system,
            "",
            p.user,
        ]
        path = out_dir / f"{p.chunk_id}.txt"
        path.write_text("\n".join(body) + "\n", encoding="utf-8")
        paths.append(path)
        manifest[p.chunk_id] = {"order": p.order, "prompt_hash": p.prompt_hash}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return paths


def import_packets(
    out_dir: Path,
) -> tuple[dict[str, int], dict[str, str], list[str]]:
    """(labels {key: 0/1}, reasons {key: reason}, problems) from every `<chunk>.result.json`."""
    manifest = json.loads((out_dir / "manifest.json").read_text("utf-8"))
    labels: dict[str, int] = {}
    reasons: dict[str, str] = {}
    problems: list[str] = []
    for cid, meta in manifest.items():
        rpath = out_dir / f"{cid}.result.json"
        if not rpath.exists():
            problems.append(f"{cid}: result file missing")
            continue
        data = json.loads(rpath.read_text("utf-8"))
        parsed = parse_relabel(json.dumps(data.get(cid) or {}), len(meta["order"]))
        if parsed is None:
            problems.append(f"{cid}: missing or invalid labels")
            continue
        lab, rea = parsed
        for pos, key in enumerate(meta["order"], 1):
            labels[key] = lab[pos]
            if pos in rea:
                reasons[key] = rea[pos]
    return labels, reasons, problems


# ---------------------------------------------------------------- mock judges and the driver

_MOCK_STOP = {
    "for",
    "a",
    "an",
    "the",
    "under",
    "women",
    "men",
    "womens",
    "mens",
    "and",
    "in",
    "with",
}
_ITEM_RE = re.compile(
    r"^(\d+)\. (.*?) \| price (?:₹(\d+)|n/a) \| colour (.*?) \| type (.*?)(?: \| .*)?$"
)


def mock_relabel_text(user: str, noise: float = 0.0, salt: int = 0) -> str:
    """Deterministic rule-based judge: relevant iff every content word of the query appears in
    the item's title/type and no opposing-gender word does (plumbing only, not a real judge)."""
    labels: dict[str, int] = {}
    reasons: dict[str, str] = {}
    query, gender = "", "any"
    for ln in user.splitlines():
        m = re.match(r'^Query: "(.*)" \(shopper gender: (.*)\)$', ln)
        if m:
            query, gender = m.group(1).lower(), m.group(2)
            continue
        im = _ITEM_RE.match(ln)
        if not im:
            continue
        n, title, price, colour, ptype = im.groups()
        text = f"{title} {ptype} {colour}".lower()
        words = [w for w in re.findall(r"[a-z]+", query) if w not in _MOCK_STOP and len(w) > 2]
        wrong_gender = (gender == "women" and re.search(r"\bmen's\b|\bmens\b", text)) or (
            gender == "men" and re.search(r"\bwomen's\b|\bwomens\b", text)
        )
        cap = re.search(r"under (\d+)", query)
        over = bool(cap and price and int(price) > int(cap.group(1)))
        ok = all(w[:4] in text for w in words) and not wrong_gender and not over
        if noise and ((int(n) * 7919 + salt * 104729) % 1000) / 1000 < noise:
            ok = not ok
        labels[n] = int(ok)
        if not ok:
            reasons[n] = "budget" if over else "type-confusion"
    return json.dumps({"labels": labels, "reasons": reasons})


MOCK_NOISE = {"openai/gpt-oss-120b": 0.0, "qwen/qwen3.8-27b": 0.04, CLAUDE: 0.02}


def _mock_responder(system: str, user: str, model: str) -> str:
    return mock_relabel_text(user, MOCK_NOISE.get(model, 0.0), salt=len(model))


def relabel_specs(budget: int = JUDGE_DAILY_TOKEN_BUDGET) -> dict[str, ModelSpec]:
    """The two Groq judges (same parameters as the retrieval-eval judges)."""
    from evals.retrieval.run import JUDGE_PARAMS

    return {
        name: ModelSpec(f"relabel-{name}", model, JUDGE_PARAMS[name], RELABEL_MAX_TOKENS, budget)
        for name, model in JUDGE_MODELS.items()
    }


def _runner(mock: bool, live: bool, out_dir: Path) -> GroqRunner:
    if mock:
        from evals.retrieval.mock import mock_transport

        return GroqRunner(
            mock_transport(_mock_responder, (3500, 700)),
            Ledger(out_dir / "ledger.jsonl"),
            ResultCache(out_dir / "cache.jsonl"),
            min_interval_s=0.0,
            sleep=lambda s: None,
        )
    assert_live_allowed(live)
    return GroqRunner(
        httpx_transport(), Ledger(out_dir / "ledger.jsonl"), ResultCache(out_dir / "cache.jsonl")
    )


def _items_path(out_dir: Path) -> Path:
    return out_dir / "items.json"


def save_items(items: Sequence[Item], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _items_path(out_dir).write_text(
        json.dumps([asdict(i) for i in items], indent=1, ensure_ascii=False), encoding="utf-8"
    )


def load_items(out_dir: Path) -> list[Item]:
    return [Item(**d) for d in json.loads(_items_path(out_dir).read_text("utf-8"))]


def stage_prepare(args: argparse.Namespace) -> int:
    """Select the items (ddb4dfd batch + drift sample + hygiene) and write items.json."""
    out_dir = Path(args.out)
    queries, older, batch = load_gold(Path(args.queries), Path(args.labels))
    drift = sample_drift(older, queries, args.drift_n, SEED)
    hygiene: list[tuple[str, str]] = []
    if args.hygiene_top5:
        from evals.retrieval.relabel_p5 import unlabelled_pairs

        hygiene = unlabelled_pairs(
            json.loads(Path(args.hygiene_top5).read_text("utf-8")), {**older, **batch}
        )
    keys = [*sorted(batch), *drift, *hygiene]
    articles = load_articles(Path(args.catalogue), [k[1] for k in keys])
    items = build_items(queries, older, batch, drift, hygiene, articles)
    save_items(items, out_dir)
    counts = Counter(i.group for i in items)
    print(f"items: {dict(counts)} (total {len(items)}) -> {_items_path(out_dir)}")
    if hygiene:
        print("hygiene unlabelled pairs:", ", ".join(f"{q}:{a}" for q, a in hygiene))
    return 0


def _save_labels(
    out_dir: Path, judge: str, labels: Mapping[str, int], reasons: Mapping[str, str]
) -> None:
    (out_dir / f"labels_{judge}.json").write_text(
        json.dumps({"labels": labels, "reasons": reasons}, indent=1), encoding="utf-8"
    )


def _load_labels(out_dir: Path) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    labels: dict[str, dict[str, int]] = {}
    reasons: dict[str, dict[str, str]] = {}
    for j in JUDGES:
        p = out_dir / f"labels_{j}.json"
        if p.exists():
            d = json.loads(p.read_text("utf-8"))
            labels[j], reasons[j] = d["labels"], d["reasons"]
    return labels, reasons


def _judge_chunk(
    runner: GroqRunner, spec: ModelSpec, p: RelabelPrompt
) -> tuple[dict[str, int], dict[str, str]] | None:
    res = runner.call(spec, p.chunk_id, p.system, p.user, p.prompt_hash)
    parsed = parse_relabel(res["text"], len(p.order)) if res else None
    if parsed is None:
        return None
    lab, rea = parsed
    return (
        {k: lab[i] for i, k in enumerate(p.order, 1)},
        {p.order[i - 1]: r for i, r in rea.items()},
    )


def stage_judge(args: argparse.Namespace) -> int:
    """Calibration chunk first (gate), then the remaining chunks, for every available judge."""
    out_dir = Path(args.out)
    items = load_items(out_dir)
    prompts = build_prompts(items)
    specs = relabel_specs(args.judge_budget)
    runner = _runner(args.mock, args.live, out_dir)
    labels: dict[str, dict[str, int]] = {j: {} for j in specs}
    reasons: dict[str, dict[str, str]] = {j: {} for j in specs}
    failed: list[str] = []

    def run_chunks(chunks: Sequence[RelabelPrompt]) -> None:
        for p in chunks:
            for j, spec in specs.items():
                got = _judge_chunk(runner, spec, p)
                if got is None:
                    failed.append(f"{j} {p.chunk_id}")
                    continue
                labels[j].update(got[0])
                reasons[j].update(got[1])

    run_chunks(prompts[:1])
    cal_labels = {j: dict(v) for j, v in labels.items()}
    claude_cal = out_dir / "labels_claude.json"
    if args.mock:  # the mock stands in for the Claude judge too
        cal = prompts[0]
        got = json.loads(mock_relabel_text(cal.user, MOCK_NOISE[CLAUDE], len(CLAUDE)))
        cal_labels[CLAUDE] = {cal.order[int(k) - 1]: v for k, v in got["labels"].items()}
    elif claude_cal.exists():
        full = json.loads(claude_cal.read_text("utf-8"))["labels"]
        cal_labels[CLAUDE] = {k: v for k, v in full.items() if k in set(prompts[0].order)}
    ok, msg = kappa_gate(cal_labels)
    if len(cal_labels) < 3 and not args.allow_partial_gate:
        ok, msg = (
            False,
            "relabel gate needs all 3 judges (Claude calibration chunk missing)\n" + msg,
        )
    print(msg)
    if not ok:
        return 2
    run_chunks(prompts[1:])
    for j in specs:
        _save_labels(out_dir, j, labels[j], reasons[j])
        print(f"{j}: {len(labels[j])} labels")
    if args.mock:
        c_lab: dict[str, int] = {}
        c_rea: dict[str, str] = {}
        for p in prompts:
            got = json.loads(mock_relabel_text(p.user, MOCK_NOISE[CLAUDE], len(CLAUDE)))
            c_lab.update({p.order[int(k) - 1]: v for k, v in got["labels"].items()})
            c_rea.update({p.order[int(k) - 1]: v for k, v in got["reasons"].items()})
        _save_labels(out_dir, CLAUDE, c_lab, c_rea)
    if failed:
        print("unparseable/failed calls (not labelled):", ", ".join(failed))
    return 0


def stage_export(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    paths = export_packets(build_prompts(load_items(out_dir)), out_dir / "claude_packets")
    print(f"wrote {len(paths)} packets to {out_dir / 'claude_packets'}")
    return 0


def stage_import(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    labels, reasons, problems = import_packets(out_dir / "claude_packets")
    _save_labels(out_dir, CLAUDE, labels, reasons)
    print(f"imported {len(labels)} Claude labels; problems: {problems or 'none'}")
    return 1 if problems else 0


def stage_consensus(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    items = load_items(out_dir)
    labels, reasons = _load_labels(out_dir)
    if not labels:
        print("no judge labels found; run --stage judge first")
        return 2
    cons = consensus(labels, items)
    rep: dict[str, Any] = {
        "n_items": len(items),
        "n_resolved": sum(1 for c in cons.values() if not c["tie"]),
        "n_ties": sum(1 for c in cons.values() if c["tie"]),
        "vs_stored": agreement_vs_stored(cons, items),
        "judge_pairs": pair_report(labels),
        "warnings": judge_count_warning(labels),
        "label_provenance": "LLM-consensus labels (majority of blind judges), not human ground truth",
    }
    (out_dir / "consensus.json").write_text(
        json.dumps({"report": rep, "items": cons}, indent=1), encoding="utf-8"
    )
    (out_dir / "proposal.yaml").write_text(
        yaml.safe_dump(build_proposal(cons, items, reasons), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(json.dumps(rep, indent=1))
    for w in rep["warnings"]:
        print("WARNING:", w)
    return 0


def stage_plan(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    p = plan(build_prompts(load_items(out_dir)), args.judge_budget)
    (out_dir / "plan.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
    print(json.dumps(p, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage",
        required=True,
        choices=["prepare", "plan", "judge", "export-packets", "import-claude", "consensus"],
    )
    ap.add_argument("--mock", action="store_true", help="rule-based fakes, no network")
    ap.add_argument("--live", action="store_true", help="live Groq (also needs STAGE2_LIVE_OK=1)")
    ap.add_argument("--out", default=str(RELABEL_DIR))
    ap.add_argument("--queries", default=str(QUERIES_PATH))
    ap.add_argument("--labels", default=str(LABELS_PATH))
    ap.add_argument("--catalogue", default=None, help="catalogue.parquet of a read-only index copy")
    ap.add_argument(
        "--hygiene-top5", default=None, help="top-5 snapshot JSON (relabel_p5 --capture)"
    )
    ap.add_argument("--drift-n", type=int, default=DRIFT_N)
    ap.add_argument("--allow-partial-gate", action="store_true")
    ap.add_argument("--judge-budget", type=int, default=JUDGE_DAILY_TOKEN_BUDGET)
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    stages = {
        "prepare": stage_prepare,
        "plan": stage_plan,
        "judge": stage_judge,
        "export-packets": stage_export,
        "import-claude": stage_import,
        "consensus": stage_consensus,
    }
    if args.stage == "prepare" and not args.catalogue:
        print("--catalogue is required for --stage prepare")
        return 2
    try:
        return stages[args.stage](args)
    except Exception as exc:  # noqa: BLE001 - guard/budget/429 stops are reported as a refusal
        print(f"STOPPED: {type(exc).__name__}: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
