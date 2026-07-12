"""Build script: season/occasion/style/fabric facet enrichment (Phase C, sample mode).

Rules pass (src/catalogue/enrichment.py, $0, deterministic) first, then a LOCAL
Ollama (llama3.1:8b) fallback ONLY for facets the rules left as None on a given
row — never for facets the rules already found. The LLM is instructed to return
"none" for anything not clearly supported by the text (see _SYSTEM_PROMPT); its
reply is validated against the fixed vocab before being kept
(src/catalogue/enrichment.py::merge_enrichment) — an invented/paraphrased label
is discarded to None, never kept.

Two modes:
    - Sample mode (default): a stratified ~400-row sample, written to
      data/processed/enrichment_sample/ — for reviewing coverage/quality/latency
      before committing to a full run.
    - Full mode (--full): every row in the catalogue, written to the separate
      data/processed/enrichment_full/ directory (never collides with sample
      artifacts). This is the ~62k-row, ~16-hour Ollama pass approved as a
      resumable, unattended multi-night run.

Resumability
------------
Progress is written incrementally to a JSONL cache
(<output-dir>/enrichment_cache.jsonl, one line per article_id) after every
chunk (--chunk-size rows). Re-running the script skips article_ids already
present in the cache, so an interrupted run resumes instead of restarting —
this is what makes --full safe to run across multiple nights.

Usage
-----
    python scripts/enrich_attributes.py                        # 400-row stratified sample
    python scripts/enrich_attributes.py --sample-size 300 --seed 42
    python scripts/enrich_attributes.py --skip-llm              # rules-only, no Ollama calls
    python scripts/enrich_attributes.py --output-dir data/processed/enrichment_sample
    python scripts/enrich_attributes.py --full                  # full ~62k-row catalogue run
    python scripts/enrich_attributes.py --full --skip-llm        # full-corpus rules-only dry run
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Repo root on sys.path so ``src.*`` imports work when run as a script.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.catalogue.enrichment import (  # noqa: E402
    FACET_VOCAB,
    append_enrichment_to_search_text,
    merge_enrichment,
    rules_pass,
)
from src.catalogue.loader import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("enrich_attributes")

_CATALOGUE_PATH = _REPO_ROOT / "data" / "processed" / "unified" / "catalogue.parquet"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "processed" / "enrichment_sample"
_DEFAULT_OUTPUT_DIR_FULL = _REPO_ROOT / "data" / "processed" / "enrichment_full"

_FACET_KEYS: tuple[str, ...] = ("season", "occasion_tag", "style_tag", "fabric")

# How often (in chunks) to print an ETA estimate during a long unattended run —
# with the default chunk-size of 25 rows, 40 chunks is ~1,000 rows.
_ETA_LOG_EVERY_N_CHUNKS = 40

# Keyword probes used to force-include a few rows per signal type in the
# sample — so the report can show concrete before/after examples for each
# facet type, plus at least a few genuinely signal-free rows (honest None).
_PROBE_KEYWORDS: dict[str, str] = {
    "summer_signal": r"\bsleeveless\b|\blinen\b|\bsummer\b",
    "boho_signal": r"\bboho\b|\bbohemian\b",
    "office_signal": r"\boffice\s*wear\b|\bformal\b",
    "wedding_signal": r"\bwedding\b|\bbridal\b",
}

# ---------------------------------------------------------------------------
# Ollama fallback prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""You are a precise product-attribute extraction assistant for a fashion catalogue.

Given a product's name, type, and description, extract ONLY facets you are
confident are DIRECTLY supported by the text. Never guess, infer loosely, or
invent a value the text does not clearly support — if the text does not
mention or strongly imply a facet, you MUST answer "none" for it.

Return STRICT JSON with EXACTLY these 4 keys, each one of the allowed values
below (or the literal string "none"). No other text, no markdown fences.

season: one of {sorted(FACET_VOCAB["season"])} or "none"
occasion_tag: one of {sorted(FACET_VOCAB["occasion_tag"])} or "none"
style_tag: one of {sorted(FACET_VOCAB["style_tag"])} or "none"
fabric: one of {sorted(FACET_VOCAB["fabric"])} or "none"

Example output:
{{"season": "summer", "occasion_tag": "none", "style_tag": "none", "fabric": "cotton"}}
"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_user_prompt(prod_name: str, product_type_name: str, detail_desc: str) -> str:
    """Build the per-row user message, truncating detail_desc to keep the call cheap."""
    desc = (detail_desc or "")[:400]
    return f"Product: {prod_name}\nType: {product_type_name}\nDescription: {desc}"


def _parse_llm_json(raw: str) -> dict[str, str | None]:
    """Parse the model's reply into a facet dict, tolerating markdown fences / stray text.

    Returns an all-None dict (never raises) if the reply isn't parseable JSON —
    a parse failure is treated the same as "the model found nothing", not as a
    crash that would abort the whole batch.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return dict.fromkeys(_FACET_KEYS)
    try:
        parsed: dict[str, Any] = json.loads(match.group(0))
    except json.JSONDecodeError:
        return dict.fromkeys(_FACET_KEYS)

    out: dict[str, str | None] = {}
    for key in _FACET_KEYS:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip().lower() not in ("none", "null", ""):
            out[key] = value.strip().lower()
        else:
            out[key] = None
    return out


def call_ollama_extract(
    llm_client: Any,
    prod_name: str,
    product_type_name: str,
    detail_desc: str,
) -> tuple[dict[str, str | None], float]:
    """Call the local LLM once for the 4 facets; returns (parsed_dict, latency_seconds).

    *llm_client* is anything satisfying ``src.llm.client.LLMClient`` (Ollama in
    practice for this script — see config.yaml's ``llm.provider``). Uses
    temperature=0.0 for extraction consistency (this repo's default 0.2 is
    tuned for conversational replies, not deterministic tagging).
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(prod_name, product_type_name, detail_desc)},
    ]
    t0 = time.monotonic()
    try:
        raw = llm_client.chat(messages, temperature=0.0, max_tokens=100)
    except Exception as exc:  # noqa: BLE001 — a single row's LLM failure must not abort the batch
        logger.warning("[ollama] extraction call failed: %s", exc)
        return dict.fromkeys(_FACET_KEYS), time.monotonic() - t0
    latency = time.monotonic() - t0
    return _parse_llm_json(raw), latency


# ---------------------------------------------------------------------------
# Stratified sample selection
# ---------------------------------------------------------------------------


def select_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Build a stratified sample across product_type_name, plus forced signal/no-signal probes.

    Strategy:
        1. Proportional-ish stratified sample across distinct product_type_name
           values (capped per type so no single huge type dominates the sample).
        2. Force-include a handful of rows matching each _PROBE_KEYWORDS pattern
           (so the report has concrete before/after examples for every facet).
        3. Force-include a handful of rows with clearly no rules-pass signal at
           all, to demonstrate honest-null behaviour.
        4. Dedupe by article_id and trim to sample_size.

    Deterministic for a fixed (df, sample_size, seed).
    """
    rng_state = seed
    types = df["product_type_name"].fillna("Unknown").unique().tolist()
    per_type_cap = max(2, sample_size // max(len(types), 1))

    strat_parts: list[pd.DataFrame] = []
    for t in types:
        subset = df[df["product_type_name"].fillna("Unknown") == t]
        n = min(len(subset), per_type_cap)
        if n:
            strat_parts.append(subset.sample(n=n, random_state=rng_state))
    stratified = pd.concat(strat_parts, ignore_index=False) if strat_parts else df.iloc[0:0]

    probe_parts: list[pd.DataFrame] = []
    combined_text = (
        df["prod_name"].fillna("") + " " + df["detail_desc"].fillna("")
    ).str.lower()
    for _, pattern in _PROBE_KEYWORDS.items():
        matches = df[combined_text.str.contains(pattern, regex=True, na=False)]
        if len(matches):
            probe_parts.append(matches.sample(n=min(8, len(matches)), random_state=rng_state))

    # No-signal probes: rows where the rules pass finds nothing at all.
    no_signal_mask = df.apply(
        lambda r: not any(
            rules_pass(r.get("prod_name"), r.get("product_type_name"), r.get("detail_desc")).values()
        ),
        axis=1,
    )
    no_signal_rows = df[no_signal_mask]
    if len(no_signal_rows):
        probe_parts.append(no_signal_rows.sample(n=min(10, len(no_signal_rows)), random_state=rng_state))

    combined = pd.concat([stratified, *probe_parts], ignore_index=False)
    combined = combined.drop_duplicates(subset="article_id")

    if len(combined) > sample_size:
        combined = combined.sample(n=sample_size, random_state=rng_state)
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Resumable cache
# ---------------------------------------------------------------------------


def _load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load already-processed rows from the JSONL cache, keyed by article_id."""
    done: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return done
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            done[record["article_id"]] = record
    return done


def _append_cache(cache_path: Path, records: list[dict[str, Any]]) -> None:
    """Append *records* to the JSONL cache (flushed immediately for resumability)."""
    with open(cache_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------


def process_chunk(
    chunk: pd.DataFrame,
    llm_client: Any | None,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Run rules pass (+ Ollama fallback when needed) over one chunk of rows.

    Returns (records, stats) — records are cache-ready dicts (one per row);
    stats accumulates counts/latency for the run-level report.
    """
    records: list[dict[str, Any]] = []
    stats = {"rule_hits": 0, "llm_hits": 0, "none_hits": 0, "llm_calls": 0, "llm_latency_total": 0.0}

    for _, row in chunk.iterrows():
        rules = rules_pass(row.get("prod_name"), row.get("product_type_name"), row.get("detail_desc"))
        needs_llm = llm_client is not None and any(v is None for v in rules.values())

        llm_result: dict[str, str | None] | None = None
        if needs_llm:
            llm_result, latency = call_ollama_extract(
                llm_client,
                row.get("prod_name") or "",
                row.get("product_type_name") or "",
                row.get("detail_desc") or "",
            )
            stats["llm_calls"] += 1
            stats["llm_latency_total"] += latency

        merged, source = merge_enrichment(rules, llm_result)
        for facet, src in source.items():
            if src == "rule":
                stats["rule_hits"] += 1
            elif src == "llm":
                stats["llm_hits"] += 1
            else:
                stats["none_hits"] += 1

        records.append(
            {
                "article_id": row["article_id"],
                **merged,
                "source": source,
            }
        )

    return records, stats


# ---------------------------------------------------------------------------
# Merge cache back into a DataFrame + report
# ---------------------------------------------------------------------------


def build_enriched_frame(sample_df: pd.DataFrame, cache: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Apply cached enrichment onto *sample_df*'s facets + search_text.

    Returns a copy of *sample_df* with the 4 new facet keys merged into the
    existing ``facets`` dict column and appended to ``search_text``.
    """
    out = sample_df.copy()
    new_facets = []
    new_search_text = []
    for _, row in out.iterrows():
        record = cache.get(row["article_id"], {})
        enrichment = {k: record.get(k) for k in _FACET_KEYS}
        facets = dict(row["facets"]) if isinstance(row["facets"], dict) else {}
        facets.update(enrichment)
        new_facets.append(facets)
        new_search_text.append(append_enrichment_to_search_text(row["search_text"], enrichment))
    out["facets"] = new_facets
    out["search_text"] = new_search_text
    return out


def print_report(
    sample_df: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
    run_stats: dict[str, float | int],
    full_corpus_size: int,
) -> None:
    """Print coverage stats, before/after examples, and a full-run time estimate."""
    n = len(sample_df)
    print(f"\n=== Enrichment sample report (n={n}) ===")

    per_facet_source: dict[str, dict[str, int]] = {
        facet: {"rule": 0, "llm": 0, "none": 0} for facet in _FACET_KEYS
    }
    for aid in sample_df["article_id"]:
        record = cache.get(aid, {})
        source = record.get("source", {})
        for facet in _FACET_KEYS:
            src = source.get(facet, "none")
            per_facet_source[facet][src] += 1

    print(f"\n  {'facet':<14} {'rule':>6} {'llm':>6} {'none':>6}")
    for facet, counts in per_facet_source.items():
        print(f"  {facet:<14} {counts['rule']:>6} {counts['llm']:>6} {counts['none']:>6}")

    llm_calls = int(run_stats.get("llm_calls", 0))
    llm_latency_total = float(run_stats.get("llm_latency_total", 0.0))
    avg_latency = llm_latency_total / llm_calls if llm_calls else 0.0
    print(f"\n  Rows needing >=1 Ollama call: {llm_calls}/{n} ({llm_calls / n:.1%})")
    print(f"  Avg Ollama call latency: {avg_latency:.2f}s")

    if llm_calls and avg_latency:
        fraction_needing_llm = llm_calls / n
        est_calls_full = fraction_needing_llm * full_corpus_size
        est_seconds = est_calls_full * avg_latency
        print(
            f"\n  Full-corpus estimate ({full_corpus_size:,} rows, "
            f"{fraction_needing_llm:.1%} needing LLM @ {avg_latency:.2f}s/call):"
        )
        print(f"    ~{est_calls_full:,.0f} Ollama calls -> ~{est_seconds / 3600:.1f} hours wall-clock")
        print("    (rules pass itself is regex-only and negligible at full scale)")

    print("\n=== Before/after examples (search_text tail + new facets) ===")
    shown = 0
    for _, row in sample_df.iterrows():
        record = cache.get(row["article_id"], {})
        source = record.get("source", {})
        if not any(v == "rule" or v == "llm" for v in source.values()):
            continue
        new_tags = {k: record.get(k) for k in _FACET_KEYS if record.get(k)}
        if not new_tags:
            continue
        print(f"\n  article_id={row['article_id']}  store={row.get('store')}")
        print(f"  prod_name: {row['prod_name']}")
        print(f"  new facets: {new_tags}  (source={ {k: source.get(k) for k in new_tags} })")
        print(f"  search_text tail: ...{row['search_text'][-160:]}")
        shown += 1
        if shown >= 10:
            break

    honest_none_examples = [
        aid for aid in sample_df["article_id"]
        if cache.get(aid, {}).get("source", {}) and all(
            v == "none" for v in cache[aid]["source"].values()
        )
    ]
    print(f"\n  Honest-null rows (no facet found by rules OR llm): {len(honest_none_examples)}/{n}")
    if honest_none_examples:
        example_row = sample_df[sample_df["article_id"] == honest_none_examples[0]].iloc[0]
        print(f"    e.g. article_id={honest_none_examples[0]}  prod_name={example_row['prod_name']!r}")

    print("\n==========================================\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample-validate season/occasion/style/fabric enrichment (rules + Ollama fallback).",
    )
    parser.add_argument("--sample-size", type=int, default=400, help="Sample rows to process (default 400).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42).")
    parser.add_argument("--chunk-size", type=int, default=25, help="Rows per resumable chunk (default 25).")
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Process every row in the catalogue instead of a stratified sample. "
            f"Defaults --output-dir to {_DEFAULT_OUTPUT_DIR_FULL} (never collides with "
            "sample-mode artifacts) and additionally writes a full "
            "catalogue_enriched.parquet at the end of the run."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for the cache + enriched parquet. Defaults to "
            f"{_DEFAULT_OUTPUT_DIR} in sample mode or {_DEFAULT_OUTPUT_DIR_FULL} with --full."
        ),
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Rules pass only — no Ollama calls (fast smoke-test of the rules coverage).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the enrichment pipeline (sample or --full): rules pass + Ollama fallback + report.

    Sample mode (default) writes:
        <output-dir>/enrichment_cache.jsonl              (resumable per-row cache)
        <output-dir>/catalogue_enriched_sample.parquet   (merged sample, for inspection)

    --full additionally writes the complete enriched catalogue (all rows, all
    original columns, facets/search_text updated):
        <output-dir>/catalogue_enriched.parquet
    """
    args = _parse_args()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _DEFAULT_OUTPUT_DIR_FULL if args.full else _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "enrichment_cache.jsonl"

    mode_label = "FULL catalogue" if args.full else f"sample (sample_size={args.sample_size}, seed={args.seed})"
    print(f"\n=== Enrichment run: {mode_label} ===")
    print(f"Output dir: {output_dir}")
    df = pd.read_parquet(_CATALOGUE_PATH)
    full_corpus_size = len(df)
    print(f"Loaded catalogue: {full_corpus_size:,} rows")

    if args.full:
        sample_df = df.reset_index(drop=True)
        print(f"Full-corpus mode: processing all {len(sample_df):,} rows")
    else:
        sample_df = select_sample(df, args.sample_size, args.seed)
        print(f"Selected sample: {len(sample_df)} rows across {sample_df['product_type_name'].nunique()} product types")

    llm_client = None
    if not args.skip_llm:
        config = load_config()
        from src.llm.client import get_llm_client  # noqa: E402 — deferred: only needed here

        llm_client = get_llm_client(config)
        print(f"Ollama fallback enabled: model={config['llm']['model']} host={config['llm']['host']}")
    else:
        print("--skip-llm: rules-only run, no Ollama calls.")

    cache = _load_cache(cache_path)
    already_done = set(cache.keys())
    remaining = sample_df[~sample_df["article_id"].isin(already_done)]
    print(f"Resuming: {len(already_done)} already cached, {len(remaining)} remaining to process")

    run_stats: dict[str, float | int] = {
        "rule_hits": 0, "llm_hits": 0, "none_hits": 0, "llm_calls": 0, "llm_latency_total": 0.0,
    }
    t_start = time.perf_counter()
    n_chunks = max(1, (len(remaining) + args.chunk_size - 1) // args.chunk_size)
    for i in range(0, len(remaining), args.chunk_size):
        chunk = remaining.iloc[i : i + args.chunk_size]
        chunk_num = i // args.chunk_size + 1
        records, chunk_stats = process_chunk(chunk, llm_client)
        _append_cache(cache_path, records)
        for r in records:
            cache[r["article_id"]] = r
        for k in run_stats:
            run_stats[k] += chunk_stats[k]
        logger.info(
            "chunk %d/%d done (%d rows, %d llm calls so far)",
            chunk_num, n_chunks, min(i + args.chunk_size, len(remaining)), int(run_stats["llm_calls"]),
        )

        # ETA — every _ETA_LOG_EVERY_N_CHUNKS chunks (~1,000 rows at the default
        # chunk-size), plus the final chunk. Needed to gauge progress from a log
        # file during an unattended multi-hour/multi-night --full run.
        if chunk_num % _ETA_LOG_EVERY_N_CHUNKS == 0 or chunk_num == n_chunks:
            rows_done = min(i + args.chunk_size, len(remaining))
            rows_left = len(remaining) - rows_done
            elapsed_so_far = time.perf_counter() - t_start
            if rows_done and elapsed_so_far > 0:
                rate = rows_done / elapsed_so_far
                eta_seconds = rows_left / rate if rate > 0 else 0.0
                logger.info(
                    "progress: %d/%d rows (%.1f%%) elapsed=%.1fmin rate=%.2f rows/s ETA=%.1fmin",
                    rows_done, len(remaining), 100.0 * rows_done / len(remaining),
                    elapsed_so_far / 60.0, rate, eta_seconds / 60.0,
                )
    elapsed = time.perf_counter() - t_start
    print(f"Processing complete in {elapsed:.1f}s")

    enriched = build_enriched_frame(sample_df, cache)
    out_filename = "catalogue_enriched.parquet" if args.full else "catalogue_enriched_sample.parquet"
    out_path = output_dir / out_filename
    enriched.to_parquet(str(out_path), index=False)
    print(f"Wrote enriched {'catalogue' if args.full else 'sample'}: {out_path}")

    print_report(sample_df, cache, run_stats, full_corpus_size)


if __name__ == "__main__":
    main()
