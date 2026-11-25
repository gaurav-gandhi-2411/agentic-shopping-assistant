"""ONNX encoder equivalence gate (rule: evals/onnx_equivalence_rule.md, pre-registered).

    python scripts/onnx_equivalence_gate.py \
        --index-dir <data/processed/unified> --clip-dir <data/processed/clip/unified> \
        --queries evals/fixtures/onnx_equivalence/queries.json --images-dir <H&M images dir> \
        --out-dir reports/onnx_equivalence

Reference vectors are the vectors STORED in the FAISS indices, not re-encoded by torch. Criteria
(all must pass; exit 1 otherwise):
  C1 MiniLM min cosine(ONNX, stored)      >= 0.9999 over the 2,000-item sample (seed 42)
  C2 CLIP-text min cosine(ONNX, stored)   >= 0.9999 over the same positions
  C3 CLIP-image min cosine(ONNX, torch)   >= 0.9999 over 200 images (no stored image vectors exist)
  C4 mean top-10 overlap torch vs ONNX    >= 0.99 for dense (60 queries), clip text (60), images
Torch is only needed for the validity control, C3 and C4; without it those are reported as NOT RUN
(the gate then cannot pass). CI runs it without images (`--skip-images`) and says so.
Writes <out-dir>/onnx_equivalence.json and .md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import export_onnx as eo  # noqa: E402  (the pinned HF revisions live there: one source of truth)

MIN_COS = 0.9999
MIN_OVERLAP = 0.99
SEED = 42
TOPK = 10
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
CLIP = "clip-ViT-B-32"


def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of two (n, d) arrays (inputs need not be normalised)."""
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return (a * b).sum(axis=1)


def cos_stats(c: np.ndarray) -> dict[str, float | int]:
    """n, min, p1, median of a cosine vector."""
    return {
        "n": int(len(c)),
        "min": float(c.min()),
        "p1": float(np.percentile(c, 1)),
        "median": float(np.median(c)),
        "max": float(c.max()),
    }


def topk_ids(index: Any, ids: np.ndarray, vecs: np.ndarray, k: int = TOPK) -> list[set[str]]:
    """Top-k article_id sets from an unrestricted IndexFlatIP search for each row of vecs."""
    _, idx = index.search(np.ascontiguousarray(vecs, dtype=np.float32), k)
    return [{str(ids[j]) for j in row if j >= 0} for row in idx]


def overlap_stats(a: list[set[str]], b: list[set[str]], k: int = TOPK) -> dict[str, Any]:
    """Per-query |A∩B|/k, summarised (mean, min, number of queries with overlap < 1)."""
    ov = np.array([len(x & y) / k for x, y in zip(a, b, strict=True)])
    hist: dict[str, int] = {}
    for v in ov:
        hist[f"{v:.1f}"] = hist.get(f"{v:.1f}", 0) + 1
    return {"n": int(len(ov)), "mean": float(ov.mean()), "min": float(ov.min()),
            "n_below_1": int((ov < 1).sum()), "histogram": dict(sorted(hist.items()))}


def sample_positions(n_total: int, n: int, seed: int = SEED) -> np.ndarray:
    """Seeded sample without replacement (the rule's `default_rng(42).choice`)."""
    return np.random.default_rng(seed).choice(n_total, size=min(n, n_total), replace=False)


MIN_REPRODUCED_FRACTION = 0.5  # below this the stored reference is not usable at all (INVALID)


def controlled_stats(onnx_v: np.ndarray, torch_v: np.ndarray, stored: np.ndarray) -> dict[str, Any]:
    """ONNX-vs-stored cosine over the items whose stored vector torch itself reproduces.

    A stored vector is only a usable reference where torch fp32 on the rebuilt text gives it back
    (cos >= MIN_COS); where it does not, the text that was indexed is unknown, so neither encoder
    can be judged against it. The pass bar applies to the reproduced subset; the full-sample
    numbers and the excluded count are reported next to it (rule amendment 1).
    """
    ctrl = cosines(torch_v, stored)
    ok = ctrl >= MIN_COS
    all_items = cosines(onnx_v, stored)
    out: dict[str, Any] = {
        "n_sample": int(len(ok)), "n_reproduced_by_torch": int(ok.sum()),
        "n_excluded": int((~ok).sum()),
        "valid": bool(ok.mean() >= MIN_REPRODUCED_FRACTION),
        "control_torch_vs_stored_all": cos_stats(ctrl),
        "onnx_vs_stored_all_items": cos_stats(all_items),
        "onnx_vs_torch_all_items": cos_stats(cosines(onnx_v, torch_v)),
    }
    if ok.any():
        out.update(cos_stats(all_items[ok]))
        out["pass"] = bool(all_items[ok].min() >= MIN_COS)
    else:
        out["pass"] = False
    return out


def brand_texts(processed_dir: Path) -> dict[str, str]:
    """article_id -> search_text from every data/processed/<brand>/catalogue.parquet that has a
    data/processed/clip/<brand> index (the CLIP builder's input; later brands win on id clashes)."""
    import pandas as pd

    out: dict[str, str] = {}
    for d in sorted((processed_dir / "clip").iterdir()):
        cat = processed_dir / d.name / "catalogue.parquet"
        if d.name == "unified" or not cat.exists():
            continue
        df = pd.read_parquet(cat, columns=["article_id", "search_text"])
        out.update(zip(df["article_id"].astype(str), df["search_text"].fillna(""), strict=True))
    return out


def load_queries(queries_dir: Path) -> list[str]:
    """The `text` field of every q*.json fixture, in file order."""
    return [json.loads(p.read_text(encoding="utf-8"))["text"] for p in sorted(queries_dir.glob("q*.json"))]


def _torch_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:  # noqa: C901 - one linear measurement script
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--clip-dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True,
                    help="directory of q*.json fixtures, or a JSON list of query strings")
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "onnx_equivalence")
    ap.add_argument("--n-text", type=int, default=2000)
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--brand-catalogues", type=Path, default=None,
                    help="data/processed dir with per-brand catalogue.parquet + clip/<brand> "
                         "(the text the CLIP builder encoded); default: unified catalogue text")
    ap.add_argument("--write-fixture", type=Path, default=None,
                    help="write the first --fixture-n sample items (text + stored vectors) as .npz")
    ap.add_argument("--fixture-n", type=int, default=20)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="exit 0 when every check that ran passed and only the image checks were skipped")
    ap.add_argument("--note", default=os.environ.get("GATE_CONTAINER", ""),
                    help="free text recorded in the report (image id / container details)")
    args = ap.parse_args()

    import faiss
    import pandas as pd

    from src.retrieval.onnx_encoders import OnnxCLIPEncoder, OnnxTextEncoder

    have_torch = _torch_available()
    report: dict[str, Any] = {"rule": "evals/onnx_equivalence_rule.md", "seed": SEED,
                              "torch_available": have_torch, "note": args.note}

    # ---- text tower: MiniLM vs stored dense vectors -----------------------------------------
    cat = pd.read_parquet(args.index_dir / "catalogue.parquet", columns=["article_id", "search_text"])
    text_of = dict(zip(cat["article_id"].astype(str), cat["search_text"].fillna(""), strict=True))
    dense = faiss.read_index(str(args.index_dir / "dense.faiss"))
    dense_ids = np.load(args.index_dir / "dense_article_ids.npy", allow_pickle=True).astype(str)
    pos = sample_positions(len(dense_ids), args.n_text)
    missing = [i for i in dense_ids[pos] if i not in text_of]
    if missing:
        sys.exit(f"{len(missing)} sampled article_ids missing from catalogue.parquet; cannot rebuild text")
    texts = [text_of[i] for i in dense_ids[pos]]
    stored = np.stack([dense.reconstruct(int(p)) for p in pos])

    onnx_text = OnnxTextEncoder(MINILM)
    onnx_vecs = onnx_text.encode(texts, batch_size=64, normalize_embeddings=True)

    torch_text = None
    if have_torch:
        from sentence_transformers import SentenceTransformer
        torch_text = SentenceTransformer(MINILM, revision=eo.MINILM_REVISION, device="cpu")
        torch_vecs = torch_text.encode(texts, normalize_embeddings=True, batch_size=64)
        report["C1_minilm_vs_stored"] = controlled_stats(onnx_vecs, torch_vecs, stored)
        reproduced = np.asarray(cosines(torch_vecs, stored) >= MIN_COS)
    else:
        report["C1_minilm_vs_stored"] = "NOT RUN (torch not installed: no validity control)"
        reproduced = np.ones(len(texts), dtype=bool)

    # ---- text tower: CLIP text vs stored clip vectors ---------------------------------------
    clip_idx = faiss.read_index(str(args.clip_dir / "clip.faiss"))
    clip_ids = np.load(args.clip_dir / "clip_article_ids.npy", allow_pickle=True).astype(str)
    clip_pos_of = {a: i for i, a in enumerate(clip_ids)}
    keep = [i for i, a in enumerate(dense_ids[pos]) if a in clip_pos_of]
    report["clip_sample_missing_from_clip_index"] = len(pos) - len(keep)
    # The unified CLIP index was concatenated from per-brand CLIP indices built BEFORE the
    # catalogue clean-up that rewrote search_text, so the per-brand catalogues hold the text the
    # CLIP builder actually encoded (rule amendment 1). Without them, unified text is used and
    # the controlled-subset logic below drops the items whose text changed.
    btext = brand_texts(args.brand_catalogues) if args.brand_catalogues else {}
    ctext_source = "per-brand catalogues (CLIP builder input)" if btext else "unified catalogue"
    ctexts = [btext.get(dense_ids[pos][i], texts[i]) for i in keep]
    cids = [dense_ids[pos][i] for i in keep]
    cstored = np.stack([clip_idx.reconstruct(clip_pos_of[a]) for a in cids])
    onnx_clip = OnnxCLIPEncoder(CLIP)
    onnx_cvecs = onnx_clip.encode_texts(ctexts)

    torch_clip = None
    creproduced = np.ones(len(ctexts), dtype=bool)
    if have_torch:
        import src.retrieval.clip_encoder as clip_mod
        from sentence_transformers import SentenceTransformer
        from src.retrieval.clip_encoder import CLIPEncoder

        # the exact production torch path (incl. truncation patch), at the pinned revision
        clip_mod._load_sentence_transformer = lambda mid: SentenceTransformer(
            mid, revision=eo.CLIP_REVISION, device="cpu")
        torch_clip = CLIPEncoder(CLIP)
        torch_cvecs = torch_clip.encode_texts(ctexts)
        report["C2_clip_text_vs_stored"] = {
            **controlled_stats(onnx_cvecs, torch_cvecs, cstored), "text_source": ctext_source}
        creproduced = np.asarray(cosines(torch_cvecs, cstored) >= MIN_COS)
    else:
        report["C2_clip_text_vs_stored"] = "NOT RUN (torch not installed: no validity control)"

    if args.write_fixture is not None:
        k = args.fixture_n
        mi, ci = np.flatnonzero(reproduced)[:k], np.flatnonzero(creproduced)[:k]
        np.savez_compressed(
            args.write_fixture, article_ids=dense_ids[pos][mi], texts=np.array(texts)[mi],
            minilm_stored=stored[mi].astype(np.float32),
            clip_texts=np.array(ctexts)[ci], clip_stored=cstored[ci].astype(np.float32))

    # ---- C4: top-10 overlap over the 60 queries ---------------------------------------------
    queries = (load_queries(args.queries) if args.queries.is_dir()
               else json.loads(args.queries.read_text(encoding="utf-8")))
    report["n_queries"] = len(queries)
    if have_torch and queries:
        qa = torch_text.encode(queries, normalize_embeddings=True)  # type: ignore[union-attr]
        qb = onnx_text.encode(queries, normalize_embeddings=True)
        report["C4_dense_query_vectors_cosine"] = cos_stats(cosines(qa, qb))
        ov = overlap_stats(topk_ids(dense, dense_ids, qa), topk_ids(dense, dense_ids, qb))
        report["C4_dense_top10_overlap"] = {**ov, "pass": bool(ov["mean"] >= MIN_OVERLAP)}
        ca = torch_clip.encode_texts(queries)  # type: ignore[union-attr]
        cb = onnx_clip.encode_texts(queries)
        ov = overlap_stats(topk_ids(clip_idx, clip_ids, ca), topk_ids(clip_idx, clip_ids, cb))
        report["C4_clip_text_top10_overlap"] = {**ov, "pass": bool(ov["mean"] >= MIN_OVERLAP)}
    else:
        report["C4_dense_top10_overlap"] = "NOT RUN (needs torch and queries)"
        report["C4_clip_text_top10_overlap"] = "NOT RUN (needs torch and queries)"

    # ---- C3 + image overlap: 200 images, torch reference ------------------------------------
    if args.skip_images or args.images_dir is None or not have_torch:
        report["C3_clip_image_vs_torch"] = "NOT RUN (images skipped or torch missing)"
        report["C4_clip_image_top10_overlap"] = "NOT RUN (images skipped or torch missing)"
    else:
        from PIL import Image
        files = sorted(args.images_dir.rglob("*.jpg"))
        chosen = random.Random(SEED).sample(files, min(args.n_images, len(files)))
        a_list, b_list = [], []
        for f in chosen:
            with Image.open(f) as im:
                im.load()
                a_list.append(torch_clip.encode_image(im))  # type: ignore[union-attr]
                b_list.append(onnx_clip.encode_image(im))
        a, b = np.stack(a_list), np.stack(b_list)
        c3 = cosines(a, b)
        report["C3_clip_image_vs_torch"] = {**cos_stats(c3), "pass": bool(c3.min() >= MIN_COS),
                                            "source": "H&M product images (primary checkout, read only)",
                                            "reference": "torch SentenceTransformer, NOT a stored vector"}
        ov = overlap_stats(topk_ids(clip_idx, clip_ids, a), topk_ids(clip_idx, clip_ids, b))
        report["C4_clip_image_top10_overlap"] = {**ov, "pass": bool(ov["mean"] >= MIN_OVERLAP)}

    # ---- environment + verdict --------------------------------------------------------------
    import onnxruntime
    import tokenizers
    env = {"platform": platform.platform(), "python": platform.python_version(),
           "cpu_count": os.cpu_count(), "onnxruntime": onnxruntime.__version__,
           "tokenizers": tokenizers.__version__, "faiss": faiss.__version__}
    if have_torch:
        import torch
        import transformers
        env.update(torch=torch.__version__, transformers=transformers.__version__)
    mdir = Path(os.environ.get("ONNX_MODEL_DIR") or ROOT / "onnx_models")
    env["models"] = {p.parent.name: json.loads(p.read_text()) for p in sorted(mdir.glob("*/manifest.json"))}
    report["environment"] = env

    gates = [k for k in report if k.startswith(("C1", "C2", "C3", "C4")) and isinstance(report[k], dict)
             and "pass" in report[k]]
    not_run = [k for k in report if k.startswith(("C1", "C2", "C3", "C4")) and isinstance(report[k], str)]
    invalid = [k for k in gates if report[k].get("valid") is False]
    report["not_run"] = not_run
    report["verdict"] = (
        f"INVALID ({', '.join(invalid)}: torch reproduces < {MIN_REPRODUCED_FRACTION:.0%} of the "
        "stored vectors from the rebuilt text, so the stored reference is unusable)"
        if invalid else
        "PASS" if all(report[k]["pass"] for k in gates) and not not_run else
        "FAIL" if any(not report[k]["pass"] for k in gates) else "INCOMPLETE")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "onnx_equivalence.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "onnx_equivalence.md").write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    image_only = set(not_run) <= {"C3_clip_image_vs_torch", "C4_clip_image_top10_overlap"}
    ok_incomplete = args.allow_incomplete and report["verdict"] == "INCOMPLETE" and image_only
    return 0 if report["verdict"] == "PASS" or ok_incomplete else 1


def render_md(r: dict[str, Any]) -> str:
    """Markdown summary of the report dict."""
    def row(name: str, v: Any) -> str:
        if isinstance(v, str):
            return f"| {name} | {v} | | | | |"
        if "n_sample" in v:
            return (f"| {name} | sample {v['n_sample']}, reproduced by torch {v['n_reproduced_by_torch']} "
                    f"(excluded {v['n_excluded']}) | n={v['n']} min {v['min']:.7f} "
                    f"| p1 {v['p1']:.7f} | median {v['median']:.7f} | {'PASS' if v['pass'] else 'FAIL'} |")
        if "min" in v and "mean" not in v:
            res = ("PASS" if v["pass"] else "FAIL") if "pass" in v else "info"
            return (f"| {name} | n={v['n']} | min {v['min']:.7f} | p1 {v['p1']:.7f} "
                    f"| median {v['median']:.7f} | {res} |")
        return (f"| {name} | n={v['n']} | mean {v['mean']:.4f} | min {v['min']:.2f} "
                f"| below-1: {v['n_below_1']} | {'PASS' if v['pass'] else 'FAIL'} |")

    lines = ["# ONNX encoder equivalence report", "", f"Verdict: **{r['verdict']}**", "",
             f"Rule: `{r['rule']}`. Seed {r['seed']}. Note: {r['note'] or '-'}", "",
             "| check | n | stat | stat | stat | result |", "|---|---|---|---|---|---|"]
    for k, v in r.items():
        if k.startswith(("C1", "C2", "C3", "C4")):
            lines.append(row(k, v))
            if isinstance(v, dict) and "n_sample" in v:
                for sub in ("control_torch_vs_stored_all", "onnx_vs_stored_all_items",
                            "onnx_vs_torch_all_items"):
                    lines.append(row(f"&nbsp;&nbsp;{k.split('_')[0]} {sub}", v[sub]))
    lines += ["", "```json", json.dumps(r["environment"], indent=2), "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
