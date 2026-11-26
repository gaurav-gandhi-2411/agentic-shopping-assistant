# ONNX encoder equivalence rule (pre-registered)

Status: PRE-REGISTERED before any measurement on branch `feat/onnx-encoder` (base `ad17f47`).
Changing a number below after seeing results requires a new commit that says why, and the PR must
say the rule was changed after measurement.

Purpose: replace the PyTorch / sentence-transformers text encoder
(`src/retrieval/dense_search.py`, `sentence-transformers/all-MiniLM-L6-v2`, 384-d) and the CLIP
encoder (`src/retrieval/clip_encoder.py`, `sentence-transformers/clip-ViT-B-32`, 512-d, text and
image towers) with ONNX Runtime CPU encoders, so the production image carries no torch.
Exports are fp32 only. No fp16, no quantisation. If the bar below is not met with fp32, the PR is
DRAFT and the failure is reported; quantisation is not used to "rescue" it.

## What exists in the index (read from the code before writing this rule)

- `data/processed/unified/dense.faiss` (+ `dense_article_ids.npy`): MiniLM vectors of
  `catalogue.parquet["search_text"]`, built by `scripts/build_unified_index.py`
  (`DenseRetriever.build_index`: `encode(texts, normalize_embeddings=True)`).
- `data/processed/clip/unified/clip.faiss` (+ `clip_article_ids.npy`): CLIP **text-tower** vectors of
  `search_text` (`scripts/build_clip_index.py`). **No CLIP image vectors are stored anywhere**: the
  image tower only runs at request time, on the uploaded photo. So the image tower cannot be
  compared with a stored vector; see "Image tower" below for what is measured instead and why that
  is weaker evidence.

## Reference and samples

Reference vectors = the vectors STORED in the index, read with `faiss.Index.reconstruct`. They are
not re-encoded by torch. Index files are read from the primary checkout, read-only.

1. Text tower, MiniLM: 2,000 stored items, `np.random.default_rng(42).choice(n, 2000,
   replace=False)` over index positions. Text = `catalogue.parquet["search_text"]` for that
   article_id (joined on `dense_article_ids`), which is exactly the string the builder encodes
   (`catalogue_df["search_text"].tolist()`; no further transformation).
2. Text tower, CLIP: the same 2,000 positions against `clip.faiss`, mapped via
   `clip_article_ids.npy`; text = the same `search_text` (the CLIP builder encodes `search_text`,
   truncated at 77 tokens).
3. Validity control (not a pass criterion): torch fp32 encode of the first 200 of the sample vs
   the stored vector. If the control itself is below 0.9999 min cosine, the stored vectors were
   built from a different text or model revision than rebuilt here, the stored-vector reference is
   not trustworthy for this model, and the result is reported as INVALID rather than pass or fail.
4. Image tower, CLIP: 200 product images, seed 42 (`random.Random(42).sample` over the sorted list
   of `.jpg` files under the H&M image directory in the primary checkout, read only). The
   reference is the torch `SentenceTransformer("clip-ViT-B-32").encode(image)` vector (the code
   path production runs today), NOT a stored vector, because none exists. This is a weaker
   reference: it shows ONNX == torch on these images, not ONNX == what was indexed.
   The H&M images are not catalogue images of the live 6+ store index; they are used because they
   are the only product-image files available locally.

## Pass criteria

All must hold, measured in the Linux CPU container built from this branch's Dockerfile (not the
Windows host):

- C1  MiniLM: min cosine(ONNX vector, stored vector) >= 0.9999 over the 2,000 sample.
- C2  CLIP text tower: min cosine(ONNX vector, stored vector) >= 0.9999 over the 2,000 sample.
- C3  CLIP image tower: min cosine(ONNX, torch) >= 0.9999 over the 200 images.
- C4  Top-10 overlap: mean over the 60 stage-2 query texts (the `text` field of
      `evals/fixtures/retrieval/q01..q60.json` in `agentic-shopping-assistant-wt-work`, read only)
      of |top10(torch query vec) ∩ top10(ONNX query vec)| / 10 against the real
      `dense.faiss`, with unrestricted IndexFlatIP search, >= 0.99. The same is reported for the
      CLIP text tower against `clip.faiss`, and for the 200 image vectors against `clip.faiss`;
      each must also be >= 0.99.
      Ties: overlap is computed on article_id sets of the top 10 from each search.
- C5  Eval gate before and after: `scripts/eval_gate.py --max-unlabeled 3` (the step the CI `eval`
      job runs; thresholds in that script) exits 0 after the change, and none of r1 precision@5,
      r1 NDCG@10, intent all-fields-exact, strict gold precision@5 is lower than the
      pre-change value by more than 0.005 absolute. "Before" = the same script on base `ad17f47`
      with torch, same index, same host; "after" = this branch with ONNX. The CI `eval` job on the
      PR is the authoritative after-run on the CI runner; if a local run is infeasible that is
      stated instead of claimed.

Reported for every criterion: n, min / p1 / median cosine, the overlap distribution.

## Not part of the pass criteria (measured and reported, no bar)

Image size, container cold start to the retriever being loaded / `/readyz`, steady RSS, encode
latency p50/p95 over the 60 queries (before = base image, after = this branch image, both built
locally under Docker on Linux). Cloud Run cold-start numbers wait for a deploy and are not claimed.

## If a criterion fails

The PR is opened as DRAFT, the failing criterion and its number are stated in the title area of
the body, and no re-tuning of the bar is done to make it pass.

## Amendment 1 (made after a development smoke run, before any official measurement)

Disclosure: a development smoke run on the Windows host (n=300 text items, 20 images; NOT the
official measurement, not a Linux container, output discarded) showed the pre-registered validity
control for the CLIP text tower failing (torch vs stored, min cosine 0.945, median 1.0). Cause,
checked afterwards by comparing text: the unified `clip.faiss` is a concatenation of per-brand CLIP
indices built (July 19-24) before the catalogue clean-up that rewrote `search_text` for the unified
catalogue; 38,225 of the 112,425 stored CLIP vectors were built from per-brand text that differs
from the unified `catalogue.parquet` text (whitespace/cleaning), the other 74,200 are identical.
The pre-registered text source for CLIP ("rebuilt exactly as the index builder builds it") is
therefore the per-brand `data/processed/<brand>/catalogue.parquet`, not the unified one. The dense
(MiniLM) side was unaffected: control min cosine 0.9999999 on 200.

Changes, in force for the official run and for any later run:

1. CLIP-text source = per-brand catalogues (`--brand-catalogues`), same 2,000 sample positions.
   Without per-brand catalogues (CI: only the unified index is downloadable) the unified text is
   used and the rule below does the work.
2. The control now runs on ALL sample items (both towers), not just the first 200. C1 and C2 are
   evaluated on the items whose stored vector torch fp32 itself reproduces (cos >= 0.9999): where
   torch cannot reproduce the stored vector the indexed text is unknown and neither encoder can be
   judged against it. The number excluded, and ONNX-vs-stored / torch-vs-stored / ONNX-vs-torch
   over ALL items, are reported next to the pass/fail number so the exclusion cannot hide a
   failure. If fewer than 50% of the sample is reproduced the result is INVALID.
3. The 0.9999 bar, the 2,000 sample, the 60 queries, the 0.99 overlap bar, fp32-only and the
   Linux-container requirement are unchanged.
4. The CI `eval` job cannot run C3 (no images on the runner); it runs C1, C2, C4 on the unified
   text with `--skip-images --allow-incomplete`. C3 is a local-container result.
