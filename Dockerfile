FROM python:3.11-slim

# libgomp1: required by faiss-cpu (OpenMP runtime)
# build-essential: needed for C-extension wheels; purged after pip install
# curl: used by the HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Baked model cache location — kept under /app so it survives layer caching
# and is owned by the non-root user we switch to at runtime.
# Both HF_HOME and SENTENCE_TRANSFORMERS_HOME point here so every path
# sentence-transformers and HuggingFace Hub resolves at runtime hits the
# pre-baked directory without any network calls.
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/sentence_transformers

# Install CPU-only torch first so sentence-transformers doesn't pull the
# ~2 GB CUDA wheel.  All other deps are pinned by requirements.txt.
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# build-essential is no longer needed after wheels are compiled.
RUN apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Bake the CLIP model weights (~350 MB) AND the dense retrieval model
# (~90 MB, sentence-transformers/all-MiniLM-L6-v2 — must match
# config.yaml retrieval.dense_model) into the image at build time so
# cold-start never triggers a runtime download.  The sentence-transformers
# cache is written to SENTENCE_TRANSFORMERS_HOME (set above).
# This layer is cached by Docker as long as requirements.txt and the torch
# install layer are unchanged.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('clip-ViT-B-32'); \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Force every runtime model load to hit the baked cache above — never the
# network.  Must be set AFTER the bake RUN so the bake step itself can still
# reach HuggingFace Hub to download weights the first time.  Without this,
# the 495s cold-start incident (unauthenticated HF revision-check stalls)
# can recur even with weights already on disk.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Disables tqdm's TMonitor background thread globally.  Without this, every
# SentenceTransformer.encode() call bootstraps/tears down a monitor thread
# whose lock waits dominated latency on constrained (1 vCPU) Cloud Run
# instances — see src/retrieval/dense_search.py and clip_encoder.py for the
# code-level fix (tqdm.tqdm.monitor_interval = 0) that also covers non-Docker
# runtimes; this env var is belt-and-suspenders for the container.
ENV TQDM_DISABLE=1

# Application source and config
COPY src/ ./src/
COPY api/ ./api/
COPY config.yaml .
COPY brands/ ./brands/

# Indices are loaded from GCS at startup via INDEX_STORE_URI.
# Create the empty directory so ensure_index_dir can write into it.
RUN mkdir -p data/processed

EXPOSE 8080

# Give the index-loading lifespan ~60 s before health checks start failing.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:${PORT:-8080}/healthz || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips '*'"]
