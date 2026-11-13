# ---------------------------------------------------------------------------
# Builder stage: compiles wheels that need a C toolchain. build-essential
# (Item 108/110) never reaches the final image -- Docker layers are append-
# only, so the old single-stage Dockerfile's `apt-get purge build-essential`
# in a later RUN never actually reclaimed the ~336MB it added; the bytes stay
# baked in regardless of what a later layer deletes. A separate builder stage
# is the only way to actually drop them: this stage's filesystem is never
# copied into the final image, only the installed Python packages are.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# build-essential: needed for C-extension wheels (faiss-cpu, etc.) at
# COMPILE time only -- unlike libgomp1 below, nothing at runtime needs it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch and requirements.txt in ONE pip invocation so pip's resolver sees both
# together and dedupes torch against the CPU-only index -- confirmed broken as two separate
# `pip install --prefix=/install` invocations (Item 110, first build attempt): each RUN is a
# separate pip process, and pip's "already satisfied" check only looks at the *running
# interpreter's own* site-packages, not an arbitrary --prefix populated by a prior, separate
# invocation. sentence-transformers pins torch as an unpinned transitive dependency, so the
# second invocation had no visibility into the first's CPU-only torch, saw torch as unmet, and
# pulled a full CUDA build (torch+nvidia-*+triton, ~4.7GB of dead weight on Cloud Run's CPU-only
# runtime) from PyPI -- inflated the image to 11.9GB instead of shrinking it. --index-url as the
# PRIMARY index (not --extra-index-url) makes pip prefer the CPU wheel for torch even though it's
# resolved transitively, while --extra-index-url PyPI still resolves everything else.
# --prefix=/install isolates everything pip installs so the final stage can COPY --from just that
# directory instead of the whole builder filesystem.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt

# ---------------------------------------------------------------------------
# Final stage: runtime only. build-essential is never installed here at all,
# so there's nothing to purge and nothing left baked in.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# libgomp1: required by faiss-cpu (OpenMP runtime) -- this one IS needed at
# runtime, unlike build-essential, so it stays in the final stage.
# curl: used by the HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pulls in everything the builder stage installed via --prefix=/install --
# python:3.11-slim's own site-packages already lives under /usr/local, so
# this lands packages exactly where Python expects them, no PYTHONPATH
# changes needed.
COPY --from=builder /install /usr/local

# ---------------------------------------------------------------------------
# Non-root runtime user. Fixed uid/gid 10001 (stable across rebuilds, so a mounted volume's
# ownership does not drift), no login shell, no home directory.
#
# Write-path audit (every place the running app writes; nothing else under /app is writable):
#   path                               when     writer                          owner needed
#   /app/data/processed/unified/       runtime  index_store.ensure_index_dir    app (GCS index download)
#   /app/data/processed/clip/<brand>/  runtime  download_supplementary_assets   app (CLIP index)
#   /app/data/processed/shopify_variants/ runtime  download_supplementary_assets app (variant maps)
#   /app/.cache/huggingface/           build    SentenceTransformer bake RUN    app (baked as app, so no
#                                                                               chown layer duplicating ~450MB;
#                                                                               runtime is offline, read-only use)
#   /tmp (HOME, MPLCONFIGDIR, XDG_*)   runtime  library caches (matplotlib via   any (always writable)
#                                               colour-science, numba, etc.)
#   /app/src, api, alembic, brands, config.yaml, alembic.ini: root-owned and read-only to
#   the app (only .pyc writes fail, silently). No log files, no sqlite, no tempfile use in
#   api/ or src/; scripts/ is not in the image (.dockerignore).
# ---------------------------------------------------------------------------
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin app \
    && mkdir -p /app/.cache /app/data/processed \
    && chown app:app /app/.cache /app/data /app/data/processed

# Home does not exist for the app user; point every library that derives a cache dir from it
# at /tmp (always writable on Cloud Run and in a read-only-rootfs container with a tmpfs).
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    NUMBA_CACHE_DIR=/tmp/numba

# Everything from here runs as the app user: the model bake below creates its files owned by
# app directly, which avoids a recursive chown that would duplicate the weights in a layer.
USER app

# Baked model cache location, under /app so it survives layer caching and is owned by the
# non-root user (the bake RUN below executes as that user).
# Both HF_HOME and SENTENCE_TRANSFORMERS_HOME point here so every path
# sentence-transformers and HuggingFace Hub resolves at runtime hits the
# pre-baked directory without any network calls.
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/sentence_transformers

# Bake the CLIP model weights (~350 MB) AND the dense retrieval model
# (~90 MB, sentence-transformers/all-MiniLM-L6-v2 — must match
# config.yaml retrieval.dense_model) into the image at build time so
# cold-start never triggers a runtime download. Unchanged from before (Item
# 110/108): still baked, still offline-forced below -- that decision stands,
# it fixed a real prior 495s cold-start incident and this refactor doesn't
# touch it.
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
# Migration scripts ship in the image only so /readyz can compare the DB's alembic_version
# with this image's head (api/routes/health.py check "schema"). Nothing runs them at startup.
COPY alembic.ini .
COPY alembic/ ./alembic/

# Indices are loaded from GCS at startup via INDEX_STORE_URI.
# data/processed is created and chowned to the app user earlier (writable download target).

# Unprivileged port (>1024); Cloud Run overrides via $PORT.
EXPOSE 8080

# Give the index-loading lifespan ~60 s before health checks start failing.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:${PORT:-8080}/healthz || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips '*'"]
