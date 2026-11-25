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

# Runtime requirements only. There is NO torch / transformers / sentence-transformers here: the
# text and CLIP encoders run on onnxruntime (src/retrieval/onnx_encoders.py), so the old
# CPU-torch-index workaround (which existed to stop sentence-transformers' transitive torch from
# resolving to a ~4.7GB CUDA build) is gone. torch lives only in the `export` stage below.
# --prefix=/install isolates everything pip installs so the final stage can COPY --from just that
# directory instead of the whole builder filesystem.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Export stage: the ONLY stage with torch. Exports the MiniLM and CLIP encoders to fp32 ONNX +
# tokenizer.json at pinned Hugging Face revisions (scripts/export_onnx.py; the script self-checks
# each export against torch and fails the build if cosine < 0.99999). Only /export is copied
# onward; nothing from this stage's site-packages reaches the runtime image. --index-url is the
# CPU wheel index (primary) so torch never resolves to a CUDA build; PyPI stays as extra index.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS export

WORKDIR /app
ENV HF_HOME=/tmp/hf \
    TQDM_DISABLE=1
COPY requirements-export.txt .
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements-export.txt
COPY src/retrieval/onnx_encoders.py ./src/retrieval/onnx_encoders.py
COPY scripts/export_onnx.py ./scripts/export_onnx.py
RUN touch src/__init__.py src/retrieval/__init__.py \
    && python scripts/export_onnx.py --out-dir /export

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
#   /tmp (HOME, MPLCONFIGDIR, XDG_*)   runtime  library caches (matplotlib via   any (always writable)
#                                               colour-science, numba, etc.)
#   /app/onnx_models: root-owned, read-only to the app (exported at build time, see below).
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

# Everything from here runs as the app user.
USER app

# The dense (MiniLM) and CLIP (text + vision towers) encoders as fp32 ONNX plus tokenizer.json,
# exported in the `export` stage at pinned HF revisions (recorded in each manifest.json). The
# weights are in the image, so cold-start never downloads anything (the 495s cold-start incident,
# unauthenticated HF revision-check stalls, cannot recur: there is no HF client in this image at
# all). ONNX_MODEL_DIR is read by src/retrieval/onnx_encoders.py.
COPY --from=export /export /app/onnx_models
ENV ONNX_MODEL_DIR=/app/onnx_models

# Disables tqdm's TMonitor background thread globally (belt-and-suspenders for
# src/retrieval/dense_search.py and clip_encoder.py, which also set monitor_interval = 0).
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
