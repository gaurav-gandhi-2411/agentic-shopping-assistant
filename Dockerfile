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

# Application source and config
COPY src/ ./src/
COPY api/ ./api/
COPY config.yaml .

# Indices are loaded from GCS at startup via INDEX_STORE_URI.
# Create the empty directory so ensure_index_dir can write into it.
RUN mkdir -p data/processed

EXPOSE 8080

# Give the index-loading lifespan ~60 s before health checks start failing.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:${PORT:-8080}/healthz || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips '*'"]
