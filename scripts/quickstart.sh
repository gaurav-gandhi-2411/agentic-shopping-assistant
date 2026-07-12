#!/usr/bin/env bash
# quickstart.sh — download catalogue, build index, and start the backend.
# Run 'make frontend' (cd frontend && npm run dev) in a second terminal.
#
# Usage:
#   BRAND=snitch PORT=8080 bash scripts/quickstart.sh
#   make demo BRAND=fashor

set -euo pipefail

BRAND="${BRAND:-snitch}"
PORT="${PORT:-8080}"

echo ""
echo "==> Quick Start: BRAND=$BRAND  port=$PORT"
echo ""

# ── 1. Derive domain from brand YAML pdp_url_template ────────────────────────
DOMAIN=$(python3 - <<EOF
import yaml, sys
from urllib.parse import urlparse
try:
    cfg = yaml.safe_load(open("brands/$BRAND.yaml"))
except FileNotFoundError:
    sys.exit("brands/$BRAND.yaml not found — is '$BRAND' a valid brand slug?")
url = cfg.get("pdp_url_template", "")
if url.startswith("http"):
    print(urlparse(url).hostname)
else:
    # Non-Shopify brands (hm, myntra, flipkart) use their own download scripts.
    sys.exit("'$BRAND' is not a Shopify brand. Use the brand-specific download script, then 'make backend BRAND=$BRAND'.")
EOF
)

# ── 2. Download catalogue if not already present ──────────────────────────────
CSV_PATH="data/raw/shopify/$BRAND/products.csv"
if [ ! -f "$CSV_PATH" ]; then
    echo "==> Downloading $BRAND catalogue from $DOMAIN..."
    python scripts/download_shopify.py --domain "$DOMAIN"
else
    echo "==> Catalogue already present at $CSV_PATH"
fi

# ── 3. Build retrieval index if not already present ───────────────────────────
INDEX_PATH="data/processed/$BRAND/dense.faiss"
if [ ! -f "$INDEX_PATH" ]; then
    echo "==> Building retrieval index for $BRAND (takes ~3 min on CPU)..."
    python scripts/01_build_retrieval.py --brand "$BRAND" --sample 0
else
    echo "==> Index already present at data/processed/$BRAND/"
fi

# ── 4. Start backend ──────────────────────────────────────────────────────────
echo ""
echo "==> Starting backend on http://127.0.0.1:$PORT"
echo "    In a second terminal run:  make frontend   (or: cd frontend && npm run dev)"
echo ""

exec JWT_VERIFICATION_DISABLED=true BRAND="$BRAND" uvicorn api.main:app --port "$PORT"
