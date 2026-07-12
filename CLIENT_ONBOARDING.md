# Client Onboarding Guide

Getting a new brand live on the Shopping Assistant platform.

---

## 1. Overview

The Shopping Assistant is a white-label conversational commerce layer for D2C fashion brands.
Shoppers ask natural-language questions ("kurtas for office wear under ₹1500", "build an outfit
around this jacket") and the assistant retrieves relevant items from your catalogue, handles
multi-turn refinement, compares products, and builds outfit bundles — all streamed in real time.

The Docker image is brand-agnostic. All brand configurations are baked into the image at build
time. The `BRAND` environment variable selects which configuration activates at container startup.
Each brand gets its own Cloud Run service (same image, different env vars), so deployments are
fast and you never need to rebuild the image for catalogue or config changes.

The three external inputs that make a deployment brand-specific are:

| Input | What it controls |
|---|---|
| `BRAND` env var | Which `brands/{slug}.yaml` config is loaded |
| `INDEX_STORE_URI` env var | Where the FAISS + BM25 indices are fetched from |
| Catalogue CSV | The product data the indices are built from |

---

## 2. Prerequisites

Before starting, make sure you have:

- **gcloud CLI** installed and authenticated:
  ```bash
  gcloud auth login
  gcloud auth configure-docker asia-south1-docker.pkg.dev
  ```
- **Docker** installed locally (needed only to verify the image locally before deploying)
- **Python 3.11** (needed to run the index-build script locally)
- **Your product catalogue** in CSV format — see Section 3 for the required column spec
- **A Google Cloud project** with these APIs enabled:
  - Cloud Run (`run.googleapis.com`)
  - Artifact Registry (`artifactregistry.googleapis.com`)
  - Cloud Storage (`storage.googleapis.com`) — for hosting the indices

---

## 3. Prepare Your Catalogue CSV

### Required columns

| Column | Description | Example value |
|---|---|---|
| `id` | Unique product identifier; used as the retrieval key | `SKU-0042`, `10023456` |
| `name` | Product display name shown in the card UI | `Slim Fit Kurta` |
| `type` | Product type / category; used by the outfit bundler to classify seed items | `Kurta`, `Dress`, `Jeans` |
| `colour` | Primary colour name; used for colour-filter faceting | `Navy Blue`, `Olive Green` |
| `department` | Gender/audience segment | `Men`, `Women`, `Unisex` |
| `description` | Free-text product description; drives semantic search quality | `A relaxed linen kurta...` |
| `price_inr` | Price in Indian Rupees as a plain number (no currency symbol) | `1299`, `2499.00` |
| `handle` | URL slug used to construct the PDP deep-link via `pdp_url_template` | `slim-fit-kurta-navy` |

**The `handle` column is critical for buy links.** It is substituted into the `pdp_url_template`
field in your brand config (see Section 4) to produce clickable product page URLs. If your site
uses numeric IDs instead of slugs, put the numeric ID in `handle` and adjust the template
accordingly (e.g. `https://yourbrand.com/p/{handle}`).

### Optional columns and defaults

| Column | Default if absent | Notes |
|---|---|---|
| `image_url` | No thumbnail shown | Full URL to product image; 300 px wide recommended |
| `gender` | Derived from `department` | Override if `department` values don't map cleanly |
| `label` | Empty | Promotional badge (e.g. "New Arrival", "Sale") |

### File location

Save the CSV as:

```
data/samples/{your-brand-slug}_feed.csv
```

Any path works as long as `catalogue_path` in your brand config points to it (see Section 4).
The `data/samples/` convention keeps catalogue files consistent across brands.

---

## 4. Create a Brand Config

Copy the bundled Indian brand starter config as your template:

```bash
cp brands/sample_in.yaml brands/{your-brand-slug}.yaml
```

Open `brands/{your-brand-slug}.yaml` and fill in each field:

```yaml
display_name: "Your Brand Name"     # Shown in the UI header and Buy CTA button
logo_url: null                       # Optional: URL to your logo image
primary_colour: "#FF5A00"            # Hex colour; used for UI theme (button, header accent)
accent_colour: "#1A1A2E"             # Hex colour; used for secondary UI elements
tagline: "Your brand tagline"        # Shown in the empty-state placeholder before first query
currency: "INR"                      # "INR" for Indian brands; used to format price display
locale: "en-IN"                      # BCP-47 locale tag
sizing_system: "IN"                  # "IN" for Indian sizes, "EU" for European, "alpha" for XS/S/M/L/XL
catalogue_path: "data/samples/{your-brand-slug}_feed.csv"  # Path to your CSV (relative to repo root)
pdp_url_template: "https://yourbrand.com/products/{handle}"  # {handle} is replaced per item
suggestion_chips:
  - "Kurtas for office wear"         # 4–6 sample queries shown on the empty state
  - "Festive outfits under ₹2000"
  - "Casual western wear"
  - "Summer dresses"
```

**Field reference:**

- `display_name` — the brand name rendered in the chat header and on the "Buy" CTA button.
- `primary_colour` / `accent_colour` — hex values injected as CSS custom properties for theming.
- `tagline` — the subtitle shown before the user sends their first message.
- `currency` — controls price formatting throughout the UI. Use `"INR"` for Indian Rupees.
- `sizing_system` — controls how size labels are displayed. `"IN"` shows Indian standard sizes
  (28/30/32 for bottoms, S/M/L in Indian sizing); `"EU"` shows European sizes; `"alpha"` shows
  XS/S/M/L/XL.
- `catalogue_path` — path to the CSV prepared in Section 3, relative to the repo root.
- `pdp_url_template` — the product detail page URL pattern. `{handle}` is replaced with each
  item's `handle` value. Test this manually by substituting one known handle value.
- `suggestion_chips` — 4 to 6 short query strings that appear as clickable chips on the empty
  state screen. Write these as your shoppers would type them.

Once the file is saved, add your brand slug to the `brands/` directory. The image build step
will automatically pick it up.

---

## 5. Build the Retrieval Indices

The assistant uses a hybrid retrieval system (FAISS dense index + BM25 sparse index) built from
your catalogue CSV. You need to build these indices once, and rebuild them whenever the catalogue
changes.

```bash
python scripts/01_build_retrieval.py --brand {your-brand-slug}
```

This script loads your catalogue CSV, adapts it to the internal schema, builds a FAISS
`IndexFlatIP` index using `all-MiniLM-L6-v2` embeddings and a BM25Okapi sparse index, then
writes both to disk alongside a processed catalogue parquet file.

**Output:**

```
data/processed/{your-brand-slug}/
├── faiss.index       # Dense vector index
├── bm25.pkl          # Serialised BM25 model
└── catalogue.parquet # Processed catalogue for facet filtering and display
```

On a CPU machine, expect roughly 2–5 minutes for a 10,000-item catalogue.

> **Tip:** Re-run this script every time your catalogue CSV changes (new products, price updates,
> description edits). The indices are derived entirely from the CSV.

---

## 6. Upload Indices to GCS

The Docker image does not bundle index files — they are large, catalogue-specific, and change
independently of the application code. At container startup, the service downloads the indices
from a Google Cloud Storage URI specified by `INDEX_STORE_URI`.

**Create a GCS bucket (once per project):**

```bash
gcloud storage buckets create gs://{your-bucket-name} \
  --location=asia-south1 \
  --uniform-bucket-level-access
```

**Upload the built indices:**

```bash
gsutil -m cp -r data/processed/{your-brand-slug}/ gs://{your-bucket-name}/{your-brand-slug}/
```

**Set the env var on your Cloud Run service:**

```
INDEX_STORE_URI=gs://{your-bucket-name}/{your-brand-slug}/
```

The service will download all files from that URI into a local cache directory on startup.
On subsequent restarts, the same URI is re-fetched — so deploying a catalogue update is:
rebuild indices → upload to GCS → restart the Cloud Run service.

> **Local dev alternative:** If you are running the service locally (not on Cloud Run), you can
> skip GCS entirely and set `INDEX_STORE_URI` to a local directory path:
> `INDEX_STORE_URI=data/processed/{your-brand-slug}/`

---

## 7. Deploy to Cloud Run

Each brand runs as its own Cloud Run service. The service is configured entirely through
environment variables — no code changes needed.

**Required environment variables:**

| Variable | Value | Notes |
|---|---|---|
| `BRAND` | `{your-brand-slug}` | Must match the filename in `brands/` (without `.yaml`). See [BRANDS.md](BRANDS.md) for all available slugs. |
| `INDEX_STORE_URI` | `gs://{bucket}/{brand}/` | GCS path to the uploaded indices. Also accepts a local path (`data/processed/{brand}/`) for dev. |
| `GROQ_API_KEY` | `gsk_...` | Groq API key for LLM calls |
| `DATABASE_URL` | `postgresql://...` | Postgres connection string for session storage |
| `SUPABASE_URL` | `https://....supabase.co` | Supabase project URL |
| `SUPABASE_ANON_KEY` | `eyJ...` | Supabase anonymous key |
| `SUPABASE_SERVICE_KEY` | `eyJ...` | Supabase service role key |

Store secrets in Google Secret Manager and reference them from the Cloud Run service using
Secret Manager mounts — do not paste raw API keys into the Cloud Run console.

**Full deployment steps are in [`DEPLOY.md`](DEPLOY.md).** This section covers only the
brand-specific configuration.

**Per-brand deployment summary:**

```bash
# 1. Deploy the service (replace placeholders)
gcloud run deploy shopping-assistant-{your-brand-slug} \
  --image {REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/shopping-assistant:latest \
  --region asia-south1 \
  --set-env-vars BRAND={your-brand-slug} \
  --set-env-vars INDEX_STORE_URI=gs://{your-bucket}/{your-brand-slug}/ \
  --update-secrets GROQ_API_KEY=groq-api-key:latest \
  --update-secrets DATABASE_URL=database-url:latest \
  --update-secrets SUPABASE_URL=supabase-url:latest \
  --update-secrets SUPABASE_ANON_KEY=supabase-anon-key:latest \
  --update-secrets SUPABASE_SERVICE_KEY=supabase-service-key:latest \
  --min-instances 1 \
  --memory 2Gi
```

Each brand gets a separate Cloud Run service (`shopping-assistant-{slug}`), separate GCS bucket
path, and separate env vars. The Docker image is shared.

---

## 8. Verify the Deployment

After the service is live, run these checks:

**Unit test suite:**

Run the backend unit tests on a fresh checkout (no pre-built index files required):

```bash
pytest -m "not requires_ollama"
```

A clean result on a fresh checkout looks like:

```
106 passed, 50 skipped, 0 errors, 0 failures
```

Tests that need pre-built FAISS/BM25 indices skip automatically when those files are absent (the test suite detects their absence via `tests/conftest.py`). Tests marked `requires_ollama` skip when Ollama is not running. Both sets run normally in a local environment after you run `python scripts/01_build_retrieval.py`.

**Health probes:**

```bash
# Liveness probe
curl https://{your-service-url}/healthz

# Readiness probe (confirms indices are loaded)
curl https://{your-service-url}/readyz
```

Both should return `200 OK`. The `/readyz` probe returns `503` until the index download from GCS
completes — wait 30–60 seconds after the first startup before checking.

**Manual UI checks:**

Open the frontend and verify:

- [ ] Brand name (`display_name`) appears in the header
- [ ] Suggestion chips match what you set in `suggestion_chips`
- [ ] Prices are displayed in ₹ (or your configured currency)
- [ ] Clicking "Buy" on a product card opens the correct PDP URL
- [ ] PDP URL contains the item's handle value (not a placeholder)
- [ ] A natural-language query returns results from your catalogue (not a different brand's items)

---

## 9. Worked Example: Myntra Fashion Catalogue

This section walks through a complete ingestion of the [Myntra Fashion Product Dataset](https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset) from Kaggle, serving as a concrete reference implementation for any Indian D2C brand.

### Step 1: Download the dataset

```bash
pip install kaggle   # one-time; configure ~/.kaggle/kaggle.json first
python scripts/download_myntra.py
```

The script downloads and unzips the dataset to `data/raw/myntra/`. Check the printed filename and verify it matches `catalogue_path` in `brands/myntra.yaml` (default: `data/raw/myntra/Myntra Fashion Products.csv`).

### Step 2: The brand config

`brands/myntra.yaml` is already included in the repo:

```yaml
display_name: "Myntra"
primary_colour: "#FF3F6C"    # Myntra brand pink
accent_colour:  "#282C3F"    # Myntra dark navy
currency: "INR"
locale: "en-IN"
sizing_system: "IN"
catalogue_path: "data/raw/myntra/Myntra Fashion Products.csv"
pdp_url_template: "{handle}"  # handle = full Myntra URL → direct product link
```

### Step 3: Build the full index

```bash
python scripts/01_build_retrieval.py --brand myntra
```

For a prospect-specific demo, narrow the index to a single brand:

```bash
# Single brand
python scripts/01_build_retrieval.py --brand myntra --brand-filter "Snitch"

# Curated streetwear preset (Snitch + Bewakoof + The Souled Store + Bonkers Corner)
python scripts/01_build_retrieval.py --brand myntra --brand-preset streetwear
```

Indices land in `data/processed/myntra/`.

### Step 4: Run locally

```bash
BRAND=myntra JWT_VERIFICATION_DISABLED=true uvicorn api.main:app --reload
```

The assistant now serves Myntra products with:
- Real INR prices (no synthetic values)
- Working `price_min` / `price_max` filtering
- "Buy" CTAs that open actual Myntra product pages

### Column mapping reference

| Myntra CSV column | Internal field | Notes |
|---|---|---|
| `title` | `prod_name` | Product display name |
| `price` | `price_inr` | Cleaned from "₹1,299" → 1299.0 |
| `color` | `colour_group_name` | Title-cased |
| `sub_category` | `product_type_name` | Falls back to `category` |
| `description` | `detail_desc` | Drives semantic search |
| `image_urls` | `image_url` | First URL extracted |
| `url` | `pdp_handle` | Full Myntra URL |
| `brand_name` | (filter only) | Used by `--brand-filter` / `--brand-preset` |

---

## 10. Shopify Stores (Automatic Ingestion)

Any brand running Shopify with a public `/products.json` endpoint can be onboarded without
any custom ETL code. The `download_shopify.py` script handles discovery, pagination, and
normalisation into the standard catalogue schema.

### How it works

Shopify exposes a read-only public API at `https://{domain}/products.json`. The script:

1. **Checks `robots.txt`** before fetching — skips if `/products.json` is disallowed.
2. **Probes** `?limit=1` to verify the endpoint is a real Shopify store (returns JSON with a `"products"` key containing `"handle"` fields).
3. **Paginates** using `?page=N&limit=250` until the response is empty or the 60-page cap (15 000 items) is reached. Pauses 500 ms between pages.
4. **Normalises** each product to the standard catalogue schema and saves as a CSV.

The output CSV uses the same column names as a manually prepared feed (Section 3), so the
generic adapter and build script work without any additional configuration.

### Step 1: Download

```bash
python scripts/download_shopify.py --domain yourbrand.com
```

Output: `data/raw/shopify/yourbrand/products.csv`

The script prints a summary: total products, top categories, and price range.

### Step 2: Create a brand config

```bash
cp brands/snitch.yaml brands/yourbrand.yaml
```

Edit the fields:

```yaml
display_name: "Your Brand Name"
primary_colour: "#HEX"
accent_colour: "#HEX"
tagline: "Brand tagline"
currency: "INR"
locale: "en-IN"
sizing_system: "IN"
catalogue_path: "data/raw/shopify/yourbrand/products.csv"
pdp_url_template: "https://yourbrand.com/products/{handle}"
suggestion_chips:
  - "Query 1"
  - "Query 2"
  - "Query 3"
  - "Query 4"
```

The `{handle}` placeholder is replaced per product with the Shopify product handle (the slug
from the store's URL, e.g. `slim-fit-linen-shirt`). The resulting PDP URL matches the exact
format Shopify uses for product pages.

### Step 3: Build the index

```bash
python scripts/01_build_retrieval.py --brand yourbrand --sample 0
```

### Step 4: Verify Buy CTA URLs

Pick 3 handles from the CSV and confirm HTTP 200:

```bash
python - << 'EOF'
import requests, pandas as pd
df = pd.read_csv("data/raw/shopify/yourbrand/products.csv")
for _, row in df.head(3).iterrows():
    url = f"https://yourbrand.com/products/{row['handle']}"
    print(url, requests.get(url, timeout=10).status_code)
EOF
```

### Schema mapping

| Shopify product field | CSV column | Internal field |
|---|---|---|
| `id` | `id` | `article_id` |
| `title` | `title` | `prod_name` |
| `product_type` | `type` | `product_type_name` |
| `vendor` | `brand` | (filter only — `--brand-filter`) |
| `body_html` (stripped) | `description` | `detail_desc` |
| `variants[0].price` | `price_inr` | `price_inr` |
| `images[0].src` | `image_url` | `image_url` |
| `handle` | `handle` | `pdp_handle` |

### Probe results for known Indian D2C brands

| Domain | Status | Notes |
|---|---|---|
| snitch.co.in | ✓ 15 000 products | Men's streetwear |
| powerlook.in | ✓ 927 products | Men's smart casual |
| fashor.com | ✓ 3 618 products | Women's ethnic + western |
| virgio.com | ✓ 1 810 products | Women's sustainable fashion |
| bewakoof.com | ✗ 404 | `/products.json` not found |
| thesouledstore.com | ✗ Not Shopify | Returns HTML, not Shopify API |
| bonkerscorner.com | ✗ Timeout | Server unreachable |
| damensch.com | ✗ 404 | `/products.json` not found |

---

## 11. Updating Your Catalogue

Catalogue updates do not require a new Docker image build or a code deploy. The update cycle is:

**Step 1 — Update your CSV:**

Edit `data/samples/{your-brand-slug}_feed.csv` with new, changed, or removed products.

**Step 2 — Rebuild the indices:**

```bash
python scripts/01_build_retrieval.py --brand {your-brand-slug}
```

**Step 3 — Upload to GCS:**

```bash
gsutil -m cp -r data/processed/{your-brand-slug}/ gs://{your-bucket}/{your-brand-slug}/
```

**Step 4 — Restart the Cloud Run service:**

```bash
gcloud run services update-traffic shopping-assistant-{your-brand-slug} \
  --region asia-south1 \
  --to-latest
```

Or trigger a new revision by updating any env var. The service will download the fresh indices
on startup.

> **Note on zero-downtime updates:** Cloud Run keeps the old revision handling traffic until the
> new revision passes its readiness probe. Because index loading happens at startup, the old
> revision continues serving requests while the new one warms up — catalogue updates are
> effectively zero-downtime.
