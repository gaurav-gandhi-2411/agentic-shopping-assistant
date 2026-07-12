# Brands Reference

Sales reference sheet — one row per demoable brand, ready to use in prospect meetings.

| Brand | BRAND= | Source | Items | Gender | Price range | Top categories | Demo query |
|---|---|---|---|---|---|---|---|
| **H&M** | `hm` | Kaggle HM dataset | ~20 K (sampled) | Men + Women | SEK | Ladieswear, Menswear, Sport | "Black slim fit jeans" |
| **Desi Drip (sample)** | `sample_in` | Bundled sample | ~500 | Mixed | ₹499–₹4,999 | Kurtas, Dresses, Basics | "Casual kurtas for office" |
| **Myntra** | `myntra` | Kaggle (hiteshsuthar101) | 14 k | Mixed | ₹169 – ₹47,999 | Kurtas, Dresses, Tops | "Black kurtas under ₹1500" |
| **Flipkart** | `flipkart` | Kaggle (aaditshukla), 15 k subsample | 15 k | Mixed | ₹99 – ₹7,799 | Topwear, Bottomwear, Winter Wear | "Winter jackets for men" |
| **Snitch** | `snitch` | snitch.co.in live | 15 k | Men | ₹219 – ₹5,699 | Shirts, T-Shirts, Jeans | "Oversized streetwear shirts" |
| **Powerlook** | `powerlook` | powerlook.in live | 927 | Men | ₹399 – ₹2,599 | Shirts, T-Shirts, Bottoms | "Formal shirts for office" |
| **Fashor** | `fashor` | fashor.com live | 3.6 k | Women | ₹499 – ₹7,799 | Kurta Sets, Dresses | "Ethnic kurta sets for wedding" |
| **Virgio** | `virgio` | virgio.com live | 1.8 k | Women | ₹359 – ₹6,298 | Dresses, Tops, Shirts | "Sustainable linen dresses" |

---

## Quick-start per brand

```bash
# H&M (bundled index — no download needed)
python scripts/01_build_retrieval.py --brand hm
BRAND=hm uvicorn api.main:app --reload

# Myntra
python scripts/download_myntra.py
python scripts/01_build_retrieval.py --brand myntra --sample 0
BRAND=myntra uvicorn api.main:app --reload

# Flipkart (JSON already downloaded by kaggle; converts to CSV automatically)
python scripts/download_flipkart.py
python scripts/01_build_retrieval.py --brand flipkart --sample 0
BRAND=flipkart uvicorn api.main:app --reload

# Snitch
python scripts/download_shopify.py --domain snitch.co.in
python scripts/01_build_retrieval.py --brand snitch --sample 0
BRAND=snitch uvicorn api.main:app --reload

# Powerlook
python scripts/download_shopify.py --domain powerlook.in
python scripts/01_build_retrieval.py --brand powerlook --sample 0
BRAND=powerlook uvicorn api.main:app --reload

# Fashor
python scripts/download_shopify.py --domain fashor.com
python scripts/01_build_retrieval.py --brand fashor --sample 0
BRAND=fashor uvicorn api.main:app --reload

# Virgio
python scripts/download_shopify.py --domain virgio.com
python scripts/01_build_retrieval.py --brand virgio --sample 0
BRAND=virgio uvicorn api.main:app --reload
```

---

## URL verification (last checked 2026-06-07)

All Buy CTA URLs confirmed HTTP 200:

| Brand | Sample PDP URL | Status |
|---|---|---|
| Flipkart | `https://www.flipkart.com/okane-checkered-men-blue-track-pants/p/itm6cadadc6fd5b1?pid=TKPFPUMH5RYSQ9V2` | 200 |
| Snitch | `https://snitch.co.in/products/abstract-oversized-shirt-4shs152-01` | 200 |
| Powerlook | `https://powerlook.in/products/white-linen-blend-zipper-shirt` | 200 |
| Fashor | `https://fashor.com/products/abstract-checks-foliage-print-straight-kurta-with-palazzo-maroon` | 200 |
| Virgio | `https://virgio.com/products/may26-brakin-100-cotton-relaxed-denim-shirt-mini-dress` | 200 |

---

## Data freshness

- **Kaggle datasets** (Myntra, Flipkart): static snapshots; re-download from Kaggle if the
  prospect needs fresher data.
- **Shopify live brands** (Snitch, Powerlook, Fashor, Virgio): re-run `download_shopify.py`
  to pull the latest catalogue. A fresh download + index rebuild takes < 10 minutes per brand.

---

## Adding a new Shopify brand

Any brand running Shopify with a public `/products.json` endpoint can be onboarded in minutes:

```bash
# 1. Probe + download
python scripts/download_shopify.py --domain newbrand.com

# 2. Create brands/newbrand.yaml  (copy any existing Shopify yaml as template)
cp brands/snitch.yaml brands/newbrand.yaml
# Edit: display_name, colours, tagline, catalogue_path, pdp_url_template, suggestion_chips

# 3. Build index
python scripts/01_build_retrieval.py --brand newbrand --sample 0

# 4. Run
BRAND=newbrand uvicorn api.main:app --reload
```

See `CLIENT_ONBOARDING.md` Section 10 for the full Shopify onboarding walkthrough.
