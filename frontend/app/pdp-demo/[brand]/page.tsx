/**
 * Mock Product Detail Page (PDP) — sales demo artifact.
 *
 * Demonstrates exactly how a brand drops the "Complete the Look" widget on a
 * real product page. The page is static/non-functional (the "Add to Bag"
 * button does nothing) — the entire point is to show the widget trigger and
 * iframe experience in realistic context.
 *
 * The widget.js script is included via a regular <script> tag in the <head>
 * (handled by next/script) with data-brand set to the current brand.
 */

import type { Metadata } from "next"
import Script from "next/script"
import { AddToBagButton } from "@/components/pdp/AddToBagButton"

// ---------------------------------------------------------------------------
// Brand-specific product catalogue snapshots
// ---------------------------------------------------------------------------

interface ProductMock {
  name: string
  brand: string
  description: string
  price: string
  mrp: string
  discount: string
  imageUrl: string
  sizes: string[]
  accentHex: string
  brandBg: string
  category: string
  sku: string
}

// exempt from design-token audit: accentHex/brandBg values are third-party
// retailer brand colors (Snitch/Myntra/Flipkart) or a neutral demo fallback
// simulating a generic storefront — not this app's design tokens.
const PRODUCTS: Record<string, ProductMock> = {
  snitch: {
    name: "Milano Slim-Fit Blazer",
    brand: "Snitch",
    description:
      "A refined Italian-inspired slim-fit blazer in midnight navy. " +
      "Crafted from premium poly-viscose stretch fabric for all-day comfort. " +
      "Notch lapel, two-button closure, side vents, and a half-canvas chest. " +
      "Perfect for business casual, date night, or cocktail occasions.",
    price: "₹3,499",
    mrp: "₹5,999",
    discount: "42% off",
    // Real Snitch blazer product shot (catalogue item 4mbz0016) — the previous
    // Unsplash photo was a man's HEADSHOT, not a blazer (sweep 2026-07-10, P2-13).
    imageUrl:
      "https://cdn.shopify.com/s/files/1/0420/7073/7058/files/4mbz0016-01_1.jpg?v=1770377146",
    sizes: ["S", "M", "L", "XL", "XXL"],
    accentHex: "#1a1a2e",
    brandBg: "#f0f0f8",
    category: "Blazers",
    sku: "SNCH-BLZ-0042",
  },
  myntra: {
    name: "Anokhi Block Print Kurta",
    brand: "Myntra",
    description:
      "Hand block-printed cotton kurta in a classic indigo-on-white Rajasthani motif. " +
      "Straight fit, mandarin collar, side slits for easy movement. " +
      "Pairs beautifully with dhoti pants, palazzo trousers, or slim churidars. " +
      "OEKO-TEX certified fabric; machine washable.",
    price: "₹1,199",
    mrp: "₹2,499",
    discount: "52% off",
    imageUrl:
      "https://images.unsplash.com/photo-1583391733956-6c78276477e2?w=600&q=80&fit=crop",
    sizes: ["XS", "S", "M", "L", "XL"],
    accentHex: "#ff3f6c",
    brandBg: "#fff0f4",
    category: "Kurtas",
    sku: "MYN-KRT-7821",
  },
  flipkart: {
    name: "Roadster Jogger Chinos",
    brand: "Flipkart Fashion",
    description:
      "Stretch twill jogger-chinos in olive grey. Features an elastic waistband " +
      "with drawstring, tapered leg, and two side zip pockets. " +
      "The hybrid construction means you get the look of chinos with " +
      "the comfort of athleisure — ideal for WFH, errands, or casual evenings out.",
    price: "₹799",
    mrp: "₹1,499",
    discount: "47% off",
    imageUrl:
      "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=600&q=80&fit=crop",
    sizes: ["28", "30", "32", "34", "36", "38"],
    accentHex: "#2874f0",
    brandBg: "#f0f5ff",
    category: "Trousers",
    sku: "FLK-TRS-3301",
  },
}

// exempt from design-token audit: neutral demo fallback simulating a generic
// storefront's own accent color, not this app's design tokens.
const FALLBACK_PRODUCT: ProductMock = {
  name: "Sample Product",
  brand: "Brand",
  description: "A great product for any occasion.",
  price: "₹999",
  mrp: "₹1,999",
  discount: "50% off",
  imageUrl:
    "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=600&q=80&fit=crop",
  sizes: ["S", "M", "L", "XL"],
  accentHex: "#111827",
  brandBg: "#f9fafb",
  category: "Clothing",
  sku: "DEMO-001",
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ brand: string }>
}): Promise<Metadata> {
  const { brand } = await params
  const product = PRODUCTS[brand] ?? FALLBACK_PRODUCT
  return {
    title: `${product.name} — ${product.brand} | Demo PDP`,
    description: product.description,
    robots: "noindex,nofollow",
  }
}

export default async function MockPdpPage({
  params,
}: {
  params: Promise<{ brand: string }>
}) {
  const { brand } = await params
  const product = PRODUCTS[brand] ?? FALLBACK_PRODUCT

  return (
    <>
      {/*
        The single-line embed snippet a brand drops on their site.
        Here we point to /widget.js on this same origin so the demo is
        self-contained. In production the brand would use the full Vercel URL.
      */}
      <Script
        src="/widget.js"
        data-brand={brand}
        data-accent={product.accentHex}
        strategy="afterInteractive"
      />

      <div
        className="min-h-screen"
        style={{ backgroundColor: product.brandBg, fontFamily: "Inter, system-ui, sans-serif" }}
      >
        {/* ------------------------------------------------------------------ */}
        {/* Simulated storefront nav */}
        {/* ------------------------------------------------------------------ */}
        <nav
          className="sticky top-0 z-10 border-b shadow-sm"
          style={{ backgroundColor: product.accentHex }}
        >
          <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
            <span className="text-white font-bold text-lg tracking-tight">
              {product.brand}
            </span>
            <div className="hidden sm:flex items-center gap-6 text-white/80 text-sm">
              <span className="hover:text-white cursor-default">Men</span>
              <span className="hover:text-white cursor-default">Women</span>
              <span className="hover:text-white cursor-default">Sale</span>
            </div>
            <div className="flex items-center gap-3">
              {/* Bag icon placeholder */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="white"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z" />
                <line x1="3" y1="6" x2="21" y2="6" />
                <path d="M16 10a4 4 0 0 1-8 0" />
              </svg>
            </div>
          </div>
        </nav>

        {/* ------------------------------------------------------------------ */}
        {/* Breadcrumb */}
        {/* ------------------------------------------------------------------ */}
        <div className="max-w-6xl mx-auto px-4 py-3">
          <p className="text-xs text-gray-500">
            Home &rsaquo; {product.category} &rsaquo;{" "}
            <span className="text-gray-800">{product.name}</span>
          </p>
        </div>

        {/* ------------------------------------------------------------------ */}
        {/* Main PDP layout */}
        {/* ------------------------------------------------------------------ */}
        <div className="max-w-6xl mx-auto px-4 pb-16">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 lg:gap-12">
            {/* Left: product image */}
            <div className="rounded-2xl overflow-hidden bg-white shadow-sm">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={product.imageUrl}
                alt={product.name}
                className="w-full object-cover"
                style={{ aspectRatio: "4/5", objectPosition: "center top" }}
              />
            </div>

            {/* Right: product info + actions */}
            <div className="flex flex-col gap-5 pt-2">
              {/* Brand */}
              <p
                className="text-sm font-semibold uppercase tracking-widest"
                style={{ color: product.accentHex }}
              >
                {product.brand}
              </p>

              {/* Product name */}
              <h1 className="text-2xl font-bold text-gray-900 leading-snug">
                {product.name}
              </h1>

              {/* SKU */}
              <p className="text-xs text-gray-400 -mt-3">SKU: {product.sku}</p>

              {/* Price block */}
              <div className="flex items-center gap-3">
                <span className="text-2xl font-bold text-gray-900">
                  {product.price}
                </span>
                <span className="text-base text-gray-400 line-through">
                  {product.mrp}
                </span>
                <span
                  className="text-sm font-semibold px-2 py-0.5 rounded-full"
                  style={{
                    backgroundColor: `${product.accentHex}18`,
                    color: product.accentHex,
                  }}
                >
                  {product.discount}
                </span>
              </div>

              {/* Description */}
              <p className="text-sm text-gray-600 leading-relaxed">
                {product.description}
              </p>

              {/* Size selector */}
              <div>
                <p className="text-sm font-semibold text-gray-700 mb-2">
                  Select Size
                </p>
                <div className="flex flex-wrap gap-2">
                  {product.sizes.map((size, i) => (
                    <button
                      key={size}
                      className="w-10 h-10 rounded-lg border-2 text-sm font-medium transition-colors"
                      // exempt from design-token audit: this mock PDP simulates a
                      // generic third-party storefront's own UI chrome, not this
                      // app's design tokens.
                      style={
                        i === 1
                          ? {
                              borderColor: product.accentHex,
                              backgroundColor: product.accentHex,
                              color: "#fff",
                            }
                          : {
                              borderColor: "#d1d5db",
                              backgroundColor: "#fff",
                              color: "#374151",
                            }
                      }
                      aria-label={`Size ${size}`}
                    >
                      {size}
                    </button>
                  ))}
                </div>
              </div>

              {/* Add to bag (non-functional — demo only) */}
              {/* Extracted to a client component: onClick is illegal in Server Components. */}
              <AddToBagButton accentHex={product.accentHex} />

              {/*
                ============================================================
                WIDGET INJECTION POINT
                widget.js injects the "Complete the Look" button here
                (after the Add-to-Bag button, before the description).
                The div below acts as a visual anchor; the script auto-locates
                the nearest submit/add-to-cart button in the DOM.
                ============================================================
              */}
              <div id="asa-widget-anchor" />

              {/* Divider */}
              <hr className="border-gray-200" />

              {/* Feature bullets */}
              <ul className="space-y-1.5 text-sm text-gray-600">
                {["Free shipping on orders over ₹999",
                  "Easy 30-day returns",
                  "Secure payments — UPI, cards, COD"].map((feat) => (
                  <li key={feat} className="flex items-center gap-2">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="14"
                      height="14"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke={product.accentHex}
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden
                    >
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    {feat}
                  </li>
                ))}
              </ul>

              {/* Demo notice */}
              <div className="rounded-xl border border-dashed border-gray-300 bg-white/60 p-4 text-center">
                <p className="text-xs text-gray-500 leading-relaxed">
                  <strong className="text-gray-700">Demo page</strong> — the &ldquo;Complete the Look&rdquo;
                  button above is injected by{" "}
                  <code className="bg-gray-100 px-1 rounded text-[11px]">/widget.js</code>.
                  On a real storefront, brands drop one{" "}
                  <code className="bg-gray-100 px-1 rounded text-[11px]">&lt;script&gt;</code> tag.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
