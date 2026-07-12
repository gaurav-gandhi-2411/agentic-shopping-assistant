/**
 * /pdp-demo — index page listing all mock brand PDP demos.
 * This is the "sales artifact" landing page shown to prospective brand partners.
 */

import type { Metadata } from "next"
import Link from "next/link"

export const metadata: Metadata = {
  title: "Complete the Look — Mock PDP Demos",
  description:
    "See exactly how the 'Complete the Look' widget appears on a live brand storefront.",
  robots: "noindex,nofollow",
}

// exempt from design-token audit: accentHex values are third-party retailer
// brand colors (Snitch/Myntra/Flipkart), not this app's design tokens.
const DEMO_BRANDS = [
  {
    id: "snitch",
    name: "Snitch",
    tagline: "Contemporary Indian menswear",
    product: "Milano Slim-Fit Blazer",
    accentHex: "#1a1a2e",
  },
  {
    id: "myntra",
    name: "Myntra",
    tagline: "India's fashion destination",
    product: "Anokhi Block Print Kurta",
    accentHex: "#ff3f6c",
  },
  {
    id: "flipkart",
    name: "Flipkart Fashion",
    tagline: "Shop fashion your way",
    product: "Roadster Jogger Chinos",
    accentHex: "#2874f0",
  },
]

export default function PdpDemoIndexPage() {
  return (
    <main className="min-h-screen bg-gray-50 py-16 px-4">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold text-gray-900 mb-3">
            Complete the Look — Live Demos
          </h1>
          <p className="text-gray-600 text-base leading-relaxed">
            See exactly how the widget appears on a real brand product page.
            Each demo loads an AI stylist inside an iframe — no backend changes,
            no CORS config, one &lt;script&gt; tag.
          </p>
        </div>

        <div className="space-y-4">
          {DEMO_BRANDS.map((brand) => (
            <Link
              key={brand.id}
              href={`/pdp-demo/${brand.id}`}
              className="flex items-center gap-4 rounded-2xl border bg-white shadow-sm p-5
                         hover:shadow-md transition-shadow group"
            >
              <div
                className="w-10 h-10 rounded-full shrink-0"
                style={{ backgroundColor: brand.accentHex }}
              />
              <div className="flex-1 min-w-0">
                <p className="font-semibold text-gray-900 group-hover:underline">
                  {brand.name}
                </p>
                <p className="text-sm text-gray-500 truncate">
                  {brand.tagline} · {brand.product}
                </p>
              </div>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-gray-400 shrink-0"
              >
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          ))}
        </div>

        <div className="mt-10 rounded-xl border border-dashed border-gray-300 bg-white p-6 text-center">
          <p className="text-sm text-gray-500 leading-relaxed mb-3">
            Each demo page embeds the widget with:
          </p>
          <code className="block bg-gray-100 rounded-lg px-4 py-3 text-xs text-left overflow-x-auto text-gray-700">
            {`<script src="https://stylemaitri.vercel.app/widget.js" data-brand="snitch" async></script>`}
          </code>
          <p className="text-xs text-gray-400 mt-3">
            One line. No CORS changes. No iframe config. The stylist experience loads instantly.
          </p>
        </div>
      </div>
    </main>
  )
}
