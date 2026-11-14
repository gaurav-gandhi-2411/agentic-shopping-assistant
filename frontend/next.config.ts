import type { NextConfig } from "next"
import { withSentryConfig } from "@sentry/nextjs"
import { EMBED_ALLOWED_ANCESTORS } from "./config/embed-allowlist"

// ---------------------------------------------------------------------------
// Embed-route CSP: frame-ancestors allowlist.
//
// Only THIS origin ('self') and explicitly listed brand storefront domains may
// embed the /embed/* route in an <iframe>.  Any other origin is blocked.
//
// The mock-PDP demo pages live at /pdp-demo/[brand] on THIS app
// (asa-stylist.vercel.app), so they frame /embed/* same-origin — 'self'
// covers them without needing a Vercel wildcard.
//
// To add a client domain: edit config/embed-allowlist.ts and redeploy.
//
// Note: X-Frame-Options is intentionally NOT set on /embed/* — it would block
// all framing. It is only set as a fallback on other routes.
// ---------------------------------------------------------------------------
const EMBED_FRAME_ANCESTORS = ["'self'", ...EMBED_ALLOWED_ANCESTORS].join(" ")

const nextConfig: NextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [{ protocol: "https", hostname: "**" }],
  },
  // /demo and /demo/chat used to be the anonymous entry points; `/` is now the assistant
  // itself (Wave 1b V2), so old links, shared URLs and bookmarks land there. Next carries
  // the query string over automatically. Skipped when the shelved per-brand picker is
  // switched on (NEXT_PUBLIC_MULTI_BRAND=1, see app/demo/page.tsx), which needs /demo.
  async redirects() {
    if (process.env.NEXT_PUBLIC_MULTI_BRAND === "1") return []
    return [
      { source: "/demo", destination: "/", permanent: false },
      { source: "/demo/chat", destination: "/", permanent: false },
    ]
  },
  async headers() {
    return [
      // -----------------------------------------------------------------------
      // Embed route: set frame-ancestors CSP; do NOT set X-Frame-Options.
      // -----------------------------------------------------------------------
      {
        source: "/embed/:path*",
        headers: [
          {
            key: "Content-Security-Policy",
            // frame-ancestors controls which origins may embed this page.
            // Scoped allowlist: our own origin + brand storefronts.
            value: `frame-ancestors ${EMBED_FRAME_ANCESTORS}`,
          },
          // Explicitly absent: X-Frame-Options (would conflict with CSP and
          // block legitimate embedding from our brand allowlist).
        ],
      },
      // -----------------------------------------------------------------------
      // All other routes: deny framing via X-Frame-Options (defence in depth).
      // -----------------------------------------------------------------------
      {
        source: "/((?!embed).*)",
        headers: [
          {
            key: "X-Frame-Options",
            value: "SAMEORIGIN",
          },
        ],
      },
    ]
  },
}

export default withSentryConfig(nextConfig, {
  // Suppress build-time output; no auth token means source map upload is skipped.
  silent: true,
})
