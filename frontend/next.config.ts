import type { NextConfig } from "next"
import { withSentryConfig } from "@sentry/nextjs"

// ---------------------------------------------------------------------------
// Embed-route CSP: frame-ancestors allowlist.
//
// Only our own Vercel deployment and known brand storefront domains may embed
// the /embed/* route in an <iframe>. Any other origin will be blocked by the
// browser's CSP frame-ancestors directive.
//
// Note: X-Frame-Options is intentionally NOT set on /embed/* — it would block
// all framing. It is only set as a fallback on other routes.
// ---------------------------------------------------------------------------
const EMBED_FRAME_ANCESTORS = [
  "'self'",
  // Vercel preview deployments
  "https://*.vercel.app",
  // Snitch
  "https://snitch.co.in",
  "https://*.snitch.co.in",
  // Powerlook
  "https://powerlook.in",
  "https://*.powerlook.in",
  // Fashor
  "https://fashor.com",
  "https://*.fashor.com",
  // Virgio
  "https://virgio.com",
  "https://*.virgio.com",
  // Myntra
  "https://myntra.com",
  "https://*.myntra.com",
  // Flipkart
  "https://flipkart.com",
  "https://*.flipkart.com",
].join(" ")

const nextConfig: NextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [{ protocol: "https", hostname: "**" }],
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
