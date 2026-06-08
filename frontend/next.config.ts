import type { NextConfig } from "next"
import { withSentryConfig } from "@sentry/nextjs"

const nextConfig: NextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [{ protocol: "https", hostname: "**" }],
  },
}

export default withSentryConfig(nextConfig, {
  // Suppress build-time output; no auth token means source map upload is skipped.
  silent: true,
})
