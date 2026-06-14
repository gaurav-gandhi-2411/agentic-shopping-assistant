/**
 * embed-allowlist.ts
 *
 * Single source of truth for the CSP frame-ancestors allowlist used on
 * /embed/* routes.  `'self'` is always prepended by next.config.ts —
 * do NOT add it here.
 *
 * HOW TO ADD A CLIENT DOMAIN
 * --------------------------
 * Append an entry to EMBED_ALLOWED_ANCESTORS below:
 *
 *   "https://client-storefront.com",
 *   "https://*.client-storefront.com",   // include if they use subdomains
 *
 * Then redeploy (the value is baked in at build time).
 * No other file needs to change.
 *
 * WHY NOT AN ENV VAR?
 * -------------------
 * next.config.ts runs at build time on Vercel. An env var works fine there
 * (EMBED_ALLOWED_ANCESTORS="https://a.com https://b.com"), but a TypeScript
 * module is easier to review, diff, and type-check.  If you need runtime
 * configurability (hot reload without redeploy) switch to the env var path.
 */

export const EMBED_ALLOWED_ANCESTORS: readonly string[] = [
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
]
