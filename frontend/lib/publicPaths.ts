// Which paths an anonymous visitor may open without being redirected to /login.
// Kept free of Next/Supabase imports so it is unit-testable in plain node and so
// middleware.ts has exactly one place to change.

/** Exact paths (the landing page itself plus the crawler files). */
const PUBLIC_EXACT = ["/", "/robots.txt", "/sitemap.xml"] as const

/**
 * Path prefixes (startsWith, deliberately the same matching the middleware always used).
 *  - /login, /auth: sign-in is optional but must stay reachable.
 *  - /demo: legacy alias of `/` (redirected in next.config.ts); /embed and /pdp-demo
 *    are public widget routes that bootstrap their own anonymous demo session.
 *  - /look: capability URL (unguessable UUID) for shared looks.
 *  - /saved-looks: reads only localStorage, no backend session.
 *  - /opengraph-image: site-wide OG card, fetched by link-preview crawlers with no session.
 */
const PUBLIC_PREFIXES = [
  "/login",
  "/auth",
  "/demo",
  "/embed",
  "/pdp-demo",
  "/look",
  "/saved-looks",
  "/opengraph-image",
] as const

/** True when an unauthenticated request for `pathname` must be served, not redirected. */
export function isPublicPath(pathname: string): boolean {
  if ((PUBLIC_EXACT as readonly string[]).includes(pathname)) return true
  return PUBLIC_PREFIXES.some((prefix) => pathname.startsWith(prefix))
}
