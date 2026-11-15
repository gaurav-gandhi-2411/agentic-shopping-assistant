import type { MetadataRoute } from "next"
import { SITE_URL } from "@/lib/site"

// Minimal on purpose: `/` is the only indexable page. /saved-looks is a personal,
// localStorage-only page and /look/[id] are capability URLs, so neither is listed.
export default function sitemap(): MetadataRoute.Sitemap {
  return [{ url: `${SITE_URL}/` }]
}
