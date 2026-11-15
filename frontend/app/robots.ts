import type { MetadataRoute } from "next"
import { SITE_URL } from "@/lib/site"

// Allow the landing page (the working assistant); keep the sign-in flow and the
// authenticated app out of search indexes.
export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        disallow: ["/login", "/auth", "/chat", "/dashboard"],
      },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
  }
}
