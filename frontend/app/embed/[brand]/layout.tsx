import type { ReactNode } from "react"
import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Complete the Look",
  description: "AI-powered fashion styling, embedded.",
  // Prevent the embed from being indexed.
  robots: "noindex,nofollow",
}

/**
 * Embed layout: no site nav, no outer chrome.
 * Fills 100% of the iframe viewport.
 * Providers (QueryClient) are inherited from the root layout via the shared
 * `app/layout.tsx`; we don't need a new provider tree here.
 */
export default function EmbedLayout({ children }: { children: ReactNode }) {
  return (
    // h-screen + overflow-hidden keeps the embed within the iframe bounds.
    <div className="flex flex-col h-screen overflow-hidden bg-background">
      {children}
    </div>
  )
}
