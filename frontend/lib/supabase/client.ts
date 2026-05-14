import { createBrowserClient } from "@supabase/ssr"

// Used in Client Components ('use client') and browser-side event handlers.
// Creates one client per call; callers that need stability should useMemo it.
export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  )
}
