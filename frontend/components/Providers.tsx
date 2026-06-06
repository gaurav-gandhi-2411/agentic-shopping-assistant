"use client"

import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useState } from "react"
import { BrandThemeInjector } from "@/components/BrandThemeInjector"

export function Providers({ children }: { children: React.ReactNode }) {
  // Stable QueryClient across re-renders — created once per component mount.
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30_000,
            refetchOnWindowFocus: false,
          },
        },
      })
  )

  return (
    <QueryClientProvider client={queryClient}>
      <BrandThemeInjector />
      {children}
    </QueryClientProvider>
  )
}
