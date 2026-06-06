"use client"

import { useEffect } from "react"
import { useBrandConfig } from "@/hooks/useBrandConfig"

/**
 * Injects brand primary/accent colours as CSS custom properties on <html>.
 * Runs client-side once brand config is fetched. Zero visual flash because
 * the fallback values in useBrandConfig match typical defaults.
 */
export function BrandThemeInjector() {
  const { data: brand } = useBrandConfig()

  useEffect(() => {
    if (!brand) return
    const root = document.documentElement
    root.style.setProperty("--brand-primary", brand.primary_colour)
    root.style.setProperty("--brand-accent", brand.accent_colour)
  }, [brand])

  return null
}
