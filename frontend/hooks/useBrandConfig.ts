import { useQuery } from "@tanstack/react-query"

export interface BrandConfig {
  display_name: string
  logo_url: string | null
  primary_colour: string
  accent_colour: string
  tagline: string | null
  currency: string
  locale: string
  sizing_system: string
  suggestion_chips: string[]
  pdp_url_template: string
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8080"

const FALLBACK_BRAND: BrandConfig = {
  display_name: "Shop",
  logo_url: null,
  primary_colour: "#000000",
  accent_colour: "#ffffff",
  tagline: null,
  currency: "INR",
  locale: "en-IN",
  sizing_system: "IN",
  suggestion_chips: [
    "Show me something to wear",
    "Help me find an outfit",
    "What's trending?",
  ],
  pdp_url_template: "",
}

export function useBrandConfig() {
  return useQuery<BrandConfig>({
    queryKey: ["brand-config"],
    queryFn: async () => {
      const res = await fetch(`${BACKEND_URL}/api/brand`)
      if (!res.ok) throw new Error("Failed to fetch brand config")
      return res.json() as Promise<BrandConfig>
    },
    staleTime: Infinity, // brand config doesn't change during a session
    retry: false,
    placeholderData: FALLBACK_BRAND,
  })
}
