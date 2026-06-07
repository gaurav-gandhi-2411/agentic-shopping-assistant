"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"

interface Brand {
  id: string
  name: string
  tagline: string
  backendUrl: string
  accentHex: string
}

// Backend URLs come from build-time env vars set per-brand in Vercel.
// If a var is unset the service is not available in this deploy.
const BRANDS: Brand[] = [
  {
    id: "snitch",
    name: "Snitch",
    tagline: "Contemporary Indian menswear",
    backendUrl: process.env.NEXT_PUBLIC_SNITCH_BACKEND_URL ?? "",
    accentHex: "#1a1a2e",
  },
  {
    id: "myntra",
    name: "Myntra",
    tagline: "India's fashion destination",
    backendUrl: process.env.NEXT_PUBLIC_MYNTRA_BACKEND_URL ?? "",
    accentHex: "#ff3f6c",
  },
  {
    id: "flipkart",
    name: "Flipkart Fashion",
    tagline: "Shop fashion your way",
    backendUrl: process.env.NEXT_PUBLIC_FLIPKART_BACKEND_URL ?? "",
    accentHex: "#2874f0",
  },
].filter((b) => b.backendUrl !== "")

interface DemoSessionResponse {
  session_token: string
  ws_ticket: string
  expires_in: number
  brand: string
}

export default function DemoPickerPage() {
  const router = useRouter()
  const [loading, setLoading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleBrandSelect(brand: Brand) {
    setLoading(brand.id)
    setError(null)
    try {
      const res = await fetch(`${brand.backendUrl}/demo/session`, {
        method: "POST",
      })
      if (res.status === 429) {
        setError("This demo has reached its daily limit — check back tomorrow.")
        setLoading(null)
        return
      }
      if (!res.ok) throw new Error(`Backend returned ${res.status}`)
      const data = (await res.json()) as DemoSessionResponse
      // Store in sessionStorage — cleared automatically when the tab closes.
      sessionStorage.setItem("demo_session_token", data.session_token)
      sessionStorage.setItem("demo_backend_url", brand.backendUrl)
      sessionStorage.setItem("demo_brand_id", brand.id)
      sessionStorage.setItem("demo_brand_name", brand.name)
      router.push("/demo/chat")
    } catch {
      setError("Could not reach the assistant — please try again in a moment.")
      setLoading(null)
    }
  }

  return (
    <main className="flex-1 flex flex-col items-center justify-center p-8">
      <div className="max-w-xl w-full text-center">
        <h1 className="text-3xl font-bold mb-2 tracking-tight">
          Agentic Shopping Assistant
        </h1>
        <p className="text-muted-foreground mb-10 text-sm">
          AI-powered fashion discovery for Indian brands.
          Pick a brand to start exploring.
        </p>

        {BRANDS.length === 0 ? (
          <p className="text-muted-foreground text-sm">
            No brand backends configured. Set{" "}
            <code className="font-mono text-xs bg-muted px-1 py-0.5 rounded">
              NEXT_PUBLIC_SNITCH_BACKEND_URL
            </code>{" "}
            etc. in your Vercel environment.
          </p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
            {BRANDS.map((brand) => (
              <button
                key={brand.id}
                onClick={() => handleBrandSelect(brand)}
                disabled={loading !== null}
                className="group relative flex flex-col items-start p-5 rounded-xl border-2
                           border-border bg-card hover:border-foreground hover:shadow-lg
                           transition-all duration-200 text-left disabled:opacity-60
                           disabled:cursor-wait"
              >
                <div
                  className="w-7 h-7 rounded-full mb-3 shrink-0"
                  style={{ backgroundColor: brand.accentHex }}
                  aria-hidden
                />
                <span className="font-semibold text-base leading-snug">
                  {brand.name}
                </span>
                <span className="text-muted-foreground text-xs mt-1">
                  {brand.tagline}
                </span>
                {loading === brand.id && (
                  <span className="absolute inset-0 flex items-center justify-center
                                   rounded-xl bg-background/80 text-sm text-muted-foreground">
                    Connecting…
                  </span>
                )}
              </button>
            ))}
          </div>
        )}

        {error && (
          <p className="text-destructive text-sm mb-4">{error}</p>
        )}

        <p className="text-xs text-muted-foreground">
          No account required · Conversations are not saved · Rate-limited
        </p>
      </div>
    </main>
  )
}
