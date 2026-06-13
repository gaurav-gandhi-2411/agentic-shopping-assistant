"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { X } from "lucide-react"
import { useChatStream } from "@/hooks/useChatStream"
import { MessageList } from "@/components/chat/MessageList"
import { ChatInput } from "@/components/chat/ChatInput"

// ---------------------------------------------------------------------------
// Known brand definitions — must match NEXT_PUBLIC_{BRAND}_BACKEND_URL vars.
// If a brand is unknown we show a friendly fallback rather than crashing.
// ---------------------------------------------------------------------------
interface KnownBrand {
  id: string
  name: string
  backendUrl: string
  accentHex: string
}

const KNOWN_BRANDS: Record<string, KnownBrand> = {
  snitch: {
    id: "snitch",
    name: "Snitch",
    backendUrl: process.env.NEXT_PUBLIC_SNITCH_BACKEND_URL ?? "",
    accentHex: "#1a1a2e",
  },
  myntra: {
    id: "myntra",
    name: "Myntra",
    backendUrl: process.env.NEXT_PUBLIC_MYNTRA_BACKEND_URL ?? "",
    accentHex: "#ff3f6c",
  },
  flipkart: {
    id: "flipkart",
    name: "Flipkart Fashion",
    backendUrl: process.env.NEXT_PUBLIC_FLIPKART_BACKEND_URL ?? "",
    accentHex: "#2874f0",
  },
}

// ---------------------------------------------------------------------------
// postMessage close signal — tells the parent widget loader to close the overlay.
// ---------------------------------------------------------------------------
function postCloseToParent(): void {
  try {
    window.parent?.postMessage({ type: "asa:close" }, "*")
  } catch {
    // Cross-origin postMessage errors are swallowed deliberately.
  }
}

// ---------------------------------------------------------------------------
// Unknown brand fallback
// ---------------------------------------------------------------------------
function UnknownBrandFallback({ brandId }: { brandId: string }) {
  return (
    <div className="flex flex-col h-full items-center justify-center gap-4 p-8 text-center">
      {/* Close button in the top-right corner */}
      <button
        onClick={postCloseToParent}
        className="absolute top-3 right-3 p-1.5 rounded-full bg-muted hover:bg-muted/80 transition-colors"
        aria-label="Close"
      >
        <X className="h-4 w-4 text-muted-foreground" />
      </button>
      <span className="text-4xl" aria-hidden>🛍️</span>
      <p className="font-semibold text-base">Brand not found</p>
      <p className="text-sm text-muted-foreground max-w-xs">
        No styling assistant is configured for <strong>{brandId}</strong>.
        Contact the brand to enable this feature.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main embed page
// ---------------------------------------------------------------------------
export default function EmbedPage() {
  const params = useParams()
  const brandId = typeof params.brand === "string" ? params.brand : ""

  const brand = KNOWN_BRANDS[brandId] ?? null
  const backendAvailable = brand !== null && brand.backendUrl !== ""

  // Session bootstrap state
  const [sessionReady, setSessionReady] = useState(false)
  const [sessionError, setSessionError] = useState<string | null>(null)
  const [brandName, setBrandName] = useState<string>(brand?.name ?? "")

  const { messages, isSending, connectionLost, sendMessage, sendImage, cancel } =
    useChatStream()

  const hasAssistantReply = messages.some((m) => m.role === "assistant")
  const showWarmup = isSending && !hasAssistantReply

  // Bootstrap a demo session — same flow as /demo/page.tsx.
  // Skipped if a session already exists in sessionStorage for this brand
  // (handles iframe reload without burning another demo slot).
  useEffect(() => {
    if (!backendAvailable || !brand) return

    const existingToken = sessionStorage.getItem("demo_session_token")
    const existingBrand = sessionStorage.getItem("demo_brand_id")

    if (existingToken && existingBrand === brandId) {
      // Reuse existing session.
      setBrandName(sessionStorage.getItem("demo_brand_name") ?? brand.name)
      setSessionReady(true)
      return
    }

    // Create a fresh anonymous demo session.
    let cancelled = false
    async function bootstrap() {
      try {
        const res = await fetch(`${brand!.backendUrl}/demo/session`, {
          method: "POST",
        })
        if (cancelled) return
        if (res.status === 429) {
          setSessionError(
            "This demo has reached its daily limit — check back tomorrow."
          )
          return
        }
        if (!res.ok) throw new Error(`Backend returned ${res.status}`)
        const data = (await res.json()) as {
          session_token: string
          ws_ticket: string
          expires_in: number
          brand: string
        }
        sessionStorage.setItem("demo_session_token", data.session_token)
        sessionStorage.setItem("demo_backend_url", brand!.backendUrl)
        sessionStorage.setItem("demo_brand_id", brandId)
        sessionStorage.setItem("demo_brand_name", brand!.name)
        setBrandName(brand!.name)
        setSessionReady(true)
      } catch {
        if (!cancelled) {
          setSessionError(
            "Could not reach the assistant — please try again in a moment."
          )
        }
      }
    }
    void bootstrap()
    return () => {
      cancelled = true
    }
  }, [brand, brandId, backendAvailable])

  function handleSend(text: string) {
    sendMessage(text, null)
  }

  // ---------------------------------------------------------------------------
  // Render paths
  // ---------------------------------------------------------------------------

  // Unknown / unconfigured brand.
  if (!backendAvailable) {
    return <UnknownBrandFallback brandId={brandId} />
  }

  // Session bootstrap error (rate limit, backend down, etc.)
  if (sessionError) {
    return (
      <div className="flex flex-col h-full items-center justify-center gap-4 p-8 text-center relative">
        <button
          onClick={postCloseToParent}
          className="absolute top-3 right-3 p-1.5 rounded-full bg-muted hover:bg-muted/80 transition-colors"
          aria-label="Close"
        >
          <X className="h-4 w-4 text-muted-foreground" />
        </button>
        <span className="text-4xl" aria-hidden>⚠️</span>
        <p className="text-sm text-muted-foreground max-w-xs">{sessionError}</p>
      </div>
    )
  }

  // Loading / bootstrapping.
  if (!sessionReady) {
    return (
      <div className="flex flex-col h-full items-center justify-center gap-3">
        <div className="flex gap-1 items-center h-4">
          <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce [animation-delay:-0.3s]" />
          <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce [animation-delay:-0.15s]" />
          <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" />
        </div>
        <p className="text-xs text-muted-foreground">Connecting to {brandName} stylist…</p>
      </div>
    )
  }

  // Ready — full chat experience.
  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Slim header: brand name + close affordance */}
      <header
        className="shrink-0 flex items-center justify-between px-4 py-2.5 border-b bg-background"
        style={{ borderColor: "hsl(var(--border))" }}
      >
        <div className="flex items-center gap-2">
          {/* Brand accent dot */}
          <span
            className="w-3 h-3 rounded-full shrink-0"
            style={{ backgroundColor: brand.accentHex }}
            aria-hidden
          />
          <span className="text-sm font-semibold leading-none">
            {brandName} Stylist
          </span>
        </div>
        {/* Close button — posts asa:close to the parent window */}
        <button
          onClick={postCloseToParent}
          className="p-1.5 rounded-full hover:bg-muted transition-colors"
          aria-label="Close styling assistant"
        >
          <X className="h-4 w-4 text-muted-foreground" />
        </button>
      </header>

      {/* Chat area — reuses the exact same MessageList component as /demo/chat */}
      <MessageList
        messages={messages}
        isSending={isSending}
        onSend={handleSend}
        brand={brandId}
      />

      {/* Connection lost banner */}
      {connectionLost && (
        <div className="mx-4 mb-2 px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/30 text-xs text-destructive text-center">
          Connection lost — please refresh.
        </div>
      )}

      {/* Cold-start warmup hint */}
      {showWarmup && (
        <div className="mx-4 mb-1 px-3 py-1.5 rounded-lg bg-muted text-xs text-muted-foreground text-center">
          Warming up the {brandName} assistant — first query may take 15–30 s…
        </div>
      )}

      {/* Chat input — reuses the exact same ChatInput component as /demo/chat */}
      <ChatInput
        onSend={handleSend}
        onCancel={cancel}
        isSending={isSending}
        onSendImage={sendImage}
      />

      {/* Powered-by footer — attribution inside the embed */}
      <div className="shrink-0 py-1.5 text-center border-t bg-background">
        <span className="text-[10px] text-muted-foreground">
          Powered by{" "}
          <a
            href="https://asa-stylist.vercel.app"
            target="_blank"
            rel="noopener noreferrer"
            className="underline underline-offset-2"
          >
            Agentic Shopping Assistant
          </a>
        </span>
      </div>
    </div>
  )
}
