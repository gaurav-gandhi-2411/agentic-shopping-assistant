"use client"

import { useEffect, useState } from "react"
import { useChatStream } from "@/hooks/useChatStream"
import { MessageList } from "@/components/chat/MessageList"
import { ChatInput } from "@/components/chat/ChatInput"
import { Logo } from "@/components/Logo"

export default function DemoChatPage() {
  const [brandName, setBrandName] = useState<string | null>(null)
  const [brandId, setBrandId] = useState<string | undefined>(undefined)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    const name = sessionStorage.getItem("demo_brand_name")
    if (!name || !sessionStorage.getItem("demo_session_token")) {
      // No valid demo session — send back to the entry page.
      window.location.replace("/demo")
      return
    }
    setBrandName(name)
    // "unified" brand id is the cross-store default; pass undefined so
    // OutfitBoard doesn't try to gate on a single Shopify brand.
    const storedId = sessionStorage.getItem("demo_brand_id")
    setBrandId(storedId === "unified" ? undefined : (storedId ?? undefined))
  }, [])

  const { messages, isSending, connectionLost, sendMessage, sendImage, cancel } =
    useChatStream()

  const hasAssistantReply = messages.some((m) => m.role === "assistant")
  const showWarmup = isSending && !hasAssistantReply

  function handleSend(text: string) {
    // No cidOverride: let useChatStream fall back to conversationIdRef.current
    // so follow-up turns (refinements like "in blue now") stay in the same
    // conversation instead of forcing a new one every message.
    sendMessage(text)
  }

  if (!mounted || brandName === null) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
        Loading…
      </div>
    )
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Header — Style Maitri alone in unified mode, "Style Maitri x <Brand>" for a
          brand-specific demo (brandId is only set in the shelved per-brand path).
          Airy premium treatment: cream bg, hairline gold bottom border, generous height. */}
      <header className="border-b border-border bg-background px-6 py-4 flex items-center shrink-0">
        {brandId ? (
          <span className="inline-flex items-center gap-2">
            <Logo showWordmark={false} iconClassName="h-6 w-6 text-champagne shrink-0" />
            <span className="font-serif font-semibold text-base tracking-tight text-foreground">
              Style Maitri <span className="text-muted-foreground font-sans font-normal">x</span> {brandName}
            </span>
          </span>
        ) : (
          <Logo />
        )}
      </header>

      {/* Chat area */}
      <MessageList
        messages={messages}
        isSending={isSending}
        onSend={handleSend}
        brand={brandId}
        onSendSuggestion={handleSend}
      />

      {connectionLost && (
        <div className="mx-4 mb-2 px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/30 text-xs text-destructive text-center">
          Connection lost — please refresh the page.
        </div>
      )}

      {showWarmup && (
        <div className="mx-4 mb-1 px-3 py-1.5 rounded-lg bg-muted text-xs text-muted-foreground text-center">
          Warming up the {brandName} assistant — first query may take 15–30 s on cold start…
        </div>
      )}

      <ChatInput
        onSend={handleSend}
        onCancel={cancel}
        isSending={isSending}
        onSendImage={sendImage}
      />
    </div>
  )
}
