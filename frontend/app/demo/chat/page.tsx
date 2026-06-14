"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { useChatStream } from "@/hooks/useChatStream"
import { MessageList } from "@/components/chat/MessageList"
import { ChatInput } from "@/components/chat/ChatInput"

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
    sendMessage(text, null)
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
      {/* Header */}
      <header className="border-b bg-background px-6 py-3 flex items-center justify-between shrink-0">
        <span className="font-semibold text-sm tracking-tight">
          {brandName} Shopping Assistant
        </span>
        <Link
          href="/demo"
          className="text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          ← Change brand
        </Link>
      </header>

      {/* Chat area */}
      <MessageList messages={messages} isSending={isSending} onSend={handleSend} brand={brandId} />

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
