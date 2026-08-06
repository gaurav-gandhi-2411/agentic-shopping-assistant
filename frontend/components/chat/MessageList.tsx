"use client"

import { useEffect, useRef } from "react"
import type { ChatMessage } from "@/lib/api/types"
import { useBrandConfig } from "@/hooks/useBrandConfig"
import { LogoMark } from "@/components/Logo"
import { MessageBubble } from "./MessageBubble"

interface Props {
  messages: ChatMessage[]
  isSending: boolean
  onSend?: (text: string) => void
  /** Brand id propagated to OutfitBoard for buy link resolution. */
  brand?: string
  /** Sends a suggestion chip's text as if the user typed it. Empty-state chips render only when this is provided. */
  onSendSuggestion?: (text: string) => void
}

export function MessageList({ messages, isSending, onSend, brand, onSendSuggestion }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const isStreamingRef = useRef(false)
  const { data: brandConfig } = useBrandConfig()

  // Scroll to bottom when messages change, using instant scroll during
  // streaming and smooth scroll for new (non-streaming) messages.
  useEffect(() => {
    const streaming = messages.some((m) => m.isStreaming)
    bottomRef.current?.scrollIntoView({
      behavior: streaming ? "instant" : "smooth",
    })
    isStreamingRef.current = streaming
  }, [messages])

  if (messages.length === 0) {
    const chips = brandConfig?.suggestion_chips ?? []
    return (
      <div className="flex flex-col items-center justify-center flex-1 gap-3 text-center px-8 select-none">
        <span aria-hidden>
          <LogoMark className="h-14 w-14 lg:h-16 lg:w-16 text-champagne" />
        </span>
        <p className="font-serif font-semibold text-xl lg:text-2xl tracking-tight text-foreground">
          What can I help you find?
        </p>
        {/* Subtle wedding-invite-style divider so the desktop-width state doesn't
            read as sparse (P3-1) — small, matches the champagne hairline palette. */}
        <span aria-hidden className="h-px w-12 bg-champagne/30" />
        <p className="text-sm text-muted-foreground max-w-xs sm:max-w-md lg:max-w-lg leading-relaxed">
          Try &ldquo;show me red summer dresses&rdquo; or &ldquo;casual blue
          jeans under ₹2,000&rdquo;
        </p>
        {onSendSuggestion && chips.length > 0 && (
          <div className="flex flex-wrap justify-center gap-2 max-w-sm sm:max-w-md lg:max-w-lg mt-3">
            {chips.map((chip) => (
              <button
                key={chip}
                onClick={() => onSendSuggestion(chip)}
                className="inline-flex items-center justify-center min-h-11 min-w-11 text-xs px-3 py-1.5 rounded-full border border-champagne/40 bg-background text-foreground hover:bg-primary hover:text-primary-foreground hover:border-primary transition-colors"
              >
                {chip}
              </button>
            ))}
          </div>
        )}
      </div>
    )
  }

  // Only the LATEST assistant message should render its suggestion chips — otherwise
  // stale chips from earlier turns would still be clickable and could resend outdated
  // refinements against the current (already-refined) look.
  let lastAssistantIndex = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      lastAssistantIndex = i
      break
    }
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-5">
      {messages.map((message, index) => (
        <MessageBubble
          key={message.id}
          message={message}
          onSend={onSend}
          brand={brand}
          isLatestAssistant={index === lastAssistantIndex}
        />
      ))}
      {/* Typing indicator while the agent is running but no token yet. Keep showing
          the dots as long as the streaming placeholder (inserted on the WS "session"
          frame) has no real content — otherwise the dots vanish the instant the
          placeholder is added, well before any text exists (#18). */}
      {isSending && !messages.some((m) => m.isStreaming && m.content.length > 0) && (
        <div className="flex items-start">
          <div className="bg-muted rounded-2xl rounded-bl-sm px-4 py-3">
            <div className="flex gap-1 items-center h-4">
              <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground animate-bounce [animation-delay:-0.3s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground animate-bounce [animation-delay:-0.15s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground animate-bounce" />
            </div>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
