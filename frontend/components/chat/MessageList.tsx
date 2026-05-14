"use client"

import { useEffect, useRef } from "react"
import type { ChatMessage } from "@/lib/api/types"
import { MessageBubble } from "./MessageBubble"

interface Props {
  messages: ChatMessage[]
  isSending: boolean
  onSend?: (text: string) => void
}

export function MessageList({ messages, isSending, onSend }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const isStreamingRef = useRef(false)

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
    return (
      <div className="flex flex-col items-center justify-center flex-1 gap-3 text-center px-8 select-none">
        <span className="text-5xl" aria-hidden>
          🛍️
        </span>
        <p className="text-lg font-semibold">What can I help you find?</p>
        <p className="text-sm text-muted-foreground max-w-xs">
          Try &ldquo;show me red summer dresses&rdquo; or &ldquo;casual blue
          jeans under £40&rdquo;
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} onSend={onSend} />
      ))}
      {/* Typing indicator while the agent is running but no token yet */}
      {isSending && !messages.some((m) => m.isStreaming) && (
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
