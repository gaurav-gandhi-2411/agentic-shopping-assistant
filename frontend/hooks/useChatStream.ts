"use client"

import { useCallback, useRef, useState } from "react"
import { getWsUrl } from "@/lib/api/client"
import type { ChatMessage, ItemSummary, WsFrame } from "@/lib/api/types"

interface UseChatStreamOptions {
  onConversationId?: (id: string) => void
  onDone?: () => void
}

export function useChatStream({
  onConversationId,
  onDone,
}: UseChatStreamOptions = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isSending, setIsSending] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)

  // Keep a ref so event handlers always see the latest conversationId without
  // stale closures — especially important when the WS session frame arrives
  // and sets a new ID that subsequent sendMessage calls should use.
  const conversationIdRef = useRef<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const sendMessage = useCallback(
    async (text: string, cidOverride?: string | null) => {
      // Close any in-flight WS before opening a new one.
      wsRef.current?.close()

      const cid =
        cidOverride !== undefined ? cidOverride : conversationIdRef.current

      // Add the user message immediately (optimistic).
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "user",
          content: text,
          items: [],
          isStreaming: false,
        },
      ])
      setIsSending(true)

      const assistantId = crypto.randomUUID()
      let pendingItems: ItemSummary[] = []
      // Tracks whether a terminal frame (done/cancelled/error) was received so
      // the onclose handler doesn't double-clear isSending.
      let finished = false

      let url: string
      try {
        url = await getWsUrl()
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            id: assistantId,
            role: "assistant",
            content: "Failed to connect — are you signed in?",
            items: [],
            isStreaming: false,
          },
        ])
        setIsSending(false)
        return
      }

      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            type: "user_message",
            conversation_id: cid,
            message: text,
          })
        )
      }

      ws.onmessage = (event: MessageEvent) => {
        const frame: WsFrame = JSON.parse(event.data as string)

        switch (frame.type) {
          case "session": {
            const newCid = frame.conversation_id
            conversationIdRef.current = newCid
            setConversationId(newCid)
            onConversationId?.(newCid)
            // Insert the empty streaming placeholder for the assistant.
            setMessages((prev) => [
              ...prev,
              {
                id: assistantId,
                role: "assistant",
                content: "",
                items: [],
                isStreaming: true,
              },
            ])
            break
          }

          case "items": {
            pendingItems = frame.items
            break
          }

          case "token": {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + frame.text }
                  : m
              )
            )
            break
          }

          case "done": {
            finished = true
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, isStreaming: false, items: pendingItems }
                  : m
              )
            )
            setIsSending(false)
            ws.close()
            onDone?.()
            break
          }

          case "cancelled": {
            finished = true
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      isStreaming: false,
                      content: m.content || "(cancelled)",
                    }
                  : m
              )
            )
            setIsSending(false)
            break
          }

          case "error": {
            finished = true
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      isStreaming: false,
                      content: `Error: ${frame.message}`,
                    }
                  : m
              )
            )
            setIsSending(false)
            break
          }
        }
      }

      ws.onerror = () => {
        if (!finished) {
          finished = true
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId && m.isStreaming
                ? {
                    ...m,
                    isStreaming: false,
                    content:
                      m.content || "Connection error. Please try again.",
                  }
                : m
            )
          )
          setIsSending(false)
        }
      }

      ws.onclose = () => {
        // Fires after done/error/cancelled too; only clean up if we haven't already.
        if (!finished) setIsSending(false)
      }
    },
    [onConversationId, onDone]
  )

  const cancel = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "cancel" }))
    }
  }, [])

  // Replace the messages list (e.g. when switching conversations).
  const resetMessages = useCallback((msgs: ChatMessage[] = []) => {
    wsRef.current?.close()
    setMessages(msgs)
    setIsSending(false)
  }, [])

  return {
    messages,
    isSending,
    conversationId,
    sendMessage,
    cancel,
    resetMessages,
    setConversationId: (id: string | null) => {
      conversationIdRef.current = id
      setConversationId(id)
    },
  }
}
