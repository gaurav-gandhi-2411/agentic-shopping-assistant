"use client"

import { useCallback, useRef, useState } from "react"
import { getWsUrl } from "@/lib/api/client"
import type { ChatMessage, ItemSummary, WsFrame } from "@/lib/api/types"

const MAX_RETRIES = 3
const RETRY_DELAYS = [1000, 2000, 4000]

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
  const [connectionLost, setConnectionLost] = useState(false)
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
      setConnectionLost(false)

      const cid =
        cidOverride !== undefined ? cidOverride : conversationIdRef.current

      // Add the user message immediately (optimistic).
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          dbId: null,
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
      // Tracks whether the assistant placeholder was added (so retries don't duplicate it).
      let placeholderAdded = false

      let url: string
      try {
        url = await getWsUrl()
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            id: assistantId,
            dbId: null,
            role: "assistant",
            content: "Failed to connect — are you signed in?",
            items: [],
            isStreaming: false,
          },
        ])
        setIsSending(false)
        return
      }

      let retryCount = 0

      function openConnection() {
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
              // Insert the empty streaming placeholder if not already present.
              if (!placeholderAdded) {
                placeholderAdded = true
                setMessages((prev) => [
                  ...prev,
                  {
                    id: assistantId,
                    dbId: null,
                    role: "assistant",
                    content: "",
                    items: [],
                    isStreaming: true,
                  },
                ])
              }
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
              // Capture the persisted DB message UUID for feedback calls.
              // frame.message_id is null when running against InMemorySessionStore.
              const dbMessageId = frame.message_id ?? null
              const lookId = frame.final_state.look_id ?? null
              const occasion = frame.final_state.occasion ?? null
              const lookGender = frame.final_state.look_gender ?? null
              const budgetTotalInr = frame.final_state.budget_total_inr ?? null
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        dbId: dbMessageId,
                        isStreaming: false,
                        items: pendingItems,
                        lookId,
                        occasion,
                        lookGender,
                        budgetTotalInr,
                      }
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
          // onclose always fires after onerror; let onclose drive the retry logic.
        }

        ws.onclose = (event: CloseEvent) => {
          if (finished) return

          // Code 1000 = clean close (should not happen unless server initiated).
          if (event.code === 1000) {
            setIsSending(false)
            return
          }

          // Code 1008 = policy violation / rate limit — do not retry.
          if (event.code === 1008) {
            finished = true
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId && m.isStreaming
                  ? {
                      ...m,
                      isStreaming: false,
                      content: m.content || "Rate limited — please wait and try again.",
                    }
                  : m
              )
            )
            setIsSending(false)
            return
          }

          // Unexpected close: retry with backoff.
          if (retryCount < MAX_RETRIES) {
            const delay = RETRY_DELAYS[retryCount]
            retryCount++
            setTimeout(openConnection, delay)
          } else {
            // All retries exhausted.
            finished = true
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId && m.isStreaming
                  ? { ...m, isStreaming: false, content: m.content || "" }
                  : m
              )
            )
            setConnectionLost(true)
            setIsSending(false)
          }
        }
      }

      openConnection()
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
    setConnectionLost(false)
  }, [])

  return {
    messages,
    isSending,
    connectionLost,
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
