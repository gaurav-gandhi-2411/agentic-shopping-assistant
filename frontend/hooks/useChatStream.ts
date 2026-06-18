"use client"

import { useCallback, useRef, useState } from "react"
import { getWsUrl } from "@/lib/api/client"
import type { ChatMessage, ItemSummary, OutfitVariant, ItemLink, WsFrame } from "@/lib/api/types"

const MAX_RETRIES = 3
const RETRY_DELAYS = [1000, 2000, 4000]

// ---------------------------------------------------------------------------
// StyleFromImageResponse — mirrors POST /style/from-image backend response.
// Same shape as an outfit chat turn's final_state fields.
// ---------------------------------------------------------------------------
interface StyleFromImageResponse {
  conversation_id?: string | null
  items: ItemSummary[]
  look_id?: string | null
  occasion?: string | null
  look_gender?: string | null
  budget_total_inr?: number | null
  outfit_rationale?: string | null
  outfit_variants?: OutfitVariant[] | null
  cart_url?: string | null
  item_links?: ItemLink[] | null
  anchor_article_id?: string | null
}

// ---------------------------------------------------------------------------
// Demo session helpers — read credentials written by the brand picker page.
// ---------------------------------------------------------------------------
function getDemoSessionToken(): string | null {
  if (typeof window === "undefined") return null
  return sessionStorage.getItem("demo_session_token")
}

function getDemoBackendBase(): string {
  if (typeof window !== "undefined") {
    const url = sessionStorage.getItem("demo_backend_url")
    if (url) return url
  }
  return process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"
}

// Map HTTP status codes from /style/from-image to user-friendly messages.
function imageUploadErrorMessage(status: number): string {
  if (status === 413) return "Image too large (max 15 MB). Please try a smaller file."
  if (status === 400) return "That doesn't look like a supported image (JPEG, PNG, WebP, HEIC)."
  if (status === 404) return "Image styling isn't available right now — try typing your request instead."
  if (status === 429) return "Rate limited — please wait a moment and try again."
  return `Something went wrong (${status}). Please try again.`
}

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
              const outfitRationale = frame.final_state.outfit_rationale ?? null
              const outfitVariants = frame.final_state.outfit_variants ?? null
              const cartUrl = frame.final_state.cart_url ?? null
              const itemLinks = frame.final_state.item_links ?? null
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
                        outfitRationale,
                        outfitVariants,
                        cartUrl,
                        itemLinks,
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

  /**
   * Upload a garment/inspiration image to POST /style/from-image and append
   * the returned outfit look as an assistant message.  The user-side bubble
   * is added optimistically; the assistant message is appended on success.
   */
  const sendImage = useCallback(async (file: File): Promise<void> => {
    if (isSending) return

    const imagePreviewUrl = URL.createObjectURL(file)

    // Optimistic user bubble — includes local preview thumbnail.
    const userMsgId = crypto.randomUUID()
    setMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        dbId: null,
        role: "user",
        content: "Styling around your uploaded photo",
        items: [],
        isStreaming: false,
        imageUrl: imagePreviewUrl,
      },
    ])
    setIsSending(true)

    // Loading placeholder — assistant side.
    const assistantId = crypto.randomUUID()
    setMessages((prev) => [
      ...prev,
      {
        id: assistantId,
        dbId: null,
        role: "assistant",
        content: "Finding your match…",
        items: [],
        isStreaming: true,
      },
    ])

    try {
      const backendBase = getDemoBackendBase()
      const token = getDemoSessionToken()

      // Re-use or seed a conversation_id so the backend can persist this
      // image look into the session store.  Subsequent sendMessage calls pass
      // conversationIdRef.current, which we update below after a successful
      // response — this is the fix for G1 (image upload context retention).
      const cid = conversationIdRef.current ?? crypto.randomUUID()

      const body = new FormData()
      body.append("file", file)
      body.append("conversation_id", cid)

      const headers: Record<string, string> = {}
      if (token) headers["Authorization"] = `Bearer ${token}`

      const res = await fetch(`${backendBase}/style/from-image`, {
        method: "POST",
        headers,
        body,
      })

      if (!res.ok) {
        const friendlyMsg = imageUploadErrorMessage(res.status)
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, isStreaming: false, content: friendlyMsg }
              : m
          )
        )
        setIsSending(false)
        return
      }

      const data = (await res.json()) as StyleFromImageResponse

      // Update the conversation_id ref so follow-up sendMessage calls resume
      // the session that now contains the image outfit context.
      const returnedCid = data.conversation_id ?? cid
      conversationIdRef.current = returnedCid
      setConversationId(returnedCid)
      onConversationId?.(returnedCid)

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: "",
                isStreaming: false,
                items: data.items ?? [],
                lookId: data.look_id ?? null,
                occasion: data.occasion ?? null,
                lookGender: data.look_gender ?? null,
                budgetTotalInr: data.budget_total_inr ?? null,
                outfitRationale: data.outfit_rationale ?? null,
                outfitVariants: data.outfit_variants ?? null,
                cartUrl: data.cart_url ?? null,
                itemLinks: data.item_links ?? null,
              }
            : m
        )
      )
      onDone?.()
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                isStreaming: false,
                content: "Could not reach the styling service — please try again.",
              }
            : m
        )
      )
    } finally {
      setIsSending(false)
    }
  }, [isSending, onDone])

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
    sendImage,
    cancel,
    resetMessages,
    setConversationId: (id: string | null) => {
      conversationIdRef.current = id
      setConversationId(id)
    },
  }
}
