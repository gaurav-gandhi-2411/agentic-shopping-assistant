"use client"

import { useEffect } from "react"
import { useConversation, useInvalidateConversations } from "@/hooks/useConversations"
import { useChatStream } from "@/hooks/useChatStream"
import { ConversationSidebar } from "./ConversationSidebar"
import { MessageList } from "./MessageList"
import { ChatInput } from "./ChatInput"
import type { ChatMessage, ConversationDetail } from "@/lib/api/types"

// Convert backend conversation history to ChatMessage objects.
// Items (retrieved_items) are attached to the last assistant message since the
// session stores a single items list, not per-message items.
function mapHistory(detail: ConversationDetail): ChatMessage[] {
  const msgs: ChatMessage[] = detail.messages.map((m, i) => ({
    // Prefer the DB UUID as the React key when available; fall back to a
    // stable client-side key for in-memory mode.
    id: m.id ?? `hist-${i}-${m.role}`,
    // dbId is the DB UUID used for feedback calls; null in in-memory mode.
    dbId: m.id ?? null,
    role: m.role,
    content: m.content,
    items: [],
    isStreaming: false,
  }))

  if (detail.retrieved_items.length > 0) {
    const lastAssistantIdx = msgs.findLastIndex((m) => m.role === "assistant")
    if (lastAssistantIdx >= 0) {
      msgs[lastAssistantIdx] = {
        ...msgs[lastAssistantIdx],
        items: detail.retrieved_items,
      }
    }
  }

  return msgs
}

interface Props {
  // If provided (e.g. from a URL param), open this conversation on mount.
  initialConversationId?: string
}

export function ChatThread({ initialConversationId }: Props) {
  const invalidate = useInvalidateConversations()

  const {
    messages,
    isSending,
    connectionLost,
    conversationId,
    sendMessage,
    cancel,
    resetMessages,
    setConversationId,
  } = useChatStream({
    onConversationId: () => invalidate(),
    onDone: () => invalidate(),
  })

  // Active conversation is either the one from the stream (after first message)
  // or one the user explicitly selected from the sidebar.
  const activeId = conversationId ?? initialConversationId ?? null

  // Fetch history whenever the user switches to an existing conversation.
  const { data: conversationData } = useConversation(
    activeId && messages.length === 0 ? activeId : null
  )

  useEffect(() => {
    if (conversationData) {
      resetMessages(mapHistory(conversationData))
      setConversationId(conversationData.conversation_id)
    }
  // Only re-run when the conversation we're loading actually changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationData?.conversation_id])

  function handleSelectConversation(cid: string) {
    if (cid === conversationId) return
    // Clear messages first; the useConversation query will load history.
    resetMessages([])
    setConversationId(cid)
  }

  function handleNewConversation() {
    resetMessages([])
    setConversationId(null)
  }

  function handleSend(text: string) {
    sendMessage(text, activeId)
  }

  return (
    <div className="flex flex-1 min-h-0">
      <ConversationSidebar
        activeId={activeId}
        onSelect={handleSelectConversation}
        onNew={handleNewConversation}
      />
      <div className="flex flex-col flex-1 min-h-0">
        <MessageList messages={messages} isSending={isSending} onSend={handleSend} />
        {connectionLost && (
          <div className="mx-4 mb-2 px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/30 text-xs text-destructive text-center">
            Connection lost — please refresh the page.
          </div>
        )}
        <ChatInput
          onSend={handleSend}
          onCancel={cancel}
          isSending={isSending}
        />
      </div>
    </div>
  )
}
