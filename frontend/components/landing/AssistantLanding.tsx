"use client"

import { lazy, Suspense, useEffect, useRef, useState } from "react"
import Link from "next/link"
import { Bookmark } from "lucide-react"
import { ChatInput } from "@/components/chat/ChatInput"
import { Logo } from "@/components/Logo"
import { useChatStream } from "@/hooks/useChatStream"
import { useDemoSession } from "@/hooks/useDemoSession"
import { useSlowStart } from "@/hooks/useSlowStart"
import { COPY, OCCASION_CHIPS } from "@/lib/copy"
import { isBusyReply } from "@/lib/limits"
import { deriveKnownGender } from "@/lib/knownGender"
import { OccasionChips } from "./OccasionChips"
import { StreamNotice } from "./StreamNotice"
import { WakingSkeleton } from "./WakingSkeleton"

// The message list pulls in react-markdown, the outfit board and the item cards. None of
// that is needed to paint the landing page, so it loads on the first query instead of
// weighing down the initial bundle (Lighthouse LCP/TBT on `/`).
const MessageList = lazy(() =>
  import("@/components/chat/MessageList").then((m) => ({ default: m.MessageList })),
)

/** A query typed or tapped before the demo session exists; sent as soon as it is ready. */
type PendingAction =
  | { id: number; kind: "text"; text: string }
  | { id: number; kind: "image"; file: File; text?: string }

/**
 * `/`: the working assistant for anonymous visitors (Wave 1b V2). Reuses the same chat
 * hook and components as the former /demo/chat. The demo session is created on load; a
 * chip tap or message sent while it is still being created is queued, so a cold start
 * never makes the visitor tap twice.
 */
export function AssistantLanding() {
  const session = useDemoSession()
  const chat = useChatStream()
  const [pending, setPending] = useState<PendingAction | null>(null)
  const nextId = useRef(1)
  const flushedId = useRef(0)

  const { sendMessage, sendImage, resetMessages, messages } = chat

  // Flush the queued action exactly once when the session becomes ready.
  useEffect(() => {
    if (session.status !== "ready" || !pending || flushedId.current === pending.id) return
    flushedId.current = pending.id
    setPending(null)
    if (pending.kind === "text") void sendMessage(pending.text)
    else void sendImage(pending.file, pending.text)
  }, [session.status, pending, sendMessage, sendImage])

  function handleSend(text: string): void {
    if (session.status === "ready") {
      void sendMessage(text)
    } else {
      setPending({ id: nextId.current++, kind: "text", text })
    }
  }

  function handleSendImage(file: File, text?: string): void {
    if (session.status === "ready") {
      void sendImage(file, text)
    } else {
      setPending({ id: nextId.current++, kind: "image", file, text })
    }
  }

  // The backend streams its "busy" apology as ordinary assistant text. Replace that bubble
  // with a proper notice that has a retry button.
  const lastMessage = messages[messages.length - 1]
  const busy =
    lastMessage?.role === "assistant" && !lastMessage.isStreaming && isBusyReply(lastMessage.content)
  const shownMessages = busy ? messages.slice(0, -1) : messages

  // Retry = drop the failed turn (the user's bubble and any busy reply) and send it again,
  // so the conversation does not show the same question twice.
  function retryLastTurn(): void {
    let lastUser = -1
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        lastUser = i
        break
      }
    }
    if (lastUser === -1) return
    const text = messages[lastUser].content
    resetMessages(messages.slice(0, lastUser))
    void sendMessage(text)
  }

  const hasAssistantReply = messages.some((m) => m.role === "assistant")
  // "Waiting" = the session is not ready yet, a query is queued, or a query was sent and the
  // connection has not produced even the first frame. Starts at page load for a cold session.
  const waiting =
    session.status === "creating" || pending != null || (chat.isSending && !hasAssistantReply)
  const slow = useSlowStart(waiting)

  const started = shownMessages.length > 0 || pending != null
  const knownGender = deriveKnownGender(messages)

  const notices = (
    <>
      {slow && <WakingSkeleton />}
      {session.status === "error" && (
        <StreamNotice kind="session_error" onRetry={session.retry} />
      )}
      {session.status === "limit" && <StreamNotice kind="demo_limit" />}
      {chat.limitNotice && (
        <StreamNotice
          kind={chat.limitNotice.kind}
          retryAfterSeconds={chat.limitNotice.retryAfterSeconds}
          onRetry={retryLastTurn}
        />
      )}
      {busy && <StreamNotice kind="busy" onRetry={retryLastTurn} />}
    </>
  )

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <header className="flex shrink-0 items-center justify-between border-b border-border bg-background px-4 py-2">
        <Logo />
        <Link
          href="/saved-looks"
          className="inline-flex min-h-11 items-center gap-1.5 px-2 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
        >
          <Bookmark className="h-4 w-4" />
          {COPY.landing.savedLooksLink}
        </Link>
      </header>

      {started ? (
        <div className="flex min-h-0 flex-1 flex-col">
          {pending && shownMessages.length === 0 ? (
            <div className="flex-1 overflow-y-auto px-4 py-6">
              <div className="flex justify-end">
                <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-primary px-4 py-3 text-sm leading-relaxed text-primary-foreground shadow-sm">
                  {pending.kind === "text" ? pending.text : (pending.text ?? COPY.landing.queuedPhoto)}
                </div>
              </div>
              {notices}
            </div>
          ) : (
            <>
              <Suspense fallback={<div className="flex-1" />}>
                <MessageList
                  messages={shownMessages}
                  isSending={chat.isSending}
                  onSend={handleSend}
                  onSendSuggestion={handleSend}
                  signInNudge
                />
              </Suspense>
              {chat.connectionLost && (
                <StreamNotice kind="session_error" onRetry={retryLastTurn} />
              )}
              {notices}
            </>
          )}
        </div>
      ) : (
        <main className="min-h-0 flex-1 overflow-y-auto px-4 pb-4 pt-6">
          <h1 className="text-balance font-serif text-2xl font-semibold tracking-tight text-foreground">
            {COPY.landing.promise}
          </h1>
          <p className="mb-5 mt-1 text-sm text-muted-foreground">{COPY.landing.noSignIn}</p>
          <OccasionChips chips={OCCASION_CHIPS} onSelect={handleSend} />
          {notices}
        </main>
      )}

      <ChatInput
        onSend={handleSend}
        onCancel={chat.cancel}
        isSending={chat.isSending}
        onSendImage={handleSendImage}
        knownGender={knownGender}
      />
    </div>
  )
}
