"use client"

import { useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import rehypeSanitize from "rehype-sanitize"
import { cn } from "@/lib/utils"
import type { ChatMessage } from "@/lib/api/types"
import { api } from "@/lib/api/client"
import { ItemCard } from "./ItemCard"
import { OutfitBoard } from "./OutfitBoard"

interface Props {
  message: ChatMessage
  onSend?: (text: string) => void
  /** Brand id from the demo session (e.g. "snitch", "myntra"). Passed to OutfitBoard. */
  brand?: string
  /**
   * True only for the most recent assistant message. Suggestion chips are only
   * rendered here — clicking a chip from an older turn would resend a stale
   * refinement against a look that has since moved on.
   */
  isLatestAssistant?: boolean
}

// ---------------------------------------------------------------------------
// FeedbackButtons — thumbs up / down for assistant messages
// ---------------------------------------------------------------------------

interface FeedbackButtonsProps {
  messageId: string  // DB UUID (non-null; callers must guard before rendering)
}

function FeedbackButtons({ messageId }: FeedbackButtonsProps) {
  const [selected, setSelected] = useState<1 | -1 | null>(null)
  const [pending, setPending] = useState(false)

  async function handleRate(rating: 1 | -1) {
    if (pending) return
    setPending(true)
    try {
      await api.feedback.submit(messageId, rating)
      setSelected(rating)
    } catch {
      // Silently swallow errors — feedback is best-effort; don't disrupt the UX.
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="flex gap-1 mt-1">
      <button
        aria-label="Thumbs up"
        onClick={() => handleRate(1)}
        disabled={pending}
        className={cn(
          "rounded px-1.5 py-0.5 text-sm transition-colors",
          selected === 1
            ? "bg-emerald-100 text-emerald-700 opacity-100"
            : "text-muted-foreground opacity-40 hover:opacity-80 hover:bg-muted"
        )}
      >
        👍
      </button>
      <button
        aria-label="Thumbs down"
        onClick={() => handleRate(-1)}
        disabled={pending}
        className={cn(
          "rounded px-1.5 py-0.5 text-sm transition-colors",
          selected === -1
            ? "bg-rose-100 text-rose-700 opacity-100"
            : "text-muted-foreground opacity-40 hover:opacity-80 hover:bg-muted"
        )}
      >
        👎
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// MessageBubble
// ---------------------------------------------------------------------------

export function MessageBubble({ message, onSend, brand, isLatestAssistant }: Props) {
  const isUser = message.role === "user"

  return (
    <div
      className={cn(
        "flex flex-col gap-3",
        isUser ? "items-end" : "items-start"
      )}
    >
      {/* Text bubble */}
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
          isUser
            ? "bg-primary text-primary-foreground rounded-br-sm"
            : "bg-muted text-foreground rounded-bl-sm"
        )}
      >
        {isUser ? (
          <>
            {message.content}
            {message.imageUrl && (
              <img
                src={message.imageUrl}
                alt="Uploaded"
                className="h-20 w-auto rounded-md object-cover mt-1"
              />
            )}
          </>
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeSanitize]}
            components={{
              table: ({ children }) => (
                <div className="overflow-x-auto my-2">
                  <table className="text-xs border-collapse w-full">{children}</table>
                </div>
              ),
              thead: ({ children }) => (
                <thead className="border-b border-border/60">{children}</thead>
              ),
              th: ({ children }) => (
                <th className="px-2 py-1 text-left font-semibold">{children}</th>
              ),
              td: ({ children }) => (
                <td className="px-2 py-1 border-t border-border/30">{children}</td>
              ),
              p: ({ children }) => <p className="mb-1 last:mb-0">{children}</p>,
              ul: ({ children }) => (
                <ul className="list-disc pl-4 mb-1">{children}</ul>
              ),
              ol: ({ children }) => (
                <ol className="list-decimal pl-4 mb-1">{children}</ol>
              ),
              strong: ({ children }) => (
                <strong className="font-semibold">{children}</strong>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        )}
        {message.isStreaming && (
          <span className="inline-block w-1.5 h-4 ml-0.5 bg-current animate-pulse rounded-sm align-middle" />
        )}
      </div>

      {/* Product cards (assistant messages only) */}
      {!isUser && message.items.length > 0 && (() => {
        const isOutfit = message.items.some((it) => it.slot_role != null)
        if (isOutfit) {
          const seed = message.items.find((it) => it.slot_role === "seed")
          return (
            <div className="w-full max-w-[85%]">
              <OutfitBoard
                items={message.items}
                lookId={message.lookId}
                occasion={message.occasion}
                lookGender={message.lookGender}
                budgetTotalInr={message.budgetTotalInr}
                outfitRationale={message.outfitRationale}
                outfitVariants={message.outfitVariants}
                cartUrl={message.cartUrl}
                itemLinks={message.itemLinks}
                suppressedSlots={message.suppressedSlots}
                lookRole={message.lookRole}
                lookTitle={message.lookTitle}
                coordinatedWith={message.coordinatedWith}
                anchorImageUrl={message.anchorImageUrl}
                sessionId={message.id}
                anchorItemId={seed?.article_id ?? ""}
                anchorCategory={seed?.product_type ?? ""}
                brand={brand}
                onSend={onSend}
              />
            </div>
          )
        }
        return (
          <div className="w-full max-w-[80%] grid grid-cols-1 sm:grid-cols-2 gap-2">
            {message.items.map((item) => (
              <ItemCard key={item.article_id} item={item} onSend={onSend} />
            ))}
          </div>
        )
      })()}

      {/* Backend-suggested follow-up chips — latest assistant message only, so a
          click always applies to the current (not a stale) look/turn. */}
      {!isUser &&
        isLatestAssistant &&
        !message.isStreaming &&
        onSend &&
        message.suggestionChips != null &&
        message.suggestionChips.length > 0 && (
          <div className="flex flex-wrap gap-1.5 max-w-[85%]">
            {message.suggestionChips.map((chip, i) => (
              <button
                key={`${chip}-${i}`}
                onClick={() => onSend(chip)}
                className="rounded-full border text-xs px-3 py-1 text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
              >
                {chip}
              </button>
            ))}
          </div>
        )}

      {/* Feedback buttons (assistant messages with a persisted DB id only) */}
      {!isUser && !message.isStreaming && message.dbId !== null && (
        <FeedbackButtons messageId={message.dbId} />
      )}
    </div>
  )
}
