"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import rehypeSanitize from "rehype-sanitize"
import { cn } from "@/lib/utils"
import type { ChatMessage } from "@/lib/api/types"
import { ItemCard } from "./ItemCard"

interface Props {
  message: ChatMessage
  onSend?: (text: string) => void
}

export function MessageBubble({ message, onSend }: Props) {
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
          message.content
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
      {!isUser && message.items.length > 0 && (
        <div className="w-full max-w-[80%] grid grid-cols-1 sm:grid-cols-2 gap-2">
          {message.items.map((item) => (
            <ItemCard key={item.article_id} item={item} onSend={onSend} />
          ))}
        </div>
      )}
    </div>
  )
}
