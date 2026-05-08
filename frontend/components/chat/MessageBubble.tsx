import { cn } from "@/lib/utils"
import type { ChatMessage } from "@/lib/api/types"
import { ItemCard } from "./ItemCard"

interface Props {
  message: ChatMessage
}

export function MessageBubble({ message }: Props) {
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
        {message.content}
        {message.isStreaming && (
          <span className="inline-block w-1.5 h-4 ml-0.5 bg-current animate-pulse rounded-sm align-middle" />
        )}
      </div>

      {/* Product cards (assistant messages only) */}
      {!isUser && message.items.length > 0 && (
        <div className="w-full max-w-[80%] grid grid-cols-1 sm:grid-cols-2 gap-2">
          {message.items.map((item) => (
            <ItemCard key={item.article_id} item={item} />
          ))}
        </div>
      )}
    </div>
  )
}
