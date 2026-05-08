"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import {
  useConversations,
  useDeleteConversation,
  usePatchConversation,
} from "@/hooks/useConversations"
import type { ConversationSummary } from "@/lib/api/types"

interface Props {
  activeId: string | null
  onSelect: (id: string) => void
  onNew: () => void
}

export function ConversationSidebar({ activeId, onSelect, onNew }: Props) {
  const { data: conversations, isLoading } = useConversations()
  const { mutate: deleteConversation } = useDeleteConversation()
  const { mutate: patchConversation } = usePatchConversation()

  return (
    <aside className="w-60 border-r bg-background flex flex-col shrink-0 min-h-0">
      {/* New chat */}
      <div className="p-3 border-b shrink-0">
        <Button onClick={onNew} variant="outline" className="w-full" size="sm">
          + New chat
        </Button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto py-2">
        {isLoading && (
          <p className="text-xs text-muted-foreground px-3 py-2">Loading…</p>
        )}
        {!isLoading && (!conversations || conversations.length === 0) && (
          <p className="text-xs text-muted-foreground px-3 py-2">
            No conversations yet.
          </p>
        )}
        {conversations?.map((c) => (
          <ConversationItem
            key={c.conversation_id}
            conversation={c}
            isActive={c.conversation_id === activeId}
            onSelect={() => onSelect(c.conversation_id)}
            onDelete={() => deleteConversation(c.conversation_id)}
            onToggleShare={() =>
              patchConversation({
                id: c.conversation_id,
                body: { is_public: !c.is_public },
              })
            }
          />
        ))}
      </div>
    </aside>
  )
}

interface ItemProps {
  conversation: ConversationSummary
  isActive: boolean
  onSelect: () => void
  onDelete: () => void
  onToggleShare: () => void
}

function ConversationItem({
  conversation,
  isActive,
  onSelect,
  onDelete,
  onToggleShare,
}: ItemProps) {
  return (
    <div
      className={cn(
        "group relative flex items-center gap-2 px-3 py-2 cursor-pointer rounded-md mx-1 hover:bg-accent transition-colors",
        isActive && "bg-accent"
      )}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onSelect()}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1">
          <p className="text-sm font-medium truncate leading-tight">
            {conversation.title}
          </p>
          {conversation.is_public && (
            <span
              className="shrink-0 text-[10px] text-muted-foreground"
              title="Shared link active"
              aria-label="Shared"
            >
              🌐
            </span>
          )}
        </div>
        <p className="text-xs text-muted-foreground truncate leading-tight mt-0.5">
          {conversation.message_count === 1
            ? "1 turn"
            : `${conversation.message_count} turns`}
        </p>
      </div>

      {/* Action buttons — visible on hover */}
      <div className="shrink-0 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          className={cn(
            "text-muted-foreground hover:text-foreground p-1 rounded text-xs",
            conversation.is_public && "text-foreground"
          )}
          onClick={(e) => {
            e.stopPropagation()
            onToggleShare()
          }}
          aria-label={
            conversation.is_public ? "Make private" : "Share conversation"
          }
          title={conversation.is_public ? "Make private" : "Share"}
        >
          🌐
        </button>
        <button
          className="text-muted-foreground hover:text-destructive p-1 rounded"
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
          aria-label={`Delete "${conversation.title}"`}
          title="Delete"
        >
          ✕
        </button>
      </div>
    </div>
  )
}
