import { ChatThread } from "@/components/chat/ChatThread"

// Auth guard is handled by middleware.ts and the layout.
// This page just mounts the client-side chat UI.
export default function ChatPage() {
  return <ChatThread />
}
