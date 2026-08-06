/**
 * Canonical "gender already known from this conversation" derivation —
 * used by every chat surface that offers BodyShapeUpload's men's/women's
 * manual-picker chips (Area 1, 2026-07-25).
 *
 * Root-cause context (live-proven bug): this logic was first written inline
 * in ChatThread.tsx (the /chat route) only. Copy-pasting it into
 * app/demo/chat/page.tsx and app/embed/[brand]/page.tsx (two OTHER,
 * independent useChatStream() call sites that also render <ChatInput>) was
 * caught as a live gap during proof — the men's chips silently never
 * appeared on /demo/chat (the actual public-facing surface used for
 * verification) because that page never derived knownGender at all. Third
 * occurrence of the same three-line logic -> factored out here so future
 * chat surfaces can't independently forget to wire it.
 */

import type { ChatMessage } from "@/lib/api/types"

/**
 * Returns the gender ("men"/"women"/"unisex") already resolved server-side
 * by the most recent message that composed an outfit, or null when
 * genuinely unknown (no outfit composed yet this session). NEVER derives
 * from message text or the photo-shape path itself — this is the same
 * signal the backend already resolved (ChatMessage.lookGender, sourced from
 * the WS "done" frame's final_state.look_gender).
 */
export function deriveKnownGender(messages: ChatMessage[]): string | null {
  return [...messages].reverse().find((m) => m.lookGender)?.lookGender ?? null
}
