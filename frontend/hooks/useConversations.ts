"use client"

import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import { api } from "@/lib/api/client"
import type { ConversationDetail, ConversationSummary } from "@/lib/api/types"

export const CONVERSATIONS_KEY = ["conversations"] as const

export function useConversations() {
  return useQuery<ConversationSummary[]>({
    queryKey: CONVERSATIONS_KEY,
    queryFn: () => api.conversations.list(),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
}

export function useConversation(id: string | null) {
  return useQuery<ConversationDetail>({
    queryKey: [...CONVERSATIONS_KEY, id],
    queryFn: () => api.conversations.get(id!),
    enabled: id !== null,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  })
}

export function useDeleteConversation() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.conversations.delete(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: CONVERSATIONS_KEY }),
  })
}

export function usePatchConversation() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: string
      body: { title?: string; is_public?: boolean }
    }) => api.conversations.patch(id, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: CONVERSATIONS_KEY }),
  })
}

export function useInvalidateConversations() {
  const qc = useQueryClient()
  return () => qc.invalidateQueries({ queryKey: CONVERSATIONS_KEY })
}
