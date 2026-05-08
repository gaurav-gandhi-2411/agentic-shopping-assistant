"use client"

interface Props {
  userEmail: string
}

export function ChatPlaceholder({ userEmail }: Props) {
  return (
    <div className="flex flex-col items-center justify-center flex-1 min-h-[60vh] gap-4 text-center px-4">
      <div className="text-5xl select-none" aria-hidden>
        🛍️
      </div>
      <h1 className="text-2xl font-semibold tracking-tight">
        Shopping Assistant
      </h1>
      <p className="text-muted-foreground text-sm max-w-xs">
        Signed in as <span className="font-medium text-foreground">{userEmail}</span>.
        The chat interface is coming in the next task.
      </p>
    </div>
  )
}
