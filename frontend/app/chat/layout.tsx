import { redirect } from "next/navigation"
import { createClient } from "@/lib/supabase/server"
import { SignOutButton } from "@/components/auth/SignOutButton"
import { Logo } from "@/components/Logo"

export default async function ChatLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/login")
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <header className="border-b-2 border-primary/15 bg-background px-6 py-3 flex items-center justify-between shrink-0">
        <Logo />
        <div className="flex items-center gap-4">
          <span className="text-sm text-muted-foreground hidden sm:block">
            {user.email}
          </span>
          <SignOutButton />
        </div>
      </header>
      <div className="flex-1 flex flex-col min-h-0">{children}</div>
    </div>
  )
}
