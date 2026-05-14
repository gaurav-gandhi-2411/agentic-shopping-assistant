import { redirect } from "next/navigation"
import { createClient } from "@/lib/supabase/server"

// Root route: redirect authenticated users to /chat, everyone else to /login.
// Middleware handles this for subsequent navigations; this covers the cold-load case.
export default async function RootPage() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (user) {
    redirect("/chat")
  } else {
    redirect("/login")
  }
}
