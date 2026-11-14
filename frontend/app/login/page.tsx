import Link from "next/link"
import { Logo } from "@/components/Logo"
import { LoginForm } from "@/components/auth/LoginForm"

interface Props {
  // In Next.js 15 searchParams is a Promise in Server Components.
  searchParams: Promise<{ error?: string }>
}

/**
 * Sign-in is optional now: `/` is the working assistant for anonymous visitors
 * (Wave 1b V2), so this page is only reached on purpose (e.g. from the optional
 * sign-in link shown after a look is saved). It keeps a brand header and a way
 * back to the assistant above the auth card (defect sweep 2026-07-10, P2-10).
 */
export default async function LoginPage({ searchParams }: Props) {
  const { error } = await searchParams
  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-sm space-y-6">
        <div className="flex flex-col items-center text-center gap-3">
          <Logo />
          <p className="text-sm text-muted-foreground leading-relaxed">
            Your AI stylist for weddings, sangeets &amp; every day.
          </p>
          <Link
            href="/"
            className="inline-block rounded-lg bg-primary text-primary-foreground text-sm font-semibold px-5 py-2.5 hover:bg-primary/90 transition-colors"
          >
            Try the live demo — no sign-in needed
          </Link>
        </div>

        <div className="relative text-center">
          <span className="relative z-10 bg-background px-3 text-[11px] uppercase tracking-wide text-muted-foreground">
            or sign in
          </span>
          <span className="absolute left-0 right-0 top-1/2 border-t" aria-hidden />
        </div>

        <LoginForm authError={error} />
      </div>
    </div>
  )
}
