import Link from "next/link"
import { Logo } from "@/components/Logo"
import { LoginForm } from "@/components/auth/LoginForm"

interface Props {
  // In Next.js 15 searchParams is a Promise in Server Components.
  searchParams: Promise<{ error?: string }>
}

/**
 * The root URL redirects anonymous visitors here, so this page IS the landing
 * page for anyone typing stylemaitri.vercel.app. It shipped as a bare unbranded
 * sign-in card — no logo, no product name, no way to reach the demo (defect
 * sweep 2026-07-10, P2-10). Brand header + demo entry above the auth card.
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
            href="/demo"
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
