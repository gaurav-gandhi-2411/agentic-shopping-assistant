import { LoginForm } from "@/components/auth/LoginForm"

interface Props {
  // In Next.js 15 searchParams is a Promise in Server Components.
  searchParams: Promise<{ error?: string }>
}

export default async function LoginPage({ searchParams }: Props) {
  const { error } = await searchParams
  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <LoginForm authError={error} />
    </div>
  )
}
