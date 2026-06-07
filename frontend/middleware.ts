import { createServerClient, type CookieOptions } from "@supabase/ssr"
import { NextResponse, type NextRequest } from "next/server"

export async function middleware(request: NextRequest) {
  // Create a mutable response we can attach refreshed cookies to.
  // Re-assigned inside setAll when Supabase rotates the session token.
  let supabaseResponse = NextResponse.next({ request })

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll()
        },
        setAll(cookiesToSet: { name: string; value: string; options: CookieOptions }[]) {
          // Forward new cookies onto the outgoing request so downstream
          // server components read the refreshed session.
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value)
          )
          // Recreate the response so it carries the updated request cookies.
          supabaseResponse = NextResponse.next({ request })
          // Also set them on the response so the browser stores them.
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          )
        },
      },
    }
  )

  // IMPORTANT: call getUser() immediately after createServerClient with no
  // code in between. Any await between creation and getUser() can cause
  // session-refresh race conditions that randomly log users out.
  const {
    data: { user },
  } = await supabase.auth.getUser()

  const { pathname } = request.nextUrl

  // Unauthenticated users may only access /login and /auth/*.
  if (
    !user &&
    !pathname.startsWith("/login") &&
    !pathname.startsWith("/auth") &&
    !pathname.startsWith("/demo")
  ) {
    const url = request.nextUrl.clone()
    url.pathname = "/login"
    return NextResponse.redirect(url)
  }

  // Authenticated users landing on /login go straight to /chat.
  if (user && pathname === "/login") {
    const url = request.nextUrl.clone()
    url.pathname = "/chat"
    return NextResponse.redirect(url)
  }

  // Return supabaseResponse (not a freshly constructed NextResponse.next())
  // so the refreshed session cookies are forwarded to the browser.
  return supabaseResponse
}

export const config = {
  matcher: [
    // Skip Next.js internals, static files, and common image extensions.
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
}
