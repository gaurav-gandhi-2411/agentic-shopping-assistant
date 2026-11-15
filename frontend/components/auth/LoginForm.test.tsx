import { afterEach, describe, expect, it, vi } from "vitest"

// LoginForm reads NEXT_PUBLIC_GOOGLE_AUTH_ENABLED into a module-level constant, so each case
// resets the module registry and imports it fresh with the variable stubbed. React and
// react-dom/server are imported after the reset too, so they share one React instance.
vi.mock("@/lib/supabase/client", () => ({ createClient: () => ({ auth: {} }) }))

afterEach(() => {
  vi.unstubAllEnvs()
  vi.resetModules()
})

async function renderLoginForm(flag: string | undefined): Promise<string> {
  vi.resetModules()
  if (flag === undefined) vi.stubEnv("NEXT_PUBLIC_GOOGLE_AUTH_ENABLED", "")
  else vi.stubEnv("NEXT_PUBLIC_GOOGLE_AUTH_ENABLED", flag)
  const React = await import("react")
  const { renderToStaticMarkup } = await import("react-dom/server")
  const { LoginForm } = await import("./LoginForm")
  return renderToStaticMarkup(React.createElement(LoginForm, {}))
}

describe("LoginForm: Google sign-in is opt-in", () => {
  it("hides the Google button, the 'or' divider and the Google wording by default", async () => {
    const html = await renderLoginForm(undefined)
    expect(html).not.toContain("Continue with Google")
    expect(html).not.toMatch(/>or</)
    expect(html).not.toMatch(/Google/)
    expect(html).toContain("Enter your email for a magic link.")
    expect(html).toContain("Send magic link") // email sign-in is unaffected
  })

  it("shows it only when the flag is exactly 'true'", async () => {
    const html = await renderLoginForm("true")
    expect(html).toContain("Continue with Google")
    expect(html).toMatch(/>or</)
    expect(html).toContain("or continue with Google")
  })

  it.each(["1", "TRUE", "yes", "false"])("keeps it hidden for flag value %s", async (flag) => {
    expect(await renderLoginForm(flag)).not.toContain("Continue with Google")
  })
})
