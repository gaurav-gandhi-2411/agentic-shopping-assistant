# Email Allow-List

Access to this application is restricted to a pre-approved set of email
addresses.  Signup attempts from addresses not on the list are rejected before
the Supabase Auth account is created.

---

## How it works

1. `public.allowed_emails` — the source of truth.  A row means the address is
   permitted.
2. `public.check_email_allowed(email TEXT) → BOOLEAN` — called by the hook.
   SECURITY DEFINER so it reads the table regardless of the caller's RLS
   context.
3. `public.before_user_created(event jsonb) → jsonb` — the Supabase Auth Hook
   function.  Returns `'{}'` to allow or an error object to block the signup.
4. The hook is wired in the Supabase dashboard:
   **Authentication → Hooks → Before User Created → Postgres Function →
   `public.before_user_created`**

Emails are **automatically lowercased** on insert or update — a BEFORE
INSERT OR UPDATE trigger calls `lower()` before anything is stored.
You can type `INSERT INTO allowed_emails VALUES ('User@Example.com', ...)` and
it will be stored and matched as `user@example.com`.  No need to remember to
lowercase manually.

---

## Adding an email

Run in the Supabase **SQL Editor** (or via `psql` with the service-role key):

```sql
INSERT INTO public.allowed_emails (email, note)
VALUES ('user@example.com', 'team member — joined 2026-05');
```

The `note` column is free text — use it to record who approved the addition
and when.  The `added_by` column accepts a `public.users.id` UUID if you want
to track which admin added the row:

```sql
INSERT INTO public.allowed_emails (email, added_by, note)
VALUES (
    'user@example.com',
    '00000000-0000-0000-0000-000000000001',   -- your user UUID
    'team member'
);
```

---

## Removing an email

```sql
DELETE FROM public.allowed_emails WHERE email = 'user@example.com';
```

Removing an email does **not** delete the Supabase Auth account or revoke
existing sessions.  To fully revoke access:

1. Delete from allow-list (above).
2. In the Supabase dashboard: **Authentication → Users → [user] → Delete**.
3. Existing JWTs expire according to their `exp` claim (default 1 hour for
   Supabase access tokens).  If immediate revocation is needed, rotate the
   JWT secret: **Project Settings → API → JWT Secret → Rotate**.

---

## Viewing the allow-list

Any authenticated user (not just admins) can read the table — this is
intentional so the app can show a human-readable "contact the team" message:

```sql
SELECT email, added_by, added_at, note
FROM public.allowed_emails
ORDER BY added_at DESC;
```

---

## Wiring the hook (one-time Supabase dashboard step)

The migration (`0003_allowed_emails.py`) creates the function but **cannot**
register it as an Auth Hook via SQL alone — the Supabase Auth Hook API
requires a dashboard action:

1. Open the Supabase dashboard for your project.
2. Go to **Authentication → Hooks**.
3. Under **Before User Created**, click **Add hook**.
4. Choose **Postgres Function**.
5. Select schema `public`, function `before_user_created`.
6. Click **Save**.

From this point on, every new signup attempt is checked against
`public.allowed_emails` before the account is created.

> **Note — local development:** The hook is not enforced on plain Postgres
> (local Docker / CI) because there is no Supabase Auth process to call it.
> Local dev connects as superuser and bypasses allow-list checks entirely.
> Add your own email to `allowed_emails` before wiring the hook on Supabase
> or you will lock yourself out.

---

## First-time setup checklist

- [ ] Run migrations: `python scripts/run_migrations.py`
- [ ] Add your own email: `INSERT INTO public.allowed_emails ...`
- [ ] Add any other team members
- [ ] Wire the hook in the dashboard (see above)
- [ ] Test: try signing up with an address NOT in the list — should get 403
- [ ] Test: sign up with an allowed address — should succeed
