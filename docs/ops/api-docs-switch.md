# DISABLE_API_DOCS: hiding /docs, /redoc and /openapi.json

`DISABLE_API_DOCS` (read in `api/main.py` `create_app`, truthy = `1`/`true`/`yes`, case-insensitive)
creates the FastAPI app with `docs_url`, `redoc_url` and `openapi_url` all `None`, so the three
routes return 404. **Default: unset = OFF = docs stay enabled (behaviour unchanged).** Every other
route, including `/ping`, `/readyz` and the API, is unaffected either way (`tests/test_disable_api_docs.py`).

## Why it stays OFF at merge

The Cloud Scheduler keep-warm job `asa-keepwarm-healthz` currently does `GET /openapi.json`. Turning
the switch on before the job is repointed would make that job get 404s and stop keeping the
instance warm. `/ping` (this PR) exists so the job has a dependency-free target that Cloud Run's
front end does not intercept (it 404s `/healthz`, hence a path not ending in `z`).

## Steps for GG (in this order, after this PR is deployed)

1. Confirm the new revision serves `/ping`:
   `curl -sS -o /dev/null -w '%{http_code}\n' https://asa-stylist-api-ec3mwppesa-el.a.run.app/ping`
   must print `200`. (The rollback revision lacks `/ping`; do not repoint while it is serving.)
2. Repoint the keep-warm job:
   ```
   gcloud scheduler jobs update http asa-keepwarm-healthz --account=gaurav.gandhi1129@gmail.com --project=stylemaitri-prod-260813 --location=asia-south1 --uri=https://asa-stylist-api-ec3mwppesa-el.a.run.app/ping
   ```
3. Confirm 200 rows for `/ping` in the Cloud Run request logs (the app does not log `/ping`
   itself; the request log is Cloud Run's own).
4. Only then set `DISABLE_API_DOCS=true` on the service.

## How to change the env var safely (deploy parity check)

`.github/workflows/deploy.yml` deploys with `gcloud run deploy --image ... --no-traffic` and passes
no env flags, so the new revision **inherits the service template's env**; the step "Assert new
revision kept prod config" then diffs the new revision's `env` against the **serving** revision
and fails the deploy (before any traffic shift) on any difference. Consequences:

- Do not add the variable only to a future deploy's flags: the workflow has no env mechanism, and
  an env that exists in the template but not on the serving revision makes the next deploy fail parity.
- Set it on the service itself, so the template and the revision that then serves both carry it:
  `gcloud run services update asa-stylist-api --account=gaurav.gandhi1129@gmail.com --project=stylemaitri-prod-260813 --region=asia-south1 --update-env-vars DISABLE_API_DOCS=true`
  (`--update-env-vars` adds/changes only this variable and keeps the others). This creates a new
  revision; check with `gcloud run services describe ... --format "json(spec.traffic,status.traffic)"`
  that it takes traffic (if traffic was pinned to a named revision, shift it deliberately, per DEPLOY.md).
- After it serves, the next deploy's parity check compares like with like (both have the variable).
- Rollback: `--remove-env-vars DISABLE_API_DOCS` the same way (or `--update-env-vars DISABLE_API_DOCS=false`).
- `deploy/service.yaml` is emergency-recovery only; add the variable there too if you use it for a
  `services replace`, or that replace will silently turn docs back on.

UNVERIFIED: none of the above was run against the cloud in this change (no cloud writes). The
inheritance behaviour is taken from the workflow's own comment and gcloud's documented behaviour.
