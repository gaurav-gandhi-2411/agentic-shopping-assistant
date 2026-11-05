# Security

## Advisory allow-list

`security/advisory-allowlist.yaml` is the only place a known dependency advisory may be ignored in
CI. It ships **empty** (`entries: []`): today no advisory is allow-listed and `pip-audit` and
`npm audit --audit-level=high` run with no ignores.

An entry needs `id`, `ecosystem` (`pip`|`npm`), `package`, `justification` (>= 40 chars, why the
vulnerable code path is unreachable), `added` (not in the future), `expires` (after `added`, at
most 30 days later, not in the past) and `approver: GG`. Ids must be valid for the ecosystem
(pip: `PYSEC-YYYY-N`, `GHSA-xxxx-xxxx-xxxx`, `CVE-YYYY-NNNN`; npm: `GHSA-...` or a numeric id).
Expired, malformed, duplicate or unreadable files fail CI.

`python scripts/check_advisory_allowlist.py check` validates the file (backend job);
`pip-args` feeds `pip-audit`; `npm-audit` replaces the plain `npm audit` step with the same
high/critical threshold plus the allow-list filter.
