<#
.SYNOPSIS
    Deploy the frontend (stylemaitri) to Vercel production and record deploy provenance.

.DESCRIPTION
    Wraps the documented Vercel deploy path (see DEPLOY.md, "Deploy Vercel frontend":
    `cd frontend && vercel --prod`). A manual `vercel --prod` deploy leaves no record
    anywhere of which git commit was actually live -- this script closes that gap:
      1. Refuses to run if frontend/ has any uncommitted change (staged, unstaged, or
         untracked) -- the deployed artifact must match a real commit.
      2. Runs the existing, documented deploy command.
      3. Parses the production URL vercel prints on success out of its own output.
      4. Appends a provenance line to reports/deploys.log.
      5. Commits and pushes that log line.

    This does NOT replace the manual steps in DEPLOY.md (Vercel env var setup, CORS
    configuration, etc.) -- it only wraps the final `vercel --prod` step with dirty-tree
    protection and a provenance record, exactly as scripts/deploy_backend.ps1 wraps the
    backend's `gcloud run deploy` step.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

# Resolve repo root relative to this script rather than the caller's cwd, so the script
# behaves the same whether invoked from repo root or elsewhere (same pattern as
# scripts/deploy_backend.ps1).
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$FrontendDir = Join-Path $RepoRoot "frontend"
if (-not (Test-Path $FrontendDir)) {
    Write-Error "frontend/ not found at $FrontendDir. Aborting."
    exit 1
}

# --- 1. Refuse to deploy from a dirty tree -------------------------------------------
# Scoped to frontend/, not the whole repo: `vercel --prod` only ever packages the
# frontend/ directory (see DEPLOY.md), so that subtree's cleanliness is what actually
# determines the deployed artifact. Uncommitted changes elsewhere in this monorepo
# (backend/, eval/, docs) don't affect what Vercel serves and shouldn't block this.
$gitStatusOutput = git status --porcelain -- frontend
if ($LASTEXITCODE -ne 0) {
    Write-Error "git status failed (exit $LASTEXITCODE). Is this a git repo?"
    exit 1
}
if ($gitStatusOutput) {
    Write-Error ("frontend/ has uncommitted changes (staged, unstaged, or untracked) " +
        "-- refusing to deploy from a dirty tree. Commit or stash first:`n$gitStatusOutput")
    exit 1
}

# --- 2. Resolve the commit SHA that's about to go live --------------------------------
$Sha = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $Sha) {
    Write-Error "Could not resolve git HEAD SHA (exit $LASTEXITCODE)."
    exit 1
}
$ShortSha = $Sha.Substring(0, 7)

# --- 3. Run the documented deploy flow (DEPLOY.md: cd frontend && vercel --prod) ------
Write-Host "Deploying frontend at commit $ShortSha to Vercel production..."
Push-Location $FrontendDir
try {
    # --yes replaces the interactive confirmation prompt DEPLOY.md's manual flow relies
    # on, since this script runs non-interactively. --prod matches DEPLOY.md exactly.
    # 2>&1 merges stderr into the captured output so the URL-parsing step below finds
    # the production URL regardless of which stream a given vercel CLI version writes it
    # to; ForEach-Object ToString() normalizes native-command stderr lines (PowerShell
    # wraps them as ErrorRecord objects) back into plain strings for regex matching.
    $rawOutput = vercel --prod --yes 2>&1
    $vercelExitCode = $LASTEXITCODE
    $vercelOutput = $rawOutput | ForEach-Object { $_.ToString() }
} finally {
    Pop-Location
}

if ($vercelExitCode -ne 0) {
    Write-Error ("vercel --prod failed (exit $vercelExitCode). Deploy log was NOT " +
        "updated. Output:`n$($vercelOutput -join "`n")")
    exit 1
}

# --- 4. Parse the production URL out of vercel's own output ---------------------------
# vercel --prod prints the resulting URL as a bare "https://..." line when stdout isn't
# a TTY (exactly the case here, piped through PowerShell) -- this is the documented CLI
# scripting convention (`url=$(vercel --prod --yes)` in Vercel's own CI examples). Take
# the LAST https:// match rather than the first: an "Inspect: https://vercel.com/..."
# dashboard link is printed before the production URL and would otherwise win instead.
$urlMatches = $vercelOutput | Select-String -Pattern 'https://\S+' -AllMatches |
    ForEach-Object { $_.Matches.Value }
if (-not $urlMatches) {
    Write-Error ("vercel --prod exited 0 but no https:// URL was found in its output " +
        "-- cannot record provenance. Raw output:`n$($vercelOutput -join "`n")")
    exit 1
}
$DeployUrl = ($urlMatches | Select-Object -Last 1).TrimEnd(']', ')', '.', ',')

Write-Host "Deployed: $DeployUrl"

# --- 5. Append the provenance line to reports/deploys.log -----------------------------
$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$LogLine = "$Timestamp  sha=$Sha  url=$DeployUrl  target=production"
$LogPath = Join-Path (Join-Path $RepoRoot "reports") "deploys.log"
try {
    Add-Content -Path $LogPath -Value $LogLine
} catch {
    Write-Error "Failed to write $LogPath. Deploy succeeded but was NOT recorded -- append the line manually:`n$LogLine`nError: $_"
    exit 1
}
Write-Host "Recorded: $LogLine"

# --- 6. Commit and push the provenance record ------------------------------------------
# reports/ is gitignored wholesale (bulk eval-report output) and *.log is gitignored
# repo-wide, so reports/deploys.log needs an explicit -f to be tracked at all -- a plain
# `git add` would silently no-op here and the commit below would then fail with
# "nothing to commit", masking the real problem.
git add -f -- reports/deploys.log
if ($LASTEXITCODE -ne 0) {
    Write-Error ("git add -f reports/deploys.log failed (exit $LASTEXITCODE). Deploy " +
        "succeeded but provenance was NOT committed -- commit it manually.")
    exit 1
}

# Pathspec-scoped commit: only stages/commits this one file, even if unrelated changes
# happen to be staged elsewhere in the tree at the time this runs.
git commit -m "chore(deploy): record production deploy $ShortSha" -- reports/deploys.log
if ($LASTEXITCODE -ne 0) {
    Write-Error ("git commit failed (exit $LASTEXITCODE). Deploy succeeded but " +
        "provenance was NOT committed -- commit reports/deploys.log manually.")
    exit 1
}

$CurrentBranch = (git rev-parse --abbrev-ref HEAD).Trim()
git push origin $CurrentBranch
if ($LASTEXITCODE -ne 0) {
    Write-Error ("git push failed (exit $LASTEXITCODE). Deploy succeeded and " +
        "provenance was committed locally on '$CurrentBranch' but NOT pushed -- push it " +
        "manually.")
    exit 1
}

Write-Host ""
Write-Host "==================== Deploy summary ===================="
Write-Host "Commit:  $Sha"
Write-Host "URL:     $DeployUrl"
Write-Host "Log:     reports/deploys.log"
Write-Host "==========================================================="
