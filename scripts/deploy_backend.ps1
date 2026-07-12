<#
.SYNOPSIS
    Build, push, and deploy the unified ASA backend service to Cloud Run.

.DESCRIPTION
    This is THE documented backend deploy path (see DEPLOY.md, "Deploy backend (unified
    service)"). The GitHub Actions workflow that used to cover this
    (.github/workflows/deploy-demo.yml) was removed 2026-07-09: it never had its required
    secrets (WIF_PROVIDER, WIF_SERVICE_ACCOUNT, GCP_PROJECT_ID, GROQ_API_KEY,
    DEMO_JWT_SECRET, DATABASE_URL, SUPABASE_URL, GCS_BUCKET) configured on the repo, so it
    never ran successfully. Every real deploy has always happened locally; this script
    codifies that path instead of pretending CI covers it.

.PARAMETER Project
    GCP project ID.

.PARAMETER Region
    Cloud Run / Artifact Registry region.

.PARAMETER Service
    Cloud Run service name.

.PARAMETER GarRepo
    Artifact Registry repository name.

.PARAMETER ImageName
    Docker image name within the Artifact Registry repo.

.PARAMETER Tag
    Docker image tag. Defaults to "wave-deploy-<git short sha>" computed from HEAD.

.PARAMETER SkipBuild
    Skip the docker build/push steps and deploy an already-pushed image tag (pass -Tag
    to identify which one).
#>

[CmdletBinding()]
param(
    [string]$Project = "iconic-reactor-496423-m4",
    [string]$Region = "asia-south1",
    [string]$Service = "asa-stylist-api",
    [string]$GarRepo = "shopping-assistant",
    [string]$ImageName = "asa-api",
    [switch]$SkipBuild,
    [string]$Tag
)

$ErrorActionPreference = "Stop"

# Resolve repo root relative to this script rather than the caller's cwd, so the script
# behaves the same whether invoked from repo root or elsewhere.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if (-not (Test-Path (Join-Path $RepoRoot "Dockerfile"))) {
    Write-Error "Dockerfile not found at repo root ($RepoRoot). Aborting."
    exit 1
}

# Dirty tree is a warning, not a hard stop: iterating on this script itself, or on a
# hotfix, often means deploying before the commit lands. The warning just makes that an
# explicit, visible choice instead of a silent surprise later.
$gitStatusOutput = git status --porcelain
if ($LASTEXITCODE -ne 0) {
    Write-Error "git status failed (exit $LASTEXITCODE). Is this a git repo?"
    exit 1
}
if ($gitStatusOutput) {
    Write-Warning "deploying a dirty working tree"
}

# Tag defaults to the current commit so a bad deploy can always be traced back to the
# exact source that produced it.
if (-not $Tag) {
    $shortSha = (git rev-parse --short HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $shortSha) {
        Write-Error "Could not resolve git HEAD short SHA (exit $LASTEXITCODE)."
        exit 1
    }
    $Tag = "wave-deploy-$shortSha"
}

$Image = "$Region-docker.pkg.dev/$Project/$GarRepo/${ImageName}:$Tag"

if (-not $SkipBuild) {
    Write-Host "Building $Image"
    docker build -t $Image .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "docker build failed (exit $LASTEXITCODE)."
        exit 1
    }

    gcloud auth configure-docker "$Region-docker.pkg.dev" --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Error "gcloud auth configure-docker failed (exit $LASTEXITCODE)."
        exit 1
    }

    Write-Host "Pushing $Image"
    docker push $Image
    if ($LASTEXITCODE -ne 0) {
        Write-Error "docker push failed (exit $LASTEXITCODE)."
        exit 1
    }
} else {
    Write-Host "SkipBuild set - deploying already-pushed image $Image"
}

# Capture the revision currently serving traffic BEFORE we deploy, so we always have a
# known-good rollback anchor. We read the revision at 100% traffic rather than
# latestReadyRevisionName: a prior deploy can leave traffic pinned to an older revision
# (see the pinned-traffic gotcha below), in which case the two differ and
# latestReadyRevisionName would be the wrong rollback target.
$describeBeforeJson = gcloud run services describe $Service --region $Region --project $Project --format json
if ($LASTEXITCODE -ne 0) {
    Write-Error "gcloud run services describe (pre-deploy) failed (exit $LASTEXITCODE)."
    exit 1
}
$serviceBefore = $describeBeforeJson | ConvertFrom-Json
$trafficBefore = $serviceBefore.status.traffic | Where-Object { $_.percent -eq 100 }
if ($trafficBefore) {
    $RollbackAnchor = $trafficBefore.revisionName
} else {
    $RollbackAnchor = $serviceBefore.status.latestReadyRevisionName
}
Write-Host "Rollback anchor (revision currently serving 100% traffic): $RollbackAnchor"

# NO --set-env-vars here, ever. gcloud run deploy --set-env-vars REPLACES the entire env
# block rather than merging into it. Deploying with --image only inherits the previous
# revision's env block untouched. We were bitten by this: --set-env-vars wiped
# manually-raised DEMO_PER_IP_HOUR_LIMIT and DEMO_DAILY_REQUEST_CAP that existed only on
# the running revision. To change env vars, run
# `gcloud run services update <service> --update-env-vars=...` as its own deliberate step,
# never bundled with an image deploy.
gcloud run deploy $Service --image=$Image --region $Region --project $Project --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error "gcloud run deploy failed (exit $LASTEXITCODE)."
    exit 1
}

# `gcloud run deploy` creates a new revision but does NOT move traffic onto it if traffic
# is pinned to a named revision (which this service's traffic history has done before) --
# that produced a silent "deployed but not live" outcome. --to-latest reassigns the
# untagged 100%-traffic slot to whichever revision is newest; it does not touch or remove
# existing revision tags (e.g. the w1test 0%-traffic rollback tag), so tagged rollback
# references survive this call.
gcloud run services update-traffic $Service --to-latest --region $Region --project $Project --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error "gcloud run services update-traffic failed (exit $LASTEXITCODE)."
    exit 1
}

# Verify traffic actually landed on the revision we just deployed. This is the check that
# would have caught the pinned-traffic incident before it reached "demo is broken."
$describeAfterJson = gcloud run services describe $Service --region $Region --project $Project --format json
if ($LASTEXITCODE -ne 0) {
    Write-Error "gcloud run services describe (post-deploy) failed (exit $LASTEXITCODE)."
    exit 1
}
$serviceAfter = $describeAfterJson | ConvertFrom-Json
$LatestReady = $serviceAfter.status.latestReadyRevisionName
$trafficAfter = $serviceAfter.status.traffic | Where-Object { $_.percent -eq 100 }
$ServingRevision = if ($trafficAfter) { $trafficAfter.revisionName } else { $null }

Write-Host "latestReadyRevisionName: $LatestReady"
Write-Host "Revision at 100% traffic: $ServingRevision"

if ($LatestReady -ne $ServingRevision) {
    Write-Error "Traffic did not move to the latest revision (latestReady=$LatestReady, serving=$ServingRevision). Deploy is NOT live."
    exit 1
}

# Live probe hits the actual serving URL, not localhost -- proof must come from what
# users would get. Cold start on this service (min-instances=0, 4Gi memory) can exceed
# 60s, hence the generous timeout; a short timeout here would false-negative a good
# deploy that's merely cold.
$ServiceUrl = $serviceAfter.status.url
$ProbeUrl = "$ServiceUrl/api/brand"
Write-Host "Probing $ProbeUrl (allowing up to 300s for a cold start)..."
try {
    $probeResponse = Invoke-RestMethod -Uri $ProbeUrl -Method Get -TimeoutSec 300
} catch {
    Write-Error "Live probe to $ProbeUrl failed: $_"
    exit 1
}
Write-Host "Live probe OK. display_name: $($probeResponse.display_name)"

Write-Host ""
Write-Host "==================== Deploy summary ===================="
Write-Host "Image:            $Image"
Write-Host "New revision:     $LatestReady"
Write-Host "Rollback anchor:  $RollbackAnchor"
Write-Host ""
Write-Host "Rollback command (run if this deploy needs to be reverted):"
Write-Host "  gcloud run services update-traffic $Service --to-revisions ${RollbackAnchor}=100 --region $Region --project $Project"
Write-Host "==========================================================="
