# bootstrap-sentinel-log-ai.ps1
# Creates/boots repo + labels + milestones + issues for sentinel-log-ai
# Requirements: git, GitHub CLI (gh) logged in (gh auth login)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------
# Config
# ---------------------------
$REPO_NAME   = "sentinel-log-ai"
$VISIBILITY  = "public"   # or "private"
$DEFAULT_BRANCH = "main"

# ---------------------------
# Helpers
# ---------------------------
function Ensure-GhAuth {
  $authResult = gh auth status 2>&1
  if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI is not authenticated. Run: gh auth login"
  }
}

function Get-GhUserLogin {
  return (gh api user --jq ".login").Trim()
}

function Ensure-GitRepo {
  if (-not (Test-Path ".git")) {
    git init 2>&1 | Out-Null
    Write-Host "Initialized new git repository"
  } else {
    Write-Host "Git repository already exists"
  }
}

function Ensure-RepoExists {
  param([string]$Owner, [string]$RepoName, [string]$Visibility)

  $full = "$Owner/$RepoName"
  $exists = $false
  
  # Temporarily allow errors to check repo existence
  $oldErrorAction = $ErrorActionPreference
  $ErrorActionPreference = "SilentlyContinue"
  $null = gh repo view $full --json name 2>&1
  if ($LASTEXITCODE -eq 0) {
    $exists = $true
  }
  $ErrorActionPreference = $oldErrorAction

  if (-not $exists) {
    Write-Host "Creating GitHub repo $full ..."
    # Create empty repo on GitHub
    gh repo create $full --$Visibility 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
      throw "Failed to create repository $full"
    }
  } else {
    Write-Host "Repo already exists: $full"
  }

  return $full
}

function Ensure-OriginRemote {
  param([string]$FullRepo)
  
  # Temporarily allow errors to check remote existence
  $oldErrorAction = $ErrorActionPreference
  $ErrorActionPreference = "SilentlyContinue"
  $remoteUrl = git remote get-url origin 2>&1
  $remoteExists = ($LASTEXITCODE -eq 0)
  $ErrorActionPreference = $oldErrorAction

  if (-not $remoteExists) {
    $https = (gh repo view $FullRepo --json url --jq ".url").Trim()
    # Convert https URL to git URL
    $gitUrl = "$https.git"
    git remote add origin $gitUrl
    Write-Host "Added remote origin: $gitUrl"
  } else {
    Write-Host "Remote origin already exists: $remoteUrl"
  }
}

function Ensure-InitialCommitAndPush {
  param([string]$FullRepo, [string]$Branch)

  # Create a README if none exists (so first push works)
  if (-not (Test-Path "README.md")) {
    "# $($FullRepo.Split('/')[1])`n`nAI-powered log intelligence engine." | Out-File -Encoding utf8 "README.md"
  }

  # Temporarily allow errors for git checks
  $oldErrorAction = $ErrorActionPreference
  $ErrorActionPreference = "SilentlyContinue"
  
  # If no commits yet, commit
  $hasCommit = $false
  $null = git rev-parse HEAD 2>&1
  if ($LASTEXITCODE -eq 0) {
    $hasCommit = $true
  }

  if (-not $hasCommit) {
    git add -A
    $null = git commit -m "chore: initial commit" 2>&1
    if ($LASTEXITCODE -ne 0) {
      Write-Host "Warning: Could not create initial commit (maybe nothing to commit)"
    } else {
      Write-Host "Created initial commit"
    }
  } else {
    Write-Host "Repository already has commits"
  }

  # Ensure branch name
  $null = git branch -M $Branch 2>&1

  # Push (handle case where remote may already have commits)
  $null = git push -u origin $Branch 2>&1
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Initial push failed. Trying with --force..."
    $null = git push -u origin $Branch --force 2>&1
    if ($LASTEXITCODE -ne 0) {
      Write-Host "Warning: Force push also failed. You may need to push manually."
    }
  } else {
    Write-Host "Pushed to origin/$Branch"
  }
  
  $ErrorActionPreference = $oldErrorAction
}

function Upsert-Label {
  param(
    [string]$FullRepo,
    [string]$Name,
    [string]$Color,
    [string]$Description = ""
  )
  
  # Temporarily allow errors for label creation
  $oldErrorAction = $ErrorActionPreference
  $ErrorActionPreference = "SilentlyContinue"
  
  # --force updates if exists
  if ($Description -and $Description.Length -gt 0) {
    $null = gh label create $Name --repo $FullRepo --color $Color --description $Description --force 2>&1
  } else {
    $null = gh label create $Name --repo $FullRepo --color $Color --force 2>&1
  }
  
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Failed to create/update label: $Name"
  } else {
    Write-Host "  Label: $Name"
  }
  
  $ErrorActionPreference = $oldErrorAction
}

function Create-MilestoneIfMissing {
  param([string]$FullRepo, [string]$Title)

  $owner, $repo = $FullRepo.Split("/")
  
  # Temporarily allow errors for API calls
  $oldErrorAction = $ErrorActionPreference
  $ErrorActionPreference = "SilentlyContinue"
  
  $milestonesJson = gh api "repos/$owner/$repo/milestones?state=all&per_page=100" 2>&1
  $milestones = @()
  if ($LASTEXITCODE -eq 0 -and $milestonesJson) {
    $milestones = $milestonesJson | ConvertFrom-Json
  } else {
    Write-Host "Warning: Could not fetch milestones, creating new one..."
  }
  
  $existing = $milestones | Where-Object { $_.title -eq $Title } | Select-Object -First 1
  if ($existing) {
    $ErrorActionPreference = $oldErrorAction
    return [int]$existing.number
  }

  $result = gh api -X POST "repos/$owner/$repo/milestones" -f title="$Title" --jq ".number" 2>&1
  $ErrorActionPreference = $oldErrorAction
  
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to create milestone: $Title - $result"
  }
  Write-Host "Created milestone: $Title"
  return [int]$result
}

function Create-Issue {
  param(
    [string]$FullRepo,
    [string]$Title,
    [string[]]$Labels,
    [int]$MilestoneNumber,
    [string]$Body
  )

  # Temporarily allow errors for issue creation
  $oldErrorAction = $ErrorActionPreference
  $ErrorActionPreference = "SilentlyContinue"
  
  $labelsCsv = $Labels -join ","
  $null = gh issue create --repo $FullRepo --title $Title --label $labelsCsv --milestone $MilestoneNumber --body $Body 2>&1
  
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Failed to create issue: $Title"
  } else {
    Write-Host "  Issue: $Title"
  }
  
  $ErrorActionPreference = $oldErrorAction
}

# ---------------------------
# Main
# ---------------------------
Ensure-GhAuth
$owner = Get-GhUserLogin

Write-Host "Bootstrapping for user: $owner"
Ensure-GitRepo

$fullRepo = Ensure-RepoExists -Owner $owner -RepoName $REPO_NAME -Visibility $VISIBILITY
Ensure-OriginRemote -FullRepo $fullRepo
Ensure-InitialCommitAndPush -FullRepo $fullRepo -Branch $DEFAULT_BRANCH

Write-Host "Creating/Updating labels..."
$labels = @(
  @{ name="epic";        color="5319e7"; desc="Epic tracking issue" },
  @{ name="core";        color="b60205"; desc="Core functionality" },
  @{ name="backend";     color="0e8a16"; desc="Backend / pipelines" },
  @{ name="ml";          color="1d76db"; desc="ML / embeddings / clustering" },
  @{ name="llm";         color="cfd3d7"; desc="LLM / prompting / explainability" },
  @{ name="infra";       color="fbca04"; desc="Infra-ish / persistence / packaging" },
  @{ name="performance"; color="d93f0b"; desc="Perf, memory, benchmarking" },
  @{ name="docs";        color="0075ca"; desc="Documentation" },
  @{ name="ux";          color="a2eeef"; desc="CLI / UX" },
  @{ name="stretch";     color="f9d0c4"; desc="Stretch goals" },
  @{ name="good-first-issue"; color="7057ff"; desc="Good first issue" }
)

foreach ($l in $labels) {
  Upsert-Label -FullRepo $fullRepo -Name $l.name -Color $l.color -Description $l.desc
}

Write-Host "Creating milestones (via GitHub API)..."
$milestoneTitles = @(
  "M1: Ingestion & Preprocessing",
  "M2: Embeddings & Vector Store",
  "M3: Clustering & Patterns",
  "M4: Novelty Detection",
  "M5: LLM Explanation",
  "M6: CLI & UX",
  "M7: Performance & Docs"
)

$milestones = @{}
foreach ($t in $milestoneTitles) {
  $milestones[$t] = Create-MilestoneIfMissing -FullRepo $fullRepo -Title $t
}

Write-Host "Creating issues..."

# ---- EPIC tracker
Create-Issue -FullRepo $fullRepo `
  -Title "EPIC: Intelligent Log Intelligence Engine (AI On-call Assistant)" `
  -Labels @("epic","core","docs") `
  -MilestoneNumber $milestones["M1: Ingestion & Preprocessing"] `
  -Body @"
## Goal
Build a local-first AI system that:
- groups similar log patterns
- detects novel/unseen errors
- explains clusters in plain English (LLM)
- is memory-efficient and usable via CLI

## Success Criteria
- Ingest 1GB+ logs without OOM (streaming)
- Similarity search < 100ms on typical laptop
- Novelty detection flags unseen patterns
- Explanations include suggested next steps + confidence

## Milestones
- M1: ingestion + preprocessing
- M2: embeddings + FAISS store
- M3: clustering summaries
- M4: novelty detection
- M5: LLM explanation + confidence
- M6: CLI polish
- M7: benchmarks + docs
"@

# ---------------------------
# M1 Issues
# ---------------------------
Create-Issue $fullRepo "Define core LogRecord data model" @("backend","core","good-first-issue") $milestones["M1: Ingestion & Preprocessing"] @"
### Description
Define a canonical LogRecord schema used across ingestion, ML, storage, and explanation.

### Requirements
- Fields: timestamp (optional), level (optional), message, source, raw, attrs (dict)
- JSON serializable
- Minimal and stable (avoid frequent schema changes)

### Acceptance Criteria
- Implement as dataclass OR pydantic model
- Unit tests cover serialization + missing fields
- Document in README under "Data Model"
"@

Create-Issue $fullRepo "Implement file-based log ingestion (batch + tail)" @("backend","core") $milestones["M1: Ingestion & Preprocessing"] @"
### Description
Implement reading logs from local files.

### Features
- Batch mode: read entire file line-by-line
- Tail mode: follow appended lines (like tail -f)
- Support plain text + JSON lines

### Acceptance Criteria
- Works with >1GB file without loading into memory
- Malformed JSON lines are handled (skip with warning)
- Emits LogRecord objects
"@

Create-Issue $fullRepo "Add log source adapters interface (File/Journald/Stdout)" @("backend") $milestones["M1: Ingestion & Preprocessing"] @"
### Description
Create a pluggable interface for log sources.

### Acceptance Criteria
- Abstract base class / protocol for sources
- FileSource implemented now
- JournaldSource stubbed behind optional dependency
"@

Create-Issue $fullRepo "Implement normalization + masking pipeline" @("ml","backend","core") $milestones["M1: Ingestion & Preprocessing"] @"
### Description
Normalize logs to reduce noise and improve clustering.

### Must mask
- IPv4/IPv6 -> <ip>
- UUID -> <uuid>
- integers -> <num>
- hex tokens -> <hex>
- timestamps -> <ts>

### Acceptance Criteria
- Deterministic output
- Configurable regex rules (yaml/toml/json)
- Unit tests for each masking type
"@

Create-Issue $fullRepo "Add structured parsing for common formats (nginx, systemd, python tracebacks)" @("backend","ml") $milestones["M1: Ingestion & Preprocessing"] @"
### Description
Add parsers that extract level/timestamp/message when possible.

### Acceptance Criteria
- Parser selection based on regex match
- If parsing fails, fallback to raw message only
- Include tests with sample log lines
"@

# ---------------------------
# M2 Issues
# ---------------------------
Create-Issue $fullRepo "Integrate sentence-transformers embeddings (CPU-first)" @("ml","core") $milestones["M2: Embeddings & Vector Store"] @"
### Description
Embed normalized log messages using sentence-transformers.

### Acceptance Criteria
- Batched embedding to reduce overhead
- Model selectable via config
- Outputs float32 numpy arrays
- Basic perf note in README (logs/sec on your machine)
"@

Create-Issue $fullRepo "Implement FAISS vector store (persist + reload)" @("ml","infra","core") $milestones["M2: Embeddings & Vector Store"] @"
### Description
Create persistent vector store:
- FAISS index on disk
- metadata store mapping vector_id -> LogRecord metadata

### Acceptance Criteria
- Create index, add vectors, search kNN
- Persist and reload without re-embedding
- Search returns IDs + distances + associated metadata
"@

Create-Issue $fullRepo "Add embedding cache (hash-based)" @("performance","ml") $milestones["M2: Embeddings & Vector Store"] @"
### Description
Avoid re-embedding identical normalized lines by caching.

### Acceptance Criteria
- Hash key = normalized message string
- Cache supports disk-backed mode
- Cache hit rate logged
"@

Create-Issue $fullRepo "Build similarity search API (top-k similar logs + cluster hint)" @("backend","ml") $milestones["M2: Embeddings & Vector Store"] @"
### Description
Expose a query function:
- input: new log line
- output: top-k similar historical logs + distances

### Acceptance Criteria
- Returns consistent results for same input
- Designed to be reused by clustering + novelty detection
"@

# ---------------------------
# M3 Issues
# ---------------------------
Create-Issue $fullRepo "Cluster embeddings with HDBSCAN and produce cluster summaries" @("ml","core") $milestones["M3: Clustering & Patterns"] @"
### Description
Cluster embedded logs into patterns.

### Output per cluster
- cluster_id
- size
- representative message
- top tokens/keywords (optional)

### Acceptance Criteria
- Noise points handled
- Summary artifacts stored (json)
"@

Create-Issue $fullRepo "Choose representative log per cluster (medoid/center)" @("ml") $milestones["M3: Clustering & Patterns"] @"
### Description
Pick a representative sample per cluster.

### Acceptance Criteria
- Use medoid (min avg distance) or nearest to centroid
- Stored in cluster summary metadata
"@

Create-Issue $fullRepo "Stabilize cluster IDs across re-runs" @("ml","enhancement") $milestones["M3: Clustering & Patterns"] @"
### Description
Make cluster IDs stable so dashboards/history make sense.

### Ideas
- Hash of representative normalized text
- Or persistent mapping between runs

### Acceptance Criteria
- Cluster IDs change minimally between runs on same dataset
"@

# ---------------------------
# M4 Issues
# ---------------------------
Create-Issue $fullRepo "Implement novelty scoring (is this error new?)" @("ml","core") $milestones["M4: Novelty Detection"] @"
### Description
Detect novel/unseen patterns for incoming logs.

### Approach
- Use kNN distance + density heuristics
- Threshold configurable

### Acceptance Criteria
- Output: is_novel, novelty_score, closest_cluster_id (optional)
- Provide default thresholds that work okay on sample logs
"@

Create-Issue $fullRepo "Novelty evaluation harness (synthetic unseen + regression tests)" @("ml","testing") $milestones["M4: Novelty Detection"] @"
### Description
Build tests to validate novelty detection.

### Acceptance Criteria
- Generate synthetic variants
- Track precision/recall over time (lightweight)
"@

# ---------------------------
# M5 Issues
# ---------------------------
Create-Issue $fullRepo "Prompt templates for root-cause explanation" @("llm","core") $milestones["M5: LLM Explanation"] @"
### Description
Create deterministic prompt templates for explaining clusters.

### Inputs
- representative log
- top similar logs
- cluster keywords
- novelty score

### Output
- probable root cause
- what to check next
- suggested remediation

### Acceptance Criteria
- Prompt versioning
- Output JSON schema option (for structured rendering)
"@

Create-Issue $fullRepo "Integrate local LLM via Ollama (pluggable)" @("llm","infra") $milestones["M5: LLM Explanation"] @"
### Description
Use Ollama as local LLM runtime.

### Acceptance Criteria
- Configurable model name
- Timeout + retry handling
- Can disable LLM and still run clustering/novelty
"@

Create-Issue $fullRepo "Explanation confidence scoring + guardrails" @("llm","ml","core") $milestones["M5: LLM Explanation"] @"
### Description
Avoid hallucinations and provide trust signals.

### Signals
- cluster cohesion/density
- novelty score
- similarity distance to known incidents

### Acceptance Criteria
- Confidence: Low/Med/High + numeric score
- If low confidence, explain why and suggest human checks
"@

# ---------------------------
# M6 Issues
# ---------------------------
Create-Issue $fullRepo "CLI: ingest/analyze/novel/explain commands" @("ux","backend","core") $milestones["M6: CLI & UX"] @"
### Description
Implement a usable CLI.

### Commands
- log-ai ingest <path>
- log-ai analyze --last 1h
- log-ai novel --follow
- log-ai explain --cluster <id>

### Acceptance Criteria
- Help text
- Config file support
- Exit codes for automation usage
"@

Create-Issue $fullRepo "CLI output: rich tables and summaries" @("ux") $milestones["M6: CLI & UX"] @"
### Description
Make output readable with table rendering.

### Acceptance Criteria
- Cluster summary table
- Novelty alerts formatted cleanly
"@

# ---------------------------
# M7 Issues
# ---------------------------
Create-Issue $fullRepo "Benchmarks: ingestion rate, embedding throughput, search latency" @("performance","core") $milestones["M7: Performance & Docs"] @"
### Description
Provide basic benchmarks and record results.

### Acceptance Criteria
- Script to run benchmarks
- Document results in README
"@

Create-Issue $fullRepo "Docs: architecture, dataflow, and failure modes" @("docs","core") $milestones["M7: Performance & Docs"] @"
### Description
Write high-quality docs.

### Must include
- architecture diagram (ascii is fine)
- design decisions
- limitations + failure modes
"@

Create-Issue $fullRepo "Demo dataset + walkthrough" @("docs","demo") $milestones["M7: Performance & Docs"] @"
### Description
Provide sample logs and step-by-step demo.

### Acceptance Criteria
- /samples folder
- README walkthrough commands
"@

# ---------------------------
# Stretch
# ---------------------------
Create-Issue $fullRepo "Stretch: Real-time streaming pipeline with backpressure" @("stretch","performance") $milestones["M7: Performance & Docs"] @"
### Description
Handle high-throughput streams safely.

### Acceptance Criteria
- Bounded queues
- Drop/compact strategy documented
"@

Create-Issue $fullRepo "Stretch: TUI dashboard (curses) for clusters + novel alerts" @("stretch","ux") $milestones["M7: Performance & Docs"] @"
### Description
Interactive terminal UI for navigating clusters and explanations.
"@

Write-Host "DONE. Repo bootstrapped with labels, milestones, and issues:"
Write-Host "  $fullRepo"
Write-Host "Open in browser:"
gh repo view $fullRepo --web
