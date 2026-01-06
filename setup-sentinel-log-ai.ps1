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
    [string]$MilestoneTitle,
    [string]$Body
  )

  # Temporarily allow errors for issue creation
  $oldErrorAction = $ErrorActionPreference
  $ErrorActionPreference = "SilentlyContinue"
  
  # Check if issue with this title already exists
  $existingIssues = gh issue list --repo $FullRepo --search "in:title $Title" --json title 2>&1 | ConvertFrom-Json
  if ($LASTEXITCODE -eq 0 -and $existingIssues) {
    $exactMatch = $existingIssues | Where-Object { $_.title -eq $Title }
    if ($exactMatch) {
      Write-Host "  Skipped (exists): $Title"
      $ErrorActionPreference = $oldErrorAction
      return
    }
  }
  
  $labelsCsv = $Labels -join ","
  $null = gh issue create --repo $FullRepo --title $Title --label $labelsCsv --milestone $MilestoneTitle --body $Body 2>&1
  
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
  @{ name="good-first-issue"; color="7057ff"; desc="Good first issue" },
  @{ name="devx";        color="c5def5"; desc="Developer experience / tooling" },
  @{ name="security";    color="d73a4a"; desc="Security & privacy" },
  @{ name="storage";     color="bfdadc"; desc="Storage & retention" },
  @{ name="alerting";    color="ff9f1c"; desc="Alerting & integrations" },
  @{ name="evaluation";  color="6f42c1"; desc="Evaluation & quality" },
  @{ name="packaging";   color="0d98ba"; desc="Packaging & release" },
  @{ name="testing";     color="bfd4f2"; desc="Testing" },
  @{ name="enhancement"; color="84b6eb"; desc="Enhancement" },
  @{ name="demo";        color="fef2c0"; desc="Demo & samples" }
)

foreach ($l in $labels) {
  Upsert-Label -FullRepo $fullRepo -Name $l.name -Color $l.color -Description $l.desc
}

Write-Host "Creating milestones (via GitHub API)..."
$milestoneTitles = @(
  "M0: Project Scaffolding & DevX",
  "M1: Ingestion & Preprocessing",
  "M2: Embeddings & Vector Store",
  "M3: Clustering & Patterns",
  "M4: Novelty Detection",
  "M5: LLM Explanation",
  "M6: CLI & UX",
  "M7: Performance & Docs",
  "M8: Storage & Retention",
  "M9: Alerting & Integrations",
  "M10: Evaluation & Quality",
  "M11: Packaging & Release",
  "M12: Security & Privacy"
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
  -MilestoneTitle "M1: Ingestion & Preprocessing" `
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
Create-Issue $fullRepo "Define core LogRecord data model" @("backend","core","good-first-issue") "M1: Ingestion & Preprocessing" @"
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

Create-Issue $fullRepo "Implement file-based log ingestion (batch + tail)" @("backend","core") "M1: Ingestion & Preprocessing" @"
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

Create-Issue $fullRepo "Add log source adapters interface (File/Journald/Stdout)" @("backend") "M1: Ingestion & Preprocessing" @"
### Description
Create a pluggable interface for log sources.

### Acceptance Criteria
- Abstract base class / protocol for sources
- FileSource implemented now
- JournaldSource stubbed behind optional dependency
"@

Create-Issue $fullRepo "Implement normalization + masking pipeline" @("ml","backend","core") "M1: Ingestion & Preprocessing" @"
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

Create-Issue $fullRepo "Add structured parsing for common formats (nginx, systemd, python tracebacks)" @("backend","ml") "M1: Ingestion & Preprocessing" @"
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
Create-Issue $fullRepo "Integrate sentence-transformers embeddings (CPU-first)" @("ml","core") "M2: Embeddings & Vector Store" @"
### Description
Embed normalized log messages using sentence-transformers.

### Acceptance Criteria
- Batched embedding to reduce overhead
- Model selectable via config
- Outputs float32 numpy arrays
- Basic perf note in README (logs/sec on your machine)
"@

Create-Issue $fullRepo "Implement FAISS vector store (persist + reload)" @("ml","infra","core") "M2: Embeddings & Vector Store" @"
### Description
Create persistent vector store:
- FAISS index on disk
- metadata store mapping vector_id -> LogRecord metadata

### Acceptance Criteria
- Create index, add vectors, search kNN
- Persist and reload without re-embedding
- Search returns IDs + distances + associated metadata
"@

Create-Issue $fullRepo "Add embedding cache (hash-based)" @("performance","ml") "M2: Embeddings & Vector Store" @"
### Description
Avoid re-embedding identical normalized lines by caching.

### Acceptance Criteria
- Hash key = normalized message string
- Cache supports disk-backed mode
- Cache hit rate logged
"@

Create-Issue $fullRepo "Build similarity search API (top-k similar logs + cluster hint)" @("backend","ml") "M2: Embeddings & Vector Store" @"
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
Create-Issue $fullRepo "Cluster embeddings with HDBSCAN and produce cluster summaries" @("ml","core") "M3: Clustering & Patterns" @"
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

Create-Issue $fullRepo "Choose representative log per cluster (medoid/center)" @("ml") "M3: Clustering & Patterns" @"
### Description
Pick a representative sample per cluster.

### Acceptance Criteria
- Use medoid (min avg distance) or nearest to centroid
- Stored in cluster summary metadata
"@

Create-Issue $fullRepo "Stabilize cluster IDs across re-runs" @("ml","enhancement") "M3: Clustering & Patterns" @"
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
Create-Issue $fullRepo "Implement novelty scoring (is this error new?)" @("ml","core") "M4: Novelty Detection" @"
### Description
Detect novel/unseen patterns for incoming logs.

### Approach
- Use kNN distance + density heuristics
- Threshold configurable

### Acceptance Criteria
- Output: is_novel, novelty_score, closest_cluster_id (optional)
- Provide default thresholds that work okay on sample logs
"@

Create-Issue $fullRepo "Novelty evaluation harness (synthetic unseen + regression tests)" @("ml","testing") "M4: Novelty Detection" @"
### Description
Build tests to validate novelty detection.

### Acceptance Criteria
- Generate synthetic variants
- Track precision/recall over time (lightweight)
"@

# ---------------------------
# M5 Issues
# ---------------------------
Create-Issue $fullRepo "Prompt templates for root-cause explanation" @("llm","core") "M5: LLM Explanation" @"
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

Create-Issue $fullRepo "Integrate local LLM via Ollama (pluggable)" @("llm","infra") "M5: LLM Explanation" @"
### Description
Use Ollama as local LLM runtime.

### Acceptance Criteria
- Configurable model name
- Timeout + retry handling
- Can disable LLM and still run clustering/novelty
"@

Create-Issue $fullRepo "Explanation confidence scoring + guardrails" @("llm","ml","core") "M5: LLM Explanation" @"
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
Create-Issue $fullRepo "CLI: ingest/analyze/novel/explain commands" @("ux","backend","core") "M6: CLI & UX" @"
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

Create-Issue $fullRepo "CLI output: rich tables and summaries" @("ux") "M6: CLI & UX" @"
### Description
Make output readable with table rendering.

### Acceptance Criteria
- Cluster summary table
- Novelty alerts formatted cleanly
"@

# ---------------------------
# M7 Issues
# ---------------------------
Create-Issue $fullRepo "Benchmarks: ingestion rate, embedding throughput, search latency" @("performance","core") "M7: Performance & Docs" @"
### Description
Provide basic benchmarks and record results.

### Acceptance Criteria
- Script to run benchmarks
- Document results in README
"@

Create-Issue $fullRepo "Docs: architecture, dataflow, and failure modes" @("docs","core") "M7: Performance & Docs" @"
### Description
Write high-quality docs.

### Must include
- architecture diagram (ascii is fine)
- design decisions
- limitations + failure modes
"@

Create-Issue $fullRepo "Demo dataset + walkthrough" @("docs","demo") "M7: Performance & Docs" @"
### Description
Provide sample logs and step-by-step demo.

### Acceptance Criteria
- /samples folder
- README walkthrough commands
"@

# ---------------------------
# Stretch
# ---------------------------
Create-Issue $fullRepo "Stretch: Real-time streaming pipeline with backpressure" @("stretch","performance") "M7: Performance & Docs" @"
### Description
Handle high-throughput streams safely.

### Acceptance Criteria
- Bounded queues
- Drop/compact strategy documented
"@

Create-Issue $fullRepo "Stretch: TUI dashboard (curses) for clusters + novel alerts" @("stretch","ux") "M7: Performance & Docs" @"
### Description
Interactive terminal UI for navigating clusters and explanations.
"@

# ---------------------------
# M0: Project Scaffolding & DevX
# ---------------------------
Create-Issue $fullRepo "Add Python project scaffold (pyproject.toml, src/ layout, ruff/black, mypy)" @("devx","infra","good-first-issue") "M0: Project Scaffolding & DevX" @"
### Description
Set up modern Python project structure.

### Requirements
- pyproject.toml with dependencies
- src/ layout
- Linting with ruff/black
- Type checking with mypy

### Acceptance Criteria
- Project installs with pip install -e .
- Linting passes
- mypy runs without critical errors
"@

Create-Issue $fullRepo "Add pre-commit hooks (ruff, formatting, trailing whitespace, end-of-file)" @("devx","infra") "M0: Project Scaffolding & DevX" @"
### Description
Set up pre-commit for code quality.

### Acceptance Criteria
- .pre-commit-config.yaml configured
- Hooks: ruff, formatting, trailing whitespace, end-of-file
- Documented in README
"@

Create-Issue $fullRepo "Add CI workflow (lint + unit tests)" @("devx","infra") "M0: Project Scaffolding & DevX" @"
### Description
Set up GitHub Actions CI.

### Acceptance Criteria
- Workflow runs on push/PR
- Runs linting and unit tests
- Reports status on PRs
"@

Create-Issue $fullRepo "Add basic logging + structured logging toggle (json/plain)" @("devx","backend") "M0: Project Scaffolding & DevX" @"
### Description
Add configurable logging for the application.

### Acceptance Criteria
- Structured JSON logging option
- Plain text logging option
- Configurable via env or config file
"@

Create-Issue $fullRepo "Add config system (yaml/toml + env overrides)" @("devx","backend","core") "M0: Project Scaffolding & DevX" @"
### Description
Implement configuration management.

### Acceptance Criteria
- Support yaml/toml config files
- Environment variable overrides
- Sensible defaults
"@

Create-Issue $fullRepo "Add make-like task runner docs (justfile / invoke / poe)" @("devx","docs") "M0: Project Scaffolding & DevX" @"
### Description
Document how to run common tasks.

### Acceptance Criteria
- justfile or equivalent
- Tasks: lint, test, build, run
- Documented in README
"@

# ---------------------------
# M1 Additional Issues
# ---------------------------
Create-Issue $fullRepo "Add stdin ingestion mode (pipe support)" @("backend") "M1: Ingestion & Preprocessing" @"
### Description
Support piping logs via stdin.

### Acceptance Criteria
- cat logs.txt | log-ai ingest - works
- Handles streaming input
"@

Create-Issue $fullRepo "Add directory ingestion (glob + rotate-aware reading)" @("backend") "M1: Ingestion & Preprocessing" @"
### Description
Ingest entire directories of log files.

### Acceptance Criteria
- Support glob patterns
- Handle rotated logs (.1, .2, .gz)
- Skip already-processed files option
"@

Create-Issue $fullRepo "Add multiline log support (stack traces, exceptions)" @("backend","ml") "M1: Ingestion & Preprocessing" @"
### Description
Handle multiline log entries like stack traces.

### Acceptance Criteria
- Configurable multiline patterns
- Stack traces grouped as single log entry
- Works with Python, Java tracebacks
"@

Create-Issue $fullRepo "Add sampling + rate limiting for high-volume streams" @("backend","performance") "M1: Ingestion & Preprocessing" @"
### Description
Handle high-volume log streams gracefully.

### Acceptance Criteria
- Configurable sampling rate
- Rate limiting with backpressure
- Metrics on dropped logs
"@

Create-Issue $fullRepo "Add deduplication window (suppress repeats for N seconds)" @("backend","performance") "M1: Ingestion & Preprocessing" @"
### Description
Suppress duplicate log lines within a time window.

### Acceptance Criteria
- Configurable window size
- Count suppressed duplicates
- Report suppression stats
"@

Create-Issue $fullRepo "Add timezone handling + timestamp parser library integration" @("backend") "M1: Ingestion & Preprocessing" @"
### Description
Robust timestamp parsing across timezones.

### Acceptance Criteria
- Parse common timestamp formats
- Handle timezone conversion
- Default timezone configurable
"@

Create-Issue $fullRepo "Add per-source tagging (service/app/env) via config" @("backend","infra") "M1: Ingestion & Preprocessing" @"
### Description
Tag logs with source metadata.

### Acceptance Criteria
- Configurable source tags
- Tags stored with log metadata
- Filterable by tag
"@

# ---------------------------
# M2 Additional Issues
# ---------------------------
Create-Issue $fullRepo "Add embedding model abstraction (swap ST, OpenAI, local LLM embeddings later)" @("ml","infra") "M2: Embeddings & Vector Store" @"
### Description
Abstract embedding model for flexibility.

### Acceptance Criteria
- Interface for embedding providers
- Support sentence-transformers
- Easy to add OpenAI, local LLM later
"@

Create-Issue $fullRepo "Add background batching queue for embeddings (throughput optimization)" @("ml","performance") "M2: Embeddings & Vector Store" @"
### Description
Optimize embedding throughput with batching.

### Acceptance Criteria
- Async queue for embeddings
- Configurable batch size
- Throughput metrics
"@

Create-Issue $fullRepo "Add vector store abstraction (FAISS now, sqlite-vss/chroma later)" @("ml","infra") "M2: Embeddings & Vector Store" @"
### Description
Abstract vector store for flexibility.

### Acceptance Criteria
- Interface for vector stores
- FAISS implementation
- Easy to add sqlite-vss, chroma later
"@

Create-Issue $fullRepo "Add metadata store implementation (SQLite) with migrations" @("infra","backend") "M2: Embeddings & Vector Store" @"
### Description
SQLite-based metadata storage.

### Acceptance Criteria
- SQLite for log metadata
- Schema migrations support
- Indexed for common queries
"@

Create-Issue $fullRepo "Add re-index command (rebuild index from metadata)" @("backend","infra") "M2: Embeddings & Vector Store" @"
### Description
Command to rebuild vector index.

### Acceptance Criteria
- log-ai reindex command
- Progress reporting
- Handles large datasets
"@

Create-Issue $fullRepo "Add compact/optimize index command" @("performance","infra") "M2: Embeddings & Vector Store" @"
### Description
Optimize index for better performance.

### Acceptance Criteria
- log-ai optimize command
- Reduces index size
- Improves query speed
"@

Create-Issue $fullRepo "Add shard support for very large datasets" @("performance","infra") "M2: Embeddings & Vector Store" @"
### Description
Support sharding for large-scale deployments.

### Acceptance Criteria
- Configurable shard size
- Automatic shard management
- Query across shards
"@

# ---------------------------
# M3 Additional Issues
# ---------------------------
Create-Issue $fullRepo "Add incremental clustering strategy (online-ish updates)" @("ml","performance") "M3: Clustering & Patterns" @"
### Description
Update clusters incrementally without full recompute.

### Acceptance Criteria
- Assign new logs to existing clusters
- Periodic full re-cluster option
- Performance within SLA
"@

Create-Issue $fullRepo "Add cluster drift detection (cluster changed over time)" @("ml") "M3: Clustering & Patterns" @"
### Description
Detect when cluster patterns change.

### Acceptance Criteria
- Track cluster evolution
- Alert on significant drift
- Historical comparison
"@

Create-Issue $fullRepo "Add cluster labeling (keywords + optional LLM title)" @("ml","llm") "M3: Clustering & Patterns" @"
### Description
Auto-generate cluster labels.

### Acceptance Criteria
- Keyword extraction
- Optional LLM-generated titles
- Human-editable labels
"@

Create-Issue $fullRepo "Add top clusters by volume report" @("ux","ml") "M3: Clustering & Patterns" @"
### Description
Report showing highest-volume clusters.

### Acceptance Criteria
- Sorted by log count
- Filterable by time range
- Export option
"@

Create-Issue $fullRepo "Add top clusters by novelty report" @("ux","ml") "M3: Clustering & Patterns" @"
### Description
Report showing most novel clusters.

### Acceptance Criteria
- Sorted by novelty score
- Shows first/last seen
- Drill-down to samples
"@

Create-Issue $fullRepo "Add template miner option (Drain3) for log pattern extraction" @("ml","backend") "M3: Clustering & Patterns" @"
### Description
Add Drain3 algorithm for pattern mining.

### Acceptance Criteria
- Optional Drain3 integration
- Extract log templates
- Compare with embedding clusters
"@

# ---------------------------
# M4 Additional Issues
# ---------------------------
Create-Issue $fullRepo "Add per-source novelty thresholds (service-specific)" @("ml","backend") "M4: Novelty Detection" @"
### Description
Different novelty thresholds per source.

### Acceptance Criteria
- Per-service configuration
- Reasonable defaults
- Easy to tune
"@

Create-Issue $fullRepo "Add novelty timeline (when pattern first seen, last seen)" @("ml","ux") "M4: Novelty Detection" @"
### Description
Track novelty over time.

### Acceptance Criteria
- First seen timestamp
- Last seen timestamp
- Frequency tracking
"@

Create-Issue $fullRepo "Add novelty suppression (acknowledge pattern to stop alerting)" @("backend","ux") "M4: Novelty Detection" @"
### Description
Allow users to acknowledge patterns.

### Acceptance Criteria
- Acknowledge command
- Suppresses future alerts
- Undo option
"@

Create-Issue $fullRepo "Add false-positive tracking + threshold tuning guide" @("ml","docs") "M4: Novelty Detection" @"
### Description
Help users tune novelty detection.

### Acceptance Criteria
- Track false positive rate
- Document tuning process
- Suggest optimal thresholds
"@

Create-Issue $fullRepo "Add novelty detection using density (LOF) as alternate method" @("ml") "M4: Novelty Detection" @"
### Description
Add Local Outlier Factor method.

### Acceptance Criteria
- LOF implementation
- Configurable method selection
- Compare with distance-based
"@

# ---------------------------
# M5 Additional Issues
# ---------------------------
Create-Issue $fullRepo "Add strict JSON output mode for explanations (schema validated)" @("llm","backend") "M5: LLM Explanation" @"
### Description
Ensure LLM output is valid JSON.

### Acceptance Criteria
- JSON schema validation
- Retry on invalid output
- Fallback handling
"@

Create-Issue $fullRepo "Add citation mode (show top similar logs used as evidence)" @("llm","ux") "M5: LLM Explanation" @"
### Description
Show evidence for explanations.

### Acceptance Criteria
- Display similar logs used
- Link to source logs
- Confidence per citation
"@

Create-Issue $fullRepo "Add prompt injection defenses for log content" @("llm","security") "M5: LLM Explanation" @"
### Description
Prevent prompt injection attacks.

### Acceptance Criteria
- Sanitize log content
- Detect injection attempts
- Safe prompt construction
"@

Create-Issue $fullRepo "Add retry/backoff and timeout controls" @("llm","backend") "M5: LLM Explanation" @"
### Description
Robust LLM API handling.

### Acceptance Criteria
- Configurable retries
- Exponential backoff
- Timeout handling
"@

Create-Issue $fullRepo "Add explanation caching (cluster_id -> explanation)" @("llm","performance") "M5: LLM Explanation" @"
### Description
Cache explanations to reduce LLM calls.

### Acceptance Criteria
- Cache by cluster ID
- Invalidation strategy
- Cache hit metrics
"@

Create-Issue $fullRepo "Add next-steps checklist generator (commands to run, files to check)" @("llm","ux") "M5: LLM Explanation" @"
### Description
Generate actionable next steps.

### Acceptance Criteria
- Specific commands
- Files to check
- Priority ordering
"@

# ---------------------------
# M6 Additional Issues
# ---------------------------
Create-Issue $fullRepo "Add report command to export markdown summary" @("ux","docs") "M6: CLI & UX" @"
### Description
Export analysis as markdown.

### Acceptance Criteria
- log-ai report command
- Includes clusters, novelty, stats
- Configurable sections
"@

Create-Issue $fullRepo "Add web export (static HTML report) command" @("ux","docs") "M6: CLI & UX" @"
### Description
Export as standalone HTML.

### Acceptance Criteria
- Single HTML file output
- Interactive elements (JS)
- No server required
"@

Create-Issue $fullRepo "Add interactive cluster drill-down (select cluster -> view samples)" @("ux") "M6: CLI & UX" @"
### Description
Interactive cluster exploration.

### Acceptance Criteria
- Select cluster from list
- View sample logs
- Navigate between clusters
"@

Create-Issue $fullRepo "Add config init command (generate sample config)" @("ux","devx") "M6: CLI & UX" @"
### Description
Generate sample configuration.

### Acceptance Criteria
- log-ai init command
- Creates sample config file
- Documented options
"@

Create-Issue $fullRepo "Add profile flag (print timing breakdown)" @("ux","performance") "M6: CLI & UX" @"
### Description
Performance profiling flag.

### Acceptance Criteria
- --profile flag
- Timing per stage
- Memory usage
"@

# ---------------------------
# M7 Additional Issues
# ---------------------------
Create-Issue $fullRepo "Add memory profiling script (peak RSS)" @("performance","testing") "M7: Performance & Docs" @"
### Description
Script to measure memory usage.

### Acceptance Criteria
- Peak memory tracking
- Per-stage breakdown
- CI integration
"@

Create-Issue $fullRepo "Add dataset scale test (10k/100k/1M lines)" @("performance","testing") "M7: Performance & Docs" @"
### Description
Benchmark at different scales.

### Acceptance Criteria
- Test datasets included
- Automated benchmarks
- Results in docs
"@

Create-Issue $fullRepo "Add troubleshooting guide (common failures)" @("docs") "M7: Performance & Docs" @"
### Description
Document common issues and solutions.

### Acceptance Criteria
- FAQ format
- Error message index
- Solutions for each
"@

Create-Issue $fullRepo "Add architecture diagram (Mermaid)" @("docs") "M7: Performance & Docs" @"
### Description
Visual architecture documentation.

### Acceptance Criteria
- Mermaid diagram
- Data flow shown
- Component relationships
"@

# ---------------------------
# M8: Storage & Retention
# ---------------------------
Create-Issue $fullRepo "Add retention policy (delete old raw logs/embeddings by age/size)" @("storage","backend") "M8: Storage & Retention" @"
### Description
Automatic data cleanup.

### Acceptance Criteria
- Age-based retention
- Size-based retention
- Configurable policies
"@

Create-Issue $fullRepo "Add snapshotting (daily index snapshots)" @("storage","infra") "M8: Storage & Retention" @"
### Description
Periodic index snapshots.

### Acceptance Criteria
- Configurable schedule
- Snapshot restore command
- Space management
"@

Create-Issue $fullRepo "Add import/export (portable bundle: sqlite + faiss + config)" @("storage","ux") "M8: Storage & Retention" @"
### Description
Portable data bundles.

### Acceptance Criteria
- Export to single archive
- Import command
- Version compatibility
"@

Create-Issue $fullRepo "Add data versioning for index formats" @("storage","infra") "M8: Storage & Retention" @"
### Description
Handle index format changes.

### Acceptance Criteria
- Version tracking
- Migration support
- Backward compatibility
"@

# ---------------------------
# M9: Alerting & Integrations
# ---------------------------
Create-Issue $fullRepo "Add Slack webhook notifier for novel events" @("alerting","backend") "M9: Alerting & Integrations" @"
### Description
Send alerts to Slack.

### Acceptance Criteria
- Webhook configuration
- Formatted messages
- Rate limiting
"@

Create-Issue $fullRepo "Add Email notifier (SMTP) for novel events" @("alerting","backend") "M9: Alerting & Integrations" @"
### Description
Send alerts via email.

### Acceptance Criteria
- SMTP configuration
- HTML/plain text
- Digest option
"@

Create-Issue $fullRepo "Add watch mode daemon that monitors and alerts" @("alerting","backend") "M9: Alerting & Integrations" @"
### Description
Background monitoring daemon.

### Acceptance Criteria
- log-ai watch command
- Configurable check interval
- Multiple notifiers
"@

Create-Issue $fullRepo "Add integration: write novel events to a local file or syslog" @("alerting","backend") "M9: Alerting & Integrations" @"
### Description
Local notification options.

### Acceptance Criteria
- File output option
- Syslog output option
- Configurable format
"@

Create-Issue $fullRepo "Add GitHub issue auto-creation output (generate issue markdown for copy-paste)" @("alerting","ux") "M9: Alerting & Integrations" @"
### Description
Generate GitHub issue content.

### Acceptance Criteria
- Formatted markdown
- Includes context
- Copy-paste ready
"@

# ---------------------------
# M10: Evaluation & Quality
# ---------------------------
Create-Issue $fullRepo "Add clustering quality metrics (silhouette, DB index where applicable)" @("evaluation","ml") "M10: Evaluation & Quality" @"
### Description
Measure clustering quality.

### Acceptance Criteria
- Silhouette score
- Davies-Bouldin index
- Trend tracking
"@

Create-Issue $fullRepo "Add human labeling tool (mark clusters as same/different)" @("evaluation","ux") "M10: Evaluation & Quality" @"
### Description
Tool for human evaluation.

### Acceptance Criteria
- Simple labeling UI
- Export labels
- Track inter-rater agreement
"@

Create-Issue $fullRepo "Add golden dataset + regression tests" @("evaluation","testing") "M10: Evaluation & Quality" @"
### Description
Standard test dataset.

### Acceptance Criteria
- Curated log samples
- Expected clusters
- CI regression tests
"@

Create-Issue $fullRepo "Add evaluation report generation" @("evaluation","docs") "M10: Evaluation & Quality" @"
### Description
Generate quality reports.

### Acceptance Criteria
- Automated report
- Metrics summary
- Trend analysis
"@

Create-Issue $fullRepo "Add ablation tests (masking on/off, model A vs B)" @("evaluation","testing") "M10: Evaluation & Quality" @"
### Description
Test component contributions.

### Acceptance Criteria
- Compare configurations
- Measure impact
- Document findings
"@

# ---------------------------
# M11: Packaging & Release
# ---------------------------
Create-Issue $fullRepo "Add Dockerfile (optional local container run)" @("packaging","infra") "M11: Packaging & Release" @"
### Description
Docker support.

### Acceptance Criteria
- Multi-stage build
- Minimal image size
- Documented usage
"@

Create-Issue $fullRepo "Add Windows packaging (PyInstaller) for single exe CLI" @("packaging","infra") "M11: Packaging & Release" @"
### Description
Windows executable.

### Acceptance Criteria
- Single .exe file
- No Python required
- Tested on Windows
"@

Create-Issue $fullRepo "Add versioning + changelog automation" @("packaging","devx") "M11: Packaging & Release" @"
### Description
Automated versioning.

### Acceptance Criteria
- Semantic versioning
- Auto-generated changelog
- Git tags
"@

Create-Issue $fullRepo "Add release workflow (build artifacts on tags)" @("packaging","infra") "M11: Packaging & Release" @"
### Description
Automated releases.

### Acceptance Criteria
- GitHub Actions workflow
- Build on tag push
- Upload artifacts
"@

Create-Issue $fullRepo "Add install docs (pipx)" @("packaging","docs") "M11: Packaging & Release" @"
### Description
Document installation methods.

### Acceptance Criteria
- pipx install instructions
- pip install instructions
- Platform-specific notes
"@

# ---------------------------
# M12: Security & Privacy
# ---------------------------
Create-Issue $fullRepo "Add PII redaction controls (emails, tokens, secrets patterns)" @("security","backend") "M12: Security & Privacy" @"
### Description
Automatic PII removal.

### Acceptance Criteria
- Email redaction
- Token/secret patterns
- Configurable rules
"@

Create-Issue $fullRepo "Add never store raw logs mode (store only normalized)" @("security","storage") "M12: Security & Privacy" @"
### Description
Privacy-preserving mode.

### Acceptance Criteria
- Raw logs discarded
- Only normalized stored
- Configurable option
"@

Create-Issue $fullRepo "Add at-rest encryption option for SQLite (document approach)" @("security","storage") "M12: Security & Privacy" @"
### Description
Encrypted storage option.

### Acceptance Criteria
- Document encryption options
- SQLCipher or alternative
- Key management guide
"@

Create-Issue $fullRepo "Add threat model doc (what can leak via embeddings)" @("security","docs") "M12: Security & Privacy" @"
### Description
Document security considerations.

### Acceptance Criteria
- Embedding leakage analysis
- Mitigation strategies
- Privacy recommendations
"@

Create-Issue $fullRepo "Add safe defaults + privacy FAQ" @("security","docs") "M12: Security & Privacy" @"
### Description
Privacy-focused documentation.

### Acceptance Criteria
- Safe default settings
- Privacy FAQ
- Compliance guidance
"@

Write-Host "DONE. Repo bootstrapped with labels, milestones, and issues:"
Write-Host "  $fullRepo"
Write-Host "Open in browser:"
gh repo view $fullRepo --web
