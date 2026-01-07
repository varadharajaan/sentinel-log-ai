# Quick Start Guide

Get up and running with Sentinel Log AI in 5 minutes.

## Prerequisites

Before you begin, ensure you have:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Go | 1.22+ | `go version` |
| Python | 3.10+ | `python --version` |
| Ollama | Latest | `ollama --version` |
| Git | Any | `git --version` |

## Step 1: Clone the Repository

```bash
git clone https://github.com/varadharajaan/sentinel-log-ai.git
cd sentinel-log-ai
```

## Step 2: Install Dependencies

### Using Make (Recommended)
```bash
make install
```

### Manual Installation

**Go Dependencies:**
```bash
go mod download
```

**Python Dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Step 3: Setup Ollama (for LLM Explanations)

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the recommended model
ollama pull llama3.2

# Verify it's running
ollama list
```

## Step 4: Build the Go Agent

```bash
make build
# or
go build -o bin/sentinel-agent ./cmd/agent
```

## Step 5: Start the ML Engine

```bash
# In one terminal
make run-server
# or
python -m sentinel_ml.server
```

You should see:
```
INFO     Starting Sentinel ML gRPC server on 0.0.0.0:50051
INFO     EmbeddingService initialized with all-MiniLM-L6-v2
INFO     VectorStore initialized with flat index
INFO     ClusteringService ready
INFO     NoveltyService ready
INFO     LLMService connected to Ollama
```

## Step 6: Analyze Your First Log File

```bash
# In another terminal
./bin/sentinel-agent ingest --file samples/app.log

# Or pipe logs directly
cat /var/log/syslog | ./bin/sentinel-agent ingest --stdin
```

## Step 7: View Results

The agent will output:
- **Cluster Summaries**: Groups of similar log patterns
- **Novel Patterns**: Logs that don't match any known cluster
- **Explanations**: LLM-powered root cause analysis

Example output:
```
┌──────────────────────────────────────────────────────────────────┐
│                        ANALYSIS RESULTS                          │
├──────────────────────────────────────────────────────────────────┤
│ Total Logs: 1,247                                                │
│ Clusters: 8                                                      │
│ Novel Patterns: 3                                                │
│ Processing Time: 4.2s                                            │
└──────────────────────────────────────────────────────────────────┘

📊 CLUSTER #1 (423 logs, cohesion: 0.92)
├── Level: INFO
├── Representative: "User login successful from {IP}"
└── Explanation: Routine authentication events...

⚠️ NOVEL PATTERN (score: 0.89)
├── Message: "Database connection pool exhausted after 30s timeout"
├── Severity: HIGH
└── Suggested Action: Check database connection limits...
```

## Common Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make build` | Build Go binary |
| `make run-server` | Start Python ML server |
| `make test` | Run all tests |
| `make lint` | Run linters |
| `make help` | Show all available commands |

## Configuration

Create a `config.yaml` for custom settings:

```yaml
server:
  host: "0.0.0.0"
  port: 50051

embedding:
  model_name: "all-MiniLM-L6-v2"
  batch_size: 32

clustering:
  min_cluster_size: 5
  metric: "euclidean"

novelty:
  threshold: 0.7
  k_neighbors: 5

llm:
  provider: "ollama"
  model: "llama3.2"
```

## Next Steps

- [[Installation]] - Detailed installation guide
- [[Configuration]] - All configuration options
- [[CLI Reference|CLI-UX]] - Complete CLI documentation
- [[Architecture Overview|Architecture-Overview]] - System design

---

*Crafted with ❤️ by [Varad](https://github.com/varadharajaan)*
