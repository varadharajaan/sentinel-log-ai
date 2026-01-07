# Sentinel Log AI

<div align="center">

[![CI](https://github.com/varadharajaan/sentinel-log-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/varadharajaan/sentinel-log-ai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Go 1.22+](https://img.shields.io/badge/go-1.22+-00ADD8.svg)](https://golang.org/dl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI-powered log intelligence engine for on-call engineers.**

*Crafted by [Varad](https://github.com/varadharajaan)*

[Features](#features) • [Quick Start](#installation) • [Architecture](#architecture) • [Demo](#demo) • [Documentation](docs/) • [Wiki](../../wiki)

</div>

---

Sentinel Log AI automatically groups similar log patterns, detects novel/unseen errors, and provides LLM-powered explanations with suggested next steps — all running locally on your machine.

## Demo

**Why ML instead of Regex?** Run the interactive demo to see the difference:

```bash
cd demo
python demo_ml_vs_regex.py
```

This demo shows:
- **75 real production logs** (Kubernetes, PostgreSQL, Kafka, Redis, OAuth, etc.)
- **4 known attacks** that regex catches (SQL injection, XSS, path traversal)
- **17 novel attacks** that regex misses but ML detects:
  - Supply chain attacks (malicious npm packages)
  - SSRF via PDF generator (cloud metadata theft)
  - Container escape attempts
  - Privilege escalation via service account impersonation
  - DNS data exfiltration (base64 encoded)

**Results:**
| Detection Method | Known Attacks | Novel Attacks |
|-----------------|---------------|---------------|
| Regex | 4/4 (100%) | 0/17 (0%) |
| ML Novelty | 4/4 (100%) | 14/17 (82%) |

> **Key insight**: Regex detects what you KNOW to look for. ML detects what you DON'T KNOW to look for.

See [docs/demo.md](docs/demo.md) for the full walkthrough.

## Features

- **Pattern Clustering**: Automatically groups similar log messages using ML embeddings and HDBSCAN
- **Novelty Detection**: Identifies unseen error patterns that don't match historical clusters
- **LLM Explanations**: Generates root cause analysis with confidence scores via Ollama
- **High Performance**: Go agent handles high-throughput log ingestion (1GB+ without OOM)
- **Local-First**: Everything runs on your laptop — no cloud dependencies
- **CLI & API**: Rich CLI for interactive use, gRPC API for integrations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Sentinel Log AI                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────┐         gRPC          ┌────────────────┐  │
│   │   Go Agent      │◄─────────────────────►│  Python ML     │  │
│   │                 │                       │  Engine        │  │
│   │  • Log Ingestion│                       │                │  │
│   │  • File Tailing │                       │  • Embeddings  │  │
│   │  • Journald     │                       │  • FAISS Store │  │
│   │  • Streaming    │                       │  • Clustering  │  │
│   │  • CLI          │                       │  • Novelty     │  │
│   │                 │                       │  • LLM/Ollama  │  │
│   └────────┬────────┘                       └───────┬────────┘  │
│            │                                        │           │
│            ▼                                        ▼           │
│   ┌─────────────────┐                      ┌────────────────┐   │
│   │  Log Sources    │                      │  Vector Store  │   │
│   │  • Files        │                      │  (FAISS)       │   │
│   │  • Journald     │                      └────────────────┘   │
│   │  • Stdin        │                                           │
│   └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why Go + Python?

| Component | Language | Rationale |
|-----------|----------|-----------|
| **Agent** | Go | Single binary, low memory, excellent concurrency for streaming |
| **ML Engine** | Python | Rich ML ecosystem (sentence-transformers, FAISS, HDBSCAN) |

Communication via **gRPC** provides strongly-typed, efficient, streaming-capable IPC.

### Ingestion Pipeline (M1)

The ingestion pipeline provides high-performance log processing:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────┐
│ Log Sources │────►│ Batch        │────►│ Preprocessing │────►│ ML Engine  │
│ File/Stdin/ │     │ Processor    │     │ Pipeline      │     │ (gRPC)     │
│ Directory   │     │ (Go)         │     │ (Python)      │     │            │
└─────────────┘     └──────────────┘     └───────────────┘     └────────────┘
                           │                     │
                    • Size-based flush    • ID Assignment
                    • Time-based flush    • Timestamp parsing
                    • Back-pressure       • Format detection
                    • Metrics tracking    • Normalization
```

**Key Components:**
- **Batch Processor**: Aggregates log records to reduce gRPC overhead (configurable size/time triggers)
- **gRPC Client**: Retry logic with exponential backoff, connection pooling, health checks
- **Preprocessing Pipeline**: Modular stages (parsing → normalization → filtering)
- **Multi-format Parsers**: JSON, Syslog, Nginx, Python traceback auto-detection

### Embeddings & Vector Store (M2)

Semantic embeddings enable intelligent log similarity and search:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────┐
│ Normalized  │────►│ Embedding    │────►│ FAISS Vector  │────►│ Similarity │
│ Log Records │     │ Service      │     │ Store         │     │ Search     │
└─────────────┘     └──────────────┘     └───────────────┘     └────────────┘
                           │                     │
                    • SentenceTransformer  • Flat/IVF/HNSW
                    • 384-dim vectors      • Persistence
                    • LRU caching          • Batch operations
                    • Mock for testing     • Metadata tracking
```

**Key Components:**
- **EmbeddingService**: Sentence-transformers with LRU cache for performance
- **VectorStore**: FAISS-based storage with multiple index strategies
- **Strategy Pattern**: Pluggable embedding providers and index types

### Clustering & Patterns (M3)

HDBSCAN clustering discovers log patterns automatically:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────┐
│ Embeddings  │────►│ HDBSCAN      │────►│ Cluster       │────►│ Pattern    │
│ Array       │     │ Algorithm    │     │ Labels        │     │ Summaries  │
└─────────────┘     └──────────────┘     └───────────────┘     └────────────┘
                           │                     │                    │
                    • Density-based        • Noise filtering   • Representative
                    • No k required        • Unique clusters     samples
                    • Handles noise        • Centroid calc     • Common level
                    • Auto-tuned                               • Time ranges
```

**Key Components:**
- **ClusteringService**: High-level API for clustering operations
- **HDBSCANAlgorithm**: Production-grade density-based clustering
- **ClusterSummary**: Rich metadata with representative samples, common levels, time ranges
- **Strategy Pattern**: Pluggable clustering algorithms (HDBSCAN, K-Means, DBSCAN)

### Novelty Detection (M4)

k-NN density-based novelty detection identifies unusual log patterns:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────┐
│ Reference   │────►│ k-NN Density │────►│ Baseline      │     │            │
│ Embeddings  │     │ Computation  │     │ Distribution  │     │            │
└─────────────┘     └──────────────┘     └───────────────┘     │            │
                                                ▼               │  Novelty   │
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     │  Scores    │
│ New         │────►│ Cross k-NN   │────►│ Density       │────►│  (0-1)     │
│ Embeddings  │     │ Distances    │     │ Scoring       │     │            │
└─────────────┘     └──────────────┘     └───────────────┘     └────────────┘
                           │                     │                    │
                    • Distance to ref      • Z-score norm      • Threshold
                    • k neighbors          • Sigmoid transform   classification
                    • Efficient search     • [0,1] range       • Explanations
```

**Key Components:**
- **NoveltyService**: High-level API for novelty detection operations
- **KNNNoveltyDetector**: k-nearest neighbors density-based scoring
- **NoveltyScore**: Per-sample scores with explanations
- **Strategy Pattern**: Pluggable detection algorithms (k-NN, LOF, Isolation Forest)

### LLM Explanation (M5)

Local LLM-powered explanations for log patterns via Ollama:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────┐
│ Cluster/    │────►│ Prompt       │────►│ Ollama LLM    │────►│ Explanation│
│ Novelty     │     │ Builder      │     │ (llama3.2)    │     │ Response   │
└─────────────┘     └──────────────┘     └───────────────┘     └────────────┘
                           │                     │                    │
                    • Template-based       • Local inference   • Root cause
                    • Structured input     • Retry logic       • Suggested actions
                    • Context building     • Timeout handling  • Severity
                    • JSON format          • Token tracking    • Confidence
```

**Key Components:**
- **LLMService**: High-level API for generating explanations
- **OllamaProvider**: Ollama REST API integration with retry logic
- **MockLLMProvider**: Deterministic mock for testing
- **Prompt Templates**: Specialized templates for cluster, novelty, error analysis
- **Strategy Pattern**: Pluggable LLM providers (Ollama, OpenAI, Mock)

**Explanation Types:**
| Type | Input | Output |
|------|-------|--------|
| `CLUSTER` | ClusterSummary | Root cause, actions, severity |
| `NOVELTY` | NoveltyScore | Why novel, potential impact |
| `ERROR_ANALYSIS` | LogRecord | Error diagnosis, fix suggestions |
| `SUMMARY` | Aggregated data | Executive summary |

### CLI & UX (M6)

Rich command-line interface with themeable output and report generation:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────┐
│ Analysis    │────►│ Console      │────►│ Theme/Format  │────►│ Terminal   │
│ Results     │     │ (Facade)     │     │ (Strategy)    │     │ Output     │
└─────────────┘     └──────────────┘     └───────────────┘     └────────────┘
                           │                     │                    │
                    • Formatters           • DARK/LIGHT       • Rich tables
                    • Progress tracker     • COLORBLIND       • JSON export
                    • Profiler             • MINIMAL/NONE     • Reports
```

**Key Components:**
- **Console**: Unified interface for all CLI output operations
- **Theme System**: 5 themes including colorblind-friendly option
- **Formatters**: Strategy pattern for JSON, Table, Cluster, Novelty, Explanation
- **Progress Tracking**: Spinners, progress bars, ETA calculation
- **Report Generation**: Markdown and HTML export with embedded styles
- **Profiler**: Timing instrumentation with hierarchical breakdown
- **Config Commands**: Generate, validate, load, show configuration

**Output Formats:**
| Format | Description |
|--------|-------------|
| `TEXT` | Human-readable colored output |
| `JSON` | Machine-readable JSON |
| `TABLE` | Rich formatted tables |
| `COMPACT` | Minimal one-line output |

**Reports:**
| Format | Features |
|--------|----------|
| Markdown | TOC, executive summary, cluster details, code blocks |
| HTML | Embedded CSS, responsive layout, cluster cards |

## Installation

### Prerequisites

- Go 1.22+
- Python 3.10+
- [Ollama](https://ollama.ai/) (for LLM explanations)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/sentinel-log-ai/sentinel-log-ai.git
cd sentinel-log-ai

# Install everything
make install

# Build the Go agent
make build-go

# Start the ML server (in one terminal)
make run-ml

# Use the CLI (in another terminal)
./bin/sentinel-log-ai ingest /var/log/syslog
./bin/sentinel-log-ai analyze --last 1h
./bin/sentinel-log-ai novel --follow
```

### Development Setup

```bash
# Install all dependencies + pre-commit hooks
make install

# Run linters
make lint

# Run tests
make test

# Format code
make fmt
```

## Usage

### Ingest Logs

```bash
# Batch mode - read entire file
sentinel-log-ai ingest /var/log/syslog

# Tail mode - follow new lines
sentinel-log-ai ingest /var/log/app.log --tail

# From stdin
cat /var/log/nginx/error.log | sentinel-log-ai ingest -

# Ingest entire directory (recursive)
sentinel-log-ai ingest /var/log/ --pattern "*.log"

# With custom batch settings
sentinel-log-ai ingest /var/log/app.log --batch-size 50 --flush-timeout 2s

# Dry run (no ML processing, just parse and display)
sentinel-log-ai ingest /var/log/syslog --dry-run

# Connect to specific ML server
sentinel-log-ai ingest /var/log/syslog --ml-server localhost:50051
```

### Analyze Patterns

```bash
# Cluster recent logs
sentinel-log-ai analyze --last 1h

# Show top clusters
sentinel-log-ai analyze --top 10
```

### Detect Novel Errors

```bash
# Check for novel patterns
sentinel-log-ai novel

# Continuous monitoring
sentinel-log-ai novel --follow --threshold 0.7
```

### Get LLM Explanations

```bash
# Explain a specific cluster
sentinel-log-ai explain cluster-abc123

# Explain the most novel pattern
sentinel-log-ai explain --novel
```

## Configuration

Create `sentinel-log-ai.yaml`:

```yaml
# Embedding configuration
embedding:
  model_name: all-MiniLM-L6-v2
  batch_size: 32
  device: cpu  # or cuda, mps

# Clustering configuration
clustering:
  min_cluster_size: 5
  min_samples: 3

# Novelty detection
novelty:
  threshold: 0.7
  k_neighbors: 5

# LLM configuration
llm:
  provider: ollama
  model: llama3.2
  base_url: http://localhost:11434

# Server configuration
server:
  host: 0.0.0.0
  port: 50051

# Logging
logging:
  level: INFO
  format: json  # or plain
```

Environment variables override config file (prefix: `SENTINEL_ML_`):

```bash
export SENTINEL_ML_LLM__MODEL=mistral
export SENTINEL_ML_EMBEDDING__DEVICE=cuda
```

## Data Model

### LogRecord

The canonical log record used across all components:

```python
class LogRecord:
    id: str | None           # Unique identifier
    message: str             # Main log message
    normalized: str | None   # Masked/normalized for ML
    level: str | None        # INFO, WARN, ERROR, etc.
    source: str              # File path, journald unit
    timestamp: datetime      # When the log occurred
    raw: str                 # Original unparsed line
    attrs: dict              # Additional structured data
```

### Normalization

Logs are normalized before embedding to improve clustering:

| Pattern | Replacement |
|---------|-------------|
| `192.168.1.100` | `<ip>` |
| `550e8400-e29b-...` | `<uuid>` |
| `2024-01-15T10:30:00Z` | `<ts>` |
| `1234567890` | `<num>` |
| `0x1a2b3c4d5e6f` | `<hex>` |

## 🎨 Design Patterns & SOLID Principles

The codebase follows enterprise-grade patterns for maintainability and extensibility:

### Go Agent
| Pattern | Usage |
|---------|-------|
| **Strategy** | Configurable flush strategies (size-based, time-based) in batch processor |
| **Observer** | Batch lifecycle hooks for monitoring and metrics |
| **Builder** | Fluent configuration for batch processor and gRPC client |
| **Factory** | Parser creation based on log format auto-detection |

### Python ML Engine
| Pattern | Usage |
|---------|-------|
| **Pipeline** | Sequential preprocessing stages (parse → normalize → filter) |
| **Strategy** | Pluggable normalization, embedding providers, clustering algorithms |
| **Factory** | Parser, normalizer, embedding service, clustering service creation |
| **Facade** | Simple gRPC interface to complex ML subsystems |
| **Template Method** | Clustering workflow with customizable steps |
| **Observer** | Statistics tracking for embeddings, vector store, clustering |

### SOLID Principles
- **Single Responsibility**: Each component handles one concern (e.g., batch processor only handles batching)
- **Open/Closed**: Extensible via interfaces (`BatchHandler`, `ProcessingStage`) without modifying core code
- **Interface Segregation**: Minimal interfaces focused on specific capabilities
- **Dependency Inversion**: Components depend on abstractions, not concrete implementations

## 🧪 Testing

```bash
# Run all tests
make test

# Go tests only
make test-go

# Python tests only
make test-python

# With coverage
pytest tests/python -v --cov=sentinel_ml
```

## 📁 Project Structure

```
sentinel-log-ai/
├── cmd/
│   └── agent/              # Go CLI entry point
│       ├── main.go
│       └── cmd/            # Cobra commands
├── internal/               # Go internal packages
│   ├── batch/              # High-performance batch processor
│   ├── errors/             # Structured error handling
│   ├── grpcclient/         # gRPC client with retry logic
│   ├── ingestion/          # Log source adapters
│   ├── logging/            # Structured JSONL logging
│   ├── models/             # Go data models
│   └── parser/             # Multi-format log parsers
├── pkg/
│   └── mlpb/               # Generated Go protobuf
├── python/
│   └── sentinel_ml/        # Python ML engine
│       ├── config.py       # Pydantic configuration
│       ├── exceptions.py   # Structured error hierarchy
│       ├── logging.py      # Structured logging (structlog)
│       ├── models.py       # Pydantic models
│       ├── normalization.py # Log normalization & masking
│       ├── parser.py       # Python log parsers
│       ├── preprocessing.py # Preprocessing pipeline
│       ├── server.py       # gRPC server
│       ├── embedding.py    # Sentence transformers (M2)
│       ├── vectorstore.py  # FAISS vector store (M2)
│       ├── clustering.py   # HDBSCAN clustering (M3)
│       ├── novelty.py      # Novelty detection (M4)
│       └── llm.py          # Ollama integration (M5)
├── proto/
│   └── ml/v1/              # Protobuf definitions
├── tests/
│   ├── go/                 # Go tests
│   └── python/             # Python tests
├── samples/                # Sample log files
├── .github/workflows/      # CI/CD
├── go.mod                  # Go module
├── pyproject.toml          # Python project
├── Makefile                # Build automation
└── README.md
```

## Roadmap

- [x] **M0**: Project scaffolding, dev tooling
- [x] **M1**: Ingestion & preprocessing pipeline
- [x] **M2**: Embeddings & FAISS vector store
- [x] **M3**: HDBSCAN clustering & pattern summaries
- [x] **M4**: Novelty detection (k-NN density-based)
- [x] **M5**: LLM explanation with confidence (Ollama integration)
- [x] **M6**: CLI polish & rich output
- [x] **M7**: Performance benchmarks & documentation
- [x] **M8**: Storage & retention policies
- [x] **M9**: Alerting & integrations
- [x] **M10**: Evaluation & quality framework
- [ ] **M11**: Packaging & release (in progress)
- [ ] **M12**: Security & privacy

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run linters and tests (`make lint && make test`)
5. Commit with conventional commits (`git commit -m 'feat: add amazing feature'`)
6. Push and open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [sentence-transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [HDBSCAN](https://hdbscan.readthedocs.io/) for clustering
- [Ollama](https://ollama.ai/) for local LLM inference

---

<div align="center">

**[⬆ Back to Top](#sentinel-log-ai)**

Made with ❤️ by [Varad](https://github.com/varadharajaan) • [Report Bug](../../issues) • [Request Feature](../../issues)

</div>
