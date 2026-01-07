# Sentinel Log AI

[![CI](https://github.com/varadharajaan/sentinel-log-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/varadharajaan/sentinel-log-ai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Go 1.22+](https://img.shields.io/badge/go-1.22+-00ADD8.svg)](https://golang.org/dl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI-powered log intelligence engine for on-call engineers.**

Sentinel Log AI automatically groups similar log patterns, detects novel/unseen errors, and provides LLM-powered explanations with suggested next steps — all running locally on your machine.

## 🎯 Features

- **Pattern Clustering**: Automatically groups similar log messages using ML embeddings and HDBSCAN
- **Novelty Detection**: Identifies unseen error patterns that don't match historical clusters
- **LLM Explanations**: Generates root cause analysis with confidence scores via Ollama
- **High Performance**: Go agent handles high-throughput log ingestion (1GB+ without OOM)
- **Local-First**: Everything runs on your laptop — no cloud dependencies
- **CLI & API**: Rich CLI for interactive use, gRPC API for integrations

## 🏗️ Architecture

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

## 📦 Installation

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

## 🚀 Usage

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

## ⚙️ Configuration

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

## 📊 Data Model

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
| **Strategy** | Pluggable normalization strategies |
| **Factory** | Parser and normalizer creation |
| **Facade** | Simple gRPC interface to complex ML subsystems |

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

## 🗺️ Roadmap

- [x] **M0**: Project scaffolding, dev tooling
- [x] **M1**: Ingestion & preprocessing pipeline
- [x] **M2**: Embeddings & FAISS vector store
- [ ] **M3**: HDBSCAN clustering & pattern summaries
- [ ] **M4**: Novelty detection
- [ ] **M5**: LLM explanation with confidence
- [ ] **M6**: CLI polish & rich output
- [ ] **M7**: Performance benchmarks & docs
- [ ] **M8**: Storage & retention policies
- [ ] **M9**: Alerting integrations
- [ ] **M10**: Evaluation framework
- [ ] **M11**: Packaging & release
- [ ] **M12**: Security & privacy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run linters and tests (`make lint && make test`)
5. Commit with conventional commits (`git commit -m 'feat: add amazing feature'`)
6. Push and open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [sentence-transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [HDBSCAN](https://hdbscan.readthedocs.io/) for clustering
- [Ollama](https://ollama.ai/) for local LLM inference
