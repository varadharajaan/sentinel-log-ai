# Architecture Overview

## System Design Philosophy

Sentinel Log AI follows a **polyglot microkernel architecture** that separates concerns optimally:

| Concern | Language | Rationale |
|---------|----------|-----------|
| Log Ingestion | Go | Single binary, low memory, excellent I/O concurrency |
| ML Processing | Python | Rich ecosystem (transformers, FAISS, scikit-learn) |
| Communication | gRPC/Protobuf | Type-safe, efficient, streaming-capable IPC |

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SENTINEL LOG AI                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        PRESENTATION LAYER                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  Go CLI     │  │  Python CLI │  │  gRPC API   │  │  Reports   │  │   │
│  │  │  (Cobra)    │  │  (Rich)     │  │  (Protobuf) │  │  (MD/HTML) │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                        APPLICATION LAYER                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  Ingestion  │  │  Embedding  │  │  Clustering │  │  Novelty   │  │   │
│  │  │  Service    │  │  Service    │  │  Service    │  │  Service   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  │  ┌─────────────┐  ┌─────────────┐                                    │   │
│  │  │  LLM        │  │  Profiling  │                                    │   │
│  │  │  Service    │  │  Service    │                                    │   │
│  │  └─────────────┘  └─────────────┘                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                         DOMAIN LAYER                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  LogRecord  │  │  Cluster    │  │  Novelty    │  │ Explanation│  │   │
│  │  │  Entity     │  │  Entity     │  │  Score      │  │  Entity    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                      INFRASTRUCTURE LAYER                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  File       │  │  FAISS      │  │  Ollama     │  │  Logging   │  │   │
│  │  │  System     │  │  VectorDB   │  │  Client     │  │  (JSONL)   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Interaction

```
┌──────────────┐        gRPC           ┌──────────────────────────────────┐
│              │ ◄───────────────────► │                                  │
│   Go Agent   │    Streaming API      │       Python ML Engine           │
│              │                       │                                  │
│  • Ingest    │                       │  ┌────────────────────────────┐  │
│  • Parse     │   PreprocessRequest   │  │     EmbeddingService       │  │
│  • Normalize │ ────────────────────► │  │  • SentenceTransformer     │  │
│  • Batch     │                       │  │  • LRU Cache               │  │
│              │   PreprocessResponse  │  └────────────┬───────────────┘  │
│              │ ◄──────────────────── │               │                  │
│              │                       │               ▼                  │
│              │    ClusterRequest     │  ┌────────────────────────────┐  │
│              │ ────────────────────► │  │     VectorStore (FAISS)    │  │
│              │                       │  │  • Flat/IVF/HNSW Index     │  │
│              │   ClusterResponse     │  │  • Similarity Search       │  │
│              │ ◄──────────────────── │  └────────────┬───────────────┘  │
│              │                       │               │                  │
│              │    NoveltyRequest     │               ▼                  │
│              │ ────────────────────► │  ┌────────────────────────────┐  │
│              │                       │  │    ClusteringService       │  │
│              │   NoveltyResponse     │  │  • HDBSCAN Algorithm       │  │
│              │ ◄──────────────────── │  │  • Pattern Summaries       │  │
│              │                       │  └────────────┬───────────────┘  │
│              │   ExplainRequest      │               │                  │
│              │ ────────────────────► │               ▼                  │
│              │                       │  ┌────────────────────────────┐  │
│              │   ExplainResponse     │  │    NoveltyService          │  │
│              │ ◄──────────────────── │  │  • k-NN Density Scoring    │  │
└──────────────┘                       │  │  • Threshold Classification│  │
                                       │  └────────────┬───────────────┘  │
                                       │               │                  │
                                       │               ▼                  │
                                       │  ┌────────────────────────────┐  │
                                       │  │       LLMService           │  │
                                       │  │  • Ollama Provider         │  │
                                       │  │  • OpenAI Provider         │  │
                                       │  │  • Prompt Templates        │  │
                                       │  └────────────────────────────┘  │
                                       │                                  │
                                       └──────────────────────────────────┘
```

## Data Flow Pipeline

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Logs   │────►│  Parse  │────►│ Normalize────►│  Embed  │────►│  Store  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
                                                                     │
    ┌────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Cluster │────►│ Novelty │────►│ Explain │────►│ Report  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

## Technology Stack

### Go Agent (Performance Layer)
| Component | Technology | Purpose |
|-----------|-----------|---------|
| CLI | Cobra | Command-line interface |
| Logging | Zap + Lumberjack | Structured JSONL with rotation |
| Concurrency | Goroutines + Channels | Parallel processing |
| gRPC | google.golang.org/grpc | ML engine communication |

### Python ML Engine (Intelligence Layer)
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Config | Pydantic | Type-safe configuration |
| Logging | Structlog | Structured JSON logging |
| Embeddings | sentence-transformers | Semantic vectors |
| Vector Store | FAISS | Fast similarity search |
| Clustering | HDBSCAN | Density-based clustering |
| LLM | Ollama/OpenAI | Natural language explanations |
| gRPC | grpcio | Server implementation |

## Scalability Considerations

### Horizontal Scaling
```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Go Agent 1    │ │   Go Agent 2    │ │   Go Agent 3    │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │  ML Engine Pool │
                    │  (Stateless)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Shared Vector  │
                    │  Store (Redis/  │
                    │  Distributed)   │
                    └─────────────────┘
```

### Memory Management
- **Go Agent**: Streaming processing, bounded channels, no full-file loading
- **Python Engine**: Batch processing, LRU caches, memory-mapped FAISS indices

---

*See also: [[Go Agent]], [[Python ML Engine]], [[Design Patterns]]*
