# System Architecture

## Overview

Sentinel Log AI is a polyglot log intelligence system that combines Go's performance for log ingestion with Python's ML/AI ecosystem for intelligent analysis.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Log Sources"
        FS[File System]
        STDIN[Stdin Stream]
        JD[Journald]
    end

    subgraph "Go Agent"
        ING[Ingestion Engine]
        PARSE[Parser Registry]
        NORM[Normalizer]
        GRPC_C[gRPC Client]
    end

    subgraph "Python ML Engine"
        GRPC_S[gRPC Server]
        EMB[Embedding Generator]
        CLUST[Clustering Engine]
        NOV[Novelty Detector]
        LLM[LLM Integration]
    end

    subgraph "Storage"
        VS[Vector Store FAISS]
        LOG[JSONL Logs]
    end

    subgraph "External"
        OLLAMA[Ollama/OpenAI]
        ATHENA[AWS Athena]
    end

    FS --> ING
    STDIN --> ING
    JD --> ING
    
    ING --> PARSE
    PARSE --> NORM
    NORM --> GRPC_C
    
    GRPC_C <--> GRPC_S
    
    GRPC_S --> EMB
    EMB --> CLUST
    EMB --> NOV
    EMB --> VS
    
    CLUST --> LLM
    NOV --> LLM
    
    LLM <--> OLLAMA
    
    ING --> LOG
    EMB --> LOG
    
    LOG --> ATHENA
```

## Component Architecture

### Go Agent Components

```mermaid
graph LR
    subgraph "cmd/agent"
        MAIN[main.go]
        CMD[cmd/root.go]
    end

    subgraph "internal/ingestion"
        SRC[Source Interface]
        FS_SRC[FileSource]
        STDIN_SRC[StdinSource]
    end

    subgraph "internal/parser"
        PREG[Parser Registry]
        JSON_P[JSON Parser]
        SYSLOG_P[Syslog Parser]
        NGINX_P[Nginx Parser]
        TB_P[Traceback Parser]
        COMMON_P[Common Parser]
    end

    subgraph "internal/models"
        LR[LogRecord]
        CS[ClusterSummary]
        EXP[Explanation]
    end

    subgraph "internal/errors"
        SE[SentinelError]
        EC[ErrorCodes]
    end

    subgraph "internal/logging"
        ZLOG[Zap Logger]
        ROLL[Rolling Handler]
    end

    MAIN --> CMD
    CMD --> SRC
    SRC --> FS_SRC
    SRC --> STDIN_SRC
    FS_SRC --> PREG
    PREG --> JSON_P
    PREG --> SYSLOG_P
    PREG --> NGINX_P
    PREG --> TB_P
    PREG --> COMMON_P
    PREG --> LR
```

### Python ML Engine Components

```mermaid
graph LR
    subgraph "sentinel_ml"
        MODELS[models.py]
        CONFIG[config.py]
        LOG[logging.py]
        NORM[normalization.py]
        PARSER[parser.py]
        SERVER[server.py]
        EXC[exceptions.py]
        PRE[preprocessing.py]
    end

    subgraph "M2 Components"
        EMB[embedding.py]
        VS[vectorstore.py]
    end

    subgraph "Future Components"
        CLUSTER[clustering.py]
        NOVELTY[novelty.py]
        LLM[llm.py]
    end

    CONFIG --> LOG
    CONFIG --> SERVER
    CONFIG --> EMB
    CONFIG --> VS
    MODELS --> PARSER
    MODELS --> EMB
    MODELS --> VS
    NORM --> PARSER
    NORM --> PRE
    EXC --> SERVER
    EXC --> EMB
    EXC --> VS
    EMB --> SERVER
    VS --> SERVER
```

## Embedding Architecture (M2)

The embedding subsystem provides semantic vector representations of log messages:

```mermaid
graph TB
    subgraph "Embedding Service"
        ES[EmbeddingService]
        CACHE[EmbeddingCache LRU]
        STATS[EmbeddingStats]
    end

    subgraph "Providers"
        EP[EmbeddingProvider ABC]
        ST[SentenceTransformerProvider]
        MOCK[MockEmbeddingProvider]
    end

    subgraph "Model"
        MODEL[all-MiniLM-L6-v2]
        TENSOR[384-dim Embeddings]
    end

    ES --> CACHE
    ES --> STATS
    ES --> EP
    EP --> ST
    EP --> MOCK
    ST --> MODEL
    MODEL --> TENSOR
```

### Embedding Provider Strategy Pattern

The embedding system uses the Strategy pattern for provider flexibility:

| Provider | Use Case |
|----------|----------|
| `SentenceTransformerProvider` | Production - Uses sentence-transformers models |
| `MockEmbeddingProvider` | Testing - Deterministic mock embeddings |

### Embedding Cache Architecture

```mermaid
flowchart LR
    INPUT[Normalized Message] --> HASH[MD5 Hash]
    HASH --> CHECK{In Cache?}
    CHECK -->|Hit| RET[Return Cached]
    CHECK -->|Miss| GEN[Generate Embedding]
    GEN --> STORE[Store in LRU Cache]
    STORE --> RET2[Return Embedding]
```

## Vector Store Architecture (M2)

The vector store provides high-performance similarity search:

```mermaid
graph TB
    subgraph "VectorStore"
        VS[VectorStore]
        META[VectorMetadata]
        STATS2[VectorStoreStats]
    end

    subgraph "Index Strategies"
        VI[VectorIndex ABC]
        FLAT[Flat Index]
        IVF[IVF Index]
        HNSW[HNSW Index]
        MOCK2[MockVectorIndex]
    end

    subgraph "FAISS"
        FAISS_FLAT[IndexFlatIP]
        FAISS_IVF[IndexIVFFlat]
        FAISS_HNSW[IndexHNSWFlat]
    end

    VS --> META
    VS --> STATS2
    VS --> VI
    VI --> FLAT
    VI --> IVF
    VI --> HNSW
    VI --> MOCK2
    FLAT --> FAISS_FLAT
    IVF --> FAISS_IVF
    HNSW --> FAISS_HNSW
```

### Index Strategy Selection

| Strategy | Best For | Trade-offs |
|----------|----------|------------|
| `Flat` | Small datasets (<10K) | Exact search, slower at scale |
| `IVF` | Medium datasets (10K-1M) | Fast approximate, requires training |
| `HNSW` | Large datasets (1M+) | Very fast, higher memory |

### Persistence Model

```mermaid
flowchart TD
    VS[VectorStore] --> SAVE[save]
    SAVE --> IDX_FILE[vectors.faiss]
    SAVE --> META_FILE[metadata.json]
    
    LOAD[load] --> IDX_FILE
    LOAD --> META_FILE
    LOAD --> VS2[Restored VectorStore]
```

## Layer Architecture

The system follows a layered architecture pattern:

```mermaid
graph TB
    subgraph "Presentation Layer"
        CLI[CLI Commands]
        GRPC[gRPC API]
    end

    subgraph "Application Layer"
        ING_SVC[Ingestion Service]
        ML_SVC[ML Service]
        EXPLAIN_SVC[Explanation Service]
    end

    subgraph "Domain Layer"
        LR_DOM[LogRecord Domain]
        CLUST_DOM[Cluster Domain]
        NOV_DOM[Novelty Domain]
    end

    subgraph "Infrastructure Layer"
        FS_INF[File System]
        VS_INF[Vector Store]
        LLM_INF[LLM Client]
        LOG_INF[Logging]
    end

    CLI --> ING_SVC
    GRPC --> ML_SVC
    GRPC --> EXPLAIN_SVC

    ING_SVC --> LR_DOM
    ML_SVC --> CLUST_DOM
    ML_SVC --> NOV_DOM
    EXPLAIN_SVC --> CLUST_DOM

    LR_DOM --> FS_INF
    LR_DOM --> LOG_INF
    CLUST_DOM --> VS_INF
    CLUST_DOM --> LLM_INF
```

## SOLID Design Principles

### Single Responsibility Principle (SRP)

Each module has one clear responsibility:

| Module | Responsibility |
|--------|---------------|
| `parser.go` | Parse log lines into structured records |
| `source.go` | Read logs from various sources |
| `logging.go` | Structured JSONL logging |
| `errors.go` | Error types and handling |
| `normalization.py` | Mask sensitive data in logs |
| `exceptions.py` | Exception hierarchy |

### Open/Closed Principle (OCP)

- Parser Registry allows adding new parsers without modifying existing code
- Source interface enables new ingestion sources
- Normalization pipeline supports custom rules

### Liskov Substitution Principle (LSP)

- All parsers implement the Parser interface
- All sources implement the Source interface
- All exceptions inherit from SentinelError

### Interface Segregation Principle (ISP)

- Small, focused interfaces (Parser, Source)
- No forced implementation of unused methods

### Dependency Inversion Principle (DIP)

- Components depend on abstractions (interfaces)
- Logging, parsing, and sources are injected

## Concurrency Model

### Go Agent

```mermaid
sequenceDiagram
    participant Main
    participant Source
    participant Parser
    participant Channel
    participant gRPC

    Main->>Source: Start reading
    loop For each line
        Source->>Parser: Parse line
        Parser->>Channel: Send record
        Channel->>gRPC: Stream to ML
    end
```

### Python ML Engine

```mermaid
sequenceDiagram
    participant gRPC
    participant ThreadPool
    participant Embedding
    participant VectorStore
    participant Clustering

    gRPC->>ThreadPool: Receive batch
    ThreadPool->>Embedding: Generate embeddings
    Embedding->>VectorStore: Store vectors
    VectorStore->>Clustering: Update clusters
    Clustering->>gRPC: Return results
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Single Host"
        AGENT[Go Agent]
        ML[Python ML Engine]
        OLLAMA_L[Ollama Local]
    end

    subgraph "Distributed"
        AGENT_D[Go Agent Pods]
        ML_D[ML Engine Pods]
        OLLAMA_D[Ollama Service]
        VS_D[Vector Store Service]
    end

    subgraph "Cloud"
        AGENT_C[Agent Container]
        ML_C[ML Container]
        OPENAI[OpenAI API]
        COSMOS[Azure Cosmos DB]
        ATHENA[AWS Athena]
    end
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Agent | Go 1.22 | High-performance log ingestion |
| CLI | Cobra | Command-line interface |
| Logging | Zap + Lumberjack | JSONL rolling logs |
| ML Engine | Python 3.10+ | ML/AI processing |
| Config | Pydantic | Configuration management |
| Logging | Structlog | Structured logging |
| IPC | gRPC + Protobuf | Agent-ML communication |
| Vector Store | FAISS | Embedding storage and search |
| Clustering | HDBSCAN | Log pattern clustering |
| LLM | Ollama/OpenAI | Log explanation generation |

## Security Considerations

1. **Data Masking**: PII/sensitive data masked during normalization
2. **Log Isolation**: Logs stored in JSONL for audit trails
3. **gRPC Security**: TLS encryption for production
4. **Error Handling**: No sensitive data in error messages
