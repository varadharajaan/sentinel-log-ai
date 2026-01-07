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

    subgraph "M3 Components"
        CLUSTER[clustering.py]
    end

    subgraph "M4 Components"
        NOVELTY[novelty.py]
    end

    subgraph "M5 Components"
        LLM[llm.py]
    end

    CONFIG --> LOG
    CONFIG --> SERVER
    CONFIG --> EMB
    CONFIG --> VS
    CONFIG --> CLUSTER
    MODELS --> PARSER
    MODELS --> EMB
    MODELS --> VS
    MODELS --> CLUSTER
    NORM --> PARSER
    NORM --> PRE
    EXC --> SERVER
    EXC --> EMB
    EXC --> VS
    EXC --> CLUSTER
    EMB --> SERVER
    VS --> SERVER
    CLUSTER --> SERVER
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

## Clustering Architecture (M3)

The clustering subsystem discovers patterns in log data using HDBSCAN:

```mermaid
graph TB
    subgraph "ClusteringService"
        CS[ClusteringService]
        STATS[ClusterStats]
        SUMMARIES[ClusterSummary List]
    end

    subgraph "Algorithm Strategies"
        CA[ClusteringAlgorithm ABC]
        HDBSCAN[HDBSCANAlgorithm]
        MOCK[MockClusteringAlgorithm]
    end

    subgraph "Output"
        RESULT[ClusteringResult]
        LABELS[Cluster Labels]
        REPS[Representative Samples]
    end

    CS --> STATS
    CS --> CA
    CA --> HDBSCAN
    CA --> MOCK
    CS --> RESULT
    RESULT --> LABELS
    RESULT --> SUMMARIES
    SUMMARIES --> REPS
```

### Clustering Algorithm Strategy Pattern

The clustering system uses the Strategy pattern for algorithm flexibility:

| Algorithm | Use Case |
|-----------|----------|
| `HDBSCANAlgorithm` | Production - Density-based clustering, handles noise |
| `MockClusteringAlgorithm` | Testing - Deterministic mock clustering |

### HDBSCAN Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cluster_size` | 5 | Minimum cluster size |
| `min_samples` | 3 | Core point threshold |
| `cluster_selection_epsilon` | 0.0 | Merge clusters within epsilon |
| `metric` | euclidean | Distance metric |

### Cluster Summary Generation

```mermaid
flowchart TD
    EMB[Embeddings + Records] --> CLUSTER[HDBSCAN Clustering]
    CLUSTER --> LABELS[Cluster Labels]
    LABELS --> FILTER[Filter Non-Noise]
    FILTER --> CENTROID[Calculate Centroids]
    CENTROID --> REPS[Select Representatives]
    REPS --> META[Extract Metadata]
    META --> SUMMARY[ClusterSummary]
    
    subgraph "Metadata Extraction"
        META --> LEVEL[Common Log Level]
        META --> SOURCE[Common Source]
        META --> TIME[Time Range]
    end
```

### Representative Selection Algorithm

1. Calculate cluster centroid from member embeddings
2. Compute distance from each member to centroid
3. Select N closest members as representatives
4. Extract their messages for summary

## Novelty Detection Architecture (M4)

The novelty detection subsystem identifies unusual log patterns using k-NN density estimation:

```mermaid
graph TB
    subgraph "NoveltyService"
        NS[NoveltyService]
        STATS[NoveltyStats]
        THRESHOLD[Threshold Config]
    end

    subgraph "Detector Strategies"
        ND[NoveltyDetector ABC]
        KNN[KNNNoveltyDetector]
        MOCK[MockNoveltyDetector]
    end

    subgraph "Output"
        RESULT[NoveltyResult]
        SCORES[Novelty Scores]
        NOVEL[NoveltyScore Objects]
    end

    NS --> STATS
    NS --> THRESHOLD
    NS --> ND
    ND --> KNN
    ND --> MOCK
    NS --> RESULT
    RESULT --> SCORES
    RESULT --> NOVEL
```

### Novelty Detector Strategy Pattern

The novelty system uses the Strategy pattern for algorithm flexibility:

| Algorithm | Use Case |
|-----------|----------|
| `KNNNoveltyDetector` | Production - k-NN density-based novelty scoring |
| `MockNoveltyDetector` | Testing - Deterministic mock scores |

### k-NN Novelty Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.7 | Score threshold for novel classification |
| `k_neighbors` | 5 | Number of neighbors for density estimation |
| `use_density` | true | Use density-based vs distance-based scoring |

### Novelty Detection Algorithm

```mermaid
flowchart TD
    REF[Reference Embeddings] --> FIT[Fit Detector]
    FIT --> COMPUTE_REF[Compute Reference Densities]
    COMPUTE_REF --> STORE_DIST[Store Density Distribution]
    
    NEW[New Embeddings] --> SCORE[Score]
    SCORE --> KNN[Compute k-NN Distances]
    KNN --> DENSITY[Calculate Local Density]
    DENSITY --> NORMALIZE[Normalize Against Reference]
    NORMALIZE --> SIGMOID[Apply Sigmoid Transform]
    SIGMOID --> CLASSIFY{Score >= Threshold?}
    CLASSIFY -->|Yes| NOVEL_OUT[Novel]
    CLASSIFY -->|No| NORMAL_OUT[Normal]
```

### k-NN Density Scoring

1. **Fit Phase**: Compute k-NN distances for reference embeddings
2. **Reference Distribution**: Calculate mean and std of densities
3. **Score Phase**: For new samples, compute k-NN distances to reference
4. **Density Estimation**: density = 1 / (mean k-NN distance + ε)
5. **Normalization**: z-score against reference distribution
6. **Transform**: Sigmoid to map to [0, 1] novelty score

### Novelty Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.0 - 0.3 | Normal - High density, close to known patterns |
| 0.3 - 0.5 | Slightly unusual - Moderate deviation |
| 0.5 - 0.7 | Unusual - Notable deviation from patterns |
| 0.7 - 0.9 | Novel - Significant anomaly |
| 0.9 - 1.0 | Highly novel - Extreme outlier |

## LLM Explanation Architecture (M5)

The LLM subsystem provides human-readable explanations for log patterns using large language models:

```mermaid
graph TB
    subgraph "LLMService"
        LS[LLMService]
        STATS[LLMStats]
        PROMPTS[Prompt Templates]
    end

    subgraph "Provider Strategies"
        LP[LLMProvider ABC]
        OLLAMA[OllamaProvider]
        MOCK[MockLLMProvider]
    end

    subgraph "Input Types"
        CLUSTER_IN[ClusterSummary]
        NOVELTY_IN[NoveltyScore]
        ERROR_IN[LogRecord]
    end

    subgraph "Output"
        RESULT[Explanation]
        SUMMARY[Summary Text]
        ROOT_CAUSE[Root Cause]
        ACTIONS[Suggested Actions]
        SEV[Severity]
    end

    subgraph "External"
        OLLAMA_SVC[Ollama Service]
    end

    CLUSTER_IN --> LS
    NOVELTY_IN --> LS
    ERROR_IN --> LS
    LS --> STATS
    LS --> PROMPTS
    LS --> LP
    LP --> OLLAMA
    LP --> MOCK
    OLLAMA --> OLLAMA_SVC
    LS --> RESULT
    RESULT --> SUMMARY
    RESULT --> ROOT_CAUSE
    RESULT --> ACTIONS
    RESULT --> SEV
```

### LLM Provider Strategy Pattern

The LLM system uses the Strategy pattern for provider flexibility:

| Provider | Use Case |
|----------|----------|
| `OllamaProvider` | Production - Ollama REST API with retry logic |
| `MockLLMProvider` | Testing - Deterministic mock responses |

### LLM Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `provider` | ollama | LLM provider type |
| `model` | llama3.2 | Model name to use |
| `base_url` | http://localhost:11434 | Ollama server URL |
| `timeout` | 120 | Request timeout in seconds |
| `max_retries` | 3 | Maximum retry attempts |
| `temperature` | 0.1 | Response temperature (lower = more focused) |

### Explanation Types

```mermaid
graph LR
    subgraph "Explanation Types"
        CLUSTER[CLUSTER]
        NOVELTY[NOVELTY]
        ERROR[ERROR_ANALYSIS]
        ROOT[ROOT_CAUSE]
        SUMMARY[SUMMARY]
    end

    subgraph "Severity Levels"
        CRITICAL[CRITICAL]
        HIGH[HIGH]
        MEDIUM[MEDIUM]
        LOW[LOW]
        INFO[INFO]
    end
```

| Explanation Type | Input | Purpose |
|------------------|-------|---------|
| `CLUSTER` | ClusterSummary | Explain cluster pattern and root cause |
| `NOVELTY` | NoveltyScore | Explain why pattern is novel |
| `ERROR_ANALYSIS` | LogRecord | Analyze error and suggest fixes |
| `ROOT_CAUSE` | Multiple inputs | Deep root cause analysis |
| `SUMMARY` | Aggregated data | Executive summary of analysis |

### LLM Explanation Flow

```mermaid
flowchart TD
    INPUT[Input Data] --> BUILD[Build Prompt]
    BUILD --> TEMPLATE[Apply Template]
    TEMPLATE --> GENERATE[Generate via Provider]
    GENERATE --> PARSE[Parse JSON Response]
    PARSE --> VALIDATE[Validate Response]
    VALIDATE --> EXTRACT[Extract Fields]
    EXTRACT --> EXPLANATION[Explanation Object]
    
    subgraph "Response Fields"
        EXPLANATION --> SUM[summary]
        EXPLANATION --> RC[root_cause]
        EXPLANATION --> ACT[suggested_actions]
        EXPLANATION --> SEV2[severity]
        EXPLANATION --> CONF[confidence]
    end
```

### Prompt Templates

Four specialized prompt templates optimize LLM responses:

| Template | Purpose | Key Fields |
|----------|---------|------------|
| `CLUSTER_EXPLANATION_PROMPT` | Cluster analysis | log_messages, cluster_size, common_level |
| `NOVELTY_EXPLANATION_PROMPT` | Novel pattern explanation | log_message, novelty_score, threshold |
| `ERROR_ANALYSIS_PROMPT` | Error diagnosis | error_message, log_level, source, context |
| `SUMMARY_PROMPT` | Executive summary | total_logs, n_clusters, n_novel |

### Retry Logic with Exponential Backoff

```mermaid
flowchart TD
    START[Request] --> TRY[Try Request]
    TRY --> SUCCESS{Success?}
    SUCCESS -->|Yes| RETURN[Return Response]
    SUCCESS -->|No| RETRY{Retries Left?}
    RETRY -->|Yes| BACKOFF[Exponential Backoff]
    BACKOFF --> TRY
    RETRY -->|No| ERROR[Raise LLMError]
```

Backoff formula: `delay = 2^attempt` seconds

### Response Parsing

The LLMService handles various LLM response formats:

1. **Pure JSON**: Directly parsed
2. **Markdown Code Block**: Extracts JSON from ```json``` blocks
3. **Invalid JSON**: Raises `LLMError.invalid_response()`

### LLM Error Handling

| Error Type | Error Code | Retryable |
|------------|------------|-----------|
| `LLM_PROVIDER_ERROR` | 6000 | Yes |
| `LLM_RATE_LIMITED` | 6001 | Yes |
| `LLM_CONTEXT_TOO_LONG` | 6002 | No |
| `LLM_INVALID_RESPONSE` | 6003 | Yes |

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
