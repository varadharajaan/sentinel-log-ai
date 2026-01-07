# Data Flow Documentation

## Overview

This document describes the end-to-end data flow in Sentinel Log AI, from log ingestion to insight generation.

## Complete Data Flow

```mermaid
flowchart TB
    subgraph "1. Ingestion"
        LS[Log Sources] --> ING[Ingestion Engine]
        ING --> PARSE[Parser Registry]
        PARSE --> NORM[Normalizer]
    end

    subgraph "2. Preprocessing"
        NORM --> LR[LogRecord]
        LR --> BATCH[Batch Aggregator]
    end

    subgraph "3. ML Processing"
        BATCH --> EMB[Embedding Generator]
        EMB --> VS[Vector Store]
        VS --> CLUST[Clustering]
        VS --> NOV[Novelty Detection]
    end

    subgraph "4. Analysis"
        CLUST --> SUMM[Cluster Summaries]
        NOV --> ALERT[Novel Pattern Alerts]
        SUMM --> LLM[LLM Explanation]
        ALERT --> LLM
    end

    subgraph "5. Output"
        LLM --> OUT[Explanations]
        BATCH --> LOG[JSONL Logs]
        LOG --> ATHENA[Analytics]
    end
```

## Phase 1: Log Ingestion

### Source Reading

```mermaid
sequenceDiagram
    participant CLI as CLI Command
    participant Src as Source
    participant File as File/Stdin
    participant Parser as Parser

    CLI->>Src: NewFileSource(path, tailMode)
    Src->>File: Open file handle
    
    loop Read lines
        File->>Src: Read line
        Src->>Parser: Parse(line)
        Parser-->>Src: LogRecord
        Src-->>CLI: Record channel
    end
```

### Parser Selection

```mermaid
flowchart TD
    LINE[Log Line] --> JSON{Is JSON?}
    JSON -->|Yes| JSON_P[JSON Parser]
    JSON -->|No| SYSLOG{Is Syslog?}
    SYSLOG -->|Yes| SYSLOG_P[Syslog Parser]
    SYSLOG -->|No| NGINX{Is Nginx?}
    NGINX -->|Yes| NGINX_P[Nginx Parser]
    NGINX -->|No| TB{Is Traceback?}
    TB -->|Yes| TB_P[Traceback Parser]
    TB -->|No| COMMON_P[Common Parser]
    
    JSON_P --> LR[LogRecord]
    SYSLOG_P --> LR
    NGINX_P --> LR
    TB_P --> LR
    COMMON_P --> LR
```

## Phase 2: Normalization

### Masking Pipeline

```mermaid
flowchart LR
    MSG[Raw Message] --> URL[Mask URLs]
    URL --> EMAIL[Mask Emails]
    EMAIL --> IP[Mask IPs]
    IP --> UUID[Mask UUIDs]
    UUID --> TIME[Mask Timestamps]
    TIME --> HEX[Mask Hex Tokens]
    HEX --> NUM[Mask Numbers]
    NUM --> CLEAN[Normalized Message]
```

### Normalization Rules

| Order | Pattern | Replacement | Example |
|-------|---------|-------------|---------|
| 1 | URLs | `<URL>` | `https://example.com` -> `<URL>` |
| 2 | Emails | `<EMAIL>` | `user@domain.com` -> `<EMAIL>` |
| 3 | IPv4 | `<IP>` | `192.168.1.1` -> `<IP>` |
| 4 | UUIDs | `<UUID>` | `550e8400-...` -> `<UUID>` |
| 5 | ISO Timestamps | `<TIMESTAMP>` | `2024-01-15T10:30:00Z` -> `<TIMESTAMP>` |
| 6 | Hex Tokens | `<HEX>` | `abc123def456` -> `<HEX>` |
| 7 | Long Numbers | `<NUM>` | `1234567890` -> `<NUM>` |

## Phase 3: gRPC Communication

### Message Flow

```mermaid
sequenceDiagram
    participant Agent as Go Agent
    participant Stream as gRPC Stream
    participant Server as Python Server
    participant Queue as Processing Queue

    Agent->>Stream: StreamLogs(batch)
    Stream->>Server: Receive batch
    Server->>Queue: Enqueue for processing
    
    loop Process batch
        Queue->>Server: Get embeddings
        Server->>Server: Store in vector DB
        Server->>Server: Update clusters
    end
    
    Server->>Stream: StreamLogsResponse
    Stream->>Agent: Ack batch
```

### Proto Message Structure

```protobuf
message LogRecord {
    string id = 1;
    string raw_message = 2;
    string normalized_message = 3;
    string level = 4;
    google.protobuf.Timestamp timestamp = 5;
    string source = 6;
    map<string, string> attributes = 7;
}

message StreamLogsRequest {
    repeated LogRecord records = 1;
}

message StreamLogsResponse {
    int32 processed_count = 1;
    repeated string errors = 2;
}
```

## Phase 4: ML Processing

### Embedding Generation

```mermaid
flowchart TD
    BATCH[Log Batch] --> TOKENIZE[Tokenize Messages]
    TOKENIZE --> MODEL[Sentence Transformer]
    MODEL --> EMB[Embeddings 384d]
    EMB --> CACHE{In Cache?}
    CACHE -->|Yes| RETRIEVE[Retrieve]
    CACHE -->|No| STORE[Store in Cache]
    STORE --> VS[Vector Store]
    RETRIEVE --> VS
```

### Embedding Generation (M2)

```mermaid
flowchart TD
    BATCH[Log Batch] --> NORM_CHECK{Normalized?}
    NORM_CHECK -->|Yes| CACHE{In Cache?}
    NORM_CHECK -->|No| NORM[Normalize Message]
    NORM --> CACHE
    CACHE -->|Hit| CACHED[Return Cached Embedding]
    CACHE -->|Miss| HASH[Generate Hash Key]
    HASH --> TOKENIZE[Tokenize Messages]
    TOKENIZE --> MODEL[Sentence Transformer<br/>all-MiniLM-L6-v2]
    MODEL --> EMB[384-dim Embedding]
    EMB --> STORE_CACHE[Store in LRU Cache]
    STORE_CACHE --> RETURN[Return Embedding]
    CACHED --> RETURN
```

### Embedding Cache Flow

```mermaid
sequenceDiagram
    participant Client as Client
    participant Service as EmbeddingService
    participant Cache as LRU Cache
    participant Provider as SentenceTransformer

    Client->>Service: embed_records(records)
    loop For each record
        Service->>Cache: get(hash(normalized))
        alt Cache Hit
            Cache-->>Service: cached_embedding
            Service->>Service: stats.cache_hits++
        else Cache Miss
            Service->>Provider: embed(normalized)
            Provider-->>Service: embedding_vector
            Service->>Cache: put(hash, embedding)
            Service->>Service: stats.cache_misses++
        end
    end
    Service-->>Client: embeddings[]
```

### Vector Store Operations (M2)

```mermaid
flowchart TD
    subgraph "Add Flow"
        EMB[Embeddings] --> NORM2[L2 Normalize]
        NORM2 --> INDEX[Add to FAISS Index]
        INDEX --> META[Store Metadata]
        META --> ID[Return IDs]
    end

    subgraph "Search Flow"
        QUERY[Query Vector] --> NORM3[L2 Normalize]
        NORM3 --> SEARCH[FAISS Search]
        SEARCH --> DIST[Distances + Indices]
        DIST --> LOOKUP[Lookup Metadata]
        LOOKUP --> RESULTS[SearchResult[]]
    end
```

### Vector Store Persistence

```mermaid
sequenceDiagram
    participant Store as VectorStore
    participant FAISS as FAISS Index
    participant FS as File System

    Note over Store,FS: Save Operation
    Store->>FAISS: write_index(index)
    FAISS->>FS: vectors.faiss
    Store->>FS: metadata.json
    Store->>FS: config.json

    Note over Store,FS: Load Operation
    FS->>FAISS: read_index(path)
    FAISS->>Store: index
    FS->>Store: metadata.json
    FS->>Store: config.json
```

### Clustering Pipeline

```mermaid
flowchart TD
    VS[Vector Store] --> QUERY[Query Similar]
    QUERY --> HDBSCAN[HDBSCAN Clustering]
    HDBSCAN --> LABELS[Cluster Labels]
    LABELS --> SUMM[Generate Summaries]
    SUMM --> REP[Representative Logs]
    REP --> OUT[Cluster Output]
```

### Novelty Detection

```mermaid
flowchart TD
    NEW[New Log] --> EMB[Embedding]
    EMB --> KNN[k-NN Search]
    KNN --> DIST[Distance Calculation]
    DIST --> DENS[Density Estimation]
    DENS --> SCORE{Score > Threshold?}
    SCORE -->|Yes| NOVEL[Mark as Novel]
    SCORE -->|No| KNOWN[Mark as Known]
    NOVEL --> ALERT[Generate Alert]
```

## Phase 5: LLM Explanation

### Explanation Flow

```mermaid
sequenceDiagram
    participant Cluster as Cluster Data
    participant Prompt as Prompt Builder
    participant LLM as Ollama/OpenAI
    participant Output as Explanation

    Cluster->>Prompt: Representative logs
    Cluster->>Prompt: Cluster metadata
    Prompt->>LLM: Generate explanation
    LLM->>Output: Root cause
    LLM->>Output: Suggested actions
    LLM->>Output: Confidence score
```

### Prompt Template

```
Analyze these log messages from cluster {cluster_id}:

{representative_logs}

Cluster Statistics:
- Size: {count} messages
- Time Range: {first_seen} to {last_seen}
- Common Level: {dominant_level}

Provide:
1. Probable root cause
2. Suggested next steps
3. Severity assessment
4. Confidence score (0-1)
```

## Phase 6: Logging and Analytics

### JSONL Log Format

```mermaid
flowchart LR
    EVENT[Log Event] --> STRUCT[Structured Fields]
    STRUCT --> JSON[JSON Serialize]
    JSON --> LINE[Single Line]
    LINE --> FILE[Rolling File]
    FILE --> ATHENA[Athena Query]
```

### Log Entry Structure

```json
{
    "timestamp": "2024-01-15T10:30:00.000Z",
    "level": "info",
    "service": "sentinel-agent",
    "hostname": "prod-server-01",
    "pid": 12345,
    "msg": "ingestion_completed",
    "path": "/var/log/app.log",
    "count": 1000,
    "duration": "2.5s",
    "request_id": "req-abc123"
}
```

### Athena Query Examples

```sql
-- Find all errors in the last hour
SELECT * FROM sentinel_logs
WHERE level = 'error'
AND timestamp > current_timestamp - interval '1' hour;

-- Count logs by service and level
SELECT service, level, count(*) as cnt
FROM sentinel_logs
WHERE date = current_date
GROUP BY service, level
ORDER BY cnt DESC;

-- Find slow operations
SELECT msg, duration, path
FROM sentinel_logs
WHERE duration > 5.0
ORDER BY duration DESC
LIMIT 100;
```

## Error Flow

### Error Handling Chain

```mermaid
flowchart TD
    OP[Operation] --> ERR{Error?}
    ERR -->|No| SUCCESS[Continue]
    ERR -->|Yes| CLASSIFY[Classify Error]
    CLASSIFY --> RETRY{Retryable?}
    RETRY -->|Yes| WAIT[Backoff Wait]
    WAIT --> RETRY_OP[Retry Operation]
    RETRY_OP --> OP
    RETRY -->|No| LOG[Log Error]
    LOG --> PROPAGATE[Propagate Up]
```

### Error Categories

| Category | Code Range | Retryable | Example |
|----------|-----------|-----------|---------|
| Configuration | 1xxx | No | Invalid config file |
| Ingestion | 2xxx | Partial | File not found (no), Timeout (yes) |
| Processing | 3xxx | Yes | OOM during embedding |
| Storage | 4xxx | Yes | Write failure |
| Communication | 5xxx | Yes | gRPC timeout |
| LLM | 6xxx | Yes | Rate limiting |
