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

### Clustering Pipeline (M3)

```mermaid
flowchart TD
    EMB[Embeddings Array] --> VALID{Valid Input?}
    VALID -->|Empty| EMPTY_RESULT[Empty Result]
    VALID -->|Valid| HDBSCAN[HDBSCAN Clustering]
    HDBSCAN --> LABELS[Cluster Labels]
    LABELS --> FILTER[Filter Noise -1]
    FILTER --> UNIQUE[Unique Clusters]
    UNIQUE --> GEN[Generate Summaries]
    
    subgraph "Summary Generation"
        GEN --> CENT[Calculate Centroid]
        CENT --> DIST[Distance to Centroid]
        DIST --> TOP_N[Select N Closest]
        TOP_N --> MSGS[Extract Messages]
        MSGS --> META[Extract Metadata]
        META --> SUMM[ClusterSummary]
    end
    
    SUMM --> RESULT[ClusteringResult]
```

### Clustering Service Sequence

```mermaid
sequenceDiagram
    participant Client as Server/Client
    participant Service as ClusteringService
    participant Algo as HDBSCANAlgorithm
    participant Stats as ClusterStats

    Client->>Service: cluster(embeddings, records)
    Service->>Algo: fit_predict(embeddings)
    Algo-->>Service: labels[]
    
    loop For each unique cluster label
        Service->>Service: extract_cluster_members()
        Service->>Service: calculate_centroid()
        Service->>Service: select_representatives()
        Service->>Service: extract_metadata()
    end
    
    Service->>Stats: update(n_clustered, n_clusters)
    Service-->>Client: ClusteringResult
```

### Representative Selection Algorithm

```mermaid
flowchart LR
    MEMBERS[Cluster Members] --> CENT[Calculate Centroid]
    CENT --> DISTS[Compute Distances]
    DISTS --> SORT[Sort by Distance]
    SORT --> TOP[Take Top N]
    TOP --> REPS[Representatives]
```

### Novelty Detection Pipeline (M4)

```mermaid
flowchart TD
    subgraph "Fit Phase"
        REF[Reference Embeddings] --> FIT_KNN[Compute k-NN Distances]
        FIT_KNN --> REF_DENS[Calculate Reference Densities]
        REF_DENS --> DIST_STATS[Store Mean/Std of Distribution]
    end

    subgraph "Detection Phase"
        NEW[New Embeddings] --> CROSS_KNN[Cross k-NN to Reference]
        CROSS_KNN --> NEW_DENS[Calculate New Densities]
        NEW_DENS --> Z_SCORE[Compute Z-Scores]
        Z_SCORE --> SIGMOID[Apply Sigmoid Transform]
        SIGMOID --> SCORES[Novelty Scores 0-1]
    end

    subgraph "Classification"
        SCORES --> THRESHOLD{Score >= Threshold?}
        THRESHOLD -->|Yes| NOVEL[Mark as Novel]
        THRESHOLD -->|No| NORMAL[Mark as Normal]
        NOVEL --> EXPLAIN[Generate Explanation]
        EXPLAIN --> ALERT[Novel Log Alert]
    end
```

### Novelty Detection Sequence

```mermaid
sequenceDiagram
    participant Client as Server/Client
    participant Service as NoveltyService
    participant Detector as KNNNoveltyDetector
    participant Stats as NoveltyStats

    Note over Client,Stats: Fit Phase (establish baseline)
    Client->>Service: fit(reference_embeddings)
    Service->>Detector: fit(embeddings)
    Detector->>Detector: compute_knn_distances()
    Detector->>Detector: calculate_density_distribution()
    Detector-->>Service: fitted

    Note over Client,Stats: Detection Phase
    Client->>Service: detect(new_embeddings, threshold)
    Service->>Detector: score(embeddings)
    Detector->>Detector: compute_cross_knn_distances()
    Detector->>Detector: calculate_densities()
    Detector->>Detector: normalize_and_transform()
    Detector-->>Service: scores[]
    
    Service->>Service: classify_by_threshold()
    Service->>Service: build_novel_scores()
    Service->>Stats: update(n_analyzed, n_novel)
    Service-->>Client: NoveltyResult
```

### k-NN Density Calculation

```mermaid
flowchart TD
    EMB[Embedding Vector] --> DIST[Compute Pairwise Distances]
    DIST --> PART[Partition k Smallest]
    PART --> KNN_D[k-NN Distances]
    KNN_D --> MEAN[Mean Distance]
    MEAN --> INV[Density = 1 / Mean + ε]
    INV --> DENS[Local Density Estimate]
```

### Novelty Score Interpretation Flow

```mermaid
flowchart LR
    SCORE[Novelty Score] --> RANGE{Score Range}
    RANGE -->|0.0-0.3| NORMAL[Normal Pattern]
    RANGE -->|0.3-0.5| SLIGHT[Slight Deviation]
    RANGE -->|0.5-0.7| UNUSUAL[Unusual]
    RANGE -->|0.7-0.9| NOVEL[Novel - Alert]
    RANGE -->|0.9-1.0| EXTREME[Highly Novel - Critical]
```

### Full Pipeline: Ingest, Embed, and Detect

```mermaid
sequenceDiagram
    participant Client
    participant Server as MLServiceServicer
    participant Preproc as Preprocessing
    participant Embed as EmbeddingService
    participant Store as VectorStore
    participant Novelty as NoveltyService

    Client->>Server: ingest_embed_and_detect_novelty(records)
    Server->>Preproc: preprocess_batch(records)
    Preproc-->>Server: processed_records
    
    Server->>Embed: embed_records(processed)
    Embed-->>Server: embeddings
    
    Server->>Store: add(embeddings, records)
    Store-->>Server: ids
    
    alt Not Fitted
        Server->>Novelty: fit(reference_embeddings)
    end
    
    Server->>Novelty: detect(embeddings)
    Novelty-->>Server: NoveltyResult
    
    Server-->>Client: (processed, embeddings, ids, novelty_result)
```

## Phase 5: LLM Explanation (M5)

### LLM Explanation Flow

```mermaid
sequenceDiagram
    participant Client as Server/Client
    participant Service as LLMService
    participant Provider as LLMProvider
    participant Ollama as Ollama API
    participant Stats as LLMStats

    Client->>Service: explain_cluster(summary)
    Service->>Service: build_prompt(template, data)
    Service->>Provider: generate(prompt, temperature)
    
    alt OllamaProvider
        Provider->>Ollama: POST /api/generate
        loop Retry on failure
            alt Success
                Ollama-->>Provider: JSON response
            else Failure
                Provider->>Provider: exponential_backoff()
                Provider->>Ollama: Retry
            end
        end
    end
    
    Provider-->>Service: (response_text, tokens)
    Service->>Service: parse_response(json)
    Service->>Service: validate_fields()
    Service->>Stats: record_request(success, tokens)
    Service-->>Client: Explanation
```

### Prompt Building Flow

```mermaid
flowchart TD
    subgraph "Input Data"
        CS[ClusterSummary]
        NS[NoveltyScore]
        LR[LogRecord]
    end

    subgraph "Prompt Templates"
        CLUSTER_T[CLUSTER_EXPLANATION_PROMPT]
        NOVELTY_T[NOVELTY_EXPLANATION_PROMPT]
        ERROR_T[ERROR_ANALYSIS_PROMPT]
        SUMMARY_T[SUMMARY_PROMPT]
    end

    subgraph "Built Prompt"
        PROMPT[Formatted Prompt]
    end

    CS --> CLUSTER_T
    NS --> NOVELTY_T
    LR --> ERROR_T
    
    CLUSTER_T --> PROMPT
    NOVELTY_T --> PROMPT
    ERROR_T --> PROMPT
    SUMMARY_T --> PROMPT
```

### Response Parsing Flow

```mermaid
flowchart TD
    RESPONSE[LLM Response Text] --> CHECK{Is Pure JSON?}
    CHECK -->|Yes| PARSE[Parse JSON]
    CHECK -->|No| REGEX[Extract from Markdown]
    REGEX --> FOUND{JSON Found?}
    FOUND -->|Yes| PARSE
    FOUND -->|No| ERROR[Raise LLMError]
    
    PARSE --> VALIDATE{Valid Fields?}
    VALIDATE -->|Yes| EXTRACT[Extract Fields]
    VALIDATE -->|No| DEFAULTS[Use Default Values]
    
    EXTRACT --> BUILD[Build Explanation]
    DEFAULTS --> BUILD
    BUILD --> SEVERITY[Map Severity Enum]
    SEVERITY --> RESULT[Explanation Object]
```

### Explanation Output Structure

```mermaid
graph LR
    subgraph "Explanation"
        ID[id: UUID]
        TYPE[explanation_type]
        SUMMARY[summary: str]
        ROOT[root_cause: str]
        ACTIONS[suggested_actions: list]
        SEV[severity: Severity]
        CONF[confidence: float]
        MODEL[model: str]
        TOKENS[token counts]
        TIME[response_time]
        META[metadata: dict]
    end
```

### Full Pipeline: Cluster to Explanation

```mermaid
sequenceDiagram
    participant Client
    participant Server as MLServiceServicer
    participant Cluster as ClusteringService
    participant LLM as LLMService
    participant Ollama as Ollama API

    Client->>Server: ingest_embed_and_cluster(records)
    Server->>Server: preprocess + embed + cluster
    Server-->>Client: ClusteringResult
    
    loop For each cluster
        Client->>Server: explain_cluster(cluster_summary)
        Server->>LLM: explain_cluster(summary)
        LLM->>LLM: build_prompt()
        LLM->>Ollama: POST /api/generate
        Ollama-->>LLM: JSON response
        LLM->>LLM: parse_response()
        LLM-->>Server: Explanation
        Server-->>Client: Explanation
    end
```

### LLM Provider Selection

```mermaid
flowchart TD
    CONFIG[LLMConfig] --> PROVIDER{provider type}
    PROVIDER -->|ollama| OLLAMA[OllamaProvider]
    PROVIDER -->|mock| MOCK[MockLLMProvider]
    
    OLLAMA --> OPTS[Configure options]
    OPTS --> URL[base_url: localhost:11434]
    OPTS --> MODEL[model: llama3.2]
    OPTS --> RETRY[max_retries: 3]
    OPTS --> TIMEOUT[timeout: 120s]
    
    MOCK --> DET[Deterministic responses]
    DET --> TEST[For unit testing]
```

### Error Handling in LLM Flow

```mermaid
flowchart TD
    GENERATE[Generate Request] --> TRY{Try Request}
    TRY -->|Success| PARSE[Parse Response]
    TRY -->|URLError| RETRY_CHECK{Retries Left?}
    TRY -->|Timeout| RETRY_CHECK
    TRY -->|HTTPError 429| RATE[Rate Limited]
    
    RATE --> WAIT[Wait retry_after]
    WAIT --> TRY
    
    RETRY_CHECK -->|Yes| BACKOFF[Exponential Backoff]
    BACKOFF --> TRY
    RETRY_CHECK -->|No| FAIL[Raise LLMError.provider_error]
    
    PARSE --> VALID{Valid JSON?}
    VALID -->|Yes| SUCCESS[Return Explanation]
    VALID -->|No| INV[Raise LLMError.invalid_response]
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

## CLI Output Flow (M6)

### Console Output Flow

```mermaid
flowchart TD
    DATA[Analysis Results] --> CONSOLE[Console]
    
    CONSOLE --> THEME[Apply Theme]
    THEME --> FORMAT{Output Format?}
    
    FORMAT -->|TEXT| TEXT_OUT[Text Output]
    FORMAT -->|JSON| JSON_OUT[JSON Output]
    FORMAT -->|TABLE| TABLE_OUT[Rich Tables]
    FORMAT -->|COMPACT| COMPACT_OUT[Minimal Output]
    
    TEXT_OUT --> TERMINAL[Terminal]
    JSON_OUT --> TERMINAL
    TABLE_OUT --> TERMINAL
    COMPACT_OUT --> TERMINAL
```

### Report Generation Flow

```mermaid
flowchart TD
    RESULTS[Analysis Results] --> REPORT_DATA[ReportData]
    
    REPORT_DATA --> MARKDOWN[Markdown Reporter]
    REPORT_DATA --> HTML[HTML Reporter]
    
    MARKDOWN --> TOC[Generate TOC]
    TOC --> SUMMARY[Executive Summary]
    SUMMARY --> CLUSTERS[Cluster Details]
    CLUSTERS --> NOVELTY[Novelty Section]
    NOVELTY --> EXPLAIN[Explanations]
    EXPLAIN --> MD_FILE[.md File]
    
    HTML --> CSS[Embed CSS]
    CSS --> RENDER[Render Sections]
    RENDER --> HTML_FILE[.html File]
```

### Progress Tracking Flow

```mermaid
sequenceDiagram
    participant User as User
    participant CLI as CLI
    participant Tracker as ProgressTracker
    participant Console as Console

    User->>CLI: Run analysis
    CLI->>Tracker: Create tracker
    Tracker->>Console: Show spinner
    
    loop Process batches
        CLI->>Tracker: Update progress
        Tracker->>Console: Update display
        Note over Console: Show ETA, rate
    end
    
    CLI->>Tracker: Complete
    Tracker->>Console: Show success
```

### Profiling Flow

```mermaid
flowchart TD
    ENTRY[Function Entry] --> START[Start Timer]
    START --> EXEC[Execute Code]
    EXEC --> NESTED{Nested Measure?}
    NESTED -->|Yes| CHILD[Child Timer]
    CHILD --> EXEC
    NESTED -->|No| END[End Timer]
    END --> RECORD[Record Timing]
    RECORD --> REPORT[Generate Report]
    
    subgraph "Timing Entry"
        REPORT --> NAME[Operation Name]
        REPORT --> DURATION[Duration ms]
        REPORT --> PARENT[Parent Operation]
    end
```

### Configuration Flow

```mermaid
flowchart TD
    CMD[Config Command] --> ACTION{Action?}
    
    ACTION -->|init| GEN[Generate Config]
    GEN --> TEMPLATE[Default Template]
    TEMPLATE --> WRITE[Write YAML]
    
    ACTION -->|validate| LOAD[Load YAML]
    LOAD --> PARSE[Parse Sections]
    PARSE --> CHECK[Validate Values]
    CHECK --> RESULT{Valid?}
    RESULT -->|Yes| OK[Show Success]
    RESULT -->|No| ERRORS[Show Errors]
    
    ACTION -->|show| READ[Read Config]
    READ --> FORMAT[Format Display]
    FORMAT --> OUTPUT[Console Output]
```

## Phase 8: Alerting and Notifications

### Watch Daemon Flow

```mermaid
flowchart TD
    START[Start Daemon] --> INIT[Initialize]
    INIT --> DISCOVER[Discover Files]
    DISCOVER --> RECORD[Record Initial Positions]
    RECORD --> POLL{Poll Loop}
    
    POLL --> CHECK[Check Files]
    CHECK --> NEW{New Lines?}
    NEW -->|No| WAIT[Wait Interval]
    WAIT --> POLL
    
    NEW -->|Yes| READ[Read New Lines]
    READ --> DETECT[Run Novelty Detector]
    DETECT --> NOVEL{Score >= Threshold?}
    
    NOVEL -->|No| POLL
    NOVEL -->|Yes| CREATE[Create AlertEvent]
    CREATE --> NOTIFY[Notify All Notifiers]
    NOTIFY --> POLL
    
    POLL -->|Stop Signal| CLEANUP[Cleanup]
    CLEANUP --> END[Stopped]
```

### Alert Routing Decision Flow

```mermaid
flowchart TD
    EVENT[AlertEvent] --> ROUTER[Alert Router]
    ROUTER --> RULES[Evaluate Rules]
    
    RULES --> R1{Rule 1 Match?}
    R1 -->|Yes| N1[Send to Notifiers]
    N1 --> STOP1{Stop on Match?}
    STOP1 -->|Yes| DONE[Routing Complete]
    STOP1 -->|No| R2{Rule 2 Match?}
    
    R1 -->|No| R2
    R2 -->|Yes| N2[Send to Notifiers]
    N2 --> STOP2{Stop on Match?}
    STOP2 -->|Yes| DONE
    STOP2 -->|No| RN{More Rules?}
    
    R2 -->|No| RN
    RN -->|Yes| CONTINUE[Continue Evaluation]
    CONTINUE --> R2
    
    RN -->|No| MATCHED{Any Matches?}
    MATCHED -->|Yes| DONE
    MATCHED -->|No| FALLBACK[Send to Fallback]
    FALLBACK --> DONE
```

### Notification Delivery Flow

```mermaid
sequenceDiagram
    participant Router as Alert Router
    participant Base as BaseNotifier
    participant Impl as Notifier Impl
    participant External as External Service

    Router->>Base: send(event)
    Base->>Base: Check enabled
    Base->>Base: Validate config
    
    loop Retry Loop
        Base->>Impl: _send_impl(event)
        Impl->>Impl: Build payload
        Impl->>External: HTTP POST
        
        alt Success
            External-->>Impl: 200 OK
            Impl-->>Base: AlertResult(SUCCESS)
            Base->>Base: Update stats
            Base-->>Router: Result
        else Failure
            External-->>Impl: Error
            Impl-->>Base: Exception
            Base->>Base: Increment retry
            alt More Retries
                Base->>Base: Wait delay
            else Max Retries
                Base->>Base: Update stats
                Base-->>Router: AlertResult(FAILED)
            end
        end
    end
```

### Health Check Aggregation

```mermaid
flowchart TD
    HC[Health Check] --> INIT[Start Check]
    INIT --> DAEMON{Watch Daemon?}
    
    DAEMON -->|Yes| CHECK_D[Check Daemon Status]
    CHECK_D --> D_STATUS{Status?}
    D_STATUS -->|RUNNING| D_OK[daemon: HEALTHY]
    D_STATUS -->|STARTING| D_DEG[daemon: DEGRADED]
    D_STATUS -->|ERROR| D_BAD[daemon: UNHEALTHY]
    
    DAEMON -->|No| NOTIFIERS
    D_OK --> NOTIFIERS
    D_DEG --> NOTIFIERS
    D_BAD --> NOTIFIERS
    
    NOTIFIERS{Notifiers?} -->|Yes| CHECK_N[Check Each Notifier]
    CHECK_N --> N_LOOP[For Each Notifier]
    N_LOOP --> N_HEALTH[health_check()]
    N_HEALTH --> N_STATUS{Healthy?}
    N_STATUS -->|Yes| N_OK[notifier: HEALTHY]
    N_STATUS -->|No| N_BAD[notifier: UNHEALTHY]
    N_OK --> N_NEXT{More?}
    N_BAD --> N_NEXT
    N_NEXT -->|Yes| N_LOOP
    N_NEXT -->|No| AGGREGATE
    
    NOTIFIERS -->|No| AGGREGATE
    
    AGGREGATE[Aggregate Status] --> WORST[Worst Status Wins]
    WORST --> RESPONSE[Build Response]
    RESPONSE --> RETURN[Return Health Status]
```
