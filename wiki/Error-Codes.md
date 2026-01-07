# Error Codes Reference

Sentinel Log AI uses structured error codes for consistent error handling across Go and Python components.

## Error Code Structure

Error codes follow a hierarchical numbering scheme:

```
XYYY
│└┴┴─ Specific error within category (001-999)
└──── Category (1-9)
```

## Error Categories

| Category | Code Range | Description |
|----------|------------|-------------|
| Configuration | 1000-1999 | Config file and validation errors |
| Ingestion | 2000-2999 | Log reading and parsing errors |
| Processing | 3000-3999 | ML processing errors |
| Storage | 4000-4999 | Vector store and persistence errors |
| Communication | 5000-5999 | gRPC and network errors |
| LLM | 6000-6999 | LLM provider errors |
| CLI | 7000-7999 | Command-line interface errors |

---

## Configuration Errors (1xxx)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 1001 | `CONFIG_NOT_FOUND` | Config file not found | Check file path and permissions |
| 1002 | `CONFIG_PARSE_ERROR` | YAML/JSON parse error | Validate config file syntax |
| 1003 | `CONFIG_VALIDATION_ERROR` | Invalid config values | Check config against schema |
| 1004 | `CONFIG_MISSING_REQUIRED` | Required field missing | Add required configuration |
| 1005 | `CONFIG_TYPE_ERROR` | Wrong type for field | Check field type in docs |
| 1010 | `CONFIG_SERVER_INVALID` | Invalid server config | Check host/port values |
| 1020 | `CONFIG_EMBEDDING_INVALID` | Invalid embedding config | Check model name, dimension |
| 1030 | `CONFIG_CLUSTERING_INVALID` | Invalid clustering config | Check algorithm parameters |
| 1040 | `CONFIG_NOVELTY_INVALID` | Invalid novelty config | Check threshold, k_neighbors |
| 1050 | `CONFIG_LLM_INVALID` | Invalid LLM config | Check provider, model, API key |

### Example

```python
from sentinel_ml.exceptions import ConfigError

try:
    config = load_config("nonexistent.yaml")
except ConfigError as e:
    print(f"Error {e.code}: {e.message}")
    # Error 1001: Configuration file not found: nonexistent.yaml
```

---

## Ingestion Errors (2xxx)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 2001 | `INGESTION_FILE_NOT_FOUND` | Log file not found | Check file path |
| 2002 | `INGESTION_PERMISSION_DENIED` | Cannot read file | Check file permissions |
| 2003 | `INGESTION_IO_ERROR` | General I/O error | Check disk/network |
| 2010 | `INGESTION_PARSE_ERROR` | Cannot parse log line | Check log format |
| 2011 | `INGESTION_INVALID_JSON` | Invalid JSON log | Validate JSON syntax |
| 2012 | `INGESTION_INVALID_SYSLOG` | Invalid syslog format | Check syslog format |
| 2020 | `INGESTION_ENCODING_ERROR` | Character encoding issue | Check file encoding (UTF-8) |
| 2030 | `INGESTION_TIMEOUT` | Read timeout | Increase timeout or check source |
| 2040 | `INGESTION_SOURCE_EXHAUSTED` | Source has no more data | Normal end of input |

### Example

```go
// Go
record, err := parser.Parse(line)
if err != nil {
    var parseErr *errors.IngestionError
    if errors.As(err, &parseErr) {
        log.Warn("Parse failed",
            zap.Int("code", parseErr.Code),
            zap.String("line", line),
        )
    }
}
```

---

## Processing Errors (3xxx)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 3001 | `PROCESSING_EMBEDDING_FAILED` | Embedding generation failed | Check model, input |
| 3002 | `PROCESSING_EMBEDDING_DIM_MISMATCH` | Dimension mismatch | Ensure consistent model |
| 3010 | `PROCESSING_CLUSTERING_FAILED` | Clustering failed | Check embeddings, params |
| 3011 | `PROCESSING_INSUFFICIENT_DATA` | Not enough data to cluster | Provide more samples |
| 3020 | `PROCESSING_NOVELTY_FAILED` | Novelty detection failed | Check reference data |
| 3021 | `PROCESSING_NOT_FITTED` | Detector not trained | Fit before predict |
| 3030 | `PROCESSING_NORMALIZATION_ERROR` | Normalization failed | Check input format |
| 3040 | `PROCESSING_OUT_OF_MEMORY` | Insufficient memory | Reduce batch size |
| 3050 | `PROCESSING_GPU_ERROR` | GPU processing error | Fallback to CPU |

### Example

```python
from sentinel_ml.exceptions import ProcessingError

try:
    embeddings = embedding_service.embed(texts)
except ProcessingError as e:
    if e.code == 3040:  # OOM
        # Retry with smaller batch
        embeddings = embedding_service.embed(texts, batch_size=16)
```

---

## Storage Errors (4xxx)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 4001 | `STORAGE_INIT_FAILED` | Vector store init failed | Check config, dependencies |
| 4002 | `STORAGE_ADD_FAILED` | Failed to add vectors | Check vector dimensions |
| 4003 | `STORAGE_SEARCH_FAILED` | Search operation failed | Check query format |
| 4004 | `STORAGE_DELETE_FAILED` | Delete operation failed | Check ID exists |
| 4010 | `STORAGE_PERSIST_FAILED` | Failed to save to disk | Check path, permissions |
| 4011 | `STORAGE_LOAD_FAILED` | Failed to load from disk | Check file exists, format |
| 4020 | `STORAGE_INDEX_CORRUPT` | Index file corrupted | Rebuild index |
| 4030 | `STORAGE_CAPACITY_EXCEEDED` | Max capacity reached | Increase capacity or prune |

### Example

```python
from sentinel_ml.exceptions import StorageError

try:
    vector_store.save(Path("index"))
except StorageError as e:
    if e.code == 4010:
        logger.error("Failed to persist index", path=str(e.details.get("path")))
```

---

## Communication Errors (5xxx)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 5001 | `GRPC_CONNECTION_FAILED` | Cannot connect to server | Check server is running |
| 5002 | `GRPC_TIMEOUT` | Request timed out | Increase timeout |
| 5003 | `GRPC_UNAVAILABLE` | Server unavailable | Check network, retry |
| 5004 | `GRPC_CANCELLED` | Request cancelled | Check client abort |
| 5010 | `GRPC_INVALID_REQUEST` | Malformed request | Check request format |
| 5011 | `GRPC_INVALID_RESPONSE` | Malformed response | Check server version |
| 5020 | `GRPC_AUTH_FAILED` | Authentication failed | Check credentials |
| 5021 | `GRPC_PERMISSION_DENIED` | Not authorized | Check permissions |
| 5030 | `GRPC_RESOURCE_EXHAUSTED` | Server overloaded | Back off and retry |

### Retry Logic

```go
// Go with exponential backoff
for attempt := 0; attempt < maxRetries; attempt++ {
    resp, err := client.Process(ctx, req)
    if err == nil {
        return resp, nil
    }
    
    if errors.IsRetryable(err) {
        time.Sleep(time.Second * time.Duration(math.Pow(2, float64(attempt))))
        continue
    }
    
    return nil, err
}
```

---

## LLM Errors (6xxx)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 6001 | `LLM_PROVIDER_ERROR` | Provider API error | Check provider status |
| 6002 | `LLM_RATE_LIMITED` | Rate limit exceeded | Wait and retry |
| 6003 | `LLM_CONTEXT_TOO_LONG` | Input too long | Reduce context size |
| 6004 | `LLM_INVALID_RESPONSE` | Cannot parse response | Retry or adjust prompt |
| 6010 | `LLM_MODEL_NOT_FOUND` | Model not available | Check model name |
| 6011 | `LLM_MODEL_LOADING` | Model still loading | Wait for model |
| 6020 | `LLM_OLLAMA_NOT_RUNNING` | Ollama not running | Start Ollama service |
| 6021 | `LLM_OPENAI_API_ERROR` | OpenAI API error | Check API key, quota |
| 6030 | `LLM_TIMEOUT` | Generation timed out | Increase timeout |

### Example

```python
from sentinel_ml.exceptions import LLMError

try:
    explanation = llm_service.explain_cluster(cluster)
except LLMError as e:
    if e.code == 6002:  # Rate limited
        time.sleep(e.details.get("retry_after", 60))
        explanation = llm_service.explain_cluster(cluster)
    elif e.code == 6003:  # Context too long
        # Truncate input
        cluster.representative = cluster.representative[:1000]
        explanation = llm_service.explain_cluster(cluster)
```

---

## CLI Errors (7xxx)

| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 7001 | `CLI_INVALID_ARGUMENT` | Invalid command argument | Check command syntax |
| 7002 | `CLI_MISSING_ARGUMENT` | Required argument missing | Provide required args |
| 7003 | `CLI_INVALID_FORMAT` | Invalid output format | Use valid format |
| 7010 | `CLI_OUTPUT_ERROR` | Cannot write output | Check stdout/file perms |
| 7011 | `CLI_INPUT_ERROR` | Cannot read input | Check stdin/file |
| 7020 | `CLI_REPORT_ERROR` | Report generation failed | Check template, data |
| 7021 | `CLI_REPORT_WRITE_ERROR` | Cannot write report | Check path, permissions |
| 7030 | `CLI_CONFIG_GENERATE_ERROR` | Cannot generate config | Check output path |

---

## Error Handling Best Practices

### Python

```python
from sentinel_ml.exceptions import (
    SentinelError,
    ConfigError,
    ProcessingError,
    LLMError,
)

try:
    result = analyze_logs(logs)
except ConfigError as e:
    # Handle configuration issues
    logger.error("Config error", code=e.code, message=e.message)
    sys.exit(1)
except ProcessingError as e:
    # Handle processing failures
    if e.is_retryable:
        result = retry_with_backoff(analyze_logs, logs)
    else:
        raise
except LLMError as e:
    # Handle LLM issues gracefully
    logger.warning("LLM unavailable, skipping explanations", error=str(e))
    result = analyze_logs_without_llm(logs)
except SentinelError as e:
    # Catch-all for other Sentinel errors
    logger.error("Unexpected error", code=e.code, details=e.details)
    raise
```

### Go

```go
import "github.com/sentinel-log-ai/sentinel-log-ai/internal/errors"

result, err := processLogs(logs)
if err != nil {
    var sentinelErr *errors.SentinelError
    if errors.As(err, &sentinelErr) {
        switch {
        case sentinelErr.Code >= 1000 && sentinelErr.Code < 2000:
            // Configuration error
            log.Fatal("Config error", zap.Error(err))
        case errors.IsRetryable(err):
            // Retry with backoff
            result, err = retryWithBackoff(processLogs, logs)
        default:
            log.Error("Processing failed", zap.Error(err))
        }
    }
}
```

---

## Error Response Format

### gRPC

```protobuf
message ErrorDetail {
  int32 code = 1;
  string message = 2;
  map<string, string> metadata = 3;
}
```

### JSON (CLI/API)

```json
{
  "error": {
    "code": 6002,
    "message": "Rate limit exceeded",
    "details": {
      "retry_after": 60,
      "limit": "100 requests/minute"
    },
    "retryable": true
  }
}
```

---

*See also: [[Testing Guide|Testing-Guide]], [[API Reference|API-Reference]]*
