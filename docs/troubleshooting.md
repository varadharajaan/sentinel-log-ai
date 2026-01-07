# Troubleshooting Guide

This guide helps diagnose and resolve common issues with Sentinel Log AI.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Connection Problems](#connection-problems)
- [Performance Issues](#performance-issues)
- [Memory Problems](#memory-problems)
- [Parsing Errors](#parsing-errors)
- [ML Engine Issues](#ml-engine-issues)
- [LLM Integration](#llm-integration)
- [Logging and Debugging](#logging-and-debugging)

## Installation Issues

### Python Dependencies Fail to Install

**Symptom**: `pip install` fails with compilation errors.

**Solution**:
1. Ensure Python 3.10+ is installed
2. Install build dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-dev build-essential
   
   # macOS
   xcode-select --install
   
   # Windows
   # Install Visual C++ Build Tools
   ```
3. Use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate   # Windows
   pip install -e .
   ```

### Go Build Fails

**Symptom**: `make build-go` fails.

**Solution**:
1. Ensure Go 1.22+ is installed:
   ```bash
   go version
   ```
2. Update dependencies:
   ```bash
   go mod tidy
   go mod download
   ```
3. Clear module cache if corrupted:
   ```bash
   go clean -modcache
   go mod download
   ```

### Protobuf Generation Errors

**Symptom**: gRPC stubs are outdated or missing.

**Solution**:
1. Install protoc compiler:
   ```bash
   # macOS
   brew install protobuf
   
   # Linux
   sudo apt-get install protobuf-compiler
   ```
2. Install buf:
   ```bash
   go install github.com/bufbuild/buf/cmd/buf@latest
   ```
3. Regenerate stubs:
   ```bash
   make proto
   ```

## Connection Problems

### gRPC Connection Refused

**Symptom**: Agent cannot connect to ML server.
```
error: connection refused to localhost:50051
```

**Solution**:
1. Ensure ML server is running:
   ```bash
   make run-ml
   ```
2. Check the port is correct:
   ```bash
   # Linux/macOS
   lsof -i :50051
   
   # Windows
   netstat -an | findstr 50051
   ```
3. Verify server health:
   ```bash
   grpcurl -plaintext localhost:50051 list
   ```

### Connection Timeout

**Symptom**: Operations timeout during processing.

**Solution**:
1. Increase timeout in configuration:
   ```yaml
   grpc:
     timeout_seconds: 60
     connect_timeout_seconds: 10
   ```
2. Check network connectivity between agent and ML server
3. Reduce batch size to decrease processing time

### TLS/SSL Errors

**Symptom**: Certificate validation errors.

**Solution**:
1. For development, use plaintext:
   ```yaml
   grpc:
     use_tls: false
   ```
2. For production, ensure certificates are valid:
   ```bash
   openssl verify -CAfile ca.crt server.crt
   ```

## Performance Issues

### Slow Log Ingestion

**Symptom**: Log processing is slower than expected.

**Diagnosis**:
1. Check batch settings:
   ```bash
   sentinel-log-ai ingest --batch-size 100 --flush-timeout 1s
   ```
2. Monitor throughput:
   ```bash
   sentinel-log-ai ingest /var/log/app.log --metrics
   ```

**Solution**:
1. Increase batch size for large files:
   ```yaml
   batch:
     size: 200
     flush_timeout: 2s
   ```
2. Use streaming mode for real-time logs:
   ```bash
   sentinel-log-ai ingest /var/log/app.log --tail
   ```

### High CPU Usage

**Symptom**: CPU usage spikes during processing.

**Diagnosis**:
```bash
# Profile Python code
python -m cProfile -o profile.stats -m sentinel_ml.server
```

**Solution**:
1. Reduce embedding batch size:
   ```yaml
   embedding:
     batch_size: 16
   ```
2. Use CPU-optimized model:
   ```yaml
   embedding:
     model_name: all-MiniLM-L6-v2
     device: cpu
   ```

### Slow Embedding Generation

**Symptom**: Embedding requests take too long.

**Solution**:
1. Enable GPU if available:
   ```yaml
   embedding:
     device: cuda  # or mps for macOS
   ```
2. Use model caching:
   ```yaml
   embedding:
     cache_size: 10000
   ```

## Memory Problems

### Out of Memory (OOM)

**Symptom**: Process killed due to memory exhaustion.

**Diagnosis**:
```bash
# Monitor memory usage
watch -n 1 'ps aux | grep sentinel'

# Use memory profiler
python -m sentinel_ml.benchmark.profiler
```

**Solution**:
1. Reduce batch sizes:
   ```yaml
   batch:
     size: 50
   embedding:
     batch_size: 8
   ```
2. Enable streaming mode instead of loading all logs:
   ```bash
   sentinel-log-ai ingest --stream /var/log/large.log
   ```
3. Limit vector store size:
   ```yaml
   vectorstore:
     max_vectors: 100000
   ```

### Memory Leak

**Symptom**: Memory usage grows continuously.

**Diagnosis**:
```python
from sentinel_ml.benchmark import MemoryProfiler

profiler = MemoryProfiler("leak_test")
profiler.set_baseline()

# Run operations
for i in range(100):
    process_logs()
    profiler.snapshot(f"iteration_{i}")

print(profiler.get_summary())
```

**Solution**:
1. Ensure proper cleanup of embedding caches
2. Clear vector store periodically:
   ```python
   vectorstore.clear()
   ```
3. Force garbage collection:
   ```python
   import gc
   gc.collect()
   ```

## Parsing Errors

### Unknown Log Format

**Symptom**: Logs not being parsed correctly.
```
warning: falling back to raw parser for line
```

**Solution**:
1. Check log format detection:
   ```bash
   sentinel-log-ai parse --debug /var/log/app.log
   ```
2. Specify format explicitly:
   ```bash
   sentinel-log-ai ingest --format json /var/log/app.log
   ```
3. Add custom parser pattern in configuration

### Timestamp Parsing Failures

**Symptom**: Timestamps not extracted correctly.

**Solution**:
1. Check supported formats in `parser.py`
2. Add custom timestamp format:
   ```yaml
   parser:
     timestamp_formats:
       - "%Y-%m-%d %H:%M:%S"
       - "%d/%b/%Y:%H:%M:%S %z"
   ```

### Character Encoding Issues

**Symptom**: Unicode errors in log parsing.

**Solution**:
1. Specify encoding:
   ```bash
   sentinel-log-ai ingest --encoding utf-8 /var/log/app.log
   ```
2. Handle binary logs:
   ```bash
   sentinel-log-ai ingest --binary-safe /var/log/binary.log
   ```

## ML Engine Issues

### Model Loading Failures

**Symptom**: Embedding model fails to load.
```
error: could not load model all-MiniLM-L6-v2
```

**Solution**:
1. Check model availability:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   ```
2. Download model manually:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```
3. Use offline model path:
   ```yaml
   embedding:
     model_name: /path/to/local/model
   ```

### FAISS Index Errors

**Symptom**: Vector store operations fail.

**Solution**:
1. Rebuild index:
   ```python
   vectorstore.rebuild_index()
   ```
2. Check dimension compatibility:
   ```python
   assert embedding_dim == vectorstore.dimension
   ```
3. Clear corrupted index:
   ```bash
   rm -rf .sentinel/vectorstore/*
   ```

### Clustering Failures

**Symptom**: HDBSCAN produces no clusters.

**Solution**:
1. Adjust clustering parameters:
   ```yaml
   clustering:
     min_cluster_size: 3
     min_samples: 2
   ```
2. Increase sample size (need more logs for clustering)
3. Check embedding quality

## LLM Integration

### Ollama Connection Failed

**Symptom**: Cannot connect to Ollama.
```
error: could not connect to Ollama at localhost:11434
```

**Solution**:
1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```
2. Check Ollama status:
   ```bash
   curl http://localhost:11434/api/tags
   ```
3. Pull required model:
   ```bash
   ollama pull llama3.2
   ```

### LLM Response Timeout

**Symptom**: LLM requests timeout.

**Solution**:
1. Increase timeout:
   ```yaml
   llm:
     timeout_seconds: 120
   ```
2. Use smaller model:
   ```yaml
   llm:
     model: phi3
   ```
3. Reduce context size in prompts

### Invalid LLM Response

**Symptom**: LLM returns unparseable response.

**Solution**:
1. Enable response validation:
   ```yaml
   llm:
     validate_response: true
   ```
2. Use structured output format
3. Add retry logic with different prompts

## Logging and Debugging

### Enable Debug Logging

```yaml
logging:
  level: DEBUG
  format: json
```

Or via environment:
```bash
export SENTINEL_ML_LOGGING__LEVEL=DEBUG
```

### View JSONL Logs

```bash
# Pretty print logs
cat logs/sentinel-ml.jsonl | jq

# Filter by level
cat logs/sentinel-ml.jsonl | jq 'select(.level == "error")'

# Filter by time range
cat logs/sentinel-ml.jsonl | jq 'select(.timestamp >= "2024-01-01")'
```

### Enable gRPC Tracing

```bash
export GRPC_TRACE=all
export GRPC_VERBOSITY=DEBUG
```

### Profile Code Execution

```python
from sentinel_ml.benchmark import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(
    name="my_operation",
    iterations=10,
    collect_memory=True,
)

class MyBenchmark(BenchmarkRunner):
    def setup(self):
        pass

    def run_iteration(self, i):
        my_operation()

    def teardown(self):
        pass

benchmark = MyBenchmark(config)
result = benchmark.execute()
print(result.to_dict())
```

## Getting Help

If you cannot resolve your issue:

1. Check the [GitHub Issues](https://github.com/varadharajaan/sentinel-log-ai/issues)
2. Search existing discussions
3. Create a new issue with:
   - Python/Go version
   - OS and version
   - Full error message
   - Minimal reproduction steps
   - Relevant configuration (sanitized)
