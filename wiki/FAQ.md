# Frequently Asked Questions

## General

### What is Sentinel Log AI?

Sentinel Log AI is an AI-powered log intelligence engine that helps on-call engineers understand their logs. It:
- **Clusters** similar log patterns automatically
- **Detects** novel/unseen error patterns
- **Explains** issues using LLM-powered analysis
- Runs **locally** on your machine

### Why Go + Python?

We chose a polyglot architecture to leverage the best of both worlds:

| Component | Language | Why |
|-----------|----------|-----|
| Log Ingestion | Go | Single binary, low memory, excellent I/O |
| ML Processing | Python | Rich ecosystem (transformers, FAISS, scikit-learn) |

Communication via gRPC provides efficient, type-safe IPC.

### Is my data sent to the cloud?

**No.** Sentinel Log AI is designed to be local-first:
- Embeddings are generated locally using sentence-transformers
- LLM inference uses local Ollama by default
- Your logs never leave your infrastructure

You *can* optionally configure OpenAI for LLM, but this is opt-in.

---

## Installation

### What are the system requirements?

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Disk | 1 GB | 10+ GB |
| Python | 3.10+ | 3.11+ |
| Go | 1.22+ | Latest |

For LLM explanations via Ollama, add:
- Additional 4-8 GB RAM for model
- GPU recommended but not required

### How do I install on Windows?

```powershell
# Clone repository
git clone https://github.com/varadharajaan/sentinel-log-ai.git
cd sentinel-log-ai

# Setup Python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Build Go agent
go build -o bin\sentinel-agent.exe .\cmd\agent
```

### How do I install Ollama?

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
```

**Windows:**
Download from [ollama.ai](https://ollama.ai/download) and install.

Then pull a model:
```powershell
ollama pull llama3.2
```

### I get "model not found" errors

Ensure you've pulled the model:
```bash
ollama list           # Check available models
ollama pull llama3.2  # Pull the default model
```

---

## Usage

### How do I analyze a log file?

```bash
# Start the ML server
python -m sentinel_ml.server

# In another terminal, run the agent
./bin/sentinel-agent ingest --file /path/to/logs.log
```

### Can I pipe logs directly?

Yes! Use stdin mode:
```bash
cat /var/log/syslog | ./bin/sentinel-agent ingest --stdin
tail -f /var/log/app.log | ./bin/sentinel-agent ingest --stdin
```

### What log formats are supported?

Auto-detected formats:
- **JSON** logs (structured)
- **Syslog** (RFC 3164/5424)
- **Nginx** access/error logs
- **Python tracebacks**
- **Common Log Format**

Unknown formats are parsed as plain text.

### How do I export results?

**Markdown report:**
```python
from sentinel_ml.cli import MarkdownReporter, ReportData
reporter = MarkdownReporter()
reporter.save(data, Path("report.md"))
```

**HTML report:**
```python
from sentinel_ml.cli import HTMLReporter
reporter = HTMLReporter()
reporter.save(data, Path("report.html"))
```

**JSON output:**
```bash
./bin/sentinel-agent ingest --file logs.log --format json > results.json
```

---

## Configuration

### Where is the config file?

Sentinel Log AI looks for configuration in:
1. `./config.yaml` (current directory)
2. `./sentinel.yaml`
3. `~/.sentinel/config.yaml`
4. `/etc/sentinel/config.yaml`

### How do I generate a default config?

```python
from sentinel_ml.cli import generate_config
generate_config(output=Path("config.yaml"))
```

### How do I change the embedding model?

In `config.yaml`:
```yaml
embedding:
  model_name: "all-mpnet-base-v2"  # Higher quality, slower
```

Available models:
- `all-MiniLM-L6-v2` (default, fast)
- `all-mpnet-base-v2` (higher quality)
- `paraphrase-MiniLM-L6-v2` (paraphrase focused)

### How do I use OpenAI instead of Ollama?

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  # Set via environment variable for security
  # api_key: "${OPENAI_API_KEY}"
```

Then set the environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

---

## Performance

### How much memory does it use?

| Component | Memory |
|-----------|--------|
| Go Agent | ~50 MB |
| Python ML Engine | ~500 MB |
| Embedding Model | ~200 MB |
| Ollama (llama3.2) | ~4 GB |

### How can I reduce memory usage?

1. **Use smaller embedding model**:
   ```yaml
   embedding:
     model_name: "all-MiniLM-L6-v2"  # 90 MB vs 420 MB
   ```

2. **Reduce batch size**:
   ```yaml
   embedding:
     batch_size: 16  # Default is 32
   ```

3. **Use flat index for small datasets**:
   ```yaml
   vector_store:
     index_type: "flat"
   ```

4. **Disable embedding cache**:
   ```yaml
   embedding:
     cache_enabled: false
   ```

### How fast is log processing?

Typical throughput on modern hardware:
- **Ingestion**: 10,000+ logs/second
- **Embedding**: 1,000+ logs/second (batched)
- **Clustering**: 10,000 logs in ~1 second
- **LLM Explanation**: 2-5 seconds per cluster

### Can I process logs in parallel?

Yes, the Go agent uses goroutines for concurrent processing. The Python ML engine uses a thread pool for parallelism.

---

## Troubleshooting

### "Connection refused" when connecting to ML server

1. Ensure the server is running:
   ```bash
   python -m sentinel_ml.server
   ```

2. Check the port:
   ```bash
   netstat -an | grep 50051  # Linux/macOS
   netstat -an | findstr 50051  # Windows
   ```

3. Check firewall settings

### Clustering produces too many small clusters

Adjust HDBSCAN parameters:
```yaml
clustering:
  min_cluster_size: 10  # Increase from default 5
  min_samples: 5        # Increase from default 3
```

### All logs are marked as novel

This usually means the reference dataset is too small. Ensure you:
1. Have at least 50+ logs for the reference set
2. The reference logs represent normal patterns

```yaml
novelty:
  min_samples_fit: 50  # Minimum samples before fitting
  threshold: 0.8       # Increase threshold (less sensitive)
```

### LLM explanations are slow

1. **Use a smaller model**:
   ```bash
   ollama pull llama3.2:3b  # 3B instead of default
   ```

2. **Increase timeout**:
   ```yaml
   llm:
     timeout: 180  # Default is 120
   ```

3. **Use GPU** (if available):
   ```bash
   # Ollama auto-detects GPU
   nvidia-smi  # Check GPU availability
   ```

### Logs aren't being parsed correctly

1. Check the format is supported
2. Try explicit format:
   ```bash
   ./bin/sentinel-agent ingest --file logs.log --parser json
   ```

3. Check for encoding issues (ensure UTF-8)

---

## Development

### How do I run tests?

```bash
# All tests
make test

# Python only
pytest tests/python/ -v

# Go only
go test ./... -v

# With coverage
pytest --cov=sentinel_ml --cov-report=html
```

### How do I add a new log parser?

1. Create parser in `internal/parser/`
2. Implement the `Parser` interface
3. Register in `parser_registry.go`
4. Add tests

See [[Development Setup]] for details.

### How do I add a new LLM provider?

1. Create provider in `python/sentinel_ml/llm/`
2. Implement the `LLMProvider` protocol
3. Add to factory in `llm.py`
4. Add tests

---

## Contributing

### How can I contribute?

See our [[Contributing]] guide! We welcome:
- Bug reports and fixes
- Feature suggestions and implementations
- Documentation improvements
- Test coverage improvements

### What's the coding style?

- **Python**: Black + Ruff, Google docstrings
- **Go**: gofmt + golangci-lint

Run `make lint` to check and `make format` to auto-fix.

---

*Have a question not covered here? [Open a Discussion](https://github.com/varadharajaan/sentinel-log-ai/discussions)!*
