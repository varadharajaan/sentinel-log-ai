# Configuration Reference

Complete reference for all configuration options in Sentinel Log AI.

## Configuration File

By default, Sentinel Log AI looks for configuration in:
1. `./config.yaml` (current directory)
2. `./sentinel.yaml`
3. `~/.sentinel/config.yaml`
4. `/etc/sentinel/config.yaml`

## Full Configuration Example

```yaml
# =============================================================================
# Sentinel Log AI Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Server Configuration
# -----------------------------------------------------------------------------
server:
  # gRPC server host
  host: "0.0.0.0"
  
  # gRPC server port
  port: 50051
  
  # Maximum concurrent workers
  max_workers: 10
  
  # Request timeout in seconds
  timeout: 30.0
  
  # Enable reflection for grpcurl/debugging
  reflection: true

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging:
  # Log level: DEBUG, INFO, WARNING, ERROR
  level: "INFO"
  
  # Output format: json, text
  format: "json"
  
  # Log file path (optional, logs to stderr if not set)
  file: "logs/sentinel-ml.jsonl"
  
  # Enable structured logging
  structured: true
  
  # Log rotation settings
  max_size_mb: 100
  max_backups: 5
  max_age_days: 30

# -----------------------------------------------------------------------------
# Embedding Configuration
# -----------------------------------------------------------------------------
embedding:
  # Model name from sentence-transformers
  # Options: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2
  model_name: "all-MiniLM-L6-v2"
  
  # Batch size for embedding generation
  batch_size: 32
  
  # Enable caching for repeated embeddings
  cache_enabled: true
  cache_size: 10000
  
  # Cache directory
  cache_dir: ".cache/embeddings"
  
  # Device: cpu, cuda, mps (auto-detected if not set)
  device: "cpu"
  
  # Use mock embeddings (for testing)
  use_mock: false

# -----------------------------------------------------------------------------
# Vector Store Configuration
# -----------------------------------------------------------------------------
vector_store:
  # Index type:
  # - flat: Exact search, best for small datasets
  # - ivf: Inverted file index, good balance
  # - hnsw: Hierarchical NSW, fast approximate search
  index_type: "flat"
  
  # Embedding dimension (must match embedding model)
  dimension: 384
  
  # Metric: cosine, l2, inner_product
  metric: "cosine"
  
  # IVF-specific settings
  ivf_nlist: 100
  ivf_nprobe: 10
  
  # HNSW-specific settings
  hnsw_m: 32
  hnsw_ef_construction: 200
  hnsw_ef_search: 64
  
  # Persistence path (optional)
  persist_path: "data/vector_store"
  
  # Use mock store (for testing)
  use_mock: false

# -----------------------------------------------------------------------------
# Clustering Configuration
# -----------------------------------------------------------------------------
clustering:
  # Algorithm: hdbscan, kmeans, dbscan
  algorithm: "hdbscan"
  
  # HDBSCAN parameters
  min_cluster_size: 5
  min_samples: 3
  cluster_selection_epsilon: 0.0
  cluster_selection_method: "eom"  # eom or leaf
  
  # K-Means parameters (if algorithm: kmeans)
  n_clusters: 10
  
  # DBSCAN parameters (if algorithm: dbscan)
  eps: 0.5
  
  # Metric for distance calculation
  metric: "euclidean"
  
  # Use mock clustering (for testing)
  use_mock: false

# -----------------------------------------------------------------------------
# Novelty Detection Configuration
# -----------------------------------------------------------------------------
novelty:
  # Algorithm: knn, lof, isolation_forest
  algorithm: "knn"
  
  # Number of neighbors for k-NN
  k_neighbors: 5
  
  # Threshold for novelty classification (0.0 - 1.0)
  threshold: 0.7
  
  # Minimum samples to fit detector
  min_samples_fit: 10
  
  # LOF-specific (if algorithm: lof)
  lof_contamination: 0.1
  
  # Isolation Forest (if algorithm: isolation_forest)
  if_n_estimators: 100
  if_contamination: 0.1
  
  # Use mock detector (for testing)
  use_mock: false

# -----------------------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------------------
llm:
  # Provider: ollama, openai, mock
  provider: "ollama"
  
  # Model name
  model: "llama3.2"
  
  # Base URL for Ollama
  base_url: "http://localhost:11434"
  
  # OpenAI API key (if provider: openai)
  # api_key: "${OPENAI_API_KEY}"
  
  # Request timeout in seconds
  timeout: 120
  
  # Maximum retries on failure
  max_retries: 3
  
  # Temperature for generation (0.0 - 1.0)
  temperature: 0.1
  
  # Use mock LLM (for testing)
  use_mock: false

# -----------------------------------------------------------------------------
# CLI Configuration
# -----------------------------------------------------------------------------
cli:
  # Theme: dark, light, minimal, colorblind, none
  theme: "dark"
  
  # Output format: text, json, table
  output_format: "text"
  
  # Enable colors (auto-detected if terminal supports)
  colors: true
  
  # Maximum output width
  max_width: 120
  
  # Show timestamps in output
  timestamps: true
  
  # Verbose output
  verbose: false

# -----------------------------------------------------------------------------
# Profiling Configuration
# -----------------------------------------------------------------------------
profiling:
  # Enable profiling
  enabled: false
  
  # Minimum duration to report (ms)
  threshold_ms: 1.0
  
  # Output file (optional)
  output_file: "profile.json"
```

---

## Environment Variables

Configuration values can be overridden with environment variables:

| Environment Variable | Config Path | Default |
|---------------------|-------------|---------|
| `SENTINEL_SERVER_HOST` | `server.host` | `0.0.0.0` |
| `SENTINEL_SERVER_PORT` | `server.port` | `50051` |
| `SENTINEL_LOG_LEVEL` | `logging.level` | `INFO` |
| `SENTINEL_EMBEDDING_MODEL` | `embedding.model_name` | `all-MiniLM-L6-v2` |
| `SENTINEL_LLM_PROVIDER` | `llm.provider` | `ollama` |
| `SENTINEL_LLM_MODEL` | `llm.model` | `llama3.2` |
| `OPENAI_API_KEY` | `llm.api_key` | - |
| `OLLAMA_HOST` | `llm.base_url` | `http://localhost:11434` |

```bash
# Example
export SENTINEL_LOG_LEVEL=DEBUG
export SENTINEL_LLM_MODEL=llama3.3
python -m sentinel_ml.server
```

---

## Configuration Sections

### Server

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `host` | string | `0.0.0.0` | gRPC server bind address |
| `port` | int | `50051` | gRPC server port |
| `max_workers` | int | `10` | Maximum concurrent gRPC workers |
| `timeout` | float | `30.0` | Request timeout in seconds |
| `reflection` | bool | `true` | Enable gRPC reflection |

### Logging

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `level` | string | `INFO` | Log level (DEBUG/INFO/WARNING/ERROR) |
| `format` | string | `json` | Output format (json/text) |
| `file` | string | - | Log file path |
| `structured` | bool | `true` | Enable structured logging |
| `max_size_mb` | int | `100` | Max log file size before rotation |
| `max_backups` | int | `5` | Number of backup files to keep |
| `max_age_days` | int | `30` | Max age of backup files |

### Embedding

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_name` | string | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `batch_size` | int | `32` | Batch size for embedding |
| `cache_enabled` | bool | `true` | Enable embedding cache |
| `cache_size` | int | `10000` | Max cached embeddings |
| `cache_dir` | string | `.cache/embeddings` | Cache directory |
| `device` | string | `cpu` | Device (cpu/cuda/mps) |
| `use_mock` | bool | `false` | Use mock provider |

### Vector Store

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `index_type` | string | `flat` | Index type (flat/ivf/hnsw) |
| `dimension` | int | `384` | Embedding dimension |
| `metric` | string | `cosine` | Distance metric |
| `ivf_nlist` | int | `100` | IVF cluster count |
| `ivf_nprobe` | int | `10` | IVF clusters to search |
| `hnsw_m` | int | `32` | HNSW max connections |
| `hnsw_ef_construction` | int | `200` | HNSW build quality |
| `hnsw_ef_search` | int | `64` | HNSW search quality |
| `persist_path` | string | - | Persistence directory |

### Clustering

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `algorithm` | string | `hdbscan` | Algorithm (hdbscan/kmeans/dbscan) |
| `min_cluster_size` | int | `5` | HDBSCAN min cluster size |
| `min_samples` | int | `3` | HDBSCAN core point threshold |
| `cluster_selection_epsilon` | float | `0.0` | HDBSCAN merge threshold |
| `cluster_selection_method` | string | `eom` | Selection method (eom/leaf) |
| `n_clusters` | int | `10` | K-Means cluster count |
| `eps` | float | `0.5` | DBSCAN epsilon |
| `metric` | string | `euclidean` | Distance metric |

### Novelty

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `algorithm` | string | `knn` | Algorithm (knn/lof/isolation_forest) |
| `k_neighbors` | int | `5` | k-NN neighbor count |
| `threshold` | float | `0.7` | Novelty classification threshold |
| `min_samples_fit` | int | `10` | Min samples to train |
| `lof_contamination` | float | `0.1` | LOF contamination rate |
| `if_n_estimators` | int | `100` | Isolation Forest trees |
| `if_contamination` | float | `0.1` | IF contamination rate |

### LLM

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | string | `ollama` | Provider (ollama/openai/mock) |
| `model` | string | `llama3.2` | Model name |
| `base_url` | string | `http://localhost:11434` | Ollama base URL |
| `api_key` | string | - | OpenAI API key |
| `timeout` | int | `120` | Request timeout |
| `max_retries` | int | `3` | Max retry attempts |
| `temperature` | float | `0.1` | Generation temperature |

### CLI

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `theme` | string | `dark` | Color theme |
| `output_format` | string | `text` | Default output format |
| `colors` | bool | `true` | Enable colors |
| `max_width` | int | `120` | Max output width |
| `timestamps` | bool | `true` | Show timestamps |
| `verbose` | bool | `false` | Verbose output |

---

## Model Recommendations

### Embedding Models

| Model | Dimension | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Default, balanced |
| `all-mpnet-base-v2` | 768 | Medium | Best | High accuracy |
| `paraphrase-MiniLM-L6-v2` | 384 | Fast | Good | Paraphrase detection |

### LLM Models (Ollama)

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `llama3.2` | 3B | Fast | Good | Default, quick analysis |
| `llama3.3` | 8B | Medium | Better | More detailed explanations |
| `mistral` | 7B | Medium | Good | Alternative to Llama |
| `codellama` | 7B | Medium | Good | Code-heavy logs |

---

## Programmatic Configuration

```python
from sentinel_ml.config import (
    Config,
    ServerConfig,
    EmbeddingConfig,
    ClusteringConfig,
    NoveltyConfig,
    LLMConfig,
)

# Create config programmatically
config = Config(
    server=ServerConfig(port=50052),
    embedding=EmbeddingConfig(
        model_name="all-mpnet-base-v2",
        batch_size=64,
    ),
    clustering=ClusteringConfig(
        min_cluster_size=10,
    ),
    novelty=NoveltyConfig(
        threshold=0.8,
    ),
    llm=LLMConfig(
        provider="ollama",
        model="llama3.3",
    ),
)

# Load from file
from sentinel_ml.cli import load_config
config = load_config(Path("config.yaml"))

# Validate config
from sentinel_ml.cli import validate_config
is_valid, errors = validate_config(Path("config.yaml"))
```

---

*See also: [[Quick Start|Quick-Start]], [[CLI UX|CLI-UX]]*
