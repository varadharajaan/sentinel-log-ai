# Demo Dataset and Walkthrough

This guide demonstrates Sentinel Log AI capabilities using sample datasets.

## Table of Contents

- [🎯 ML vs Regex Demo](#-ml-vs-regex-demo)
- [Quick Demo](#quick-demo)
- [Generating Demo Data](#generating-demo-data)
- [Step-by-Step Walkthrough](#step-by-step-walkthrough)
- [Interpreting Results](#interpreting-results)

---

## 🎯 ML vs Regex Demo

**The definitive demo showing why ML detection beats regex for security logs.**

### Why This Demo Matters

When teams ask "Why can't we just use regex?", this demo provides the answer with real production log patterns.

### The Core Problem

| Approach | How It Works | Limitation |
|----------|--------------|------------|
| **Regex** | Pattern match on TEXT characters | Only catches attacks you've already seen |
| **ML** | Semantic match on MEANING | Catches attacks that *mean* the same thing |

### Run the Demo

```bash
cd demo
python demo_ml_vs_regex.py
```

### What the Demo Shows

1. **Baseline Learning** - ML builds understanding of "normal" from 75 production logs
2. **Known Attack Detection** - Both regex and ML catch 4/4 SQL injection, XSS, path traversal
3. **Novel Attack Detection** - The key difference:

| Attack Type | Regex | ML |
|-------------|-------|-----|
| Supply Chain (malicious package) | ❌ | ✅ |
| Credential Stuffing (distributed) | ❌ | ✅ |
| Container Escape (Docker socket) | ❌ | ✅ |
| SSRF (internal service probe) | ❌ | ✅ |
| Privilege Escalation (sudo/setuid) | ❌ | ✅ |
| DNS Exfiltration (encoded data) | ❌ | ✅ |

### The Demo Dataset

Located in `demo/demo_logs.jsonl` - 96 real-world production log patterns:

- **75 Normal Logs**: Kubernetes pods, PostgreSQL queries, Kafka consumers, Redis cache, OAuth tokens, GraphQL operations, cron jobs, gRPC calls, nginx access, Prometheus metrics
- **4 Known Attacks**: SQL injection, XSS, path traversal (regex catches these)
- **17 Novel Attacks**: Supply chain, credential stuffing, container escape, SSRF, privilege escalation, DNS exfiltration (regex misses ALL of these)

### Key Insight

> **Regex catches 4/4 known attacks but 0/17 novel attacks (0%)**  
> **ML catches 4/4 known attacks AND 14/17 novel attacks (82%)**

Attackers don't reuse the same patterns. They innovate. Your detection must too.

---

## Quick Demo

Run a quick demonstration with synthetic logs:

```bash
# Generate demo dataset
python -c "
from sentinel_ml.benchmark import generate_test_logs
logs = generate_test_logs(1000, seed=42)
with open('demo_logs.jsonl', 'w') as f:
    for log in logs:
        f.write(log.to_json_line() + '\n')
print(f'Generated {len(logs)} demo logs')
"

# Start the ML server
make run-ml &

# Ingest the demo logs
./bin/sentinel-log-ai ingest demo_logs.jsonl

# Analyze patterns
./bin/sentinel-log-ai analyze --top 10

# Detect novelties
./bin/sentinel-log-ai novel --threshold 0.7
```

## Generating Demo Data

### Using the Dataset Generator

```python
from sentinel_ml.benchmark import (
    DatasetGenerator,
    DatasetConfig,
    LogPattern,
    LogLevel,
)

# Create custom dataset configuration
config = DatasetConfig(
    name="production_simulation",
    size=10000,
    patterns={
        LogPattern.HTTP_REQUEST: 0.5,    # 50% HTTP requests
        LogPattern.DATABASE_QUERY: 0.2,  # 20% DB queries
        LogPattern.AUTH_EVENT: 0.15,     # 15% auth events
        LogPattern.ERROR_STACK: 0.1,     # 10% errors
        LogPattern.SYSTEM_EVENT: 0.05,   # 5% system events
    },
    level_distribution={
        LogLevel.DEBUG: 0.05,
        LogLevel.INFO: 0.60,
        LogLevel.WARNING: 0.20,
        LogLevel.ERROR: 0.12,
        LogLevel.CRITICAL: 0.03,
    },
    time_range_hours=24,
    seed=42,  # For reproducibility
)

# Generate logs
generator = DatasetGenerator(config)
logs = generator.generate_all()

# Save to file
with open("production_sim.jsonl", "w") as f:
    for log in logs:
        f.write(log.to_json_line() + "\n")

print(f"Generated {len(logs)} logs")
```

### Scale Testing Datasets

```python
from sentinel_ml.benchmark import create_scale_datasets, DatasetGenerator

# Get predefined scale configurations
configs = create_scale_datasets()

for name, config in configs.items():
    print(f"{name}: {config.size:,} logs")
    
    # Generate small dataset for demo
    if name == "small":
        generator = DatasetGenerator(config)
        logs = generator.generate_all()
        
        with open(f"demo_{name}.jsonl", "w") as f:
            for log in logs:
                f.write(log.to_json_line() + "\n")
```

### Streaming Generation for Large Datasets

```python
from sentinel_ml.benchmark import DatasetGenerator, DatasetConfig

config = DatasetConfig(name="large_demo", size=100000, seed=42)
generator = DatasetGenerator(config)

# Stream logs to file without loading all into memory
with open("large_demo.jsonl", "w") as f:
    for log in generator.generate_iter(batch_size=1000):
        f.write(log.to_json_line() + "\n")
```

## Step-by-Step Walkthrough

### Step 1: Environment Setup

```bash
# Install dependencies
make install

# Build the Go agent
make build-go

# Verify installation
./bin/sentinel-log-ai --version
python -c "import sentinel_ml; print('ML engine ready')"
```

### Step 2: Generate Sample Data

```bash
# Create demo directory
mkdir -p demo

# Generate varied log samples
python << 'EOF'
from sentinel_ml.benchmark import generate_test_logs

# Standard logs
logs = generate_test_logs(500, name="standard", seed=1)
with open("demo/standard.jsonl", "w") as f:
    for log in logs:
        f.write(log.to_json_line() + "\n")

# Error-heavy logs  
from sentinel_ml.benchmark import DatasetConfig, DatasetGenerator, LogLevel

error_config = DatasetConfig(
    name="errors",
    size=200,
    level_distribution={
        LogLevel.ERROR: 0.7,
        LogLevel.CRITICAL: 0.2,
        LogLevel.WARNING: 0.1,
    },
    seed=2,
)
gen = DatasetGenerator(error_config)
with open("demo/errors.jsonl", "w") as f:
    for log in gen.generate_all():
        f.write(log.to_json_line() + "\n")

print("Demo data generated in demo/")
EOF
```

### Step 3: Start the ML Server

```bash
# Terminal 1: Start the server
make run-ml

# Wait for "Server started on port 50051"
```

### Step 4: Ingest Logs

```bash
# Terminal 2: Ingest standard logs
./bin/sentinel-log-ai ingest demo/standard.jsonl

# Ingest error logs
./bin/sentinel-log-ai ingest demo/errors.jsonl

# Ingest with verbose output
./bin/sentinel-log-ai ingest demo/ --pattern "*.jsonl" -v
```

### Step 5: Analyze Patterns

```bash
# View top clusters
./bin/sentinel-log-ai analyze --top 5

# Output format: JSON
./bin/sentinel-log-ai analyze --top 5 --format json

# Filter by time
./bin/sentinel-log-ai analyze --last 1h
```

### Step 6: Detect Novelties

```bash
# Find novel patterns
./bin/sentinel-log-ai novel

# With custom threshold
./bin/sentinel-log-ai novel --threshold 0.8

# Continuous monitoring
./bin/sentinel-log-ai novel --follow --threshold 0.7
```

### Step 7: Get LLM Explanations

```bash
# Ensure Ollama is running
ollama serve &
ollama pull llama3.2

# Explain a cluster
./bin/sentinel-log-ai explain cluster-001

# Explain novel patterns
./bin/sentinel-log-ai explain --novel
```

## Interpreting Results

### Cluster Analysis Output

```
Cluster: cluster-001
  Size: 127 logs
  Representative: "GET /api/v1/users 200 45ms - 192.168.1.100"
  Common Level: INFO
  Time Range: 2024-01-01 10:00:00 - 2024-01-01 11:30:00
  
  Interpretation:
  - This cluster represents successful HTTP GET requests to the users API
  - Response times are consistently low (45ms average)
  - All requests from the same IP range suggest a single client or load balancer
```

### Novelty Detection Output

```
Novel Patterns Found: 3

1. Score: 0.89 (High)
   Message: "DatabaseError: Connection pool exhausted after 30s timeout"
   Explanation: This error pattern has not been seen before.
   Recommended Action: Check database connection pool configuration.

2. Score: 0.76 (Medium)
   Message: "AuthenticationError: Invalid token signature for user=admin"
   Explanation: Unusual authentication failure pattern.
   Recommended Action: Investigate potential security incident.

3. Score: 0.71 (Medium)
   Message: "GET /api/v1/internal/debug 401 - 10.0.0.50"
   Explanation: Access attempt to internal endpoint from unusual IP.
   Recommended Action: Review access control policies.
```

### LLM Explanation Output

```
Cluster Analysis: cluster-002
---------------------------------
Pattern Type: Database Connection Errors
Severity: HIGH
Confidence: 0.87

Root Cause Analysis:
The logs indicate a pattern of database connection failures occurring
during peak traffic hours. The connection pool is becoming exhausted
due to slow query execution times.

Suggested Actions:
1. Increase connection pool size from 10 to 25
2. Add query timeout of 5 seconds for long-running queries
3. Implement connection retry with exponential backoff
4. Review indexes on frequently queried tables

Related Patterns:
- cluster-005: Slow database queries (may be related)
- cluster-008: Application timeout errors (downstream effect)
```

## Benchmark the Demo

Run performance benchmarks on the demo data:

```python
from sentinel_ml.benchmark import (
    BenchmarkSuite,
    BenchmarkConfig,
    FunctionBenchmark,
    DatasetGenerator,
    DatasetConfig,
)

# Create benchmark suite
suite = BenchmarkSuite("demo_benchmark")

# Benchmark dataset generation
def generate_logs():
    config = DatasetConfig(name="bench", size=1000, seed=42)
    gen = DatasetGenerator(config)
    return gen.generate_all()

suite.add_function(generate_logs, name="log_generation", iterations=5)

# Run benchmarks
results = suite.run_all()

# Print summary
for result in results:
    print(f"{result.config.name}: {result.duration_seconds:.3f}s")
    if "timing" in result.metrics:
        timing = result.metrics["timing"]
        for op, stats in timing.items():
            print(f"  {op}: mean={stats['mean_seconds']:.4f}s")
```

## Next Steps

1. Try with your own log files
2. Customize the dataset patterns to match your production logs
3. Tune clustering and novelty detection parameters
4. Set up continuous monitoring with `--follow` mode
5. Integrate with your alerting system
