# CLI & UX Guide

The CLI module provides a rich, themeable command-line interface for interacting with Sentinel Log AI.

## Overview

```
┌────────────────────────────────────────────────────────────────┐
│                         CLI MODULE                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     Console (Facade)                      │  │
│  │  Unified interface for all CLI output operations          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Themes   │     │  Formatters │     │   Progress  │       │
│  │  (Strategy)│     │  (Strategy) │     │  (Observer) │       │
│  └────────────┘     └─────────────┘     └─────────────┘       │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  Profiler  │     │   Reports   │     │   Config    │       │
│  │ (Decorator)│     │  (Template) │     │  Commands   │       │
│  └────────────┘     └─────────────┘     └─────────────┘       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Console

The `Console` class is the main entry point for all CLI operations.

### Basic Usage

```python
from sentinel_ml.cli import Console, ConsoleConfig, Theme, OutputFormat

# Create console with default settings
console = Console()

# Or with custom configuration
config = ConsoleConfig(
    theme=Theme.DARK,
    format=OutputFormat.TEXT,
    colors=True,
    verbose=True
)
console = Console(config)

# Output methods
console.info("Processing logs...")
console.success("Analysis complete!")
console.warning("Some logs could not be parsed")
console.error("Connection failed")
console.debug("Verbose debug info")  # Only shown if verbose=True
```

### Output Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `TEXT` | Human-readable colored output | Interactive terminal |
| `JSON` | Machine-readable JSON | Scripting, piping |
| `TABLE` | Rich formatted tables | Data presentation |
| `COMPACT` | Minimal one-line output | Log aggregation |

```python
# Print structured data
console.print_json({"clusters": 8, "novel": 3})

# Print tables
from sentinel_ml.cli import TableData, TableColumn

columns = [
    TableColumn(name="ID", key="id", width=10),
    TableColumn(name="Size", key="size", align="right"),
    TableColumn(name="Level", key="level"),
]
rows = [
    {"id": "cluster-1", "size": 423, "level": "INFO"},
    {"id": "cluster-2", "size": 156, "level": "ERROR"},
]
console.print_table(TableData(columns=columns, rows=rows))
```

Output:
```
┌────────────┬──────┬───────┐
│ ID         │ Size │ Level │
├────────────┼──────┼───────┤
│ cluster-1  │  423 │ INFO  │
│ cluster-2  │  156 │ ERROR │
└────────────┴──────┴───────┘
```

---

## Themes

Five built-in themes provide accessibility and preference options.

### Available Themes

| Theme | Description | Best For |
|-------|-------------|----------|
| `DARK` | Cyan/green on dark background | Dark terminals (default) |
| `LIGHT` | Blue/dark colors | Light terminal backgrounds |
| `MINIMAL` | Muted color palette | Reduced visual noise |
| `COLORBLIND` | High-contrast, deuteranopia-safe | Color vision deficiency |
| `NONE` | No colors, plain text | Logs, non-TTY output |

### Theme Colors

```python
from sentinel_ml.cli import Theme, get_theme

colors = get_theme(Theme.DARK)
print(colors.primary)     # "cyan"
print(colors.success)     # "green"
print(colors.warning)     # "yellow"
print(colors.error)       # "red"
print(colors.info)        # "blue"
```

### Semantic Color Functions

```python
from sentinel_ml.cli import (
    get_severity_color,
    get_novelty_color,
    get_confidence_color,
    get_log_level_color,
)

# Colors based on severity (0.0 - 1.0)
color = get_severity_color(0.9)  # "red" (high severity)
color = get_severity_color(0.3)  # "green" (low severity)

# Colors based on novelty score
color = get_novelty_color(0.85)  # "red" (highly novel)

# Colors based on confidence
color = get_confidence_color(0.95)  # "green" (high confidence)

# Colors for log levels
color = get_log_level_color("ERROR")  # "red"
color = get_log_level_color("INFO")   # "blue"
```

---

## Formatters

Strategy pattern implementations for different output types.

### JSON Formatter

```python
from sentinel_ml.cli import JSONFormatter, FormatOptions

formatter = JSONFormatter()
output = formatter.format(
    data={"clusters": clusters, "novel": novel_scores},
    options=FormatOptions(indent=2, sort_keys=True)
)
print(output)
```

### Table Formatter

```python
from sentinel_ml.cli import TableFormatter, TableColumn, TableData

formatter = TableFormatter()

# Define columns with formatting options
columns = [
    TableColumn(name="Cluster", key="id", width=15),
    TableColumn(name="Size", key="size", align="right", format="{:,}"),
    TableColumn(name="Cohesion", key="cohesion", format="{:.3f}"),
]

data = TableData(columns=columns, rows=cluster_data)
output = formatter.format(data)
```

### Cluster Formatter

```python
from sentinel_ml.cli import ClusterFormatter

formatter = ClusterFormatter()
output = formatter.format(clusters, options=FormatOptions(verbose=True))
```

Output:
```
📊 CLUSTER abc123def456 (423 logs)
├── Cohesion: 0.923
├── Level: INFO
├── Time Range: 2026-01-07 10:00 - 10:45
├── Representative: "User login successful from {IP}"
└── Tags: authentication, success
```

### Novelty Formatter

```python
from sentinel_ml.cli import NoveltyFormatter

formatter = NoveltyFormatter()
output = formatter.format(novel_scores)
```

Output:
```
⚠️ NOVEL PATTERN (score: 0.89, HIGH)
├── Message: "Database connection pool exhausted..."
├── Distance: 2.341
└── Explanation: Significantly different from known patterns
```

---

## Progress Tracking

Track long-running operations with spinners and progress bars.

### Spinner

```python
from sentinel_ml.cli import spinner

# Simple spinner context
with spinner("Processing logs..."):
    process_logs()

# With status updates
with spinner("Analyzing") as spin:
    for i, batch in enumerate(batches):
        spin.update(f"Batch {i+1}/{len(batches)}")
        process_batch(batch)
```

Output:
```
⠋ Processing logs...
⠙ Analyzing - Batch 5/10
✓ Complete
```

### Progress Bar

```python
from sentinel_ml.cli import ProgressTracker

tracker = ProgressTracker(total=1000, description="Embedding logs")

with tracker:
    for log in logs:
        embed(log)
        tracker.advance()
```

Output:
```
Embedding logs ━━━━━━━━━━━━━━━━━━━━ 100% 1000/1000 [00:04<00:00, 250.0 items/s]
```

### Timed Operations

```python
from sentinel_ml.cli import timed_operation

with timed_operation("Clustering"):
    result = cluster(embeddings)
# Output: ✓ Clustering completed in 1.234s
```

---

## Profiler

Hierarchical timing instrumentation for performance analysis.

### Basic Profiling

```python
from sentinel_ml.cli import Profiler, measure, profile

# Global profiler
profiler = Profiler(enabled=True, threshold_ms=1.0)

# Context manager
with profiler.measure("embedding"):
    embeddings = embed(logs)

# Decorator
@profiler.profile("clustering")
def cluster_logs(embeddings):
    return cluster(embeddings)

# Get report
print(profiler.format_report())
```

Output:
```
Performance Profile
════════════════════════════════════════════════════════
  embedding                    89.45ms  (59.5%)
  clustering                   60.12ms  (40.0%)
    ├── hdbscan                45.00ms  (30.0%)
    └── summary_generation     15.12ms  (10.0%)
────────────────────────────────────────────────────────
  Total                       150.32ms (100.0%)
```

### Nested Profiling

```python
with profiler.measure("analyze"):
    with profiler.measure("embed"):
        embeddings = embed(logs)
    with profiler.measure("cluster"):
        with profiler.measure("hdbscan"):
            clusters = hdbscan(embeddings)
        with profiler.measure("summaries"):
            summaries = summarize(clusters)
```

### Global Profiler

```python
from sentinel_ml.cli import enable_profiling, disable_profiling, get_profiler

# Enable globally
enable_profiling()

# Use module-level convenience functions
from sentinel_ml.cli import measure

with measure("operation"):
    do_something()

# Get results
profiler = get_profiler()
print(profiler.to_dict())
```

---

## Report Generation

Export analysis results to Markdown or HTML.

### Markdown Reports

```python
from sentinel_ml.cli import MarkdownReporter, ReportData, ReportConfig

config = ReportConfig(
    title="Log Analysis Report",
    include_toc=True,
    include_summary=True,
)

data = ReportData(
    timestamp=datetime.now(),
    total_logs=1247,
    clusters=cluster_summaries,
    novel_scores=novel_patterns,
    explanations=explanations,
)

reporter = MarkdownReporter(config)
markdown = reporter.generate(data)
reporter.save(data, Path("report.md"))
```

### HTML Reports

```python
from sentinel_ml.cli import HTMLReporter

reporter = HTMLReporter(config)
html = reporter.generate(data)
reporter.save(data, Path("report.html"))
```

Features:
- Embedded CSS (no external dependencies)
- Responsive layout
- Cluster cards with color coding
- Syntax-highlighted code blocks
- Executive summary with charts

---

## Configuration Commands

### Generate Config

```python
from sentinel_ml.cli import generate_config

# Generate default config
yaml_content = generate_config()

# Generate minimal config
yaml_content = generate_config(minimal=True)

# Save to file
generate_config(output=Path("config.yaml"))
```

### Validate Config

```python
from sentinel_ml.cli import validate_config

is_valid, errors = validate_config(Path("config.yaml"))

if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

### Load Config

```python
from sentinel_ml.cli import load_config

config = load_config(Path("config.yaml"))
print(config.server.port)  # 50051
```

### Show Config

```python
from sentinel_ml.cli import show_config

output = show_config(config)
print(output)
```

Output:
```
Server:
  host: 0.0.0.0
  port: 50051

Embedding:
  model: all-MiniLM-L6-v2
  batch_size: 32

Clustering:
  min_cluster_size: 5
  metric: euclidean

Novelty:
  threshold: 0.7
  k_neighbors: 5

LLM:
  provider: ollama
  model: llama3.2
```

---

## Output Capture (Testing)

Capture console output for testing:

```python
from sentinel_ml.cli import Console, CapturedOutput

console = Console()

with console.capture() as captured:
    console.info("Test message")
    console.success("Done!")

print(captured.stdout)  # Contains both messages
print(captured.lines)   # List of output lines
```

---

*See also: [[Configuration Reference|Configuration-Reference]], [[API Reference|API-Reference]]*
