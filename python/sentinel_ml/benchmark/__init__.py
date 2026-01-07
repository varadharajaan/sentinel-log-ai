"""
Benchmark and performance testing module.

Provides utilities for:
- Ingestion rate benchmarking
- Embedding throughput measurement
- Clustering performance analysis
- Memory profiling and tracking
- Dataset scale testing

Design Patterns:
- Strategy Pattern: Pluggable benchmark runners
- Template Method: Common benchmark execution flow
- Observer Pattern: Real-time metrics collection
"""

from sentinel_ml.benchmark.datasets import (
    DatasetConfig,
    DatasetGenerator,
    GeneratedLogRecord,
    LogLevel,
    LogPattern,
    create_scale_datasets,
    generate_test_logs,
)
from sentinel_ml.benchmark.metrics import (
    MemoryTracker,
    MetricsCollector,
    ThroughputMetrics,
    TimingMetrics,
)
from sentinel_ml.benchmark.profiler import (
    MemoryProfiler,
    MemorySnapshot,
    memory_profile,
    profile_memory,
)
from sentinel_ml.benchmark.runner import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkStatus,
    BenchmarkSuite,
    FunctionBenchmark,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkStatus",
    "BenchmarkSuite",
    "create_scale_datasets",
    "DatasetConfig",
    "DatasetGenerator",
    "FunctionBenchmark",
    "generate_test_logs",
    "GeneratedLogRecord",
    "LogLevel",
    "LogPattern",
    "memory_profile",
    "MemoryProfiler",
    "MemorySnapshot",
    "MemoryTracker",
    "MetricsCollector",
    "profile_memory",
    "ThroughputMetrics",
    "TimingMetrics",
]
