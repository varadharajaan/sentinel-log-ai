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
    "DatasetConfig",
    "DatasetGenerator",
    "FunctionBenchmark",
    "GeneratedLogRecord",
    "LogLevel",
    "LogPattern",
    "MemoryProfiler",
    "MemorySnapshot",
    "MemoryTracker",
    "MetricsCollector",
    "ThroughputMetrics",
    "TimingMetrics",
    "create_scale_datasets",
    "generate_test_logs",
    "memory_profile",
    "profile_memory",
]
