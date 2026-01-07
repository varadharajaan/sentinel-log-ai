"""
Unit tests for benchmark module.

Tests metrics, profiler, runner, and datasets components.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

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
    MetricsCollector,
    MemoryTracker,
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

if TYPE_CHECKING:
    pass


class TestTimingMetrics:
    """Tests for TimingMetrics class."""

    def test_init_empty(self) -> None:
        """Test initialization with no samples."""
        metrics = TimingMetrics(name="test")
        assert metrics.count == 0
        assert metrics.mean == 0.0
        assert metrics.samples == []

    def test_add_sample(self) -> None:
        """Test adding samples."""
        metrics = TimingMetrics(name="test")
        metrics.add_sample(1.0)
        metrics.add_sample(2.0)
        metrics.add_sample(3.0)

        assert metrics.count == 3
        assert metrics.samples == [1.0, 2.0, 3.0]

    def test_mean_calculation(self) -> None:
        """Test mean calculation."""
        metrics = TimingMetrics(name="test")
        for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
            metrics.add_sample(val)

        assert metrics.mean == 3.0

    def test_median_odd_count(self) -> None:
        """Test median with odd number of samples."""
        metrics = TimingMetrics(name="test")
        for val in [1.0, 5.0, 3.0, 2.0, 4.0]:
            metrics.add_sample(val)

        assert metrics.median == 3.0

    def test_median_even_count(self) -> None:
        """Test median with even number of samples."""
        metrics = TimingMetrics(name="test")
        for val in [1.0, 2.0, 3.0, 4.0]:
            metrics.add_sample(val)

        assert metrics.median == 2.5

    def test_min_max(self) -> None:
        """Test min and max values."""
        metrics = TimingMetrics(name="test")
        for val in [5.0, 1.0, 10.0, 3.0]:
            metrics.add_sample(val)

        assert metrics.min_value == 1.0
        assert metrics.max_value == 10.0

    def test_std_dev(self) -> None:
        """Test standard deviation calculation."""
        metrics = TimingMetrics(name="test")
        for val in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            metrics.add_sample(val)

        # Using sample std dev (n-1 denominator)
        assert metrics.std_dev == pytest.approx(2.138, rel=0.01)

    def test_percentiles(self) -> None:
        """Test percentile calculations."""
        metrics = TimingMetrics(name="test")
        for val in range(1, 101):
            metrics.add_sample(float(val))

        assert metrics.p50 == pytest.approx(50.0, rel=0.02)
        assert metrics.p95 == pytest.approx(95.0, rel=0.02)
        assert metrics.p99 == pytest.approx(99.0, rel=0.02)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = TimingMetrics(name="test")
        metrics.add_sample(1.0)
        metrics.add_sample(2.0)

        result = metrics.to_dict()

        assert "count" in result
        assert "name" in result
        assert result["name"] == "test"


class TestThroughputMetrics:
    """Tests for ThroughputMetrics class."""

    def test_init(self) -> None:
        """Test initialization."""
        metrics = ThroughputMetrics(name="test")
        assert metrics.items_processed == 0
        assert metrics.bytes_processed == 0

    def test_record_items(self) -> None:
        """Test recording items."""
        metrics = ThroughputMetrics(name="test")
        metrics.record(items=100)
        metrics.record(items=50)

        assert metrics.items_processed == 150

    def test_record_bytes(self) -> None:
        """Test recording bytes."""
        metrics = ThroughputMetrics(name="test")
        metrics.record(items=0, bytes_count=1024)
        metrics.record(items=0, bytes_count=2048)

        assert metrics.bytes_processed == 3072

    def test_items_per_second(self) -> None:
        """Test items per second calculation."""
        metrics = ThroughputMetrics(name="test")
        metrics.record(items=1000, duration=2.0)

        assert metrics.items_per_second == pytest.approx(500.0, rel=0.01)

    def test_mb_per_second(self) -> None:
        """Test MB per second calculation."""
        metrics = ThroughputMetrics(name="test")
        bytes_count = 10 * 1024 * 1024
        metrics.record(items=0, bytes_count=bytes_count, duration=2.0)

        assert metrics.mb_per_second == pytest.approx(5.0, rel=0.01)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = ThroughputMetrics(name="test")
        metrics.record(items=100, bytes_count=1024, duration=1.0)

        result = metrics.to_dict()

        assert "items_processed" in result
        assert "bytes_processed" in result
        assert "items_per_second" in result
        assert "mb_per_second" in result


class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    def test_init(self) -> None:
        """Test initialization."""
        tracker = MemoryTracker()
        assert len(tracker.snapshots) == 0

    def test_snapshot(self) -> None:
        """Test taking a snapshot."""
        tracker = MemoryTracker()
        tracker.snapshot("test")

        assert len(tracker.snapshots) == 1

    def test_multiple_snapshots(self) -> None:
        """Test multiple snapshots."""
        tracker = MemoryTracker()
        tracker.snapshot("first")
        tracker.snapshot("second")
        tracker.snapshot("third")

        assert len(tracker.snapshots) == 3

    def test_get_snapshots(self) -> None:
        """Test getting snapshots."""
        tracker = MemoryTracker()
        tracker.snapshot("test1")
        tracker.snapshot("test2")

        snapshots = tracker.snapshots

        assert len(snapshots) == 2
        assert snapshots[0]["label"] == "test1"
        assert snapshots[1]["label"] == "test2"

    def test_peak_rss_mb(self) -> None:
        """Test peak RSS tracking."""
        tracker = MemoryTracker()
        tracker.snapshot("test")

        peak = tracker.peak_rss_mb
        assert peak >= 0


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_init(self) -> None:
        """Test initialization."""
        collector = MetricsCollector(name="test")
        assert collector is not None
        assert collector.name == "test"

    def test_time_operation(self) -> None:
        """Test timing an operation."""
        collector = MetricsCollector(name="test")

        with collector.time_operation("test_op"):
            time.sleep(0.01)

        timing = collector.get_timing("test_op")
        assert timing.count == 1

    def test_get_throughput(self) -> None:
        """Test getting throughput metrics."""
        collector = MetricsCollector(name="test")
        throughput = collector.get_throughput("process")
        throughput.record(items=100, bytes_count=1024, duration=1.0)

        assert throughput.items_processed == 100

    def test_to_dict(self) -> None:
        """Test getting summary dictionary."""
        collector = MetricsCollector(name="test")

        with collector.time_operation("op1"):
            pass

        result = collector.to_dict()

        assert "timing" in result
        assert "throughput" in result
        assert "memory" in result


class TestMemorySnapshot:
    """Tests for MemorySnapshot class."""

    def test_init(self) -> None:
        """Test initialization."""
        snap = MemorySnapshot(
            label="test",
            timestamp=1000.0,
            rss_bytes=1024 * 1024 * 100,
            vms_bytes=1024 * 1024 * 200,
        )

        assert snap.label == "test"
        assert snap.rss_bytes == 1024 * 1024 * 100

    def test_rss_mb(self) -> None:
        """Test RSS in MB."""
        snap = MemorySnapshot(
            label="test",
            timestamp=1000.0,
            rss_bytes=1024 * 1024 * 100,
            vms_bytes=0,
        )

        assert snap.rss_mb == 100.0

    def test_vms_mb(self) -> None:
        """Test VMS in MB."""
        snap = MemorySnapshot(
            label="test",
            timestamp=1000.0,
            rss_bytes=0,
            vms_bytes=1024 * 1024 * 200,
        )

        assert snap.vms_mb == 200.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        snap = MemorySnapshot(
            label="test",
            timestamp=1000.0,
            rss_bytes=1024 * 1024,
            vms_bytes=2 * 1024 * 1024,
            gc_objects=1000,
        )

        result = snap.to_dict()

        assert result["label"] == "test"
        assert result["rss_mb"] == 1.0
        assert result["vms_mb"] == 2.0
        assert result["gc_objects"] == 1000


class TestMemoryProfiler:
    """Tests for MemoryProfiler class."""

    def test_init(self) -> None:
        """Test initialization."""
        profiler = MemoryProfiler("test")
        assert profiler.name == "test"

    def test_set_baseline(self) -> None:
        """Test setting baseline."""
        profiler = MemoryProfiler("test")
        baseline = profiler.set_baseline()

        assert baseline is not None
        assert baseline.label == "baseline"

    def test_snapshot(self) -> None:
        """Test taking snapshot."""
        profiler = MemoryProfiler("test")
        snap = profiler.snapshot("test_point")

        assert snap.label == "test_point"

    def test_force_gc(self) -> None:
        """Test forcing garbage collection."""
        profiler = MemoryProfiler("test")
        result = profiler.force_gc()

        assert "collected" in result

    def test_peak_rss_mb(self) -> None:
        """Test peak RSS tracking."""
        profiler = MemoryProfiler("test")
        profiler.snapshot("first")
        profiler.snapshot("second")

        peak = profiler.peak_rss_mb
        assert peak >= 0

    def test_total_growth_mb(self) -> None:
        """Test growth calculation."""
        profiler = MemoryProfiler("test")
        profiler.set_baseline()
        profiler.snapshot("after")

        growth = profiler.total_growth_mb
        assert isinstance(growth, float)

    def test_get_summary(self) -> None:
        """Test getting summary."""
        profiler = MemoryProfiler("test")
        profiler.set_baseline()
        profiler.snapshot("point1")

        summary = profiler.get_summary()

        assert summary["name"] == "test"
        assert summary["snapshots"] == 2

    def test_track_context_manager(self) -> None:
        """Test track context manager."""
        profiler = MemoryProfiler("test")

        with profiler.track("operation") as before:
            _ = [i for i in range(1000)]

        snapshots = profiler.get_all_snapshots()
        assert len(snapshots) == 2
        assert "operation_before" in snapshots[0]["label"]


class TestProfileMemoryContextManager:
    """Tests for profile_memory context manager."""

    def test_basic_usage(self) -> None:
        """Test basic context manager usage."""
        with profile_memory("test") as profiler:
            profiler.set_baseline()
            _ = list(range(1000))
            profiler.snapshot("after")

        summary = profiler.get_summary()
        assert summary["name"] == "test"


class TestMemoryProfileDecorator:
    """Tests for memory_profile decorator."""

    def test_decorator(self) -> None:
        """Test decorator wraps function."""

        @memory_profile
        def sample_function() -> int:
            return sum(range(1000))

        result = sample_function()
        assert result == 499500


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig class."""

    def test_init_defaults(self) -> None:
        """Test default values."""
        config = BenchmarkConfig(name="test")

        assert config.name == "test"
        assert config.warmup_iterations == 3
        assert config.iterations == 10
        assert config.collect_memory is True

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = BenchmarkConfig(
            name="custom",
            warmup_iterations=5,
            iterations=20,
            timeout_seconds=60.0,
            collect_memory=False,
        )

        assert config.warmup_iterations == 5
        assert config.iterations == 20
        assert config.timeout_seconds == 60.0
        assert config.collect_memory is False

    def test_validate_negative_warmup(self) -> None:
        """Test validation rejects negative warmup."""
        config = BenchmarkConfig(name="test", warmup_iterations=-1)

        with pytest.raises(ValueError, match="warmup_iterations"):
            config.validate()

    def test_validate_zero_iterations(self) -> None:
        """Test validation rejects zero iterations."""
        config = BenchmarkConfig(name="test", iterations=0)

        with pytest.raises(ValueError, match="iterations"):
            config.validate()

    def test_validate_negative_timeout(self) -> None:
        """Test validation rejects negative timeout."""
        config = BenchmarkConfig(name="test", timeout_seconds=-1.0)

        with pytest.raises(ValueError, match="timeout_seconds"):
            config.validate()


class TestBenchmarkResult:
    """Tests for BenchmarkResult class."""

    def test_init(self) -> None:
        """Test initialization."""
        config = BenchmarkConfig(name="test")
        result = BenchmarkResult(config=config)

        assert result.config == config
        assert result.status == BenchmarkStatus.PENDING

    def test_duration_seconds(self) -> None:
        """Test duration calculation."""
        config = BenchmarkConfig(name="test")
        result = BenchmarkResult(config=config, start_time=100.0, end_time=105.0)

        assert result.duration_seconds == 5.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = BenchmarkConfig(name="test", iterations=5)
        result = BenchmarkResult(
            config=config,
            status=BenchmarkStatus.COMPLETED,
            metrics={"timing": {"mean": 1.0}},
        )

        data = result.to_dict()

        assert data["name"] == "test"
        assert data["status"] == "completed"
        assert data["iterations"] == 5


class TestFunctionBenchmark:
    """Tests for FunctionBenchmark class."""

    def test_benchmark_function(self) -> None:
        """Test benchmarking a simple function."""
        call_count = 0

        def sample_func() -> None:
            nonlocal call_count
            call_count += 1

        config = BenchmarkConfig(name="sample", iterations=5, warmup_iterations=2)
        benchmark = FunctionBenchmark(sample_func, config)

        result = benchmark.execute()

        assert result.status == BenchmarkStatus.COMPLETED
        assert call_count == 7

    def test_benchmark_with_args(self) -> None:
        """Test benchmarking with arguments."""
        results: list[int] = []

        def add_func(a: int, b: int) -> None:
            results.append(a + b)

        config = BenchmarkConfig(name="add", iterations=3, warmup_iterations=0)
        benchmark = FunctionBenchmark(add_func, config, args=(2, 3))

        result = benchmark.execute()

        assert result.status == BenchmarkStatus.COMPLETED
        assert results == [5, 5, 5]

    def test_benchmark_with_kwargs(self) -> None:
        """Test benchmarking with keyword arguments."""
        results: list[str] = []

        def greet(name: str = "world") -> None:
            results.append(f"hello {name}")

        config = BenchmarkConfig(name="greet", iterations=2, warmup_iterations=0)
        benchmark = FunctionBenchmark(greet, config, kwargs={"name": "test"})

        result = benchmark.execute()

        assert result.status == BenchmarkStatus.COMPLETED
        assert results == ["hello test", "hello test"]

    def test_benchmark_failure(self) -> None:
        """Test benchmark failure handling."""

        def failing_func() -> None:
            raise RuntimeError("test error")

        config = BenchmarkConfig(name="fail", iterations=1, warmup_iterations=0)
        benchmark = FunctionBenchmark(failing_func, config)

        with pytest.raises(RuntimeError, match="test error"):
            benchmark.execute()


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite class."""

    def test_init(self) -> None:
        """Test initialization."""
        suite = BenchmarkSuite("test_suite")
        assert suite.name == "test_suite"

    def test_add_benchmark(self) -> None:
        """Test adding benchmarks."""
        suite = BenchmarkSuite("test")

        def func() -> None:
            pass

        config = BenchmarkConfig(name="func", iterations=1)
        benchmark = FunctionBenchmark(func, config)

        result = suite.add(benchmark)

        assert result is suite

    def test_add_function(self) -> None:
        """Test adding function directly."""
        suite = BenchmarkSuite("test")

        def sample() -> None:
            pass

        result = suite.add_function(sample, iterations=2)

        assert result is suite

    def test_run_all(self) -> None:
        """Test running all benchmarks."""
        suite = BenchmarkSuite("test")

        call_counts = {"a": 0, "b": 0}

        def func_a() -> None:
            call_counts["a"] += 1

        def func_b() -> None:
            call_counts["b"] += 1

        suite.add_function(func_a, iterations=2)
        suite.add_function(func_b, iterations=3)

        results = suite.run_all()

        assert len(results) == 2
        assert all(r.status == BenchmarkStatus.COMPLETED for r in results)

    def test_get_summary(self) -> None:
        """Test getting suite summary."""
        suite = BenchmarkSuite("test")

        def sample() -> None:
            pass

        suite.add_function(sample, iterations=1)
        suite.run_all()

        summary = suite.get_summary()

        assert summary["name"] == "test"
        assert summary["total_benchmarks"] == 1
        assert summary["completed"] == 1


class TestDatasetConfig:
    """Tests for DatasetConfig class."""

    def test_init_defaults(self) -> None:
        """Test default values."""
        config = DatasetConfig(name="test", size=100)

        assert config.name == "test"
        assert config.size == 100
        assert config.time_range_hours == 24
        assert len(config.patterns) > 0
        assert len(config.level_distribution) > 0

    def test_custom_patterns(self) -> None:
        """Test custom pattern distribution."""
        patterns = {LogPattern.HTTP_REQUEST: 1.0}
        config = DatasetConfig(name="test", size=100, patterns=patterns)

        assert config.patterns == patterns

    def test_custom_levels(self) -> None:
        """Test custom level distribution."""
        levels = {LogLevel.ERROR: 1.0}
        config = DatasetConfig(name="test", size=100, level_distribution=levels)

        assert config.level_distribution == levels


class TestGeneratedLogRecord:
    """Tests for GeneratedLogRecord class."""

    def test_init(self) -> None:
        """Test initialization."""
        from datetime import datetime, timezone

        record = GeneratedLogRecord(
            id="log_001",
            message="test message",
            level="INFO",
            source="/var/log/test.log",
            timestamp=datetime.now(timezone.utc),
            raw="raw log line",
        )

        assert record.id == "log_001"
        assert record.message == "test message"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        from datetime import datetime, timezone

        record = GeneratedLogRecord(
            id="log_001",
            message="test",
            level="INFO",
            source="/var/log/test.log",
            timestamp=datetime.now(timezone.utc),
            raw="raw",
        )

        data = record.to_dict()

        assert data["id"] == "log_001"
        assert "timestamp" in data

    def test_to_json_line(self) -> None:
        """Test JSON line generation."""
        import json
        from datetime import datetime, timezone

        record = GeneratedLogRecord(
            id="log_001",
            message="test",
            level="INFO",
            source="/var/log/test.log",
            timestamp=datetime.now(timezone.utc),
            raw="raw",
        )

        line = record.to_json_line()
        parsed = json.loads(line)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "test"


class TestDatasetGenerator:
    """Tests for DatasetGenerator class."""

    def test_init(self) -> None:
        """Test initialization."""
        config = DatasetConfig(name="test", size=100, seed=42)
        generator = DatasetGenerator(config)

        assert generator.config == config

    def test_generate_one(self) -> None:
        """Test generating single record."""
        config = DatasetConfig(name="test", size=100, seed=42)
        generator = DatasetGenerator(config)

        record = generator.generate_one()

        assert record.id is not None
        assert record.message is not None
        assert record.level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_generate_batch(self) -> None:
        """Test generating batch."""
        config = DatasetConfig(name="test", size=100, seed=42)
        generator = DatasetGenerator(config)

        records = generator.generate_batch(50)

        assert len(records) == 50

    def test_generate_all(self) -> None:
        """Test generating all records."""
        config = DatasetConfig(name="test", size=100, seed=42)
        generator = DatasetGenerator(config)

        records = generator.generate_all()

        assert len(records) == 100

    def test_generate_iter(self) -> None:
        """Test iterator generation."""
        config = DatasetConfig(name="test", size=50, seed=42)
        generator = DatasetGenerator(config)

        records = list(generator.generate_iter(batch_size=10))

        assert len(records) == 50

    def test_reproducibility(self) -> None:
        """Test seed ensures reproducibility."""
        config1 = DatasetConfig(name="test", size=10, seed=42)
        config2 = DatasetConfig(name="test", size=10, seed=42)

        gen1 = DatasetGenerator(config1)
        gen2 = DatasetGenerator(config2)

        records1 = gen1.generate_all()
        records2 = gen2.generate_all()

        for r1, r2 in zip(records1, records2, strict=True):
            assert r1.message == r2.message
            assert r1.level == r2.level


class TestGenerateTestLogs:
    """Tests for generate_test_logs function."""

    def test_basic_generation(self) -> None:
        """Test basic log generation."""
        logs = generate_test_logs(100)

        assert len(logs) == 100

    def test_with_seed(self) -> None:
        """Test reproducible generation."""
        logs1 = generate_test_logs(10, seed=42)
        logs2 = generate_test_logs(10, seed=42)

        for l1, l2 in zip(logs1, logs2, strict=True):
            assert l1.message == l2.message

    def test_with_name(self) -> None:
        """Test custom dataset name."""
        logs = generate_test_logs(10, name="custom")

        assert len(logs) == 10


class TestCreateScaleDatasets:
    """Tests for create_scale_datasets function."""

    def test_creates_standard_configs(self) -> None:
        """Test creating standard scale configs."""
        configs = create_scale_datasets()

        assert "small" in configs
        assert "medium" in configs
        assert "large" in configs
        assert "xlarge" in configs

    def test_sizes_are_correct(self) -> None:
        """Test dataset sizes are as expected."""
        configs = create_scale_datasets()

        assert configs["small"].size == 1000
        assert configs["medium"].size == 10000
        assert configs["large"].size == 100000
        assert configs["xlarge"].size == 1000000

    def test_seeds_are_consistent(self) -> None:
        """Test all configs have same seed for reproducibility."""
        configs = create_scale_datasets()

        seeds = [c.seed for c in configs.values()]
        assert all(s == 42 for s in seeds)
