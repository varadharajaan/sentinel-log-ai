"""
Metrics collection for benchmarking.

Provides structured metrics collection following the Observer pattern.
All metrics are designed for JSONL output compatibility.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass
class TimingMetrics:
    """
    Timing metrics for a single operation.

    Attributes:
        name: Name of the timed operation.
        samples: List of timing samples in seconds.
        unit: Time unit for display (seconds, milliseconds).
    """

    name: str
    samples: list[float] = field(default_factory=list)
    unit: str = "seconds"

    def add_sample(self, duration: float) -> None:
        """Add a timing sample."""
        self.samples.append(duration)
        logger.debug(
            "timing_sample_added",
            operation=self.name,
            duration_seconds=round(duration, 6),
        )

    @property
    def count(self) -> int:
        """Number of samples."""
        return len(self.samples)

    @property
    def total(self) -> float:
        """Total time across all samples."""
        return sum(self.samples)

    @property
    def mean(self) -> float:
        """Mean duration."""
        if not self.samples:
            return 0.0
        return statistics.mean(self.samples)

    @property
    def median(self) -> float:
        """Median duration."""
        if not self.samples:
            return 0.0
        return statistics.median(self.samples)

    @property
    def std_dev(self) -> float:
        """Standard deviation."""
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(self.samples)

    @property
    def min_value(self) -> float:
        """Minimum duration."""
        if not self.samples:
            return 0.0
        return min(self.samples)

    @property
    def max_value(self) -> float:
        """Maximum duration."""
        if not self.samples:
            return 0.0
        return max(self.samples)

    @property
    def p50(self) -> float:
        """50th percentile (median)."""
        return self.median

    @property
    def p95(self) -> float:
        """95th percentile."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def p99(self) -> float:
        """99th percentile."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "count": self.count,
            "total_seconds": round(self.total, 6),
            "mean_seconds": round(self.mean, 6),
            "median_seconds": round(self.median, 6),
            "std_dev_seconds": round(self.std_dev, 6),
            "min_seconds": round(self.min_value, 6),
            "max_seconds": round(self.max_value, 6),
            "p50_seconds": round(self.p50, 6),
            "p95_seconds": round(self.p95, 6),
            "p99_seconds": round(self.p99, 6),
        }


@dataclass
class ThroughputMetrics:
    """
    Throughput metrics for measuring processing rates.

    Attributes:
        name: Name of the throughput measurement.
        items_processed: Total items processed.
        bytes_processed: Total bytes processed.
        duration_seconds: Total duration in seconds.
    """

    name: str
    items_processed: int = 0
    bytes_processed: int = 0
    duration_seconds: float = 0.0

    def record(
        self,
        items: int,
        bytes_count: int = 0,
        duration: float = 0.0,
    ) -> None:
        """Record processing metrics."""
        self.items_processed += items
        self.bytes_processed += bytes_count
        self.duration_seconds += duration
        logger.debug(
            "throughput_recorded",
            operation=self.name,
            items=items,
            bytes=bytes_count,
            duration_seconds=round(duration, 6),
        )

    @property
    def items_per_second(self) -> float:
        """Calculate items per second."""
        if self.duration_seconds <= 0:
            return 0.0
        return self.items_processed / self.duration_seconds

    @property
    def bytes_per_second(self) -> float:
        """Calculate bytes per second."""
        if self.duration_seconds <= 0:
            return 0.0
        return self.bytes_processed / self.duration_seconds

    @property
    def mb_per_second(self) -> float:
        """Calculate megabytes per second."""
        return self.bytes_per_second / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "items_processed": self.items_processed,
            "bytes_processed": self.bytes_processed,
            "duration_seconds": round(self.duration_seconds, 6),
            "items_per_second": round(self.items_per_second, 2),
            "bytes_per_second": round(self.bytes_per_second, 2),
            "mb_per_second": round(self.mb_per_second, 4),
        }


class MemoryTracker:
    """
    Track memory usage over time.

    Uses psutil for accurate memory measurements.
    Follows the Observer pattern for memory change notifications.
    """

    def __init__(self) -> None:
        """Initialize memory tracker."""
        self._snapshots: list[dict[str, Any]] = []
        self._baseline_rss: int = 0
        self._peak_rss: int = 0
        self._observers: list[Callable[[dict[str, Any]], None]] = []
        logger.info("memory_tracker_initialized")

    def add_observer(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Add an observer for memory changes."""
        self._observers.append(callback)

    def _notify_observers(self, snapshot: dict[str, Any]) -> None:
        """Notify all observers of a new snapshot."""
        for observer in self._observers:
            try:
                observer(snapshot)
            except Exception as e:
                logger.warning(
                    "observer_notification_failed",
                    error=str(e),
                )

    def _get_memory_info(self) -> dict[str, int]:
        """Get current memory information."""
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss": mem_info.rss,
                "vms": mem_info.vms,
                "shared": getattr(mem_info, "shared", 0),
                "data": getattr(mem_info, "data", 0),
            }
        except ImportError:
            logger.warning("psutil_not_available", fallback="resource_module")
            try:
                import resource

                usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
                return {
                    "rss": usage.ru_maxrss * 1024,
                    "vms": 0,
                    "shared": 0,
                    "data": 0,
                }
            except ImportError:
                logger.warning("resource_not_available", fallback="zero")
                return {"rss": 0, "vms": 0, "shared": 0, "data": 0}

    def set_baseline(self) -> dict[str, Any]:
        """Set the baseline memory measurement."""
        mem_info = self._get_memory_info()
        self._baseline_rss = mem_info["rss"]
        self._peak_rss = mem_info["rss"]

        snapshot = {
            "type": "baseline",
            "timestamp": time.time(),
            "rss_bytes": mem_info["rss"],
            "vms_bytes": mem_info["vms"],
            "rss_mb": round(mem_info["rss"] / (1024 * 1024), 2),
        }
        self._snapshots.append(snapshot)
        logger.info(
            "memory_baseline_set",
            rss_mb=snapshot["rss_mb"],
        )
        return snapshot

    def snapshot(self, label: str = "") -> dict[str, Any]:
        """Take a memory snapshot."""
        mem_info = self._get_memory_info()
        current_rss = mem_info["rss"]

        if current_rss > self._peak_rss:
            self._peak_rss = current_rss

        delta_from_baseline = current_rss - self._baseline_rss

        snapshot = {
            "type": "snapshot",
            "label": label,
            "timestamp": time.time(),
            "rss_bytes": current_rss,
            "vms_bytes": mem_info["vms"],
            "rss_mb": round(current_rss / (1024 * 1024), 2),
            "delta_bytes": delta_from_baseline,
            "delta_mb": round(delta_from_baseline / (1024 * 1024), 2),
        }
        self._snapshots.append(snapshot)
        self._notify_observers(snapshot)

        logger.debug(
            "memory_snapshot",
            label=label,
            rss_mb=snapshot["rss_mb"],
            delta_mb=snapshot["delta_mb"],
        )
        return snapshot

    @property
    def peak_rss_bytes(self) -> int:
        """Get peak RSS in bytes."""
        return self._peak_rss

    @property
    def peak_rss_mb(self) -> float:
        """Get peak RSS in megabytes."""
        return self._peak_rss / (1024 * 1024)

    @property
    def current_rss_mb(self) -> float:
        """Get current RSS in megabytes."""
        mem_info = self._get_memory_info()
        return mem_info["rss"] / (1024 * 1024)

    @property
    def snapshots(self) -> list[dict[str, Any]]:
        """Get all snapshots."""
        return self._snapshots.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get memory tracking summary."""
        return {
            "baseline_rss_mb": round(self._baseline_rss / (1024 * 1024), 2),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "current_rss_mb": round(self.current_rss_mb, 2),
            "total_growth_mb": round(
                (self._peak_rss - self._baseline_rss) / (1024 * 1024),
                2,
            ),
            "snapshot_count": len(self._snapshots),
        }


class MetricsCollector:
    """
    Central metrics collector for benchmarks.

    Aggregates timing, throughput, and memory metrics.
    Follows the Facade pattern for simplified metric management.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize metrics collector.

        Args:
            name: Name of the benchmark or test.
        """
        self.name = name
        self._timing_metrics: dict[str, TimingMetrics] = {}
        self._throughput_metrics: dict[str, ThroughputMetrics] = {}
        self._memory_tracker = MemoryTracker()
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._metadata: dict[str, Any] = {}
        logger.info("metrics_collector_initialized", name=name)

    def start(self) -> None:
        """Start the metrics collection."""
        self._start_time = time.time()
        self._memory_tracker.set_baseline()
        logger.info("metrics_collection_started", name=self.name)

    def stop(self) -> None:
        """Stop the metrics collection."""
        self._end_time = time.time()
        self._memory_tracker.snapshot("final")
        logger.info(
            "metrics_collection_stopped",
            name=self.name,
            duration_seconds=round(self.total_duration, 2),
        )

    @property
    def total_duration(self) -> float:
        """Get total duration in seconds."""
        if self._end_time <= 0:
            return time.time() - self._start_time
        return self._end_time - self._start_time

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the metrics."""
        self._metadata[key] = value

    def get_timing(self, name: str) -> TimingMetrics:
        """Get or create a timing metrics tracker."""
        if name not in self._timing_metrics:
            self._timing_metrics[name] = TimingMetrics(name=name)
        return self._timing_metrics[name]

    def get_throughput(self, name: str) -> ThroughputMetrics:
        """Get or create a throughput metrics tracker."""
        if name not in self._throughput_metrics:
            self._throughput_metrics[name] = ThroughputMetrics(name=name)
        return self._throughput_metrics[name]

    @property
    def memory(self) -> MemoryTracker:
        """Get the memory tracker."""
        return self._memory_tracker

    def time_operation(self, name: str) -> _TimingContext:
        """Context manager for timing an operation."""
        return _TimingContext(self.get_timing(name))

    def to_dict(self) -> dict[str, Any]:
        """Convert all metrics to dictionary."""
        return {
            "name": self.name,
            "total_duration_seconds": round(self.total_duration, 2),
            "metadata": self._metadata,
            "timing": {name: m.to_dict() for name, m in self._timing_metrics.items()},
            "throughput": {name: m.to_dict() for name, m in self._throughput_metrics.items()},
            "memory": self._memory_tracker.get_summary(),
        }


class _TimingContext:
    """Context manager for timing operations."""

    def __init__(self, timing: TimingMetrics) -> None:
        self._timing = timing
        self._start: float = 0.0

    def __enter__(self) -> _TimingContext:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        duration = time.perf_counter() - self._start
        self._timing.add_sample(duration)
