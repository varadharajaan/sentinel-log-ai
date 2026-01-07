"""
Memory profiling utilities.

Provides memory profiling for detecting memory leaks and
measuring memory consumption of operations.
"""

from __future__ import annotations

import functools
import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

logger = get_logger(__name__)

F = TypeVar("F", bound="Callable[..., Any]")


@dataclass
class MemorySnapshot:
    """
    A point-in-time memory measurement.

    Attributes:
        label: Description of this snapshot point.
        timestamp: Unix timestamp when snapshot was taken.
        rss_bytes: Resident Set Size in bytes.
        vms_bytes: Virtual Memory Size in bytes.
        gc_objects: Number of tracked objects by garbage collector.
        gc_collections: GC collection counts by generation.
    """

    label: str
    timestamp: float
    rss_bytes: int
    vms_bytes: int
    gc_objects: int = 0
    gc_collections: tuple[int, int, int] = (0, 0, 0)

    @property
    def rss_mb(self) -> float:
        """RSS in megabytes."""
        return self.rss_bytes / (1024 * 1024)

    @property
    def vms_mb(self) -> float:
        """VMS in megabytes."""
        return self.vms_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "timestamp": self.timestamp,
            "rss_bytes": self.rss_bytes,
            "rss_mb": round(self.rss_mb, 2),
            "vms_bytes": self.vms_bytes,
            "vms_mb": round(self.vms_mb, 2),
            "gc_objects": self.gc_objects,
            "gc_collections": list(self.gc_collections),
        }


class MemoryProfiler:
    """
    Memory profiler for tracking memory usage.

    Provides detailed memory profiling with GC integration
    and leak detection capabilities.
    """

    def __init__(self, name: str = "default") -> None:
        """
        Initialize memory profiler.

        Args:
            name: Name for this profiler instance.
        """
        self.name = name
        self._snapshots: list[MemorySnapshot] = []
        self._baseline: MemorySnapshot | None = None
        self._gc_enabled_initially = gc.isenabled()
        logger.info("memory_profiler_initialized", name=name)

    def _get_memory_info(self) -> tuple[int, int]:
        """Get current RSS and VMS."""
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            return mem_info.rss, mem_info.vms
        except ImportError:
            try:
                import resource

                usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
                rss = usage.ru_maxrss * 1024
                return rss, 0
            except ImportError:
                return 0, 0

    def _create_snapshot(self, label: str) -> MemorySnapshot:
        """Create a memory snapshot."""
        rss, vms = self._get_memory_info()
        gc_stats = gc.get_count()

        return MemorySnapshot(
            label=label,
            timestamp=time.time(),
            rss_bytes=rss,
            vms_bytes=vms,
            gc_objects=len(gc.get_objects()),
            gc_collections=gc_stats,
        )

    def set_baseline(self, label: str = "baseline") -> MemorySnapshot:
        """
        Set the baseline memory measurement.

        Args:
            label: Label for the baseline snapshot.

        Returns:
            The baseline snapshot.
        """
        gc.collect()
        self._baseline = self._create_snapshot(label)
        self._snapshots.append(self._baseline)

        logger.info(
            "memory_baseline_set",
            profiler=self.name,
            rss_mb=round(self._baseline.rss_mb, 2),
            gc_objects=self._baseline.gc_objects,
        )
        return self._baseline

    def snapshot(self, label: str) -> MemorySnapshot:
        """
        Take a memory snapshot.

        Args:
            label: Label for this snapshot.

        Returns:
            The snapshot taken.
        """
        snap = self._create_snapshot(label)
        self._snapshots.append(snap)

        delta_mb = 0.0
        if self._baseline:
            delta_mb = snap.rss_mb - self._baseline.rss_mb

        logger.debug(
            "memory_snapshot_taken",
            profiler=self.name,
            label=label,
            rss_mb=round(snap.rss_mb, 2),
            delta_mb=round(delta_mb, 2),
        )
        return snap

    def force_gc(self) -> dict[str, int]:
        """
        Force garbage collection and return collection counts.

        Returns:
            Dictionary with collection counts per generation.
        """
        before = gc.get_count()
        collected = gc.collect()
        after = gc.get_count()

        result = {
            "collected": collected,
            "gen0_before": before[0],
            "gen0_after": after[0],
            "gen1_before": before[1],
            "gen1_after": after[1],
            "gen2_before": before[2],
            "gen2_after": after[2],
        }

        logger.debug("gc_forced", **result)
        return result

    @property
    def peak_rss_mb(self) -> float:
        """Get peak RSS across all snapshots."""
        if not self._snapshots:
            return 0.0
        return max(s.rss_mb for s in self._snapshots)

    @property
    def total_growth_mb(self) -> float:
        """Get total memory growth from baseline."""
        if not self._baseline or len(self._snapshots) < 2:
            return 0.0
        latest = self._snapshots[-1]
        return latest.rss_mb - self._baseline.rss_mb

    def get_summary(self) -> dict[str, Any]:
        """Get profiling summary."""
        if not self._snapshots:
            return {"name": self.name, "snapshots": 0}

        baseline_mb = self._baseline.rss_mb if self._baseline else 0.0
        latest = self._snapshots[-1]

        return {
            "name": self.name,
            "snapshots": len(self._snapshots),
            "baseline_rss_mb": round(baseline_mb, 2),
            "current_rss_mb": round(latest.rss_mb, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "total_growth_mb": round(self.total_growth_mb, 2),
            "gc_objects": latest.gc_objects,
        }

    def get_all_snapshots(self) -> list[dict[str, Any]]:
        """Get all snapshots as dictionaries."""
        return [s.to_dict() for s in self._snapshots]

    @contextmanager
    def track(self, label: str) -> Generator[MemorySnapshot, None, None]:
        """
        Context manager for tracking memory during an operation.

        Args:
            label: Label for this tracking session.

        Yields:
            The before snapshot.
        """
        before = self.snapshot(f"{label}_before")
        try:
            yield before
        finally:
            self.snapshot(f"{label}_after")


@contextmanager
def profile_memory(
    name: str = "profile",
) -> Generator[MemoryProfiler, None, None]:
    """
    Context manager for memory profiling.

    Args:
        name: Name for the profiler.

    Yields:
        A MemoryProfiler instance.

    Example:
        with profile_memory("my_operation") as profiler:
            profiler.set_baseline()
            do_something()
            profiler.snapshot("after_operation")
        print(profiler.get_summary())
    """
    profiler = MemoryProfiler(name)
    try:
        yield profiler
    finally:
        summary = profiler.get_summary()
        logger.info("memory_profile_complete", **summary)


def memory_profile(func: F) -> F:
    """
    Decorator for memory profiling a function.

    Logs memory usage before and after function execution.

    Args:
        func: Function to profile.

    Returns:
        Wrapped function with memory profiling.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        profiler = MemoryProfiler(func.__name__)
        profiler.set_baseline("before")

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.snapshot("after")
            summary = profiler.get_summary()
            logger.info(
                "function_memory_profile",
                function=func.__name__,
                growth_mb=summary["total_growth_mb"],
                peak_mb=summary["peak_rss_mb"],
            )

    return wrapper  # type: ignore[return-value]
