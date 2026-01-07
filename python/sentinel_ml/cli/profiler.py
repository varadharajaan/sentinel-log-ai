"""
Profiler for timing and performance analysis.

Provides detailed timing breakdown for operations,
useful for performance debugging and optimization.

Design Pattern: Decorator Pattern for transparent timing.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from sentinel_ml.cli.formatters import ProfileFormatter, ProfileTiming
from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

logger = get_logger(__name__)

F = TypeVar("F", bound="Callable[..., Any]")


@dataclass
class TimingEntry:
    """A single timing entry."""

    name: str
    start_time: float
    end_time: float | None = None
    parent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class Profiler:
    """
    Performance profiler for timing operations.

    Tracks nested timing information and generates
    hierarchical timing reports.

    Example:
        profiler = Profiler()

        with profiler.measure("total"):
            with profiler.measure("loading"):
                load_data()
            with profiler.measure("processing"):
                process_data()

        profiler.print_report()
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        threshold_ms: float = 0.0,
    ) -> None:
        """
        Initialize profiler.

        Args:
            enabled: Whether profiling is enabled.
            threshold_ms: Minimum duration to report (ms).
        """
        self.enabled = enabled
        self.threshold_ms = threshold_ms
        self._entries: list[TimingEntry] = []
        self._stack: list[str] = []
        self._start_time: float | None = None

        logger.debug(
            "profiler_initialized",
            enabled=enabled,
            threshold_ms=threshold_ms,
        )

    @contextmanager
    def measure(
        self,
        name: str,
        **metadata: Any,
    ) -> Generator[TimingEntry, None, None]:
        """
        Context manager for measuring operation timing.

        Args:
            name: Operation name.
            **metadata: Additional metadata to store.

        Yields:
            TimingEntry for the operation.

        Example:
            with profiler.measure("data_loading", source="file.csv"):
                load_data()
        """
        if not self.enabled:
            yield TimingEntry(name=name, start_time=0)
            return

        # Initialize start time if this is the first measurement
        if self._start_time is None:
            self._start_time = time.perf_counter()

        # Create entry
        parent = self._stack[-1] if self._stack else None
        entry = TimingEntry(
            name=name,
            start_time=time.perf_counter(),
            parent=parent,
            metadata=metadata,
        )

        # Push to stack
        self._stack.append(name)

        logger.debug("profiler_start", operation=name, parent=parent)

        try:
            yield entry
        finally:
            # Pop from stack
            self._stack.pop()

            # Record end time
            entry.end_time = time.perf_counter()

            # Only record if above threshold
            if entry.duration_ms >= self.threshold_ms:
                self._entries.append(entry)

            logger.debug(
                "profiler_end",
                operation=name,
                duration_ms=entry.duration_ms,
            )

    def profile(self, name: str | None = None) -> Callable[[F], F]:
        """
        Decorator for profiling functions.

        Args:
            name: Operation name (defaults to function name).

        Returns:
            Decorated function.

        Example:
            @profiler.profile()
            def expensive_operation():
                ...
        """

        def decorator(func: F) -> F:
            operation_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.measure(operation_name):
                    return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def get_timings(self) -> list[ProfileTiming]:
        """
        Get timing data in report format.

        Returns:
            List of ProfileTiming objects for formatting.
        """
        # Build hierarchy
        root_entries = [e for e in self._entries if e.parent is None]
        total_ms = sum(e.duration_ms for e in root_entries)

        result: list[ProfileTiming] = []
        for entry in root_entries:
            timing = self._build_timing(entry, total_ms)
            result.append(timing)

        return result

    def _build_timing(
        self,
        entry: TimingEntry,
        total_ms: float,
    ) -> ProfileTiming:
        """Build ProfileTiming from entry."""
        children = [e for e in self._entries if e.parent == entry.name]
        child_timings = [self._build_timing(c, total_ms) for c in children]

        percentage = (entry.duration_ms / total_ms * 100) if total_ms > 0 else 0

        return ProfileTiming(
            name=entry.name,
            duration_ms=entry.duration_ms,
            percentage=percentage,
            children=child_timings,
        )

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with summary stats.
        """
        if not self._entries:
            return {"total_ms": 0, "operations": 0}

        durations = [e.duration_ms for e in self._entries]

        return {
            "total_ms": sum(durations),
            "operations": len(self._entries),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "avg_ms": sum(durations) / len(durations),
        }

    def format_report(self) -> str:
        """
        Format timing report as string.

        Returns:
            Formatted report string.
        """
        timings = self.get_timings()
        formatter = ProfileFormatter()
        return formatter.format(timings)

    def print_report(self) -> None:
        """Print timing report to stdout."""
        print(self.format_report())

    def clear(self) -> None:
        """Clear all timing data."""
        self._entries.clear()
        self._stack.clear()
        self._start_time = None
        logger.debug("profiler_cleared")

    def to_dict(self) -> dict[str, Any]:
        """
        Export timing data as dictionary.

        Returns:
            Dictionary with all timing entries.
        """
        return {
            "entries": [
                {
                    "name": e.name,
                    "duration_ms": e.duration_ms,
                    "parent": e.parent,
                    "metadata": e.metadata,
                }
                for e in self._entries
            ],
            "summary": self.get_summary(),
        }


# Global profiler instance
_global_profiler: Profiler | None = None


def get_profiler() -> Profiler:
    """
    Get the global profiler instance.

    Returns:
        Global Profiler instance.
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = Profiler(enabled=False)
    return _global_profiler


def enable_profiling(threshold_ms: float = 0.0) -> Profiler:
    """
    Enable global profiling.

    Args:
        threshold_ms: Minimum duration to report.

    Returns:
        Configured Profiler instance.
    """
    global _global_profiler
    _global_profiler = Profiler(enabled=True, threshold_ms=threshold_ms)
    logger.info("profiling_enabled", threshold_ms=threshold_ms)
    return _global_profiler


def disable_profiling() -> None:
    """Disable global profiling."""
    global _global_profiler
    if _global_profiler:
        _global_profiler.enabled = False
    logger.info("profiling_disabled")


@contextmanager
def measure(name: str, **metadata: Any) -> Generator[TimingEntry, None, None]:
    """
    Convenience function for profiling.

    Uses the global profiler instance.

    Args:
        name: Operation name.
        **metadata: Additional metadata.

    Yields:
        TimingEntry for the operation.

    Example:
        with measure("loading"):
            load_data()
    """
    profiler = get_profiler()
    with profiler.measure(name, **metadata) as entry:
        yield entry


def profile(name: str | None = None) -> Callable[[F], F]:
    """
    Decorator for profiling functions.

    Uses the global profiler instance.

    Args:
        name: Operation name.

    Returns:
        Decorated function.

    Example:
        @profile()
        def expensive_function():
            ...
    """

    def decorator(func: F) -> F:
        operation_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = get_profiler()
            with profiler.measure(operation_name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
