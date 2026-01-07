"""
Benchmark runner for performance testing.

Provides a framework for running benchmarks with configurable
parameters and result collection.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

from sentinel_ml.benchmark.metrics import MetricsCollector
from sentinel_ml.benchmark.profiler import MemoryProfiler

logger = get_logger(__name__)


class BenchmarkStatus(Enum):
    """Status of a benchmark run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BenchmarkConfig:
    """
    Configuration for a benchmark run.

    Attributes:
        name: Name of the benchmark.
        warmup_iterations: Number of warmup runs before measurement.
        iterations: Number of measured iterations.
        timeout_seconds: Maximum time per iteration.
        collect_memory: Whether to collect memory metrics.
        gc_between_iterations: Whether to run GC between iterations.
        params: Additional benchmark parameters.
    """

    name: str
    warmup_iterations: int = 3
    iterations: int = 10
    timeout_seconds: float = 300.0
    collect_memory: bool = True
    gc_between_iterations: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration values."""
        if self.warmup_iterations < 0:
            msg = "warmup_iterations must be non-negative"
            raise ValueError(msg)
        if self.iterations < 1:
            msg = "iterations must be at least 1"
            raise ValueError(msg)
        if self.timeout_seconds <= 0:
            msg = "timeout_seconds must be positive"
            raise ValueError(msg)


@dataclass
class BenchmarkResult:
    """
    Result of a benchmark run.

    Attributes:
        config: The benchmark configuration used.
        status: Final status of the benchmark.
        metrics: Collected metrics from the run.
        error: Error message if failed.
        start_time: When the benchmark started.
        end_time: When the benchmark ended.
        metadata: Additional result metadata.
    """

    config: BenchmarkConfig
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Total benchmark duration."""
        if self.end_time > self.start_time:
            return self.end_time - self.start_time
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.config.name,
            "status": self.status.value,
            "duration_seconds": round(self.duration_seconds, 3),
            "iterations": self.config.iterations,
            "metrics": self.metrics,
            "error": self.error,
            "params": self.config.params,
            "metadata": self.metadata,
        }


class BenchmarkRunner(ABC):
    """
    Abstract base class for benchmark runners.

    Implements Template Method pattern for benchmark execution.
    Subclasses implement the specific benchmark logic.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        config.validate()
        self.config = config
        self._metrics = MetricsCollector(name=config.name)
        self._profiler: MemoryProfiler | None = None
        self._result: BenchmarkResult | None = None
        logger.info(
            "benchmark_runner_initialized",
            name=config.name,
            iterations=config.iterations,
        )

    @abstractmethod
    def setup(self) -> None:
        """
        Setup before benchmark runs.

        Override to initialize resources needed for the benchmark.
        """

    @abstractmethod
    def run_iteration(self, iteration: int) -> None:
        """
        Run a single benchmark iteration.

        Args:
            iteration: Current iteration number (0-indexed).
        """

    @abstractmethod
    def teardown(self) -> None:
        """
        Cleanup after benchmark runs.

        Override to release resources.
        """

    def _warmup(self) -> None:
        """Run warmup iterations."""
        for i in range(self.config.warmup_iterations):
            logger.debug(
                "warmup_iteration",
                name=self.config.name,
                iteration=i + 1,
                total=self.config.warmup_iterations,
            )
            self.run_iteration(i)

    def _run_measured(self) -> None:
        """Run measured iterations with metric collection."""
        import gc

        for i in range(self.config.iterations):
            if self.config.gc_between_iterations:
                gc.collect()

            with self._metrics.time_operation(f"iteration_{i}"):
                self.run_iteration(i)

            if self._profiler and self.config.collect_memory:
                self._profiler.snapshot(f"iteration_{i}")

            logger.debug(
                "measured_iteration",
                name=self.config.name,
                iteration=i + 1,
                total=self.config.iterations,
            )

    def execute(self) -> BenchmarkResult:
        """
        Execute the full benchmark.

        Returns:
            BenchmarkResult with collected metrics.
        """
        result = BenchmarkResult(config=self.config)
        result.start_time = time.time()
        result.status = BenchmarkStatus.RUNNING

        logger.info(
            "benchmark_started",
            name=self.config.name,
            iterations=self.config.iterations,
            warmup=self.config.warmup_iterations,
        )

        try:
            self.setup()

            if self.config.collect_memory:
                self._profiler = MemoryProfiler(self.config.name)
                self._profiler.set_baseline()

            if self.config.warmup_iterations > 0:
                self._warmup()

            self._run_measured()

            result.metrics = self._collect_metrics()
            result.status = BenchmarkStatus.COMPLETED

            logger.info(
                "benchmark_completed",
                name=self.config.name,
                duration=round(time.time() - result.start_time, 3),
            )

        except Exception as exc:
            result.status = BenchmarkStatus.FAILED
            result.error = str(exc)
            logger.error(
                "benchmark_failed",
                name=self.config.name,
                error=str(exc),
            )
            raise

        finally:
            try:
                self.teardown()
            except Exception as teardown_exc:
                logger.warning(
                    "benchmark_teardown_error",
                    name=self.config.name,
                    error=str(teardown_exc),
                )
            result.end_time = time.time()

        self._result = result
        return result

    def _collect_metrics(self) -> dict[str, Any]:
        """Collect all metrics into a dictionary."""
        metrics: dict[str, Any] = self._metrics.to_dict()

        if self._profiler:
            metrics["memory"] = self._profiler.get_summary()

        return metrics


class FunctionBenchmark(BenchmarkRunner):
    """
    Benchmark runner for a single function.

    Wraps any callable for benchmarking.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        config: BenchmarkConfig,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize function benchmark.

        Args:
            func: Function to benchmark.
            config: Benchmark configuration.
            args: Positional arguments for the function.
            kwargs: Keyword arguments for the function.
        """
        super().__init__(config)
        self._func = func
        self._args = args or ()
        self._kwargs = kwargs or {}

    def setup(self) -> None:
        """No setup needed for function benchmarks."""

    def run_iteration(self, iteration: int) -> None:  # noqa: ARG002
        """Run the function once."""
        self._func(*self._args, **self._kwargs)

    def teardown(self) -> None:
        """No teardown needed for function benchmarks."""


class BenchmarkSuite:
    """
    Collection of benchmarks to run together.

    Manages multiple benchmarks and aggregates results.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize benchmark suite.

        Args:
            name: Name of the suite.
        """
        self.name = name
        self._benchmarks: list[BenchmarkRunner] = []
        self._results: list[BenchmarkResult] = []
        logger.info("benchmark_suite_created", name=name)

    def add(self, benchmark: BenchmarkRunner) -> BenchmarkSuite:
        """
        Add a benchmark to the suite.

        Args:
            benchmark: Benchmark runner to add.

        Returns:
            Self for method chaining.
        """
        self._benchmarks.append(benchmark)
        logger.debug(
            "benchmark_added_to_suite",
            suite=self.name,
            benchmark=benchmark.config.name,
        )
        return self

    def add_function(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        iterations: int = 10,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkSuite:
        """
        Add a function benchmark to the suite.

        Args:
            func: Function to benchmark.
            name: Benchmark name (defaults to function name).
            iterations: Number of iterations.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Self for method chaining.
        """
        config = BenchmarkConfig(
            name=name or func.__name__,
            iterations=iterations,
        )
        benchmark = FunctionBenchmark(func, config, args, kwargs)
        return self.add(benchmark)

    def run_all(self, stop_on_failure: bool = False) -> list[BenchmarkResult]:
        """
        Run all benchmarks in the suite.

        Args:
            stop_on_failure: Whether to stop if a benchmark fails.

        Returns:
            List of benchmark results.
        """
        self._results = []
        total = len(self._benchmarks)

        logger.info(
            "benchmark_suite_started",
            suite=self.name,
            total_benchmarks=total,
        )

        for i, benchmark in enumerate(self._benchmarks):
            logger.info(
                "running_benchmark",
                suite=self.name,
                benchmark=benchmark.config.name,
                index=i + 1,
                total=total,
            )

            try:
                result = benchmark.execute()
                self._results.append(result)
            except Exception:
                if stop_on_failure:
                    logger.error(
                        "suite_stopped_on_failure",
                        suite=self.name,
                        failed_benchmark=benchmark.config.name,
                    )
                    break
                result = BenchmarkResult(
                    config=benchmark.config,
                    status=BenchmarkStatus.FAILED,
                )
                self._results.append(result)

        logger.info(
            "benchmark_suite_completed",
            suite=self.name,
            completed=len(self._results),
            total=total,
        )

        return self._results

    def get_summary(self) -> dict[str, Any]:
        """Get suite summary."""
        completed = sum(
            1 for r in self._results if r.status == BenchmarkStatus.COMPLETED
        )
        failed = sum(1 for r in self._results if r.status == BenchmarkStatus.FAILED)

        return {
            "name": self.name,
            "total_benchmarks": len(self._benchmarks),
            "completed": completed,
            "failed": failed,
            "results": [r.to_dict() for r in self._results],
        }
