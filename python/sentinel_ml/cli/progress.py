"""
Progress tracking and spinners for CLI.

Provides visual feedback for long-running operations.
Uses the rich library for beautiful progress bars and spinners.

Design Pattern: Observer Pattern for progress updates.
"""

from __future__ import annotations

import contextlib
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

logger = get_logger(__name__)


class SpinnerType(str, Enum):
    """Available spinner types."""

    DOTS = "dots"
    LINE = "line"
    ARROW = "arrow"
    BOUNCE = "bounce"
    CLOCK = "clock"


# Spinner frames for each type
SPINNER_FRAMES: dict[SpinnerType, list[str]] = {
    SpinnerType.DOTS: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    SpinnerType.LINE: ["-", "\\", "|", "/"],
    SpinnerType.ARROW: ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
    SpinnerType.BOUNCE: ["⠁", "⠂", "⠄", "⠂"],
    SpinnerType.CLOCK: ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"],
}


@dataclass
class TaskProgress:
    """
    Progress state for a single task.

    Attributes:
        name: Task name for display.
        total: Total number of items (0 = indeterminate).
        completed: Number of completed items.
        status: Current status message.
        started_at: Start timestamp.
        finished_at: Finish timestamp.
    """

    name: str
    total: int = 0
    completed: int = 0
    status: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.completed / self.total) * 100)

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.finished_at is not None

    @property
    def items_per_second(self) -> float:
        """Calculate processing rate."""
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.completed / elapsed

    @property
    def eta_seconds(self) -> float | None:
        """Estimate remaining time in seconds."""
        if self.total <= 0 or self.completed <= 0:
            return None
        rate = self.items_per_second
        if rate <= 0:
            return None
        remaining = self.total - self.completed
        return remaining / rate

    def update(self, completed: int | None = None, status: str | None = None) -> None:
        """Update progress state."""
        if completed is not None:
            self.completed = completed
        if status is not None:
            self.status = status

    def advance(self, amount: int = 1) -> None:
        """Advance completed count."""
        self.completed += amount

    def finish(self, status: str | None = None) -> None:
        """Mark task as finished."""
        self.finished_at = time.time()
        if status:
            self.status = status


class ProgressTracker:
    """
    Track and display progress for multiple tasks.

    Provides a simple interface for tracking progress of long-running
    operations with visual feedback.

    Example:
        tracker = ProgressTracker()
        with tracker.task("Processing", total=100) as task:
            for item in items:
                process(item)
                task.advance()
    """

    def __init__(
        self,
        *,
        show_eta: bool = True,
        show_rate: bool = True,
        show_percentage: bool = True,
        spinner_type: SpinnerType = SpinnerType.DOTS,
        use_colors: bool = True,
        output: Any = None,
    ) -> None:
        """
        Initialize progress tracker.

        Args:
            show_eta: Show estimated time remaining.
            show_rate: Show processing rate.
            show_percentage: Show percentage complete.
            spinner_type: Type of spinner for indeterminate progress.
            use_colors: Use colored output.
            output: Output stream (default: sys.stderr).
        """
        self.show_eta = show_eta
        self.show_rate = show_rate
        self.show_percentage = show_percentage
        self.spinner_type = spinner_type
        self.use_colors = use_colors
        self.output = output or sys.stderr

        self._tasks: dict[str, TaskProgress] = {}
        self._active_task: str | None = None
        self._spinner_frame: int = 0

        logger.debug(
            "progress_tracker_initialized",
            spinner_type=spinner_type.value,
            use_colors=use_colors,
        )

    @contextmanager
    def task(
        self,
        name: str,
        total: int = 0,
        status: str = "",
    ) -> Generator[TaskProgress, None, None]:
        """
        Context manager for tracking a task.

        Args:
            name: Task name for display.
            total: Total items (0 = indeterminate).
            status: Initial status message.

        Yields:
            TaskProgress object for updating progress.

        Example:
            with tracker.task("Loading", total=100) as task:
                for i in range(100):
                    do_work()
                    task.advance()
        """
        task = TaskProgress(name=name, total=total, status=status)
        self._tasks[name] = task
        self._active_task = name

        logger.info("task_started", task=name, total=total)

        try:
            self._render_start(task)
            yield task
            task.finish(status="Done" if not task.status else task.status)
            self._render_finish(task)
            logger.info(
                "task_completed",
                task=name,
                elapsed_seconds=task.elapsed_seconds,
                items=task.completed,
            )
        except Exception as e:
            task.finish(status=f"Failed: {e}")
            self._render_finish(task, success=False)
            logger.error("task_failed", task=name, error=str(e))
            raise
        finally:
            self._active_task = None

    def _render_start(self, task: TaskProgress) -> None:
        """Render task start."""
        if task.total > 0:
            line = f"⏳ {task.name} [0/{task.total}]"
        else:
            line = f"⏳ {task.name}..."

        if task.status:
            line += f" - {task.status}"

        self._write(line)

    def _render_finish(self, task: TaskProgress, *, success: bool = True) -> None:
        """Render task completion."""
        elapsed = task.elapsed_seconds

        if success:
            icon = "✅" if self.use_colors else "[OK]"
            status = task.status or "Done"
        else:
            icon = "❌" if self.use_colors else "[FAIL]"
            status = task.status or "Failed"

        line = f"{icon} {task.name}: {status} ({elapsed:.2f}s)"

        if task.completed > 0:
            rate = task.items_per_second
            line += f" - {task.completed} items ({rate:.1f}/s)"

        self._write(line)

    def _write(self, text: str) -> None:
        """Write to output stream."""
        self.output.write(text + "\n")
        self.output.flush()

    def create_progress_bar(
        self,
        current: int,
        total: int,
        width: int = 40,
    ) -> str:
        """
        Create a text-based progress bar.

        Args:
            current: Current progress value.
            total: Total value.
            width: Width of the bar in characters.

        Returns:
            Progress bar string.
        """
        if total <= 0:
            return self._get_spinner()

        percentage = min(1.0, current / total)
        filled = int(percentage * width)
        empty = width - filled

        bar = "█" * filled + "░" * empty
        pct_text = f"{percentage * 100:.1f}%"

        return f"[{bar}] {pct_text}"

    def _get_spinner(self) -> str:
        """Get current spinner frame."""
        frames = SPINNER_FRAMES[self.spinner_type]
        frame = frames[self._spinner_frame % len(frames)]
        self._spinner_frame += 1
        return frame

    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration for display.

        Args:
            seconds: Duration in seconds.

        Returns:
            Human-readable duration string.
        """
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

    @staticmethod
    def format_rate(items_per_second: float, unit: str = "items") -> str:
        """
        Format rate for display.

        Args:
            items_per_second: Rate in items per second.
            unit: Unit name (e.g., "items", "records", "logs").

        Returns:
            Human-readable rate string.
        """
        if items_per_second >= 1000000:
            return f"{items_per_second / 1000000:.1f}M {unit}/s"
        if items_per_second >= 1000:
            return f"{items_per_second / 1000:.1f}K {unit}/s"
        if items_per_second >= 1:
            return f"{items_per_second:.1f} {unit}/s"
        return f"{items_per_second * 60:.1f} {unit}/min"


class SpinnerContext:
    """
    Simple spinner for indeterminate operations.

    A lighter-weight alternative to ProgressTracker for simple
    operations that just need a spinner.

    Example:
        with SpinnerContext("Loading...") as spinner:
            do_something()
            spinner.update("Processing...")
    """

    def __init__(
        self,
        message: str,
        spinner_type: SpinnerType = SpinnerType.DOTS,
        output: Any = None,
    ) -> None:
        """
        Initialize spinner context.

        Args:
            message: Message to display.
            spinner_type: Type of spinner animation.
            output: Output stream.
        """
        self.message = message
        self.spinner_type = spinner_type
        self.output = output or sys.stderr
        self._frames = SPINNER_FRAMES[spinner_type]
        self._frame_idx = 0
        self._started_at = 0.0

    def __enter__(self) -> SpinnerContext:
        """Start spinner."""
        self._started_at = time.time()
        self._write_start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop spinner."""
        elapsed = time.time() - self._started_at

        if exc_type is None:
            self._write_end(success=True, elapsed=elapsed)
        else:
            self._write_end(success=False, elapsed=elapsed)

    def update(self, message: str) -> None:
        """Update spinner message."""
        self.message = message

    def _write_start(self) -> None:
        """Write initial spinner state."""
        frame = self._frames[0]
        self.output.write(f"{frame} {self.message}")
        self.output.flush()

    def _write_end(self, *, success: bool, elapsed: float) -> None:
        """Write final spinner state."""
        icon = "✓" if success else "✗"
        self.output.write(f"\r{icon} {self.message} ({elapsed:.2f}s)\n")
        self.output.flush()


@contextmanager
def spinner(
    message: str,
    spinner_type: SpinnerType = SpinnerType.DOTS,
) -> Iterator[SpinnerContext]:
    """
    Convenience function for creating a spinner context.

    Args:
        message: Message to display.
        spinner_type: Type of spinner.

    Yields:
        SpinnerContext for updating the message.

    Example:
        with spinner("Processing...") as s:
            for item in items:
                process(item)
                s.update(f"Processing {item}...")
    """
    ctx = SpinnerContext(message, spinner_type)
    with ctx:
        yield ctx


@contextmanager
def timed_operation(name: str) -> Generator[dict[str, Any], None, None]:
    """
    Context manager for timing operations.

    Captures timing information for profiling.

    Args:
        name: Operation name.

    Yields:
        Dictionary to store timing results.

    Example:
        with timed_operation("data_loading") as timing:
            load_data()
        print(f"Loading took {timing['duration_ms']:.2f}ms")
    """
    result: dict[str, Any] = {"name": name, "started_at": time.time()}
    logger.debug("operation_started", operation=name)

    try:
        yield result
    finally:
        end_time = time.time()
        result["ended_at"] = end_time
        result["duration_ms"] = (end_time - result["started_at"]) * 1000

        logger.debug(
            "operation_completed",
            operation=name,
            duration_ms=result["duration_ms"],
        )
