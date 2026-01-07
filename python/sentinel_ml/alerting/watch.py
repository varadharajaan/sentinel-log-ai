"""
Watch mode daemon for continuous log monitoring.

Monitors log directories for new entries and triggers alerts
for novel events detected by the ML pipeline.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

import structlog

from sentinel_ml.alerting.base import AlertEvent, AlertPriority

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sentinel_ml.alerting.base import BaseNotifier

logger = structlog.get_logger(__name__)


class WatchState(str, Enum):
    """State of the watch daemon."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class WatchEvent:
    """
    Event detected by the watch daemon.

    Attributes:
        file_path: Path to the file that changed.
        event_type: Type of change (created, modified, deleted).
        timestamp: When the event occurred.
        lines_added: Number of new lines detected.
        novel_count: Number of novel events detected.
    """

    file_path: Path
    event_type: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    lines_added: int = 0
    novel_count: int = 0


@dataclass
class WatchConfig:
    """
    Configuration for watch daemon.

    Attributes:
        watch_paths: Directories or files to monitor.
        file_patterns: Glob patterns to match (e.g., *.log).
        poll_interval_seconds: How often to check for changes.
        novelty_threshold: Minimum novelty score to trigger alert.
        batch_size: Maximum lines to process per cycle.
        max_file_size_mb: Skip files larger than this.
        recursive: Whether to watch subdirectories.
        enabled: Whether daemon is active.
    """

    watch_paths: list[Path] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=lambda: ["*.log", "*.jsonl"])
    poll_interval_seconds: float = 10.0
    novelty_threshold: float = 0.5
    batch_size: int = 100
    max_file_size_mb: float = 100.0
    recursive: bool = True
    enabled: bool = True


class WatchDaemon:
    """
    Watch mode daemon for continuous monitoring.

    Monitors specified paths for new log entries and integrates
    with the ML pipeline to detect and alert on novel events.
    """

    def __init__(
        self,
        config: WatchConfig,
        notifiers: Sequence[BaseNotifier] | None = None,
        novelty_detector: Callable[[str], float] | None = None,
    ) -> None:
        """
        Initialize watch daemon.

        Args:
            config: Watch configuration.
            notifiers: List of notifiers for alert delivery.
            novelty_detector: Function to compute novelty score.
        """
        self._config = config
        self._notifiers = list(notifiers) if notifiers else []
        self._novelty_detector = novelty_detector or self._default_detector
        self._logger = logger.bind(component="watch-daemon")

        self._state = WatchState.STOPPED
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._file_positions: dict[Path, int] = {}
        self._stats = WatchStats()

    @property
    def state(self) -> WatchState:
        """Get current daemon state."""
        return self._state

    @property
    def stats(self) -> WatchStats:
        """Get daemon statistics."""
        return self._stats

    def add_notifier(self, notifier: BaseNotifier) -> None:
        """Add a notifier for alert delivery."""
        self._notifiers.append(notifier)
        self._logger.debug(
            "notifier_added",
            notifier=notifier.name,
        )

    def start(self) -> None:
        """
        Start the watch daemon.

        Raises:
            RuntimeError: If daemon is already running.
        """
        if self._state == WatchState.RUNNING:
            raise RuntimeError("Daemon is already running")

        self._state = WatchState.STARTING
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run,
            name="sentinel-watch-daemon",
            daemon=True,
        )
        self._thread.start()

        self._state = WatchState.RUNNING
        self._logger.info(
            "watch_daemon_started",
            paths=[str(p) for p in self._config.watch_paths],
            poll_interval=self._config.poll_interval_seconds,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the watch daemon.

        Args:
            timeout: Maximum time to wait for shutdown.
        """
        if self._state != WatchState.RUNNING:
            return

        self._state = WatchState.STOPPING
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self._logger.warning("watch_daemon_stop_timeout")
            self._thread = None

        self._state = WatchState.STOPPED
        self._logger.info("watch_daemon_stopped")

    def _run(self) -> None:
        """Main daemon loop."""
        try:
            while not self._stop_event.is_set():
                try:
                    self._poll_cycle()
                except Exception as e:
                    self._logger.error(
                        "watch_poll_error",
                        error=str(e),
                    )
                    self._stats.error_count += 1

                self._stop_event.wait(self._config.poll_interval_seconds)

        except Exception as e:
            self._logger.error(
                "watch_daemon_fatal",
                error=str(e),
            )
            self._state = WatchState.ERROR

    def _poll_cycle(self) -> None:
        """Execute one polling cycle."""
        self._stats.poll_count += 1
        files = self._discover_files()

        for file_path in files:
            if self._stop_event.is_set():
                break

            try:
                new_lines = self._read_new_lines(file_path)
                if new_lines:
                    self._process_lines(file_path, new_lines)
            except Exception as e:
                self._logger.warning(
                    "file_read_error",
                    path=str(file_path),
                    error=str(e),
                )

    def _discover_files(self) -> list[Path]:
        """Discover files matching patterns in watch paths."""
        files: list[Path] = []
        max_size = self._config.max_file_size_mb * 1024 * 1024

        for watch_path in self._config.watch_paths:
            if not watch_path.exists():
                continue

            if watch_path.is_file():
                if watch_path.stat().st_size <= max_size:
                    files.append(watch_path)
                continue

            for pattern in self._config.file_patterns:
                if self._config.recursive:
                    matched = watch_path.rglob(pattern)
                else:
                    matched = watch_path.glob(pattern)

                for match in matched:
                    if match.is_file() and match.stat().st_size <= max_size:
                        files.append(match)

        return files

    def _read_new_lines(self, file_path: Path) -> list[str]:
        """Read new lines from file since last position."""
        current_size = file_path.stat().st_size
        last_pos = self._file_positions.get(file_path, 0)

        # Handle file truncation
        if current_size < last_pos:
            self._logger.debug(
                "file_truncated",
                path=str(file_path),
            )
            last_pos = 0

        if current_size == last_pos:
            return []

        new_lines: list[str] = []

        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.rstrip("\n\r")
                    if line:
                        new_lines.append(line)
                        if len(new_lines) >= self._config.batch_size:
                            break
                self._file_positions[file_path] = f.tell()

        except OSError as e:
            self._logger.warning(
                "file_read_failed",
                path=str(file_path),
                error=str(e),
            )

        self._stats.lines_processed += len(new_lines)
        return new_lines

    def _process_lines(self, file_path: Path, lines: list[str]) -> None:
        """Process new lines and detect novel events."""
        novel_events: list[AlertEvent] = []

        for line in lines:
            try:
                score = self._novelty_detector(line)

                if score >= self._config.novelty_threshold:
                    event = AlertEvent(
                        title=f"Novel event detected in {file_path.name}",
                        message=line[:500],  # Truncate long lines
                        priority=AlertPriority.from_score(score),
                        source=str(file_path),
                        metadata={
                            "novelty_score": score,
                            "file_path": str(file_path),
                            "line_preview": line[:100],
                        },
                        tags=["novel-event", file_path.suffix.lstrip(".")],
                    )
                    novel_events.append(event)
                    self._stats.novel_count += 1

            except Exception as e:
                self._logger.warning(
                    "novelty_detection_failed",
                    error=str(e),
                )

        # Send alerts
        for event in novel_events:
            self._send_alerts(event)

    def _send_alerts(self, event: AlertEvent) -> None:
        """Send event to all notifiers."""
        for notifier in self._notifiers:
            if notifier.is_enabled:
                try:
                    result = notifier.send(event)
                    if result.is_success:
                        self._stats.alerts_sent += 1
                    else:
                        self._stats.alerts_failed += 1
                except Exception as e:
                    self._logger.error(
                        "notifier_error",
                        notifier=notifier.name,
                        error=str(e),
                    )
                    self._stats.alerts_failed += 1

    def _default_detector(self, line: str) -> float:  # noqa: ARG002
        """Default novelty detector (always returns 0)."""
        return 0.0

    def get_status(self) -> dict[str, Any]:
        """Get daemon status summary."""
        return {
            "state": self._state.value,
            "stats": {
                "poll_count": self._stats.poll_count,
                "lines_processed": self._stats.lines_processed,
                "novel_count": self._stats.novel_count,
                "alerts_sent": self._stats.alerts_sent,
                "alerts_failed": self._stats.alerts_failed,
                "error_count": self._stats.error_count,
            },
            "watched_files": len(self._file_positions),
            "notifiers": len(self._notifiers),
            "config": {
                "poll_interval": self._config.poll_interval_seconds,
                "novelty_threshold": self._config.novelty_threshold,
            },
        }


@dataclass
class WatchStats:
    """Statistics for watch daemon."""

    poll_count: int = 0
    lines_processed: int = 0
    novel_count: int = 0
    alerts_sent: int = 0
    alerts_failed: int = 0
    error_count: int = 0
    started_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    @property
    def uptime_seconds(self) -> float:
        """Get daemon uptime in seconds."""
        delta = datetime.now(tz=timezone.utc) - self.started_at
        return delta.total_seconds()
