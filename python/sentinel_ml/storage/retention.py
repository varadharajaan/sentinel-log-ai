"""
Retention policy engine for automatic data cleanup.

This module implements configurable retention policies for managing
log data lifecycle based on age and size criteria.

Design Patterns:
- Strategy Pattern: Pluggable retention criteria
- Composite Pattern: Combine multiple criteria
- Observer Pattern: Notify on retention events
- Template Method: Common cleanup workflow
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = get_logger(__name__)


class RetentionEventType(str, Enum):
    """Types of retention events."""

    FILE_DELETED = "file_deleted"
    DIRECTORY_CLEANED = "directory_cleaned"
    RETENTION_STARTED = "retention_started"
    RETENTION_COMPLETED = "retention_completed"
    RETENTION_ERROR = "retention_error"


@dataclass
class RetentionEvent:
    """Event emitted during retention operations."""

    event_type: RetentionEventType
    path: Path | None = None
    size_bytes: int = 0
    age_days: float = 0.0
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type.value,
            "path": str(self.path) if self.path else None,
            "size_bytes": self.size_bytes,
            "age_days": round(self.age_days, 2),
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


class RetentionObserver(Protocol):
    """Protocol for retention event observers."""

    def on_retention_event(self, event: RetentionEvent) -> None:
        """Handle a retention event."""
        ...


@dataclass
class RetentionResult:
    """Result of a retention operation."""

    files_deleted: int = 0
    directories_cleaned: int = 0
    bytes_freed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None

    @property
    def success(self) -> bool:
        """Check if retention completed without errors."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "files_deleted": self.files_deleted,
            "directories_cleaned": self.directories_cleaned,
            "bytes_freed": self.bytes_freed,
            "bytes_freed_mb": round(self.bytes_freed / (1024 * 1024), 2),
            "errors": self.errors,
            "error_count": len(self.errors),
            "duration_seconds": round(self.duration_seconds, 3),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success": self.success,
        }


class RetentionCriteria(ABC):
    """Abstract base class for retention criteria."""

    @abstractmethod
    def should_delete(self, path: Path) -> bool:
        """
        Determine if a file should be deleted.

        Args:
            path: Path to the file.

        Returns:
            True if the file should be deleted.
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get a human-readable description of the criteria."""
        pass


@dataclass
class AgeCriteria(RetentionCriteria):
    """
    Age-based retention criteria.

    Deletes files older than the specified number of days.
    """

    max_age_days: float
    reference_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def should_delete(self, path: Path) -> bool:
        """Check if file is older than max age."""
        if not path.exists():
            return False

        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            age = self.reference_time - mtime
            return age.total_seconds() > self.max_age_days * 86400
        except OSError:
            logger.warning("age_criteria_stat_failed", path=str(path))
            return False

    def get_age_days(self, path: Path) -> float:
        """Get the age of a file in days."""
        if not path.exists():
            return 0.0

        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            age = self.reference_time - mtime
            return age.total_seconds() / 86400
        except OSError:
            return 0.0

    def get_description(self) -> str:
        """Get criteria description."""
        return f"Files older than {self.max_age_days} days"


@dataclass
class SizeCriteria(RetentionCriteria):
    """
    Size-based retention criteria.

    Deletes files when total directory size exceeds the threshold.
    """

    max_size_bytes: int
    directory: Path | None = None

    def should_delete(self, path: Path) -> bool:
        """
        Check if file should be deleted based on size.

        This criteria works at directory level - if the directory
        exceeds max size, oldest files are marked for deletion.
        """
        if self.directory is None:
            return False

        current_size = self._get_directory_size()
        return current_size > self.max_size_bytes and path.exists()

    def _get_directory_size(self) -> int:
        """Calculate total size of directory."""
        if self.directory is None or not self.directory.exists():
            return 0

        total_size = 0
        try:
            for entry in self.directory.rglob("*"):
                if entry.is_file():
                    with contextlib.suppress(OSError):
                        total_size += entry.stat().st_size
        except OSError:
            logger.warning("size_criteria_scan_failed", directory=str(self.directory))

        return total_size

    def get_current_size(self) -> int:
        """Get current directory size in bytes."""
        return self._get_directory_size()

    def get_description(self) -> str:
        """Get criteria description."""
        max_mb = self.max_size_bytes / (1024 * 1024)
        return f"Directory size exceeds {max_mb:.1f} MB"


@dataclass
class CompositeCriteria(RetentionCriteria):
    """
    Composite retention criteria combining multiple criteria.

    Uses logical AND or OR to combine criteria.
    """

    criteria: list[RetentionCriteria] = field(default_factory=list)
    require_all: bool = True  # AND if True, OR if False

    def should_delete(self, path: Path) -> bool:
        """Check if file meets composite criteria."""
        if not self.criteria:
            return False

        if self.require_all:
            return all(c.should_delete(path) for c in self.criteria)
        return any(c.should_delete(path) for c in self.criteria)

    def add_criteria(self, criteria: RetentionCriteria) -> None:
        """Add a criteria to the composite."""
        self.criteria.append(criteria)

    def get_description(self) -> str:
        """Get criteria description."""
        operator = " AND " if self.require_all else " OR "
        descriptions = [c.get_description() for c in self.criteria]
        return f"({operator.join(descriptions)})"


@dataclass
class RetentionConfig:
    """Configuration for retention policies."""

    max_age_days: float | None = None
    max_size_bytes: int | None = None
    max_size_mb: float | None = None
    file_patterns: list[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: list[str] = field(default_factory=list)
    dry_run: bool = False
    recursive: bool = True

    def __post_init__(self) -> None:
        """Convert max_size_mb to bytes if provided."""
        if self.max_size_mb is not None and self.max_size_bytes is None:
            self.max_size_bytes = int(self.max_size_mb * 1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_age_days": self.max_age_days,
            "max_size_bytes": self.max_size_bytes,
            "max_size_mb": self.max_size_mb,
            "file_patterns": self.file_patterns,
            "exclude_patterns": self.exclude_patterns,
            "dry_run": self.dry_run,
            "recursive": self.recursive,
        }


class RetentionPolicy:
    """
    Retention policy engine for automatic data cleanup.

    Implements the Template Method pattern for cleanup operations
    and Observer pattern for event notifications.
    """

    def __init__(
        self,
        config: RetentionConfig | None = None,
        observers: Sequence[RetentionObserver] | None = None,
    ) -> None:
        """
        Initialize the retention policy.

        Args:
            config: Retention configuration.
            observers: Event observers to notify.
        """
        self._config = config or RetentionConfig()
        self._observers: list[RetentionObserver] = list(observers) if observers else []
        self._criteria: list[RetentionCriteria] = []

        self._build_criteria()

        logger.info(
            "retention_policy_initialized",
            config=self._config.to_dict(),
            criteria_count=len(self._criteria),
        )

    def _build_criteria(self) -> None:
        """Build retention criteria from configuration."""
        if self._config.max_age_days is not None:
            self._criteria.append(AgeCriteria(max_age_days=self._config.max_age_days))

        if self._config.max_size_bytes is not None:
            self._criteria.append(SizeCriteria(max_size_bytes=self._config.max_size_bytes))

    def add_observer(self, observer: RetentionObserver) -> None:
        """Add an event observer."""
        self._observers.append(observer)

    def remove_observer(self, observer: RetentionObserver) -> None:
        """Remove an event observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_observers(self, event: RetentionEvent) -> None:
        """Notify all observers of an event."""
        for observer in self._observers:
            try:
                observer.on_retention_event(event)
            except Exception as e:
                logger.warning(
                    "observer_notification_failed",
                    observer=type(observer).__name__,
                    error=str(e),
                )

    def _should_include_file(self, path: Path) -> bool:
        """Check if file matches inclusion patterns."""

        # Check exclusion patterns first
        for pattern in self._config.exclude_patterns:
            if path.match(pattern):
                return False

        # Check inclusion patterns
        return any(path.match(pattern) for pattern in self._config.file_patterns)

    def _should_delete(self, path: Path) -> bool:
        """Check if a file should be deleted based on all criteria."""
        if not self._criteria:
            return False

        # Any criteria match triggers deletion (OR logic)
        return any(c.should_delete(path) for c in self._criteria)

    def _get_candidate_files(self, directory: Path) -> list[tuple[Path, float, int]]:
        """
        Get candidate files for deletion.

        Returns:
            List of (path, age_days, size_bytes) tuples sorted by age (oldest first).
        """
        candidates: list[tuple[Path, float, int]] = []

        if not directory.exists():
            return candidates

        try:
            if self._config.recursive:
                files = list(directory.rglob("*"))
            else:
                files = list(directory.glob("*"))

            for path in files:
                if not path.is_file():
                    continue

                if not self._should_include_file(path):
                    continue

                try:
                    stat = path.stat()
                    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    age = datetime.now(timezone.utc) - mtime
                    age_days = age.total_seconds() / 86400
                    candidates.append((path, age_days, stat.st_size))
                except OSError:
                    pass

        except OSError as e:
            logger.warning(
                "candidate_scan_failed",
                directory=str(directory),
                error=str(e),
            )

        # Sort by age descending (oldest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def apply(self, directory: Path) -> RetentionResult:
        """
        Apply retention policy to a directory.

        Args:
            directory: Directory to clean up.

        Returns:
            Result of the retention operation.
        """
        import time

        start_time = time.perf_counter()
        result = RetentionResult()

        self._notify_observers(
            RetentionEvent(
                event_type=RetentionEventType.RETENTION_STARTED,
                path=directory,
                message=f"Starting retention on {directory}",
            )
        )

        logger.info(
            "retention_started",
            directory=str(directory),
            config=self._config.to_dict(),
        )

        # Update size criteria with directory
        for criteria in self._criteria:
            if isinstance(criteria, SizeCriteria):
                criteria.directory = directory

        try:
            candidates = self._get_candidate_files(directory)
            current_size = sum(c[2] for c in candidates)

            logger.debug(
                "retention_candidates",
                directory=str(directory),
                candidate_count=len(candidates),
                current_size_bytes=current_size,
            )

            for path, age_days, size_bytes in candidates:
                should_delete = False

                # Check age criteria
                if self._config.max_age_days is not None and age_days > self._config.max_age_days:
                    should_delete = True

                # Check size criteria
                if self._config.max_size_bytes is not None:
                    remaining_size = current_size - result.bytes_freed
                    if remaining_size > self._config.max_size_bytes:
                        should_delete = True

                if not should_delete:
                    continue

                try:
                    if not self._config.dry_run:
                        path.unlink()

                    result.files_deleted += 1
                    result.bytes_freed += size_bytes

                    self._notify_observers(
                        RetentionEvent(
                            event_type=RetentionEventType.FILE_DELETED,
                            path=path,
                            size_bytes=size_bytes,
                            age_days=age_days,
                            message=f"Deleted file: {path.name}",
                        )
                    )

                    logger.debug(
                        "file_deleted",
                        path=str(path),
                        size_bytes=size_bytes,
                        age_days=round(age_days, 2),
                        dry_run=self._config.dry_run,
                    )

                except OSError as e:
                    error_msg = f"Failed to delete {path}: {e}"
                    result.errors.append(error_msg)
                    logger.warning("file_delete_failed", path=str(path), error=str(e))

            # Clean empty directories
            if not self._config.dry_run:
                result.directories_cleaned = self._clean_empty_directories(directory)

        except Exception as e:
            error_msg = f"Retention failed: {e}"
            result.errors.append(error_msg)

            self._notify_observers(
                RetentionEvent(
                    event_type=RetentionEventType.RETENTION_ERROR,
                    path=directory,
                    message=error_msg,
                    error=str(e),
                )
            )

            logger.exception("retention_failed", directory=str(directory), error=str(e))

        result.end_time = datetime.now(timezone.utc)
        result.duration_seconds = time.perf_counter() - start_time

        self._notify_observers(
            RetentionEvent(
                event_type=RetentionEventType.RETENTION_COMPLETED,
                path=directory,
                size_bytes=result.bytes_freed,
                message=f"Retention completed: {result.files_deleted} files deleted",
            )
        )

        logger.info(
            "retention_completed",
            directory=str(directory),
            result=result.to_dict(),
        )

        return result

    def _clean_empty_directories(self, directory: Path) -> int:
        """
        Remove empty directories.

        Args:
            directory: Root directory to clean.

        Returns:
            Number of directories removed.
        """
        removed = 0

        try:
            # Walk bottom-up to remove nested empty directories
            for dirpath in sorted(directory.rglob("*"), reverse=True):
                if not dirpath.is_dir():
                    continue

                try:
                    # Check if directory is empty
                    if not any(dirpath.iterdir()):
                        dirpath.rmdir()
                        removed += 1

                        self._notify_observers(
                            RetentionEvent(
                                event_type=RetentionEventType.DIRECTORY_CLEANED,
                                path=dirpath,
                                message=f"Removed empty directory: {dirpath.name}",
                            )
                        )

                        logger.debug("empty_directory_removed", path=str(dirpath))

                except OSError:
                    pass

        except OSError as e:
            logger.warning(
                "directory_cleanup_failed",
                directory=str(directory),
                error=str(e),
            )

        return removed

    def preview(self, directory: Path) -> list[dict[str, Any]]:
        """
        Preview what would be deleted without actually deleting.

        Args:
            directory: Directory to preview.

        Returns:
            List of file information dictionaries.
        """
        # Update size criteria with directory
        for criteria in self._criteria:
            if isinstance(criteria, SizeCriteria):
                criteria.directory = directory

        candidates = self._get_candidate_files(directory)
        current_size = sum(c[2] for c in candidates)
        preview_list: list[dict[str, Any]] = []
        bytes_to_free = 0

        for path, age_days, size_bytes in candidates:
            would_delete = False
            reasons: list[str] = []

            if self._config.max_age_days is not None and age_days > self._config.max_age_days:
                would_delete = True
                reasons.append(f"Age: {age_days:.1f} days > {self._config.max_age_days}")

            if self._config.max_size_bytes is not None:
                remaining = current_size - bytes_to_free
                if remaining > self._config.max_size_bytes:
                    would_delete = True
                    max_mb = self._config.max_size_bytes / (1024 * 1024)
                    reasons.append(f"Size limit: {max_mb:.1f} MB exceeded")

            if would_delete:
                bytes_to_free += size_bytes
                preview_list.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "size_bytes": size_bytes,
                        "size_mb": round(size_bytes / (1024 * 1024), 2),
                        "age_days": round(age_days, 2),
                        "reasons": reasons,
                    }
                )

        return preview_list

    def get_statistics(self, directory: Path) -> dict[str, Any]:
        """
        Get statistics about a directory.

        Args:
            directory: Directory to analyze.

        Returns:
            Dictionary of statistics.
        """
        candidates = self._get_candidate_files(directory)

        total_size = sum(c[2] for c in candidates)
        file_count = len(candidates)
        oldest_age = max((c[1] for c in candidates), default=0.0)
        newest_age = min((c[1] for c in candidates), default=0.0)

        return {
            "directory": str(directory),
            "file_count": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_file_age_days": round(oldest_age, 2),
            "newest_file_age_days": round(newest_age, 2),
            "retention_config": self._config.to_dict(),
        }
