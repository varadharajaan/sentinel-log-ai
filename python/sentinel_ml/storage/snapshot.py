"""
Snapshot management for periodic index backups.

This module provides snapshot capabilities for creating and managing
periodic backups of the vector store and associated metadata.

Design Patterns:
- Command Pattern: Snapshot operations as commands
- Factory Pattern: Create snapshots with metadata
- Repository Pattern: Manage snapshot storage
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from sentinel_ml.logging import get_logger

logger = get_logger(__name__)


class SnapshotStatus(str, Enum):
    """Status of a snapshot."""

    CREATING = "creating"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"
    DELETED = "deleted"


@dataclass
class SnapshotMetadata:
    """Metadata for a snapshot."""

    snapshot_id: str
    name: str
    status: SnapshotStatus
    created_at: datetime
    size_bytes: int = 0
    file_count: int = 0
    source_directory: str = ""
    checksum: str = ""
    compression: str = "gzip"
    format_version: str = "1.0"
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "snapshot_id": self.snapshot_id,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "source_directory": self.source_directory,
            "checksum": self.checksum,
            "compression": self.compression,
            "format_version": self.format_version,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotMetadata:
        """Create from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            name=data["name"],
            status=SnapshotStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            size_bytes=data.get("size_bytes", 0),
            file_count=data.get("file_count", 0),
            source_directory=data.get("source_directory", ""),
            checksum=data.get("checksum", ""),
            compression=data.get("compression", "gzip"),
            format_version=data.get("format_version", "1.0"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> SnapshotMetadata:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Snapshot:
    """
    Represents a snapshot of data.

    Contains the metadata and path to the snapshot archive.
    """

    metadata: SnapshotMetadata
    path: Path

    @property
    def snapshot_id(self) -> str:
        """Get the snapshot ID."""
        return self.metadata.snapshot_id

    @property
    def exists(self) -> bool:
        """Check if the snapshot file exists."""
        return self.path.exists()

    @property
    def is_valid(self) -> bool:
        """Check if the snapshot is valid."""
        return (
            self.exists
            and self.metadata.status == SnapshotStatus.COMPLETED
            and self.verify_checksum()
        )

    def verify_checksum(self) -> bool:
        """Verify the snapshot file checksum."""
        if not self.exists:
            return False

        if not self.metadata.checksum:
            return True  # No checksum to verify

        calculated = self._calculate_checksum()
        return calculated == self.metadata.checksum

    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of the snapshot file."""
        sha256 = hashlib.sha256()
        buffer_size = 65536

        try:
            with self.path.open("rb") as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    sha256.update(data)
            return sha256.hexdigest()
        except OSError:
            return ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "path": str(self.path),
            "exists": self.exists,
            "is_valid": self.is_valid if self.exists else False,
        }


@dataclass
class SnapshotConfig:
    """Configuration for snapshot operations."""

    snapshot_directory: Path = field(default_factory=lambda: Path(".data/snapshots"))
    compression_level: int = 6
    max_snapshots: int = 10
    auto_cleanup: bool = True
    include_patterns: list[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: list[str] = field(default_factory=lambda: ["*.tmp", "*.lock"])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_directory": str(self.snapshot_directory),
            "compression_level": self.compression_level,
            "max_snapshots": self.max_snapshots,
            "auto_cleanup": self.auto_cleanup,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
        }


class SnapshotManager:
    """
    Manages snapshot creation, restoration, and cleanup.

    Implements the Repository pattern for snapshot storage management.
    """

    METADATA_FILENAME = "snapshot_metadata.json"
    INDEX_FILENAME = "snapshot_index.json"

    def __init__(self, config: SnapshotConfig | None = None) -> None:
        """
        Initialize the snapshot manager.

        Args:
            config: Snapshot configuration.
        """
        self._config = config or SnapshotConfig()
        self._ensure_snapshot_directory()

        logger.info(
            "snapshot_manager_initialized",
            config=self._config.to_dict(),
        )

    def _ensure_snapshot_directory(self) -> None:
        """Ensure the snapshot directory exists."""
        self._config.snapshot_directory.mkdir(parents=True, exist_ok=True)

    def _generate_snapshot_id(self) -> str:
        """Generate a unique snapshot ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_suffix = hashlib.md5(f"{timestamp}_{time.perf_counter()}".encode()).hexdigest()[:8]
        return f"snapshot_{timestamp}_{unique_suffix}"

    def _should_include_file(self, path: Path, base_path: Path) -> bool:
        """Check if a file should be included in the snapshot."""
        relative = path.relative_to(base_path)

        # Check exclusions
        for pattern in self._config.exclude_patterns:
            if path.match(pattern) or relative.match(pattern):
                return False

        # Check inclusions
        for pattern in self._config.include_patterns:
            if path.match(pattern) or relative.match(pattern):
                return True

        return False

    def create(
        self,
        source_directory: Path,
        name: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
    ) -> Snapshot:
        """
        Create a new snapshot of a directory.

        Args:
            source_directory: Directory to snapshot.
            name: Optional name for the snapshot.
            description: Optional description.
            tags: Optional tags for categorization.

        Returns:
            The created Snapshot object.

        Raises:
            OSError: If snapshot creation fails.
        """
        snapshot_id = self._generate_snapshot_id()
        name = name or snapshot_id
        tags = tags or []

        snapshot_path = self._config.snapshot_directory / f"{snapshot_id}.tar.gz"

        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            name=name,
            status=SnapshotStatus.CREATING,
            created_at=datetime.now(timezone.utc),
            source_directory=str(source_directory),
            description=description,
            tags=tags,
        )

        logger.info(
            "snapshot_creation_started",
            snapshot_id=snapshot_id,
            source=str(source_directory),
        )

        start_time = time.perf_counter()
        file_count = 0
        total_size = 0

        try:
            with tarfile.open(
                snapshot_path, "w:gz", compresslevel=self._config.compression_level
            ) as tar:
                for file_path in source_directory.rglob("*"):
                    if not file_path.is_file():
                        continue

                    if not self._should_include_file(file_path, source_directory):
                        continue

                    arcname = file_path.relative_to(source_directory)
                    tar.add(file_path, arcname=str(arcname))
                    file_count += 1
                    total_size += file_path.stat().st_size

                # Add metadata file
                metadata_content = metadata.to_json().encode("utf-8")
                import io

                metadata_info = tarfile.TarInfo(name=self.METADATA_FILENAME)
                metadata_info.size = len(metadata_content)
                tar.addfile(metadata_info, io.BytesIO(metadata_content))

            # Update metadata with final stats
            metadata.file_count = file_count
            metadata.size_bytes = snapshot_path.stat().st_size

            # Calculate checksum
            snapshot = Snapshot(metadata=metadata, path=snapshot_path)
            metadata.checksum = snapshot._calculate_checksum()
            metadata.status = SnapshotStatus.COMPLETED

            duration = time.perf_counter() - start_time

            logger.info(
                "snapshot_created",
                snapshot_id=snapshot_id,
                path=str(snapshot_path),
                file_count=file_count,
                size_bytes=metadata.size_bytes,
                duration_seconds=round(duration, 3),
            )

            # Auto cleanup old snapshots
            if self._config.auto_cleanup:
                self._cleanup_old_snapshots()

            # Update index
            self._update_index(snapshot)

            return snapshot

        except Exception as e:
            metadata.status = SnapshotStatus.FAILED

            # Clean up failed snapshot
            if snapshot_path.exists():
                snapshot_path.unlink()

            logger.exception(
                "snapshot_creation_failed",
                snapshot_id=snapshot_id,
                error=str(e),
            )
            raise

    def restore(self, snapshot: Snapshot, target_directory: Path) -> None:
        """
        Restore a snapshot to a target directory.

        Args:
            snapshot: Snapshot to restore.
            target_directory: Directory to restore to.

        Raises:
            ValueError: If snapshot is invalid.
            OSError: If restoration fails.
        """
        if not snapshot.exists:
            msg = f"Snapshot file not found: {snapshot.path}"
            logger.error("snapshot_restore_failed", error=msg)
            raise ValueError(msg)

        if not snapshot.verify_checksum():
            msg = f"Snapshot checksum verification failed: {snapshot.snapshot_id}"
            logger.error("snapshot_restore_failed", error=msg)
            raise ValueError(msg)

        logger.info(
            "snapshot_restore_started",
            snapshot_id=snapshot.snapshot_id,
            target=str(target_directory),
        )

        start_time = time.perf_counter()

        try:
            target_directory.mkdir(parents=True, exist_ok=True)

            with tarfile.open(snapshot.path, "r:gz") as tar:
                # Filter out metadata file
                members = [m for m in tar.getmembers() if m.name != self.METADATA_FILENAME]

                # Security check: prevent path traversal
                for member in members:
                    member_path = target_directory / member.name
                    if not str(member_path.resolve()).startswith(str(target_directory.resolve())):
                        msg = f"Path traversal detected: {member.name}"
                        raise ValueError(msg)

                tar.extractall(path=target_directory, members=members, filter="data")

            duration = time.perf_counter() - start_time

            logger.info(
                "snapshot_restored",
                snapshot_id=snapshot.snapshot_id,
                target=str(target_directory),
                file_count=snapshot.metadata.file_count,
                duration_seconds=round(duration, 3),
            )

        except Exception as e:
            logger.exception(
                "snapshot_restore_failed",
                snapshot_id=snapshot.snapshot_id,
                error=str(e),
            )
            raise

    def list_snapshots(self) -> list[Snapshot]:
        """
        List all available snapshots.

        Returns:
            List of Snapshot objects sorted by creation time (newest first).
        """
        snapshots: list[Snapshot] = []

        for path in self._config.snapshot_directory.glob("*.tar.gz"):
            try:
                metadata = self._read_snapshot_metadata(path)
                if metadata:
                    snapshots.append(Snapshot(metadata=metadata, path=path))
            except Exception as e:
                logger.warning(
                    "snapshot_metadata_read_failed",
                    path=str(path),
                    error=str(e),
                )

        # Sort by creation time (newest first)
        snapshots.sort(key=lambda s: s.metadata.created_at, reverse=True)

        return snapshots

    def _read_snapshot_metadata(self, path: Path) -> SnapshotMetadata | None:
        """Read metadata from a snapshot archive."""
        try:
            with tarfile.open(path, "r:gz") as tar:
                try:
                    member = tar.getmember(self.METADATA_FILENAME)
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8")
                        return SnapshotMetadata.from_json(content)
                    return None
                except KeyError:
                    # No metadata file, create basic metadata
                    stat = path.stat()
                    return SnapshotMetadata(
                        snapshot_id=path.stem,
                        name=path.stem,
                        status=SnapshotStatus.COMPLETED,
                        created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                        size_bytes=stat.st_size,
                    )
        except Exception as e:
            logger.warning(
                "snapshot_metadata_read_error",
                path=str(path),
                error=str(e),
            )
            return None

    def get_snapshot(self, snapshot_id: str) -> Snapshot | None:
        """
        Get a snapshot by ID.

        Args:
            snapshot_id: The snapshot ID to find.

        Returns:
            The Snapshot if found, None otherwise.
        """
        for snapshot in self.list_snapshots():
            if snapshot.snapshot_id == snapshot_id:
                return snapshot
        return None

    def delete(self, snapshot: Snapshot) -> bool:
        """
        Delete a snapshot.

        Args:
            snapshot: Snapshot to delete.

        Returns:
            True if deleted successfully.
        """
        if not snapshot.exists:
            return False

        try:
            snapshot.path.unlink()
            snapshot.metadata.status = SnapshotStatus.DELETED

            logger.info(
                "snapshot_deleted",
                snapshot_id=snapshot.snapshot_id,
                path=str(snapshot.path),
            )

            self._remove_from_index(snapshot.snapshot_id)
            return True

        except OSError as e:
            logger.error(
                "snapshot_delete_failed",
                snapshot_id=snapshot.snapshot_id,
                error=str(e),
            )
            return False

    def _cleanup_old_snapshots(self) -> int:
        """
        Remove old snapshots exceeding the maximum count.

        Returns:
            Number of snapshots deleted.
        """
        snapshots = self.list_snapshots()

        if len(snapshots) <= self._config.max_snapshots:
            return 0

        # Delete oldest snapshots
        to_delete = snapshots[self._config.max_snapshots :]
        deleted = 0

        for snapshot in to_delete:
            if self.delete(snapshot):
                deleted += 1

        if deleted > 0:
            logger.info(
                "old_snapshots_cleaned",
                deleted_count=deleted,
                remaining_count=len(snapshots) - deleted,
            )

        return deleted

    def _update_index(self, snapshot: Snapshot) -> None:
        """Update the snapshot index file."""
        index_path = self._config.snapshot_directory / self.INDEX_FILENAME
        index: dict[str, Any] = {}

        if index_path.exists():
            try:
                with index_path.open() as f:
                    index = json.load(f)
            except Exception:
                index = {}

        if "snapshots" not in index:
            index["snapshots"] = {}

        index["snapshots"][snapshot.snapshot_id] = snapshot.metadata.to_dict()
        index["last_updated"] = datetime.now(timezone.utc).isoformat()

        with index_path.open("w") as f:
            json.dump(index, f, indent=2)

    def _remove_from_index(self, snapshot_id: str) -> None:
        """Remove a snapshot from the index."""
        index_path = self._config.snapshot_directory / self.INDEX_FILENAME

        if not index_path.exists():
            return

        try:
            with index_path.open() as f:
                index = json.load(f)

            if "snapshots" in index and snapshot_id in index["snapshots"]:
                del index["snapshots"][snapshot_id]
                index["last_updated"] = datetime.now(timezone.utc).isoformat()

                with index_path.open("w") as f:
                    json.dump(index, f, indent=2)

        except Exception as e:
            logger.warning(
                "index_update_failed",
                snapshot_id=snapshot_id,
                error=str(e),
            )

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about snapshots.

        Returns:
            Dictionary of statistics.
        """
        snapshots = self.list_snapshots()

        total_size = sum(s.metadata.size_bytes for s in snapshots)
        valid_count = sum(1 for s in snapshots if s.is_valid)

        oldest = min(
            (s.metadata.created_at for s in snapshots),
            default=None,
        )
        newest = max(
            (s.metadata.created_at for s in snapshots),
            default=None,
        )

        return {
            "snapshot_count": len(snapshots),
            "valid_count": valid_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_snapshot": oldest.isoformat() if oldest else None,
            "newest_snapshot": newest.isoformat() if newest else None,
            "snapshot_directory": str(self._config.snapshot_directory),
            "max_snapshots": self._config.max_snapshots,
        }
