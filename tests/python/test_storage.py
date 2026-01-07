"""
Unit tests for the storage module.

Tests cover:
- Retention policy engine (age-based, size-based, composite)
- Snapshot management (create, restore, list, delete)
- Import/export bundle operations
- Data versioning and migrations
"""

from __future__ import annotations

import tarfile
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from sentinel_ml.storage import (
    AgeCriteria,
    BundleExporter,
    BundleImporter,
    BundleManifest,
    BundleMetadata,
    CompositeCriteria,
    DataVersion,
    ExportConfig,
    ImportConfig,
    ImportResult,
    Migration,
    MigrationRegistry,
    RetentionConfig,
    RetentionEvent,
    RetentionEventType,
    RetentionObserver,
    RetentionPolicy,
    RetentionResult,
    SchemaVersion,
    SizeCriteria,
    Snapshot,
    SnapshotConfig,
    SnapshotManager,
    SnapshotMetadata,
    SnapshotStatus,
    VersionManager,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestRetentionEventType:
    """Tests for RetentionEventType enum."""

    def test_file_deleted_value(self) -> None:
        """Test FILE_DELETED enum value."""
        assert RetentionEventType.FILE_DELETED.value == "file_deleted"

    def test_directory_cleaned_value(self) -> None:
        """Test DIRECTORY_CLEANED enum value."""
        assert RetentionEventType.DIRECTORY_CLEANED.value == "directory_cleaned"

    def test_retention_started_value(self) -> None:
        """Test RETENTION_STARTED enum value."""
        assert RetentionEventType.RETENTION_STARTED.value == "retention_started"

    def test_retention_completed_value(self) -> None:
        """Test RETENTION_COMPLETED enum value."""
        assert RetentionEventType.RETENTION_COMPLETED.value == "retention_completed"

    def test_retention_error_value(self) -> None:
        """Test RETENTION_ERROR enum value."""
        assert RetentionEventType.RETENTION_ERROR.value == "retention_error"


class TestRetentionEvent:
    """Tests for RetentionEvent dataclass."""

    def test_create_basic_event(self) -> None:
        """Test creating a basic retention event."""
        event = RetentionEvent(event_type=RetentionEventType.FILE_DELETED)
        assert event.event_type == RetentionEventType.FILE_DELETED
        assert event.path is None
        assert event.size_bytes == 0
        assert event.age_days == 0.0
        assert event.message == ""
        assert event.error is None

    def test_create_full_event(self, tmp_path: Path) -> None:
        """Test creating event with all fields."""
        event = RetentionEvent(
            event_type=RetentionEventType.FILE_DELETED,
            path=tmp_path / "test.log",
            size_bytes=1024,
            age_days=7.5,
            message="Test deletion",
            error=None,
        )
        assert event.path == tmp_path / "test.log"
        assert event.size_bytes == 1024
        assert event.age_days == 7.5
        assert event.message == "Test deletion"

    def test_event_to_dict(self, tmp_path: Path) -> None:
        """Test converting event to dictionary."""
        path = tmp_path / "test.log"
        event = RetentionEvent(
            event_type=RetentionEventType.FILE_DELETED,
            path=path,
            size_bytes=2048,
            age_days=14.0,
            message="Deleted old file",
        )
        result = event.to_dict()

        assert result["event_type"] == "file_deleted"
        assert result["path"] == str(path)
        assert result["size_bytes"] == 2048
        assert result["age_days"] == 14.0
        assert result["message"] == "Deleted old file"
        assert "timestamp" in result

    def test_event_with_error(self) -> None:
        """Test event with error field."""
        event = RetentionEvent(
            event_type=RetentionEventType.RETENTION_ERROR,
            message="Operation failed",
            error="Permission denied",
        )
        assert event.error == "Permission denied"
        result = event.to_dict()
        assert result["error"] == "Permission denied"


class TestRetentionResult:
    """Tests for RetentionResult dataclass."""

    def test_create_default_result(self) -> None:
        """Test creating default result."""
        result = RetentionResult()
        assert result.files_deleted == 0
        assert result.directories_cleaned == 0
        assert result.bytes_freed == 0
        assert result.errors == []
        assert result.duration_seconds == 0.0
        assert result.success is True

    def test_result_with_errors(self) -> None:
        """Test result with errors."""
        result = RetentionResult(
            files_deleted=5,
            bytes_freed=10240,
            errors=["Failed to delete file1", "Failed to delete file2"],
        )
        assert result.success is False
        assert len(result.errors) == 2

    def test_result_to_dict(self) -> None:
        """Test converting result to dictionary."""
        result = RetentionResult(
            files_deleted=10,
            directories_cleaned=2,
            bytes_freed=5 * 1024 * 1024,
            duration_seconds=1.5,
        )
        data = result.to_dict()

        assert data["files_deleted"] == 10
        assert data["directories_cleaned"] == 2
        assert data["bytes_freed"] == 5 * 1024 * 1024
        assert data["bytes_freed_mb"] == 5.0
        assert data["duration_seconds"] == 1.5
        assert data["success"] is True


class TestAgeCriteria:
    """Tests for AgeCriteria class."""

    def test_create_age_criteria(self) -> None:
        """Test creating age criteria."""
        criteria = AgeCriteria(max_age_days=30)
        assert criteria.max_age_days == 30

    def test_should_delete_old_file(self, tmp_path: Path) -> None:
        """Test that old files should be deleted."""
        # Create a file
        test_file = tmp_path / "old_file.log"
        test_file.write_text("test content")

        # Set reference time to 31 days in future
        ref_time = datetime.now(timezone.utc) + timedelta(days=31)
        criteria = AgeCriteria(max_age_days=30, reference_time=ref_time)

        assert criteria.should_delete(test_file) is True

    def test_should_not_delete_recent_file(self, tmp_path: Path) -> None:
        """Test that recent files should not be deleted."""
        test_file = tmp_path / "recent_file.log"
        test_file.write_text("test content")

        criteria = AgeCriteria(max_age_days=30)
        assert criteria.should_delete(test_file) is False

    def test_should_not_delete_nonexistent_file(self, tmp_path: Path) -> None:
        """Test handling of non-existent files."""
        criteria = AgeCriteria(max_age_days=30)
        assert criteria.should_delete(tmp_path / "nonexistent.log") is False

    def test_get_age_days(self, tmp_path: Path) -> None:
        """Test getting file age in days."""
        test_file = tmp_path / "test.log"
        test_file.write_text("content")

        criteria = AgeCriteria(max_age_days=30)
        age = criteria.get_age_days(test_file)

        assert age >= 0
        assert age < 1  # Just created

    def test_get_description(self) -> None:
        """Test getting criteria description."""
        criteria = AgeCriteria(max_age_days=14)
        desc = criteria.get_description()
        assert "14" in desc
        assert "days" in desc


class TestSizeCriteria:
    """Tests for SizeCriteria class."""

    def test_create_size_criteria(self) -> None:
        """Test creating size criteria."""
        criteria = SizeCriteria(max_size_bytes=100 * 1024 * 1024)
        assert criteria.max_size_bytes == 100 * 1024 * 1024

    def test_should_delete_when_over_limit(self, tmp_path: Path) -> None:
        """Test deletion when directory exceeds size limit."""
        # Create files exceeding limit
        for i in range(10):
            (tmp_path / f"file_{i}.log").write_bytes(b"x" * 1024)

        criteria = SizeCriteria(
            max_size_bytes=5 * 1024,  # 5KB limit
            directory=tmp_path,
        )

        test_file = tmp_path / "file_0.log"
        assert criteria.should_delete(test_file) is True

    def test_should_not_delete_under_limit(self, tmp_path: Path) -> None:
        """Test no deletion when under size limit."""
        test_file = tmp_path / "small.log"
        test_file.write_bytes(b"x" * 100)

        criteria = SizeCriteria(
            max_size_bytes=1024 * 1024,  # 1MB limit
            directory=tmp_path,
        )

        assert criteria.should_delete(test_file) is False

    def test_get_current_size(self, tmp_path: Path) -> None:
        """Test getting current directory size."""
        # Create 5 files of 1KB each
        for i in range(5):
            (tmp_path / f"file_{i}.log").write_bytes(b"x" * 1024)

        criteria = SizeCriteria(max_size_bytes=10 * 1024, directory=tmp_path)
        size = criteria.get_current_size()

        assert size == 5 * 1024

    def test_get_description(self) -> None:
        """Test getting criteria description."""
        criteria = SizeCriteria(max_size_bytes=100 * 1024 * 1024)
        desc = criteria.get_description()
        assert "100" in desc or "MB" in desc


class TestCompositeCriteria:
    """Tests for CompositeCriteria class."""

    def test_create_composite_criteria(self) -> None:
        """Test creating composite criteria."""
        criteria = CompositeCriteria()
        assert criteria.criteria == []
        assert criteria.require_all is True

    def test_add_criteria(self) -> None:
        """Test adding criteria to composite."""
        composite = CompositeCriteria()
        age = AgeCriteria(max_age_days=30)
        composite.add_criteria(age)

        assert len(composite.criteria) == 1

    def test_and_logic(self, tmp_path: Path) -> None:
        """Test AND logic (require_all=True)."""
        test_file = tmp_path / "test.log"
        test_file.write_text("content")

        # One criteria passes, one fails
        ref_time = datetime.now(timezone.utc) + timedelta(days=10)
        age1 = AgeCriteria(max_age_days=5, reference_time=ref_time)  # Should delete
        age2 = AgeCriteria(max_age_days=20, reference_time=ref_time)  # Should not delete

        composite = CompositeCriteria(criteria=[age1, age2], require_all=True)
        assert composite.should_delete(test_file) is False  # Both must pass

    def test_or_logic(self, tmp_path: Path) -> None:
        """Test OR logic (require_all=False)."""
        test_file = tmp_path / "test.log"
        test_file.write_text("content")

        ref_time = datetime.now(timezone.utc) + timedelta(days=10)
        age1 = AgeCriteria(max_age_days=5, reference_time=ref_time)  # Should delete
        age2 = AgeCriteria(max_age_days=20, reference_time=ref_time)  # Should not delete

        composite = CompositeCriteria(criteria=[age1, age2], require_all=False)
        assert composite.should_delete(test_file) is True  # Any can pass

    def test_empty_criteria_returns_false(self, tmp_path: Path) -> None:
        """Test that empty criteria returns False."""
        test_file = tmp_path / "test.log"
        test_file.write_text("content")

        composite = CompositeCriteria()
        assert composite.should_delete(test_file) is False

    def test_get_description(self) -> None:
        """Test getting composite description."""
        age1 = AgeCriteria(max_age_days=30)
        age2 = AgeCriteria(max_age_days=60)
        composite = CompositeCriteria(criteria=[age1, age2], require_all=True)

        desc = composite.get_description()
        assert "AND" in desc


class TestRetentionConfig:
    """Tests for RetentionConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = RetentionConfig()
        assert config.max_age_days is None
        assert config.max_size_bytes is None
        assert config.dry_run is False
        assert config.recursive is True

    def test_config_with_age(self) -> None:
        """Test configuration with age limit."""
        config = RetentionConfig(max_age_days=30)
        assert config.max_age_days == 30

    def test_config_with_size_mb(self) -> None:
        """Test configuration with size in MB."""
        config = RetentionConfig(max_size_mb=100)
        assert config.max_size_bytes == 100 * 1024 * 1024

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = RetentionConfig(
            max_age_days=14,
            max_size_mb=50,
            dry_run=True,
        )
        data = config.to_dict()

        assert data["max_age_days"] == 14
        assert data["max_size_mb"] == 50
        assert data["dry_run"] is True


class TestRetentionPolicy:
    """Tests for RetentionPolicy class."""

    def test_create_policy(self) -> None:
        """Test creating a retention policy."""
        config = RetentionConfig(max_age_days=30)
        policy = RetentionPolicy(config)
        assert policy is not None

    def test_apply_deletes_old_files(self, tmp_path: Path) -> None:
        """Test that apply deletes old files."""
        import os

        # Create test files
        for i in range(5):
            f = tmp_path / f"file_{i}.log"
            f.write_text(f"content {i}")
            # Set mtime to 2 days ago
            old_time = time.time() - (2 * 86400)
            os.utime(f, (old_time, old_time))

        # Use age criteria of 1 day
        config = RetentionConfig(max_age_days=1)
        policy = RetentionPolicy(config)

        result = policy.apply(tmp_path)

        # Files should be deleted (they are 2 days old, limit is 1 day)
        assert result.files_deleted == 5
        assert result.success is True

    def test_apply_dry_run(self, tmp_path: Path) -> None:
        """Test dry run does not delete files."""
        import os

        test_file = tmp_path / "test.log"
        test_file.write_text("content")
        # Set mtime to 2 days ago
        old_time = time.time() - (2 * 86400)
        os.utime(test_file, (old_time, old_time))

        config = RetentionConfig(max_age_days=1, dry_run=True)
        policy = RetentionPolicy(config)

        result = policy.apply(tmp_path)

        # File should still exist
        assert test_file.exists()
        assert result.files_deleted == 1  # Counted but not actually deleted

    def test_preview(self, tmp_path: Path) -> None:
        """Test preview functionality."""
        import os

        # Create files with old timestamps
        for i in range(3):
            f = tmp_path / f"file_{i}.log"
            f.write_text(f"content {i}")
            old_time = time.time() - (2 * 86400)
            os.utime(f, (old_time, old_time))

        config = RetentionConfig(max_age_days=1)
        policy = RetentionPolicy(config)

        preview = policy.preview(tmp_path)

        assert len(preview) == 3
        assert "path" in preview[0]
        assert "size_bytes" in preview[0]
        assert "age_days" in preview[0]

    def test_get_statistics(self, tmp_path: Path) -> None:
        """Test getting directory statistics."""
        # Create files
        for i in range(5):
            (tmp_path / f"file_{i}.log").write_bytes(b"x" * 1000)

        config = RetentionConfig()
        policy = RetentionPolicy(config)
        stats = policy.get_statistics(tmp_path)

        assert stats["file_count"] == 5
        assert stats["total_size_bytes"] == 5000
        assert "directory" in stats

    def test_observer_notification(self, tmp_path: Path) -> None:
        """Test observer receives notifications."""
        import os

        # Create observer mock
        observer = MagicMock(spec=RetentionObserver)

        test_file = tmp_path / "test.log"
        test_file.write_text("content")
        # Set mtime to 2 days ago
        old_time = time.time() - (2 * 86400)
        os.utime(test_file, (old_time, old_time))

        config = RetentionConfig(max_age_days=1)
        policy = RetentionPolicy(config, observers=[observer])

        policy.apply(tmp_path)

        # Observer should have been called
        assert observer.on_retention_event.called

    def test_add_remove_observer(self) -> None:
        """Test adding and removing observers."""
        policy = RetentionPolicy()
        observer = MagicMock(spec=RetentionObserver)

        policy.add_observer(observer)
        assert observer in policy._observers

        policy.remove_observer(observer)
        assert observer not in policy._observers


class TestSnapshotStatus:
    """Tests for SnapshotStatus enum."""

    def test_status_values(self) -> None:
        """Test status enum values."""
        assert SnapshotStatus.CREATING.value == "creating"
        assert SnapshotStatus.COMPLETED.value == "completed"
        assert SnapshotStatus.FAILED.value == "failed"
        assert SnapshotStatus.CORRUPTED.value == "corrupted"
        assert SnapshotStatus.DELETED.value == "deleted"


class TestSnapshotMetadata:
    """Tests for SnapshotMetadata dataclass."""

    def test_create_metadata(self) -> None:
        """Test creating snapshot metadata."""
        metadata = SnapshotMetadata(
            snapshot_id="snap_001",
            name="Test Snapshot",
            status=SnapshotStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
        )
        assert metadata.snapshot_id == "snap_001"
        assert metadata.name == "Test Snapshot"
        assert metadata.status == SnapshotStatus.COMPLETED

    def test_to_dict(self) -> None:
        """Test converting metadata to dictionary."""
        now = datetime.now(timezone.utc)
        metadata = SnapshotMetadata(
            snapshot_id="snap_002",
            name="Backup",
            status=SnapshotStatus.COMPLETED,
            created_at=now,
            size_bytes=1024,
            file_count=10,
        )
        data = metadata.to_dict()

        assert data["snapshot_id"] == "snap_002"
        assert data["name"] == "Backup"
        assert data["status"] == "completed"
        assert data["size_bytes"] == 1024
        assert data["file_count"] == 10

    def test_from_dict(self) -> None:
        """Test creating metadata from dictionary."""
        data = {
            "snapshot_id": "snap_003",
            "name": "Restored",
            "status": "completed",
            "created_at": "2024-01-01T00:00:00+00:00",
            "size_bytes": 2048,
        }
        metadata = SnapshotMetadata.from_dict(data)

        assert metadata.snapshot_id == "snap_003"
        assert metadata.status == SnapshotStatus.COMPLETED

    def test_json_serialization(self) -> None:
        """Test JSON serialization round-trip."""
        original = SnapshotMetadata(
            snapshot_id="snap_004",
            name="JSON Test",
            status=SnapshotStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
        )
        json_str = original.to_json()
        restored = SnapshotMetadata.from_json(json_str)

        assert restored.snapshot_id == original.snapshot_id
        assert restored.name == original.name


class TestSnapshot:
    """Tests for Snapshot class."""

    def test_create_snapshot(self, tmp_path: Path) -> None:
        """Test creating a snapshot object."""
        metadata = SnapshotMetadata(
            snapshot_id="snap_001",
            name="Test",
            status=SnapshotStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
        )
        snapshot = Snapshot(metadata=metadata, path=tmp_path / "snap.tar.gz")

        assert snapshot.snapshot_id == "snap_001"
        assert snapshot.exists is False  # File not created

    def test_exists_property(self, tmp_path: Path) -> None:
        """Test exists property."""
        path = tmp_path / "snap.tar.gz"
        metadata = SnapshotMetadata(
            snapshot_id="snap_002",
            name="Exists Test",
            status=SnapshotStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
        )

        snapshot = Snapshot(metadata=metadata, path=path)
        assert snapshot.exists is False

        path.write_bytes(b"test")
        assert snapshot.exists is True

    def test_to_dict(self, tmp_path: Path) -> None:
        """Test converting snapshot to dictionary."""
        metadata = SnapshotMetadata(
            snapshot_id="snap_003",
            name="Dict Test",
            status=SnapshotStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
        )
        snapshot = Snapshot(metadata=metadata, path=tmp_path / "snap.tar.gz")
        data = snapshot.to_dict()

        assert "metadata" in data
        assert "path" in data
        assert "exists" in data


class TestSnapshotConfig:
    """Tests for SnapshotConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default snapshot configuration."""
        config = SnapshotConfig()
        assert config.compression_level == 6
        assert config.max_snapshots == 10
        assert config.auto_cleanup is True

    def test_custom_config(self, tmp_path: Path) -> None:
        """Test custom configuration."""
        config = SnapshotConfig(
            snapshot_directory=tmp_path,
            compression_level=9,
            max_snapshots=5,
        )
        assert config.snapshot_directory == tmp_path
        assert config.compression_level == 9
        assert config.max_snapshots == 5

    def test_to_dict(self, tmp_path: Path) -> None:
        """Test converting config to dictionary."""
        config = SnapshotConfig(snapshot_directory=tmp_path)
        data = config.to_dict()

        assert "snapshot_directory" in data
        assert "compression_level" in data
        assert "max_snapshots" in data


class TestSnapshotManager:
    """Tests for SnapshotManager class."""

    def test_create_manager(self, tmp_path: Path) -> None:
        """Test creating a snapshot manager."""
        config = SnapshotConfig(snapshot_directory=tmp_path / "snapshots")
        SnapshotManager(config)

        assert (tmp_path / "snapshots").exists()

    def test_create_snapshot(self, tmp_path: Path) -> None:
        """Test creating a snapshot."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("content 1")
        (source_dir / "file2.txt").write_text("content 2")

        config = SnapshotConfig(snapshot_directory=tmp_path / "snapshots")
        manager = SnapshotManager(config)

        snapshot = manager.create(source_dir, name="test_snapshot")

        assert snapshot.exists
        assert snapshot.metadata.status == SnapshotStatus.COMPLETED
        assert snapshot.metadata.file_count == 2

    def test_restore_snapshot(self, tmp_path: Path) -> None:
        """Test restoring a snapshot."""
        # Create source with files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "data.txt").write_text("important data")

        # Create snapshot
        config = SnapshotConfig(snapshot_directory=tmp_path / "snapshots")
        manager = SnapshotManager(config)
        snapshot = manager.create(source_dir, name="backup")

        # Restore to new location
        restore_dir = tmp_path / "restored"
        manager.restore(snapshot, restore_dir)

        assert (restore_dir / "data.txt").exists()
        assert (restore_dir / "data.txt").read_text() == "important data"

    def test_list_snapshots(self, tmp_path: Path) -> None:
        """Test listing snapshots."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("content")

        config = SnapshotConfig(snapshot_directory=tmp_path / "snapshots")
        manager = SnapshotManager(config)

        # Create multiple snapshots
        manager.create(source_dir, name="snap1")
        manager.create(source_dir, name="snap2")

        snapshots = manager.list_snapshots()
        assert len(snapshots) == 2

    def test_get_snapshot(self, tmp_path: Path) -> None:
        """Test getting snapshot by ID."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("content")

        config = SnapshotConfig(snapshot_directory=tmp_path / "snapshots")
        manager = SnapshotManager(config)

        created = manager.create(source_dir, name="findable")

        found = manager.get_snapshot(created.snapshot_id)
        assert found is not None
        assert found.snapshot_id == created.snapshot_id

    def test_delete_snapshot(self, tmp_path: Path) -> None:
        """Test deleting a snapshot."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("content")

        config = SnapshotConfig(snapshot_directory=tmp_path / "snapshots")
        manager = SnapshotManager(config)

        snapshot = manager.create(source_dir, name="deletable")
        assert snapshot.exists

        result = manager.delete(snapshot)
        assert result is True
        assert not snapshot.path.exists()

    def test_auto_cleanup(self, tmp_path: Path) -> None:
        """Test automatic cleanup of old snapshots."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("content")

        config = SnapshotConfig(
            snapshot_directory=tmp_path / "snapshots",
            max_snapshots=2,
            auto_cleanup=True,
        )
        manager = SnapshotManager(config)

        # Create 3 snapshots (exceeds max)
        manager.create(source_dir, name="snap1")
        manager.create(source_dir, name="snap2")
        manager.create(source_dir, name="snap3")

        snapshots = manager.list_snapshots()
        assert len(snapshots) == 2  # Oldest should be cleaned

    def test_get_statistics(self, tmp_path: Path) -> None:
        """Test getting snapshot statistics."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("content")

        config = SnapshotConfig(snapshot_directory=tmp_path / "snapshots")
        manager = SnapshotManager(config)
        manager.create(source_dir, name="stats_test")

        stats = manager.get_statistics()

        assert "snapshot_count" in stats
        assert "total_size_bytes" in stats
        assert stats["snapshot_count"] == 1


class TestBundleMetadata:
    """Tests for BundleMetadata dataclass."""

    def test_create_metadata(self) -> None:
        """Test creating bundle metadata."""
        metadata = BundleMetadata(
            bundle_id="bundle_001",
            name="Test Bundle",
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
        )
        assert metadata.bundle_id == "bundle_001"
        assert metadata.name == "Test Bundle"
        assert metadata.version == "1.0.0"

    def test_to_dict(self) -> None:
        """Test converting metadata to dictionary."""
        metadata = BundleMetadata(
            bundle_id="bundle_002",
            name="Dict Test",
            version="2.0.0",
            created_at=datetime.now(timezone.utc),
            components=["vectors", "metadata"],
        )
        data = metadata.to_dict()

        assert data["bundle_id"] == "bundle_002"
        assert data["version"] == "2.0.0"
        assert "vectors" in data["components"]

    def test_json_roundtrip(self) -> None:
        """Test JSON serialization round-trip."""
        original = BundleMetadata(
            bundle_id="bundle_003",
            name="JSON Test",
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
            description="Test description",
        )
        json_str = original.to_json()
        restored = BundleMetadata.from_json(json_str)

        assert restored.bundle_id == original.bundle_id
        assert restored.description == original.description


class TestBundleManifest:
    """Tests for BundleManifest dataclass."""

    def test_create_manifest(self) -> None:
        """Test creating a manifest."""
        manifest = BundleManifest()
        assert manifest.files == []

    def test_add_file(self) -> None:
        """Test adding files to manifest."""
        manifest = BundleManifest()
        manifest.add_file(
            path="vectors/index.faiss",
            size_bytes=1024,
            checksum="abc123",
            component="vectors",
        )

        assert len(manifest.files) == 1
        assert manifest.files[0]["path"] == "vectors/index.faiss"

    def test_to_dict(self) -> None:
        """Test converting manifest to dictionary."""
        manifest = BundleManifest()
        manifest.add_file("file1.txt", 100, "hash1", "data")
        manifest.add_file("file2.txt", 200, "hash2", "data")

        data = manifest.to_dict()

        assert data["file_count"] == 2
        assert data["total_size_bytes"] == 300


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default export configuration."""
        config = ExportConfig()
        assert config.compression_level == 6
        assert config.include_config is True
        assert config.include_vectors is True

    def test_to_dict(self, tmp_path: Path) -> None:
        """Test converting config to dictionary."""
        config = ExportConfig(
            output_directory=tmp_path,
            compression_level=9,
        )
        data = config.to_dict()

        assert "output_directory" in data
        assert data["compression_level"] == 9


class TestImportConfig:
    """Tests for ImportConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default import configuration."""
        config = ImportConfig()
        assert config.overwrite is False
        assert config.verify_checksums is True

    def test_to_dict(self, tmp_path: Path) -> None:
        """Test converting config to dictionary."""
        config = ImportConfig(
            target_directory=tmp_path,
            overwrite=True,
        )
        data = config.to_dict()

        assert data["overwrite"] is True


class TestImportResult:
    """Tests for ImportResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful import result."""
        result = ImportResult(
            success=True,
            files_imported=10,
            bytes_imported=5000,
        )
        assert result.success is True
        assert result.files_imported == 10

    def test_failed_result(self) -> None:
        """Test failed import result."""
        result = ImportResult(
            success=False,
            errors=["File not found", "Checksum mismatch"],
        )
        assert result.success is False
        assert len(result.errors) == 2

    def test_to_dict(self) -> None:
        """Test converting result to dictionary."""
        result = ImportResult(
            success=True,
            files_imported=5,
            bytes_imported=1024 * 1024,
            duration_seconds=2.5,
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["files_imported"] == 5
        assert data["bytes_imported_mb"] == 1.0


class TestBundleExporter:
    """Tests for BundleExporter class."""

    def test_create_exporter(self, tmp_path: Path) -> None:
        """Test creating an exporter."""
        config = ExportConfig(output_directory=tmp_path / "exports")
        BundleExporter(config)

        assert (tmp_path / "exports").exists()

    def test_export_single_directory(self, tmp_path: Path) -> None:
        """Test exporting a single directory."""
        # Create source data
        source = tmp_path / "data"
        source.mkdir()
        (source / "file1.txt").write_text("content 1")
        (source / "file2.txt").write_text("content 2")

        config = ExportConfig(output_directory=tmp_path / "exports")
        exporter = BundleExporter(config)

        bundle_path = exporter.export(
            source_directories={"data": source},
            name="test_bundle",
            version="1.0.0",
        )

        assert bundle_path.exists()
        assert bundle_path.suffix == ".gz"

    def test_export_multiple_directories(self, tmp_path: Path) -> None:
        """Test exporting multiple directories."""
        # Create source directories
        vectors_dir = tmp_path / "vectors"
        vectors_dir.mkdir()
        (vectors_dir / "index.bin").write_bytes(b"vector data")

        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        (meta_dir / "meta.json").write_text('{"key": "value"}')

        config = ExportConfig(output_directory=tmp_path / "exports")
        exporter = BundleExporter(config)

        bundle_path = exporter.export(
            source_directories={
                "vectors": vectors_dir,
                "metadata": meta_dir,
            },
            name="multi_component",
            version="2.0.0",
        )

        # Verify bundle contains both directories
        with tarfile.open(bundle_path, "r:gz") as tar:
            names = tar.getnames()
            assert any("vectors" in n for n in names)
            assert any("metadata" in n for n in names)


class TestBundleImporter:
    """Tests for BundleImporter class."""

    def test_create_importer(self) -> None:
        """Test creating an importer."""
        config = ImportConfig()
        importer = BundleImporter(config)
        assert importer is not None

    def test_inspect_bundle(self, tmp_path: Path) -> None:
        """Test inspecting a bundle."""
        # Create a bundle first
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("content")

        exporter = BundleExporter(ExportConfig(output_directory=tmp_path / "exports"))
        bundle_path = exporter.export(
            source_directories={"data": source},
            name="inspect_test",
            version="1.0.0",
        )

        # Inspect it
        importer = BundleImporter()
        metadata = importer.inspect(bundle_path)

        assert metadata is not None
        assert metadata.name == "inspect_test"
        assert metadata.version == "1.0.0"

    def test_import_bundle(self, tmp_path: Path) -> None:
        """Test importing a bundle."""
        # Create and export
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.txt").write_text("important data")

        exporter = BundleExporter(ExportConfig(output_directory=tmp_path / "exports"))
        bundle_path = exporter.export(
            source_directories={"data": source},
            name="import_test",
            version="1.0.0",
        )

        # Import to new location
        target = tmp_path / "imported"
        config = ImportConfig(target_directory=target)
        importer = BundleImporter(config)

        result = importer.import_bundle(bundle_path)

        assert result.success is True
        assert result.files_imported == 1
        assert (target / "data" / "data.txt").exists()

    def test_import_nonexistent_bundle(self, tmp_path: Path) -> None:
        """Test importing non-existent bundle."""
        config = ImportConfig(target_directory=tmp_path / "target")
        importer = BundleImporter(config)

        result = importer.import_bundle(tmp_path / "nonexistent.tar.gz")

        assert result.success is False
        assert len(result.errors) > 0

    def test_list_bundles(self, tmp_path: Path) -> None:
        """Test listing bundles in directory."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("content")

        export_dir = tmp_path / "exports"
        exporter = BundleExporter(ExportConfig(output_directory=export_dir))

        # Create multiple bundles
        exporter.export({"data": source}, name="bundle1", version="1.0.0")
        exporter.export({"data": source}, name="bundle2", version="1.0.0")

        importer = BundleImporter()
        bundles = importer.list_bundles(export_dir)

        assert len(bundles) == 2


class TestSchemaVersion:
    """Tests for SchemaVersion class."""

    def test_create_version(self) -> None:
        """Test creating a version."""
        version = SchemaVersion(1, 2, 3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_from_string(self) -> None:
        """Test parsing version from string."""
        version = SchemaVersion.from_string("2.1.0")
        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 0

    def test_from_string_without_patch(self) -> None:
        """Test parsing version without patch."""
        version = SchemaVersion.from_string("3.0")
        assert version.major == 3
        assert version.minor == 0
        assert version.patch == 0

    def test_from_string_invalid(self) -> None:
        """Test parsing invalid version string."""
        with pytest.raises(ValueError):
            SchemaVersion.from_string("invalid")

    def test_str_representation(self) -> None:
        """Test string representation."""
        version = SchemaVersion(1, 2, 3)
        assert str(version) == "1.2.3"

    def test_equality(self) -> None:
        """Test version equality."""
        v1 = SchemaVersion(1, 0, 0)
        v2 = SchemaVersion(1, 0, 0)
        v3 = SchemaVersion(1, 0, 1)

        assert v1 == v2
        assert v1 != v3

    def test_comparison(self) -> None:
        """Test version comparison."""
        v1 = SchemaVersion(1, 0, 0)
        v2 = SchemaVersion(1, 1, 0)
        v3 = SchemaVersion(2, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert v3 > v1

    def test_less_than_or_equal(self) -> None:
        """Test less than or equal comparison."""
        v1 = SchemaVersion(1, 0, 0)
        v2 = SchemaVersion(1, 0, 0)
        v3 = SchemaVersion(1, 1, 0)

        assert v1 <= v2
        assert v1 <= v3

    def test_greater_than_or_equal(self) -> None:
        """Test greater than or equal comparison."""
        v1 = SchemaVersion(2, 0, 0)
        v2 = SchemaVersion(2, 0, 0)
        v3 = SchemaVersion(1, 0, 0)

        assert v1 >= v2
        assert v1 >= v3

    def test_hash(self) -> None:
        """Test version hashing."""
        v1 = SchemaVersion(1, 0, 0)
        v2 = SchemaVersion(1, 0, 0)

        assert hash(v1) == hash(v2)

        versions = {v1, v2}
        assert len(versions) == 1

    def test_compatibility(self) -> None:
        """Test version compatibility check."""
        v1 = SchemaVersion(1, 0, 0)
        v2 = SchemaVersion(1, 5, 0)
        v3 = SchemaVersion(2, 0, 0)

        assert v1.is_compatible_with(v2)
        assert not v1.is_compatible_with(v3)

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        version = SchemaVersion(1, 2, 3)
        data = version.to_dict()

        assert data["major"] == 1
        assert data["minor"] == 2
        assert data["patch"] == 3

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {"major": 2, "minor": 1, "patch": 0}
        version = SchemaVersion.from_dict(data)

        assert version.major == 2
        assert version.minor == 1


class TestDataVersion:
    """Tests for DataVersion dataclass."""

    def test_create_version(self) -> None:
        """Test creating data version."""
        version = DataVersion(schema_version=SchemaVersion(1, 0, 0))
        assert version.schema_version == SchemaVersion(1, 0, 0)
        assert version.migration_history == []

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        version = DataVersion(
            schema_version=SchemaVersion(1, 0, 0),
            migration_history=["m001", "m002"],
        )
        data = version.to_dict()

        assert data["schema_version"] == "1.0.0"
        assert "m001" in data["migration_history"]

    def test_json_roundtrip(self) -> None:
        """Test JSON serialization round-trip."""
        original = DataVersion(
            schema_version=SchemaVersion(1, 0, 0),
            metadata={"key": "value"},
        )
        json_str = original.to_json()
        restored = DataVersion.from_json(json_str)

        assert restored.schema_version == original.schema_version
        assert restored.metadata == original.metadata

    def test_record_migration(self) -> None:
        """Test recording migration in history."""
        version = DataVersion(schema_version=SchemaVersion(1, 0, 0))
        version.record_migration("migration_001")

        assert "migration_001" in version.migration_history


class TestMigration:
    """Tests for Migration abstract class."""

    def test_concrete_migration(self, tmp_path: Path) -> None:
        """Test concrete migration implementation."""

        class TestMigration(Migration):
            @property
            def migration_id(self) -> str:
                return "test_001"

            @property
            def from_version(self) -> SchemaVersion:
                return SchemaVersion(1, 0, 0)

            @property
            def to_version(self) -> SchemaVersion:
                return SchemaVersion(1, 1, 0)

            @property
            def description(self) -> str:
                return "Test migration"

            def up(self, data_path: Path) -> None:
                (data_path / "migrated.txt").write_text("migrated")

            def down(self, data_path: Path) -> None:
                (data_path / "migrated.txt").unlink()

        migration = TestMigration()
        assert migration.migration_id == "test_001"
        assert migration.from_version == SchemaVersion(1, 0, 0)
        assert migration.to_version == SchemaVersion(1, 1, 0)

        # Test up migration
        tmp_path.mkdir(exist_ok=True)
        migration.up(tmp_path)
        assert (tmp_path / "migrated.txt").exists()

        # Test down migration
        migration.down(tmp_path)
        assert not (tmp_path / "migrated.txt").exists()


class TestMigrationRegistry:
    """Tests for MigrationRegistry class."""

    def _create_test_migration(
        self,
        migration_id: str,
        from_ver: SchemaVersion,
        to_ver: SchemaVersion,
    ) -> Migration:
        """Create a test migration."""

        class TestMigration(Migration):
            @property
            def migration_id(self) -> str:
                return migration_id

            @property
            def from_version(self) -> SchemaVersion:
                return from_ver

            @property
            def to_version(self) -> SchemaVersion:
                return to_ver

            def up(self, data_path: Path) -> None:
                pass

            def down(self, data_path: Path) -> None:
                pass

        return TestMigration()

    def test_create_registry(self) -> None:
        """Test creating a registry."""
        registry = MigrationRegistry()
        assert registry.list_migrations() == []

    def test_register_migration(self) -> None:
        """Test registering a migration."""
        registry = MigrationRegistry()
        migration = self._create_test_migration(
            "m001",
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 1, 0),
        )
        registry.register(migration)

        assert len(registry.list_migrations()) == 1

    def test_get_migration(self) -> None:
        """Test getting migration by ID."""
        registry = MigrationRegistry()
        migration = self._create_test_migration(
            "m002",
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 1, 0),
        )
        registry.register(migration)

        found = registry.get_migration("m002")
        assert found is not None
        assert found.migration_id == "m002"

    def test_get_migrations_from(self) -> None:
        """Test getting migrations from a version."""
        registry = MigrationRegistry()

        m1 = self._create_test_migration(
            "m001",
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 1, 0),
        )
        m2 = self._create_test_migration(
            "m002",
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 2, 0),
        )
        registry.register(m1)
        registry.register(m2)

        migrations = registry.get_migrations_from(SchemaVersion(1, 0, 0))
        assert len(migrations) == 2

    def test_get_migration_path(self) -> None:
        """Test finding migration path."""
        registry = MigrationRegistry()

        m1 = self._create_test_migration(
            "m001",
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 1, 0),
        )
        m2 = self._create_test_migration(
            "m002",
            SchemaVersion(1, 1, 0),
            SchemaVersion(1, 2, 0),
        )
        registry.register(m1)
        registry.register(m2)

        path = registry.get_migration_path(
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 2, 0),
        )

        assert len(path) == 2
        assert path[0].migration_id == "m001"
        assert path[1].migration_id == "m002"

    def test_no_path_available(self) -> None:
        """Test when no migration path exists."""
        registry = MigrationRegistry()

        path = registry.get_migration_path(
            SchemaVersion(1, 0, 0),
            SchemaVersion(3, 0, 0),
        )

        assert len(path) == 0


class TestVersionManager:
    """Tests for VersionManager class."""

    def test_create_manager(self, tmp_path: Path) -> None:
        """Test creating a version manager."""
        manager = VersionManager(data_directory=tmp_path)
        assert manager is not None

    def test_get_current_version_none(self, tmp_path: Path) -> None:
        """Test getting version when none exists."""
        manager = VersionManager(data_directory=tmp_path)
        version = manager.get_current_version()
        assert version is None

    def test_initialize(self, tmp_path: Path) -> None:
        """Test initializing versioning."""
        manager = VersionManager(data_directory=tmp_path)
        version = manager.initialize()

        assert version is not None
        assert version.schema_version == SchemaVersion(1, 0, 0)

    def test_initialize_custom_version(self, tmp_path: Path) -> None:
        """Test initializing with custom version."""
        manager = VersionManager(data_directory=tmp_path)
        version = manager.initialize(SchemaVersion(2, 0, 0))

        assert version.schema_version == SchemaVersion(2, 0, 0)

    def test_set_and_get_version(self, tmp_path: Path) -> None:
        """Test setting and getting version."""
        manager = VersionManager(data_directory=tmp_path)

        data_version = DataVersion(
            schema_version=SchemaVersion(1, 5, 0),
            metadata={"test": True},
        )
        manager.set_version(data_version)

        retrieved = manager.get_current_version()
        assert retrieved is not None
        assert retrieved.schema_version == SchemaVersion(1, 5, 0)
        assert retrieved.metadata == {"test": True}

    def test_needs_migration(self, tmp_path: Path) -> None:
        """Test checking if migration is needed."""
        manager = VersionManager(data_directory=tmp_path)
        manager.initialize(SchemaVersion(0, 9, 0))

        assert manager.needs_migration() is True

    def test_no_migration_needed(self, tmp_path: Path) -> None:
        """Test when no migration is needed."""
        manager = VersionManager(data_directory=tmp_path)
        manager.initialize(SchemaVersion(1, 0, 0))

        assert manager.needs_migration() is False

    def test_get_status(self, tmp_path: Path) -> None:
        """Test getting version status."""
        manager = VersionManager(data_directory=tmp_path)
        manager.initialize()

        status = manager.get_status()

        assert "current_version" in status
        assert "target_version" in status
        assert "needs_migration" in status

    def test_check_compatibility(self, tmp_path: Path) -> None:
        """Test checking version compatibility."""
        manager = VersionManager(data_directory=tmp_path)
        manager.initialize(SchemaVersion(1, 0, 0))

        compatible = manager.check_compatibility(SchemaVersion(1, 5, 0))
        assert compatible["compatible"] is True

        incompatible = manager.check_compatibility(SchemaVersion(2, 0, 0))
        assert incompatible["compatible"] is False
