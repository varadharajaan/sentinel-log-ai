"""
Storage and retention management for Sentinel Log AI.

This module provides data lifecycle management including:
- Retention policies for automatic data cleanup
- Snapshotting for periodic backups
- Import/export for portable data bundles
- Data versioning for format migrations

Design Patterns:
- Strategy Pattern: Pluggable retention policies
- Command Pattern: Snapshot and export operations
- Template Method: Common lifecycle operations
- Observer Pattern: Retention event notifications

SOLID Principles:
- Single Responsibility: Each class has one purpose
- Open/Closed: Extensible via policy plugins
- Liskov Substitution: All policies implement same interface
- Interface Segregation: Separate read/write interfaces
- Dependency Inversion: Depends on abstractions
"""

from sentinel_ml.storage.bundle import (
    BundleExporter,
    BundleImporter,
    BundleManifest,
    BundleMetadata,
    ExportConfig,
    ImportConfig,
    ImportResult,
)
from sentinel_ml.storage.retention import (
    AgeCriteria,
    CompositeCriteria,
    RetentionConfig,
    RetentionCriteria,
    RetentionEvent,
    RetentionEventType,
    RetentionObserver,
    RetentionPolicy,
    RetentionResult,
    SizeCriteria,
)
from sentinel_ml.storage.snapshot import (
    Snapshot,
    SnapshotConfig,
    SnapshotManager,
    SnapshotMetadata,
    SnapshotStatus,
)
from sentinel_ml.storage.versioning import (
    DataVersion,
    Migration,
    MigrationRegistry,
    SchemaVersion,
    VersionManager,
)

__all__ = [
    # Retention
    "AgeCriteria",
    # Bundle
    "BundleExporter",
    "BundleImporter",
    "BundleManifest",
    "BundleMetadata",
    "CompositeCriteria",
    # Versioning
    "DataVersion",
    "ExportConfig",
    "ImportConfig",
    "ImportResult",
    "Migration",
    "MigrationRegistry",
    "RetentionConfig",
    "RetentionCriteria",
    "RetentionEvent",
    "RetentionEventType",
    "RetentionObserver",
    "RetentionPolicy",
    "RetentionResult",
    "SchemaVersion",
    "SizeCriteria",
    # Snapshot
    "Snapshot",
    "SnapshotConfig",
    "SnapshotManager",
    "SnapshotMetadata",
    "SnapshotStatus",
    "VersionManager",
]
