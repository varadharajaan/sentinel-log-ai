"""
Data versioning and migration system.

This module provides version tracking and migration support for
data format changes, ensuring backward compatibility.

Design Patterns:
- Registry Pattern: Central migration registry
- Command Pattern: Migrations as executable commands
- Chain of Responsibility: Sequential migration application
- Strategy Pattern: Pluggable migration strategies
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


class SchemaVersion:
    """
    Semantic versioning for data schemas.

    Follows semver: MAJOR.MINOR.PATCH
    - MAJOR: Breaking changes
    - MINOR: Backward-compatible additions
    - PATCH: Backward-compatible fixes
    """

    def __init__(self, major: int, minor: int, patch: int = 0) -> None:
        """
        Initialize version.

        Args:
            major: Major version number.
            minor: Minor version number.
            patch: Patch version number.
        """
        self.major = major
        self.minor = minor
        self.patch = patch

    @classmethod
    def from_string(cls, version_str: str) -> SchemaVersion:
        """
        Parse version from string.

        Args:
            version_str: Version string (e.g., "1.2.3").

        Returns:
            SchemaVersion instance.
        """
        parts = version_str.split(".")
        if len(parts) < 2:
            msg = f"Invalid version format: {version_str}"
            raise ValueError(msg)

        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2]) if len(parts) > 2 else 0

        return cls(major, minor, patch)

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"SchemaVersion({self.major}, {self.minor}, {self.patch})"

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return self.major == other.major and self.minor == other.minor and self.patch == other.patch

    def __lt__(self, other: SchemaVersion) -> bool:
        """Less than comparison."""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch

    def __le__(self, other: SchemaVersion) -> bool:
        """Less than or equal comparison."""
        return self == other or self < other

    def __gt__(self, other: SchemaVersion) -> bool:
        """Greater than comparison."""
        return not self <= other

    def __ge__(self, other: SchemaVersion) -> bool:
        """Greater than or equal comparison."""
        return not self < other

    def __hash__(self) -> int:
        """Return hash for use in sets/dicts."""
        return hash((self.major, self.minor, self.patch))

    def is_compatible_with(self, other: SchemaVersion) -> bool:
        """
        Check if this version is compatible with another.

        Compatible means same major version.

        Args:
            other: Version to compare.

        Returns:
            True if compatible.
        """
        return self.major == other.major

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> SchemaVersion:
        """Create from dictionary."""
        return cls(
            major=data["major"],
            minor=data["minor"],
            patch=data.get("patch", 0),
        )


@dataclass
class DataVersion:
    """
    Version information for a data store.

    Tracks the current schema version and migration history.
    """

    schema_version: SchemaVersion
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    migration_history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_version": str(self.schema_version),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "migration_history": self.migration_history,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataVersion:
        """Create from dictionary."""
        return cls(
            schema_version=SchemaVersion.from_string(data["schema_version"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            migration_history=data.get("migration_history", []),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> DataVersion:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def record_migration(self, migration_id: str) -> None:
        """Record a migration in the history."""
        self.migration_history.append(migration_id)
        self.updated_at = datetime.now(timezone.utc)


class Migration(ABC):
    """
    Abstract base class for data migrations.

    Migrations transform data from one version to another.
    """

    @property
    @abstractmethod
    def migration_id(self) -> str:
        """Unique identifier for this migration."""
        pass

    @property
    @abstractmethod
    def from_version(self) -> SchemaVersion:
        """Source version this migration applies to."""
        pass

    @property
    @abstractmethod
    def to_version(self) -> SchemaVersion:
        """Target version after migration."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of the migration."""
        return ""

    @abstractmethod
    def up(self, data_path: Path) -> None:
        """
        Apply the migration.

        Args:
            data_path: Path to the data directory.
        """
        pass

    @abstractmethod
    def down(self, data_path: Path) -> None:
        """
        Revert the migration.

        Args:
            data_path: Path to the data directory.
        """
        pass

    def validate(self, data_path: Path) -> bool:
        """
        Validate that migration can be applied.

        Args:
            data_path: Path to the data directory.

        Returns:
            True if migration can be applied.
        """
        return data_path.exists()


class MigrationRegistry:
    """
    Registry for data migrations.

    Implements the Registry pattern for managing migrations.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._migrations: dict[str, Migration] = {}
        self._version_map: dict[SchemaVersion, list[Migration]] = {}

        logger.debug("migration_registry_initialized")

    def register(self, migration: Migration) -> None:
        """
        Register a migration.

        Args:
            migration: Migration to register.
        """
        if migration.migration_id in self._migrations:
            logger.warning(
                "migration_already_registered",
                migration_id=migration.migration_id,
            )
            return

        self._migrations[migration.migration_id] = migration

        # Index by from_version
        from_ver = migration.from_version
        if from_ver not in self._version_map:
            self._version_map[from_ver] = []
        self._version_map[from_ver].append(migration)

        logger.debug(
            "migration_registered",
            migration_id=migration.migration_id,
            from_version=str(from_ver),
            to_version=str(migration.to_version),
        )

    def get_migration(self, migration_id: str) -> Migration | None:
        """
        Get a migration by ID.

        Args:
            migration_id: Migration identifier.

        Returns:
            Migration if found, None otherwise.
        """
        return self._migrations.get(migration_id)

    def get_migrations_from(self, version: SchemaVersion) -> list[Migration]:
        """
        Get all migrations from a specific version.

        Args:
            version: Source version.

        Returns:
            List of applicable migrations.
        """
        return self._version_map.get(version, [])

    def get_migration_path(
        self,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> list[Migration]:
        """
        Find the migration path between two versions.

        Uses BFS to find shortest path.

        Args:
            from_version: Starting version.
            to_version: Target version.

        Returns:
            List of migrations to apply in order.
        """
        if from_version == to_version:
            return []

        # BFS to find path
        from collections import deque

        visited: set[SchemaVersion] = set()
        queue: deque[tuple[SchemaVersion, list[Migration]]] = deque()
        queue.append((from_version, []))

        while queue:
            current, path = queue.popleft()

            if current in visited:
                continue
            visited.add(current)

            if current == to_version:
                return path

            for migration in self.get_migrations_from(current):
                if migration.to_version not in visited:
                    queue.append((migration.to_version, [*path, migration]))

        # No path found
        logger.warning(
            "no_migration_path",
            from_version=str(from_version),
            to_version=str(to_version),
        )
        return []

    def list_migrations(self) -> list[Migration]:
        """
        List all registered migrations.

        Returns:
            List of all migrations.
        """
        return list(self._migrations.values())


class VersionManager:
    """
    Manages data versioning and migrations.

    Coordinates version tracking and migration execution.
    """

    VERSION_FILENAME = "data_version.json"
    CURRENT_VERSION = SchemaVersion(1, 0, 0)

    def __init__(
        self,
        data_directory: Path,
        registry: MigrationRegistry | None = None,
    ) -> None:
        """
        Initialize the version manager.

        Args:
            data_directory: Path to the data directory.
            registry: Migration registry to use.
        """
        self._data_directory = data_directory
        self._registry = registry or MigrationRegistry()
        self._version_file = data_directory / self.VERSION_FILENAME

        logger.info(
            "version_manager_initialized",
            data_directory=str(data_directory),
        )

    def get_current_version(self) -> DataVersion | None:
        """
        Get the current data version.

        Returns:
            Current version if exists, None for new data.
        """
        if not self._version_file.exists():
            return None

        try:
            with self._version_file.open() as f:
                return DataVersion.from_json(f.read())
        except Exception as e:
            logger.warning(
                "version_read_failed",
                path=str(self._version_file),
                error=str(e),
            )
            return None

    def set_version(self, version: DataVersion) -> None:
        """
        Set the data version.

        Args:
            version: Version to set.
        """
        self._data_directory.mkdir(parents=True, exist_ok=True)

        with self._version_file.open("w") as f:
            f.write(version.to_json())

        logger.info(
            "version_updated",
            version=str(version.schema_version),
        )

    def initialize(self, version: SchemaVersion | None = None) -> DataVersion:
        """
        Initialize versioning for new data.

        Args:
            version: Initial version (defaults to CURRENT_VERSION).

        Returns:
            Created DataVersion.
        """
        version = version or self.CURRENT_VERSION

        data_version = DataVersion(
            schema_version=version,
            metadata={"initialized_by": "version_manager"},
        )

        self.set_version(data_version)

        logger.info(
            "versioning_initialized",
            version=str(version),
        )

        return data_version

    def needs_migration(self, target_version: SchemaVersion | None = None) -> bool:
        """
        Check if data needs migration.

        Args:
            target_version: Target version (defaults to CURRENT_VERSION).

        Returns:
            True if migration is needed.
        """
        current = self.get_current_version()
        if current is None:
            return False

        target = target_version or self.CURRENT_VERSION
        return current.schema_version != target

    def migrate(
        self,
        target_version: SchemaVersion | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """
        Migrate data to target version.

        Args:
            target_version: Target version (defaults to CURRENT_VERSION).
            dry_run: If True, only plan without executing.

        Returns:
            List of applied migration IDs.
        """
        current = self.get_current_version()
        if current is None:
            logger.info("no_version_info", message="Cannot migrate unversioned data")
            return []

        target = target_version or self.CURRENT_VERSION

        if current.schema_version == target:
            logger.info(
                "already_at_target_version",
                version=str(target),
            )
            return []

        # Find migration path
        migrations = self._registry.get_migration_path(
            current.schema_version,
            target,
        )

        if not migrations:
            logger.warning(
                "no_migration_path_found",
                from_version=str(current.schema_version),
                to_version=str(target),
            )
            return []

        logger.info(
            "migration_planned",
            from_version=str(current.schema_version),
            to_version=str(target),
            migration_count=len(migrations),
            dry_run=dry_run,
        )

        applied: list[str] = []

        for migration in migrations:
            if dry_run:
                logger.info(
                    "migration_would_apply",
                    migration_id=migration.migration_id,
                    description=migration.description,
                )
                applied.append(migration.migration_id)
                continue

            try:
                start_time = time.perf_counter()

                # Validate
                if not migration.validate(self._data_directory):
                    msg = f"Migration validation failed: {migration.migration_id}"
                    logger.error("migration_validation_failed", migration_id=migration.migration_id)
                    raise ValueError(msg)

                # Apply
                migration.up(self._data_directory)

                duration = time.perf_counter() - start_time

                # Update version
                current.schema_version = migration.to_version
                current.record_migration(migration.migration_id)
                self.set_version(current)

                applied.append(migration.migration_id)

                logger.info(
                    "migration_applied",
                    migration_id=migration.migration_id,
                    to_version=str(migration.to_version),
                    duration_seconds=round(duration, 3),
                )

            except Exception as e:
                logger.exception(
                    "migration_failed",
                    migration_id=migration.migration_id,
                    error=str(e),
                )
                raise

        return applied

    def rollback(self, migration_id: str) -> bool:
        """
        Rollback a specific migration.

        Args:
            migration_id: ID of migration to rollback.

        Returns:
            True if rollback succeeded.
        """
        current = self.get_current_version()
        if current is None:
            logger.warning("cannot_rollback", message="No version info")
            return False

        if migration_id not in current.migration_history:
            logger.warning(
                "migration_not_in_history",
                migration_id=migration_id,
            )
            return False

        migration = self._registry.get_migration(migration_id)
        if migration is None:
            logger.error(
                "migration_not_found",
                migration_id=migration_id,
            )
            return False

        try:
            start_time = time.perf_counter()

            migration.down(self._data_directory)

            duration = time.perf_counter() - start_time

            # Update version
            current.schema_version = migration.from_version
            current.migration_history.remove(migration_id)
            self.set_version(current)

            logger.info(
                "migration_rolled_back",
                migration_id=migration_id,
                to_version=str(migration.from_version),
                duration_seconds=round(duration, 3),
            )

            return True

        except Exception as e:
            logger.exception(
                "rollback_failed",
                migration_id=migration_id,
                error=str(e),
            )
            return False

    def get_status(self) -> dict[str, Any]:
        """
        Get version and migration status.

        Returns:
            Status dictionary.
        """
        current = self.get_current_version()

        return {
            "data_directory": str(self._data_directory),
            "current_version": str(current.schema_version) if current else None,
            "target_version": str(self.CURRENT_VERSION),
            "needs_migration": self.needs_migration(),
            "migration_history": current.migration_history if current else [],
            "available_migrations": [
                {
                    "id": m.migration_id,
                    "from": str(m.from_version),
                    "to": str(m.to_version),
                    "description": m.description,
                }
                for m in self._registry.list_migrations()
            ],
        }

    def check_compatibility(self, version: SchemaVersion) -> dict[str, Any]:
        """
        Check compatibility with a version.

        Args:
            version: Version to check.

        Returns:
            Compatibility information.
        """
        current = self.get_current_version()
        if current is None:
            return {
                "compatible": True,
                "reason": "No existing version",
            }

        current_ver = current.schema_version
        is_compatible = current_ver.is_compatible_with(version)

        migration_path = self._registry.get_migration_path(current_ver, version)

        return {
            "compatible": is_compatible,
            "current_version": str(current_ver),
            "check_version": str(version),
            "migration_available": len(migration_path) > 0,
            "migration_steps": len(migration_path),
            "reason": (
                "Compatible (same major version)"
                if is_compatible
                else "Incompatible (different major versions)"
            ),
        }
