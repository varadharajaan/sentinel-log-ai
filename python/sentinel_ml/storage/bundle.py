"""
Import/export functionality for portable data bundles.

This module provides capabilities to export and import complete
data bundles including vector indices, metadata, and configuration.

Design Patterns:
- Builder Pattern: Construct bundles with multiple components
- Strategy Pattern: Pluggable compression and serialization
- Command Pattern: Export/import as reversible operations
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


class BundleFormat(str, Enum):
    """Supported bundle formats."""

    TAR_GZ = "tar.gz"
    ZIP = "zip"


@dataclass
class BundleMetadata:
    """Metadata for a data bundle."""

    bundle_id: str
    name: str
    version: str
    created_at: datetime
    created_by: str = ""
    description: str = ""
    format_version: str = "1.0"
    components: list[str] = field(default_factory=list)
    file_count: int = 0
    total_size_bytes: int = 0
    checksum: str = ""
    tags: list[str] = field(default_factory=list)
    source_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bundle_id": self.bundle_id,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "format_version": self.format_version,
            "components": self.components,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "checksum": self.checksum,
            "tags": self.tags,
            "source_config": self.source_config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleMetadata:
        """Create from dictionary."""
        return cls(
            bundle_id=data["bundle_id"],
            name=data["name"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by", ""),
            description=data.get("description", ""),
            format_version=data.get("format_version", "1.0"),
            components=data.get("components", []),
            file_count=data.get("file_count", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
            checksum=data.get("checksum", ""),
            tags=data.get("tags", []),
            source_config=data.get("source_config", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> BundleMetadata:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class BundleManifest:
    """Manifest listing all files in a bundle."""

    files: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_file(
        self,
        path: str,
        size_bytes: int,
        checksum: str,
        component: str,
    ) -> None:
        """Add a file to the manifest."""
        self.files.append(
            {
                "path": path,
                "size_bytes": size_bytes,
                "checksum": checksum,
                "component": component,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "files": self.files,
            "created_at": self.created_at.isoformat(),
            "file_count": len(self.files),
            "total_size_bytes": sum(f["size_bytes"] for f in self.files),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleManifest:
        """Create from dictionary."""
        manifest = cls(
            files=data.get("files", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
        )
        return manifest

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> BundleManifest:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ExportConfig:
    """Configuration for export operations."""

    output_path: Path | None = None
    output_directory: Path = field(default_factory=lambda: Path(".data/exports"))
    bundle_format: BundleFormat = BundleFormat.TAR_GZ
    compression_level: int = 6
    include_config: bool = True
    include_vectors: bool = True
    include_metadata: bool = True
    include_logs: bool = False
    exclude_patterns: list[str] = field(default_factory=lambda: ["*.tmp", "*.lock", "__pycache__"])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_path": str(self.output_path) if self.output_path else None,
            "output_directory": str(self.output_directory),
            "bundle_format": self.bundle_format.value,
            "compression_level": self.compression_level,
            "include_config": self.include_config,
            "include_vectors": self.include_vectors,
            "include_metadata": self.include_metadata,
            "include_logs": self.include_logs,
            "exclude_patterns": self.exclude_patterns,
        }


@dataclass
class ImportConfig:
    """Configuration for import operations."""

    target_directory: Path = field(default_factory=lambda: Path(".data"))
    overwrite: bool = False
    verify_checksums: bool = True
    restore_config: bool = True
    restore_vectors: bool = True
    restore_metadata: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_directory": str(self.target_directory),
            "overwrite": self.overwrite,
            "verify_checksums": self.verify_checksums,
            "restore_config": self.restore_config,
            "restore_vectors": self.restore_vectors,
            "restore_metadata": self.restore_metadata,
        }


@dataclass
class ImportResult:
    """Result of an import operation."""

    success: bool
    bundle_metadata: BundleMetadata | None = None
    files_imported: int = 0
    bytes_imported: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "bundle_id": self.bundle_metadata.bundle_id if self.bundle_metadata else None,
            "files_imported": self.files_imported,
            "bytes_imported": self.bytes_imported,
            "bytes_imported_mb": round(self.bytes_imported / (1024 * 1024), 2),
            "duration_seconds": round(self.duration_seconds, 3),
            "errors": self.errors,
            "warnings": self.warnings,
        }


class BundleExporter:
    """
    Exports data as portable bundles.

    Implements the Builder pattern for constructing bundles
    with multiple components.
    """

    METADATA_FILENAME = "bundle_metadata.json"
    MANIFEST_FILENAME = "bundle_manifest.json"

    def __init__(self, config: ExportConfig | None = None) -> None:
        """
        Initialize the exporter.

        Args:
            config: Export configuration.
        """
        self._config = config or ExportConfig()
        self._config.output_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            "bundle_exporter_initialized",
            config=self._config.to_dict(),
        )

    def _generate_bundle_id(self) -> str:
        """Generate a unique bundle ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_suffix = hashlib.md5(f"{timestamp}_{time.perf_counter()}".encode()).hexdigest()[:8]
        return f"bundle_{timestamp}_{unique_suffix}"

    def _calculate_file_checksum(self, path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        buffer_size = 65536

        try:
            with path.open("rb") as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    sha256.update(data)
            return sha256.hexdigest()
        except OSError:
            return ""

    def _should_include_file(self, path: Path, base_path: Path) -> bool:
        """Check if a file should be included in the bundle."""
        relative = path.relative_to(base_path)

        for pattern in self._config.exclude_patterns:
            if path.match(pattern) or relative.match(pattern):
                return False

        return True

    def export(
        self,
        source_directories: dict[str, Path],
        name: str,
        version: str = "1.0.0",
        description: str = "",
        tags: list[str] | None = None,
    ) -> Path:
        """
        Export data to a bundle.

        Args:
            source_directories: Dict mapping component names to directories.
            name: Bundle name.
            version: Bundle version.
            description: Bundle description.
            tags: Optional tags.

        Returns:
            Path to the created bundle file.
        """
        bundle_id = self._generate_bundle_id()
        tags = tags or []

        # Determine output path
        if self._config.output_path:
            bundle_path = self._config.output_path
        else:
            extension = self._config.bundle_format.value
            bundle_path = self._config.output_directory / f"{bundle_id}.{extension}"

        metadata = BundleMetadata(
            bundle_id=bundle_id,
            name=name,
            version=version,
            created_at=datetime.now(timezone.utc),
            description=description,
            tags=tags,
            components=list(source_directories.keys()),
        )

        manifest = BundleManifest()

        logger.info(
            "bundle_export_started",
            bundle_id=bundle_id,
            components=list(source_directories.keys()),
        )

        start_time = time.perf_counter()
        file_count = 0
        total_size = 0

        try:
            with tarfile.open(
                bundle_path, "w:gz", compresslevel=self._config.compression_level
            ) as tar:
                for component_name, source_dir in source_directories.items():
                    if not source_dir.exists():
                        logger.warning(
                            "component_directory_not_found",
                            component=component_name,
                            path=str(source_dir),
                        )
                        continue

                    for file_path in source_dir.rglob("*"):
                        if not file_path.is_file():
                            continue

                        if not self._should_include_file(file_path, source_dir):
                            continue

                        # Construct archive path with component prefix
                        relative_path = file_path.relative_to(source_dir)
                        arcname = f"{component_name}/{relative_path}"

                        tar.add(file_path, arcname=arcname)

                        file_size = file_path.stat().st_size
                        file_checksum = self._calculate_file_checksum(file_path)

                        manifest.add_file(
                            path=arcname,
                            size_bytes=file_size,
                            checksum=file_checksum,
                            component=component_name,
                        )

                        file_count += 1
                        total_size += file_size

                # Add metadata and manifest
                self._add_json_to_archive(tar, self.METADATA_FILENAME, metadata.to_json())
                self._add_json_to_archive(tar, self.MANIFEST_FILENAME, manifest.to_json())

            # Update metadata with final stats
            metadata.file_count = file_count
            metadata.total_size_bytes = bundle_path.stat().st_size
            metadata.checksum = self._calculate_file_checksum(bundle_path)

            duration = time.perf_counter() - start_time

            logger.info(
                "bundle_exported",
                bundle_id=bundle_id,
                path=str(bundle_path),
                file_count=file_count,
                size_bytes=metadata.total_size_bytes,
                duration_seconds=round(duration, 3),
            )

            return bundle_path

        except Exception as e:
            # Clean up failed bundle
            if bundle_path.exists():
                bundle_path.unlink()

            logger.exception(
                "bundle_export_failed",
                bundle_id=bundle_id,
                error=str(e),
            )
            raise

    def _add_json_to_archive(
        self,
        tar: tarfile.TarFile,
        filename: str,
        content: str,
    ) -> None:
        """Add a JSON string to the archive."""
        import io

        content_bytes = content.encode("utf-8")
        info = tarfile.TarInfo(name=filename)
        info.size = len(content_bytes)
        tar.addfile(info, io.BytesIO(content_bytes))


class BundleImporter:
    """
    Imports data from portable bundles.

    Implements verification and restoration of bundle components.
    """

    METADATA_FILENAME = "bundle_metadata.json"
    MANIFEST_FILENAME = "bundle_manifest.json"

    def __init__(self, config: ImportConfig | None = None) -> None:
        """
        Initialize the importer.

        Args:
            config: Import configuration.
        """
        self._config = config or ImportConfig()

        logger.info(
            "bundle_importer_initialized",
            config=self._config.to_dict(),
        )

    def _calculate_file_checksum(self, path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        buffer_size = 65536

        try:
            with path.open("rb") as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    sha256.update(data)
            return sha256.hexdigest()
        except OSError:
            return ""

    def inspect(self, bundle_path: Path) -> BundleMetadata | None:
        """
        Inspect a bundle without importing.

        Args:
            bundle_path: Path to the bundle file.

        Returns:
            Bundle metadata if valid, None otherwise.
        """
        try:
            with tarfile.open(bundle_path, "r:gz") as tar:
                try:
                    member = tar.getmember(self.METADATA_FILENAME)
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8")
                        return BundleMetadata.from_json(content)
                    return None
                except KeyError:
                    logger.warning(
                        "bundle_metadata_not_found",
                        path=str(bundle_path),
                    )
                    return None

        except Exception as e:
            logger.warning(
                "bundle_inspect_failed",
                path=str(bundle_path),
                error=str(e),
            )
            return None

    def import_bundle(self, bundle_path: Path) -> ImportResult:
        """
        Import a bundle to the target directory.

        Args:
            bundle_path: Path to the bundle file.

        Returns:
            Result of the import operation.
        """
        result = ImportResult(success=False)
        start_time = time.perf_counter()

        if not bundle_path.exists():
            result.errors.append(f"Bundle not found: {bundle_path}")
            return result

        logger.info(
            "bundle_import_started",
            path=str(bundle_path),
        )

        try:
            with tarfile.open(bundle_path, "r:gz") as tar:
                # Read metadata
                try:
                    metadata_member = tar.getmember(self.METADATA_FILENAME)
                    f = tar.extractfile(metadata_member)
                    if f:
                        metadata_content = f.read().decode("utf-8")
                        result.bundle_metadata = BundleMetadata.from_json(metadata_content)
                except KeyError:
                    result.warnings.append("Bundle metadata not found, using defaults")

                # Read manifest
                manifest: BundleManifest | None = None
                try:
                    manifest_member = tar.getmember(self.MANIFEST_FILENAME)
                    f = tar.extractfile(manifest_member)
                    if f:
                        manifest_content = f.read().decode("utf-8")
                        manifest = BundleManifest.from_json(manifest_content)
                except KeyError:
                    result.warnings.append("Bundle manifest not found, skipping verification")

                # Create target directory
                self._config.target_directory.mkdir(parents=True, exist_ok=True)

                # Extract files
                members = [
                    m
                    for m in tar.getmembers()
                    if m.name not in (self.METADATA_FILENAME, self.MANIFEST_FILENAME)
                ]

                for member in members:
                    target_path = self._config.target_directory / member.name

                    # Security check
                    if not str(target_path.resolve()).startswith(
                        str(self._config.target_directory.resolve())
                    ):
                        result.errors.append(f"Path traversal detected: {member.name}")
                        continue

                    # Check overwrite
                    if target_path.exists() and not self._config.overwrite:
                        result.warnings.append(f"Skipping existing file: {member.name}")
                        continue

                    # Extract
                    tar.extract(member, path=self._config.target_directory, filter="data")

                    result.files_imported += 1
                    result.bytes_imported += member.size

                    # Verify checksum if manifest available
                    if manifest and self._config.verify_checksums:
                        expected = next(
                            (f["checksum"] for f in manifest.files if f["path"] == member.name),
                            None,
                        )
                        if expected:
                            actual = self._calculate_file_checksum(target_path)
                            if actual != expected:
                                result.errors.append(f"Checksum mismatch for {member.name}")

            result.success = len(result.errors) == 0
            result.duration_seconds = time.perf_counter() - start_time

            logger.info(
                "bundle_imported",
                bundle_id=result.bundle_metadata.bundle_id if result.bundle_metadata else "unknown",
                files_imported=result.files_imported,
                bytes_imported=result.bytes_imported,
                duration_seconds=round(result.duration_seconds, 3),
                success=result.success,
            )

        except Exception as e:
            result.errors.append(f"Import failed: {e}")
            result.duration_seconds = time.perf_counter() - start_time

            logger.exception(
                "bundle_import_failed",
                path=str(bundle_path),
                error=str(e),
            )

        return result

    def list_bundles(self, directory: Path) -> list[BundleMetadata]:
        """
        List all bundles in a directory.

        Args:
            directory: Directory to search.

        Returns:
            List of bundle metadata.
        """
        bundles: list[BundleMetadata] = []

        for path in directory.glob("*.tar.gz"):
            metadata = self.inspect(path)
            if metadata:
                bundles.append(metadata)

        # Sort by creation time (newest first)
        bundles.sort(key=lambda m: m.created_at, reverse=True)

        return bundles
