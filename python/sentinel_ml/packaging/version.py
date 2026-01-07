"""
Semantic Version Management for Sentinel Log AI.

This module provides comprehensive version management following
Semantic Versioning 2.0.0 specification (https://semver.org/).

Design Patterns:
- Value Object: SemanticVersion is immutable
- Strategy Pattern: Different bump strategies for version increments
- Factory Method: Version parsing and creation
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class VersionBumpType(Enum):
    """
    Types of version bumps following semantic versioning.

    Attributes:
        MAJOR: Breaking changes - increments major version, resets minor and patch.
        MINOR: New features, backward compatible - increments minor, resets patch.
        PATCH: Bug fixes, backward compatible - increments patch only.
        PRERELEASE: Pre-release version (alpha, beta, rc).
        BUILD: Build metadata only, no version change.
    """

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRERELEASE = "prerelease"
    BUILD = "build"


@dataclass(frozen=True, order=True)
class SemanticVersion:
    """
    Immutable semantic version representation.

    Follows Semantic Versioning 2.0.0 specification.
    Format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

    Attributes:
        major: Major version number (breaking changes).
        minor: Minor version number (new features).
        patch: Patch version number (bug fixes).
        prerelease: Pre-release identifier (e.g., "alpha.1", "beta.2", "rc.1").
        build: Build metadata (e.g., git commit hash).
    """

    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    build: str | None = None

    def __post_init__(self) -> None:
        """Validate version components."""
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValueError("Version components must be non-negative integers")

        if self.prerelease is not None and not self._validate_prerelease(self.prerelease):
            raise ValueError(f"Invalid prerelease identifier: {self.prerelease}")

        if self.build is not None and not self._validate_build(self.build):
            raise ValueError(f"Invalid build metadata: {self.build}")

    @staticmethod
    def _validate_prerelease(prerelease: str) -> bool:
        """Validate prerelease identifier format."""
        pattern = r"^[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*$"
        return bool(re.match(pattern, prerelease))

    @staticmethod
    def _validate_build(build: str) -> bool:
        """Validate build metadata format."""
        pattern = r"^[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*$"
        return bool(re.match(pattern, build))

    def __str__(self) -> str:
        """Return string representation of the version."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "prerelease": self.prerelease,
            "build": self.build,
            "string": str(self),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticVersion:
        """Create from dictionary representation."""
        return cls(
            major=data["major"],
            minor=data["minor"],
            patch=data["patch"],
            prerelease=data.get("prerelease"),
            build=data.get("build"),
        )

    def bump(
        self,
        bump_type: VersionBumpType,
        prerelease_id: str | None = None,
        build_metadata: str | None = None,
    ) -> SemanticVersion:
        """
        Create a new version with the specified bump applied.

        Args:
            bump_type: Type of version bump to apply.
            prerelease_id: Pre-release identifier for PRERELEASE bumps.
            build_metadata: Build metadata to include.

        Returns:
            New SemanticVersion with bump applied.

        Raises:
            ValueError: If prerelease bump requested without identifier.
        """
        if bump_type == VersionBumpType.MAJOR:
            return SemanticVersion(
                major=self.major + 1,
                minor=0,
                patch=0,
                prerelease=None,
                build=build_metadata,
            )
        elif bump_type == VersionBumpType.MINOR:
            return SemanticVersion(
                major=self.major,
                minor=self.minor + 1,
                patch=0,
                prerelease=None,
                build=build_metadata,
            )
        elif bump_type == VersionBumpType.PATCH:
            return SemanticVersion(
                major=self.major,
                minor=self.minor,
                patch=self.patch + 1,
                prerelease=None,
                build=build_metadata,
            )
        elif bump_type == VersionBumpType.PRERELEASE:
            if prerelease_id is None:
                raise ValueError("Prerelease identifier required for PRERELEASE bump")
            return SemanticVersion(
                major=self.major,
                minor=self.minor,
                patch=self.patch,
                prerelease=prerelease_id,
                build=build_metadata,
            )
        elif bump_type == VersionBumpType.BUILD:
            return SemanticVersion(
                major=self.major,
                minor=self.minor,
                patch=self.patch,
                prerelease=self.prerelease,
                build=build_metadata,
            )
        else:
            raise ValueError(f"Unknown bump type: {bump_type}")

    def is_prerelease(self) -> bool:
        """Check if this is a pre-release version."""
        return self.prerelease is not None

    def is_stable(self) -> bool:
        """Check if this is a stable release (not pre-release, major > 0)."""
        return self.prerelease is None and self.major > 0

    def is_compatible_with(self, other: SemanticVersion) -> bool:
        """
        Check if this version is API-compatible with another.

        Two versions are compatible if they have the same major version
        and neither is a pre-release.
        """
        if self.major == 0 or other.major == 0:
            return self.major == other.major and self.minor == other.minor
        return self.major == other.major


def parse_version(version_string: str) -> SemanticVersion:
    """
    Parse a version string into a SemanticVersion.

    Args:
        version_string: Version string in format MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD].

    Returns:
        Parsed SemanticVersion.

    Raises:
        ValueError: If version string is invalid.
    """
    pattern = r"""
        ^
        (?P<major>0|[1-9]\d*)
        \.
        (?P<minor>0|[1-9]\d*)
        \.
        (?P<patch>0|[1-9]\d*)
        (?:-(?P<prerelease>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?
        (?:\+(?P<build>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?
        $
    """

    match = re.match(pattern, version_string.strip(), re.VERBOSE)
    if not match:
        raise ValueError(f"Invalid version string: {version_string}")

    logger.debug("version_parsed", version=version_string)

    return SemanticVersion(
        major=int(match.group("major")),
        minor=int(match.group("minor")),
        patch=int(match.group("patch")),
        prerelease=match.group("prerelease"),
        build=match.group("build"),
    )


def get_current_version() -> SemanticVersion:
    """
    Get the current version from pyproject.toml.

    Returns:
        Current SemanticVersion.

    Raises:
        FileNotFoundError: If pyproject.toml not found.
        ValueError: If version not found or invalid.
    """
    pyproject_path = _find_pyproject_toml()
    if pyproject_path is None:
        raise FileNotFoundError("pyproject.toml not found")

    content = pyproject_path.read_text(encoding="utf-8")

    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not version_match:
        raise ValueError("Version not found in pyproject.toml")

    version = parse_version(version_match.group(1))
    logger.info("current_version_retrieved", version=str(version))
    return version


def _find_pyproject_toml() -> Path | None:
    """Find pyproject.toml by walking up directory tree."""
    current = Path.cwd()
    while current != current.parent:
        pyproject = current / "pyproject.toml"
        if pyproject.exists():
            return pyproject
        current = current.parent
    return None


class VersionManager:
    """
    Manages version updates across project files.

    This class handles version synchronization across multiple files
    including pyproject.toml, __init__.py, and other configuration files.

    Attributes:
        project_root: Root directory of the project.
        version_files: List of files containing version information.
    """

    VERSION_PATTERNS: dict[str, str] = {
        "pyproject.toml": r'version\s*=\s*"([^"]+)"',
        "__init__.py": r'__version__\s*=\s*"([^"]+)"',
        "package.json": r'"version"\s*:\s*"([^"]+)"',
        "Cargo.toml": r'version\s*=\s*"([^"]+)"',
    }

    def __init__(self, project_root: Path | None = None) -> None:
        """
        Initialize the version manager.

        Args:
            project_root: Root directory of the project. Defaults to current directory.
        """
        self.project_root = project_root or Path.cwd()
        self._version_files: list[Path] = []
        self._current_version: SemanticVersion | None = None
        self._discover_version_files()

        logger.info(
            "version_manager_initialized",
            project_root=str(self.project_root),
            version_files=[str(f) for f in self._version_files],
        )

    def _discover_version_files(self) -> None:
        """Discover files containing version information."""
        self._version_files = []

        for filename in self.VERSION_PATTERNS:
            if filename == "pyproject.toml":
                path = self.project_root / filename
                if path.exists():
                    self._version_files.append(path)
            elif filename == "__init__.py":
                for init_file in self.project_root.rglob("__init__.py"):
                    if "__version__" in init_file.read_text(encoding="utf-8"):
                        self._version_files.append(init_file)

    @property
    def current_version(self) -> SemanticVersion:
        """Get the current version."""
        if self._current_version is None:
            self._current_version = self._read_version()
        return self._current_version

    def _read_version(self) -> SemanticVersion:
        """Read version from primary version file (pyproject.toml)."""
        pyproject = self.project_root / "pyproject.toml"
        if not pyproject.exists():
            raise FileNotFoundError("pyproject.toml not found")

        content = pyproject.read_text(encoding="utf-8")
        match = re.search(self.VERSION_PATTERNS["pyproject.toml"], content)
        if not match:
            raise ValueError("Version not found in pyproject.toml")

        return parse_version(match.group(1))

    def bump_version(
        self,
        bump_type: VersionBumpType,
        prerelease_id: str | None = None,
        build_metadata: str | None = None,
        dry_run: bool = False,
    ) -> SemanticVersion:
        """
        Bump the version across all version files.

        Args:
            bump_type: Type of version bump.
            prerelease_id: Pre-release identifier for PRERELEASE bumps.
            build_metadata: Build metadata to include.
            dry_run: If True, don't actually modify files.

        Returns:
            The new version.
        """
        old_version = self.current_version
        new_version = old_version.bump(bump_type, prerelease_id, build_metadata)

        logger.info(
            "version_bump_initiated",
            old_version=str(old_version),
            new_version=str(new_version),
            bump_type=bump_type.value,
            dry_run=dry_run,
        )

        if not dry_run:
            self._update_version_files(old_version, new_version)
            self._current_version = new_version

        return new_version

    def _update_version_files(
        self,
        old_version: SemanticVersion,
        new_version: SemanticVersion,
    ) -> None:
        """Update version in all discovered version files."""
        old_str = str(old_version)
        new_str = str(new_version)

        for file_path in self._version_files:
            content = file_path.read_text(encoding="utf-8")
            updated_content = content.replace(f'"{old_str}"', f'"{new_str}"')

            if content != updated_content:
                file_path.write_text(updated_content, encoding="utf-8")
                logger.debug(
                    "version_file_updated",
                    file=str(file_path),
                    old_version=old_str,
                    new_version=new_str,
                )

    def set_version(
        self,
        version: SemanticVersion | str,
        dry_run: bool = False,
    ) -> SemanticVersion:
        """
        Set the version to a specific value.

        Args:
            version: The version to set.
            dry_run: If True, don't actually modify files.

        Returns:
            The new version.
        """
        if isinstance(version, str):
            version = parse_version(version)

        old_version = self.current_version

        logger.info(
            "version_set_initiated",
            old_version=str(old_version),
            new_version=str(version),
            dry_run=dry_run,
        )

        if not dry_run:
            self._update_version_files(old_version, version)
            self._current_version = version

        return version

    def get_version_history(self) -> list[dict[str, Any]]:
        """
        Get version history from git tags.

        Returns:
            List of version history entries.
        """
        import subprocess

        history: list[dict[str, Any]] = []

        try:
            result = subprocess.run(
                ["git", "tag", "-l", "v*", "--sort=-version:refname"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                check=True,
            )

            for tag in result.stdout.strip().split("\n"):
                if not tag:
                    continue

                version_str = tag.lstrip("v")
                try:
                    version = parse_version(version_str)

                    tag_date_result = subprocess.run(
                        ["git", "log", "-1", "--format=%ai", tag],
                        capture_output=True,
                        text=True,
                        cwd=self.project_root,
                        check=True,
                    )

                    history.append({
                        "version": version.to_dict(),
                        "tag": tag,
                        "date": tag_date_result.stdout.strip(),
                    })
                except ValueError:
                    logger.warning("invalid_version_tag", tag=tag)
                    continue

        except subprocess.CalledProcessError as e:
            logger.warning("git_tag_retrieval_failed", error=str(e))

        return history

    def create_version_tag(
        self,
        message: str | None = None,
        push: bool = False,
    ) -> str:
        """
        Create a git tag for the current version.

        Args:
            message: Tag message. Defaults to version string.
            push: If True, push the tag to remote.

        Returns:
            The created tag name.
        """
        import subprocess

        tag_name = f"v{self.current_version}"
        tag_message = message or f"Release {self.current_version}"

        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", tag_message],
            cwd=self.project_root,
            check=True,
        )

        logger.info("version_tag_created", tag=tag_name, message=tag_message)

        if push:
            subprocess.run(
                ["git", "push", "origin", tag_name],
                cwd=self.project_root,
                check=True,
            )
            logger.info("version_tag_pushed", tag=tag_name)

        return tag_name

    def validate_version_consistency(self) -> list[str]:
        """
        Validate that all version files have consistent versions.

        Returns:
            List of inconsistency messages, empty if all consistent.
        """
        issues: list[str] = []
        expected_version = str(self.current_version)

        for file_path in self._version_files:
            content = file_path.read_text(encoding="utf-8")
            filename = file_path.name

            if filename in self.VERSION_PATTERNS:
                pattern = self.VERSION_PATTERNS[filename]
                match = re.search(pattern, content)
                if match:
                    file_version = match.group(1)
                    if file_version != expected_version:
                        issues.append(
                            f"{file_path}: expected {expected_version}, found {file_version}"
                        )

        if issues:
            logger.warning("version_inconsistencies_found", issues=issues)
        else:
            logger.info("version_consistency_validated", version=expected_version)

        return issues

    def to_dict(self) -> dict[str, Any]:
        """Convert manager state to dictionary."""
        return {
            "project_root": str(self.project_root),
            "current_version": self.current_version.to_dict(),
            "version_files": [str(f) for f in self._version_files],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
