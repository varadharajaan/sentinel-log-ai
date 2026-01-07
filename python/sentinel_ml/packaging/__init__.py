"""
Packaging and Release Module for Sentinel Log AI.

This module provides utilities for:
- Version management and semantic versioning
- Changelog generation and automation
- Build configuration and validation
- Release artifact management
- Installation verification

Design Patterns:
- Strategy Pattern: Different version bump strategies
- Factory Pattern: Changelog entry creation
- Builder Pattern: Release artifact construction
- Observer Pattern: Version change notifications
"""

from sentinel_ml.packaging.build import (
    BuildArtifact,
    BuildConfig,
    BuildPlatform,
    BuildRunner,
    BuildTarget,
    BuildValidator,
)
from sentinel_ml.packaging.changelog import (
    ChangelogEntry,
    ChangelogEntryType,
    ChangelogGenerator,
    ChangelogManager,
    ChangelogRelease,
)
from sentinel_ml.packaging.installer import (
    DependencyChecker,
    DependencyInfo,
    DependencyStatus,
    InstallationResult,
    InstallationVerifier,
)
from sentinel_ml.packaging.release import (
    Release,
    ReleaseArtifact,
    ReleaseConfig,
    ReleaseManager,
    ReleaseNotesGenerator,
    ReleaseStatus,
)
from sentinel_ml.packaging.version import (
    SemanticVersion,
    VersionBumpType,
    VersionManager,
    get_current_version,
    parse_version,
)

__all__ = [
    "BuildArtifact",
    "BuildConfig",
    "BuildPlatform",
    "BuildRunner",
    "BuildTarget",
    "BuildValidator",
    "ChangelogEntry",
    "ChangelogEntryType",
    "ChangelogGenerator",
    "ChangelogManager",
    "ChangelogRelease",
    "DependencyChecker",
    "DependencyInfo",
    "DependencyStatus",
    "InstallationResult",
    "InstallationVerifier",
    "Release",
    "ReleaseArtifact",
    "ReleaseConfig",
    "ReleaseManager",
    "ReleaseNotesGenerator",
    "ReleaseStatus",
    "SemanticVersion",
    "VersionBumpType",
    "VersionManager",
    "get_current_version",
    "parse_version",
]
