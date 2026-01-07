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

from sentinel_ml.packaging.version import (
    SemanticVersion,
    VersionBumpType,
    VersionManager,
    get_current_version,
    parse_version,
)
from sentinel_ml.packaging.changelog import (
    ChangelogEntry,
    ChangelogEntryType,
    ChangelogGenerator,
    ChangelogManager,
)
from sentinel_ml.packaging.build import (
    BuildConfig,
    BuildTarget,
    BuildValidator,
    BuildArtifact,
)
from sentinel_ml.packaging.release import (
    ReleaseConfig,
    ReleaseManager,
    ReleaseArtifact,
    ReleaseStatus,
)
from sentinel_ml.packaging.installer import (
    InstallationVerifier,
    InstallationResult,
    DependencyChecker,
)

__all__ = [
    # Version
    "SemanticVersion",
    "VersionBumpType",
    "VersionManager",
    "get_current_version",
    "parse_version",
    # Changelog
    "ChangelogEntry",
    "ChangelogEntryType",
    "ChangelogGenerator",
    "ChangelogManager",
    # Build
    "BuildConfig",
    "BuildTarget",
    "BuildValidator",
    "BuildArtifact",
    # Release
    "ReleaseConfig",
    "ReleaseManager",
    "ReleaseArtifact",
    "ReleaseStatus",
    # Installer
    "InstallationVerifier",
    "InstallationResult",
    "DependencyChecker",
]
