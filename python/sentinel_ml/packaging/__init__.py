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
    BuildTarget,
    BuildValidator,
)
from sentinel_ml.packaging.changelog import (
    ChangelogEntry,
    ChangelogEntryType,
    ChangelogGenerator,
    ChangelogManager,
)
from sentinel_ml.packaging.installer import (
    DependencyChecker,
    InstallationResult,
    InstallationVerifier,
)
from sentinel_ml.packaging.release import (
    ReleaseArtifact,
    ReleaseConfig,
    ReleaseManager,
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
    "BuildTarget",
    "BuildValidator",
    "ChangelogEntry",
    "ChangelogEntryType",
    "ChangelogGenerator",
    "ChangelogManager",
    "DependencyChecker",
    "InstallationResult",
    "InstallationVerifier",
    "ReleaseArtifact",
    "ReleaseConfig",
    "ReleaseManager",
    "ReleaseStatus",
    "SemanticVersion",
    "VersionBumpType",
    "VersionManager",
    "get_current_version",
    "parse_version",
]
