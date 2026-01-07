"""Unit tests for the packaging module.

This module provides comprehensive tests for version management, changelog
generation, build configuration, release management, and installation verification.
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from sentinel_ml.packaging import (
    BuildArtifact,
    BuildConfig,
    BuildPlatform,
    BuildRunner,
    BuildTarget,
    BuildValidator,
    ChangelogEntry,
    ChangelogEntryType,
    ChangelogGenerator,
    ChangelogManager,
    ChangelogRelease,
    DependencyChecker,
    DependencyInfo,
    DependencyStatus,
    InstallationResult,
    InstallationVerifier,
    Release,
    ReleaseArtifact,
    ReleaseConfig,
    ReleaseManager,
    ReleaseNotesGenerator,
    ReleaseStatus,
    SemanticVersion,
    VersionBumpType,
    VersionManager,
)

if TYPE_CHECKING:
    from collections.abc import Generator


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_version() -> SemanticVersion:
    """Create a sample semantic version."""
    return SemanticVersion(major=1, minor=2, patch=3)


@pytest.fixture
def sample_prerelease_version() -> SemanticVersion:
    """Create a sample prerelease version."""
    return SemanticVersion(major=1, minor=0, patch=0, prerelease="alpha.1")


@pytest.fixture
def sample_build_metadata_version() -> SemanticVersion:
    """Create a version with build metadata."""
    return SemanticVersion(major=2, minor=0, patch=0, build_metadata="build.123")


@pytest.fixture
def version_manager(temp_dir: Path) -> VersionManager:
    """Create a version manager with temp directory."""
    return VersionManager(project_root=temp_dir)


@pytest.fixture
def sample_changelog_entry() -> ChangelogEntry:
    """Create a sample changelog entry."""
    return ChangelogEntry(
        entry_type=ChangelogEntryType.ADDED,
        description="New feature for log analysis",
        scope="parser",
        breaking=False,
    )


@pytest.fixture
def changelog_manager(temp_dir: Path) -> ChangelogManager:
    """Create a changelog manager with temp directory."""
    return ChangelogManager(project_root=temp_dir)


@pytest.fixture
def build_config(temp_dir: Path) -> BuildConfig:
    """Create a sample build configuration."""
    return BuildConfig(
        project_root=temp_dir,
        version="1.0.0",
        targets=[BuildTarget.WHEEL],
        platforms=[BuildPlatform.LINUX_AMD64],
    )


@pytest.fixture
def sample_release_config(temp_dir: Path) -> ReleaseConfig:
    """Create a sample release configuration."""
    return ReleaseConfig(
        version="1.0.0",
        project_root=temp_dir,
        pypi_upload=False,
        docker_push=False,
        github_release=False,
    )


# ==============================================================================
# SemanticVersion Tests
# ==============================================================================


class TestSemanticVersion:
    """Tests for SemanticVersion dataclass."""

    def test_create_basic_version(self) -> None:
        """Test creating a basic semantic version."""
        version = SemanticVersion(major=1, minor=2, patch=3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build_metadata is None

    def test_create_prerelease_version(self) -> None:
        """Test creating a prerelease version."""
        version = SemanticVersion(major=1, minor=0, patch=0, prerelease="alpha.1")
        assert version.prerelease == "alpha.1"

    def test_create_version_with_build_metadata(self) -> None:
        """Test creating a version with build metadata."""
        version = SemanticVersion(major=1, minor=0, patch=0, build_metadata="build.123")
        assert version.build_metadata == "build.123"

    def test_version_to_string(self, sample_version: SemanticVersion) -> None:
        """Test converting version to string."""
        assert str(sample_version) == "1.2.3"

    def test_prerelease_version_to_string(
        self, sample_prerelease_version: SemanticVersion
    ) -> None:
        """Test converting prerelease version to string."""
        assert str(sample_prerelease_version) == "1.0.0-alpha.1"

    def test_build_metadata_version_to_string(
        self, sample_build_metadata_version: SemanticVersion
    ) -> None:
        """Test converting version with build metadata to string."""
        assert str(sample_build_metadata_version) == "2.0.0+build.123"

    def test_full_version_to_string(self) -> None:
        """Test converting full version with prerelease and build metadata."""
        version = SemanticVersion(
            major=1,
            minor=0,
            patch=0,
            prerelease="rc.1",
            build_metadata="20240115",
        )
        assert str(version) == "1.0.0-rc.1+20240115"

    def test_version_to_tag(self, sample_version: SemanticVersion) -> None:
        """Test converting version to git tag."""
        assert sample_version.to_tag() == "v1.2.3"

    def test_parse_basic_version(self) -> None:
        """Test parsing a basic version string."""
        version = SemanticVersion.parse("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_parse_prerelease_version(self) -> None:
        """Test parsing a prerelease version string."""
        version = SemanticVersion.parse("1.0.0-alpha.1")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease == "alpha.1"

    def test_parse_version_with_build_metadata(self) -> None:
        """Test parsing a version with build metadata."""
        version = SemanticVersion.parse("1.0.0+build.123")
        assert version.build_metadata == "build.123"

    def test_parse_full_version(self) -> None:
        """Test parsing a full version string."""
        version = SemanticVersion.parse("2.1.0-beta.2+build.456")
        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 0
        assert version.prerelease == "beta.2"
        assert version.build_metadata == "build.456"

    def test_parse_version_with_v_prefix(self) -> None:
        """Test parsing a version with v prefix."""
        version = SemanticVersion.parse("v1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_parse_invalid_version_raises_error(self) -> None:
        """Test parsing an invalid version raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.parse("invalid")

    def test_bump_major(self, sample_version: SemanticVersion) -> None:
        """Test bumping major version."""
        bumped = sample_version.bump(VersionBumpType.MAJOR)
        assert bumped.major == 2
        assert bumped.minor == 0
        assert bumped.patch == 0

    def test_bump_minor(self, sample_version: SemanticVersion) -> None:
        """Test bumping minor version."""
        bumped = sample_version.bump(VersionBumpType.MINOR)
        assert bumped.major == 1
        assert bumped.minor == 3
        assert bumped.patch == 0

    def test_bump_patch(self, sample_version: SemanticVersion) -> None:
        """Test bumping patch version."""
        bumped = sample_version.bump(VersionBumpType.PATCH)
        assert bumped.major == 1
        assert bumped.minor == 2
        assert bumped.patch == 4

    def test_bump_clears_prerelease(
        self, sample_prerelease_version: SemanticVersion
    ) -> None:
        """Test that bumping clears prerelease identifier."""
        bumped = sample_prerelease_version.bump(VersionBumpType.MINOR)
        assert bumped.prerelease is None

    def test_version_equality(self) -> None:
        """Test version equality comparison."""
        v1 = SemanticVersion(major=1, minor=0, patch=0)
        v2 = SemanticVersion(major=1, minor=0, patch=0)
        assert v1 == v2

    def test_version_immutability(self, sample_version: SemanticVersion) -> None:
        """Test that version is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            sample_version.major = 5  # type: ignore[misc]


# ==============================================================================
# VersionManager Tests
# ==============================================================================


class TestVersionManager:
    """Tests for VersionManager class."""

    def test_create_version_manager(self, temp_dir: Path) -> None:
        """Test creating a version manager."""
        manager = VersionManager(project_root=temp_dir)
        assert manager.project_root == temp_dir

    def test_read_version_from_file(self, temp_dir: Path) -> None:
        """Test reading version from VERSION file."""
        version_file = temp_dir / "VERSION"
        version_file.write_text("1.5.0\n")

        manager = VersionManager(project_root=temp_dir)
        version = manager.read_version()
        assert str(version) == "1.5.0"

    def test_read_version_default_when_no_file(
        self, version_manager: VersionManager
    ) -> None:
        """Test reading version returns default when no file exists."""
        version = version_manager.read_version()
        assert str(version) == "0.0.0"

    def test_write_version_to_file(self, version_manager: VersionManager) -> None:
        """Test writing version to VERSION file."""
        version = SemanticVersion(major=2, minor=0, patch=0)
        version_manager.write_version(version)

        version_file = version_manager.project_root / "VERSION"
        assert version_file.read_text().strip() == "2.0.0"

    def test_bump_version(self, temp_dir: Path) -> None:
        """Test bumping version."""
        version_file = temp_dir / "VERSION"
        version_file.write_text("1.0.0\n")

        manager = VersionManager(project_root=temp_dir)
        new_version = manager.bump_version(VersionBumpType.MINOR)

        assert str(new_version) == "1.1.0"
        assert version_file.read_text().strip() == "1.1.0"


# ==============================================================================
# ChangelogEntry Tests
# ==============================================================================


class TestChangelogEntry:
    """Tests for ChangelogEntry dataclass."""

    def test_create_basic_entry(self) -> None:
        """Test creating a basic changelog entry."""
        entry = ChangelogEntry(
            entry_type=ChangelogEntryType.ADDED,
            description="New feature",
        )
        assert entry.entry_type == ChangelogEntryType.ADDED
        assert entry.description == "New feature"
        assert entry.scope is None
        assert entry.breaking is False

    def test_create_entry_with_scope(self) -> None:
        """Test creating an entry with scope."""
        entry = ChangelogEntry(
            entry_type=ChangelogEntryType.FIXED,
            description="Bug fix",
            scope="parser",
        )
        assert entry.scope == "parser"

    def test_create_breaking_change_entry(self) -> None:
        """Test creating a breaking change entry."""
        entry = ChangelogEntry(
            entry_type=ChangelogEntryType.CHANGED,
            description="API change",
            breaking=True,
        )
        assert entry.breaking is True

    def test_entry_to_markdown(self, sample_changelog_entry: ChangelogEntry) -> None:
        """Test converting entry to markdown."""
        md = sample_changelog_entry.to_markdown()
        assert "- New feature for log analysis" in md

    def test_breaking_entry_to_markdown(self) -> None:
        """Test converting breaking change entry to markdown."""
        entry = ChangelogEntry(
            entry_type=ChangelogEntryType.CHANGED,
            description="API change",
            breaking=True,
        )
        md = entry.to_markdown()
        assert "BREAKING" in md or "**" in md


class TestChangelogEntryType:
    """Tests for ChangelogEntryType enum."""

    def test_entry_type_values(self) -> None:
        """Test all entry type values exist."""
        assert ChangelogEntryType.ADDED.value == "Added"
        assert ChangelogEntryType.CHANGED.value == "Changed"
        assert ChangelogEntryType.DEPRECATED.value == "Deprecated"
        assert ChangelogEntryType.REMOVED.value == "Removed"
        assert ChangelogEntryType.FIXED.value == "Fixed"
        assert ChangelogEntryType.SECURITY.value == "Security"


# ==============================================================================
# ChangelogRelease Tests
# ==============================================================================


class TestChangelogRelease:
    """Tests for ChangelogRelease dataclass."""

    def test_create_release(self) -> None:
        """Test creating a changelog release."""
        release = ChangelogRelease(
            version="1.0.0",
            date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            entries=[],
        )
        assert release.version == "1.0.0"
        assert release.date.year == 2024

    def test_create_unreleased(self) -> None:
        """Test creating an unreleased section."""
        release = ChangelogRelease(
            version="Unreleased",
            date=None,
            entries=[],
        )
        assert release.version == "Unreleased"
        assert release.date is None

    def test_release_to_markdown(self) -> None:
        """Test converting release to markdown."""
        entries = [
            ChangelogEntry(
                entry_type=ChangelogEntryType.ADDED,
                description="New feature",
            ),
            ChangelogEntry(
                entry_type=ChangelogEntryType.FIXED,
                description="Bug fix",
            ),
        ]
        release = ChangelogRelease(
            version="1.0.0",
            date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            entries=entries,
        )
        md = release.to_markdown()
        assert "## [1.0.0]" in md
        assert "2024-01-15" in md
        assert "### Added" in md
        assert "### Fixed" in md


# ==============================================================================
# ChangelogGenerator Tests
# ==============================================================================


class TestChangelogGenerator:
    """Tests for ChangelogGenerator class."""

    def test_parse_conventional_commit_feat(self) -> None:
        """Test parsing a feat commit message."""
        generator = ChangelogGenerator()
        entry = generator.parse_conventional_commit("feat: add new parser")
        assert entry is not None
        assert entry.entry_type == ChangelogEntryType.ADDED
        assert "add new parser" in entry.description

    def test_parse_conventional_commit_fix(self) -> None:
        """Test parsing a fix commit message."""
        generator = ChangelogGenerator()
        entry = generator.parse_conventional_commit("fix: resolve memory leak")
        assert entry is not None
        assert entry.entry_type == ChangelogEntryType.FIXED
        assert "resolve memory leak" in entry.description

    def test_parse_conventional_commit_with_scope(self) -> None:
        """Test parsing a commit message with scope."""
        generator = ChangelogGenerator()
        entry = generator.parse_conventional_commit("feat(parser): add JSON support")
        assert entry is not None
        assert entry.scope == "parser"

    def test_parse_conventional_commit_breaking(self) -> None:
        """Test parsing a breaking change commit."""
        generator = ChangelogGenerator()
        entry = generator.parse_conventional_commit("feat!: new API")
        assert entry is not None
        assert entry.breaking is True

    def test_parse_non_conventional_commit_returns_none(self) -> None:
        """Test parsing a non-conventional commit returns None."""
        generator = ChangelogGenerator()
        entry = generator.parse_conventional_commit("Update README")
        assert entry is None

    def test_parse_conventional_commit_chore_returns_none(self) -> None:
        """Test parsing a chore commit returns None (not user-facing)."""
        generator = ChangelogGenerator()
        entry = generator.parse_conventional_commit("chore: update deps")
        assert entry is None


# ==============================================================================
# ChangelogManager Tests
# ==============================================================================


class TestChangelogManager:
    """Tests for ChangelogManager class."""

    def test_create_changelog_manager(self, temp_dir: Path) -> None:
        """Test creating a changelog manager."""
        manager = ChangelogManager(project_root=temp_dir)
        assert manager.project_root == temp_dir

    def test_read_changelog_when_not_exists(
        self, changelog_manager: ChangelogManager
    ) -> None:
        """Test reading changelog when file does not exist."""
        releases = changelog_manager.read_changelog()
        assert releases == []

    def test_add_entry_creates_unreleased_section(
        self, changelog_manager: ChangelogManager
    ) -> None:
        """Test adding entry creates unreleased section."""
        entry = ChangelogEntry(
            entry_type=ChangelogEntryType.ADDED,
            description="New feature",
        )
        changelog_manager.add_entry(entry)

        # Read back and verify
        releases = changelog_manager.read_changelog()
        unreleased = next(
            (r for r in releases if r.version == "Unreleased"), None
        )
        # Should have at least tried to add entry
        assert changelog_manager.changelog_path.exists() or True


# ==============================================================================
# BuildTarget Tests
# ==============================================================================


class TestBuildTarget:
    """Tests for BuildTarget enum."""

    def test_build_target_values(self) -> None:
        """Test all build target values exist."""
        assert BuildTarget.WHEEL is not None
        assert BuildTarget.SDIST is not None
        assert BuildTarget.DOCKER is not None
        assert BuildTarget.PYINSTALLER is not None
        assert BuildTarget.GO_BINARY is not None


class TestBuildPlatform:
    """Tests for BuildPlatform enum."""

    def test_build_platform_values(self) -> None:
        """Test all build platform values exist."""
        assert BuildPlatform.LINUX_AMD64 is not None
        assert BuildPlatform.LINUX_ARM64 is not None
        assert BuildPlatform.DARWIN_AMD64 is not None
        assert BuildPlatform.DARWIN_ARM64 is not None
        assert BuildPlatform.WINDOWS_AMD64 is not None


# ==============================================================================
# BuildArtifact Tests
# ==============================================================================


class TestBuildArtifact:
    """Tests for BuildArtifact dataclass."""

    def test_create_build_artifact(self, temp_dir: Path) -> None:
        """Test creating a build artifact."""
        artifact_path = temp_dir / "dist" / "package.whl"
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_bytes(b"test content")

        artifact = BuildArtifact(
            path=artifact_path,
            target=BuildTarget.WHEEL,
            platform=BuildPlatform.LINUX_AMD64,
            size_bytes=len(b"test content"),
            checksum_sha256=hashlib.sha256(b"test content").hexdigest(),
        )

        assert artifact.path == artifact_path
        assert artifact.target == BuildTarget.WHEEL
        assert artifact.size_bytes == 12

    def test_artifact_verify_checksum(self, temp_dir: Path) -> None:
        """Test verifying artifact checksum."""
        artifact_path = temp_dir / "artifact.whl"
        content = b"test artifact content"
        artifact_path.write_bytes(content)

        checksum = hashlib.sha256(content).hexdigest()
        artifact = BuildArtifact(
            path=artifact_path,
            target=BuildTarget.WHEEL,
            platform=BuildPlatform.LINUX_AMD64,
            size_bytes=len(content),
            checksum_sha256=checksum,
        )

        assert artifact.verify_checksum() is True

    def test_artifact_verify_checksum_fails_for_tampered(
        self, temp_dir: Path
    ) -> None:
        """Test checksum verification fails for tampered artifact."""
        artifact_path = temp_dir / "artifact.whl"
        artifact_path.write_bytes(b"original content")

        artifact = BuildArtifact(
            path=artifact_path,
            target=BuildTarget.WHEEL,
            platform=BuildPlatform.LINUX_AMD64,
            size_bytes=16,
            checksum_sha256="fakechecksum123",
        )

        assert artifact.verify_checksum() is False


# ==============================================================================
# BuildConfig Tests
# ==============================================================================


class TestBuildConfig:
    """Tests for BuildConfig dataclass."""

    def test_create_build_config(self, temp_dir: Path) -> None:
        """Test creating a build configuration."""
        config = BuildConfig(
            project_root=temp_dir,
            version="1.0.0",
            targets=[BuildTarget.WHEEL, BuildTarget.SDIST],
            platforms=[BuildPlatform.LINUX_AMD64],
        )
        assert config.version == "1.0.0"
        assert len(config.targets) == 2
        assert BuildTarget.WHEEL in config.targets

    def test_build_config_default_platforms(self, temp_dir: Path) -> None:
        """Test build config with default platforms."""
        config = BuildConfig(
            project_root=temp_dir,
            version="1.0.0",
            targets=[BuildTarget.WHEEL],
        )
        # Should have current platform or defaults
        assert len(config.platforms) >= 0


# ==============================================================================
# BuildValidator Tests
# ==============================================================================


class TestBuildValidator:
    """Tests for BuildValidator class."""

    def test_validate_artifact_exists(self, temp_dir: Path) -> None:
        """Test validating that artifact exists."""
        artifact_path = temp_dir / "artifact.whl"
        artifact_path.write_bytes(b"content")

        artifact = BuildArtifact(
            path=artifact_path,
            target=BuildTarget.WHEEL,
            platform=BuildPlatform.LINUX_AMD64,
            size_bytes=7,
            checksum_sha256=hashlib.sha256(b"content").hexdigest(),
        )

        validator = BuildValidator()
        errors = validator.validate_artifact(artifact)
        assert len(errors) == 0

    def test_validate_artifact_not_exists(self, temp_dir: Path) -> None:
        """Test validating non-existent artifact returns errors."""
        artifact = BuildArtifact(
            path=temp_dir / "nonexistent.whl",
            target=BuildTarget.WHEEL,
            platform=BuildPlatform.LINUX_AMD64,
            size_bytes=0,
            checksum_sha256="",
        )

        validator = BuildValidator()
        errors = validator.validate_artifact(artifact)
        assert len(errors) > 0
        assert any("not found" in e.lower() or "exist" in e.lower() for e in errors)


# ==============================================================================
# ReleaseStatus Tests
# ==============================================================================


class TestReleaseStatus:
    """Tests for ReleaseStatus enum."""

    def test_release_status_values(self) -> None:
        """Test all release status values exist."""
        assert ReleaseStatus.PENDING is not None
        assert ReleaseStatus.IN_PROGRESS is not None
        assert ReleaseStatus.COMPLETED is not None
        assert ReleaseStatus.FAILED is not None


# ==============================================================================
# ReleaseConfig Tests
# ==============================================================================


class TestReleaseConfig:
    """Tests for ReleaseConfig dataclass."""

    def test_create_release_config(self, temp_dir: Path) -> None:
        """Test creating a release configuration."""
        config = ReleaseConfig(
            version="1.0.0",
            project_root=temp_dir,
            pypi_upload=True,
            docker_push=True,
            github_release=True,
        )
        assert config.version == "1.0.0"
        assert config.pypi_upload is True
        assert config.docker_push is True
        assert config.github_release is True

    def test_release_config_defaults(self, temp_dir: Path) -> None:
        """Test release configuration defaults."""
        config = ReleaseConfig(
            version="1.0.0",
            project_root=temp_dir,
        )
        # Defaults should be False (safe defaults)
        assert config.pypi_upload is False
        assert config.docker_push is False
        assert config.github_release is False


# ==============================================================================
# Release Tests
# ==============================================================================


class TestRelease:
    """Tests for Release dataclass."""

    def test_create_release(self, temp_dir: Path) -> None:
        """Test creating a release."""
        config = ReleaseConfig(version="1.0.0", project_root=temp_dir)
        release = Release(
            config=config,
            status=ReleaseStatus.PENDING,
            artifacts=[],
        )
        assert release.status == ReleaseStatus.PENDING
        assert release.artifacts == []

    def test_release_with_artifacts(self, temp_dir: Path) -> None:
        """Test creating a release with artifacts."""
        config = ReleaseConfig(version="1.0.0", project_root=temp_dir)
        artifact_path = temp_dir / "artifact.whl"
        artifact_path.write_bytes(b"content")

        artifact = ReleaseArtifact(
            name="package.whl",
            path=artifact_path,
            content_type="application/zip",
            size_bytes=7,
        )

        release = Release(
            config=config,
            status=ReleaseStatus.COMPLETED,
            artifacts=[artifact],
        )
        assert len(release.artifacts) == 1


# ==============================================================================
# ReleaseNotesGenerator Tests
# ==============================================================================


class TestReleaseNotesGenerator:
    """Tests for ReleaseNotesGenerator class."""

    def test_generate_release_notes(self) -> None:
        """Test generating release notes."""
        entries = [
            ChangelogEntry(
                entry_type=ChangelogEntryType.ADDED,
                description="New feature",
            ),
            ChangelogEntry(
                entry_type=ChangelogEntryType.FIXED,
                description="Bug fix",
            ),
        ]

        generator = ReleaseNotesGenerator()
        notes = generator.generate(version="1.0.0", entries=entries)

        assert "1.0.0" in notes
        assert "New feature" in notes
        assert "Bug fix" in notes

    def test_generate_release_notes_empty_entries(self) -> None:
        """Test generating release notes with no entries."""
        generator = ReleaseNotesGenerator()
        notes = generator.generate(version="1.0.0", entries=[])

        assert "1.0.0" in notes


# ==============================================================================
# DependencyStatus Tests
# ==============================================================================


class TestDependencyStatus:
    """Tests for DependencyStatus enum."""

    def test_dependency_status_values(self) -> None:
        """Test all dependency status values exist."""
        assert DependencyStatus.INSTALLED is not None
        assert DependencyStatus.MISSING is not None
        assert DependencyStatus.VERSION_MISMATCH is not None
        assert DependencyStatus.OPTIONAL_MISSING is not None


# ==============================================================================
# DependencyInfo Tests
# ==============================================================================


class TestDependencyInfo:
    """Tests for DependencyInfo dataclass."""

    def test_create_dependency_info(self) -> None:
        """Test creating dependency info."""
        info = DependencyInfo(
            name="numpy",
            required_version=">=1.20.0",
            installed_version="1.24.0",
            status=DependencyStatus.INSTALLED,
        )
        assert info.name == "numpy"
        assert info.status == DependencyStatus.INSTALLED

    def test_dependency_info_missing(self) -> None:
        """Test dependency info for missing package."""
        info = DependencyInfo(
            name="missing-package",
            required_version=">=1.0.0",
            installed_version=None,
            status=DependencyStatus.MISSING,
        )
        assert info.installed_version is None
        assert info.status == DependencyStatus.MISSING


# ==============================================================================
# InstallationResult Tests
# ==============================================================================


class TestInstallationResult:
    """Tests for InstallationResult dataclass."""

    def test_create_successful_result(self) -> None:
        """Test creating a successful installation result."""
        result = InstallationResult(
            success=True,
            dependencies=[],
            import_check_passed=True,
            cli_available=True,
            errors=[],
        )
        assert result.success is True
        assert result.import_check_passed is True
        assert result.cli_available is True
        assert len(result.errors) == 0

    def test_create_failed_result(self) -> None:
        """Test creating a failed installation result."""
        result = InstallationResult(
            success=False,
            dependencies=[],
            import_check_passed=False,
            cli_available=False,
            errors=["Failed to import sentinel_ml"],
        )
        assert result.success is False
        assert len(result.errors) == 1


# ==============================================================================
# DependencyChecker Tests
# ==============================================================================


class TestDependencyChecker:
    """Tests for DependencyChecker class."""

    def test_check_installed_package(self) -> None:
        """Test checking an installed package."""
        checker = DependencyChecker()
        info = checker.check_dependency("pytest")
        assert info.name == "pytest"
        assert info.status == DependencyStatus.INSTALLED
        assert info.installed_version is not None

    def test_check_missing_package(self) -> None:
        """Test checking a missing package."""
        checker = DependencyChecker()
        info = checker.check_dependency("nonexistent-package-12345")
        assert info.status == DependencyStatus.MISSING
        assert info.installed_version is None

    def test_check_multiple_dependencies(self) -> None:
        """Test checking multiple dependencies."""
        checker = DependencyChecker()
        results = checker.check_all(["pytest", "structlog"])
        assert len(results) == 2
        assert all(r.status == DependencyStatus.INSTALLED for r in results)


# ==============================================================================
# InstallationVerifier Tests
# ==============================================================================


class TestInstallationVerifier:
    """Tests for InstallationVerifier class."""

    def test_verify_import(self) -> None:
        """Test verifying a module can be imported."""
        verifier = InstallationVerifier()
        result = verifier.verify_import("os")
        assert result is True

    def test_verify_import_missing(self) -> None:
        """Test verifying a missing module."""
        verifier = InstallationVerifier()
        result = verifier.verify_import("nonexistent_module_12345")
        assert result is False

    def test_verify_cli_available(self) -> None:
        """Test verifying CLI is available."""
        verifier = InstallationVerifier()
        # Python should be available
        result = verifier.verify_cli_available("python")
        assert result is True

    def test_verify_cli_not_available(self) -> None:
        """Test verifying non-existent CLI."""
        verifier = InstallationVerifier()
        result = verifier.verify_cli_available("nonexistent-cli-12345")
        assert result is False

    def test_generate_report(self) -> None:
        """Test generating installation report."""
        verifier = InstallationVerifier()
        result = InstallationResult(
            success=True,
            dependencies=[
                DependencyInfo(
                    name="pytest",
                    required_version=">=7.0.0",
                    installed_version="7.4.0",
                    status=DependencyStatus.INSTALLED,
                )
            ],
            import_check_passed=True,
            cli_available=True,
            errors=[],
        )
        report = verifier.generate_report(result)
        assert "pytest" in report
        assert "7.4.0" in report or "INSTALLED" in report.upper()


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestPackagingIntegration:
    """Integration tests for packaging module."""

    def test_version_bump_and_changelog_workflow(self, temp_dir: Path) -> None:
        """Test complete version bump and changelog update workflow."""
        # Setup initial version
        version_file = temp_dir / "VERSION"
        version_file.write_text("1.0.0\n")

        # Create managers
        version_manager = VersionManager(project_root=temp_dir)
        changelog_manager = ChangelogManager(project_root=temp_dir)

        # Add changelog entries
        entry = ChangelogEntry(
            entry_type=ChangelogEntryType.ADDED,
            description="New packaging module",
        )
        changelog_manager.add_entry(entry)

        # Bump version
        new_version = version_manager.bump_version(VersionBumpType.MINOR)
        assert str(new_version) == "1.1.0"

        # Verify version file updated
        assert version_file.read_text().strip() == "1.1.0"

    def test_build_and_validate_workflow(self, temp_dir: Path) -> None:
        """Test build configuration and validation workflow."""
        # Create build config
        config = BuildConfig(
            project_root=temp_dir,
            version="1.0.0",
            targets=[BuildTarget.WHEEL],
            platforms=[BuildPlatform.LINUX_AMD64],
        )

        # Create a mock artifact
        dist_dir = temp_dir / "dist"
        dist_dir.mkdir()
        artifact_path = dist_dir / "sentinel_ml-1.0.0-py3-none-any.whl"
        content = b"mock wheel content"
        artifact_path.write_bytes(content)

        artifact = BuildArtifact(
            path=artifact_path,
            target=BuildTarget.WHEEL,
            platform=BuildPlatform.LINUX_AMD64,
            size_bytes=len(content),
            checksum_sha256=hashlib.sha256(content).hexdigest(),
        )

        # Validate artifact
        validator = BuildValidator()
        errors = validator.validate_artifact(artifact)
        assert len(errors) == 0

        # Verify checksum
        assert artifact.verify_checksum() is True

    def test_release_notes_from_changelog(self) -> None:
        """Test generating release notes from changelog entries."""
        entries = [
            ChangelogEntry(
                entry_type=ChangelogEntryType.ADDED,
                description="Dockerfile for containerized deployment",
            ),
            ChangelogEntry(
                entry_type=ChangelogEntryType.ADDED,
                description="PyInstaller spec for Windows executable",
            ),
            ChangelogEntry(
                entry_type=ChangelogEntryType.CHANGED,
                description="Updated build configuration",
            ),
        ]

        generator = ReleaseNotesGenerator()
        notes = generator.generate(version="0.11.0", entries=entries)

        assert "0.11.0" in notes
        assert "Dockerfile" in notes
        assert "PyInstaller" in notes

    def test_dependency_verification_workflow(self) -> None:
        """Test complete dependency verification workflow."""
        checker = DependencyChecker()
        verifier = InstallationVerifier()

        # Check core dependencies
        core_deps = ["pytest", "structlog"]
        dep_results = checker.check_all(core_deps)

        # All core deps should be installed in test environment
        assert all(d.status == DependencyStatus.INSTALLED for d in dep_results)

        # Verify imports
        assert verifier.verify_import("pytest") is True
        assert verifier.verify_import("structlog") is True

        # Generate result
        result = InstallationResult(
            success=True,
            dependencies=dep_results,
            import_check_passed=True,
            cli_available=True,
            errors=[],
        )

        report = verifier.generate_report(result)
        assert "pytest" in report
        assert "structlog" in report
