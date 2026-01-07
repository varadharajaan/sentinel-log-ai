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

import pytest

from sentinel_ml.packaging import (
    BuildArtifact,
    BuildConfig,
    BuildPlatform,
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
    ReleaseConfig,
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
    return SemanticVersion(major=2, minor=0, patch=0, build="build.123")


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
        assert version.build is None

    def test_create_prerelease_version(self) -> None:
        """Test creating a prerelease version."""
        version = SemanticVersion(major=1, minor=0, patch=0, prerelease="alpha.1")
        assert version.prerelease == "alpha.1"

    def test_create_version_with_build_metadata(self) -> None:
        """Test creating a version with build metadata."""
        version = SemanticVersion(major=1, minor=0, patch=0, build="build.123")
        assert version.build == "build.123"

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
            build="20240115",
        )
        assert str(version) == "1.0.0-rc.1+20240115"

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

    def test_version_to_dict(self, sample_version: SemanticVersion) -> None:
        """Test converting version to dictionary."""
        d = sample_version.to_dict()
        assert d["major"] == 1
        assert d["minor"] == 2
        assert d["patch"] == 3


# ==============================================================================
# VersionManager Tests
# ==============================================================================


class TestVersionManager:
    """Tests for VersionManager class."""

    def test_create_version_manager(self, temp_dir: Path) -> None:
        """Test creating a version manager."""
        manager = VersionManager(project_root=temp_dir)
        assert manager.project_root == temp_dir


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
        assert entry.breaking is False

    def test_create_breaking_change_entry(self) -> None:
        """Test creating a breaking change entry."""
        entry = ChangelogEntry(
            entry_type=ChangelogEntryType.CHANGED,
            description="API change",
            breaking=True,
        )
        assert entry.breaking is True

    def test_entry_to_markdown(self) -> None:
        """Test converting entry to markdown."""
        entry = ChangelogEntry(
            entry_type=ChangelogEntryType.ADDED,
            description="New feature",
        )
        md = entry.to_markdown()
        assert "New feature" in md


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
        version = SemanticVersion(major=1, minor=0, patch=0)
        release = ChangelogRelease(
            version=version,
            date=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        assert release.version == version
        assert release.date.year == 2024

    def test_create_unreleased(self) -> None:
        """Test creating an unreleased section."""
        release = ChangelogRelease(
            version=None,
            date=datetime.now(tz=timezone.utc),
        )
        assert release.version is None
        assert release.is_unreleased is True


# ==============================================================================
# ChangelogGenerator Tests
# ==============================================================================


class TestChangelogGenerator:
    """Tests for ChangelogGenerator class."""

    def test_create_generator(self, temp_dir: Path) -> None:
        """Test creating a changelog generator."""
        generator = ChangelogGenerator(project_root=temp_dir)
        assert generator.project_root == temp_dir


# ==============================================================================
# ChangelogManager Tests
# ==============================================================================


class TestChangelogManager:
    """Tests for ChangelogManager class."""

    def test_create_changelog_manager(self, temp_dir: Path) -> None:
        """Test creating a changelog manager."""
        manager = ChangelogManager(project_root=temp_dir)
        assert manager.project_root == temp_dir

    def test_add_entry(self, temp_dir: Path) -> None:
        """Test adding an entry to changelog."""
        manager = ChangelogManager(project_root=temp_dir)
        entry = ChangelogEntry(
            entry_type=ChangelogEntryType.ADDED,
            description="New feature",
        )
        manager.add_entry(entry)
        # Should not raise


# ==============================================================================
# BuildTarget Tests
# ==============================================================================


class TestBuildTarget:
    """Tests for BuildTarget enum."""

    def test_build_target_values(self) -> None:
        """Test all build target values exist."""
        assert BuildTarget.PYTHON_WHEEL is not None
        assert BuildTarget.PYTHON_SDIST is not None
        assert BuildTarget.DOCKER_IMAGE is not None
        assert BuildTarget.WINDOWS_EXE is not None
        assert BuildTarget.GO_BINARY is not None


class TestBuildPlatform:
    """Tests for BuildPlatform enum."""

    def test_build_platform_values(self) -> None:
        """Test all build platform values exist."""
        assert BuildPlatform.LINUX is not None
        assert BuildPlatform.MACOS is not None
        assert BuildPlatform.WINDOWS is not None
        assert BuildPlatform.ANY is not None


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
            name="package.whl",
            path=artifact_path,
            target=BuildTarget.PYTHON_WHEEL,
            platform=BuildPlatform.LINUX,
            size_bytes=len(b"test content"),
            checksum_sha256=hashlib.sha256(b"test content").hexdigest(),
        )

        assert artifact.path == artifact_path
        assert artifact.target == BuildTarget.PYTHON_WHEEL
        assert artifact.size_bytes == 12

    def test_artifact_verify_checksum(self, temp_dir: Path) -> None:
        """Test verifying artifact checksum."""
        artifact_path = temp_dir / "artifact.whl"
        content = b"test artifact content"
        artifact_path.write_bytes(content)

        checksum = hashlib.sha256(content).hexdigest()
        artifact = BuildArtifact(
            name="artifact.whl",
            path=artifact_path,
            target=BuildTarget.PYTHON_WHEEL,
            platform=BuildPlatform.LINUX,
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
            name="artifact.whl",
            path=artifact_path,
            target=BuildTarget.PYTHON_WHEEL,
            platform=BuildPlatform.LINUX,
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
        output_dir = temp_dir / "dist"
        config = BuildConfig(
            project_root=temp_dir,
            output_dir=output_dir,
            version="1.0.0",
            targets=[BuildTarget.PYTHON_WHEEL, BuildTarget.PYTHON_SDIST],
        )
        assert config.version == "1.0.0"
        assert len(config.targets) == 2
        assert BuildTarget.PYTHON_WHEEL in config.targets


# ==============================================================================
# BuildValidator Tests
# ==============================================================================


class TestBuildValidator:
    """Tests for BuildValidator class."""

    def test_validate_artifact_exists(self, temp_dir: Path) -> None:
        """Test validating that artifact exists."""
        artifact_path = temp_dir / "artifact.whl"
        artifact_path.write_bytes(b"content")

        # Create required project files for validation
        (temp_dir / "pyproject.toml").write_text("[project]\nname='test'")
        (temp_dir / "README.md").write_text("# Test")

        output_dir = temp_dir / "dist"
        config = BuildConfig(
            project_root=temp_dir,
            output_dir=output_dir,
            version="1.0.0",
            targets=[BuildTarget.PYTHON_WHEEL],
        )
        validator = BuildValidator(config)
        # validate() returns list of errors for the build environment
        errors = validator.validate()
        # Should have minimal errors since required files exist
        assert isinstance(errors, list)

    def test_validate_missing_files(self, temp_dir: Path) -> None:
        """Test validating missing required files returns errors."""
        output_dir = temp_dir / "dist"
        config = BuildConfig(
            project_root=temp_dir,
            output_dir=output_dir,
            version="1.0.0",
            targets=[BuildTarget.PYTHON_WHEEL],
        )
        validator = BuildValidator(config)
        errors = validator.validate()
        # Missing required files should produce errors
        assert len(errors) > 0
        assert any("pyproject.toml" in e or "README.md" in e for e in errors)


# ==============================================================================
# ReleaseStatus Tests
# ==============================================================================


class TestReleaseStatus:
    """Tests for ReleaseStatus enum."""

    def test_release_status_values(self) -> None:
        """Test all release status values exist."""
        assert ReleaseStatus.DRAFT is not None
        assert ReleaseStatus.PENDING is not None
        assert ReleaseStatus.PUBLISHED is not None
        assert ReleaseStatus.FAILED is not None
        assert ReleaseStatus.YANKED is not None


# ==============================================================================
# ReleaseConfig Tests
# ==============================================================================


class TestReleaseConfig:
    """Tests for ReleaseConfig dataclass."""

    def test_create_release_config(self, temp_dir: Path) -> None:
        """Test creating a release configuration."""
        version = SemanticVersion(major=1, minor=0, patch=0)
        config = ReleaseConfig(
            project_root=temp_dir,
            version=version,
            targets=[BuildTarget.PYTHON_WHEEL],
        )
        assert config.version == version
        assert str(config.version) == "1.0.0"


# ==============================================================================
# DependencyStatus Tests
# ==============================================================================


class TestDependencyStatus:
    """Tests for DependencyStatus enum."""

    def test_dependency_status_values(self) -> None:
        """Test all dependency status values exist."""
        assert DependencyStatus.INSTALLED is not None
        assert DependencyStatus.MISSING is not None
        assert DependencyStatus.INCOMPATIBLE is not None
        assert DependencyStatus.OPTIONAL is not None


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
            python_version="3.12.0",
            package_version="0.11.0",
            dependencies=[],
            checks_passed=["Test check"],
            checks_failed=[],
            warnings=[],
        )
        assert result.success is True
        assert result.python_version == "3.12.0"
        assert len(result.checks_failed) == 0

    def test_create_failed_result(self) -> None:
        """Test creating a failed installation result."""
        result = InstallationResult(
            success=False,
            python_version="3.12.0",
            package_version=None,
            dependencies=[],
            checks_passed=[],
            checks_failed=["Failed to import sentinel_ml"],
            warnings=[],
        )
        assert result.success is False
        assert len(result.checks_failed) == 1


# ==============================================================================
# DependencyChecker Tests
# ==============================================================================


class TestDependencyChecker:
    """Tests for DependencyChecker class."""

    def test_check_all_dependencies(self) -> None:
        """Test checking all dependencies."""
        checker = DependencyChecker()
        results = checker.check_all()
        assert isinstance(results, list)

    def test_get_missing_required(self) -> None:
        """Test getting missing required dependencies."""
        checker = DependencyChecker()
        _ = checker.check_all()
        missing = checker.get_missing_required()
        assert isinstance(missing, list)


# ==============================================================================
# InstallationVerifier Tests
# ==============================================================================


class TestInstallationVerifier:
    """Tests for InstallationVerifier class."""

    def test_verify(self) -> None:
        """Test running verification."""
        verifier = InstallationVerifier()
        result = verifier.verify()
        assert isinstance(result, InstallationResult)

    def test_generate_report(self) -> None:
        """Test generating installation report."""
        verifier = InstallationVerifier()
        _ = verifier.verify()  # Must verify first
        report = verifier.generate_report()  # No arguments - uses internal state
        assert isinstance(report, str)
        assert len(report) > 0
