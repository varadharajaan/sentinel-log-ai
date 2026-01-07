"""
Release Management for Sentinel Log AI.

This module provides comprehensive release management including
artifact generation, distribution, and release notes.

Design Patterns:
- Facade Pattern: ReleaseManager provides unified interface
- State Pattern: ReleaseStatus for release workflow
- Builder Pattern: Release configuration and execution
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from sentinel_ml.packaging.build import BuildArtifact, BuildConfig, BuildRunner, BuildTarget
from sentinel_ml.packaging.changelog import ChangelogManager
from sentinel_ml.packaging.version import SemanticVersion, VersionBumpType, VersionManager

logger = structlog.get_logger(__name__)


class ReleaseStatus(Enum):
    """
    Status of a release.

    Attributes:
        DRAFT: Release is being prepared.
        PENDING: Release is ready for review.
        PUBLISHED: Release has been published.
        FAILED: Release failed.
        YANKED: Release was withdrawn.
    """

    DRAFT = "draft"
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"
    YANKED = "yanked"


@dataclass
class ReleaseArtifact:
    """
    Release artifact with distribution metadata.

    Attributes:
        build_artifact: The underlying build artifact.
        download_url: URL where artifact can be downloaded.
        upload_status: Upload status.
        uploaded_at: Upload timestamp.
    """

    build_artifact: BuildArtifact
    download_url: str | None = None
    upload_status: str = "pending"
    uploaded_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "build_artifact": self.build_artifact.to_dict(),
            "download_url": self.download_url,
            "upload_status": self.upload_status,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
        }


@dataclass
class ReleaseConfig:
    """
    Release configuration.

    Attributes:
        project_root: Root directory of the project.
        version: Version to release.
        targets: Build targets to include.
        draft: Create as draft release.
        prerelease: Mark as pre-release.
        generate_notes: Auto-generate release notes.
        upload_pypi: Upload to PyPI.
        upload_docker: Upload to Docker registry.
    """

    project_root: Path
    version: SemanticVersion
    targets: list[BuildTarget] = field(default_factory=list)
    draft: bool = False
    prerelease: bool = False
    generate_notes: bool = True
    upload_pypi: bool = False
    upload_docker: bool = False
    docker_registry: str = "ghcr.io"
    pypi_repository: str = "pypi"

    def __post_init__(self) -> None:
        """Set default targets if not specified."""
        if not self.targets:
            self.targets = [
                BuildTarget.PYTHON_WHEEL,
                BuildTarget.PYTHON_SDIST,
            ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "project_root": str(self.project_root),
            "version": self.version.to_dict(),
            "targets": [t.value for t in self.targets],
            "draft": self.draft,
            "prerelease": self.prerelease,
            "generate_notes": self.generate_notes,
            "upload_pypi": self.upload_pypi,
            "upload_docker": self.upload_docker,
        }


@dataclass
class Release:
    """
    A complete release.

    Attributes:
        version: Release version.
        status: Release status.
        artifacts: Release artifacts.
        notes: Release notes (markdown).
        created_at: Creation timestamp.
        published_at: Publication timestamp.
        tag_name: Git tag name.
        commit_sha: Git commit SHA.
    """

    version: SemanticVersion
    status: ReleaseStatus = ReleaseStatus.DRAFT
    artifacts: list[ReleaseArtifact] = field(default_factory=list)
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: datetime | None = None
    tag_name: str = ""
    commit_sha: str = ""

    def __post_init__(self) -> None:
        """Set default tag name."""
        if not self.tag_name:
            self.tag_name = f"v{self.version}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "version": self.version.to_dict(),
            "status": self.status.value,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "tag_name": self.tag_name,
            "commit_sha": self.commit_sha,
        }


class ReleaseNotesGenerator:
    """
    Generates release notes from changelog and git history.

    This class creates formatted release notes suitable for
    GitHub releases and other platforms.
    """

    def __init__(self, project_root: Path) -> None:
        """
        Initialize the generator.

        Args:
            project_root: Root directory of the project.
        """
        self.project_root = project_root
        self._changelog_manager = ChangelogManager(project_root=project_root)

        logger.info("release_notes_generator_initialized")

    def generate(
        self,
        version: SemanticVersion,
        include_contributors: bool = True,
        include_checksums: bool = True,
        artifacts: list[BuildArtifact] | None = None,
    ) -> str:
        """
        Generate release notes for a version.

        Args:
            version: Version to generate notes for.
            include_contributors: Include contributor list.
            include_checksums: Include artifact checksums.
            artifacts: Build artifacts to include.

        Returns:
            Markdown formatted release notes.
        """
        sections: list[str] = []

        sections.append(f"# Release {version}")
        sections.append("")

        changelog_section = self._get_changelog_section(version)
        if changelog_section:
            sections.append(changelog_section)

        if include_contributors:
            contributors = self._get_contributors()
            if contributors:
                sections.append("## Contributors")
                sections.append("")
                for contributor in contributors:
                    sections.append(f"- @{contributor}")
                sections.append("")

        if artifacts and include_checksums:
            sections.append("## Checksums")
            sections.append("")
            sections.append("| File | SHA-256 |")
            sections.append("|------|---------|")
            for artifact in artifacts:
                short_checksum = artifact.checksum_sha256[:16] + "..."
                sections.append(f"| {artifact.name} | `{short_checksum}` |")
            sections.append("")

        sections.append("## Installation")
        sections.append("")
        sections.append("```bash")
        sections.append(f"pip install sentinel-log-ai-ml=={version}")
        sections.append("```")
        sections.append("")

        logger.info("release_notes_generated", version=str(version))

        return "\n".join(sections)

    def _get_changelog_section(self, version: SemanticVersion) -> str:
        """Get changelog section for a version."""
        for release in self._changelog_manager.releases:
            if release.version == version:
                lines: list[str] = []
                lines.append("## What's Changed")
                lines.append("")

                for entry_type, entries in release.entries.items():
                    if entries:
                        lines.append(f"### {entry_type.value}")
                        lines.append("")
                        for entry in entries:
                            lines.append(entry.to_markdown())
                        lines.append("")

                return "\n".join(lines)

        return ""

    def _get_contributors(self) -> list[str]:
        """Get list of contributors from recent commits."""
        try:
            result = subprocess.run(
                [
                    "git", "log",
                    "--format=%aN",
                    "-n", "50",
                ],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                check=True,
            )

            contributors = set()
            for name in result.stdout.strip().split("\n"):
                if name and name != "dependabot[bot]":
                    contributors.add(name)

            return sorted(contributors)

        except subprocess.CalledProcessError:
            return []


class ReleaseManager:
    """
    Manages the complete release process.

    This class coordinates version bumping, building, changelog
    generation, and release publishing.

    Attributes:
        config: Release configuration.
        release: Current release being managed.
    """

    def __init__(self, config: ReleaseConfig) -> None:
        """
        Initialize the release manager.

        Args:
            config: Release configuration.
        """
        self.config = config
        self._version_manager = VersionManager(config.project_root)
        self._changelog_manager = ChangelogManager(project_root=config.project_root)
        self._notes_generator = ReleaseNotesGenerator(config.project_root)
        self.release: Release | None = None

        logger.info(
            "release_manager_initialized",
            version=str(config.version),
            targets=[t.value for t in config.targets],
        )

    def prepare(self) -> Release:
        """
        Prepare the release.

        This creates the release object, builds artifacts,
        and generates release notes.

        Returns:
            The prepared release.
        """
        logger.info("release_preparation_started", version=str(self.config.version))

        self.release = Release(
            version=self.config.version,
            status=ReleaseStatus.DRAFT,
        )

        try:
            self.release.commit_sha = self._get_current_commit()
        except subprocess.CalledProcessError:
            logger.warning("could_not_get_commit_sha")

        build_config = BuildConfig(
            project_root=self.config.project_root,
            output_dir=self.config.project_root / "dist",
            targets=self.config.targets,
            version=str(self.config.version),
        )

        builder = BuildRunner(build_config)
        build_artifacts = builder.build(validate=True)

        for artifact in build_artifacts:
            release_artifact = ReleaseArtifact(build_artifact=artifact)
            self.release.artifacts.append(release_artifact)

        if self.config.generate_notes:
            self.release.notes = self._notes_generator.generate(
                self.config.version,
                artifacts=[a.build_artifact for a in self.release.artifacts],
            )

        self.release.status = ReleaseStatus.PENDING

        logger.info(
            "release_prepared",
            version=str(self.config.version),
            artifact_count=len(self.release.artifacts),
        )

        return self.release

    def publish(self) -> Release:
        """
        Publish the release.

        This creates the git tag, uploads artifacts, and
        creates the GitHub release.

        Returns:
            The published release.

        Raises:
            RuntimeError: If release not prepared or publication fails.
        """
        if self.release is None:
            raise RuntimeError("Release not prepared. Call prepare() first.")

        if self.release.status not in (ReleaseStatus.PENDING, ReleaseStatus.DRAFT):
            raise RuntimeError(f"Cannot publish release in status: {self.release.status}")

        logger.info("release_publication_started", version=str(self.config.version))

        try:
            self._create_git_tag()

            if self.config.upload_pypi:
                self._upload_to_pypi()

            if self.config.upload_docker:
                self._upload_to_docker()

            self._create_github_release()

            self.release.status = ReleaseStatus.PUBLISHED
            self.release.published_at = datetime.now(timezone.utc)

            logger.info(
                "release_published",
                version=str(self.config.version),
                tag=self.release.tag_name,
            )

        except Exception as e:
            self.release.status = ReleaseStatus.FAILED
            logger.error("release_publication_failed", error=str(e))
            raise

        return self.release

    def _get_current_commit(self) -> str:
        """Get current git commit SHA."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=self.config.project_root,
            check=True,
        )
        return result.stdout.strip()

    def _create_git_tag(self) -> None:
        """Create and push git tag."""
        if self.release is None:
            return

        tag_name = self.release.tag_name
        message = f"Release {self.config.version}"

        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", message],
            cwd=self.config.project_root,
            check=True,
        )

        subprocess.run(
            ["git", "push", "origin", tag_name],
            cwd=self.config.project_root,
            check=True,
        )

        logger.info("git_tag_created", tag=tag_name)

    def _upload_to_pypi(self) -> None:
        """Upload packages to PyPI."""
        if self.release is None:
            return

        dist_dir = self.config.project_root / "dist"

        subprocess.run(
            [
                "python", "-m", "twine", "upload",
                "--repository", self.config.pypi_repository,
                str(dist_dir / "*.whl"),
                str(dist_dir / "*.tar.gz"),
            ],
            cwd=self.config.project_root,
            check=True,
        )

        for artifact in self.release.artifacts:
            if artifact.build_artifact.target in (
                BuildTarget.PYTHON_WHEEL,
                BuildTarget.PYTHON_SDIST,
            ):
                artifact.upload_status = "uploaded"
                artifact.uploaded_at = datetime.now(timezone.utc)

        logger.info("pypi_upload_completed")

    def _upload_to_docker(self) -> None:
        """Upload Docker image to registry."""
        if self.release is None:
            return

        image_name = f"sentinel-log-ai:{self.config.version}"
        registry_image = f"{self.config.docker_registry}/sentinel-log-ai:{self.config.version}"

        subprocess.run(
            ["docker", "tag", image_name, registry_image],
            check=True,
        )

        subprocess.run(
            ["docker", "push", registry_image],
            check=True,
        )

        for artifact in self.release.artifacts:
            if artifact.build_artifact.target == BuildTarget.DOCKER_IMAGE:
                artifact.upload_status = "uploaded"
                artifact.uploaded_at = datetime.now(timezone.utc)
                artifact.download_url = registry_image

        logger.info("docker_upload_completed", image=registry_image)

    def _create_github_release(self) -> None:
        """Create GitHub release."""
        if self.release is None:
            return

        args = [
            "gh", "release", "create",
            self.release.tag_name,
            "--title", f"Release {self.config.version}",
            "--notes", self.release.notes,
        ]

        if self.config.draft:
            args.append("--draft")

        if self.config.prerelease or self.config.version.is_prerelease():
            args.append("--prerelease")

        for artifact in self.release.artifacts:
            if artifact.build_artifact.path.exists():
                args.append(str(artifact.build_artifact.path))

        subprocess.run(
            args,
            cwd=self.config.project_root,
            check=True,
        )

        logger.info("github_release_created", tag=self.release.tag_name)

    def bump_and_prepare(
        self,
        bump_type: VersionBumpType,
        prerelease_id: str | None = None,
    ) -> Release:
        """
        Bump version and prepare release in one step.

        Args:
            bump_type: Type of version bump.
            prerelease_id: Pre-release identifier if applicable.

        Returns:
            The prepared release.
        """
        new_version = self._version_manager.bump_version(
            bump_type,
            prerelease_id=prerelease_id,
        )

        self._changelog_manager.create_release(new_version)
        self._changelog_manager.save()

        self.config.version = new_version

        return self.prepare()

    def save_release_manifest(self, path: Path | None = None) -> Path:
        """
        Save release manifest to JSON file.

        Args:
            path: Output path. Defaults to dist/release.json.

        Returns:
            Path to saved manifest.
        """
        if self.release is None:
            raise RuntimeError("No release to save")

        if path is None:
            path = self.config.project_root / "dist" / "release.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "release": self.release.to_dict(),
            "config": self.config.to_dict(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        logger.info("release_manifest_saved", path=str(path))

        return path

    def to_dict(self) -> dict[str, Any]:
        """Convert manager state to dictionary."""
        return {
            "config": self.config.to_dict(),
            "release": self.release.to_dict() if self.release else None,
        }
