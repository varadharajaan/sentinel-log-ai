"""
Changelog Generation and Management for Sentinel Log AI.

This module provides tools for generating and managing changelogs
following the Keep a Changelog format (https://keepachangelog.com/).

Design Patterns:
- Factory Pattern: ChangelogEntry creation
- Template Method: Changelog formatting
- Strategy Pattern: Different output formats
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from sentinel_ml.packaging.version import SemanticVersion, parse_version

logger = structlog.get_logger(__name__)


class ChangelogEntryType(Enum):
    """
    Types of changelog entries following Keep a Changelog.

    Attributes:
        ADDED: New features.
        CHANGED: Changes in existing functionality.
        DEPRECATED: Soon-to-be removed features.
        REMOVED: Removed features.
        FIXED: Bug fixes.
        SECURITY: Security vulnerability fixes.
    """

    ADDED = "Added"
    CHANGED = "Changed"
    DEPRECATED = "Deprecated"
    REMOVED = "Removed"
    FIXED = "Fixed"
    SECURITY = "Security"


@dataclass
class ChangelogEntry:
    """
    A single changelog entry.

    Attributes:
        entry_type: Type of change.
        description: Description of the change.
        issue_refs: Related issue references (e.g., "#123").
        pr_refs: Related PR references (e.g., "#456").
        author: Author of the change.
        commit_hash: Git commit hash.
        breaking: Whether this is a breaking change.
    """

    entry_type: ChangelogEntryType
    description: str
    issue_refs: list[str] = field(default_factory=list)
    pr_refs: list[str] = field(default_factory=list)
    author: str | None = None
    commit_hash: str | None = None
    breaking: bool = False

    def __post_init__(self) -> None:
        """Validate entry."""
        if not self.description.strip():
            raise ValueError("Changelog entry description cannot be empty")

    def to_markdown(self, include_refs: bool = True) -> str:
        """
        Convert entry to markdown format.

        Args:
            include_refs: Whether to include issue/PR references.

        Returns:
            Markdown formatted entry.
        """
        prefix = "**BREAKING:** " if self.breaking else ""
        line = f"- {prefix}{self.description}"

        if include_refs:
            refs = []
            refs.extend(self.issue_refs)
            refs.extend(self.pr_refs)
            if refs:
                line += f" ({', '.join(refs)})"

        return line

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entry_type": self.entry_type.value,
            "description": self.description,
            "issue_refs": self.issue_refs,
            "pr_refs": self.pr_refs,
            "author": self.author,
            "commit_hash": self.commit_hash,
            "breaking": self.breaking,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangelogEntry:
        """Create from dictionary representation."""
        return cls(
            entry_type=ChangelogEntryType(data["entry_type"]),
            description=data["description"],
            issue_refs=data.get("issue_refs", []),
            pr_refs=data.get("pr_refs", []),
            author=data.get("author"),
            commit_hash=data.get("commit_hash"),
            breaking=data.get("breaking", False),
        )

    @classmethod
    def from_commit_message(cls, message: str) -> ChangelogEntry | None:
        """
        Parse a conventional commit message into a changelog entry.

        Supports conventional commits format:
        type(scope): description

        Args:
            message: Git commit message.

        Returns:
            ChangelogEntry or None if not parseable.
        """
        pattern = r"^(?P<type>\w+)(?:\((?P<scope>[^)]+)\))?\s*(?P<breaking>!)?\s*:\s*(?P<desc>.+)"
        match = re.match(pattern, message.strip().split("\n")[0])

        if not match:
            return None

        commit_type = match.group("type").lower()
        description = match.group("desc").strip()
        breaking = match.group("breaking") == "!"

        type_mapping = {
            "feat": ChangelogEntryType.ADDED,
            "feature": ChangelogEntryType.ADDED,
            "fix": ChangelogEntryType.FIXED,
            "bugfix": ChangelogEntryType.FIXED,
            "docs": ChangelogEntryType.CHANGED,
            "style": ChangelogEntryType.CHANGED,
            "refactor": ChangelogEntryType.CHANGED,
            "perf": ChangelogEntryType.CHANGED,
            "test": ChangelogEntryType.CHANGED,
            "chore": ChangelogEntryType.CHANGED,
            "build": ChangelogEntryType.CHANGED,
            "ci": ChangelogEntryType.CHANGED,
            "revert": ChangelogEntryType.REMOVED,
            "security": ChangelogEntryType.SECURITY,
            "deprecated": ChangelogEntryType.DEPRECATED,
        }

        entry_type = type_mapping.get(commit_type, ChangelogEntryType.CHANGED)

        issue_refs = re.findall(r"#(\d+)", message)

        return cls(
            entry_type=entry_type,
            description=description,
            issue_refs=[f"#{ref}" for ref in issue_refs],
            breaking=breaking,
        )


@dataclass
class ChangelogRelease:
    """
    A changelog release containing multiple entries.

    Attributes:
        version: Release version.
        date: Release date.
        entries: List of changelog entries.
        yanked: Whether this release was yanked.
        compare_url: URL to compare with previous version.
    """

    version: SemanticVersion | None
    date: datetime
    entries: dict[ChangelogEntryType, list[ChangelogEntry]] = field(default_factory=dict)
    yanked: bool = False
    compare_url: str | None = None

    @property
    def is_unreleased(self) -> bool:
        """Check if this is the unreleased section."""
        return self.version is None

    def add_entry(self, entry: ChangelogEntry) -> None:
        """Add an entry to the release."""
        if entry.entry_type not in self.entries:
            self.entries[entry.entry_type] = []
        self.entries[entry.entry_type].append(entry)

    def to_markdown(self) -> str:
        """Convert release to markdown format."""
        lines: list[str] = []

        if self.is_unreleased:
            header = "## [Unreleased]"
        else:
            version_str = str(self.version)
            date_str = self.date.strftime("%Y-%m-%d")
            yanked_suffix = " [YANKED]" if self.yanked else ""
            header = f"## [{version_str}] - {date_str}{yanked_suffix}"

        lines.append(header)
        lines.append("")

        for entry_type in ChangelogEntryType:
            if self.entries.get(entry_type):
                lines.append(f"### {entry_type.value}")
                lines.append("")
                for entry in self.entries[entry_type]:
                    lines.append(entry.to_markdown())
                lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "version": self.version.to_dict() if self.version else None,
            "date": self.date.isoformat(),
            "entries": {
                entry_type.value: [e.to_dict() for e in entries]
                for entry_type, entries in self.entries.items()
            },
            "yanked": self.yanked,
            "compare_url": self.compare_url,
        }


class ChangelogGenerator:
    """
    Generates changelog entries from git history.

    This class analyzes git commits and generates structured
    changelog entries based on conventional commit messages.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        """
        Initialize the changelog generator.

        Args:
            project_root: Root directory of the project.
        """
        self.project_root = project_root or Path.cwd()
        logger.info("changelog_generator_initialized", project_root=str(self.project_root))

    def generate_from_commits(
        self,
        from_ref: str | None = None,
        to_ref: str = "HEAD",
    ) -> list[ChangelogEntry]:
        """
        Generate changelog entries from git commits.

        Args:
            from_ref: Starting git reference (tag, commit, etc.).
            to_ref: Ending git reference.

        Returns:
            List of changelog entries.
        """
        import subprocess

        entries: list[ChangelogEntry] = []

        range_spec = f"{from_ref}..{to_ref}" if from_ref else to_ref

        try:
            result = subprocess.run(
                [
                    "git",
                    "log",
                    range_spec,
                    "--pretty=format:%H|%s|%an",
                    "--no-merges",
                ],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                check=True,
            )

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split("|", 2)
                if len(parts) < 3:
                    continue

                commit_hash, message, author = parts

                entry = ChangelogEntry.from_commit_message(message)
                if entry:
                    entry.commit_hash = commit_hash[:8]
                    entry.author = author
                    entries.append(entry)

            logger.info(
                "changelog_entries_generated",
                from_ref=from_ref,
                to_ref=to_ref,
                entry_count=len(entries),
            )

        except subprocess.CalledProcessError as e:
            logger.error("git_log_failed", error=str(e))

        return entries

    def get_latest_tag(self) -> str | None:
        """Get the latest git tag."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None


class ChangelogManager:
    """
    Manages the project changelog file.

    This class handles reading, writing, and updating the CHANGELOG.md file
    following the Keep a Changelog format.

    Attributes:
        changelog_path: Path to the CHANGELOG.md file.
        releases: List of changelog releases.
    """

    HEADER_TEMPLATE = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

"""

    def __init__(
        self,
        changelog_path: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        """
        Initialize the changelog manager.

        Args:
            changelog_path: Path to CHANGELOG.md. Defaults to project_root/CHANGELOG.md.
            project_root: Root directory of the project.
        """
        self.project_root = project_root or Path.cwd()
        self.changelog_path = changelog_path or self.project_root / "CHANGELOG.md"
        self.releases: list[ChangelogRelease] = []
        self._generator = ChangelogGenerator(self.project_root)

        if self.changelog_path.exists():
            self._parse_changelog()

        logger.info(
            "changelog_manager_initialized",
            changelog_path=str(self.changelog_path),
            release_count=len(self.releases),
        )

    def _parse_changelog(self) -> None:
        """Parse existing changelog file."""
        content = self.changelog_path.read_text(encoding="utf-8")
        self.releases = []

        release_pattern = r"## \[([^\]]+)\](?: - (\d{4}-\d{2}-\d{2}))?( \[YANKED\])?"
        section_pattern = r"### (\w+)"
        entry_pattern = r"^- (.+)$"

        current_release: ChangelogRelease | None = None
        current_section: ChangelogEntryType | None = None

        for line in content.split("\n"):
            release_match = re.match(release_pattern, line)
            if release_match:
                version_str = release_match.group(1)
                date_str = release_match.group(2)
                yanked = release_match.group(3) is not None

                if version_str.lower() == "unreleased":
                    version = None
                    date = datetime.now(timezone.utc)
                else:
                    try:
                        version = parse_version(version_str)
                    except ValueError:
                        continue
                    date = (
                        datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        if date_str
                        else datetime.now(timezone.utc)
                    )

                current_release = ChangelogRelease(
                    version=version,
                    date=date,
                    yanked=yanked,
                )
                self.releases.append(current_release)
                current_section = None
                continue

            section_match = re.match(section_pattern, line)
            if section_match and current_release:
                section_name = section_match.group(1)
                try:
                    current_section = ChangelogEntryType(section_name)
                except ValueError:
                    current_section = None
                continue

            entry_match = re.match(entry_pattern, line)
            if entry_match and current_release and current_section:
                description = entry_match.group(1)
                breaking = description.startswith("**BREAKING:**")
                if breaking:
                    description = description.replace("**BREAKING:** ", "")

                issue_refs = re.findall(r"(#\d+)", description)

                entry = ChangelogEntry(
                    entry_type=current_section,
                    description=re.sub(r"\s*\([^)]+\)\s*$", "", description),
                    issue_refs=issue_refs,
                    breaking=breaking,
                )
                current_release.add_entry(entry)

    def get_unreleased(self) -> ChangelogRelease | None:
        """Get the unreleased section."""
        for release in self.releases:
            if release.is_unreleased:
                return release
        return None

    def add_entry(self, entry: ChangelogEntry) -> None:
        """
        Add an entry to the unreleased section.

        Args:
            entry: Changelog entry to add.
        """
        unreleased = self.get_unreleased()
        if unreleased is None:
            unreleased = ChangelogRelease(
                version=None,
                date=datetime.now(timezone.utc),
            )
            self.releases.insert(0, unreleased)

        unreleased.add_entry(entry)

        logger.debug(
            "changelog_entry_added",
            entry_type=entry.entry_type.value,
            description=entry.description[:50],
        )

    def create_release(
        self,
        version: SemanticVersion,
        date: datetime | None = None,
    ) -> ChangelogRelease:
        """
        Create a new release from unreleased entries.

        Args:
            version: Version for the release.
            date: Release date. Defaults to now.

        Returns:
            The created release.
        """
        unreleased = self.get_unreleased()
        release_date = date or datetime.now(timezone.utc)

        release = ChangelogRelease(
            version=version,
            date=release_date,
            entries=unreleased.entries if unreleased else {},
        )

        if unreleased:
            self.releases.remove(unreleased)

        new_unreleased = ChangelogRelease(
            version=None,
            date=datetime.now(timezone.utc),
        )
        self.releases.insert(0, new_unreleased)
        self.releases.insert(1, release)

        logger.info(
            "changelog_release_created",
            version=str(version),
            entry_count=sum(len(entries) for entries in release.entries.values()),
        )

        return release

    def generate_unreleased(self, from_tag: str | None = None) -> int:
        """
        Generate unreleased entries from git commits.

        Args:
            from_tag: Starting tag. Defaults to latest tag.

        Returns:
            Number of entries generated.
        """
        if from_tag is None:
            from_tag = self._generator.get_latest_tag()

        entries = self._generator.generate_from_commits(from_ref=from_tag)

        for entry in entries:
            self.add_entry(entry)

        logger.info(
            "unreleased_entries_generated",
            from_tag=from_tag,
            entry_count=len(entries),
        )

        return len(entries)

    def save(self) -> None:
        """Save the changelog to file."""
        content = self.HEADER_TEMPLATE

        for release in self.releases:
            content += release.to_markdown()
            content += "\n"

        self.changelog_path.write_text(content, encoding="utf-8")

        logger.info("changelog_saved", path=str(self.changelog_path))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "changelog_path": str(self.changelog_path),
            "releases": [r.to_dict() for r in self.releases],
        }
