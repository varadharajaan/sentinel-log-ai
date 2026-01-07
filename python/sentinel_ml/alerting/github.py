"""
GitHub issue creator for novel event tracking.

Automatically creates GitHub issues for novel log events,
enabling tracking and investigation workflows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import structlog

from sentinel_ml.alerting.base import (
    AlertEvent,
    AlertPriority,
    BaseNotifier,
    NotifierConfig,
    NotifierFactory,
)

logger = structlog.get_logger(__name__)


# Priority to label mapping
PRIORITY_LABELS: dict[AlertPriority, str] = {
    AlertPriority.CRITICAL: "priority: critical",
    AlertPriority.HIGH: "priority: high",
    AlertPriority.MEDIUM: "priority: medium",
    AlertPriority.LOW: "priority: low",
    AlertPriority.INFO: "priority: info",
}


@dataclass
class GitHubConfig(NotifierConfig):
    """
    Configuration for GitHub issue creator.

    Attributes:
        token: GitHub personal access token or GitHub App token.
        owner: Repository owner (user or organization).
        repo: Repository name.
        api_base_url: GitHub API base URL (for Enterprise).
        default_labels: Labels to add to all issues.
        default_assignees: Users to assign to all issues.
        milestone: Milestone ID to assign (optional).
        include_metadata: Include event metadata in issue body.
        deduplicate: Check for existing issues with same title.
        dedupe_label: Label to filter for deduplication.
    """

    token: str = ""
    owner: str = ""
    repo: str = ""
    api_base_url: str = "https://api.github.com"
    default_labels: list[str] = field(default_factory=lambda: ["sentinel-ml", "auto-generated"])
    default_assignees: list[str] = field(default_factory=list)
    milestone: int | None = None
    include_metadata: bool = True
    deduplicate: bool = True
    dedupe_label: str = "sentinel-ml"

    def __post_init__(self) -> None:
        if not self.name or self.name == "base-notifier":
            self.name = "github-notifier"


class GitHubIssueCreator(BaseNotifier):
    """
    GitHub issue creator for novel events.

    Creates GitHub issues from alert events with proper formatting,
    labels, and optional deduplication.
    """

    def __init__(self, config: GitHubConfig) -> None:
        """
        Initialize GitHub issue creator.

        Args:
            config: GitHub-specific configuration.
        """
        super().__init__(config)
        self._github_config = config
        self._logger = logger.bind(
            notifier=config.name,
            repo=f"{config.owner}/{config.repo}",
        )

    def _send(self, event: AlertEvent) -> dict[str, Any]:
        """
        Create GitHub issue for alert.

        Args:
            event: Alert event to create issue for.

        Returns:
            Response data including issue URL.

        Raises:
            ValueError: If configuration is incomplete.
            HTTPError: If GitHub API returns an error.
        """
        self._validate_required_config()

        # Check for duplicates if enabled
        if self._github_config.deduplicate:
            existing = self._find_existing_issue(event.title)
            if existing:
                self._logger.info(
                    "github_issue_exists",
                    event_id=event.event_id,
                    issue_url=existing["html_url"],
                )
                return {
                    "status": "duplicate",
                    "issue_url": existing["html_url"],
                    "issue_number": existing["number"],
                }

        # Create new issue
        issue = self._create_issue(event)

        return {
            "status": "created",
            "issue_url": issue["html_url"],
            "issue_number": issue["number"],
        }

    def _validate_required_config(self) -> None:
        """Validate required configuration fields."""
        if not self._github_config.token:
            raise ValueError("GitHub token is required")
        if not self._github_config.owner:
            raise ValueError("Repository owner is required")
        if not self._github_config.repo:
            raise ValueError("Repository name is required")

    def _create_issue(self, event: AlertEvent) -> dict[str, Any]:
        """
        Create a new GitHub issue.

        Args:
            event: Alert event to create issue for.

        Returns:
            Created issue data.
        """
        body = self._build_issue_body(event)
        labels = self._build_labels(event)

        payload: dict[str, Any] = {
            "title": event.title,
            "body": body,
            "labels": labels,
        }

        if self._github_config.default_assignees:
            payload["assignees"] = self._github_config.default_assignees

        if self._github_config.milestone:
            payload["milestone"] = self._github_config.milestone

        url = f"{self._github_config.api_base_url}/repos/{self._github_config.owner}/{self._github_config.repo}/issues"

        response = self._api_request("POST", url, payload)

        self._logger.info(
            "github_issue_created",
            event_id=event.event_id,
            issue_number=response["number"],
        )

        return response

    def _build_issue_body(self, event: AlertEvent) -> str:
        """Build GitHub issue body in Markdown format."""
        lines = [
            "## Alert Details",
            "",
            f"**Priority:** {event.priority.value.upper()}",
            f"**Source:** {event.source}",
            f"**Event ID:** `{event.event_id}`",
            f"**Timestamp:** {event.timestamp.isoformat()}",
            "",
            "## Message",
            "",
            event.message,
        ]

        if event.tags:
            lines.extend(
                [
                    "",
                    "## Tags",
                    "",
                    ", ".join(f"`{tag}`" for tag in event.tags),
                ]
            )

        if self._github_config.include_metadata and event.metadata:
            lines.extend(
                [
                    "",
                    "## Metadata",
                    "",
                    "| Key | Value |",
                    "|-----|-------|",
                ]
            )
            for key, value in event.metadata.items():
                # Escape pipe characters in values
                safe_value = str(value).replace("|", "\\|")[:200]
                lines.append(f"| {key} | {safe_value} |")

        lines.extend(
            [
                "",
                "---",
                "",
                "*This issue was automatically created by Sentinel ML.*",
            ]
        )

        return "\n".join(lines)

    def _build_labels(self, event: AlertEvent) -> list[str]:
        """Build list of labels for the issue."""
        labels = list(self._github_config.default_labels)

        # Add priority label
        priority_label = PRIORITY_LABELS.get(event.priority)
        if priority_label:
            labels.append(priority_label)

        # Add event tags as labels (sanitized)
        for tag in event.tags[:5]:  # Limit to 5 tags
            safe_tag = tag.lower().replace(" ", "-")[:50]
            labels.append(safe_tag)

        return labels

    def _find_existing_issue(self, title: str) -> dict[str, Any] | None:
        """
        Search for existing issue with same title.

        Args:
            title: Issue title to search for.

        Returns:
            Existing issue data or None.
        """
        # Build search query
        query_parts = [
            f"repo:{self._github_config.owner}/{self._github_config.repo}",
            "is:issue",
            "is:open",
            f"label:{self._github_config.dedupe_label}",
            f'"{title}"',
        ]
        query = " ".join(query_parts)

        url = f"{self._github_config.api_base_url}/search/issues"

        try:
            response = self._api_request(
                "GET",
                url,
                params={"q": query, "per_page": "1"},
            )

            if response.get("total_count", 0) > 0:
                items = response.get("items", [])
                if items:
                    return dict(items[0])

        except HTTPError as e:
            if e.code != 404:
                self._logger.warning(
                    "github_search_failed",
                    error=str(e),
                )

        return None

    def _api_request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make GitHub API request.

        Args:
            method: HTTP method.
            url: API endpoint URL.
            payload: Request body for POST/PUT/PATCH.
            params: Query parameters for GET.

        Returns:
            Response JSON data.

        Raises:
            HTTPError: If request fails.
        """
        headers = {
            "Authorization": f"Bearer {self._github_config.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Sentinel-ML/1.0",
        }

        if params:
            from urllib.parse import urlencode

            url = f"{url}?{urlencode(params)}"

        data = None
        if payload:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(url, data=data, headers=headers, method=method)

        try:
            with urlopen(request, timeout=self._github_config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                result: dict[str, Any] = json.loads(body)
                return result
        except HTTPError as e:
            self._logger.error(
                "github_api_error",
                status_code=e.code,
                reason=e.reason,
                url=url,
            )
            raise
        except URLError as e:
            self._logger.error(
                "github_connection_error",
                reason=str(e.reason),
            )
            raise

    def validate_config(self) -> list[str]:
        """Validate GitHub configuration."""
        errors = super().validate_config()

        if not self._github_config.token:
            errors.append("GitHub token is required")
        if not self._github_config.owner:
            errors.append("Repository owner is required")
        if not self._github_config.repo:
            errors.append("Repository name is required")

        return errors

    def health_check(self) -> bool:
        """Check if GitHub notifier is operational."""
        if not super().health_check():
            return False

        try:
            url = f"{self._github_config.api_base_url}/repos/{self._github_config.owner}/{self._github_config.repo}"
            self._api_request("GET", url)
            return True
        except Exception:
            return False


# Register with factory
NotifierFactory.register("github", GitHubIssueCreator)
