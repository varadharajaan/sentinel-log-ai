"""
Slack webhook notifier for alert delivery.

Sends formatted messages to Slack channels via incoming webhooks.
Supports rich message formatting with attachments and blocks.
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


# Priority to color mapping for Slack attachments
PRIORITY_COLORS: dict[AlertPriority, str] = {
    AlertPriority.CRITICAL: "#dc3545",  # Red
    AlertPriority.HIGH: "#fd7e14",  # Orange
    AlertPriority.MEDIUM: "#ffc107",  # Yellow
    AlertPriority.LOW: "#17a2b8",  # Blue
    AlertPriority.INFO: "#6c757d",  # Gray
}


@dataclass
class SlackConfig(NotifierConfig):
    """
    Configuration for Slack notifier.

    Attributes:
        webhook_url: Slack incoming webhook URL.
        channel: Optional channel override.
        username: Bot username to display.
        icon_url: URL for bot avatar icon.
        include_metadata: Whether to include metadata in message.
        mention_users: User IDs to mention for critical alerts.
        mention_groups: Group IDs to mention for critical alerts.
    """

    webhook_url: str = ""
    channel: str | None = None
    username: str = "Sentinel ML"
    icon_url: str | None = None
    include_metadata: bool = True
    mention_users: list[str] = field(default_factory=list)
    mention_groups: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name or self.name == "base-notifier":
            self.name = "slack-notifier"


class SlackNotifier(BaseNotifier):
    """
    Slack webhook notifier.

    Sends formatted alert messages to Slack channels using incoming webhooks.
    Supports rich formatting with color-coded priority levels and optional
    metadata display.
    """

    def __init__(self, config: SlackConfig) -> None:
        """
        Initialize Slack notifier.

        Args:
            config: Slack-specific configuration.
        """
        super().__init__(config)
        self._slack_config = config
        self._logger = logger.bind(
            notifier=config.name,
            channel=config.channel,
        )

    def _send(self, event: AlertEvent) -> dict[str, Any]:
        """
        Send alert to Slack.

        Args:
            event: Alert event to send.

        Returns:
            Response data from Slack API.

        Raises:
            ValueError: If webhook URL is not configured.
            URLError: If connection fails.
            HTTPError: If Slack API returns an error.
        """
        if not self._slack_config.webhook_url:
            raise ValueError("Slack webhook URL is required")

        payload = self._build_payload(event)
        response = self._post_webhook(payload)

        return {"status": "ok", "response": response}

    def _build_payload(self, event: AlertEvent) -> dict[str, Any]:
        """
        Build Slack message payload.

        Args:
            event: Alert event to format.

        Returns:
            Slack message payload.
        """
        color = PRIORITY_COLORS.get(event.priority, "#6c757d")

        # Build mention string for critical alerts
        mentions = ""
        if event.priority == AlertPriority.CRITICAL:
            user_mentions = [f"<@{uid}>" for uid in self._slack_config.mention_users]
            group_mentions = [
                f"<!subteam^{gid}>" for gid in self._slack_config.mention_groups
            ]
            all_mentions = user_mentions + group_mentions
            if all_mentions:
                mentions = " ".join(all_mentions) + " "

        # Build fields for attachment
        fields: list[dict[str, Any]] = [
            {
                "title": "Priority",
                "value": event.priority.value.upper(),
                "short": True,
            },
            {
                "title": "Source",
                "value": event.source,
                "short": True,
            },
        ]

        if event.tags:
            fields.append({
                "title": "Tags",
                "value": ", ".join(event.tags),
                "short": True,
            })

        if self._slack_config.include_metadata and event.metadata:
            for key, value in list(event.metadata.items())[:5]:
                fields.append({
                    "title": key,
                    "value": str(value)[:100],
                    "short": True,
                })

        payload: dict[str, Any] = {
            "attachments": [
                {
                    "fallback": f"[{event.priority.value.upper()}] {event.title}",
                    "color": color,
                    "title": f"{mentions}{event.title}",
                    "text": event.message,
                    "fields": fields,
                    "footer": f"Event ID: {event.event_id}",
                    "ts": int(event.timestamp.timestamp()),
                }
            ]
        }

        if self._slack_config.channel:
            payload["channel"] = self._slack_config.channel

        if self._slack_config.username:
            payload["username"] = self._slack_config.username

        if self._slack_config.icon_url:
            payload["icon_url"] = self._slack_config.icon_url

        return payload

    def _post_webhook(self, payload: dict[str, Any]) -> str:
        """
        Post payload to Slack webhook.

        Args:
            payload: Message payload.

        Returns:
            Response body from Slack.

        Raises:
            URLError: If connection fails.
            HTTPError: If request fails.
        """
        data = json.dumps(payload).encode("utf-8")

        request = Request(
            self._slack_config.webhook_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Sentinel-ML/1.0",
            },
            method="POST",
        )

        try:
            with urlopen(
                request, timeout=self._slack_config.timeout_seconds
            ) as response:
                return response.read().decode("utf-8")
        except HTTPError as e:
            self._logger.error(
                "slack_http_error",
                status_code=e.code,
                reason=e.reason,
            )
            raise
        except URLError as e:
            self._logger.error(
                "slack_connection_error",
                reason=str(e.reason),
            )
            raise

    def validate_config(self) -> list[str]:
        """Validate Slack configuration."""
        errors = super().validate_config()

        if not self._slack_config.webhook_url:
            errors.append("Slack webhook URL is required")
        elif not self._slack_config.webhook_url.startswith("https://hooks.slack.com/"):
            if not self._slack_config.webhook_url.startswith("https://"):
                errors.append("Slack webhook URL must use HTTPS")

        return errors

    def health_check(self) -> bool:
        """Check if Slack notifier is operational."""
        if not super().health_check():
            return False
        return bool(self._slack_config.webhook_url)


# Register with factory
NotifierFactory.register("slack", SlackNotifier)
