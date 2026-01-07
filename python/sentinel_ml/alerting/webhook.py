"""
Generic webhook notifier for alert delivery.

Sends JSON payloads to arbitrary webhook endpoints.
Supports custom headers, authentication, and payload templates.
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
    BaseNotifier,
    NotifierConfig,
    NotifierFactory,
)

logger = structlog.get_logger(__name__)


@dataclass
class WebhookConfig(NotifierConfig):
    """
    Configuration for webhook notifier.

    Attributes:
        url: Webhook endpoint URL.
        method: HTTP method (POST, PUT).
        headers: Custom HTTP headers.
        auth_type: Authentication type (none, bearer, basic, api_key).
        auth_token: Bearer token or API key.
        auth_username: Basic auth username.
        auth_password: Basic auth password.
        api_key_header: Header name for API key auth.
        include_timestamp: Include ISO timestamp in payload.
        custom_fields: Additional fields to include in payload.
        payload_template: Custom payload structure (if None, uses default).
    """

    url: str = ""
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    auth_type: str = "none"
    auth_token: str = ""
    auth_username: str = ""
    auth_password: str = ""
    api_key_header: str = "X-API-Key"
    include_timestamp: bool = True
    custom_fields: dict[str, Any] = field(default_factory=dict)
    payload_template: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name == "base-notifier":
            self.name = "webhook-notifier"


class WebhookNotifier(BaseNotifier):
    """
    Generic webhook notifier.

    Sends JSON payloads to webhook endpoints with configurable
    authentication and headers.
    """

    def __init__(self, config: WebhookConfig) -> None:
        """
        Initialize webhook notifier.

        Args:
            config: Webhook-specific configuration.
        """
        super().__init__(config)
        self._webhook_config = config
        self._logger = logger.bind(
            notifier=config.name,
            url=config.url,
        )

    def _send(self, event: AlertEvent) -> dict[str, Any]:
        """
        Send alert via webhook.

        Args:
            event: Alert event to send.

        Returns:
            Response data from webhook.

        Raises:
            ValueError: If URL is not configured.
            URLError: If connection fails.
            HTTPError: If request fails.
        """
        if not self._webhook_config.url:
            raise ValueError("Webhook URL is required")

        payload = self._build_payload(event)
        headers = self._build_headers()

        response = self._post_webhook(payload, headers)

        return {
            "status": "sent",
            "response_code": response.get("code", 200),
            "response_body": response.get("body", ""),
        }

    def _build_payload(self, event: AlertEvent) -> dict[str, Any]:
        """
        Build webhook payload.

        Args:
            event: Alert event to format.

        Returns:
            Payload dictionary.
        """
        if self._webhook_config.payload_template:
            # Use custom template with variable substitution
            return self._apply_template(
                self._webhook_config.payload_template, event
            )

        # Default payload structure
        payload: dict[str, Any] = {
            "event_id": event.event_id,
            "title": event.title,
            "message": event.message,
            "priority": event.priority.value,
            "source": event.source,
            "tags": event.tags,
            "metadata": event.metadata,
        }

        if self._webhook_config.include_timestamp:
            payload["timestamp"] = event.timestamp.isoformat()

        # Add custom fields
        payload.update(self._webhook_config.custom_fields)

        return payload

    def _apply_template(
        self, template: dict[str, Any], event: AlertEvent
    ) -> dict[str, Any]:
        """
        Apply event data to payload template.

        Args:
            template: Template dictionary.
            event: Event data source.

        Returns:
            Populated payload.
        """
        result: dict[str, Any] = {}
        event_dict = event.to_dict()

        for key, value in template.items():
            if isinstance(value, str) and value.startswith("$"):
                # Variable substitution
                var_name = value[1:]
                result[key] = event_dict.get(var_name, value)
            elif isinstance(value, dict):
                result[key] = self._apply_template(value, event)
            else:
                result[key] = value

        return result

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers including authentication."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Sentinel-ML/1.0",
        }

        # Add custom headers
        headers.update(self._webhook_config.headers)

        # Add authentication
        config = self._webhook_config
        if config.auth_type == "bearer" and config.auth_token:
            headers["Authorization"] = f"Bearer {config.auth_token}"
        elif config.auth_type == "api_key" and config.auth_token:
            headers[config.api_key_header] = config.auth_token
        elif config.auth_type == "basic" and config.auth_username:
            import base64

            credentials = f"{config.auth_username}:{config.auth_password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        return headers

    def _post_webhook(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        """
        Post payload to webhook.

        Args:
            payload: JSON payload.
            headers: HTTP headers.

        Returns:
            Response data.

        Raises:
            URLError: If connection fails.
            HTTPError: If request fails.
        """
        data = json.dumps(payload).encode("utf-8")

        request = Request(
            self._webhook_config.url,
            data=data,
            headers=headers,
            method=self._webhook_config.method,
        )

        try:
            with urlopen(
                request, timeout=self._webhook_config.timeout_seconds
            ) as response:
                body = response.read().decode("utf-8")
                return {
                    "code": response.status,
                    "body": body,
                }
        except HTTPError as e:
            self._logger.error(
                "webhook_http_error",
                status_code=e.code,
                reason=e.reason,
            )
            raise
        except URLError as e:
            self._logger.error(
                "webhook_connection_error",
                reason=str(e.reason),
            )
            raise

    def validate_config(self) -> list[str]:
        """Validate webhook configuration."""
        errors = super().validate_config()

        if not self._webhook_config.url:
            errors.append("Webhook URL is required")
        elif not self._webhook_config.url.startswith(("http://", "https://")):
            errors.append("Webhook URL must use HTTP or HTTPS")

        if self._webhook_config.method not in ("POST", "PUT", "PATCH"):
            errors.append("HTTP method must be POST, PUT, or PATCH")

        if self._webhook_config.auth_type not in ("none", "bearer", "basic", "api_key"):
            errors.append("Invalid auth_type")

        return errors

    def health_check(self) -> bool:
        """Check if webhook notifier is operational."""
        if not super().health_check():
            return False
        return bool(self._webhook_config.url)


# Register with factory
NotifierFactory.register("webhook", WebhookNotifier)
