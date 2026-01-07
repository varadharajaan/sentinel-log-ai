"""
Base classes for the alerting framework.

Provides abstract base classes and common types for notification delivery.
All notifier implementations extend BaseNotifier and implement the
required interface for consistent alert handling.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


class AlertPriority(str, Enum):
    """Alert priority levels for routing and display."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @classmethod
    def from_score(cls, novelty_score: float) -> AlertPriority:
        """
        Determine priority from a novelty score.

        Args:
            novelty_score: Score from 0.0 to 1.0.

        Returns:
            Appropriate priority level.
        """
        if novelty_score >= 0.9:
            return cls.CRITICAL
        if novelty_score >= 0.7:
            return cls.HIGH
        if novelty_score >= 0.5:
            return cls.MEDIUM
        if novelty_score >= 0.3:
            return cls.LOW
        return cls.INFO


class AlertStatus(str, Enum):
    """Status of an alert delivery attempt."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


@dataclass
class AlertEvent:
    """
    Represents an alert to be delivered.

    Attributes:
        event_id: Unique identifier for this alert.
        title: Short summary of the alert.
        message: Detailed alert message.
        priority: Priority level for routing.
        source: Origin of the alert (e.g., component name).
        timestamp: When the event occurred.
        metadata: Additional structured data.
        tags: Labels for categorization.
    """

    title: str
    message: str
    priority: AlertPriority = AlertPriority.MEDIUM
    source: str = "sentinel-ml"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "title": self.title,
            "message": self.message,
            "priority": self.priority.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags,
        }


@dataclass
class AlertResult:
    """
    Result of an alert delivery attempt.

    Attributes:
        event_id: ID of the alert that was processed.
        status: Delivery status.
        notifier_name: Name of the notifier that processed this.
        attempts: Number of delivery attempts.
        error: Error message if failed.
        response_data: Response from the notification service.
        delivered_at: When the alert was successfully delivered.
    """

    event_id: str
    status: AlertStatus
    notifier_name: str
    attempts: int = 1
    error: str | None = None
    response_data: dict[str, Any] = field(default_factory=dict)
    delivered_at: datetime | None = None

    @property
    def is_success(self) -> bool:
        """Check if delivery was successful."""
        return self.status == AlertStatus.SENT


@dataclass
class NotifierConfig:
    """
    Base configuration for notifiers.

    Attributes:
        name: Unique name for this notifier instance.
        enabled: Whether this notifier is active.
        max_retries: Maximum retry attempts on failure.
        retry_delay_seconds: Delay between retries.
        timeout_seconds: Request timeout.
        batch_size: Maximum alerts per batch.
        rate_limit_per_minute: Maximum alerts per minute.
    """

    name: str = "base-notifier"
    enabled: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0
    batch_size: int = 10
    rate_limit_per_minute: int = 60


class BaseNotifier(ABC):
    """
    Abstract base class for all notifier implementations.

    Provides common functionality for alert delivery including:
    - Retry logic with exponential backoff
    - Rate limiting
    - Logging and metrics

    Subclasses must implement the _send method for their
    specific notification channel.
    """

    def __init__(self, config: NotifierConfig) -> None:
        """
        Initialize the notifier.

        Args:
            config: Notifier configuration.
        """
        self._config = config
        self._logger = logger.bind(notifier=config.name)
        self._sent_count = 0
        self._failed_count = 0

    @property
    def name(self) -> str:
        """Get the notifier name."""
        return self._config.name

    @property
    def is_enabled(self) -> bool:
        """Check if the notifier is enabled."""
        return self._config.enabled

    @property
    def stats(self) -> dict[str, int]:
        """Get delivery statistics."""
        return {
            "sent": self._sent_count,
            "failed": self._failed_count,
            "total": self._sent_count + self._failed_count,
        }

    def send(self, event: AlertEvent) -> AlertResult:
        """
        Send an alert with retry logic.

        Args:
            event: The alert event to send.

        Returns:
            Result of the delivery attempt.
        """
        if not self._config.enabled:
            self._logger.debug(
                "notifier_disabled",
                event_id=event.event_id,
            )
            return AlertResult(
                event_id=event.event_id,
                status=AlertStatus.SKIPPED,
                notifier_name=self.name,
            )

        attempts = 0
        last_error: str | None = None

        while attempts < self._config.max_retries:
            attempts += 1
            try:
                self._logger.debug(
                    "sending_alert",
                    event_id=event.event_id,
                    attempt=attempts,
                )

                response_data = self._send(event)

                self._sent_count += 1
                self._logger.info(
                    "alert_sent",
                    event_id=event.event_id,
                    priority=event.priority.value,
                    attempts=attempts,
                )

                return AlertResult(
                    event_id=event.event_id,
                    status=AlertStatus.SENT,
                    notifier_name=self.name,
                    attempts=attempts,
                    response_data=response_data,
                    delivered_at=datetime.now(tz=timezone.utc),
                )

            except Exception as e:
                last_error = str(e)
                self._logger.warning(
                    "alert_send_failed",
                    event_id=event.event_id,
                    attempt=attempts,
                    error=last_error,
                )

                if attempts < self._config.max_retries:
                    import time

                    delay = self._config.retry_delay_seconds * (2 ** (attempts - 1))
                    time.sleep(delay)

        self._failed_count += 1
        self._logger.error(
            "alert_delivery_exhausted",
            event_id=event.event_id,
            attempts=attempts,
            error=last_error,
        )

        return AlertResult(
            event_id=event.event_id,
            status=AlertStatus.FAILED,
            notifier_name=self.name,
            attempts=attempts,
            error=last_error,
        )

    def send_batch(self, events: Sequence[AlertEvent]) -> list[AlertResult]:
        """
        Send multiple alerts.

        Args:
            events: Sequence of alert events.

        Returns:
            List of results for each event.
        """
        results: list[AlertResult] = []
        batch_size = self._config.batch_size

        for i in range(0, len(events), batch_size):
            batch = events[i : i + batch_size]
            for event in batch:
                results.append(self.send(event))

        return results

    @abstractmethod
    def _send(self, event: AlertEvent) -> dict[str, Any]:
        """
        Perform the actual send operation.

        Args:
            event: The alert event to send.

        Returns:
            Response data from the notification service.

        Raises:
            Exception: If sending fails.
        """

    def validate_config(self) -> list[str]:
        """
        Validate the notifier configuration.

        Returns:
            List of validation error messages.
        """
        errors: list[str] = []
        if not self._config.name:
            errors.append("Notifier name is required")
        if self._config.max_retries < 0:
            errors.append("max_retries must be non-negative")
        if self._config.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        return errors

    def health_check(self) -> bool:
        """
        Check if the notifier is operational.

        Returns:
            True if healthy, False otherwise.
        """
        return self._config.enabled


class NotifierFactory:
    """
    Factory for creating notifier instances.

    Implements the Factory Pattern for dynamic notifier creation
    based on configuration.
    """

    _registry: ClassVar[dict[str, type[BaseNotifier]]] = {}

    @classmethod
    def register(cls, notifier_type: str, notifier_class: type[BaseNotifier]) -> None:
        """
        Register a notifier type.

        Args:
            notifier_type: Unique type identifier.
            notifier_class: Notifier class to register.
        """
        cls._registry[notifier_type] = notifier_class
        logger.debug(
            "notifier_registered",
            notifier_type=notifier_type,
        )

    @classmethod
    def create(cls, notifier_type: str, config: NotifierConfig) -> BaseNotifier:
        """
        Create a notifier instance.

        Args:
            notifier_type: Type of notifier to create.
            config: Configuration for the notifier.

        Returns:
            New notifier instance.

        Raises:
            ValueError: If notifier type is not registered.
        """
        if notifier_type not in cls._registry:
            raise ValueError(f"Unknown notifier type: {notifier_type}")

        notifier_class = cls._registry[notifier_type]
        return notifier_class(config)

    @classmethod
    def available_types(cls) -> list[str]:
        """Get list of registered notifier types."""
        return list(cls._registry.keys())
