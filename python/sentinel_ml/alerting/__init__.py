"""
Alerting and integration module for Sentinel Log AI.

This module provides notification and integration capabilities:
- Multi-channel notification delivery (Slack, Email, Webhooks)
- Watch mode daemon for continuous monitoring
- GitHub issue auto-creation for novel events
- Health check endpoints for monitoring

Design Patterns:
- Strategy Pattern: Pluggable notification channels
- Observer Pattern: Event subscription and delivery
- Template Method: Common notification workflow
- Factory Pattern: Notifier creation
- Chain of Responsibility: Alert routing

SOLID Principles:
- Single Responsibility: Each notifier handles one channel
- Open/Closed: Extensible via new notifier implementations
- Liskov Substitution: All notifiers implement BaseNotifier
- Interface Segregation: Separate sync/async interfaces
- Dependency Inversion: Depends on notifier abstractions
"""

from sentinel_ml.alerting.base import (
    AlertEvent,
    AlertPriority,
    AlertResult,
    AlertStatus,
    BaseNotifier,
    NotifierConfig,
)
from sentinel_ml.alerting.email import (
    EmailConfig,
    EmailNotifier,
)
from sentinel_ml.alerting.github import (
    GitHubConfig,
    GitHubIssueCreator,
)
from sentinel_ml.alerting.health import (
    HealthCheck,
    HealthConfig,
    HealthStatus,
)
from sentinel_ml.alerting.router import (
    AlertRouter,
    RoutingConfig,
    RoutingRule,
)
from sentinel_ml.alerting.slack import (
    SlackConfig,
    SlackNotifier,
)
from sentinel_ml.alerting.watch import (
    WatchConfig,
    WatchDaemon,
    WatchEvent,
    WatchState,
)
from sentinel_ml.alerting.webhook import (
    WebhookConfig,
    WebhookNotifier,
)

__all__ = [
    "AlertEvent",
    "AlertPriority",
    "AlertResult",
    "AlertRouter",
    "AlertStatus",
    "BaseNotifier",
    "EmailConfig",
    "EmailNotifier",
    "GitHubConfig",
    "GitHubIssueCreator",
    "HealthCheck",
    "HealthConfig",
    "HealthStatus",
    "NotifierConfig",
    "RoutingConfig",
    "RoutingRule",
    "SlackConfig",
    "SlackNotifier",
    "WatchConfig",
    "WatchDaemon",
    "WatchEvent",
    "WatchState",
    "WebhookConfig",
    "WebhookNotifier",
]
