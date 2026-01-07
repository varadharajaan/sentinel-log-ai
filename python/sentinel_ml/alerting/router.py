"""
Alert routing for multi-channel notification delivery.

Routes alerts to appropriate notifiers based on configurable rules
including priority, tags, and source filters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

from sentinel_ml.alerting.base import (
    AlertEvent,
    AlertPriority,
    AlertResult,
    AlertStatus,
    BaseNotifier,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


class MatchOperator(str, Enum):
    """Match operators for routing rules."""

    EQUALS = "equals"
    CONTAINS = "contains"
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"
    GTE = "gte"
    LTE = "lte"


@dataclass
class RoutingRule:
    """
    Routing rule for alert distribution.

    Attributes:
        name: Rule name for identification.
        notifiers: Notifier names to route to.
        priority: Match specific priorities.
        min_priority: Minimum priority level.
        tags: Match events with these tags.
        source_pattern: Regex pattern for source matching.
        metadata_filters: Key-value filters on metadata.
        enabled: Whether rule is active.
        stop_on_match: Stop processing after this rule matches.
    """

    name: str
    notifiers: list[str]
    priority: list[AlertPriority] | None = None
    min_priority: AlertPriority | None = None
    tags: list[str] | None = None
    source_pattern: str | None = None
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    stop_on_match: bool = False

    def matches(self, event: AlertEvent) -> bool:
        """
        Check if rule matches an event.

        Args:
            event: Alert event to check.

        Returns:
            True if event matches rule criteria.
        """
        if not self.enabled:
            return False

        # Check priority filter
        if self.priority is not None and event.priority not in self.priority:
            return False

        # Check minimum priority
        if self.min_priority is not None:
            priority_order = [
                AlertPriority.INFO,
                AlertPriority.LOW,
                AlertPriority.MEDIUM,
                AlertPriority.HIGH,
                AlertPriority.CRITICAL,
            ]
            event_idx = priority_order.index(event.priority)
            min_idx = priority_order.index(self.min_priority)
            if event_idx < min_idx:
                return False

        # Check tags filter
        if self.tags is not None and not any(tag in event.tags for tag in self.tags):
            return False

        # Check source pattern
        if self.source_pattern is not None and not re.search(self.source_pattern, event.source):
            return False

        # Check metadata filters
        for key, expected in self.metadata_filters.items():
            actual = event.metadata.get(key)
            if not self._match_value(actual, expected):
                return False

        return True

    def _match_value(self, actual: Any, expected: Any) -> bool:
        """Match a single value against expected criteria."""
        if isinstance(expected, dict):
            operator = expected.get("op", MatchOperator.EQUALS)
            value = expected.get("value")

            if operator == MatchOperator.EQUALS:
                return actual == value
            elif operator == MatchOperator.CONTAINS:
                return value in str(actual)
            elif operator == MatchOperator.REGEX:
                return bool(re.search(value, str(actual)))
            elif operator == MatchOperator.IN:
                return actual in value
            elif operator == MatchOperator.NOT_IN:
                return actual not in value
            elif operator == MatchOperator.GTE:
                return actual >= value
            elif operator == MatchOperator.LTE:
                return actual <= value
            return False

        return actual == expected


@dataclass
class RoutingConfig:
    """
    Configuration for alert router.

    Attributes:
        rules: List of routing rules.
        default_notifiers: Notifiers for unmatched events.
        fallback_enabled: Send to default if no rules match.
        parallel_delivery: Send to all matching notifiers in parallel.
    """

    rules: list[RoutingRule] = field(default_factory=list)
    default_notifiers: list[str] = field(default_factory=list)
    fallback_enabled: bool = True
    parallel_delivery: bool = False


class AlertRouter:
    """
    Alert router for multi-channel delivery.

    Routes alerts to appropriate notifiers based on configurable
    rules and priority levels. Implements the Chain of Responsibility
    pattern for flexible alert distribution.
    """

    def __init__(self, config: RoutingConfig | None = None) -> None:
        """
        Initialize alert router.

        Args:
            config: Routing configuration.
        """
        self._config = config or RoutingConfig()
        self._notifiers: dict[str, BaseNotifier] = {}
        self._logger = logger.bind(component="alert-router")
        self._stats = RouterStats()

    def register_notifier(self, notifier: BaseNotifier) -> None:
        """
        Register a notifier for routing.

        Args:
            notifier: Notifier instance to register.
        """
        self._notifiers[notifier.name] = notifier
        self._logger.debug(
            "notifier_registered",
            name=notifier.name,
        )

    def add_rule(self, rule: RoutingRule) -> None:
        """
        Add a routing rule.

        Args:
            rule: Rule to add.
        """
        self._config.rules.append(rule)
        self._logger.debug(
            "rule_added",
            name=rule.name,
            notifiers=rule.notifiers,
        )

    def route(self, event: AlertEvent) -> list[AlertResult]:
        """
        Route an alert to matching notifiers.

        Args:
            event: Alert event to route.

        Returns:
            List of delivery results.
        """
        self._stats.events_routed += 1

        # Find matching notifiers
        target_notifiers = self._find_targets(event)

        if not target_notifiers:
            self._logger.warning(
                "no_notifiers_matched",
                event_id=event.event_id,
            )
            self._stats.events_unrouted += 1
            return []

        # Deliver to targets
        results = self._deliver(event, target_notifiers)
        return results

    def route_batch(self, events: Sequence[AlertEvent]) -> list[AlertResult]:
        """
        Route multiple alerts.

        Args:
            events: Alert events to route.

        Returns:
            List of all delivery results.
        """
        all_results: list[AlertResult] = []

        for event in events:
            results = self.route(event)
            all_results.extend(results)

        return all_results

    def _find_targets(self, event: AlertEvent) -> list[BaseNotifier]:
        """Find notifiers matching the event."""
        matched_names: set[str] = set()

        for rule in self._config.rules:
            if rule.matches(event):
                matched_names.update(rule.notifiers)
                self._stats.rules_matched += 1

                if rule.stop_on_match:
                    break

        # Apply fallback if no matches
        if not matched_names and self._config.fallback_enabled:
            matched_names.update(self._config.default_notifiers)

        # Resolve to notifier instances
        targets: list[BaseNotifier] = []
        for name in matched_names:
            notifier = self._notifiers.get(name)
            if notifier and notifier.is_enabled:
                targets.append(notifier)
            else:
                self._logger.warning(
                    "notifier_not_found",
                    name=name,
                )

        return targets

    def _deliver(self, event: AlertEvent, notifiers: list[BaseNotifier]) -> list[AlertResult]:
        """Deliver event to notifiers."""
        results: list[AlertResult] = []

        for notifier in notifiers:
            try:
                result = notifier.send(event)
                results.append(result)

                if result.is_success:
                    self._stats.successful_deliveries += 1
                else:
                    self._stats.failed_deliveries += 1

            except Exception as e:
                self._logger.error(
                    "delivery_error",
                    notifier=notifier.name,
                    error=str(e),
                )
                results.append(
                    AlertResult(
                        event_id=event.event_id,
                        status=AlertStatus.FAILED,
                        notifier_name=notifier.name,
                        error=str(e),
                    )
                )
                self._stats.failed_deliveries += 1

        return results

    def get_stats(self) -> dict[str, int]:
        """Get routing statistics."""
        return {
            "events_routed": self._stats.events_routed,
            "events_unrouted": self._stats.events_unrouted,
            "rules_matched": self._stats.rules_matched,
            "successful_deliveries": self._stats.successful_deliveries,
            "failed_deliveries": self._stats.failed_deliveries,
        }

    def get_notifiers(self) -> list[str]:
        """Get list of registered notifier names."""
        return list(self._notifiers.keys())

    def get_rules(self) -> list[RoutingRule]:
        """Get list of routing rules."""
        return list(self._config.rules)


@dataclass
class RouterStats:
    """Statistics for alert router."""

    events_routed: int = 0
    events_unrouted: int = 0
    rules_matched: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
