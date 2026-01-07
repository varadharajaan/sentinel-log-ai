"""
Comprehensive unit tests for the alerting module.

Tests all notifier implementations, watch daemon, health check,
and alert routing functionality.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from sentinel_ml.alerting.base import (
    AlertEvent,
    AlertPriority,
    AlertResult,
    AlertStatus,
    BaseNotifier,
    NotifierConfig,
    NotifierFactory,
)
from sentinel_ml.alerting.email import EmailConfig, EmailNotifier
from sentinel_ml.alerting.github import GitHubConfig, GitHubIssueCreator
from sentinel_ml.alerting.health import HealthCheck, HealthConfig
from sentinel_ml.alerting.router import (
    AlertRouter,
    RoutingConfig,
    RoutingRule,
)
from sentinel_ml.alerting.slack import SlackConfig, SlackNotifier
from sentinel_ml.alerting.watch import WatchConfig, WatchDaemon, WatchEvent, WatchState
from sentinel_ml.alerting.webhook import WebhookConfig, WebhookNotifier

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_event() -> AlertEvent:
    """Create a sample alert event."""
    return AlertEvent(
        title="Test Alert",
        message="This is a test alert message",
        priority=AlertPriority.HIGH,
        source="test-source",
        metadata={"key": "value"},
        tags=["test", "unit-test"],
    )


@pytest.fixture
def low_priority_event() -> AlertEvent:
    """Create a low priority event."""
    return AlertEvent(
        title="Low Priority Alert",
        message="This is a low priority message",
        priority=AlertPriority.LOW,
        source="test-source",
    )


@pytest.fixture
def critical_event() -> AlertEvent:
    """Create a critical priority event."""
    return AlertEvent(
        title="Critical Alert",
        message="This is a critical message",
        priority=AlertPriority.CRITICAL,
        source="production",
        tags=["urgent"],
    )


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Create a temporary log directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


# =============================================================================
# AlertPriority Tests
# =============================================================================


class TestAlertPriority:
    """Tests for AlertPriority enum."""

    def test_priority_values(self) -> None:
        """Test priority value strings."""
        assert AlertPriority.CRITICAL.value == "critical"
        assert AlertPriority.HIGH.value == "high"
        assert AlertPriority.MEDIUM.value == "medium"
        assert AlertPriority.LOW.value == "low"
        assert AlertPriority.INFO.value == "info"

    def test_from_score_critical(self) -> None:
        """Test critical priority from high score."""
        assert AlertPriority.from_score(0.95) == AlertPriority.CRITICAL
        assert AlertPriority.from_score(1.0) == AlertPriority.CRITICAL
        assert AlertPriority.from_score(0.9) == AlertPriority.CRITICAL

    def test_from_score_high(self) -> None:
        """Test high priority from medium-high score."""
        assert AlertPriority.from_score(0.7) == AlertPriority.HIGH
        assert AlertPriority.from_score(0.85) == AlertPriority.HIGH

    def test_from_score_medium(self) -> None:
        """Test medium priority from medium score."""
        assert AlertPriority.from_score(0.5) == AlertPriority.MEDIUM
        assert AlertPriority.from_score(0.65) == AlertPriority.MEDIUM

    def test_from_score_low(self) -> None:
        """Test low priority from low score."""
        assert AlertPriority.from_score(0.3) == AlertPriority.LOW
        assert AlertPriority.from_score(0.45) == AlertPriority.LOW

    def test_from_score_info(self) -> None:
        """Test info priority from very low score."""
        assert AlertPriority.from_score(0.0) == AlertPriority.INFO
        assert AlertPriority.from_score(0.2) == AlertPriority.INFO


# =============================================================================
# AlertEvent Tests
# =============================================================================


class TestAlertEvent:
    """Tests for AlertEvent dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        event = AlertEvent(title="Test", message="Message")
        assert event.priority == AlertPriority.MEDIUM
        assert event.source == "sentinel-ml"
        assert event.event_id is not None
        assert event.timestamp is not None
        assert event.metadata == {}
        assert event.tags == []

    def test_custom_values(self, sample_event: AlertEvent) -> None:
        """Test custom field values."""
        assert sample_event.title == "Test Alert"
        assert sample_event.priority == AlertPriority.HIGH
        assert sample_event.source == "test-source"
        assert "key" in sample_event.metadata
        assert "test" in sample_event.tags

    def test_to_dict(self, sample_event: AlertEvent) -> None:
        """Test serialization to dictionary."""
        data = sample_event.to_dict()
        assert data["title"] == "Test Alert"
        assert data["priority"] == "high"
        assert "event_id" in data
        assert "timestamp" in data

    def test_unique_event_ids(self) -> None:
        """Test that each event gets a unique ID."""
        events = [AlertEvent(title="Test", message="Msg") for _ in range(10)]
        ids = [e.event_id for e in events]
        assert len(set(ids)) == 10


# =============================================================================
# AlertResult Tests
# =============================================================================


class TestAlertResult:
    """Tests for AlertResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result."""
        result = AlertResult(
            event_id="test-123",
            status=AlertStatus.SENT,
            notifier_name="test-notifier",
            delivered_at=datetime.now(tz=timezone.utc),
        )
        assert result.is_success is True

    def test_failed_result(self) -> None:
        """Test failed result."""
        result = AlertResult(
            event_id="test-123",
            status=AlertStatus.FAILED,
            notifier_name="test-notifier",
            error="Connection failed",
        )
        assert result.is_success is False
        assert result.error == "Connection failed"

    def test_skipped_result(self) -> None:
        """Test skipped result."""
        result = AlertResult(
            event_id="test-123",
            status=AlertStatus.SKIPPED,
            notifier_name="disabled-notifier",
        )
        assert result.is_success is False


# =============================================================================
# NotifierConfig Tests
# =============================================================================


class TestNotifierConfig:
    """Tests for NotifierConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = NotifierConfig()
        assert config.name == "base-notifier"
        assert config.enabled is True
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.timeout_seconds == 30.0
        assert config.batch_size == 10
        assert config.rate_limit_per_minute == 60

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = NotifierConfig(
            name="custom",
            enabled=False,
            max_retries=5,
        )
        assert config.name == "custom"
        assert config.enabled is False
        assert config.max_retries == 5


# =============================================================================
# BaseNotifier Tests
# =============================================================================


class ConcreteNotifier(BaseNotifier):
    """Concrete implementation for testing."""

    def __init__(self, config: NotifierConfig, should_fail: bool = False) -> None:
        super().__init__(config)
        self._should_fail = should_fail
        self.send_calls: list[AlertEvent] = []

    def _send(self, event: AlertEvent) -> dict[str, Any]:
        self.send_calls.append(event)
        if self._should_fail:
            raise RuntimeError("Simulated failure")
        return {"status": "ok"}


class TestBaseNotifier:
    """Tests for BaseNotifier abstract class."""

    def test_send_success(self, sample_event: AlertEvent) -> None:
        """Test successful send."""
        config = NotifierConfig(name="test")
        notifier = ConcreteNotifier(config)

        result = notifier.send(sample_event)

        assert result.is_success
        assert result.status == AlertStatus.SENT
        assert len(notifier.send_calls) == 1

    def test_send_disabled(self, sample_event: AlertEvent) -> None:
        """Test send with disabled notifier."""
        config = NotifierConfig(name="test", enabled=False)
        notifier = ConcreteNotifier(config)

        result = notifier.send(sample_event)

        assert result.status == AlertStatus.SKIPPED
        assert len(notifier.send_calls) == 0

    def test_send_retry_on_failure(self, sample_event: AlertEvent) -> None:
        """Test retry logic on failure."""
        config = NotifierConfig(name="test", max_retries=3, retry_delay_seconds=0.01)
        notifier = ConcreteNotifier(config, should_fail=True)

        result = notifier.send(sample_event)

        assert result.status == AlertStatus.FAILED
        assert result.attempts == 3
        assert len(notifier.send_calls) == 3

    def test_stats_tracking(self, sample_event: AlertEvent) -> None:
        """Test statistics tracking."""
        config = NotifierConfig(name="test")
        notifier = ConcreteNotifier(config)

        notifier.send(sample_event)
        notifier.send(sample_event)

        stats = notifier.stats
        assert stats["sent"] == 2
        assert stats["failed"] == 0
        assert stats["total"] == 2

    def test_send_batch(self) -> None:
        """Test batch sending."""
        config = NotifierConfig(name="test", batch_size=2)
        notifier = ConcreteNotifier(config)
        events = [AlertEvent(title=f"Event {i}", message="Msg") for i in range(5)]

        results = notifier.send_batch(events)

        assert len(results) == 5
        assert all(r.is_success for r in results)

    def test_validate_config(self) -> None:
        """Test configuration validation."""
        config = NotifierConfig(name="", max_retries=-1, timeout_seconds=0)
        notifier = ConcreteNotifier(config)

        errors = notifier.validate_config()

        assert "Notifier name is required" in errors
        assert "max_retries must be non-negative" in errors
        assert "timeout_seconds must be positive" in errors

    def test_health_check(self) -> None:
        """Test health check."""
        config = NotifierConfig(name="test", enabled=True)
        notifier = ConcreteNotifier(config)
        assert notifier.health_check() is True

        config = NotifierConfig(name="test", enabled=False)
        notifier = ConcreteNotifier(config)
        assert notifier.health_check() is False


# =============================================================================
# NotifierFactory Tests
# =============================================================================


class TestNotifierFactory:
    """Tests for NotifierFactory."""

    def test_register_and_create(self) -> None:
        """Test notifier registration and creation."""
        NotifierFactory.register("concrete", ConcreteNotifier)

        config = NotifierConfig(name="test")
        notifier = NotifierFactory.create("concrete", config)

        assert isinstance(notifier, ConcreteNotifier)
        assert notifier.name == "test"

    def test_create_unknown_type(self) -> None:
        """Test error for unknown type."""
        with pytest.raises(ValueError, match="Unknown notifier type"):
            NotifierFactory.create("nonexistent", NotifierConfig())

    def test_available_types(self) -> None:
        """Test listing available types."""
        NotifierFactory.register("test-type", ConcreteNotifier)
        types = NotifierFactory.available_types()
        assert "test-type" in types


# =============================================================================
# SlackNotifier Tests
# =============================================================================


class TestSlackConfig:
    """Tests for SlackConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = SlackConfig()
        assert config.name == "slack-notifier"
        assert config.webhook_url == ""
        assert config.username == "Sentinel ML"

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = SlackConfig(
            webhook_url="https://hooks.slack.com/test",
            channel="#alerts",
        )
        assert config.webhook_url == "https://hooks.slack.com/test"
        assert config.channel == "#alerts"


class TestSlackNotifier:
    """Tests for SlackNotifier."""

    def test_validate_config_missing_url(self) -> None:
        """Test validation with missing webhook URL."""
        config = SlackConfig(webhook_url="")
        notifier = SlackNotifier(config)

        errors = notifier.validate_config()
        assert "Slack webhook URL is required" in errors

    def test_validate_config_valid(self) -> None:
        """Test validation with valid config."""
        config = SlackConfig(webhook_url="https://hooks.slack.com/services/xxx")
        notifier = SlackNotifier(config)

        errors = notifier.validate_config()
        assert len(errors) == 0

    @patch("sentinel_ml.alerting.slack.urlopen")
    def test_send_success(self, mock_urlopen: Mock, sample_event: AlertEvent) -> None:
        """Test successful Slack send."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"ok"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        notifier = SlackNotifier(config)

        result = notifier.send(sample_event)

        assert result.is_success
        mock_urlopen.assert_called_once()

    def test_build_payload(self, sample_event: AlertEvent) -> None:
        """Test payload building."""
        config = SlackConfig(
            webhook_url="https://hooks.slack.com/test",
            channel="#test",
        )
        notifier = SlackNotifier(config)

        payload = notifier._build_payload(sample_event)

        assert "attachments" in payload
        assert payload["channel"] == "#test"
        assert len(payload["attachments"]) == 1

    def test_critical_mentions(self, critical_event: AlertEvent) -> None:
        """Test mention formatting for critical alerts."""
        config = SlackConfig(
            webhook_url="https://hooks.slack.com/test",
            mention_users=["U123", "U456"],
            mention_groups=["G789"],
        )
        notifier = SlackNotifier(config)

        payload = notifier._build_payload(critical_event)

        title = payload["attachments"][0]["title"]
        assert "<@U123>" in title
        assert "<@U456>" in title
        assert "<!subteam^G789>" in title


# =============================================================================
# EmailNotifier Tests
# =============================================================================


class TestEmailConfig:
    """Tests for EmailConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = EmailConfig()
        assert config.name == "email-notifier"
        assert config.smtp_host == "localhost"
        assert config.smtp_port == 587
        assert config.use_tls is True

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = EmailConfig(
            smtp_host="smtp.example.com",
            from_address="alerts@example.com",
            to_addresses=["admin@example.com"],
        )
        assert config.smtp_host == "smtp.example.com"
        assert config.from_address == "alerts@example.com"


class TestEmailNotifier:
    """Tests for EmailNotifier."""

    def test_validate_config_missing_fields(self) -> None:
        """Test validation with missing required fields."""
        config = EmailConfig()
        notifier = EmailNotifier(config)

        errors = notifier.validate_config()
        assert "From address is required" in errors
        assert "At least one recipient address is required" in errors

    def test_validate_config_ssl_tls_conflict(self) -> None:
        """Test validation with both SSL and TLS."""
        config = EmailConfig(
            from_address="test@example.com",
            to_addresses=["admin@example.com"],
            use_ssl=True,
            use_tls=True,
        )
        notifier = EmailNotifier(config)

        errors = notifier.validate_config()
        assert "Cannot use both SSL and TLS simultaneously" in errors

    def test_build_message(self, sample_event: AlertEvent) -> None:
        """Test message building."""
        config = EmailConfig(
            from_address="alerts@example.com",
            from_name="Sentinel",
            to_addresses=["admin@example.com"],
        )
        notifier = EmailNotifier(config)

        message = notifier._build_message(sample_event)

        assert "[HIGH]" in message["Subject"]
        assert "Test Alert" in message["Subject"]
        assert message["From"] == "Sentinel <alerts@example.com>"

    def test_x_priority_mapping(self) -> None:
        """Test X-Priority header mapping."""
        config = EmailConfig(
            from_address="test@example.com",
            to_addresses=["admin@example.com"],
        )
        notifier = EmailNotifier(config)

        assert notifier._get_x_priority(AlertPriority.CRITICAL) == "1"
        assert notifier._get_x_priority(AlertPriority.HIGH) == "2"
        assert notifier._get_x_priority(AlertPriority.MEDIUM) == "3"
        assert notifier._get_x_priority(AlertPriority.LOW) == "4"
        assert notifier._get_x_priority(AlertPriority.INFO) == "5"


# =============================================================================
# WebhookNotifier Tests
# =============================================================================


class TestWebhookConfig:
    """Tests for WebhookConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = WebhookConfig()
        assert config.name == "webhook-notifier"
        assert config.url == ""
        assert config.method == "POST"
        assert config.auth_type == "none"

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = WebhookConfig(
            url="https://api.example.com/webhook",
            auth_type="bearer",
            auth_token="token123",
        )
        assert config.url == "https://api.example.com/webhook"
        assert config.auth_type == "bearer"


class TestWebhookNotifier:
    """Tests for WebhookNotifier."""

    def test_validate_config_missing_url(self) -> None:
        """Test validation with missing URL."""
        config = WebhookConfig()
        notifier = WebhookNotifier(config)

        errors = notifier.validate_config()
        assert "Webhook URL is required" in errors

    def test_validate_config_invalid_method(self) -> None:
        """Test validation with invalid HTTP method."""
        config = WebhookConfig(url="https://example.com", method="GET")
        notifier = WebhookNotifier(config)

        errors = notifier.validate_config()
        assert "HTTP method must be POST, PUT, or PATCH" in errors

    def test_build_payload_default(self, sample_event: AlertEvent) -> None:
        """Test default payload building."""
        config = WebhookConfig(url="https://example.com")
        notifier = WebhookNotifier(config)

        payload = notifier._build_payload(sample_event)

        assert payload["title"] == "Test Alert"
        assert payload["priority"] == "high"
        assert "timestamp" in payload

    def test_build_payload_with_template(self, sample_event: AlertEvent) -> None:
        """Test payload building with template."""
        config = WebhookConfig(
            url="https://example.com",
            payload_template={
                "alert_title": "$title",
                "level": "$priority",
                "custom": "static_value",
            },
        )
        notifier = WebhookNotifier(config)

        payload = notifier._build_payload(sample_event)

        assert payload["alert_title"] == "Test Alert"
        assert payload["level"] == "high"
        assert payload["custom"] == "static_value"

    def test_build_headers_bearer_auth(self) -> None:
        """Test header building with bearer auth."""
        config = WebhookConfig(
            url="https://example.com",
            auth_type="bearer",
            auth_token="mytoken",
        )
        notifier = WebhookNotifier(config)

        headers = notifier._build_headers()

        assert headers["Authorization"] == "Bearer mytoken"

    def test_build_headers_api_key_auth(self) -> None:
        """Test header building with API key auth."""
        config = WebhookConfig(
            url="https://example.com",
            auth_type="api_key",
            auth_token="apikey123",
            api_key_header="X-Custom-Key",
        )
        notifier = WebhookNotifier(config)

        headers = notifier._build_headers()

        assert headers["X-Custom-Key"] == "apikey123"


# =============================================================================
# GitHubIssueCreator Tests
# =============================================================================


class TestGitHubConfig:
    """Tests for GitHubConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = GitHubConfig()
        assert config.name == "github-notifier"
        assert config.api_base_url == "https://api.github.com"
        assert "sentinel-ml" in config.default_labels

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = GitHubConfig(
            token="ghp_xxx",
            owner="testorg",
            repo="testrepo",
        )
        assert config.token == "ghp_xxx"
        assert config.owner == "testorg"
        assert config.repo == "testrepo"


class TestGitHubIssueCreator:
    """Tests for GitHubIssueCreator."""

    def test_validate_config_missing_fields(self) -> None:
        """Test validation with missing required fields."""
        config = GitHubConfig()
        creator = GitHubIssueCreator(config)

        errors = creator.validate_config()
        assert "GitHub token is required" in errors
        assert "Repository owner is required" in errors
        assert "Repository name is required" in errors

    def test_validate_config_valid(self) -> None:
        """Test validation with valid config."""
        config = GitHubConfig(
            token="ghp_xxx",
            owner="testorg",
            repo="testrepo",
        )
        creator = GitHubIssueCreator(config)

        errors = creator.validate_config()
        assert len(errors) == 0

    def test_build_issue_body(self, sample_event: AlertEvent) -> None:
        """Test issue body building."""
        config = GitHubConfig(
            token="ghp_xxx",
            owner="testorg",
            repo="testrepo",
        )
        creator = GitHubIssueCreator(config)

        body = creator._build_issue_body(sample_event)

        assert "## Alert Details" in body
        assert "HIGH" in body
        assert sample_event.message in body
        assert sample_event.event_id in body

    def test_build_labels(self, sample_event: AlertEvent) -> None:
        """Test label building."""
        config = GitHubConfig(
            token="ghp_xxx",
            owner="testorg",
            repo="testrepo",
            default_labels=["alert"],
        )
        creator = GitHubIssueCreator(config)

        labels = creator._build_labels(sample_event)

        assert "alert" in labels
        assert "priority: high" in labels
        assert "test" in labels


# =============================================================================
# WatchDaemon Tests
# =============================================================================


class TestWatchConfig:
    """Tests for WatchConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = WatchConfig()
        assert config.poll_interval_seconds == 10.0
        assert config.novelty_threshold == 0.5
        assert config.recursive is True

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = WatchConfig(
            watch_paths=[Path("/var/log")],
            poll_interval_seconds=5.0,
            novelty_threshold=0.7,
        )
        assert config.poll_interval_seconds == 5.0
        assert config.novelty_threshold == 0.7


class TestWatchEvent:
    """Tests for WatchEvent."""

    def test_creation(self) -> None:
        """Test event creation."""
        event = WatchEvent(
            file_path=Path("/var/log/test.log"),
            event_type="modified",
            lines_added=10,
        )
        assert event.file_path == Path("/var/log/test.log")
        assert event.event_type == "modified"
        assert event.lines_added == 10


class TestWatchDaemon:
    """Tests for WatchDaemon."""

    def test_initial_state(self) -> None:
        """Test initial daemon state."""
        config = WatchConfig()
        daemon = WatchDaemon(config)
        assert daemon.state == WatchState.STOPPED

    def test_start_stop(self, temp_log_dir: Path) -> None:
        """Test daemon start and stop."""
        config = WatchConfig(
            watch_paths=[temp_log_dir],
            poll_interval_seconds=0.1,
        )
        daemon = WatchDaemon(config)

        daemon.start()
        assert daemon.state == WatchState.RUNNING

        daemon.stop()
        assert daemon.state == WatchState.STOPPED

    def test_start_already_running(self, temp_log_dir: Path) -> None:
        """Test error when starting already running daemon."""
        config = WatchConfig(
            watch_paths=[temp_log_dir],
            poll_interval_seconds=0.1,
        )
        daemon = WatchDaemon(config)

        daemon.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                daemon.start()
        finally:
            daemon.stop()

    def test_add_notifier(self) -> None:
        """Test adding notifier."""
        config = WatchConfig()
        daemon = WatchDaemon(config)

        notifier = ConcreteNotifier(NotifierConfig(name="test"))
        daemon.add_notifier(notifier)

        assert len(daemon._notifiers) == 1

    def test_discover_files(self, temp_log_dir: Path) -> None:
        """Test file discovery."""
        # Create test files
        (temp_log_dir / "app.log").write_text("test")
        (temp_log_dir / "error.jsonl").write_text("test")
        (temp_log_dir / "data.txt").write_text("test")

        config = WatchConfig(
            watch_paths=[temp_log_dir],
            file_patterns=["*.log", "*.jsonl"],
        )
        daemon = WatchDaemon(config)

        files = daemon._discover_files()

        assert len(files) == 2
        names = [f.name for f in files]
        assert "app.log" in names
        assert "error.jsonl" in names

    def test_read_new_lines(self, temp_log_dir: Path) -> None:
        """Test reading new lines from file."""
        log_file = temp_log_dir / "test.log"
        log_file.write_text("line1\nline2\nline3\n")

        config = WatchConfig(watch_paths=[temp_log_dir])
        daemon = WatchDaemon(config)

        # First read
        lines = daemon._read_new_lines(log_file)
        assert len(lines) == 3

        # No new content
        lines = daemon._read_new_lines(log_file)
        assert len(lines) == 0

        # Append new content
        with log_file.open("a") as f:
            f.write("line4\nline5\n")

        lines = daemon._read_new_lines(log_file)
        assert len(lines) == 2
        assert "line4" in lines
        assert "line5" in lines

    def test_get_status(self, temp_log_dir: Path) -> None:
        """Test status reporting."""
        config = WatchConfig(watch_paths=[temp_log_dir])
        daemon = WatchDaemon(config)

        status = daemon.get_status()

        assert status["state"] == "stopped"
        assert "stats" in status
        assert "watched_files" in status


# =============================================================================
# HealthCheck Tests
# =============================================================================


class TestHealthConfig:
    """Tests for HealthConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = HealthConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.path == "/health"

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = HealthConfig(
            host="127.0.0.1",
            port=9090,
        )
        assert config.host == "127.0.0.1"
        assert config.port == 9090


class TestHealthCheck:
    """Tests for HealthCheck."""

    def test_get_health_no_components(self) -> None:
        """Test health with no components configured."""
        config = HealthConfig()
        health = HealthCheck(config)

        result = health.get_health()

        assert result["status"] == "healthy"
        assert "timestamp" in result

    def test_get_health_with_notifiers(self) -> None:
        """Test health with notifiers."""
        config = HealthConfig()
        notifier = ConcreteNotifier(NotifierConfig(name="test"))
        health = HealthCheck(config, notifiers=[notifier])

        result = health.get_health()

        assert "components" in result
        assert "notifiers" in result["components"]

    def test_get_health_with_daemon(self, temp_log_dir: Path) -> None:
        """Test health with watch daemon."""
        health_config = HealthConfig()
        watch_config = WatchConfig(watch_paths=[temp_log_dir])
        daemon = WatchDaemon(watch_config)

        health = HealthCheck(health_config, watch_daemon=daemon)

        result = health.get_health()

        assert "components" in result
        assert "watch_daemon" in result["components"]

    def test_start_stop_server(self) -> None:
        """Test server start and stop."""
        config = HealthConfig(port=18080)  # Use non-standard port
        health = HealthCheck(config)

        health.start()
        time.sleep(0.1)  # Allow server to start

        health.stop()


# =============================================================================
# AlertRouter Tests
# =============================================================================


class TestRoutingRule:
    """Tests for RoutingRule."""

    def test_match_by_priority(self, sample_event: AlertEvent) -> None:
        """Test matching by priority."""
        rule = RoutingRule(
            name="high-priority",
            notifiers=["slack"],
            priority=[AlertPriority.HIGH, AlertPriority.CRITICAL],
        )

        assert rule.matches(sample_event) is True

    def test_match_by_min_priority(
        self, sample_event: AlertEvent, low_priority_event: AlertEvent
    ) -> None:
        """Test matching by minimum priority."""
        rule = RoutingRule(
            name="important",
            notifiers=["email"],
            min_priority=AlertPriority.MEDIUM,
        )

        assert rule.matches(sample_event) is True
        assert rule.matches(low_priority_event) is False

    def test_match_by_tags(self, sample_event: AlertEvent) -> None:
        """Test matching by tags."""
        rule = RoutingRule(
            name="test-events",
            notifiers=["webhook"],
            tags=["test", "staging"],
        )

        assert rule.matches(sample_event) is True

        rule2 = RoutingRule(
            name="production",
            notifiers=["webhook"],
            tags=["production"],
        )

        assert rule2.matches(sample_event) is False

    def test_match_by_source_pattern(self, sample_event: AlertEvent) -> None:
        """Test matching by source pattern."""
        rule = RoutingRule(
            name="test-source",
            notifiers=["slack"],
            source_pattern=r"test-.*",
        )

        assert rule.matches(sample_event) is True

        rule2 = RoutingRule(
            name="prod-source",
            notifiers=["slack"],
            source_pattern=r"prod-.*",
        )

        assert rule2.matches(sample_event) is False

    def test_match_disabled_rule(self, sample_event: AlertEvent) -> None:
        """Test disabled rule never matches."""
        rule = RoutingRule(
            name="disabled",
            notifiers=["slack"],
            enabled=False,
        )

        assert rule.matches(sample_event) is False


class TestAlertRouter:
    """Tests for AlertRouter."""

    def test_register_notifier(self) -> None:
        """Test notifier registration."""
        router = AlertRouter()
        notifier = ConcreteNotifier(NotifierConfig(name="test"))

        router.register_notifier(notifier)

        assert "test" in router.get_notifiers()

    def test_add_rule(self) -> None:
        """Test rule addition."""
        router = AlertRouter()
        rule = RoutingRule(name="test", notifiers=["slack"])

        router.add_rule(rule)

        assert len(router.get_rules()) == 1

    def test_route_with_matching_rule(self, sample_event: AlertEvent) -> None:
        """Test routing with matching rule."""
        router = AlertRouter()
        notifier = ConcreteNotifier(NotifierConfig(name="test-notifier"))
        router.register_notifier(notifier)

        rule = RoutingRule(
            name="high-priority",
            notifiers=["test-notifier"],
            priority=[AlertPriority.HIGH],
        )
        router.add_rule(rule)

        results = router.route(sample_event)

        assert len(results) == 1
        assert results[0].is_success

    def test_route_with_fallback(self, sample_event: AlertEvent) -> None:
        """Test routing with fallback."""
        config = RoutingConfig(
            default_notifiers=["fallback"],
            fallback_enabled=True,
        )
        router = AlertRouter(config)
        notifier = ConcreteNotifier(NotifierConfig(name="fallback"))
        router.register_notifier(notifier)

        results = router.route(sample_event)

        assert len(results) == 1

    def test_route_no_match(self, sample_event: AlertEvent) -> None:
        """Test routing with no matches."""
        config = RoutingConfig(fallback_enabled=False)
        router = AlertRouter(config)

        results = router.route(sample_event)

        assert len(results) == 0

    def test_route_stop_on_match(self, sample_event: AlertEvent) -> None:
        """Test stop_on_match behavior."""
        router = AlertRouter()
        notifier1 = ConcreteNotifier(NotifierConfig(name="first"))
        notifier2 = ConcreteNotifier(NotifierConfig(name="second"))
        router.register_notifier(notifier1)
        router.register_notifier(notifier2)

        rule1 = RoutingRule(
            name="first-rule",
            notifiers=["first"],
            stop_on_match=True,
        )
        rule2 = RoutingRule(
            name="second-rule",
            notifiers=["second"],
        )
        router.add_rule(rule1)
        router.add_rule(rule2)

        results = router.route(sample_event)

        # Only first rule should match due to stop_on_match
        assert len(results) == 1
        assert results[0].notifier_name == "first"

    def test_route_batch(self) -> None:
        """Test batch routing."""
        router = AlertRouter()
        notifier = ConcreteNotifier(NotifierConfig(name="test"))
        router.register_notifier(notifier)

        rule = RoutingRule(name="all", notifiers=["test"])
        router.add_rule(rule)

        events = [AlertEvent(title=f"Event {i}", message="Msg") for i in range(3)]

        results = router.route_batch(events)

        assert len(results) == 3

    def test_get_stats(self, sample_event: AlertEvent) -> None:
        """Test statistics tracking."""
        router = AlertRouter()
        notifier = ConcreteNotifier(NotifierConfig(name="test"))
        router.register_notifier(notifier)
        router.add_rule(RoutingRule(name="all", notifiers=["test"]))

        router.route(sample_event)

        stats = router.get_stats()
        assert stats["events_routed"] == 1
        assert stats["successful_deliveries"] == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestAlertingIntegration:
    """Integration tests for alerting module."""

    def test_end_to_end_routing(self) -> None:
        """Test complete alert routing flow."""
        # Setup router with multiple notifiers
        router = AlertRouter()

        slack_notifier = ConcreteNotifier(NotifierConfig(name="slack"))
        email_notifier = ConcreteNotifier(NotifierConfig(name="email"))
        webhook_notifier = ConcreteNotifier(NotifierConfig(name="webhook"))

        router.register_notifier(slack_notifier)
        router.register_notifier(email_notifier)
        router.register_notifier(webhook_notifier)

        # Add rules
        router.add_rule(
            RoutingRule(
                name="critical-all",
                notifiers=["slack", "email"],
                priority=[AlertPriority.CRITICAL],
            )
        )
        router.add_rule(
            RoutingRule(
                name="high-slack",
                notifiers=["slack"],
                priority=[AlertPriority.HIGH],
            )
        )
        router.add_rule(
            RoutingRule(
                name="webhook-all",
                notifiers=["webhook"],
            )
        )

        # Route events
        critical = AlertEvent(
            title="Critical",
            message="Critical issue",
            priority=AlertPriority.CRITICAL,
        )
        high = AlertEvent(
            title="High",
            message="High issue",
            priority=AlertPriority.HIGH,
        )

        critical_results = router.route(critical)
        high_results = router.route(high)

        # Verify routing
        assert len(critical_results) == 3  # slack, email, webhook
        assert len(high_results) == 2  # slack, webhook

    def test_watch_daemon_with_notifiers(self, temp_log_dir: Path) -> None:
        """Test watch daemon with notifiers."""
        # Create log file
        log_file = temp_log_dir / "test.log"
        log_file.touch()

        def detector(line: str) -> float:
            return 0.8 if "ERROR" in line else 0.2

        notifier = ConcreteNotifier(NotifierConfig(name="test"))
        config = WatchConfig(
            watch_paths=[temp_log_dir],
            poll_interval_seconds=0.1,
            novelty_threshold=0.5,
        )

        daemon = WatchDaemon(config, notifiers=[notifier], novelty_detector=detector)

        # Start and add log entries
        daemon.start()
        try:
            time.sleep(0.15)

            with log_file.open("a") as f:
                f.write("INFO: Normal log\n")
                f.write("ERROR: Something failed\n")

            time.sleep(0.3)
        finally:
            daemon.stop()

        # Verify
        assert daemon.stats.lines_processed >= 2
