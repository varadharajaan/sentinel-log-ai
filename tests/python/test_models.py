"""Tests for the models module."""

from datetime import datetime

import pytest

from sentinel_ml.models import (
    ClusterSummary,
    ConfidenceLevel,
    Explanation,
    LogRecord,
    NoveltyResult,
    SearchResult,
)


class TestLogRecord:
    """Test cases for LogRecord model."""

    def test_create_minimal(self) -> None:
        """Test creating a LogRecord with minimal fields."""
        record = LogRecord(
            message="Error occurred",
            source="/var/log/app.log",
            raw="2024-01-15 ERROR Error occurred",
        )
        assert record.message == "Error occurred"
        assert record.source == "/var/log/app.log"
        assert record.level is None
        assert record.timestamp is None
        assert record.attrs == {}

    def test_create_full(self) -> None:
        """Test creating a LogRecord with all fields."""
        now = datetime.now()
        record = LogRecord(
            id="log-123",
            message="Connection failed",
            normalized="Connection failed",
            level="ERROR",
            source="/var/log/app.log",
            timestamp=now,
            raw="2024-01-15 ERROR Connection failed",
            attrs={"host": "server-01", "port": 5432},
        )
        assert record.id == "log-123"
        assert record.level == "ERROR"
        assert record.timestamp == now
        assert record.attrs["host"] == "server-01"

    def test_json_serialization(self) -> None:
        """Test JSON serialization of LogRecord."""
        record = LogRecord(
            message="Test",
            source="test",
            raw="test raw",
        )
        json_str = record.model_dump_json()
        assert "Test" in json_str
        assert "test" in json_str


class TestClusterSummary:
    """Test cases for ClusterSummary model."""

    def test_create(self) -> None:
        """Test creating a ClusterSummary."""
        cluster = ClusterSummary(
            cluster_id="cluster-001",
            size=150,
            representative="Connection timeout to database",
            keywords=["connection", "timeout", "database"],
            cohesion=0.85,
            novelty_score=0.2,
        )
        assert cluster.cluster_id == "cluster-001"
        assert cluster.size == 150
        assert len(cluster.keywords) == 3
        assert cluster.cohesion == 0.85


class TestExplanation:
    """Test cases for Explanation model."""

    def test_create(self) -> None:
        """Test creating an Explanation."""
        explanation = Explanation(
            cluster_id="cluster-001",
            root_cause="Database connection pool exhausted",
            next_steps=["Check connection pool size", "Review slow queries"],
            remediation="Increase pool size in config",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.92,
        )
        assert explanation.confidence == ConfidenceLevel.HIGH
        assert explanation.confidence_score == 0.92
        assert len(explanation.next_steps) == 2

    def test_confidence_levels(self) -> None:
        """Test confidence level enum values."""
        assert ConfidenceLevel.LOW.value == "LOW"
        assert ConfidenceLevel.MEDIUM.value == "MEDIUM"
        assert ConfidenceLevel.HIGH.value == "HIGH"


class TestNoveltyResult:
    """Test cases for NoveltyResult model."""

    def test_novel_result(self) -> None:
        """Test a novel detection result."""
        result = NoveltyResult(
            is_novel=True,
            novelty_score=0.95,
            closest_cluster_id=None,
            reason="No similar patterns found in historical data",
        )
        assert result.is_novel is True
        assert result.novelty_score == 0.95

    def test_known_pattern_result(self) -> None:
        """Test a known pattern detection result."""
        result = NoveltyResult(
            is_novel=False,
            novelty_score=0.15,
            closest_cluster_id="cluster-005",
            distance_to_cluster=0.08,
            reason="Matches known pattern: connection timeout",
        )
        assert result.is_novel is False
        assert result.closest_cluster_id == "cluster-005"
