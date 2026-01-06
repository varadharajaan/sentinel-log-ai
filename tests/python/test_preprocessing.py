"""Comprehensive tests for the preprocessing pipeline."""

from datetime import datetime, timezone

import pytest

from sentinel_ml.models import ClusterSummary, Explanation, LogRecord, NoveltyResult


class TestLogRecordPreprocessing:
    """Test log record creation and preprocessing integration."""

    def test_log_record_with_all_fields(self):
        """Test creating a log record with all fields populated."""
        now = datetime.now(timezone.utc)
        record = LogRecord(
            timestamp=now,
            level="ERROR",
            message="Connection failed to database server",
            source="/var/log/app.log",
            raw="2024-01-15 ERROR Connection failed to database server",
            attrs={"host": "db-01", "port": 5432, "retries": 3},
            normalized="Connection failed to database server",
        )

        assert record.timestamp == now
        assert record.level == "ERROR"
        assert record.message == "Connection failed to database server"
        assert record.source == "/var/log/app.log"
        assert record.attrs["host"] == "db-01"
        assert record.attrs["port"] == 5432

    def test_log_record_minimal(self):
        """Test creating a minimal log record."""
        record = LogRecord(
            message="test",
            source="stdin",
            raw="test",
        )

        assert record.message == "test"
        assert record.source == "stdin"
        assert record.timestamp is None
        assert record.level is None
        assert record.attrs == {}  # default_factory=dict

    def test_log_record_json_serialization(self):
        """Test JSON serialization round-trip."""
        record = LogRecord(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            level="WARN",
            message="Disk space low",
            source="/var/log/system.log",
            raw="WARN Disk space low",
            attrs={"disk": "/dev/sda1", "percent_used": 95.5},
        )

        # Serialize
        json_str = record.model_dump_json()
        assert "Disk space low" in json_str
        assert "WARN" in json_str

        # Deserialize
        restored = LogRecord.model_validate_json(json_str)
        assert restored.message == record.message
        assert restored.level == record.level
        assert restored.attrs["percent_used"] == 95.5

    def test_log_record_dict_conversion(self):
        """Test dict conversion."""
        record = LogRecord(
            message="test message",
            source="test.log",
            raw="raw line",
            attrs={"key": "value"},
        )

        d = record.model_dump()
        assert isinstance(d, dict)
        assert d["message"] == "test message"
        assert d["attrs"]["key"] == "value"

    def test_log_record_equality(self):
        """Test record equality comparison."""
        r1 = LogRecord(message="test", source="a.log", raw="test")
        r2 = LogRecord(message="test", source="a.log", raw="test")
        r3 = LogRecord(message="different", source="a.log", raw="different")

        assert r1 == r2
        assert r1 != r3


class TestClusterSummaryPreprocessing:
    """Test cluster summary creation."""

    def test_cluster_summary_creation(self):
        """Test creating a cluster summary."""
        now = datetime.now(timezone.utc)
        summary = ClusterSummary(
            cluster_id="cluster-001",
            size=150,
            representative="Connection timeout to database",
            keywords=["connection", "timeout", "database"],
            novelty_score=0.15,
            first_seen=now,
            last_seen=now,
        )

        assert summary.cluster_id == "cluster-001"
        assert summary.size == 150
        assert len(summary.keywords) == 3
        assert summary.novelty_score == 0.15

    def test_cluster_summary_validation(self):
        """Test cluster summary validation."""
        now = datetime.now(timezone.utc)

        # Valid novelty score
        summary = ClusterSummary(
            cluster_id="test",
            size=10,
            representative="test",
            novelty_score=0.5,
            first_seen=now,
            last_seen=now,
        )
        assert 0.0 <= summary.novelty_score <= 1.0

    def test_cluster_summary_json_round_trip(self):
        """Test JSON serialization."""
        now = datetime.now(timezone.utc)
        summary = ClusterSummary(
            cluster_id="cluster-002",
            size=50,
            representative="Memory allocation failed",
            keywords=["memory", "allocation", "failed"],
            novelty_score=0.8,
            first_seen=now,
            last_seen=now,
        )

        json_str = summary.model_dump_json()
        restored = ClusterSummary.model_validate_json(json_str)

        assert restored.cluster_id == summary.cluster_id
        assert restored.size == summary.size
        assert restored.novelty_score == summary.novelty_score


class TestExplanationModel:
    """Test explanation model."""

    def test_explanation_creation(self):
        """Test creating an explanation."""
        now = datetime.now(timezone.utc)
        explanation = Explanation(
            cluster_id="cluster-001",
            root_cause="Database connection pool exhausted",
            next_steps=["Check connection pool size", "Review slow queries"],
            remediation="Increase pool size in config.yaml",
            confidence="HIGH",
            confidence_score=0.92,
            reasoning="Strong correlation with historical patterns",
            generated_at=now,
        )

        assert explanation.cluster_id == "cluster-001"
        assert explanation.confidence == "HIGH"
        assert explanation.confidence_score == 0.92
        assert len(explanation.next_steps) == 2

    def test_explanation_without_remediation(self):
        """Test explanation without remediation."""
        now = datetime.now(timezone.utc)
        explanation = Explanation(
            cluster_id="cluster-002",
            root_cause="Unknown network issue",
            next_steps=["Check network logs", "Verify DNS"],
            confidence="LOW",
            confidence_score=0.35,
            generated_at=now,
        )

        assert explanation.remediation is None
        assert explanation.confidence_reasoning is None  # correct field name

    def test_explanation_json_round_trip(self):
        """Test JSON serialization."""
        now = datetime.now(timezone.utc)
        explanation = Explanation(
            cluster_id="cluster-003",
            root_cause="Memory leak in worker process",
            next_steps=["Analyze heap dumps", "Check for circular refs"],
            remediation="Restart workers periodically",
            confidence="MEDIUM",
            confidence_score=0.65,
            reasoning="Memory growth pattern detected",
            generated_at=now,
        )

        json_str = explanation.model_dump_json()
        restored = Explanation.model_validate_json(json_str)

        assert restored.cluster_id == explanation.cluster_id
        assert restored.root_cause == explanation.root_cause
        assert restored.confidence_score == explanation.confidence_score


class TestNoveltyResult:
    """Test novelty result model."""

    def test_novelty_result_creation(self):
        """Test creating a novelty result."""
        result = NoveltyResult(
            is_novel=True,
            novelty_score=0.87,
            closest_cluster_id="cluster-005",
            reason="Pattern not seen in last 24 hours",
        )

        assert result.is_novel is True
        assert result.novelty_score == 0.87
        assert result.closest_cluster_id == "cluster-005"

    def test_novelty_result_not_novel(self):
        """Test novelty result for known pattern."""
        result = NoveltyResult(
            is_novel=False,
            novelty_score=0.12,
            closest_cluster_id="cluster-001",
        )

        assert result.is_novel is False
        assert result.novelty_score < 0.5
        assert result.reason is None

    def test_novelty_result_json_round_trip(self):
        """Test JSON serialization."""
        result = NoveltyResult(
            is_novel=True,
            novelty_score=0.95,
            closest_cluster_id="cluster-010",
            reason="Completely new error pattern",
        )

        json_str = result.model_dump_json()
        restored = NoveltyResult.model_validate_json(json_str)

        assert restored.is_novel == result.is_novel
        assert restored.novelty_score == result.novelty_score


class TestPreprocessingEdgeCases:
    """Test edge cases in preprocessing."""

    def test_empty_message(self):
        """Test handling empty message."""
        record = LogRecord(message="", source="test.log", raw="")
        assert record.message == ""

    def test_unicode_message(self):
        """Test handling unicode in message."""
        record = LogRecord(
            message="Error: 日本語テスト émoji 🔥 message",
            source="test.log",
            raw="Error: 日本語テスト émoji 🔥 message",
        )
        assert "日本語" in record.message
        assert "🔥" in record.message

    def test_very_long_message(self):
        """Test handling very long message."""
        long_msg = "A" * 100000
        record = LogRecord(
            message=long_msg,
            source="test.log",
            raw=long_msg,
        )
        assert len(record.message) == 100000

    def test_special_characters_in_attrs(self):
        """Test special characters in attrs."""
        record = LogRecord(
            message="test",
            source="test.log",
            raw="test",
            attrs={
                "path": "/var/log/app\\data.log",
                "query": "SELECT * FROM users WHERE name='O'Brien'",
                "json_str": '{"key": "value"}',
            },
        )

        assert "\\" in record.attrs["path"]
        assert "O'Brien" in record.attrs["query"]

    def test_nested_attrs(self):
        """Test nested attrs structure."""
        record = LogRecord(
            message="test",
            source="test.log",
            raw="test",
            attrs={
                "metadata": {
                    "host": {
                        "name": "server-01",
                        "ip": "10.0.0.1",
                    },
                    "tags": ["production", "critical"],
                },
                "count": 42,
            },
        )

        assert record.attrs["metadata"]["host"]["name"] == "server-01"
        assert record.attrs["count"] == 42

    def test_timestamp_naive(self):
        """Test handling naive datetime (no timezone)."""
        naive_dt = datetime(2024, 1, 15, 10, 30, 0)
        record = LogRecord(
            timestamp=naive_dt,
            message="test",
            source="test.log",
            raw="test",
        )
        # Should accept naive datetime
        assert record.timestamp == naive_dt

    def test_timestamp_aware(self):
        """Test handling aware datetime (with timezone)."""
        aware_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        record = LogRecord(
            timestamp=aware_dt,
            message="test",
            source="test.log",
            raw="test",
        )
        assert record.timestamp.tzinfo is not None


class TestModelValidation:
    """Test model validation behavior."""

    def test_log_record_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(TypeError):  # Missing required positional arguments
            LogRecord()  # Missing required fields

    def test_log_record_type_coercion(self):
        """Test type coercion in attrs."""
        record = LogRecord(
            message="test",
            source="test.log",
            raw="test",
            attrs={"count": "42"},  # String instead of int
        )
        # Should preserve the string (no automatic coercion in attrs)
        assert record.attrs["count"] == "42"

    def test_cluster_summary_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(TypeError):
            ClusterSummary()  # Missing required fields

    def test_explanation_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(TypeError):
            Explanation()  # Missing required fields


class TestBatchProcessing:
    """Test batch processing scenarios."""

    def test_batch_log_records(self):
        """Test creating multiple log records."""
        records = [
            LogRecord(
                message=f"Log message {i}",
                source="/var/log/app.log",
                raw=f"Log message {i}",
                attrs={"line_num": i},
            )
            for i in range(100)
        ]

        assert len(records) == 100
        assert records[50].attrs["line_num"] == 50

    def test_batch_serialization(self):
        """Test serializing a batch of records."""
        records = [
            LogRecord(
                message=f"Message {i}",
                source="test.log",
                raw=f"Message {i}",
            )
            for i in range(10)
        ]

        # Serialize all
        json_strings = [r.model_dump_json() for r in records]
        assert len(json_strings) == 10

        # Deserialize all
        restored = [LogRecord.model_validate_json(j) for j in json_strings]
        assert len(restored) == 10
        assert restored[5].message == "Message 5"
