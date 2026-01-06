"""Tests for the preprocessing service module."""

from datetime import datetime

import pytest

from sentinel_ml.models import LogRecord
from sentinel_ml.preprocessing import (
    FilterStage,
    IDAssignmentStage,
    NormalizationStage,
    ParsingStage,
    PreprocessingPipeline,
    PreprocessingService,
    ProcessingStage,
    ProcessingStats,
    TimestampStage,
)


class TestProcessingStats:
    """Test ProcessingStats dataclass."""

    def test_initial_stats(self) -> None:
        """Test initial stats values."""
        stats = ProcessingStats()

        assert stats.records_received == 0
        assert stats.records_parsed == 0
        assert stats.records_normalized == 0
        assert stats.records_output == 0
        assert stats.parse_errors == 0
        assert stats.normalize_errors == 0
        assert stats.end_time is None

    def test_duration_calculation(self) -> None:
        """Test duration calculation."""
        stats = ProcessingStats()

        # Duration should be positive
        assert stats.duration_seconds >= 0

        # With explicit end time
        stats.end_time = stats.start_time + 10.0
        assert stats.duration_seconds == 10.0

    def test_records_per_second(self) -> None:
        """Test records per second calculation."""
        stats = ProcessingStats()
        stats.records_output = 100
        stats.end_time = stats.start_time + 10.0

        assert stats.records_per_second == 10.0

    def test_records_per_second_zero_duration(self) -> None:
        """Test records per second with zero duration."""
        stats = ProcessingStats()
        stats.records_output = 100
        stats.end_time = stats.start_time  # Zero duration

        assert stats.records_per_second == 0.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        stats = ProcessingStats()
        stats.records_received = 100
        stats.records_output = 95

        d = stats.to_dict()

        assert isinstance(d, dict)
        assert d["records_received"] == 100
        assert d["records_output"] == 95
        assert "duration_seconds" in d
        assert "records_per_second" in d


class TestIDAssignmentStage:
    """Test ID assignment stage."""

    def test_assigns_id_when_missing(self) -> None:
        """Test that stage assigns ID when missing."""
        stage = IDAssignmentStage()
        record = LogRecord(message="test", source="test.log", raw="test")

        result = stage.process(record)

        assert result is not None
        assert result.id is not None
        assert len(result.id) == 36  # UUID format

    def test_preserves_existing_id(self) -> None:
        """Test that stage preserves existing ID."""
        stage = IDAssignmentStage()
        record = LogRecord(
            id="existing-id",
            message="test",
            source="test.log",
            raw="test",
        )

        result = stage.process(record)

        assert result is not None
        assert result.id == "existing-id"

    def test_stage_name(self) -> None:
        """Test stage name."""
        stage = IDAssignmentStage()
        assert stage.name == "id_assignment"


class TestTimestampStage:
    """Test timestamp stage."""

    def test_assigns_timestamp_when_missing(self) -> None:
        """Test that stage assigns timestamp when missing."""
        stage = TimestampStage()
        record = LogRecord(message="test", source="test.log", raw="test")

        result = stage.process(record)

        assert result is not None
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    def test_preserves_existing_timestamp(self) -> None:
        """Test that stage preserves existing timestamp."""
        stage = TimestampStage()
        original_ts = datetime(2024, 1, 15, 10, 30, 0)
        record = LogRecord(
            timestamp=original_ts,
            message="test",
            source="test.log",
            raw="test",
        )

        result = stage.process(record)

        assert result is not None
        assert result.timestamp == original_ts

    def test_stage_name(self) -> None:
        """Test stage name."""
        stage = TimestampStage()
        assert stage.name == "timestamp"


class TestParsingStage:
    """Test parsing stage."""

    def test_parses_json_log(self) -> None:
        """Test parsing JSON log line."""
        stage = ParsingStage()
        raw = '{"message": "test message", "level": "ERROR"}'
        record = LogRecord(
            message=raw,
            source="test.log",
            raw=raw,
        )

        result = stage.process(record)

        assert result is not None
        assert result.message == "test message"
        assert result.level == "ERROR"

    def test_parses_syslog(self) -> None:
        """Test parsing syslog line."""
        stage = ParsingStage()
        raw = "Jan 15 10:30:00 hostname program[1234]: test message"
        record = LogRecord(
            message=raw,
            source="syslog",
            raw=raw,
        )

        result = stage.process(record)

        assert result is not None
        assert "test message" in result.message

    def test_handles_empty_raw(self) -> None:
        """Test handling empty raw line."""
        stage = ParsingStage()
        record = LogRecord(message="", source="test.log", raw="")

        result = stage.process(record)

        assert result is not None

    def test_preserves_existing_message(self) -> None:
        """Test that stage preserves existing message if different from raw."""
        stage = ParsingStage()
        record = LogRecord(
            message="existing message",
            source="test.log",
            raw="raw line",
        )

        result = stage.process(record)

        assert result is not None
        assert result.message == "existing message"

    def test_stage_name(self) -> None:
        """Test stage name."""
        stage = ParsingStage()
        assert stage.name == "parsing"


class TestNormalizationStage:
    """Test normalization stage."""

    def test_normalizes_message(self) -> None:
        """Test message normalization."""
        stage = NormalizationStage()
        record = LogRecord(
            message="Connection from 192.168.1.100 failed",
            source="test.log",
            raw="Connection from 192.168.1.100 failed",
        )

        result = stage.process(record)

        assert result is not None
        assert result.normalized is not None
        # IP should be masked
        assert "192.168.1.100" not in result.normalized

    def test_handles_empty_message(self) -> None:
        """Test handling empty message."""
        stage = NormalizationStage()
        record = LogRecord(message="", source="test.log", raw="")

        result = stage.process(record)

        assert result is not None

    def test_stage_name(self) -> None:
        """Test stage name."""
        stage = NormalizationStage()
        assert stage.name == "normalization"


class TestFilterStage:
    """Test filter stage."""

    def test_passes_matching_records(self) -> None:
        """Test that matching records pass through."""
        stage = FilterStage(lambda r: r.level == "ERROR")
        record = LogRecord(
            message="test",
            source="test.log",
            raw="test",
            level="ERROR",
        )

        result = stage.process(record)

        assert result is not None

    def test_drops_non_matching_records(self) -> None:
        """Test that non-matching records are dropped."""
        stage = FilterStage(lambda r: r.level == "ERROR")
        record = LogRecord(
            message="test",
            source="test.log",
            raw="test",
            level="INFO",
        )

        result = stage.process(record)

        assert result is None

    def test_custom_stage_name(self) -> None:
        """Test custom stage name."""
        stage = FilterStage(lambda _r: True, stage_name="error_filter")
        assert stage.name == "error_filter"

    def test_default_stage_name(self) -> None:
        """Test default stage name."""
        stage = FilterStage(lambda _r: True)
        assert stage.name == "filter"


class TestPreprocessingPipeline:
    """Test preprocessing pipeline."""

    def test_default_pipeline(self) -> None:
        """Test pipeline with default stages."""
        pipeline = PreprocessingPipeline()
        record = LogRecord(message="test message", source="test.log", raw="test message")

        result = pipeline.process_one(record)

        assert result is not None
        assert result.id is not None
        assert result.timestamp is not None
        assert result.normalized is not None

    def test_custom_stages(self) -> None:
        """Test pipeline with custom stages."""
        stages = [IDAssignmentStage()]
        pipeline = PreprocessingPipeline(stages=stages)
        record = LogRecord(message="test", source="test.log", raw="test")

        result = pipeline.process_one(record)

        assert result is not None
        assert result.id is not None
        # No normalization stage, so normalized should be None
        assert result.normalized is None

    def test_process_batch(self) -> None:
        """Test batch processing."""
        pipeline = PreprocessingPipeline()
        records = [
            LogRecord(message=f"message {i}", source="test.log", raw=f"message {i}")
            for i in range(10)
        ]

        results = pipeline.process_batch(records)

        assert len(results) == 10
        assert all(r.id is not None for r in results)

    def test_process_stream(self) -> None:
        """Test stream processing."""
        pipeline = PreprocessingPipeline()
        records = [
            LogRecord(message=f"message {i}", source="test.log", raw=f"message {i}")
            for i in range(5)
        ]

        results = list(pipeline.process_stream(records))

        assert len(results) == 5

    def test_filter_drops_records(self) -> None:
        """Test that filter stage can drop records."""
        stages = [
            IDAssignmentStage(),
            FilterStage(lambda r: "keep" in r.message),
        ]
        pipeline = PreprocessingPipeline(stages=stages)
        records = [
            LogRecord(message="keep this", source="test.log", raw="keep this"),
            LogRecord(message="drop this", source="test.log", raw="drop this"),
            LogRecord(message="keep me too", source="test.log", raw="keep me too"),
        ]

        results = pipeline.process_batch(records)

        assert len(results) == 2
        assert all("keep" in r.message for r in results)

    def test_stats_tracking(self) -> None:
        """Test statistics tracking."""
        pipeline = PreprocessingPipeline()
        records = [
            LogRecord(message=f"message {i}", source="test.log", raw=f"message {i}")
            for i in range(5)
        ]

        pipeline.process_batch(records)

        stats = pipeline.get_stats()
        assert stats.records_received == 5
        assert stats.records_output == 5

    def test_reset_stats(self) -> None:
        """Test statistics reset."""
        pipeline = PreprocessingPipeline()
        record = LogRecord(message="test", source="test.log", raw="test")
        pipeline.process_one(record)

        pipeline.reset_stats()

        stats = pipeline.get_stats()
        assert stats.records_received == 0

    def test_add_stage(self) -> None:
        """Test adding a stage to the pipeline."""
        pipeline = PreprocessingPipeline(stages=[])
        pipeline.add_stage(IDAssignmentStage())

        record = LogRecord(message="test", source="test.log", raw="test")
        result = pipeline.process_one(record)

        assert result is not None
        assert result.id is not None

    def test_add_stage_at_position(self) -> None:
        """Test adding a stage at specific position."""
        pipeline = PreprocessingPipeline(stages=[IDAssignmentStage()])
        pipeline.add_stage(TimestampStage(), position=0)

        # Timestamp stage should now be first
        record = LogRecord(message="test", source="test.log", raw="test")
        result = pipeline.process_one(record)

        assert result is not None
        assert result.timestamp is not None


class TestPreprocessingService:
    """Test preprocessing service."""

    def test_preprocess_single_record(self) -> None:
        """Test preprocessing a single record."""
        service = PreprocessingService()
        record = LogRecord(message="test message", source="test.log", raw="test message")

        result = service.preprocess(record)

        assert result is not None
        assert result.id is not None
        assert result.normalized is not None

    def test_preprocess_batch(self) -> None:
        """Test preprocessing a batch of records."""
        service = PreprocessingService()
        records = [
            LogRecord(message=f"message {i}", source="test.log", raw=f"message {i}")
            for i in range(5)
        ]

        results = service.preprocess_batch(records)

        assert len(results) == 5
        assert all(r.id is not None for r in results)

    def test_create_record_from_raw(self) -> None:
        """Test creating a record from raw log line."""
        service = PreprocessingService()
        record = service.create_record_from_raw(
            raw="2024-01-15 ERROR Connection failed",
            source="/var/log/app.log",
            attrs={"line_num": 42},
        )

        assert record.raw == "2024-01-15 ERROR Connection failed"
        assert record.source == "/var/log/app.log"
        assert record.attrs["line_num"] == 42

    def test_get_stats(self) -> None:
        """Test getting statistics."""
        service = PreprocessingService()
        record = LogRecord(message="test", source="test.log", raw="test")
        service.preprocess(record)

        stats = service.get_stats()

        assert isinstance(stats, dict)
        assert "records_received" in stats
        assert stats["records_received"] >= 1

    def test_reset_stats(self) -> None:
        """Test resetting statistics."""
        service = PreprocessingService()
        record = LogRecord(message="test", source="test.log", raw="test")
        service.preprocess(record)

        service.reset_stats()

        stats = service.get_stats()
        assert stats["records_received"] == 0

    def test_preprocess_json_log(self) -> None:
        """Test preprocessing a JSON log line."""
        service = PreprocessingService()
        raw = '{"timestamp": "2024-01-15T10:30:00Z", "level": "ERROR", "message": "Connection failed"}'
        record = service.create_record_from_raw(raw=raw, source="app.jsonl")

        result = service.preprocess(record)

        assert result is not None
        assert result.message == "Connection failed"
        assert result.level == "ERROR"

    def test_preprocess_with_normalization(self) -> None:
        """Test that normalization masks sensitive data."""
        service = PreprocessingService()
        raw = "User 192.168.1.100 logged in with email user@example.com"
        record = service.create_record_from_raw(raw=raw, source="auth.log")

        result = service.preprocess(record)

        assert result is not None
        assert result.normalized is not None
        # Should mask IP and email
        assert "192.168.1.100" not in result.normalized
        assert "user@example.com" not in result.normalized


class TestProcessingStageInterface:
    """Test that all stages implement the interface correctly."""

    @pytest.mark.parametrize(
        "stage_class",
        [
            IDAssignmentStage,
            TimestampStage,
            ParsingStage,
            NormalizationStage,
        ],
    )
    def test_stage_has_name(self, stage_class: type[ProcessingStage]) -> None:
        """Test that all stages have a name property."""
        stage = stage_class()
        assert isinstance(stage.name, str)
        assert len(stage.name) > 0

    @pytest.mark.parametrize(
        "stage_class",
        [
            IDAssignmentStage,
            TimestampStage,
            ParsingStage,
            NormalizationStage,
        ],
    )
    def test_stage_processes_record(self, stage_class: type[ProcessingStage]) -> None:
        """Test that all stages can process a record."""
        stage = stage_class()
        record = LogRecord(message="test", source="test.log", raw="test")

        result = stage.process(record)

        # Result should be either a LogRecord or None
        assert result is None or isinstance(result, LogRecord)
