"""
Preprocessing service for log ingestion.

This module provides a comprehensive preprocessing pipeline that:
- Parses raw log lines into structured LogRecords
- Normalizes messages for ML processing
- Batches records for efficient embedding

Design Patterns:
- Pipeline Pattern: Sequential processing stages
- Strategy Pattern: Configurable normalization strategies
- Factory Pattern: Parser and normalizer creation

SOLID Principles:
- Single Responsibility: Each stage handles one transformation
- Open/Closed: Extensible via custom processors
- Dependency Inversion: Depends on abstractions
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sentinel_ml.exceptions import PreprocessingError
from sentinel_ml.logging import get_logger
from sentinel_ml.models import LogRecord
from sentinel_ml.normalization import normalize
from sentinel_ml.parser import ParserRegistry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

logger = get_logger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for preprocessing pipeline."""

    records_received: int = 0
    records_parsed: int = 0
    records_normalized: int = 0
    records_output: int = 0
    parse_errors: int = 0
    normalize_errors: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    @property
    def duration_seconds(self) -> float:
        """Get the processing duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def records_per_second(self) -> float:
        """Get the processing rate."""
        duration = self.duration_seconds
        if duration == 0:
            return 0.0
        return self.records_output / duration

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging."""
        return {
            "records_received": self.records_received,
            "records_parsed": self.records_parsed,
            "records_normalized": self.records_normalized,
            "records_output": self.records_output,
            "parse_errors": self.parse_errors,
            "normalize_errors": self.normalize_errors,
            "duration_seconds": round(self.duration_seconds, 3),
            "records_per_second": round(self.records_per_second, 2),
        }


class ProcessingStage(ABC):
    """Abstract base class for processing stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stage name."""
        pass

    @abstractmethod
    def process(self, record: LogRecord) -> LogRecord | None:
        """
        Process a single record.

        Args:
            record: The log record to process.

        Returns:
            The processed record, or None to drop it.
        """
        pass


class ParsingStage(ProcessingStage):
    """Stage that parses raw log lines."""

    def __init__(self) -> None:
        self._parser_registry = ParserRegistry()

    @property
    def name(self) -> str:
        return "parsing"

    def process(self, record: LogRecord) -> LogRecord | None:
        """Parse the raw log line to extract structured fields."""
        if not record.raw:
            return record

        # Only parse if message is not already set
        if record.message and record.message != record.raw:
            return record

        try:
            parsed = self._parser_registry.parse(record.raw, record.source)
            if parsed:
                # Merge parsed fields into existing record
                if parsed.message:
                    record.message = parsed.message
                if parsed.level and not record.level:
                    record.level = parsed.level
                if parsed.timestamp and not record.timestamp:
                    record.timestamp = parsed.timestamp
                if parsed.attrs:
                    record.attrs.update(parsed.attrs)
            return record
        except Exception as e:
            logger.warning("parsing_failed", error=str(e), raw=record.raw[:200])
            return record


class NormalizationStage(ProcessingStage):
    """Stage that normalizes log messages for ML processing."""

    @property
    def name(self) -> str:
        return "normalization"

    def process(self, record: LogRecord) -> LogRecord | None:
        """Normalize the log message."""
        if not record.message:
            return record

        try:
            record.normalized = normalize(record.message)
            return record
        except Exception as e:
            logger.warning(
                "normalization_failed",
                error=str(e),
                message=record.message[:200],
            )
            # Keep original message as normalized
            record.normalized = record.message
            return record


class IDAssignmentStage(ProcessingStage):
    """Stage that assigns unique IDs to records."""

    @property
    def name(self) -> str:
        return "id_assignment"

    def process(self, record: LogRecord) -> LogRecord | None:
        """Assign a unique ID if not present."""
        if not record.id:
            record.id = str(uuid.uuid4())
        return record


class TimestampStage(ProcessingStage):
    """Stage that ensures records have timestamps."""

    @property
    def name(self) -> str:
        return "timestamp"

    def process(self, record: LogRecord) -> LogRecord | None:
        """Set current timestamp if not present."""
        if not record.timestamp:
            record.timestamp = datetime.now()
        return record


class FilterStage(ProcessingStage):
    """Stage that filters records based on a predicate."""

    def __init__(
        self,
        predicate: Callable[[LogRecord], bool],
        stage_name: str = "filter",
    ) -> None:
        self._predicate = predicate
        self._name = stage_name

    @property
    def name(self) -> str:
        return self._name

    def process(self, record: LogRecord) -> LogRecord | None:
        """Return the record if predicate passes, None otherwise."""
        if self._predicate(record):
            return record
        return None


class PreprocessingPipeline:
    """
    Preprocessing pipeline for log records.

    Applies a sequence of processing stages to transform raw log data
    into normalized records ready for ML processing.
    """

    def __init__(self, stages: list[ProcessingStage] | None = None) -> None:
        """
        Initialize the pipeline.

        Args:
            stages: Custom stages to use. If None, uses default stages.
        """
        if stages is None:
            stages = self._default_stages()
        self._stages = stages
        self._stats = ProcessingStats()
        logger.info(
            "preprocessing_pipeline_initialized",
            stages=[s.name for s in self._stages],
        )

    @staticmethod
    def _default_stages() -> list[ProcessingStage]:
        """Return the default processing stages."""
        return [
            IDAssignmentStage(),
            ParsingStage(),
            TimestampStage(),
            NormalizationStage(),
        ]

    def process_one(self, record: LogRecord) -> LogRecord | None:
        """
        Process a single record through the pipeline.

        Args:
            record: The record to process.

        Returns:
            The processed record, or None if dropped.
        """
        self._stats.records_received += 1

        current = record
        for stage in self._stages:
            try:
                result = stage.process(current)
                if result is None:
                    logger.debug("record_dropped", stage=stage.name)
                    return None
                current = result
            except Exception as e:
                logger.error(
                    "stage_failed",
                    stage=stage.name,
                    error=str(e),
                )
                raise PreprocessingError(
                    f"Stage '{stage.name}' failed: {e}",
                    record_id=record.id,
                ) from e

        self._stats.records_output += 1
        return current

    def process_batch(
        self,
        records: Iterable[LogRecord],
        skip_errors: bool = True,
    ) -> list[LogRecord]:
        """
        Process a batch of records.

        Args:
            records: The records to process.
            skip_errors: If True, skip records that fail processing.

        Returns:
            List of successfully processed records.
        """
        results = []
        for record in records:
            try:
                result = self.process_one(record)
                if result is not None:
                    results.append(result)
            except PreprocessingError as e:
                if skip_errors:
                    logger.warning("record_skipped", error=str(e))
                else:
                    raise

        return results

    def process_stream(
        self,
        records: Iterable[LogRecord],
        skip_errors: bool = True,
    ) -> Iterator[LogRecord]:
        """
        Process records as a stream (generator).

        Args:
            records: The records to process.
            skip_errors: If True, skip records that fail processing.

        Yields:
            Successfully processed records.
        """
        for record in records:
            try:
                result = self.process_one(record)
                if result is not None:
                    yield result
            except PreprocessingError as e:
                if skip_errors:
                    logger.warning("record_skipped", error=str(e))
                else:
                    raise

    def get_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._stats = ProcessingStats()

    def add_stage(self, stage: ProcessingStage, position: int | None = None) -> None:
        """
        Add a processing stage to the pipeline.

        Args:
            stage: The stage to add.
            position: Position to insert at. If None, appends to end.
        """
        if position is None:
            self._stages.append(stage)
        else:
            self._stages.insert(position, stage)
        logger.info("stage_added", stage=stage.name, position=position)


class PreprocessingService:
    """
    High-level preprocessing service.

    Provides a simple interface for preprocessing log records,
    suitable for use from gRPC handlers.
    """

    def __init__(self, pipeline: PreprocessingPipeline | None = None) -> None:
        """
        Initialize the service.

        Args:
            pipeline: Custom pipeline to use. If None, uses default.
        """
        self._pipeline = pipeline or PreprocessingPipeline()
        logger.info("preprocessing_service_initialized")

    def preprocess(self, record: LogRecord) -> LogRecord | None:
        """
        Preprocess a single record.

        Args:
            record: The record to preprocess.

        Returns:
            The preprocessed record, or None if dropped.
        """
        return self._pipeline.process_one(record)

    def preprocess_batch(
        self,
        records: list[LogRecord],
        skip_errors: bool = True,
    ) -> list[LogRecord]:
        """
        Preprocess a batch of records.

        Args:
            records: The records to preprocess.
            skip_errors: If True, skip failed records.

        Returns:
            List of preprocessed records.
        """
        start_time = time.time()
        results = self._pipeline.process_batch(records, skip_errors)
        duration = time.time() - start_time

        logger.info(
            "batch_preprocessed",
            input_count=len(records),
            output_count=len(results),
            duration_ms=round(duration * 1000, 2),
        )

        return results

    def create_record_from_raw(
        self,
        raw: str,
        source: str,
        attrs: dict[str, Any] | None = None,
    ) -> LogRecord:
        """
        Create a LogRecord from raw log line.

        Args:
            raw: The raw log line.
            source: The source of the log.
            attrs: Additional attributes.

        Returns:
            A new LogRecord.
        """
        return LogRecord(
            raw=raw,
            message=raw,  # Will be updated by parsing
            source=source,
            attrs=attrs or {},
        )

    def get_stats(self) -> dict[str, Any]:
        """Get preprocessing statistics as dictionary."""
        return self._pipeline.get_stats().to_dict()

    def reset_stats(self) -> None:
        """Reset preprocessing statistics."""
        self._pipeline.reset_stats()
