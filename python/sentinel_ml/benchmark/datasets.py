"""
Dataset generation for benchmarking.

Provides utilities for generating synthetic log datasets
of various sizes for scale testing.
"""

from __future__ import annotations

import random
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)


class LogLevel(Enum):
    """Log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogPattern(Enum):
    """Common log message patterns."""

    HTTP_REQUEST = "http_request"
    DATABASE_QUERY = "database_query"
    AUTH_EVENT = "auth_event"
    ERROR_STACK = "error_stack"
    METRIC_LOG = "metric_log"
    SYSTEM_EVENT = "system_event"
    CUSTOM = "custom"


@dataclass
class DatasetConfig:
    """
    Configuration for dataset generation.

    Attributes:
        name: Name of the dataset.
        size: Number of log records to generate.
        patterns: Distribution of log patterns.
        level_distribution: Distribution of log levels.
        time_range_hours: Time range for timestamps.
        include_errors: Percentage of error logs (0.0-1.0).
        include_stacktraces: Whether to include stack traces.
        sources: List of source identifiers.
        seed: Random seed for reproducibility.
    """

    name: str
    size: int
    patterns: dict[LogPattern, float] = field(default_factory=dict)
    level_distribution: dict[LogLevel, float] = field(default_factory=dict)
    time_range_hours: int = 24
    include_errors: float = 0.1
    include_stacktraces: bool = True
    sources: list[str] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        """Set defaults after initialization."""
        if not self.patterns:
            self.patterns = {
                LogPattern.HTTP_REQUEST: 0.4,
                LogPattern.DATABASE_QUERY: 0.2,
                LogPattern.AUTH_EVENT: 0.15,
                LogPattern.SYSTEM_EVENT: 0.15,
                LogPattern.ERROR_STACK: 0.1,
            }
        if not self.level_distribution:
            self.level_distribution = {
                LogLevel.DEBUG: 0.1,
                LogLevel.INFO: 0.5,
                LogLevel.WARNING: 0.2,
                LogLevel.ERROR: 0.15,
                LogLevel.CRITICAL: 0.05,
            }
        if not self.sources:
            self.sources = [
                "/var/log/app/web.log",
                "/var/log/app/api.log",
                "/var/log/app/worker.log",
            ]


@dataclass
class GeneratedLogRecord:
    """
    A generated log record.

    Attributes:
        id: Unique identifier.
        message: Log message.
        level: Log level.
        source: Source identifier.
        timestamp: Log timestamp.
        raw: Raw log line.
        attrs: Additional attributes.
    """

    id: str
    message: str
    level: str
    source: str
    timestamp: datetime
    raw: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "message": self.message,
            "level": self.level,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "raw": self.raw,
            "attrs": self.attrs,
        }

    def to_json_line(self) -> str:
        """Convert to JSON log line."""
        import json

        data = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "source": self.source,
            **self.attrs,
        }
        return json.dumps(data)


class DatasetGenerator:
    """
    Generator for synthetic log datasets.

    Creates realistic log data for benchmarking and testing.
    """

    HTTP_METHODS: ClassVar[list[str]] = ["GET", "POST", "PUT", "DELETE", "PATCH"]
    HTTP_PATHS: ClassVar[list[str]] = [
        "/api/v1/users",
        "/api/v1/orders",
        "/api/v1/products",
        "/api/v1/auth/login",
        "/api/v1/auth/logout",
        "/api/v1/health",
        "/api/v1/metrics",
    ]
    HTTP_STATUS_CODES: ClassVar[list[int]] = [200, 201, 204, 400, 401, 403, 404, 500, 502, 503]
    DB_OPERATIONS: ClassVar[list[str]] = ["SELECT", "INSERT", "UPDATE", "DELETE"]
    DB_TABLES: ClassVar[list[str]] = ["users", "orders", "products", "sessions", "logs"]
    AUTH_ACTIONS: ClassVar[list[str]] = ["login", "logout", "token_refresh", "password_reset"]
    ERROR_TYPES: ClassVar[list[str]] = [
        "ConnectionError",
        "TimeoutError",
        "ValueError",
        "KeyError",
        "RuntimeError",
        "DatabaseError",
        "AuthenticationError",
    ]
    SERVICES: ClassVar[list[str]] = ["web", "api", "worker", "scheduler", "cache"]
    USERNAMES: ClassVar[list[str]] = ["alice", "bob", "charlie", "david", "eve", "frank"]

    def __init__(self, config: DatasetConfig) -> None:
        """
        Initialize dataset generator.

        Args:
            config: Dataset configuration.
        """
        self.config = config
        self._rng = random.Random(config.seed)
        self._generated_count = 0
        logger.info(
            "dataset_generator_initialized",
            name=config.name,
            size=config.size,
            seed=config.seed,
        )

    def _random_ip(self) -> str:
        """Generate random IP address."""
        return ".".join(str(self._rng.randint(1, 255)) for _ in range(4))

    def _random_uuid(self) -> str:
        """Generate random UUID."""
        return str(uuid.UUID(int=self._rng.getrandbits(128)))

    def _random_string(self, length: int = 8) -> str:
        """Generate random alphanumeric string."""
        chars = string.ascii_lowercase + string.digits
        return "".join(self._rng.choice(chars) for _ in range(length))

    def _random_timestamp(self) -> datetime:
        """Generate random timestamp within configured range."""
        now = datetime.now(timezone.utc)
        offset = timedelta(seconds=self._rng.randint(0, self.config.time_range_hours * 3600))
        return now - offset

    def _weighted_choice(self, choices: dict[Any, float]) -> Any:
        """Make weighted random choice."""
        items = list(choices.keys())
        weights = list(choices.values())
        return self._rng.choices(items, weights=weights, k=1)[0]

    def _generate_http_request(self) -> str:
        """Generate HTTP request log message."""
        method = self._rng.choice(self.HTTP_METHODS)
        path = self._rng.choice(self.HTTP_PATHS)
        status = self._rng.choice(self.HTTP_STATUS_CODES)
        duration = self._rng.randint(1, 5000)
        ip = self._random_ip()

        return f'{method} {path} {status} {duration}ms - {ip} - "{self._random_string(32)}"'

    def _generate_database_query(self) -> str:
        """Generate database query log message."""
        operation = self._rng.choice(self.DB_OPERATIONS)
        table = self._rng.choice(self.DB_TABLES)
        duration = self._rng.randint(1, 1000)
        rows = self._rng.randint(0, 10000)

        return f"Query executed: {operation} on {table} - {duration}ms, {rows} rows affected"

    def _generate_auth_event(self) -> str:
        """Generate authentication event log message."""
        action = self._rng.choice(self.AUTH_ACTIONS)
        user = self._rng.choice(self.USERNAMES)
        ip = self._random_ip()
        session_id = self._random_uuid()

        success = self._rng.random() > 0.1
        status = "successful" if success else "failed"

        return f"Auth {action} {status} for user={user} from ip={ip} session={session_id}"

    def _generate_error_stack(self) -> str:
        """Generate error with stack trace."""
        error_type = self._rng.choice(self.ERROR_TYPES)
        message = f"Operation failed: {self._random_string(16)}"

        if not self.config.include_stacktraces:
            return f"{error_type}: {message}"

        frames = [
            f'  File "/app/src/{self._random_string(8)}.py", line {self._rng.randint(1, 500)}, in {self._random_string(10)}'
            for _ in range(self._rng.randint(2, 5))
        ]
        stack = "\n".join(frames)

        return f"{error_type}: {message}\nTraceback:\n{stack}"

    def _generate_metric_log(self) -> str:
        """Generate metric log message."""
        metric = self._rng.choice(
            ["cpu_usage", "memory_usage", "request_count", "error_rate", "latency_p99"]
        )
        value = round(self._rng.uniform(0, 100), 2)
        service = self._rng.choice(self.SERVICES)

        return f"Metric {metric}={value} service={service}"

    def _generate_system_event(self) -> str:
        """Generate system event log message."""
        events = [
            f"Service {self._rng.choice(self.SERVICES)} started",
            f"Service {self._rng.choice(self.SERVICES)} stopped",
            f"Health check passed for {self._rng.choice(self.SERVICES)}",
            f"Configuration reloaded for {self._rng.choice(self.SERVICES)}",
            f"Connection pool resized to {self._rng.randint(10, 100)}",
            f"Cache cleared: {self._rng.randint(100, 10000)} entries removed",
            f"Scheduled task completed: job_{self._random_string(6)}",
        ]
        return self._rng.choice(events)

    def _generate_message(self, pattern: LogPattern) -> str:
        """Generate message for given pattern."""
        generators = {
            LogPattern.HTTP_REQUEST: self._generate_http_request,
            LogPattern.DATABASE_QUERY: self._generate_database_query,
            LogPattern.AUTH_EVENT: self._generate_auth_event,
            LogPattern.ERROR_STACK: self._generate_error_stack,
            LogPattern.METRIC_LOG: self._generate_metric_log,
            LogPattern.SYSTEM_EVENT: self._generate_system_event,
        }

        generator = generators.get(pattern, self._generate_system_event)
        return generator()

    def generate_one(self) -> GeneratedLogRecord:
        """
        Generate a single log record.

        Returns:
            Generated log record.
        """
        pattern = self._weighted_choice(self.config.patterns)
        level = self._weighted_choice(self.config.level_distribution)
        source = self._rng.choice(self.config.sources)
        timestamp = self._random_timestamp()
        message = self._generate_message(pattern)

        record_id = f"log_{self._generated_count:010d}"
        self._generated_count += 1

        raw = f"{timestamp.isoformat()} [{level.value}] {message}"

        return GeneratedLogRecord(
            id=record_id,
            message=message,
            level=level.value,
            source=source,
            timestamp=timestamp,
            raw=raw,
            attrs={"pattern": pattern.value},
        )

    def generate_batch(self, size: int) -> list[GeneratedLogRecord]:
        """
        Generate a batch of log records.

        Args:
            size: Number of records to generate.

        Returns:
            List of generated log records.
        """
        records = [self.generate_one() for _ in range(size)]
        logger.debug(
            "batch_generated",
            name=self.config.name,
            size=size,
            total_generated=self._generated_count,
        )
        return records

    def generate_all(self) -> list[GeneratedLogRecord]:
        """
        Generate all records as specified in config.

        Returns:
            List of all generated log records.
        """
        logger.info(
            "generating_dataset",
            name=self.config.name,
            size=self.config.size,
        )

        records = self.generate_batch(self.config.size)

        logger.info(
            "dataset_generated",
            name=self.config.name,
            size=len(records),
        )
        return records

    def generate_iter(self, batch_size: int = 1000) -> Iterator[GeneratedLogRecord]:
        """
        Generate records as an iterator.

        Args:
            batch_size: Internal batch size for generation.

        Yields:
            Generated log records.
        """
        remaining = self.config.size

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            yield from self.generate_batch(current_batch)
            remaining -= current_batch


def generate_test_logs(
    size: int,
    name: str = "test_dataset",
    seed: int | None = None,
) -> list[GeneratedLogRecord]:
    """
    Convenience function to generate test logs.

    Args:
        size: Number of logs to generate.
        name: Dataset name.
        seed: Random seed for reproducibility.

    Returns:
        List of generated log records.
    """
    config = DatasetConfig(name=name, size=size, seed=seed)
    generator = DatasetGenerator(config)
    return generator.generate_all()


def create_scale_datasets() -> dict[str, DatasetConfig]:
    """
    Create standard scale test dataset configurations.

    Returns:
        Dictionary of dataset configs for different scales.
    """
    return {
        "small": DatasetConfig(name="small", size=1000, seed=42),
        "medium": DatasetConfig(name="medium", size=10000, seed=42),
        "large": DatasetConfig(name="large", size=100000, seed=42),
        "xlarge": DatasetConfig(name="xlarge", size=1000000, seed=42),
    }
