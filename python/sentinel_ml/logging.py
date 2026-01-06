"""
Structured logging for the ML engine.

Supports JSONL (JSON Lines) output format for Athena analysis.
Features:
- Rolling log files with configurable size and backup count
- JSONL format (one JSON object per line) for analytics pipelines
- Structured context binding for correlation
- No ASCII art or decorations - clean machine-readable output
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from sentinel_ml.config import get_config

if TYPE_CHECKING:
    from structlog.types import Processor

# Default log directory
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "sentinel-ml.jsonl"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


class JSONLRotatingHandler(RotatingFileHandler):
    """
    Rotating file handler that writes JSONL format.

    Each log entry is a single JSON object on its own line,
    compatible with AWS Athena, Spark, and other analytics tools.
    """

    def __init__(
        self,
        filename: str | Path,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        encoding: str = "utf-8",
    ):
        """
        Initialize the JSONL rotating handler.

        Args:
            filename: Path to the log file (should end in .jsonl)
            max_bytes: Maximum size per log file before rotation
            backup_count: Number of backup files to keep
            encoding: File encoding (default: utf-8)
        """
        # Ensure parent directory exists
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename=str(path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
        )


def _add_service_info(
    _logger: logging.Logger, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Add service metadata to each log entry for Athena analysis."""
    event_dict["service"] = "sentinel-ml"
    event_dict["hostname"] = os.environ.get("HOSTNAME", "unknown")
    event_dict["pid"] = os.getpid()
    return event_dict


def _add_timestamp_utc(
    _logger: logging.Logger, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Add ISO8601 UTC timestamp for consistent time-based queries."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def _format_exception(
    _logger: logging.Logger, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Format exceptions as structured data instead of multiline strings."""
    if "exception" in event_dict:
        exc_info = event_dict.pop("exception")
        if exc_info:
            event_dict["exception"] = {
                "type": type(exc_info).__name__ if exc_info else None,
                "message": str(exc_info) if exc_info else None,
            }
    return event_dict


def setup_logging(
    level: str | None = None,
    format: str | None = None,
    log_file: str | None = None,
    log_dir: str | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
    enable_console: bool = True,
    enable_file: bool = True,
) -> None:
    """
    Configure structured JSONL logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to config value.
        format: Log format (json, plain). Defaults to config value.
        log_file: Log filename (not full path). Defaults to sentinel-ml.jsonl.
        log_dir: Directory for log files. Defaults to ./logs.
        max_bytes: Maximum size per log file before rotation. Default 10MB.
        backup_count: Number of backup files to keep. Default 5.
        enable_console: Whether to log to console. Default True.
        enable_file: Whether to log to file. Default True.
    """
    config = get_config()

    level = level or config.logging.level
    format = format or config.logging.format
    log_file = log_file or DEFAULT_LOG_FILE
    log_dir = log_dir or DEFAULT_LOG_DIR
    max_bytes = max_bytes or DEFAULT_MAX_BYTES
    backup_count = backup_count or DEFAULT_BACKUP_COUNT

    # Convert level string to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Processors for JSONL format - optimized for Athena queries
    jsonl_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _add_timestamp_utc,
        _add_service_info,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        _format_exception,
        structlog.processors.UnicodeDecoder(),
    ]

    # Common processors for console output
    console_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure structlog
    structlog.configure(
        processors=[
            *jsonl_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handlers: list[logging.Handler] = []

    # JSONL file handler with rotation
    if enable_file:
        log_path = Path(log_dir) / log_file
        jsonl_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=jsonl_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )

        file_handler = JSONLRotatingHandler(
            filename=log_path,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
        file_handler.setFormatter(jsonl_formatter)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    # Console handler
    if enable_console:
        if format.lower() == "json":
            console_renderer: Processor = structlog.processors.JSONRenderer()
        else:
            # Plain text for development - no colors for clean output
            console_renderer = structlog.dev.ConsoleRenderer(
                colors=False,
                exception_formatter=structlog.dev.plain_traceback,
            )

        console_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=console_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                console_renderer,
            ],
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        handlers.append(console_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(log_level)

    # Reduce noise from third-party libraries
    for lib in ["urllib3", "httpx", "grpc", "asyncio", "concurrent"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> Any:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured structlog logger

    Example:
        logger = get_logger(__name__)
        logger.info("processing_started", batch_size=100)
        logger.warning("high_memory_usage", memory_mb=1024)
        logger.error("processing_failed", error_code="SENTINEL_3001")
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables to all subsequent log messages in this context.

    Use for request/operation correlation across log entries.

    Example:
        bind_context(request_id="abc123", user_id="user456")
        logger.info("processing_request")  # Will include request_id and user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


@contextmanager
def with_context(**kwargs: Any) -> Iterator[None]:
    """
    Context manager for temporary context binding.

    Example:
        with with_context(operation="embedding"):
            logger.info("started")
            # ... do work ...
            logger.info("completed")
    """
    with structlog.contextvars.bound_contextvars(**kwargs):
        yield


# Initialize logging on import with defaults
# Can be reconfigured later with setup_logging()
_initialized = False


def ensure_logging() -> None:
    """Ensure logging is initialized (idempotent)."""
    global _initialized
    if not _initialized:
        setup_logging()
        _initialized = True


def get_log_file_path(log_dir: str | None = None, log_file: str | None = None) -> Path:
    """
    Get the full path to the current log file.

    Useful for log analysis or monitoring tools.
    """
    return Path(log_dir or DEFAULT_LOG_DIR) / (log_file or DEFAULT_LOG_FILE)
