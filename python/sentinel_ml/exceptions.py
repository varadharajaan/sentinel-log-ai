"""
Custom exception hierarchy for Sentinel ML Engine.

Follows SOLID principles with a clear exception hierarchy:
- SentinelError: Base exception for all sentinel-specific errors
- ConfigurationError: Configuration and validation issues
- IngestionError: Log ingestion and parsing failures
- ProcessingError: ML processing errors (embedding, clustering)
- StorageError: Vector store and persistence errors
- CommunicationError: gRPC and network errors

Each exception includes:
- error_code: Machine-readable error identifier
- context: Additional structured data for debugging
- is_retryable: Whether the operation can be retried
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Machine-readable error codes for categorization and monitoring."""

    # Configuration errors (1xxx)
    CONFIG_INVALID = "SENTINEL_1001"
    CONFIG_MISSING = "SENTINEL_1002"
    CONFIG_VALIDATION = "SENTINEL_1003"

    # Ingestion errors (2xxx)
    INGEST_FILE_NOT_FOUND = "SENTINEL_2001"
    INGEST_PERMISSION_DENIED = "SENTINEL_2002"
    INGEST_PARSE_FAILED = "SENTINEL_2003"
    INGEST_ENCODING_ERROR = "SENTINEL_2004"
    INGEST_TIMEOUT = "SENTINEL_2005"

    # Processing errors (3xxx)
    PROCESS_EMBEDDING_FAILED = "SENTINEL_3001"
    PROCESS_CLUSTERING_FAILED = "SENTINEL_3002"
    PROCESS_NOVELTY_FAILED = "SENTINEL_3003"
    PROCESS_MODEL_LOAD_FAILED = "SENTINEL_3004"
    PROCESS_RESOURCE_EXHAUSTED = "SENTINEL_3005"
    PREPROCESSING_FAILED = "SENTINEL_3006"
    PREPROCESSING_STAGE_FAILED = "SENTINEL_3007"

    # Storage errors (4xxx)
    STORAGE_READ_FAILED = "SENTINEL_4001"
    STORAGE_WRITE_FAILED = "SENTINEL_4002"
    STORAGE_INDEX_CORRUPTED = "SENTINEL_4003"
    STORAGE_CAPACITY_EXCEEDED = "SENTINEL_4004"

    # Communication errors (5xxx)
    COMM_CONNECTION_FAILED = "SENTINEL_5001"
    COMM_TIMEOUT = "SENTINEL_5002"
    COMM_PROTOCOL_ERROR = "SENTINEL_5003"
    COMM_AUTH_FAILED = "SENTINEL_5004"

    # LLM errors (6xxx)
    LLM_PROVIDER_ERROR = "SENTINEL_6001"
    LLM_RATE_LIMITED = "SENTINEL_6002"
    LLM_CONTEXT_TOO_LONG = "SENTINEL_6003"
    LLM_INVALID_RESPONSE = "SENTINEL_6004"

    # General errors (9xxx)
    UNKNOWN = "SENTINEL_9999"


@dataclass
class SentinelError(Exception):
    """
    Base exception for all Sentinel ML errors.

    Provides structured error information for logging and monitoring.

    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error identifier
        context: Additional structured data for debugging
        is_retryable: Whether the operation can be safely retried
        cause: Original exception that caused this error
    """

    message: str
    error_code: ErrorCode = ErrorCode.UNKNOWN
    context: dict[str, Any] = field(default_factory=dict)
    is_retryable: bool = False
    cause: Exception | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [f"[{self.error_code.value}] {self.message}"]
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f" ({context_str})")
        return "".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"context={self.context!r}, "
            f"is_retryable={self.is_retryable})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code.value,
            "message": self.message,
            "context": self.context,
            "is_retryable": self.is_retryable,
            "cause": str(self.cause) if self.cause else None,
        }


@dataclass
class ConfigurationError(SentinelError):
    """Raised when configuration is invalid or missing."""

    error_code: ErrorCode = ErrorCode.CONFIG_INVALID

    @classmethod
    def missing_file(cls, path: str) -> ConfigurationError:
        """Create error for missing configuration file."""
        return cls(
            message=f"Configuration file not found: {path}",
            error_code=ErrorCode.CONFIG_MISSING,
            context={"path": path},
        )

    @classmethod
    def validation_failed(cls, field: str, value: Any, reason: str) -> ConfigurationError:
        """Create error for validation failure."""
        return cls(
            message=f"Configuration validation failed for '{field}': {reason}",
            error_code=ErrorCode.CONFIG_VALIDATION,
            context={"field": field, "value": str(value), "reason": reason},
        )


@dataclass
class IngestionError(SentinelError):
    """Raised when log ingestion fails."""

    error_code: ErrorCode = ErrorCode.INGEST_PARSE_FAILED
    is_retryable: bool = True

    @classmethod
    def file_not_found(cls, path: str) -> IngestionError:
        """Create error for missing log file."""
        return cls(
            message=f"Log file not found: {path}",
            error_code=ErrorCode.INGEST_FILE_NOT_FOUND,
            context={"path": path},
            is_retryable=False,
        )

    @classmethod
    def permission_denied(cls, path: str) -> IngestionError:
        """Create error for permission issues."""
        return cls(
            message=f"Permission denied reading: {path}",
            error_code=ErrorCode.INGEST_PERMISSION_DENIED,
            context={"path": path},
            is_retryable=False,
        )

    @classmethod
    def parse_failed(cls, line: str, line_number: int, reason: str) -> IngestionError:
        """Create error for parse failure."""
        # Truncate long lines for context
        truncated = line[:200] + "..." if len(line) > 200 else line
        return cls(
            message=f"Failed to parse log line {line_number}: {reason}",
            error_code=ErrorCode.INGEST_PARSE_FAILED,
            context={"line": truncated, "line_number": line_number, "reason": reason},
        )

    @classmethod
    def encoding_error(cls, path: str, encoding: str) -> IngestionError:
        """Create error for encoding issues."""
        return cls(
            message=f"Encoding error reading {path} with {encoding}",
            error_code=ErrorCode.INGEST_ENCODING_ERROR,
            context={"path": path, "encoding": encoding},
        )


@dataclass
class ProcessingError(SentinelError):
    """Raised when ML processing fails."""

    error_code: ErrorCode = ErrorCode.PROCESS_EMBEDDING_FAILED

    @classmethod
    def embedding_failed(cls, batch_size: int, reason: str) -> ProcessingError:
        """Create error for embedding failure."""
        return cls(
            message=f"Embedding generation failed: {reason}",
            error_code=ErrorCode.PROCESS_EMBEDDING_FAILED,
            context={"batch_size": batch_size, "reason": reason},
            is_retryable=True,
        )

    @classmethod
    def clustering_failed(cls, n_samples: int, reason: str) -> ProcessingError:
        """Create error for clustering failure."""
        return cls(
            message=f"Clustering failed: {reason}",
            error_code=ErrorCode.PROCESS_CLUSTERING_FAILED,
            context={"n_samples": n_samples, "reason": reason},
        )

    @classmethod
    def model_load_failed(cls, model_name: str, reason: str) -> ProcessingError:
        """Create error for model loading failure."""
        return cls(
            message=f"Failed to load model '{model_name}': {reason}",
            error_code=ErrorCode.PROCESS_MODEL_LOAD_FAILED,
            context={"model_name": model_name, "reason": reason},
            is_retryable=True,
        )

    @classmethod
    def resource_exhausted(cls, resource: str, limit: str) -> ProcessingError:
        """Create error for resource exhaustion."""
        return cls(
            message=f"Resource exhausted: {resource} exceeded {limit}",
            error_code=ErrorCode.PROCESS_RESOURCE_EXHAUSTED,
            context={"resource": resource, "limit": limit},
            is_retryable=True,
        )


@dataclass
class PreprocessingError(SentinelError):
    """Raised when preprocessing operations fail."""

    error_code: ErrorCode = ErrorCode.PREPROCESSING_FAILED

    @classmethod
    def stage_failed(cls, stage_name: str, reason: str) -> PreprocessingError:
        """Create error for stage failure."""
        return cls(
            message=f"Preprocessing stage '{stage_name}' failed: {reason}",
            error_code=ErrorCode.PREPROCESSING_STAGE_FAILED,
            context={"stage_name": stage_name, "reason": reason},
            is_retryable=True,
        )

    @classmethod
    def pipeline_failed(cls, reason: str, records_processed: int = 0) -> PreprocessingError:
        """Create error for pipeline failure."""
        return cls(
            message=f"Preprocessing pipeline failed: {reason}",
            error_code=ErrorCode.PREPROCESSING_FAILED,
            context={"reason": reason, "records_processed": records_processed},
            is_retryable=True,
        )

    @classmethod
    def invalid_record(cls, record_id: str, reason: str) -> PreprocessingError:
        """Create error for invalid record."""
        return cls(
            message=f"Invalid record '{record_id}': {reason}",
            error_code=ErrorCode.PREPROCESSING_FAILED,
            context={"record_id": record_id, "reason": reason},
            is_retryable=False,
        )


@dataclass
class StorageError(SentinelError):
    """Raised when storage operations fail."""

    error_code: ErrorCode = ErrorCode.STORAGE_WRITE_FAILED

    @classmethod
    def read_failed(cls, path: str, reason: str) -> StorageError:
        """Create error for read failure."""
        return cls(
            message=f"Failed to read from storage: {reason}",
            error_code=ErrorCode.STORAGE_READ_FAILED,
            context={"path": path, "reason": reason},
            is_retryable=True,
        )

    @classmethod
    def write_failed(cls, path: str, reason: str) -> StorageError:
        """Create error for write failure."""
        return cls(
            message=f"Failed to write to storage: {reason}",
            error_code=ErrorCode.STORAGE_WRITE_FAILED,
            context={"path": path, "reason": reason},
            is_retryable=True,
        )

    @classmethod
    def index_corrupted(cls, path: str) -> StorageError:
        """Create error for corrupted index."""
        return cls(
            message=f"Vector index is corrupted: {path}",
            error_code=ErrorCode.STORAGE_INDEX_CORRUPTED,
            context={"path": path},
            is_retryable=False,
        )


@dataclass
class CommunicationError(SentinelError):
    """Raised when network communication fails."""

    error_code: ErrorCode = ErrorCode.COMM_CONNECTION_FAILED
    is_retryable: bool = True

    @classmethod
    def connection_failed(cls, address: str, reason: str) -> CommunicationError:
        """Create error for connection failure."""
        return cls(
            message=f"Failed to connect to {address}: {reason}",
            error_code=ErrorCode.COMM_CONNECTION_FAILED,
            context={"address": address, "reason": reason},
        )

    @classmethod
    def timeout(cls, operation: str, timeout_seconds: float) -> CommunicationError:
        """Create error for timeout."""
        return cls(
            message=f"Operation '{operation}' timed out after {timeout_seconds}s",
            error_code=ErrorCode.COMM_TIMEOUT,
            context={"operation": operation, "timeout_seconds": timeout_seconds},
        )

    @classmethod
    def protocol_error(cls, details: str) -> CommunicationError:
        """Create error for protocol issues."""
        return cls(
            message=f"Protocol error: {details}",
            error_code=ErrorCode.COMM_PROTOCOL_ERROR,
            context={"details": details},
        )


@dataclass
class LLMError(SentinelError):
    """Raised when LLM operations fail."""

    error_code: ErrorCode = ErrorCode.LLM_PROVIDER_ERROR

    @classmethod
    def provider_error(cls, provider: str, reason: str) -> LLMError:
        """Create error for LLM provider issues."""
        return cls(
            message=f"LLM provider '{provider}' error: {reason}",
            error_code=ErrorCode.LLM_PROVIDER_ERROR,
            context={"provider": provider, "reason": reason},
            is_retryable=True,
        )

    @classmethod
    def rate_limited(cls, provider: str, retry_after: float | None = None) -> LLMError:
        """Create error for rate limiting."""
        return cls(
            message=f"Rate limited by LLM provider '{provider}'",
            error_code=ErrorCode.LLM_RATE_LIMITED,
            context={"provider": provider, "retry_after": retry_after},
            is_retryable=True,
        )

    @classmethod
    def context_too_long(cls, tokens: int, max_tokens: int) -> LLMError:
        """Create error for context length exceeded."""
        return cls(
            message=f"Context length {tokens} exceeds maximum {max_tokens}",
            error_code=ErrorCode.LLM_CONTEXT_TOO_LONG,
            context={"tokens": tokens, "max_tokens": max_tokens},
        )

    @classmethod
    def invalid_response(cls, reason: str) -> LLMError:
        """Create error for invalid LLM response."""
        return cls(
            message=f"Invalid LLM response: {reason}",
            error_code=ErrorCode.LLM_INVALID_RESPONSE,
            context={"reason": reason},
            is_retryable=True,
        )
