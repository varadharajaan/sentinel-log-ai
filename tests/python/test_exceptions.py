"""
Tests for the Sentinel ML exception hierarchy.

Verifies:
- Exception creation and message formatting
- Error code assignment
- Context data handling
- Factory method behavior
- Serialization to dict for logging
"""

import pytest

from sentinel_ml.exceptions import (
    CommunicationError,
    ConfigurationError,
    ErrorCode,
    IngestionError,
    LLMError,
    ProcessingError,
    SentinelError,
    StorageError,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_error_codes_are_strings(self) -> None:
        """Error codes should be string values."""
        assert isinstance(ErrorCode.CONFIG_INVALID.value, str)
        assert ErrorCode.CONFIG_INVALID.value == "SENTINEL_1001"

    def test_error_code_ranges(self) -> None:
        """Error codes should follow the defined ranges."""
        # Configuration: 1xxx
        assert ErrorCode.CONFIG_INVALID.value.startswith("SENTINEL_1")
        assert ErrorCode.CONFIG_MISSING.value.startswith("SENTINEL_1")

        # Ingestion: 2xxx
        assert ErrorCode.INGEST_FILE_NOT_FOUND.value.startswith("SENTINEL_2")
        assert ErrorCode.INGEST_PARSE_FAILED.value.startswith("SENTINEL_2")

        # Processing: 3xxx
        assert ErrorCode.PROCESS_EMBEDDING_FAILED.value.startswith("SENTINEL_3")
        assert ErrorCode.PROCESS_CLUSTERING_FAILED.value.startswith("SENTINEL_3")

        # Storage: 4xxx
        assert ErrorCode.STORAGE_READ_FAILED.value.startswith("SENTINEL_4")
        assert ErrorCode.STORAGE_WRITE_FAILED.value.startswith("SENTINEL_4")

        # Communication: 5xxx
        assert ErrorCode.COMM_CONNECTION_FAILED.value.startswith("SENTINEL_5")
        assert ErrorCode.COMM_TIMEOUT.value.startswith("SENTINEL_5")

        # LLM: 6xxx
        assert ErrorCode.LLM_PROVIDER_ERROR.value.startswith("SENTINEL_6")
        assert ErrorCode.LLM_RATE_LIMITED.value.startswith("SENTINEL_6")


class TestSentinelError:
    """Tests for the base SentinelError class."""

    def test_basic_creation(self) -> None:
        """Basic error creation with message."""
        error = SentinelError(message="Something went wrong")
        assert error.message == "Something went wrong"
        assert error.error_code == ErrorCode.UNKNOWN
        assert error.is_retryable is False
        assert error.cause is None

    def test_with_error_code(self) -> None:
        """Error with specific error code."""
        error = SentinelError(
            message="Config invalid",
            error_code=ErrorCode.CONFIG_INVALID,
        )
        assert error.error_code == ErrorCode.CONFIG_INVALID

    def test_with_context(self) -> None:
        """Error with context data."""
        error = SentinelError(
            message="Parse failed",
            context={"line": 42, "file": "app.log"},
        )
        assert error.context["line"] == 42
        assert error.context["file"] == "app.log"

    def test_str_without_context(self) -> None:
        """String representation without context."""
        error = SentinelError(
            message="Test error",
            error_code=ErrorCode.CONFIG_INVALID,
        )
        assert str(error) == "[SENTINEL_1001] Test error"

    def test_str_with_context(self) -> None:
        """String representation includes context."""
        error = SentinelError(
            message="Test error",
            error_code=ErrorCode.CONFIG_INVALID,
            context={"key": "value"},
        )
        result = str(error)
        assert "[SENTINEL_1001] Test error" in result
        assert "key=value" in result

    def test_repr(self) -> None:
        """Repr shows all fields."""
        error = SentinelError(
            message="Test",
            error_code=ErrorCode.UNKNOWN,
            is_retryable=True,
        )
        result = repr(error)
        assert "SentinelError" in result
        assert "message='Test'" in result
        assert "is_retryable=True" in result

    def test_to_dict(self) -> None:
        """Serialization to dictionary."""
        cause = ValueError("Original error")
        error = SentinelError(
            message="Wrapped error",
            error_code=ErrorCode.PROCESS_EMBEDDING_FAILED,
            context={"batch": 32},
            is_retryable=True,
            cause=cause,
        )
        result = error.to_dict()

        assert result["error_type"] == "SentinelError"
        assert result["error_code"] == "SENTINEL_3001"
        assert result["message"] == "Wrapped error"
        assert result["context"] == {"batch": 32}
        assert result["is_retryable"] is True
        assert "Original error" in result["cause"]

    def test_is_exception(self) -> None:
        """SentinelError is a proper Exception."""
        error = SentinelError(message="Test error")
        assert isinstance(error, Exception)

        # Can be raised and caught
        with pytest.raises(SentinelError) as exc_info:
            raise error
        assert exc_info.value.message == "Test error"


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_default_error_code(self) -> None:
        """Default error code is CONFIG_INVALID."""
        error = ConfigurationError(message="Bad config")
        assert error.error_code == ErrorCode.CONFIG_INVALID

    def test_missing_file_factory(self) -> None:
        """Factory method for missing config file."""
        error = ConfigurationError.missing_file("/path/to/config.yaml")
        assert error.error_code == ErrorCode.CONFIG_MISSING
        assert "/path/to/config.yaml" in error.message
        assert error.context["path"] == "/path/to/config.yaml"

    def test_validation_failed_factory(self) -> None:
        """Factory method for validation failure."""
        error = ConfigurationError.validation_failed(
            field="batch_size",
            value=-1,
            reason="must be positive",
        )
        assert error.error_code == ErrorCode.CONFIG_VALIDATION
        assert "batch_size" in error.message
        assert error.context["field"] == "batch_size"
        assert error.context["reason"] == "must be positive"


class TestIngestionError:
    """Tests for IngestionError."""

    def test_default_retryable(self) -> None:
        """Ingestion errors are retryable by default."""
        error = IngestionError(message="Temporary failure")
        assert error.is_retryable is True

    def test_file_not_found_not_retryable(self) -> None:
        """File not found is not retryable."""
        error = IngestionError.file_not_found("/var/log/missing.log")
        assert error.error_code == ErrorCode.INGEST_FILE_NOT_FOUND
        assert error.is_retryable is False

    def test_permission_denied(self) -> None:
        """Permission denied factory."""
        error = IngestionError.permission_denied("/var/log/secure")
        assert error.error_code == ErrorCode.INGEST_PERMISSION_DENIED
        assert "/var/log/secure" in error.message

    def test_parse_failed_truncates_long_lines(self) -> None:
        """Long log lines are truncated in context."""
        long_line = "x" * 500
        error = IngestionError.parse_failed(long_line, 100, "invalid format")
        # Line should be truncated to 200 chars + "..."
        assert len(error.context["line"]) == 203
        assert error.context["line"].endswith("...")

    def test_parse_failed_short_line(self) -> None:
        """Short lines are not truncated."""
        short_line = "ERROR: something happened"
        error = IngestionError.parse_failed(short_line, 42, "unknown format")
        assert error.context["line"] == short_line
        assert error.context["line_number"] == 42


class TestProcessingError:
    """Tests for ProcessingError."""

    def test_embedding_failed(self) -> None:
        """Embedding failure factory."""
        error = ProcessingError.embedding_failed(32, "out of memory")
        assert error.error_code == ErrorCode.PROCESS_EMBEDDING_FAILED
        assert error.is_retryable is True
        assert error.context["batch_size"] == 32

    def test_clustering_failed(self) -> None:
        """Clustering failure factory."""
        error = ProcessingError.clustering_failed(100, "insufficient samples")
        assert error.error_code == ErrorCode.PROCESS_CLUSTERING_FAILED
        assert error.context["n_samples"] == 100

    def test_model_load_failed(self) -> None:
        """Model loading failure factory."""
        error = ProcessingError.model_load_failed(
            "all-MiniLM-L6-v2",
            "network error",
        )
        assert error.error_code == ErrorCode.PROCESS_MODEL_LOAD_FAILED
        assert "all-MiniLM-L6-v2" in error.message

    def test_resource_exhausted(self) -> None:
        """Resource exhaustion factory."""
        error = ProcessingError.resource_exhausted("GPU memory", "16GB")
        assert error.error_code == ErrorCode.PROCESS_RESOURCE_EXHAUSTED
        assert error.is_retryable is True


class TestStorageError:
    """Tests for StorageError."""

    def test_read_failed(self) -> None:
        """Storage read failure factory."""
        error = StorageError.read_failed("/data/index.faiss", "file corrupted")
        assert error.error_code == ErrorCode.STORAGE_READ_FAILED
        assert error.is_retryable is True

    def test_write_failed(self) -> None:
        """Storage write failure factory."""
        error = StorageError.write_failed("/data/index.faiss", "disk full")
        assert error.error_code == ErrorCode.STORAGE_WRITE_FAILED

    def test_index_corrupted_not_retryable(self) -> None:
        """Corrupted index is not retryable."""
        error = StorageError.index_corrupted("/data/index.faiss")
        assert error.error_code == ErrorCode.STORAGE_INDEX_CORRUPTED
        assert error.is_retryable is False


class TestCommunicationError:
    """Tests for CommunicationError."""

    def test_connection_failed(self) -> None:
        """Connection failure factory."""
        error = CommunicationError.connection_failed(
            "localhost:50051",
            "connection refused",
        )
        assert error.error_code == ErrorCode.COMM_CONNECTION_FAILED
        assert error.is_retryable is True
        assert error.context["address"] == "localhost:50051"

    def test_timeout(self) -> None:
        """Timeout factory."""
        error = CommunicationError.timeout("embedding", 30.0)
        assert error.error_code == ErrorCode.COMM_TIMEOUT
        assert error.context["timeout_seconds"] == 30.0

    def test_protocol_error(self) -> None:
        """Protocol error factory."""
        error = CommunicationError.protocol_error("invalid message format")
        assert error.error_code == ErrorCode.COMM_PROTOCOL_ERROR


class TestLLMError:
    """Tests for LLMError."""

    def test_provider_error(self) -> None:
        """LLM provider error factory."""
        error = LLMError.provider_error("ollama", "server unavailable")
        assert error.error_code == ErrorCode.LLM_PROVIDER_ERROR
        assert "ollama" in error.message

    def test_rate_limited(self) -> None:
        """Rate limiting factory."""
        error = LLMError.rate_limited("openai", retry_after=60.0)
        assert error.error_code == ErrorCode.LLM_RATE_LIMITED
        assert error.is_retryable is True
        assert error.context["retry_after"] == 60.0

    def test_context_too_long(self) -> None:
        """Context too long factory."""
        error = LLMError.context_too_long(tokens=50000, max_tokens=32000)
        assert error.error_code == ErrorCode.LLM_CONTEXT_TOO_LONG
        assert error.context["tokens"] == 50000
        assert error.context["max_tokens"] == 32000

    def test_invalid_response(self) -> None:
        """Invalid response factory."""
        error = LLMError.invalid_response("empty response body")
        assert error.error_code == ErrorCode.LLM_INVALID_RESPONSE
        assert error.is_retryable is True


class TestExceptionHierarchy:
    """Tests for exception hierarchy and catching."""

    def test_all_inherit_from_sentinel_error(self) -> None:
        """All custom exceptions inherit from SentinelError."""
        assert issubclass(ConfigurationError, SentinelError)
        assert issubclass(IngestionError, SentinelError)
        assert issubclass(ProcessingError, SentinelError)
        assert issubclass(StorageError, SentinelError)
        assert issubclass(CommunicationError, SentinelError)
        assert issubclass(LLMError, SentinelError)

    def test_can_catch_specific_type(self) -> None:
        """Can catch specific exception types."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError.missing_file("/config.yaml")

    def test_can_catch_as_base_type(self) -> None:
        """Can catch as base SentinelError."""
        with pytest.raises(SentinelError):
            raise IngestionError.file_not_found("/missing.log")

    def test_can_catch_as_exception(self) -> None:
        """Can catch as generic Exception."""
        with pytest.raises(ProcessingError):
            raise ProcessingError.embedding_failed(32, "error")
