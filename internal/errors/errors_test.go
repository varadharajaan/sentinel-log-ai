// Package errors_test provides tests for the sentinel error types.
package errors_test

import (
	"errors"
	"testing"

	sentinelerrors "sentinel-log-ai/internal/errors"
)

func TestErrorCodes(t *testing.T) {
	t.Run("error codes follow ranges", func(t *testing.T) {
		// Configuration: 1xxx
		if sentinelerrors.ErrCodeConfigInvalid[:11] != "SENTINEL_10" {
			t.Errorf("config errors should be 1xxx, got %s", sentinelerrors.ErrCodeConfigInvalid)
		}

		// Ingestion: 2xxx
		if sentinelerrors.ErrCodeIngestFileNotFound[:11] != "SENTINEL_20" {
			t.Errorf("ingest errors should be 2xxx, got %s", sentinelerrors.ErrCodeIngestFileNotFound)
		}

		// Processing: 3xxx
		if sentinelerrors.ErrCodeProcessEmbeddingFailed[:11] != "SENTINEL_30" {
			t.Errorf("process errors should be 3xxx, got %s", sentinelerrors.ErrCodeProcessEmbeddingFailed)
		}

		// Storage: 4xxx
		if sentinelerrors.ErrCodeStorageReadFailed[:11] != "SENTINEL_40" {
			t.Errorf("storage errors should be 4xxx, got %s", sentinelerrors.ErrCodeStorageReadFailed)
		}

		// Communication: 5xxx
		if sentinelerrors.ErrCodeCommConnectionFailed[:11] != "SENTINEL_50" {
			t.Errorf("comm errors should be 5xxx, got %s", sentinelerrors.ErrCodeCommConnectionFailed)
		}
	})
}

func TestSentinelError(t *testing.T) {
	t.Run("Error method formats correctly", func(t *testing.T) {
		err := sentinelerrors.NewSentinelError(
			sentinelerrors.ErrCodeConfigInvalid,
			"test error",
			nil,
		)
		expected := "[SENTINEL_1001] test error"
		if err.Error() != expected {
			t.Errorf("expected %q, got %q", expected, err.Error())
		}
	})

	t.Run("Error with cause includes cause", func(t *testing.T) {
		cause := errors.New("original error")
		err := sentinelerrors.NewSentinelError(
			sentinelerrors.ErrCodeConfigInvalid,
			"wrapped error",
			cause,
		)
		result := err.Error()
		if result != "[SENTINEL_1001] wrapped error: original error" {
			t.Errorf("unexpected error string: %s", result)
		}
	})

	t.Run("WithContext adds context", func(t *testing.T) {
		err := sentinelerrors.NewSentinelError(
			sentinelerrors.ErrCodeConfigInvalid,
			"test",
			nil,
		)
		err = err.WithContext("key", "value")
		if err.Context["key"] != "value" {
			t.Error("context not set correctly")
		}
	})

	t.Run("ToMap serializes correctly", func(t *testing.T) {
		err := sentinelerrors.NewSentinelError(
			sentinelerrors.ErrCodeConfigInvalid,
			"test error",
			nil,
		)
		err.IsRetryable = true
		err.Context["field"] = "value"

		m := err.ToMap()
		if m["error_code"] != "SENTINEL_1001" {
			t.Errorf("unexpected error_code: %v", m["error_code"])
		}
		if m["message"] != "test error" {
			t.Errorf("unexpected message: %v", m["message"])
		}
		if m["is_retryable"] != true {
			t.Error("is_retryable should be true")
		}
	})
}

func TestUnwrap(t *testing.T) {
	t.Run("Unwrap returns cause", func(t *testing.T) {
		cause := errors.New("root cause")
		err := sentinelerrors.NewSentinelError(
			sentinelerrors.ErrCodeConfigInvalid,
			"wrapped",
			cause,
		)
		if err.Unwrap() != cause {
			t.Error("Unwrap should return the cause")
		}
	})

	t.Run("errors.Is works with cause", func(t *testing.T) {
		err := sentinelerrors.NewConfigMissingError("/config.yaml")
		if !errors.Is(err, sentinelerrors.ErrConfigMissing) {
			t.Error("errors.Is should match the cause")
		}
	})

	t.Run("errors.As works with SentinelError", func(t *testing.T) {
		err := sentinelerrors.NewIngestFileNotFoundError("/missing.log")
		var sentinelErr *sentinelerrors.SentinelError
		if !errors.As(err, &sentinelErr) {
			t.Error("errors.As should work with SentinelError")
		}
		if sentinelErr.Code != sentinelerrors.ErrCodeIngestFileNotFound {
			t.Errorf("unexpected code: %s", sentinelErr.Code)
		}
	})
}

func TestConfigurationErrors(t *testing.T) {
	t.Run("NewConfigInvalidError", func(t *testing.T) {
		cause := errors.New("parse error")
		err := sentinelerrors.NewConfigInvalidError("failed to parse", cause)
		if err.Code != sentinelerrors.ErrCodeConfigInvalid {
			t.Errorf("unexpected code: %s", err.Code)
		}
		if err.IsRetryable {
			t.Error("config errors should not be retryable")
		}
	})

	t.Run("NewConfigMissingError", func(t *testing.T) {
		err := sentinelerrors.NewConfigMissingError("/etc/app/config.yaml")
		if err.Context["path"] != "/etc/app/config.yaml" {
			t.Errorf("unexpected path: %v", err.Context["path"])
		}
	})

	t.Run("NewConfigValidationError", func(t *testing.T) {
		err := sentinelerrors.NewConfigValidationError("port", 70000, "must be < 65536")
		if err.Context["field"] != "port" {
			t.Errorf("unexpected field: %v", err.Context["field"])
		}
		if err.Context["reason"] != "must be < 65536" {
			t.Errorf("unexpected reason: %v", err.Context["reason"])
		}
	})
}

func TestIngestionErrors(t *testing.T) {
	t.Run("NewIngestFileNotFoundError", func(t *testing.T) {
		err := sentinelerrors.NewIngestFileNotFoundError("/var/log/app.log")
		if err.Code != sentinelerrors.ErrCodeIngestFileNotFound {
			t.Errorf("unexpected code: %s", err.Code)
		}
		if err.IsRetryable {
			t.Error("file not found should not be retryable")
		}
	})

	t.Run("NewIngestParseError truncates long lines", func(t *testing.T) {
		longLine := string(make([]byte, 500))
		for i := range longLine {
			longLine = longLine[:i] + "x" + longLine[i+1:]
		}
		// Create a proper long string
		longLine = ""
		for i := 0; i < 500; i++ {
			longLine += "x"
		}
		err := sentinelerrors.NewIngestParseError(longLine, 100, "invalid format")
		line := err.Context["line"].(string)
		if len(line) != 203 { // 200 + "..."
			t.Errorf("line should be truncated to 203 chars, got %d", len(line))
		}
	})

	t.Run("NewIngestParseError keeps short lines", func(t *testing.T) {
		shortLine := "ERROR: something happened"
		err := sentinelerrors.NewIngestParseError(shortLine, 42, "unknown")
		if err.Context["line"] != shortLine {
			t.Errorf("short line should not be truncated")
		}
		if err.Context["line_number"] != 42 {
			t.Errorf("unexpected line_number: %v", err.Context["line_number"])
		}
	})

	t.Run("NewIngestTimeoutError", func(t *testing.T) {
		err := sentinelerrors.NewIngestTimeoutError("/var/log/app.log", 30.0)
		if err.Code != sentinelerrors.ErrCodeIngestTimeout {
			t.Errorf("unexpected code: %s", err.Code)
		}
		if !err.IsRetryable {
			t.Error("timeout should be retryable")
		}
	})
}

func TestProcessingErrors(t *testing.T) {
	t.Run("NewProcessEmbeddingError", func(t *testing.T) {
		err := sentinelerrors.NewProcessEmbeddingError(32, "out of memory")
		if err.Code != sentinelerrors.ErrCodeProcessEmbeddingFailed {
			t.Errorf("unexpected code: %s", err.Code)
		}
		if err.Context["batch_size"] != 32 {
			t.Errorf("unexpected batch_size: %v", err.Context["batch_size"])
		}
		if !err.IsRetryable {
			t.Error("embedding errors should be retryable")
		}
	})

	t.Run("NewProcessClusteringError", func(t *testing.T) {
		err := sentinelerrors.NewProcessClusteringError(100, "insufficient samples")
		if err.Context["n_samples"] != 100 {
			t.Errorf("unexpected n_samples: %v", err.Context["n_samples"])
		}
	})

	t.Run("NewProcessModelLoadError", func(t *testing.T) {
		err := sentinelerrors.NewProcessModelLoadError("all-MiniLM-L6-v2", "network error")
		if err.Context["model_name"] != "all-MiniLM-L6-v2" {
			t.Errorf("unexpected model_name: %v", err.Context["model_name"])
		}
	})
}

func TestCommunicationErrors(t *testing.T) {
	t.Run("NewCommConnectionError", func(t *testing.T) {
		err := sentinelerrors.NewCommConnectionError("localhost:50051", "connection refused")
		if err.Code != sentinelerrors.ErrCodeCommConnectionFailed {
			t.Errorf("unexpected code: %s", err.Code)
		}
		if !err.IsRetryable {
			t.Error("connection errors should be retryable")
		}
	})

	t.Run("NewCommTimeoutError", func(t *testing.T) {
		err := sentinelerrors.NewCommTimeoutError("embedding", 30.0)
		if err.Context["operation"] != "embedding" {
			t.Errorf("unexpected operation: %v", err.Context["operation"])
		}
		if err.Context["timeout_seconds"] != 30.0 {
			t.Errorf("unexpected timeout_seconds: %v", err.Context["timeout_seconds"])
		}
	})
}

func TestStorageErrors(t *testing.T) {
	t.Run("NewStorageReadError", func(t *testing.T) {
		err := sentinelerrors.NewStorageReadError("/data/index.faiss", "file corrupted")
		if err.Code != sentinelerrors.ErrCodeStorageReadFailed {
			t.Errorf("unexpected code: %s", err.Code)
		}
		if !err.IsRetryable {
			t.Error("storage read errors should be retryable")
		}
	})

	t.Run("NewStorageWriteError", func(t *testing.T) {
		err := sentinelerrors.NewStorageWriteError("/data/index.faiss", "disk full")
		if err.Context["reason"] != "disk full" {
			t.Errorf("unexpected reason: %v", err.Context["reason"])
		}
	})
}

func TestHelperFunctions(t *testing.T) {
	t.Run("IsRetryableError returns true for retryable", func(t *testing.T) {
		err := sentinelerrors.NewIngestTimeoutError("source", 30.0)
		if !sentinelerrors.IsRetryableError(err) {
			t.Error("timeout should be retryable")
		}
	})

	t.Run("IsRetryableError returns false for non-retryable", func(t *testing.T) {
		err := sentinelerrors.NewIngestFileNotFoundError("/missing.log")
		if sentinelerrors.IsRetryableError(err) {
			t.Error("file not found should not be retryable")
		}
	})

	t.Run("IsRetryableError returns false for non-sentinel error", func(t *testing.T) {
		err := errors.New("regular error")
		if sentinelerrors.IsRetryableError(err) {
			t.Error("regular errors should not be retryable")
		}
	})

	t.Run("GetErrorCode returns code for sentinel error", func(t *testing.T) {
		err := sentinelerrors.NewConfigMissingError("/config.yaml")
		code := sentinelerrors.GetErrorCode(err)
		if code != sentinelerrors.ErrCodeConfigMissing {
			t.Errorf("unexpected code: %s", code)
		}
	})

	t.Run("GetErrorCode returns UNKNOWN for non-sentinel error", func(t *testing.T) {
		err := errors.New("regular error")
		code := sentinelerrors.GetErrorCode(err)
		if code != sentinelerrors.ErrCodeUnknown {
			t.Errorf("expected UNKNOWN, got: %s", code)
		}
	})
}
