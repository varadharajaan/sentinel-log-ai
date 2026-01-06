// Package errors provides structured error types for the sentinel-log-ai agent.
//
// This package follows Go best practices for error handling:
// - Sentinel errors for type checking with errors.Is()
// - Error wrapping with context using fmt.Errorf("%w", err)
// - Structured error types for detailed information
// - Error codes for machine-readable categorization
//
// Error code ranges:
// - 1xxx: Configuration errors
// - 2xxx: Ingestion errors
// - 3xxx: Processing errors
// - 4xxx: Storage errors
// - 5xxx: Communication errors
// - 9xxx: General errors
package errors

import (
	"errors"
	"fmt"
)

// ErrorCode represents a machine-readable error identifier.
type ErrorCode string

// Configuration error codes (1xxx)
const (
	ErrCodeConfigInvalid    ErrorCode = "SENTINEL_1001"
	ErrCodeConfigMissing    ErrorCode = "SENTINEL_1002"
	ErrCodeConfigValidation ErrorCode = "SENTINEL_1003"
)

// Ingestion error codes (2xxx)
const (
	ErrCodeIngestFileNotFound     ErrorCode = "SENTINEL_2001"
	ErrCodeIngestPermissionDenied ErrorCode = "SENTINEL_2002"
	ErrCodeIngestParseFailed      ErrorCode = "SENTINEL_2003"
	ErrCodeIngestEncodingError    ErrorCode = "SENTINEL_2004"
	ErrCodeIngestTimeout          ErrorCode = "SENTINEL_2005"
)

// Processing error codes (3xxx)
const (
	ErrCodeProcessEmbeddingFailed   ErrorCode = "SENTINEL_3001"
	ErrCodeProcessClusteringFailed  ErrorCode = "SENTINEL_3002"
	ErrCodeProcessNoveltyFailed     ErrorCode = "SENTINEL_3003"
	ErrCodeProcessModelLoadFailed   ErrorCode = "SENTINEL_3004"
	ErrCodeProcessResourceExhausted ErrorCode = "SENTINEL_3005"
)

// Storage error codes (4xxx)
const (
	ErrCodeStorageReadFailed       ErrorCode = "SENTINEL_4001"
	ErrCodeStorageWriteFailed      ErrorCode = "SENTINEL_4002"
	ErrCodeStorageIndexCorrupted   ErrorCode = "SENTINEL_4003"
	ErrCodeStorageCapacityExceeded ErrorCode = "SENTINEL_4004"
)

// Communication error codes (5xxx)
const (
	ErrCodeCommConnectionFailed ErrorCode = "SENTINEL_5001"
	ErrCodeCommTimeout          ErrorCode = "SENTINEL_5002"
	ErrCodeCommProtocolError    ErrorCode = "SENTINEL_5003"
	ErrCodeCommAuthFailed       ErrorCode = "SENTINEL_5004"
)

// General error codes (9xxx)
const (
	ErrCodeUnknown ErrorCode = "SENTINEL_9999"
)

// Sentinel errors for type checking with errors.Is()
var (
	// Configuration errors
	ErrConfigInvalid    = errors.New("invalid configuration")
	ErrConfigMissing    = errors.New("configuration not found")
	ErrConfigValidation = errors.New("configuration validation failed")

	// Ingestion errors
	ErrIngestFileNotFound     = errors.New("log file not found")
	ErrIngestPermissionDenied = errors.New("permission denied")
	ErrIngestParseFailed      = errors.New("log parsing failed")
	ErrIngestEncodingError    = errors.New("encoding error")
	ErrIngestTimeout          = errors.New("ingestion timeout")

	// Processing errors
	ErrProcessEmbeddingFailed   = errors.New("embedding generation failed")
	ErrProcessClusteringFailed  = errors.New("clustering failed")
	ErrProcessNoveltyFailed     = errors.New("novelty detection failed")
	ErrProcessModelLoadFailed   = errors.New("model loading failed")
	ErrProcessResourceExhausted = errors.New("resource exhausted")

	// Storage errors
	ErrStorageReadFailed       = errors.New("storage read failed")
	ErrStorageWriteFailed      = errors.New("storage write failed")
	ErrStorageIndexCorrupted   = errors.New("index corrupted")
	ErrStorageCapacityExceeded = errors.New("storage capacity exceeded")

	// Communication errors
	ErrCommConnectionFailed = errors.New("connection failed")
	ErrCommTimeout          = errors.New("communication timeout")
	ErrCommProtocolError    = errors.New("protocol error")
	ErrCommAuthFailed       = errors.New("authentication failed")
)

// SentinelError is the base error type with structured information.
type SentinelError struct {
	Code        ErrorCode
	Message     string
	Context     map[string]interface{}
	IsRetryable bool
	Cause       error
}

// Error implements the error interface.
func (e *SentinelError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("[%s] %s: %v", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

// Unwrap returns the underlying cause for errors.Is/As support.
func (e *SentinelError) Unwrap() error {
	return e.Cause
}

// Is checks if the target error matches this error's cause.
func (e *SentinelError) Is(target error) bool {
	if e.Cause != nil {
		return errors.Is(e.Cause, target)
	}
	return false
}

// WithContext adds context information to the error.
func (e *SentinelError) WithContext(key string, value interface{}) *SentinelError {
	if e.Context == nil {
		e.Context = make(map[string]interface{})
	}
	e.Context[key] = value
	return e
}

// ToMap converts the error to a map for structured logging.
func (e *SentinelError) ToMap() map[string]interface{} {
	m := map[string]interface{}{
		"error_code":   string(e.Code),
		"message":      e.Message,
		"is_retryable": e.IsRetryable,
	}
	if e.Context != nil {
		m["context"] = e.Context
	}
	if e.Cause != nil {
		m["cause"] = e.Cause.Error()
	}
	return m
}

// NewSentinelError creates a new SentinelError.
func NewSentinelError(code ErrorCode, message string, cause error) *SentinelError {
	return &SentinelError{
		Code:    code,
		Message: message,
		Cause:   cause,
		Context: make(map[string]interface{}),
	}
}

// Configuration Error constructors

// NewConfigInvalidError creates a configuration invalid error.
func NewConfigInvalidError(message string, cause error) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeConfigInvalid,
		Message:     message,
		Cause:       cause,
		IsRetryable: false,
		Context:     make(map[string]interface{}),
	}
}

// NewConfigMissingError creates a configuration missing error.
func NewConfigMissingError(path string) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeConfigMissing,
		Message:     fmt.Sprintf("configuration file not found: %s", path),
		Cause:       ErrConfigMissing,
		IsRetryable: false,
		Context: map[string]interface{}{
			"path": path,
		},
	}
}

// NewConfigValidationError creates a configuration validation error.
func NewConfigValidationError(field string, value interface{}, reason string) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeConfigValidation,
		Message:     fmt.Sprintf("validation failed for '%s': %s", field, reason),
		Cause:       ErrConfigValidation,
		IsRetryable: false,
		Context: map[string]interface{}{
			"field":  field,
			"value":  fmt.Sprintf("%v", value),
			"reason": reason,
		},
	}
}

// Ingestion Error constructors

// NewIngestFileNotFoundError creates a file not found error.
func NewIngestFileNotFoundError(path string) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeIngestFileNotFound,
		Message:     fmt.Sprintf("log file not found: %s", path),
		Cause:       ErrIngestFileNotFound,
		IsRetryable: false,
		Context: map[string]interface{}{
			"path": path,
		},
	}
}

// NewIngestPermissionDeniedError creates a permission denied error.
func NewIngestPermissionDeniedError(path string) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeIngestPermissionDenied,
		Message:     fmt.Sprintf("permission denied reading: %s", path),
		Cause:       ErrIngestPermissionDenied,
		IsRetryable: false,
		Context: map[string]interface{}{
			"path": path,
		},
	}
}

// NewIngestParseError creates a parse error.
func NewIngestParseError(line string, lineNumber int, reason string) *SentinelError {
	// Truncate long lines
	truncated := line
	if len(line) > 200 {
		truncated = line[:200] + "..."
	}
	return &SentinelError{
		Code:        ErrCodeIngestParseFailed,
		Message:     fmt.Sprintf("failed to parse log line %d: %s", lineNumber, reason),
		Cause:       ErrIngestParseFailed,
		IsRetryable: true,
		Context: map[string]interface{}{
			"line":        truncated,
			"line_number": lineNumber,
			"reason":      reason,
		},
	}
}

// NewIngestEncodingError creates an encoding error.
func NewIngestEncodingError(path string, encoding string) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeIngestEncodingError,
		Message:     fmt.Sprintf("encoding error reading %s with %s", path, encoding),
		Cause:       ErrIngestEncodingError,
		IsRetryable: false,
		Context: map[string]interface{}{
			"path":     path,
			"encoding": encoding,
		},
	}
}

// NewIngestTimeoutError creates a timeout error.
func NewIngestTimeoutError(source string, timeoutSeconds float64) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeIngestTimeout,
		Message:     fmt.Sprintf("ingestion from %s timed out after %.1fs", source, timeoutSeconds),
		Cause:       ErrIngestTimeout,
		IsRetryable: true,
		Context: map[string]interface{}{
			"source":          source,
			"timeout_seconds": timeoutSeconds,
		},
	}
}

// Processing Error constructors

// NewProcessEmbeddingError creates an embedding error.
func NewProcessEmbeddingError(batchSize int, reason string) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeProcessEmbeddingFailed,
		Message:     fmt.Sprintf("embedding generation failed: %s", reason),
		Cause:       ErrProcessEmbeddingFailed,
		IsRetryable: true,
		Context: map[string]interface{}{
			"batch_size": batchSize,
			"reason":     reason,
		},
	}
}

// NewProcessClusteringError creates a clustering error.
func NewProcessClusteringError(nSamples int, reason string) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeProcessClusteringFailed,
		Message:     fmt.Sprintf("clustering failed: %s", reason),
		Cause:       ErrProcessClusteringFailed,
		IsRetryable: false,
		Context: map[string]interface{}{
			"n_samples": nSamples,
			"reason":    reason,
		},
	}
}

// NewProcessModelLoadError creates a model loading error.
func NewProcessModelLoadError(modelName string, reason string) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeProcessModelLoadFailed,
		Message:     fmt.Sprintf("failed to load model '%s': %s", modelName, reason),
		Cause:       ErrProcessModelLoadFailed,
		IsRetryable: true,
		Context: map[string]interface{}{
			"model_name": modelName,
			"reason":     reason,
		},
	}
}

// Communication Error constructors

// NewCommConnectionError creates a connection error.
func NewCommConnectionError(address string, reason string) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeCommConnectionFailed,
		Message:     fmt.Sprintf("failed to connect to %s: %s", address, reason),
		Cause:       ErrCommConnectionFailed,
		IsRetryable: true,
		Context: map[string]interface{}{
			"address": address,
			"reason":  reason,
		},
	}
}

// NewCommTimeoutError creates a communication timeout error.
func NewCommTimeoutError(operation string, timeoutSeconds float64) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeCommTimeout,
		Message:     fmt.Sprintf("operation '%s' timed out after %.1fs", operation, timeoutSeconds),
		Cause:       ErrCommTimeout,
		IsRetryable: true,
		Context: map[string]interface{}{
			"operation":       operation,
			"timeout_seconds": timeoutSeconds,
		},
	}
}

// Storage Error constructors

// NewStorageReadError creates a storage read error.
func NewStorageReadError(path string, reason string) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeStorageReadFailed,
		Message:     fmt.Sprintf("failed to read from storage: %s", reason),
		Cause:       ErrStorageReadFailed,
		IsRetryable: true,
		Context: map[string]interface{}{
			"path":   path,
			"reason": reason,
		},
	}
}

// NewStorageWriteError creates a storage write error.
func NewStorageWriteError(path string, reason string) *SentinelError {
	return &SentinelError{
		Code:        ErrCodeStorageWriteFailed,
		Message:     fmt.Sprintf("failed to write to storage: %s", reason),
		Cause:       ErrStorageWriteFailed,
		IsRetryable: true,
		Context: map[string]interface{}{
			"path":   path,
			"reason": reason,
		},
	}
}

// IsRetryableError checks if an error is retryable.
func IsRetryableError(err error) bool {
	var sentinelErr *SentinelError
	if errors.As(err, &sentinelErr) {
		return sentinelErr.IsRetryable
	}
	return false
}

// GetErrorCode extracts the error code from an error.
func GetErrorCode(err error) ErrorCode {
	var sentinelErr *SentinelError
	if errors.As(err, &sentinelErr) {
		return sentinelErr.Code
	}
	return ErrCodeUnknown
}
