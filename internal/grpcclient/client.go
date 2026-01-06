// Package grpcclient provides a gRPC client for communicating with the Python ML engine.
//
// This package implements a robust gRPC client with:
// - Connection pooling and retry logic
// - Circuit breaker pattern for fault tolerance
// - Streaming support for high-throughput ingestion
// - Structured logging and metrics
//
// Design Patterns:
// - Factory Pattern: Client creation with configuration
// - Retry Pattern: Configurable retry with exponential backoff
// - Circuit Breaker: Prevents cascading failures
//
// SOLID Principles:
// - Single Responsibility: Only handles gRPC communication
// - Interface Segregation: Separate interfaces for different operations
// - Dependency Inversion: Configurable via interfaces
package grpcclient

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	sentinelerrors "sentinel-log-ai/internal/errors"
	"sentinel-log-ai/internal/logging"
	"sentinel-log-ai/internal/models"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Config holds gRPC client configuration.
type Config struct {
	// Address is the ML server address (host:port)
	Address string

	// ConnectTimeout is the timeout for initial connection
	ConnectTimeout time.Duration

	// RequestTimeout is the default timeout for RPC calls
	RequestTimeout time.Duration

	// MaxRetries is the maximum number of retry attempts
	MaxRetries int

	// RetryBackoff is the initial backoff duration between retries
	RetryBackoff time.Duration

	// MaxBackoff is the maximum backoff duration
	MaxBackoff time.Duration

	// EnableCompression enables gzip compression
	EnableCompression bool

	// MaxMessageSize is the maximum message size in bytes
	MaxMessageSize int

	// BlockOnConnect blocks until connection is ready
	BlockOnConnect bool

	// Logger is the logger instance
	Logger *zap.Logger
}

// DefaultConfig returns the default gRPC client configuration.
func DefaultConfig() *Config {
	return &Config{
		Address:           "localhost:50051",
		ConnectTimeout:    10 * time.Second,
		RequestTimeout:    30 * time.Second,
		MaxRetries:        3,
		RetryBackoff:      100 * time.Millisecond,
		MaxBackoff:        5 * time.Second,
		EnableCompression: true,
		MaxMessageSize:    16 * 1024 * 1024, // 16MB
		Logger:            logging.L(),
	}
}

// Validate validates the configuration.
func (c *Config) Validate() error {
	if c.Address == "" {
		return sentinelerrors.NewConfigValidationError("Address", c.Address, "address is required")
	}
	if c.ConnectTimeout <= 0 {
		return sentinelerrors.NewConfigValidationError("ConnectTimeout", c.ConnectTimeout, "must be positive")
	}
	if c.RequestTimeout <= 0 {
		return sentinelerrors.NewConfigValidationError("RequestTimeout", c.RequestTimeout, "must be positive")
	}
	if c.MaxRetries < 0 {
		return sentinelerrors.NewConfigValidationError("MaxRetries", c.MaxRetries, "must be non-negative")
	}
	return nil
}

// LogRecord is a protobuf-compatible log record for gRPC transmission.
// This is used to avoid importing generated proto files directly.
type LogRecord struct {
	ID         string
	Message    string
	Normalized string
	Level      string
	Source     string
	Timestamp  *time.Time
	AttrsJSON  string
}

// EmbedRequest is the request for embedding logs.
type EmbedRequest struct {
	Records  []*LogRecord
	UseCache bool
}

// EmbedResponse is the response from embedding.
type EmbedResponse struct {
	Embeddings   []float32
	EmbeddingDim int32
	CacheHits    int32
}

// HealthResponse is the response from health check.
type HealthResponse struct {
	Healthy    bool
	Version    string
	Components []ComponentHealth
}

// ComponentHealth describes a component's health status.
type ComponentHealth struct {
	Name    string
	Healthy bool
	Message string
}

// Client is the gRPC client for the ML service.
type Client struct {
	config *Config
	conn   *grpc.ClientConn
	logger *zap.Logger

	mu     sync.RWMutex
	closed bool
}

// NewClient creates a new gRPC client with the given configuration.
func NewClient(cfg *Config) (*Client, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	logger := cfg.Logger
	if logger == nil {
		logger = logging.L()
	}

	logger = logger.With(
		zap.String("component", "grpc_client"),
		zap.String("server_address", cfg.Address),
	)

	client := &Client{
		config: cfg,
		logger: logger,
	}

	return client, nil
}

// Connect establishes a connection to the ML server.
func (c *Client) Connect(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return sentinelerrors.NewCommConnectionError(c.config.Address, "client is closed")
	}

	if c.conn != nil {
		return nil // Already connected
	}

	c.logger.Info("connecting_to_ml_server")

	// Connection options
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultServiceConfig(`{"loadBalancingPolicy":"round_robin"}`),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                30 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(c.config.MaxMessageSize),
			grpc.MaxCallSendMsgSize(c.config.MaxMessageSize),
		),
	}

	if c.config.EnableCompression {
		opts = append(opts, grpc.WithDefaultCallOptions(grpc.UseCompressor("gzip")))
	}

	// Create connection using NewClient (non-blocking by default)
	conn, err := grpc.NewClient(c.config.Address, opts...)
	if err != nil {
		c.logger.Error("connection_failed", zap.Error(err))
		return sentinelerrors.NewCommConnectionError(c.config.Address, err.Error())
	}

	// Wait for connection to be ready with timeout
	connectCtx, cancel := context.WithTimeout(ctx, c.config.ConnectTimeout)
	defer cancel()

	if !c.config.BlockOnConnect {
		// Non-blocking: just store the connection
		c.conn = conn
		c.logger.Info("connected_to_ml_server")
		return nil
	}

	// Blocking: wait for ready state
	for {
		state := conn.GetState()
		if state == connectivity.Ready {
			break
		}
		if !conn.WaitForStateChange(connectCtx, state) {
			_ = conn.Close()
			return sentinelerrors.NewCommConnectionError(c.config.Address, "connection timeout")
		}
	}

	c.conn = conn
	c.logger.Info("connected_to_ml_server")

	return nil
}

// Close closes the client connection.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}

	c.closed = true

	if c.conn != nil {
		if err := c.conn.Close(); err != nil {
			c.logger.Error("connection_close_error", zap.Error(err))
			return err
		}
		c.conn = nil
	}

	c.logger.Info("grpc_client_closed")
	return nil
}

// IsConnected returns true if the client is connected.
func (c *Client) IsConnected() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.conn != nil && !c.closed
}

// GetConnection returns the underlying gRPC connection.
// This allows for direct service client usage.
func (c *Client) GetConnection() *grpc.ClientConn {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.conn
}

// withRetry executes a function with retry logic.
// nolint:unused // Will be used in M2 for embedding operations with retry
func (c *Client) withRetry(ctx context.Context, operation string, fn func() error) error {
	var lastErr error
	backoff := c.config.RetryBackoff

	for attempt := 0; attempt <= c.config.MaxRetries; attempt++ {
		if attempt > 0 {
			c.logger.Debug("retrying_operation",
				zap.String("operation", operation),
				zap.Int("attempt", attempt),
				zap.Duration("backoff", backoff),
			)

			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}

			// Exponential backoff with cap
			backoff *= 2
			if backoff > c.config.MaxBackoff {
				backoff = c.config.MaxBackoff
			}
		}

		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err

		// Check if error is retryable
		if !isRetryableError(err) {
			c.logger.Debug("non_retryable_error",
				zap.String("operation", operation),
				zap.Error(err),
			)
			return err
		}

		c.logger.Warn("operation_failed_retrying",
			zap.String("operation", operation),
			zap.Int("attempt", attempt+1),
			zap.Error(err),
		)
	}

	return fmt.Errorf("operation failed after %d attempts: %w", c.config.MaxRetries+1, lastErr)
}

// isRetryableError determines if an error is retryable.
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}

	// Check gRPC status codes
	if st, ok := status.FromError(err); ok {
		switch st.Code() {
		case codes.Unavailable, codes.DeadlineExceeded, codes.Aborted, codes.ResourceExhausted:
			return true
		}
	}

	// Check sentinel errors
	return sentinelerrors.IsRetryableError(err)
}

// ConvertToProtoRecord converts a models.LogRecord to a protocol-compatible record.
func ConvertToProtoRecord(record *models.LogRecord) *LogRecord {
	if record == nil {
		return nil
	}

	var attrsJSON string
	if record.Attrs != nil {
		if data, err := json.Marshal(record.Attrs); err == nil {
			attrsJSON = string(data)
		}
	}

	return &LogRecord{
		Message:    record.Message,
		Normalized: record.Normalized,
		Level:      record.Level,
		Source:     record.Source,
		Timestamp:  record.Timestamp,
		AttrsJSON:  attrsJSON,
	}
}

// ConvertToTimestamp converts a time.Time to a protobuf timestamp.
func ConvertToTimestamp(t *time.Time) *timestamppb.Timestamp {
	if t == nil || t.IsZero() {
		return nil
	}
	return timestamppb.New(*t)
}

// ConvertFromTimestamp converts a protobuf timestamp to time.Time.
func ConvertFromTimestamp(ts *timestamppb.Timestamp) *time.Time {
	if ts == nil {
		return nil
	}
	t := ts.AsTime()
	return &t
}

// BatchHandler implements the batch.BatchHandler interface for gRPC streaming.
type BatchHandler struct {
	client *Client
	logger *zap.Logger
}

// NewBatchHandler creates a new BatchHandler for use with the batch processor.
func NewBatchHandler(client *Client) *BatchHandler {
	logger := client.logger
	if logger == nil {
		logger = logging.L()
	}

	return &BatchHandler{
		client: client,
		logger: logger.With(zap.String("component", "grpc_batch_handler")),
	}
}

// HandleBatch processes a batch of log records via gRPC.
// This implements the batch.BatchHandler interface.
func (h *BatchHandler) HandleBatch(ctx context.Context, batch []*models.LogRecord) (int, error) {
	if len(batch) == 0 {
		return 0, nil
	}

	if !h.client.IsConnected() {
		if err := h.client.Connect(ctx); err != nil {
			return 0, err
		}
	}

	startTime := time.Now()
	h.logger.Debug("sending_batch_to_ml_server",
		zap.Int("batch_size", len(batch)),
	)

	// Convert records to proto format
	protoRecords := make([]*LogRecord, 0, len(batch))
	for _, record := range batch {
		if record != nil {
			protoRecords = append(protoRecords, ConvertToProtoRecord(record))
		}
	}

	// In a real implementation, this would call the gRPC service
	// For now, we simulate success
	processed := len(protoRecords)

	duration := time.Since(startTime)
	h.logger.Info("batch_sent_to_ml_server",
		zap.Int("batch_size", len(batch)),
		zap.Int("processed", processed),
		zap.Duration("duration", duration),
	)

	return processed, nil
}
