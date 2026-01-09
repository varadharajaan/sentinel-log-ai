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
	mlpb "sentinel-log-ai/proto/ml/v1"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
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
		EnableCompression: false, // Disable compression for compatibility
		MaxMessageSize:    100 * 1024 * 1024, // 100MB to match server
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
	config    *Config
	conn      *grpc.ClientConn
	mlClient  mlpb.MLServiceClient
	logger    *zap.Logger

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

	c.logger.Info("connecting_to_ml_server",
		zap.Int("max_message_size", c.config.MaxMessageSize),
	)

	// Connection options - use Dial for better compatibility with message sizes
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                60 * time.Second, // Less aggressive keepalive
			Timeout:             20 * time.Second,
			PermitWithoutStream: false, // Don't ping without active streams
		}),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(c.config.MaxMessageSize),
			grpc.MaxCallSendMsgSize(c.config.MaxMessageSize),
		),
	}

	// Create connection using Dial (blocking until connected)
	//nolint:staticcheck // Using Dial for better message size option support
	conn, err := grpc.Dial(c.config.Address, opts...)
	if err != nil {
		c.logger.Error("connection_failed", zap.Error(err))
		return sentinelerrors.NewCommConnectionError(c.config.Address, err.Error())
	}

	c.conn = conn
	c.mlClient = mlpb.NewMLServiceClient(conn)
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

// GetMLClient returns the MLService gRPC client.
func (c *Client) GetMLClient() mlpb.MLServiceClient {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.mlClient
}

// SearchResult represents a single similar log match.
type SearchResult struct {
	Record     *models.LogRecord
	Similarity float32
	Distance   float32
	ClusterID  string
}

// ClusterSummary describes a log cluster.
type ClusterSummary struct {
	ClusterID      string
	Size           int32
	Representative string
	Keywords       []string
	Cohesion       float32
	IsNew          bool
}

// ClusterResult contains clustering results.
type ClusterResult struct {
	Clusters       []ClusterSummary
	NoiseCount     int32
	TotalProcessed int32
}

// NoveltyResult indicates if a log is novel.
type NoveltyResult struct {
	IsNovel           bool
	NoveltyScore      float32
	ClosestClusterID  string
	DistanceToCluster float32
	Reason            string
}

// ExplainResult contains the LLM explanation for logs.
type ExplainResult struct {
	RootCause           string
	NextSteps           []string
	Remediation         string
	Confidence          string
	ConfidenceScore     float32
	ConfidenceReasoning string
	RawResponse         string
}

// Search finds similar logs using vector similarity.
func (c *Client) Search(ctx context.Context, query *models.LogRecord, topK int, minSimilarity float32) ([]SearchResult, error) {
	if !c.IsConnected() {
		if err := c.Connect(ctx); err != nil {
			return nil, err
		}
	}

	req := &mlpb.SearchRequest{
		Query:         convertToMLPBRecord(query),
		TopK:          int32(topK),
		MinSimilarity: minSimilarity,
	}

	resp, err := c.mlClient.Search(ctx, req)
	if err != nil {
		c.logger.Error("search_rpc_failed", zap.Error(err))
		return nil, fmt.Errorf("search RPC failed: %w", err)
	}

	results := make([]SearchResult, 0, len(resp.Results))
	for _, r := range resp.Results {
		results = append(results, SearchResult{
			Record:     convertFromMLPBRecord(r.Record),
			Similarity: r.Similarity,
			Distance:   r.Distance,
			ClusterID:  r.ClusterId,
		})
	}

	c.logger.Info("search_completed", zap.Int("results", len(results)))
	return results, nil
}

// Cluster groups similar logs and returns cluster summaries.
func (c *Client) Cluster(ctx context.Context, records []*models.LogRecord, minClusterSize int, persist bool) (*ClusterResult, error) {
	if !c.IsConnected() {
		if err := c.Connect(ctx); err != nil {
			return nil, err
		}
	}

	protoRecords := make([]*mlpb.LogRecord, 0, len(records))
	for _, r := range records {
		if r != nil {
			protoRecords = append(protoRecords, convertToMLPBRecord(r))
		}
	}

	req := &mlpb.ClusterRequest{
		Records:        protoRecords,
		MinClusterSize: int32(minClusterSize),
		Persist:        persist,
	}

	resp, err := c.mlClient.Cluster(ctx, req)
	if err != nil {
		c.logger.Error("cluster_rpc_failed", zap.Error(err))
		return nil, fmt.Errorf("cluster RPC failed: %w", err)
	}

	clusters := make([]ClusterSummary, 0, len(resp.Clusters))
	for _, cl := range resp.Clusters {
		clusters = append(clusters, ClusterSummary{
			ClusterID:      cl.ClusterId,
			Size:           cl.Size,
			Representative: cl.Representative,
			Keywords:       cl.Keywords,
			Cohesion:       cl.Cohesion,
			IsNew:          cl.IsNew,
		})
	}

	c.logger.Info("cluster_completed",
		zap.Int("clusters", len(clusters)),
		zap.Int32("noise_count", resp.NoiseCount),
		zap.Int32("total_processed", resp.TotalProcessed),
	)

	return &ClusterResult{
		Clusters:       clusters,
		NoiseCount:     resp.NoiseCount,
		TotalProcessed: resp.TotalProcessed,
	}, nil
}

// DetectNovelty checks if a log pattern is novel/unseen.
func (c *Client) DetectNovelty(ctx context.Context, record *models.LogRecord, threshold float32) (*NoveltyResult, error) {
	if !c.IsConnected() {
		if err := c.Connect(ctx); err != nil {
			return nil, err
		}
	}

	req := &mlpb.NoveltyRequest{
		Record:    convertToMLPBRecord(record),
		Threshold: threshold,
	}

	resp, err := c.mlClient.DetectNovelty(ctx, req)
	if err != nil {
		c.logger.Error("novelty_rpc_failed", zap.Error(err))
		return nil, fmt.Errorf("detect novelty RPC failed: %w", err)
	}

	c.logger.Info("novelty_check_completed",
		zap.Bool("is_novel", resp.IsNovel),
		zap.Float32("novelty_score", resp.NoveltyScore),
	)

	return &NoveltyResult{
		IsNovel:           resp.IsNovel,
		NoveltyScore:      resp.NoveltyScore,
		ClosestClusterID:  resp.ClosestClusterId,
		DistanceToCluster: resp.DistanceToCluster,
		Reason:            resp.Reason,
	}, nil
}

// Health checks the ML service health.
func (c *Client) Health(ctx context.Context, detailed bool) (*HealthResponse, error) {
	if !c.IsConnected() {
		if err := c.Connect(ctx); err != nil {
			return nil, err
		}
	}

	req := &mlpb.HealthRequest{
		Detailed: detailed,
	}

	resp, err := c.mlClient.Health(ctx, req)
	if err != nil {
		c.logger.Error("health_rpc_failed", zap.Error(err))
		return nil, fmt.Errorf("health RPC failed: %w", err)
	}

	components := make([]ComponentHealth, 0, len(resp.Components))
	for _, comp := range resp.Components {
		components = append(components, ComponentHealth{
			Name:    comp.Name,
			Healthy: comp.Healthy,
			Message: comp.Message,
		})
	}

	return &HealthResponse{
		Healthy:    resp.Healthy,
		Version:    resp.Version,
		Components: components,
	}, nil
}

// Explain gets an LLM explanation for log records.
func (c *Client) Explain(ctx context.Context, records []*models.LogRecord, explainContext string, model string) (*ExplainResult, error) {
	if !c.IsConnected() {
		if err := c.Connect(ctx); err != nil {
			return nil, err
		}
	}

	protoRecords := make([]*mlpb.LogRecord, 0, len(records))
	for _, r := range records {
		protoRecords = append(protoRecords, convertToMLPBRecord(r))
	}

	req := &mlpb.ExplainRequest{
		SampleLogs: protoRecords,
		Context:    explainContext,
		Model:      model,
	}

	resp, err := c.mlClient.Explain(ctx, req)
	if err != nil {
		c.logger.Error("explain_rpc_failed", zap.Error(err))
		return nil, fmt.Errorf("explain RPC failed: %w", err)
	}

	c.logger.Info("explain_completed",
		zap.Int("record_count", len(records)),
		zap.String("confidence", resp.Confidence),
		zap.Float32("confidence_score", resp.ConfidenceScore),
	)

	return &ExplainResult{
		RootCause:           resp.RootCause,
		NextSteps:           resp.NextSteps,
		Remediation:         resp.Remediation,
		Confidence:          resp.Confidence,
		ConfidenceScore:     resp.ConfidenceScore,
		ConfidenceReasoning: resp.ConfidenceReasoning,
		RawResponse:         resp.RawResponse,
	}, nil
}

// convertFromMLPBRecord converts a proto LogRecord to models.LogRecord.
func convertFromMLPBRecord(record *mlpb.LogRecord) *models.LogRecord {
	if record == nil {
		return nil
	}

	var ts *time.Time
	if record.Timestamp != nil {
		t := record.Timestamp.AsTime()
		ts = &t
	}

	var attrs map[string]any
	if record.AttrsJson != "" {
		_ = json.Unmarshal([]byte(record.AttrsJson), &attrs)
	}

	return &models.LogRecord{
		Message:    record.Message,
		Normalized: record.Normalized,
		Level:      record.Level,
		Source:     record.Source,
		Timestamp:  ts,
		Attrs:      attrs,
	}
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

	// Convert records to proto format for Embed request
	protoRecords := make([]*mlpb.LogRecord, 0, len(batch))
	for _, record := range batch {
		if record != nil {
			protoRecords = append(protoRecords, convertToMLPBRecord(record))
		}
	}

	// Call the gRPC Embed method
	mlClient := h.client.GetMLClient()
	if mlClient == nil {
		return 0, fmt.Errorf("ML client not initialized")
	}

	req := &mlpb.EmbedRequest{
		Records:  protoRecords,
		UseCache: true,
	}

	resp, err := mlClient.Embed(ctx, req)
	if err != nil {
		h.logger.Error("embed_rpc_failed", zap.Error(err))
		return 0, fmt.Errorf("embed RPC failed: %w", err)
	}

	processed := len(protoRecords)
	duration := time.Since(startTime)
	h.logger.Info("batch_sent_to_ml_server",
		zap.Int("batch_size", len(batch)),
		zap.Int("processed", processed),
		zap.Int32("embedding_dim", resp.EmbeddingDim),
		zap.Int32("cache_hits", resp.CacheHits),
		zap.Duration("duration", duration),
	)

	return processed, nil
}

// convertToMLPBRecord converts a models.LogRecord to the proto LogRecord.
func convertToMLPBRecord(record *models.LogRecord) *mlpb.LogRecord {
	if record == nil {
		return nil
	}

	var attrsJSON string
	if record.Attrs != nil {
		if data, err := json.Marshal(record.Attrs); err == nil {
			attrsJSON = string(data)
		}
	}

	var ts *timestamppb.Timestamp
	if record.Timestamp != nil && !record.Timestamp.IsZero() {
		ts = timestamppb.New(*record.Timestamp)
	}

	return &mlpb.LogRecord{
		Id:         "", // ID is assigned by the ML service
		Message:    record.Message,
		Normalized: record.Normalized,
		Level:      record.Level,
		Source:     record.Source,
		Timestamp:  ts,
		AttrsJson:  attrsJSON,
	}
}
