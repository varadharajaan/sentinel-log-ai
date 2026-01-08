// Package cmd provides the CLI commands for the sentinel-log-ai agent.
package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"sentinel-log-ai/internal/batch"
	"sentinel-log-ai/internal/grpcclient"
	"sentinel-log-ai/internal/ingestion"
	"sentinel-log-ai/internal/logging"
	"sentinel-log-ai/internal/models"
	"sentinel-log-ai/internal/parser"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

// IngestOptions holds options for the ingest command.
type IngestOptions struct {
	Path           string
	TailMode       bool
	Pattern        string
	MLServerAddr   string
	BatchSize      int
	BatchTimeout   time.Duration
	BufferSize     int
	OutputFile     string
	DryRun         bool
	ConnectTimeout time.Duration
}

// DefaultIngestOptions returns the default ingest options.
func DefaultIngestOptions() *IngestOptions {
	return &IngestOptions{
		TailMode:       false,
		Pattern:        "*",
		MLServerAddr:   "localhost:50051",
		BatchSize:      100,
		BatchTimeout:   5 * time.Second,
		BufferSize:     10000,
		OutputFile:     "",
		DryRun:         false,
		ConnectTimeout: 10 * time.Second,
	}
}

// IngestRunner handles the ingestion workflow.
type IngestRunner struct {
	options    *IngestOptions
	logger     *zap.Logger
	parser     *parser.Registry
	processor  *batch.Processor
	grpcClient *grpcclient.Client
}

// NewIngestRunner creates a new ingest runner with the given options.
func NewIngestRunner(opts *IngestOptions) (*IngestRunner, error) {
	if opts == nil {
		opts = DefaultIngestOptions()
	}

	// Set up logging
	logCfg := logging.DefaultConfig()
	logCfg.ConsoleFormat = "plain"
	if verbose {
		logCfg.Level = "debug"
	}
	if err := logging.Setup(logCfg); err != nil {
		return nil, fmt.Errorf("failed to setup logging: %w", err)
	}

	logger := logging.L().With(
		zap.String("command", "ingest"),
		zap.String("path", opts.Path),
	)

	// Create parser registry
	parserRegistry := parser.NewRegistry()

	runner := &IngestRunner{
		options: opts,
		logger:  logger,
		parser:  parserRegistry,
	}

	return runner, nil
}

// Run executes the ingestion workflow.
func (r *IngestRunner) Run(ctx context.Context) error {
	r.logger.Info("ingestion_starting",
		zap.Bool("tail_mode", r.options.TailMode),
		zap.String("ml_server", r.options.MLServerAddr),
		zap.Int("batch_size", r.options.BatchSize),
	)

	// Set up context with cancellation
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Handle signals for graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		r.logger.Info("received_signal", zap.String("signal", sig.String()))
		cancel()
	}()

	// Initialize gRPC client if not in dry-run mode
	if !r.options.DryRun {
		grpcCfg := &grpcclient.Config{
			Address:        r.options.MLServerAddr,
			ConnectTimeout: r.options.ConnectTimeout,
			RequestTimeout: 120 * time.Second, // 2 minutes for initial model loading
			MaxRetries:     3,
			RetryBackoff:   100 * time.Millisecond,
			MaxBackoff:     5 * time.Second,
			MaxMessageSize: 100 * 1024 * 1024, // 100MB to support large batches
			Logger:         r.logger,
		}

		client, err := grpcclient.NewClient(grpcCfg)
		if err != nil {
			return fmt.Errorf("failed to create gRPC client: %w", err)
		}
		r.grpcClient = client
		defer func() { _ = r.grpcClient.Close() }()
	}

	// Create batch handler
	handler := r.createBatchHandler()

	// Initialize batch processor
	batchCfg := &batch.Config{
		MaxBatchSize: r.options.BatchSize,
		MaxWaitTime:  r.options.BatchTimeout,
		BufferSize:   r.options.BufferSize,
		FlushTimeout: 30 * time.Second,
		Logger:       r.logger,
	}

	processor, err := batch.NewProcessor(batchCfg, handler)
	if err != nil {
		return fmt.Errorf("failed to create batch processor: %w", err)
	}
	r.processor = processor
	defer func() {
		if err := r.processor.Close(); err != nil {
			r.logger.Error("batch_processor_close_error", zap.Error(err))
		}
	}()

	// Create source
	source, err := r.createSource()
	if err != nil {
		return fmt.Errorf("failed to create source: %w", err)
	}
	defer func() { _ = source.Close() }()

	// Create record channel
	recordCh := make(chan *models.LogRecord, 1000)

	// Start reading from source
	readErrCh := make(chan error, 1)
	go func() {
		readErrCh <- source.Read(ctx, recordCh)
		close(recordCh)
	}()

	// Process records
	recordCount := 0
	startTime := time.Now()

	for record := range recordCh {
		// Parse the record
		parsedRecord := r.parser.Parse(record.Raw, record.Source)
		if parsedRecord.Timestamp == nil {
			parsedRecord.Timestamp = record.Timestamp
		}

		// Add to batch processor
		if err := r.processor.Add(parsedRecord); err != nil {
			r.logger.Warn("failed_to_add_record", zap.Error(err))
			continue
		}
		recordCount++

		// Log progress periodically
		if recordCount%1000 == 0 {
			r.logger.Info("ingestion_progress",
				zap.Int("records_processed", recordCount),
				zap.Duration("elapsed", time.Since(startTime)),
			)
		}
	}

	// Wait for source reading to complete
	if err := <-readErrCh; err != nil && ctx.Err() == nil {
		return fmt.Errorf("source read error: %w", err)
	}

	// Final flush
	if err := r.processor.Flush(context.Background()); err != nil {
		r.logger.Error("final_flush_failed", zap.Error(err))
	}

	// Log completion
	metrics := r.processor.GetMetrics()
	r.logger.Info("ingestion_complete",
		zap.Int("total_records", recordCount),
		zap.Int64("processed", metrics.TotalProcessed),
		zap.Int64("dropped", metrics.TotalDropped),
		zap.Int64("batches", metrics.TotalBatches),
		zap.Duration("duration", time.Since(startTime)),
	)

	return nil
}

// createSource creates the appropriate log source based on the path.
func (r *IngestRunner) createSource() (ingestion.Source, error) {
	path := r.options.Path

	// Handle stdin
	if path == "-" {
		r.logger.Info("reading_from_stdin")
		return ingestion.NewStdinSource(r.logger), nil
	}

	// Check if path exists
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("path not found: %s: %w", path, err)
	}

	// Handle directory
	if info.IsDir() {
		return r.createDirectorySource(path)
	}

	// Handle file
	r.logger.Info("reading_from_file",
		zap.String("path", path),
		zap.Bool("tail", r.options.TailMode),
	)
	return ingestion.NewFileSource(path, r.options.TailMode, r.logger), nil
}

// createDirectorySource creates a source that reads from all matching files in a directory.
func (r *IngestRunner) createDirectorySource(dir string) (ingestion.Source, error) {
	pattern := filepath.Join(dir, r.options.Pattern)
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("invalid pattern: %s: %w", r.options.Pattern, err)
	}

	if len(matches) == 0 {
		return nil, fmt.Errorf("no files match pattern: %s", pattern)
	}

	r.logger.Info("reading_from_directory",
		zap.String("directory", dir),
		zap.String("pattern", r.options.Pattern),
		zap.Int("file_count", len(matches)),
	)

	// For now, just use the first matching file
	// TODO: Implement multi-file source
	return ingestion.NewFileSource(matches[0], r.options.TailMode, r.logger), nil
}

// createBatchHandler creates the appropriate batch handler.
func (r *IngestRunner) createBatchHandler() batch.BatchHandler {
	if r.options.DryRun {
		r.logger.Info("dry_run_mode_enabled")
		return batch.BatchHandlerFunc(r.dryRunHandler)
	}

	if r.grpcClient != nil {
		return grpcclient.NewBatchHandler(r.grpcClient)
	}

	return batch.BatchHandlerFunc(r.dryRunHandler)
}

// dryRunHandler logs batches without sending them.
func (r *IngestRunner) dryRunHandler(ctx context.Context, records []*models.LogRecord) (int, error) {
	r.logger.Debug("dry_run_batch",
		zap.Int("batch_size", len(records)),
	)
	return len(records), nil
}

// Close releases resources.
func (r *IngestRunner) Close() error {
	if r.processor != nil {
		if err := r.processor.Close(); err != nil {
			return err
		}
	}
	if r.grpcClient != nil {
		if err := r.grpcClient.Close(); err != nil {
			return err
		}
	}
	return logging.Close()
}

// RunIngestCommand executes the ingest command with the given options.
func RunIngestCommand(opts *IngestOptions) error {
	runner, err := NewIngestRunner(opts)
	if err != nil {
		return err
	}
	defer func() { _ = runner.Close() }()

	return runner.Run(context.Background())
}

// setupIngestCmd configures the ingest command.
func setupIngestCmd() *cobra.Command {
	opts := DefaultIngestOptions()

	cmd := &cobra.Command{
		Use:   "ingest [path]",
		Short: "Ingest logs from a file or directory",
		Long: `Ingest logs from the specified file or directory.
Supports batch mode (read entire file) and tail mode (follow new lines).

Examples:
  sentinel-log-ai ingest /var/log/syslog
  sentinel-log-ai ingest /var/log/nginx/ --pattern "*.log"
  sentinel-log-ai ingest /var/log/app.log --tail
  sentinel-log-ai ingest - (read from stdin)
  sentinel-log-ai ingest /var/log/app.log --dry-run`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.Path = args[0]
			return RunIngestCommand(opts)
		},
	}

	cmd.Flags().BoolVar(&opts.TailMode, "tail", false, "follow file for new lines (like tail -f)")
	cmd.Flags().StringVar(&opts.Pattern, "pattern", "*", "glob pattern for files in directory mode")
	cmd.Flags().StringVar(&opts.MLServerAddr, "ml-addr", "localhost:50051", "ML engine gRPC address")
	cmd.Flags().IntVar(&opts.BatchSize, "batch-size", 100, "number of records per batch")
	cmd.Flags().DurationVar(&opts.BatchTimeout, "batch-timeout", 5*time.Second, "max time before flushing batch")
	cmd.Flags().IntVar(&opts.BufferSize, "buffer-size", 10000, "internal buffer size")
	cmd.Flags().StringVar(&opts.OutputFile, "output", "", "write processed records to file (JSONL)")
	cmd.Flags().BoolVar(&opts.DryRun, "dry-run", false, "process logs without sending to ML server")
	cmd.Flags().DurationVar(&opts.ConnectTimeout, "connect-timeout", 10*time.Second, "ML server connection timeout")

	return cmd
}
