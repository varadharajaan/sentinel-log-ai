// Package batch provides a high-performance batch processor for log ingestion.
//
// The batch processor aggregates log records before sending them to the ML engine,
// reducing gRPC overhead and improving throughput. It supports configurable batch
// sizes, timeouts, and back-pressure handling.
//
// Design Patterns:
// - Strategy Pattern: Configurable flush strategies (size-based, time-based)
// - Observer Pattern: Batch lifecycle hooks for monitoring
// - Builder Pattern: Fluent configuration
//
// SOLID Principles:
// - Single Responsibility: Only handles batching logic
// - Open/Closed: Extensible via BatchHandler interface
// - Interface Segregation: Minimal interfaces
// - Dependency Inversion: Depends on abstractions (BatchHandler)
package batch

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	sentinelerrors "sentinel-log-ai/internal/errors"
	"sentinel-log-ai/internal/logging"
	"sentinel-log-ai/internal/models"

	"go.uber.org/zap"
)

// Common errors for batch processing.
var (
	// ErrProcessorClosed is returned when operations are attempted on a closed processor.
	ErrProcessorClosed = errors.New("batch processor is closed")

	// ErrBatchFull is returned when the batch buffer is at capacity.
	ErrBatchFull = errors.New("batch buffer is full")

	// ErrHandlerFailed is returned when the batch handler fails.
	ErrHandlerFailed = errors.New("batch handler failed")
)

// BatchHandler is the interface for processing batches of log records.
// Implementations may send batches to gRPC, write to disk, or perform other operations.
type BatchHandler interface {
	// HandleBatch processes a batch of log records.
	// Returns the number of successfully processed records and any error.
	HandleBatch(ctx context.Context, batch []*models.LogRecord) (processed int, err error)
}

// BatchHandlerFunc is a function adapter for BatchHandler.
type BatchHandlerFunc func(ctx context.Context, batch []*models.LogRecord) (int, error)

// HandleBatch implements BatchHandler.
func (f BatchHandlerFunc) HandleBatch(ctx context.Context, batch []*models.LogRecord) (int, error) {
	return f(ctx, batch)
}

// Metrics holds batch processor statistics.
type Metrics struct {
	// TotalRecords is the total number of records added.
	TotalRecords int64

	// TotalBatches is the total number of batches flushed.
	TotalBatches int64

	// TotalProcessed is the total number of successfully processed records.
	TotalProcessed int64

	// TotalDropped is the total number of dropped records due to errors.
	TotalDropped int64

	// LastFlushTime is the timestamp of the last flush.
	LastFlushTime time.Time

	// LastFlushDuration is the duration of the last flush.
	LastFlushDuration time.Duration

	// LastBatchSize is the size of the last batch.
	LastBatchSize int
}

// Config holds batch processor configuration.
type Config struct {
	// MaxBatchSize is the maximum number of records before auto-flush.
	// Default: 100
	MaxBatchSize int

	// MaxWaitTime is the maximum time to wait before flushing.
	// Default: 5 seconds
	MaxWaitTime time.Duration

	// BufferSize is the capacity of the internal record buffer.
	// Default: 10000
	BufferSize int

	// FlushTimeout is the timeout for flushing a batch to the handler.
	// Default: 30 seconds
	FlushTimeout time.Duration

	// DropOnFull determines behavior when buffer is full.
	// If true, new records are dropped. If false, Add() blocks.
	// Default: false
	DropOnFull bool

	// Logger is the logger instance.
	Logger *zap.Logger
}

// DefaultConfig returns the default batch processor configuration.
func DefaultConfig() *Config {
	return &Config{
		MaxBatchSize: 100,
		MaxWaitTime:  5 * time.Second,
		BufferSize:   10000,
		FlushTimeout: 30 * time.Second,
		DropOnFull:   false,
		Logger:       logging.L(),
	}
}

// Validate validates the configuration.
func (c *Config) Validate() error {
	if c.MaxBatchSize <= 0 {
		return sentinelerrors.NewConfigValidationError("MaxBatchSize", c.MaxBatchSize, "must be positive")
	}
	if c.MaxWaitTime <= 0 {
		return sentinelerrors.NewConfigValidationError("MaxWaitTime", c.MaxWaitTime, "must be positive")
	}
	if c.BufferSize <= 0 {
		return sentinelerrors.NewConfigValidationError("BufferSize", c.BufferSize, "must be positive")
	}
	if c.FlushTimeout <= 0 {
		return sentinelerrors.NewConfigValidationError("FlushTimeout", c.FlushTimeout, "must be positive")
	}
	return nil
}

// Processor is a high-performance batch processor for log records.
// It accumulates records and flushes them to a handler when either the batch
// size or time threshold is reached.
type Processor struct {
	config  *Config
	handler BatchHandler
	logger  *zap.Logger

	// Input channel for records
	recordCh chan *models.LogRecord

	// Current batch being accumulated
	batch   []*models.LogRecord
	batchMu sync.Mutex

	// Lifecycle management
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	closed    atomic.Bool
	closeOnce sync.Once

	// Metrics
	metrics     Metrics
	metricsLock sync.RWMutex
}

// NewProcessor creates a new batch processor with the given configuration and handler.
func NewProcessor(cfg *Config, handler BatchHandler) (*Processor, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	if handler == nil {
		return nil, sentinelerrors.NewConfigValidationError("handler", nil, "batch handler is required")
	}

	logger := cfg.Logger
	if logger == nil {
		logger = logging.L()
	}

	ctx, cancel := context.WithCancel(context.Background())

	p := &Processor{
		config:   cfg,
		handler:  handler,
		logger:   logger.With(zap.String("component", "batch_processor")),
		recordCh: make(chan *models.LogRecord, cfg.BufferSize),
		batch:    make([]*models.LogRecord, 0, cfg.MaxBatchSize),
		ctx:      ctx,
		cancel:   cancel,
	}

	// Start the processing goroutine
	p.wg.Add(1)
	go p.processLoop()

	p.logger.Info("batch_processor_started",
		zap.Int("max_batch_size", cfg.MaxBatchSize),
		zap.Duration("max_wait_time", cfg.MaxWaitTime),
		zap.Int("buffer_size", cfg.BufferSize),
	)

	return p, nil
}

// Add adds a log record to the batch processor.
// If DropOnFull is false, this will block when the buffer is full.
// If DropOnFull is true, the record will be dropped and ErrBatchFull returned.
func (p *Processor) Add(record *models.LogRecord) error {
	if p.closed.Load() {
		return ErrProcessorClosed
	}

	if record == nil {
		return nil
	}

	if p.config.DropOnFull {
		select {
		case p.recordCh <- record:
			atomic.AddInt64(&p.metrics.TotalRecords, 1)
			return nil
		default:
			atomic.AddInt64(&p.metrics.TotalDropped, 1)
			p.logger.Warn("record_dropped_buffer_full",
				zap.String("source", record.Source),
			)
			return ErrBatchFull
		}
	}

	select {
	case p.recordCh <- record:
		atomic.AddInt64(&p.metrics.TotalRecords, 1)
		return nil
	case <-p.ctx.Done():
		return ErrProcessorClosed
	}
}

// AddBatch adds multiple log records to the batch processor.
func (p *Processor) AddBatch(records []*models.LogRecord) error {
	for _, record := range records {
		if err := p.Add(record); err != nil {
			return err
		}
	}
	return nil
}

// Flush forces an immediate flush of the current batch.
func (p *Processor) Flush(ctx context.Context) error {
	if p.closed.Load() {
		return ErrProcessorClosed
	}

	p.batchMu.Lock()
	batch := p.batch
	p.batch = make([]*models.LogRecord, 0, p.config.MaxBatchSize)
	p.batchMu.Unlock()

	if len(batch) == 0 {
		return nil
	}

	return p.flushBatch(ctx, batch)
}

// Close stops the processor and flushes any remaining records.
func (p *Processor) Close() error {
	var closeErr error
	p.closeOnce.Do(func() {
		p.closed.Store(true)
		close(p.recordCh)
		p.cancel()

		// Wait for processing to complete
		p.wg.Wait()

		// Flush any remaining records
		p.batchMu.Lock()
		remaining := p.batch
		p.batch = nil
		p.batchMu.Unlock()

		if len(remaining) > 0 {
			ctx, cancel := context.WithTimeout(context.Background(), p.config.FlushTimeout)
			defer cancel()
			if err := p.flushBatch(ctx, remaining); err != nil {
				p.logger.Error("final_flush_failed", zap.Error(err))
				closeErr = err
			}
		}

		p.logger.Info("batch_processor_stopped",
			zap.Int64("total_records", p.metrics.TotalRecords),
			zap.Int64("total_batches", p.metrics.TotalBatches),
			zap.Int64("total_processed", p.metrics.TotalProcessed),
			zap.Int64("total_dropped", p.metrics.TotalDropped),
		)
	})
	return closeErr
}

// GetMetrics returns a copy of the current metrics.
func (p *Processor) GetMetrics() Metrics {
	p.metricsLock.RLock()
	defer p.metricsLock.RUnlock()

	return Metrics{
		TotalRecords:      atomic.LoadInt64(&p.metrics.TotalRecords),
		TotalBatches:      atomic.LoadInt64(&p.metrics.TotalBatches),
		TotalProcessed:    atomic.LoadInt64(&p.metrics.TotalProcessed),
		TotalDropped:      atomic.LoadInt64(&p.metrics.TotalDropped),
		LastFlushTime:     p.metrics.LastFlushTime,
		LastFlushDuration: p.metrics.LastFlushDuration,
		LastBatchSize:     p.metrics.LastBatchSize,
	}
}

// processLoop is the main processing goroutine.
func (p *Processor) processLoop() {
	defer p.wg.Done()

	ticker := time.NewTicker(p.config.MaxWaitTime)
	defer ticker.Stop()

	for {
		select {
		case <-p.ctx.Done():
			return

		case record, ok := <-p.recordCh:
			if !ok {
				return
			}

			p.batchMu.Lock()
			p.batch = append(p.batch, record)
			shouldFlush := len(p.batch) >= p.config.MaxBatchSize
			var batch []*models.LogRecord
			if shouldFlush {
				batch = p.batch
				p.batch = make([]*models.LogRecord, 0, p.config.MaxBatchSize)
			}
			p.batchMu.Unlock()

			if shouldFlush {
				ctx, cancel := context.WithTimeout(p.ctx, p.config.FlushTimeout)
				if err := p.flushBatch(ctx, batch); err != nil {
					p.logger.Error("batch_flush_failed", zap.Error(err))
				}
				cancel()
			}

		case <-ticker.C:
			p.batchMu.Lock()
			batch := p.batch
			p.batch = make([]*models.LogRecord, 0, p.config.MaxBatchSize)
			p.batchMu.Unlock()

			if len(batch) > 0 {
				ctx, cancel := context.WithTimeout(p.ctx, p.config.FlushTimeout)
				if err := p.flushBatch(ctx, batch); err != nil {
					p.logger.Error("periodic_flush_failed", zap.Error(err))
				}
				cancel()
			}
		}
	}
}

// flushBatch flushes a batch to the handler.
func (p *Processor) flushBatch(ctx context.Context, batch []*models.LogRecord) error {
	if len(batch) == 0 {
		return nil
	}

	startTime := time.Now()
	p.logger.Debug("flushing_batch",
		zap.Int("batch_size", len(batch)),
	)

	processed, err := p.handler.HandleBatch(ctx, batch)
	duration := time.Since(startTime)

	// Update metrics
	atomic.AddInt64(&p.metrics.TotalBatches, 1)
	atomic.AddInt64(&p.metrics.TotalProcessed, int64(processed))

	p.metricsLock.Lock()
	p.metrics.LastFlushTime = time.Now()
	p.metrics.LastFlushDuration = duration
	p.metrics.LastBatchSize = len(batch)
	p.metricsLock.Unlock()

	if err != nil {
		dropped := len(batch) - processed
		atomic.AddInt64(&p.metrics.TotalDropped, int64(dropped))

		p.logger.Error("batch_handler_failed",
			zap.Int("batch_size", len(batch)),
			zap.Int("processed", processed),
			zap.Int("dropped", dropped),
			zap.Duration("duration", duration),
			zap.Error(err),
		)

		return fmt.Errorf("%w: %v", ErrHandlerFailed, err)
	}

	p.logger.Info("batch_flushed",
		zap.Int("batch_size", len(batch)),
		zap.Int("processed", processed),
		zap.Duration("duration", duration),
	)

	return nil
}
