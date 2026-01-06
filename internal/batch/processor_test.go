package batch

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"sentinel-log-ai/internal/logging"
	"sentinel-log-ai/internal/models"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// TestNewProcessor tests processor creation with various configurations.
func TestNewProcessor(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	logger := zap.NewNop()
	handler := BatchHandlerFunc(func(_ context.Context, _ []*models.LogRecord) (int, error) {
		return 0, nil
	})

	tests := []struct {
		name      string
		config    *Config
		handler   BatchHandler
		wantErr   bool
		errSubstr string
	}{
		{
			name:    "default config",
			config:  nil,
			handler: handler,
			wantErr: false,
		},
		{
			name: "valid custom config",
			config: &Config{
				MaxBatchSize: 50,
				MaxWaitTime:  1 * time.Second,
				BufferSize:   1000,
				FlushTimeout: 10 * time.Second,
				Logger:       logger,
			},
			handler: handler,
			wantErr: false,
		},
		{
			name: "invalid MaxBatchSize",
			config: &Config{
				MaxBatchSize: 0,
				MaxWaitTime:  1 * time.Second,
				BufferSize:   1000,
				FlushTimeout: 10 * time.Second,
			},
			handler:   handler,
			wantErr:   true,
			errSubstr: "MaxBatchSize",
		},
		{
			name: "invalid MaxWaitTime",
			config: &Config{
				MaxBatchSize: 100,
				MaxWaitTime:  0,
				BufferSize:   1000,
				FlushTimeout: 10 * time.Second,
			},
			handler:   handler,
			wantErr:   true,
			errSubstr: "MaxWaitTime",
		},
		{
			name: "invalid BufferSize",
			config: &Config{
				MaxBatchSize: 100,
				MaxWaitTime:  1 * time.Second,
				BufferSize:   0,
				FlushTimeout: 10 * time.Second,
			},
			handler:   handler,
			wantErr:   true,
			errSubstr: "BufferSize",
		},
		{
			name: "nil handler",
			config: &Config{
				MaxBatchSize: 100,
				MaxWaitTime:  1 * time.Second,
				BufferSize:   1000,
				FlushTimeout: 10 * time.Second,
			},
			handler:   nil,
			wantErr:   true,
			errSubstr: "handler",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proc, err := NewProcessor(tt.config, tt.handler)
			if tt.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.errSubstr)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, proc)
			defer func() { _ = proc.Close() }()
		})
	}
}

// TestProcessorAdd tests adding records to the processor.
func TestProcessorAdd(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	var received int64
	handler := BatchHandlerFunc(func(_ context.Context, batch []*models.LogRecord) (int, error) {
		atomic.AddInt64(&received, int64(len(batch)))
		return len(batch), nil
	})

	cfg := &Config{
		MaxBatchSize: 10,
		MaxWaitTime:  100 * time.Millisecond,
		BufferSize:   100,
		FlushTimeout: 5 * time.Second,
		Logger:       zap.NewNop(),
	}

	proc, err := NewProcessor(cfg, handler)
	require.NoError(t, err)

	// Add records
	for i := 0; i < 25; i++ {
		record := &models.LogRecord{
			Message: "test message",
			Source:  "test",
		}
		err := proc.Add(record)
		require.NoError(t, err)
	}

	// Wait for batches to be processed
	time.Sleep(300 * time.Millisecond)

	// Close and wait for final flush
	err = proc.Close()
	require.NoError(t, err)

	// Verify all records were processed
	metrics := proc.GetMetrics()
	assert.Equal(t, int64(25), metrics.TotalRecords)
	assert.Equal(t, int64(25), metrics.TotalProcessed)
	assert.GreaterOrEqual(t, metrics.TotalBatches, int64(2))
}

// TestProcessorBatchSize tests that batches are flushed at max size.
func TestProcessorBatchSize(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	var batchSizes []int
	var mu sync.Mutex

	handler := BatchHandlerFunc(func(_ context.Context, batch []*models.LogRecord) (int, error) {
		mu.Lock()
		batchSizes = append(batchSizes, len(batch))
		mu.Unlock()
		return len(batch), nil
	})

	cfg := &Config{
		MaxBatchSize: 5,
		MaxWaitTime:  10 * time.Second, // Long wait to ensure size-based flush
		BufferSize:   100,
		FlushTimeout: 5 * time.Second,
		Logger:       zap.NewNop(),
	}

	proc, err := NewProcessor(cfg, handler)
	require.NoError(t, err)

	// Add exactly MaxBatchSize records
	for i := 0; i < 5; i++ {
		err := proc.Add(&models.LogRecord{Message: "test"})
		require.NoError(t, err)
	}

	// Wait for flush
	time.Sleep(100 * time.Millisecond)

	mu.Lock()
	sizes := make([]int, len(batchSizes))
	copy(sizes, batchSizes)
	mu.Unlock()

	// Should have flushed one batch of size 5
	require.Len(t, sizes, 1)
	assert.Equal(t, 5, sizes[0])

	err = proc.Close()
	require.NoError(t, err)
}

// TestProcessorTimeBasedFlush tests that batches are flushed after max wait time.
func TestProcessorTimeBasedFlush(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	flushCh := make(chan int, 10)

	handler := BatchHandlerFunc(func(_ context.Context, batch []*models.LogRecord) (int, error) {
		flushCh <- len(batch)
		return len(batch), nil
	})

	cfg := &Config{
		MaxBatchSize: 100, // High batch size to ensure time-based flush
		MaxWaitTime:  50 * time.Millisecond,
		BufferSize:   100,
		FlushTimeout: 5 * time.Second,
		Logger:       zap.NewNop(),
	}

	proc, err := NewProcessor(cfg, handler)
	require.NoError(t, err)

	// Add some records (less than MaxBatchSize)
	for i := 0; i < 3; i++ {
		err := proc.Add(&models.LogRecord{Message: "test"})
		require.NoError(t, err)
	}

	// Wait for time-based flush
	select {
	case size := <-flushCh:
		assert.Equal(t, 3, size)
	case <-time.After(200 * time.Millisecond):
		t.Fatal("expected time-based flush did not occur")
	}

	err = proc.Close()
	require.NoError(t, err)
}

// TestProcessorClose tests proper shutdown behavior.
func TestProcessorClose(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	var finalBatch []*models.LogRecord
	var mu sync.Mutex

	handler := BatchHandlerFunc(func(_ context.Context, batch []*models.LogRecord) (int, error) {
		mu.Lock()
		finalBatch = append(finalBatch, batch...)
		mu.Unlock()
		return len(batch), nil
	})

	cfg := &Config{
		MaxBatchSize: 100,
		MaxWaitTime:  10 * time.Second,
		BufferSize:   100,
		FlushTimeout: 5 * time.Second,
		Logger:       zap.NewNop(),
	}

	proc, err := NewProcessor(cfg, handler)
	require.NoError(t, err)

	// Add records without triggering flush
	for i := 0; i < 7; i++ {
		err := proc.Add(&models.LogRecord{Message: "test"})
		require.NoError(t, err)
	}

	// Give processLoop time to move records from channel to batch
	time.Sleep(50 * time.Millisecond)

	// Close should flush remaining records
	err = proc.Close()
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	assert.Len(t, finalBatch, 7)
}

// TestProcessorAddAfterClose tests that Add returns error after Close.
func TestProcessorAddAfterClose(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	handler := BatchHandlerFunc(func(_ context.Context, _ []*models.LogRecord) (int, error) {
		return 0, nil
	})

	proc, err := NewProcessor(nil, handler)
	require.NoError(t, err)

	err = proc.Close()
	require.NoError(t, err)

	err = proc.Add(&models.LogRecord{Message: "test"})
	assert.ErrorIs(t, err, ErrProcessorClosed)
}

// TestProcessorDropOnFull tests the DropOnFull behavior.
func TestProcessorDropOnFull(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	// Slow handler to create backpressure
	handler := BatchHandlerFunc(func(_ context.Context, batch []*models.LogRecord) (int, error) {
		time.Sleep(100 * time.Millisecond)
		return len(batch), nil
	})

	cfg := &Config{
		MaxBatchSize: 5,
		MaxWaitTime:  10 * time.Millisecond,
		BufferSize:   5, // Small buffer
		FlushTimeout: 5 * time.Second,
		DropOnFull:   true,
		Logger:       zap.NewNop(),
	}

	proc, err := NewProcessor(cfg, handler)
	require.NoError(t, err)

	// Try to add more records than buffer can hold
	droppedCount := 0
	for i := 0; i < 20; i++ {
		err := proc.Add(&models.LogRecord{Message: "test"})
		if errors.Is(err, ErrBatchFull) {
			droppedCount++
		}
	}

	// Some records should have been dropped
	assert.Greater(t, droppedCount, 0, "expected some records to be dropped")

	err = proc.Close()
	require.NoError(t, err)

	metrics := proc.GetMetrics()
	assert.Greater(t, metrics.TotalDropped, int64(0))
}

// TestProcessorHandlerError tests handling of handler errors.
func TestProcessorHandlerError(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	handlerErr := errors.New("handler error")
	handler := BatchHandlerFunc(func(_ context.Context, batch []*models.LogRecord) (int, error) {
		// Process half, then fail
		return len(batch) / 2, handlerErr
	})

	cfg := &Config{
		MaxBatchSize: 4,
		MaxWaitTime:  10 * time.Millisecond,
		BufferSize:   100,
		FlushTimeout: 5 * time.Second,
		Logger:       zap.NewNop(),
	}

	proc, err := NewProcessor(cfg, handler)
	require.NoError(t, err)

	// Add records to trigger flush
	for i := 0; i < 4; i++ {
		err := proc.Add(&models.LogRecord{Message: "test"})
		require.NoError(t, err)
	}

	// Wait for flush
	time.Sleep(50 * time.Millisecond)

	err = proc.Close()
	require.NoError(t, err)

	metrics := proc.GetMetrics()
	assert.Greater(t, metrics.TotalDropped, int64(0))
}

// TestProcessorManualFlush tests manual flush operation.
func TestProcessorManualFlush(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	flushCount := int64(0)
	handler := BatchHandlerFunc(func(_ context.Context, batch []*models.LogRecord) (int, error) {
		atomic.AddInt64(&flushCount, 1)
		return len(batch), nil
	})

	cfg := &Config{
		MaxBatchSize: 100,
		MaxWaitTime:  10 * time.Second,
		BufferSize:   100,
		FlushTimeout: 5 * time.Second,
		Logger:       zap.NewNop(),
	}

	proc, err := NewProcessor(cfg, handler)
	require.NoError(t, err)

	// Add some records
	for i := 0; i < 5; i++ {
		err := proc.Add(&models.LogRecord{Message: "test"})
		require.NoError(t, err)
	}

	// Give processLoop time to move records from channel to batch
	time.Sleep(50 * time.Millisecond)

	// Manually flush
	err = proc.Flush(context.Background())
	require.NoError(t, err)

	assert.Equal(t, int64(1), atomic.LoadInt64(&flushCount))

	err = proc.Close()
	require.NoError(t, err)
}

// TestProcessorMetrics tests metrics tracking.
func TestProcessorMetrics(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	handler := BatchHandlerFunc(func(_ context.Context, batch []*models.LogRecord) (int, error) {
		return len(batch), nil
	})

	cfg := &Config{
		MaxBatchSize: 5,
		MaxWaitTime:  10 * time.Millisecond,
		BufferSize:   100,
		FlushTimeout: 5 * time.Second,
		Logger:       zap.NewNop(),
	}

	proc, err := NewProcessor(cfg, handler)
	require.NoError(t, err)

	// Add records
	for i := 0; i < 12; i++ {
		err := proc.Add(&models.LogRecord{Message: "test"})
		require.NoError(t, err)
	}

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	err = proc.Close()
	require.NoError(t, err)

	metrics := proc.GetMetrics()
	assert.Equal(t, int64(12), metrics.TotalRecords)
	assert.Equal(t, int64(12), metrics.TotalProcessed)
	assert.Equal(t, int64(0), metrics.TotalDropped)
	assert.GreaterOrEqual(t, metrics.TotalBatches, int64(2))
	assert.False(t, metrics.LastFlushTime.IsZero())
}

// TestProcessorConcurrency tests concurrent access.
func TestProcessorConcurrency(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	var processedCount int64
	handler := BatchHandlerFunc(func(_ context.Context, batch []*models.LogRecord) (int, error) {
		atomic.AddInt64(&processedCount, int64(len(batch)))
		return len(batch), nil
	})

	cfg := &Config{
		MaxBatchSize: 10,
		MaxWaitTime:  50 * time.Millisecond,
		BufferSize:   10000,
		FlushTimeout: 5 * time.Second,
		Logger:       zap.NewNop(),
	}

	proc, err := NewProcessor(cfg, handler)
	require.NoError(t, err)

	// Launch concurrent producers
	numGoroutines := 10
	recordsPerGoroutine := 100
	var wg sync.WaitGroup

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < recordsPerGoroutine; j++ {
				_ = proc.Add(&models.LogRecord{Message: "test"})
			}
		}()
	}

	wg.Wait()

	// Give processLoop time to process pending records before closing
	time.Sleep(100 * time.Millisecond)

	err = proc.Close()
	require.NoError(t, err)

	expectedTotal := int64(numGoroutines * recordsPerGoroutine)
	assert.Equal(t, expectedTotal, atomic.LoadInt64(&processedCount))
}

// TestProcessorNilRecord tests handling of nil records.
func TestProcessorNilRecord(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	handler := BatchHandlerFunc(func(_ context.Context, _ []*models.LogRecord) (int, error) {
		return 0, nil
	})

	proc, err := NewProcessor(nil, handler)
	require.NoError(t, err)

	err = proc.Add(nil)
	assert.NoError(t, err)

	err = proc.Close()
	require.NoError(t, err)
}

// TestProcessorFlushAfterClose tests Flush after Close.
func TestProcessorFlushAfterClose(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	handler := BatchHandlerFunc(func(_ context.Context, _ []*models.LogRecord) (int, error) {
		return 0, nil
	})

	proc, err := NewProcessor(nil, handler)
	require.NoError(t, err)

	err = proc.Close()
	require.NoError(t, err)

	err = proc.Flush(context.Background())
	assert.ErrorIs(t, err, ErrProcessorClosed)
}

// TestBatchHandlerFunc tests the BatchHandlerFunc adapter.
func TestBatchHandlerFunc(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	called := false
	f := BatchHandlerFunc(func(_ context.Context, batch []*models.LogRecord) (int, error) {
		called = true
		return len(batch), nil
	})

	batch := []*models.LogRecord{{Message: "test"}}
	n, err := f.HandleBatch(context.Background(), batch)

	assert.True(t, called)
	assert.NoError(t, err)
	assert.Equal(t, 1, n)
}

// TestConfigValidate tests Config validation.
func TestConfigValidate(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	tests := []struct {
		name    string
		config  *Config
		wantErr bool
	}{
		{
			name:    "valid config",
			config:  DefaultConfig(),
			wantErr: false,
		},
		{
			name: "invalid FlushTimeout",
			config: &Config{
				MaxBatchSize: 100,
				MaxWaitTime:  1 * time.Second,
				BufferSize:   1000,
				FlushTimeout: 0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
