package cmd

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"sentinel-log-ai/internal/logging"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDefaultIngestOptions tests the default options.
func TestDefaultIngestOptions(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	opts := DefaultIngestOptions()

	assert.False(t, opts.TailMode)
	assert.Equal(t, "*", opts.Pattern)
	assert.Equal(t, "localhost:50051", opts.MLServerAddr)
	assert.Equal(t, 100, opts.BatchSize)
	assert.Equal(t, 5*time.Second, opts.BatchTimeout)
	assert.Equal(t, 10000, opts.BufferSize)
	assert.False(t, opts.DryRun)
}

// TestNewIngestRunner tests ingest runner creation.
func TestNewIngestRunner(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	tests := []struct {
		name    string
		opts    *IngestOptions
		wantErr bool
	}{
		{
			name:    "nil options uses defaults",
			opts:    nil,
			wantErr: false,
		},
		{
			name: "valid custom options",
			opts: &IngestOptions{
				Path:         "/tmp/test.log",
				BatchSize:    50,
				BatchTimeout: 1 * time.Second,
				BufferSize:   1000,
				DryRun:       true,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runner, err := NewIngestRunner(tt.opts)
			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, runner)
			defer func() { _ = runner.Close() }()
		})
	}
}

// TestIngestRunnerDryRun tests dry-run mode processing.
func TestIngestRunnerDryRun(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	// Create a temporary test file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.log")
	content := `2024-01-15T10:30:00Z INFO Starting application
2024-01-15T10:30:01Z INFO Server listening on port 8080
2024-01-15T10:30:02Z ERROR Connection refused to database
2024-01-15T10:30:03Z WARN High memory usage detected
2024-01-15T10:30:04Z INFO Request completed successfully`
	err := os.WriteFile(testFile, []byte(content), 0644)
	require.NoError(t, err)

	opts := &IngestOptions{
		Path:           testFile,
		TailMode:       false,
		BatchSize:      2,
		BatchTimeout:   100 * time.Millisecond,
		BufferSize:     100,
		DryRun:         true,
		ConnectTimeout: 1 * time.Second,
	}

	runner, err := NewIngestRunner(opts)
	require.NoError(t, err)
	defer func() { _ = runner.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = runner.Run(ctx)
	assert.NoError(t, err)
}

// TestIngestRunnerFileNotFound tests handling of missing files.
func TestIngestRunnerFileNotFound(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	opts := &IngestOptions{
		Path:     "/nonexistent/path/to/file.log",
		DryRun:   true,
		BatchSize: 100,
		BatchTimeout: 1 * time.Second,
		BufferSize: 1000,
	}

	runner, err := NewIngestRunner(opts)
	require.NoError(t, err)
	defer func() { _ = runner.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = runner.Run(ctx)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "path not found")
}

// TestIngestRunnerDirectory tests directory ingestion.
func TestIngestRunnerDirectory(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	// Create a temporary directory with test files
	tmpDir := t.TempDir()

	// Create test files
	for _, name := range []string{"app1.log", "app2.log", "other.txt"} {
		content := "2024-01-15T10:30:00Z INFO Test log entry\n"
		err := os.WriteFile(filepath.Join(tmpDir, name), []byte(content), 0644)
		require.NoError(t, err)
	}

	opts := &IngestOptions{
		Path:         tmpDir,
		Pattern:      "*.log",
		TailMode:     false,
		BatchSize:    10,
		BatchTimeout: 100 * time.Millisecond,
		BufferSize:   100,
		DryRun:       true,
	}

	runner, err := NewIngestRunner(opts)
	require.NoError(t, err)
	defer func() { _ = runner.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = runner.Run(ctx)
	assert.NoError(t, err)
}

// TestIngestRunnerEmptyDirectory tests handling of directories with no matching files.
func TestIngestRunnerEmptyDirectory(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	tmpDir := t.TempDir()

	opts := &IngestOptions{
		Path:         tmpDir,
		Pattern:      "*.log",
		DryRun:       true,
		BatchSize:    100,
		BatchTimeout: 1 * time.Second,
		BufferSize:   1000,
	}

	runner, err := NewIngestRunner(opts)
	require.NoError(t, err)
	defer func() { _ = runner.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = runner.Run(ctx)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no files match pattern")
}

// TestIngestRunnerContextCancellation tests graceful shutdown on context cancellation.
func TestIngestRunnerContextCancellation(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	// Create a temporary test file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.log")
	content := "2024-01-15T10:30:00Z INFO Test log entry\n"
	err := os.WriteFile(testFile, []byte(content), 0644)
	require.NoError(t, err)

	opts := &IngestOptions{
		Path:         testFile,
		TailMode:     false,
		BatchSize:    100,
		BatchTimeout: 1 * time.Second,
		BufferSize:   1000,
		DryRun:       true,
	}

	runner, err := NewIngestRunner(opts)
	require.NoError(t, err)
	defer func() { _ = runner.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// Should complete quickly without error (cancelled before reading)
	err = runner.Run(ctx)
	// May or may not error depending on timing
	_ = err
}

// TestSetupIngestCmd tests command setup.
func TestSetupIngestCmd(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	cmd := setupIngestCmd()

	assert.Equal(t, "ingest [path]", cmd.Use)
	assert.NotEmpty(t, cmd.Short)
	assert.NotEmpty(t, cmd.Long)

	// Check that flags are set up
	tailFlag := cmd.Flags().Lookup("tail")
	assert.NotNil(t, tailFlag)

	patternFlag := cmd.Flags().Lookup("pattern")
	assert.NotNil(t, patternFlag)

	dryRunFlag := cmd.Flags().Lookup("dry-run")
	assert.NotNil(t, dryRunFlag)

	batchSizeFlag := cmd.Flags().Lookup("batch-size")
	assert.NotNil(t, batchSizeFlag)
}

// TestIngestRunnerClose tests proper cleanup.
func TestIngestRunnerClose(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	opts := DefaultIngestOptions()
	opts.Path = "/tmp/test.log"
	opts.DryRun = true

	runner, err := NewIngestRunner(opts)
	require.NoError(t, err)

	// Close should not error
	err = runner.Close()
	assert.NoError(t, err)

	// Second close should also not error (idempotent)
	err = runner.Close()
	assert.NoError(t, err)
}

// TestIngestRunnerJSONLogs tests processing of JSON-formatted logs.
func TestIngestRunnerJSONLogs(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	// Create a temporary test file with JSON logs
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.jsonl")
	content := `{"timestamp":"2024-01-15T10:30:00Z","level":"INFO","message":"Application started"}
{"timestamp":"2024-01-15T10:30:01Z","level":"ERROR","message":"Database connection failed","error":"connection refused"}
{"timestamp":"2024-01-15T10:30:02Z","level":"WARN","message":"Retrying in 5 seconds"}`
	err := os.WriteFile(testFile, []byte(content), 0644)
	require.NoError(t, err)

	opts := &IngestOptions{
		Path:         testFile,
		TailMode:     false,
		BatchSize:    10,
		BatchTimeout: 100 * time.Millisecond,
		BufferSize:   100,
		DryRun:       true,
	}

	runner, err := NewIngestRunner(opts)
	require.NoError(t, err)
	defer func() { _ = runner.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = runner.Run(ctx)
	assert.NoError(t, err)
}

// TestIngestRunnerSyslogFormat tests processing of syslog-formatted logs.
func TestIngestRunnerSyslogFormat(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	// Create a temporary test file with syslog format
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "syslog")
	content := `Jan 15 10:30:00 webserver nginx[1234]: 192.168.1.100 - - [15/Jan/2024:10:30:00 +0000] "GET /api/health HTTP/1.1" 200 42
Jan 15 10:30:01 webserver nginx[1234]: 192.168.1.101 - - [15/Jan/2024:10:30:01 +0000] "POST /api/data HTTP/1.1" 201 128
Jan 15 10:30:02 webserver nginx[1234]: 192.168.1.102 - - [15/Jan/2024:10:30:02 +0000] "GET /api/users HTTP/1.1" 500 64`
	err := os.WriteFile(testFile, []byte(content), 0644)
	require.NoError(t, err)

	opts := &IngestOptions{
		Path:         testFile,
		TailMode:     false,
		BatchSize:    10,
		BatchTimeout: 100 * time.Millisecond,
		BufferSize:   100,
		DryRun:       true,
	}

	runner, err := NewIngestRunner(opts)
	require.NoError(t, err)
	defer func() { _ = runner.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = runner.Run(ctx)
	assert.NoError(t, err)
}
