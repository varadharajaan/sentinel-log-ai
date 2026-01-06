package ingestion

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sentinel-log-ai/sentinel-log-ai/internal/models"
	"go.uber.org/zap"
)

func TestNewFileSource(t *testing.T) {
	logger := zap.NewNop()

	tests := []struct {
		name     string
		path     string
		tailMode bool
	}{
		{"batch mode", "/var/log/test.log", false},
		{"tail mode", "/var/log/app.log", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source := NewFileSource(tt.path, tt.tailMode, logger)
			if source == nil {
				t.Fatal("NewFileSource returned nil")
			}
			if source.path != tt.path {
				t.Errorf("path = %v, want %v", source.path, tt.path)
			}
			if source.tail != tt.tailMode {
				t.Errorf("tail = %v, want %v", source.tail, tt.tailMode)
			}
		})
	}
}

func TestFileSource_Name(t *testing.T) {
	logger := zap.NewNop()
	source := NewFileSource("/var/log/test.log", false, logger)

	name := source.Name()
	if !strings.HasPrefix(name, "file:") {
		t.Errorf("Name() = %v, want prefix 'file:'", name)
	}
	if !strings.Contains(name, "test.log") {
		t.Errorf("Name() = %v, should contain 'test.log'", name)
	}
}

func TestFileSource_ReadBatch(t *testing.T) {
	logger := zap.NewNop()

	// Create temp file with test content
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "test.log")

	content := `first line
second line
third line
ERROR: something failed
{"message": "json log", "level": "INFO"}`

	if err := os.WriteFile(tmpFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	source := NewFileSource(tmpFile, false, logger)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	records := make(chan *models.LogRecord, 10)

	// Read in goroutine
	errChan := make(chan error, 1)
	go func() {
		errChan <- source.Read(ctx, records)
		close(records)
	}()

	// Collect records
	var collected []*models.LogRecord
	for record := range records {
		collected = append(collected, record)
	}

	if err := <-errChan; err != nil {
		t.Fatalf("Read() error = %v", err)
	}

	if len(collected) != 5 {
		t.Errorf("Read() collected %v records, want 5", len(collected))
	}

	// Verify first record
	if collected[0].Raw != "first line" {
		t.Errorf("First record raw = %v, want 'first line'", collected[0].Raw)
	}

	// Verify source is set
	for i, r := range collected {
		if r.Source != tmpFile {
			t.Errorf("Record %d source = %v, want %v", i, r.Source, tmpFile)
		}
	}

	// Verify JSON parsing
	lastRecord := collected[4]
	if lastRecord.Message != "json log" {
		t.Errorf("JSON record message = %v, want 'json log'", lastRecord.Message)
	}
	if lastRecord.Level != "INFO" {
		t.Errorf("JSON record level = %v, want 'INFO'", lastRecord.Level)
	}
}

func TestFileSource_ReadBatch_EmptyFile(t *testing.T) {
	logger := zap.NewNop()

	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "empty.log")

	if err := os.WriteFile(tmpFile, []byte(""), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	source := NewFileSource(tmpFile, false, logger)

	ctx := context.Background()
	records := make(chan *models.LogRecord, 10)

	errChan := make(chan error, 1)
	go func() {
		errChan <- source.Read(ctx, records)
		close(records)
	}()

	var count int
	for range records {
		count++
	}

	if err := <-errChan; err != nil {
		t.Fatalf("Read() error = %v", err)
	}

	if count != 0 {
		t.Errorf("Empty file should produce 0 records, got %d", count)
	}
}

func TestFileSource_ReadBatch_LargeLines(t *testing.T) {
	logger := zap.NewNop()

	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "large.log")

	// Create a line larger than default buffer
	largeLine := strings.Repeat("a", 100000) // 100KB line
	content := "normal line\n" + largeLine + "\nanother normal line"

	if err := os.WriteFile(tmpFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	source := NewFileSource(tmpFile, false, logger)

	ctx := context.Background()
	records := make(chan *models.LogRecord, 10)

	errChan := make(chan error, 1)
	go func() {
		errChan <- source.Read(ctx, records)
		close(records)
	}()

	var collected []*models.LogRecord
	for record := range records {
		collected = append(collected, record)
	}

	if err := <-errChan; err != nil {
		t.Fatalf("Read() error = %v", err)
	}

	if len(collected) != 3 {
		t.Errorf("Read() collected %v records, want 3", len(collected))
	}

	// Verify large line was read correctly
	if len(collected[1].Raw) != 100000 {
		t.Errorf("Large line length = %v, want 100000", len(collected[1].Raw))
	}
}

func TestFileSource_ReadBatch_NonExistent(t *testing.T) {
	logger := zap.NewNop()
	source := NewFileSource("/nonexistent/path/file.log", false, logger)

	ctx := context.Background()
	records := make(chan *models.LogRecord, 10)

	err := source.Read(ctx, records)
	if err == nil {
		t.Error("Read() should return error for non-existent file")
	}
}

func TestFileSource_ReadBatch_ContextCancellation(t *testing.T) {
	logger := zap.NewNop()

	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "test.log")

	// Create file with many lines
	var lines []string
	for i := 0; i < 10000; i++ {
		lines = append(lines, "line content")
	}
	content := strings.Join(lines, "\n")

	if err := os.WriteFile(tmpFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	source := NewFileSource(tmpFile, false, logger)

	ctx, cancel := context.WithCancel(context.Background())

	records := make(chan *models.LogRecord, 100)

	errChan := make(chan error, 1)
	go func() {
		errChan <- source.Read(ctx, records)
	}()

	// Read a few records then cancel
	for i := 0; i < 10; i++ {
		<-records
	}
	cancel()

	err := <-errChan
	if err != context.Canceled {
		t.Errorf("Read() error = %v, want context.Canceled", err)
	}
}

func TestFileSource_Close(t *testing.T) {
	logger := zap.NewNop()
	source := NewFileSource("/var/log/test.log", false, logger)

	err := source.Close()
	if err != nil {
		t.Errorf("Close() error = %v", err)
	}
}

func TestNewStdinSource(t *testing.T) {
	logger := zap.NewNop()
	source := NewStdinSource(logger)

	if source == nil {
		t.Fatal("NewStdinSource returned nil")
	}

	if source.Name() != "stdin" {
		t.Errorf("Name() = %v, want 'stdin'", source.Name())
	}
}

func TestStdinSource_Close(t *testing.T) {
	logger := zap.NewNop()
	source := NewStdinSource(logger)

	err := source.Close()
	if err != nil {
		t.Errorf("Close() error = %v", err)
	}
}

// Test JSON parsing edge cases
func TestFileSource_JSONParsing(t *testing.T) {
	logger := zap.NewNop()

	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "json.log")

	content := `{"message": "test1", "level": "INFO", "timestamp": "2024-01-15T10:30:00Z"}
{"msg": "test2", "level": "WARN"}
{"level": "ERROR"}
not json at all`

	if err := os.WriteFile(tmpFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	source := NewFileSource(tmpFile, false, logger)

	ctx := context.Background()
	records := make(chan *models.LogRecord, 10)

	errChan := make(chan error, 1)
	go func() {
		errChan <- source.Read(ctx, records)
		close(records)
	}()

	var collected []*models.LogRecord
	for record := range records {
		collected = append(collected, record)
	}

	if err := <-errChan; err != nil {
		t.Fatalf("Read() error = %v", err)
	}

	if len(collected) != 4 {
		t.Fatalf("Read() collected %v records, want 4", len(collected))
	}

	// First record - full JSON
	if collected[0].Message != "test1" {
		t.Errorf("Record 0 message = %v, want 'test1'", collected[0].Message)
	}
	if collected[0].Timestamp == nil {
		t.Error("Record 0 should have timestamp")
	}

	// Second record - msg field
	if collected[1].Message != "test2" {
		t.Errorf("Record 1 message = %v, want 'test2'", collected[1].Message)
	}

	// Third record - no message
	if collected[2].Level != "ERROR" {
		t.Errorf("Record 2 level = %v, want 'ERROR'", collected[2].Level)
	}

	// Fourth record - not JSON
	if collected[3].Raw != "not json at all" {
		t.Errorf("Record 3 raw = %v, want 'not json at all'", collected[3].Raw)
	}
}

// Benchmark tests
func BenchmarkFileSource_ReadBatch(b *testing.B) {
	logger := zap.NewNop()

	tmpDir := b.TempDir()
	tmpFile := filepath.Join(tmpDir, "bench.log")

	// Create file with 10000 lines
	var lines []string
	for i := 0; i < 10000; i++ {
		lines = append(lines, `{"message": "benchmark log line", "level": "INFO"}`)
	}
	content := strings.Join(lines, "\n")

	if err := os.WriteFile(tmpFile, []byte(content), 0644); err != nil {
		b.Fatalf("Failed to create test file: %v", err)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		source := NewFileSource(tmpFile, false, logger)
		ctx := context.Background()
		records := make(chan *models.LogRecord, 1000)

		go func() {
			source.Read(ctx, records)
			close(records)
		}()

		for range records {
			// consume
		}
	}
}
