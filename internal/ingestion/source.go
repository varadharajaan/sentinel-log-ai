// Package ingestion provides log source adapters for reading logs from various sources.
package ingestion

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"sentinel-log-ai/internal/models"

	"github.com/nxadm/tail"
	"go.uber.org/zap"
)

// Source is the interface that log source adapters must implement.
type Source interface {
	// Read reads log records and sends them to the provided channel.
	// It should return when the context is cancelled or the source is exhausted.
	Read(ctx context.Context, records chan<- *models.LogRecord) error

	// Name returns a human-readable name for this source.
	Name() string

	// Close releases any resources held by the source.
	Close() error
}

// FileSource reads logs from a file.
type FileSource struct {
	path   string
	tail   bool
	logger *zap.Logger
}

// NewFileSource creates a new file source.
func NewFileSource(path string, tailMode bool, logger *zap.Logger) *FileSource {
	return &FileSource{
		path:   path,
		tail:   tailMode,
		logger: logger,
	}
}

// Name returns the source name.
func (f *FileSource) Name() string {
	return fmt.Sprintf("file:%s", f.path)
}

// Read reads log records from the file.
func (f *FileSource) Read(ctx context.Context, records chan<- *models.LogRecord) error {
	if f.tail {
		return f.readTail(ctx, records)
	}
	return f.readBatch(ctx, records)
}

// readBatch reads the entire file.
func (f *FileSource) readBatch(ctx context.Context, records chan<- *models.LogRecord) error {
	file, err := os.Open(f.path)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	// Increase buffer size for long lines
	const maxCapacity = 1024 * 1024 // 1MB
	buf := make([]byte, maxCapacity)
	scanner.Buffer(buf, maxCapacity)

	lineNum := 0
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		lineNum++
		line := scanner.Text()
		record := f.parseLine(line, lineNum)
		records <- record
	}

	return scanner.Err()
}

// readTail follows the file for new lines.
func (f *FileSource) readTail(ctx context.Context, records chan<- *models.LogRecord) error {
	t, err := tail.TailFile(f.path, tail.Config{
		Follow:    true,
		ReOpen:    true,
		MustExist: true,
		Logger:    tail.DiscardingLogger,
	})
	if err != nil {
		return fmt.Errorf("failed to tail file: %w", err)
	}
	defer t.Stop()

	lineNum := 0
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case line, ok := <-t.Lines:
			if !ok {
				return nil
			}
			if line.Err != nil {
				f.logger.Warn("Error reading line", zap.Error(line.Err))
				continue
			}
			lineNum++
			record := f.parseLine(line.Text, lineNum)
			records <- record
		}
	}
}

// parseLine parses a log line into a LogRecord.
func (f *FileSource) parseLine(line string, lineNum int) *models.LogRecord {
	record := &models.LogRecord{
		Raw:    line,
		Source: f.path,
		Attrs:  map[string]any{"line_num": lineNum},
	}

	// Try to parse as JSON first
	if strings.HasPrefix(strings.TrimSpace(line), "{") {
		var jsonLog map[string]any
		if err := json.Unmarshal([]byte(line), &jsonLog); err == nil {
			record.Attrs = jsonLog
			if msg, ok := jsonLog["message"].(string); ok {
				record.Message = msg
			} else if msg, ok := jsonLog["msg"].(string); ok {
				record.Message = msg
			}
			if level, ok := jsonLog["level"].(string); ok {
				record.Level = level
			}
			if ts, ok := jsonLog["timestamp"].(string); ok {
				if t, err := time.Parse(time.RFC3339, ts); err == nil {
					record.Timestamp = &t
				}
			}
			return record
		}
	}

	// Fallback to raw message
	record.Message = line
	return record
}

// Close releases resources.
func (f *FileSource) Close() error {
	return nil
}

// StdinSource reads logs from standard input.
type StdinSource struct {
	logger *zap.Logger
}

// NewStdinSource creates a new stdin source.
func NewStdinSource(logger *zap.Logger) *StdinSource {
	return &StdinSource{logger: logger}
}

// Name returns the source name.
func (s *StdinSource) Name() string {
	return "stdin"
}

// Read reads log records from stdin.
func (s *StdinSource) Read(ctx context.Context, records chan<- *models.LogRecord) error {
	scanner := bufio.NewScanner(os.Stdin)
	lineNum := 0

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		lineNum++
		line := scanner.Text()
		record := &models.LogRecord{
			Raw:     line,
			Message: line,
			Source:  "stdin",
			Attrs:   map[string]any{"line_num": lineNum},
		}
		records <- record
	}

	return scanner.Err()
}

// Close releases resources.
func (s *StdinSource) Close() error {
	return nil
}
