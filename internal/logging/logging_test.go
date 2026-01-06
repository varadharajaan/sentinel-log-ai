// Package logging_test provides tests for the sentinel logging package.
package logging_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"sentinel-log-ai/internal/logging"
)

func TestDefaultConfig(t *testing.T) {
	cfg := logging.DefaultConfig()

	if cfg.Level != "info" {
		t.Errorf("expected level 'info', got %q", cfg.Level)
	}
	if cfg.LogDir != "logs" {
		t.Errorf("expected log dir 'logs', got %q", cfg.LogDir)
	}
	if cfg.LogFile != "sentinel-agent.jsonl" {
		t.Errorf("expected log file 'sentinel-agent.jsonl', got %q", cfg.LogFile)
	}
	if cfg.MaxSizeMB != 10 {
		t.Errorf("expected max size 10MB, got %d", cfg.MaxSizeMB)
	}
	if cfg.MaxBackups != 5 {
		t.Errorf("expected max backups 5, got %d", cfg.MaxBackups)
	}
	if !cfg.EnableConsole {
		t.Error("console should be enabled by default")
	}
	if !cfg.EnableFile {
		t.Error("file should be enabled by default")
	}
}

func TestSetupWithDefaults(t *testing.T) {
	// Use temp directory for logs
	tmpDir := t.TempDir()
	cfg := &logging.Config{
		Level:         "debug",
		LogDir:        tmpDir,
		LogFile:       "test.jsonl",
		MaxSizeMB:     1,
		MaxBackups:    2,
		EnableConsole: false, // Disable console to avoid test output noise
		EnableFile:    true,
		ConsoleFormat: "plain",
	}

	err := logging.Setup(cfg)
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}
	defer logging.Sync()

	// Log something
	logger := logging.L()
	logger.Info("test message", logging.Path("/var/log/app.log"))

	// Verify log file was created
	logPath := filepath.Join(tmpDir, "test.jsonl")
	if _, err := os.Stat(logPath); os.IsNotExist(err) {
		t.Error("log file was not created")
	}
}

func TestLoggerOutputsJSONL(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &logging.Config{
		Level:         "info",
		LogDir:        tmpDir,
		LogFile:       "jsonl-test.jsonl",
		MaxSizeMB:     1,
		MaxBackups:    1,
		EnableConsole: false,
		EnableFile:    true,
	}

	err := logging.Setup(cfg)
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}

	logger := logging.L()
	logger.Info("test_event", logging.Count(42), logging.Path("/test/path"))
	logging.Sync()

	// Read and parse the log file
	logPath := filepath.Join(tmpDir, "jsonl-test.jsonl")
	content, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read log file: %v", err)
	}

	// Each line should be valid JSON
	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) == 0 {
		t.Fatal("no log lines written")
	}

	for i, line := range lines {
		if line == "" {
			continue
		}
		var entry map[string]interface{}
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			t.Errorf("line %d is not valid JSON: %v\nLine: %s", i, err, line)
		}

		// Verify expected fields for Athena compatibility
		if _, ok := entry["timestamp"]; !ok {
			t.Error("log entry missing 'timestamp' field")
		}
		if _, ok := entry["level"]; !ok {
			t.Error("log entry missing 'level' field")
		}
		if _, ok := entry["msg"]; !ok {
			t.Error("log entry missing 'msg' field")
		}
		if _, ok := entry["service"]; !ok {
			t.Error("log entry missing 'service' field")
		}
	}
}

func TestLoggerWithContext(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &logging.Config{
		Level:         "debug",
		LogDir:        tmpDir,
		LogFile:       "context-test.jsonl",
		MaxSizeMB:     1,
		MaxBackups:    1,
		EnableConsole: false,
		EnableFile:    true,
	}

	err := logging.Setup(cfg)
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}

	// Create child logger with context
	logger := logging.WithContext("req-123", "ingestion")
	logger.Info("processing_started")
	logging.Sync()

	// Read and verify context fields
	logPath := filepath.Join(tmpDir, "context-test.jsonl")
	content, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read log file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) == 0 {
		t.Fatal("no log lines written")
	}

	var entry map[string]interface{}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &entry); err != nil {
		t.Fatalf("failed to parse log line: %v", err)
	}

	if entry["request_id"] != "req-123" {
		t.Errorf("expected request_id 'req-123', got %v", entry["request_id"])
	}
	if entry["operation"] != "ingestion" {
		t.Errorf("expected operation 'ingestion', got %v", entry["operation"])
	}
}

func TestFieldConstructors(t *testing.T) {
	// Test that field constructors don't panic
	fields := []struct {
		name  string
		field interface{}
	}{
		{"Path", logging.Path("/var/log/app.log")},
		{"Count", logging.Count(100)},
		{"Duration", logging.Duration(1000)},
		{"ErrorCode", logging.ErrorCode("SENTINEL_1001")},
		{"BatchSize", logging.BatchSize(32)},
		{"Source", logging.Source("file")},
		{"ClusterID", logging.ClusterID("cluster-1")},
		{"NoveltyScore", logging.NoveltyScore(0.85)},
	}

	for _, tc := range fields {
		t.Run(tc.name, func(t *testing.T) {
			if tc.field == nil {
				t.Errorf("%s returned nil", tc.name)
			}
		})
	}
}

func TestLogLevels(t *testing.T) {
	tmpDir := t.TempDir()

	testCases := []struct {
		level    string
		logFunc  func()
		expected bool // whether the message should appear
	}{
		{
			level: "debug",
			logFunc: func() {
				logging.L().Debug("debug message")
			},
			expected: true,
		},
		{
			level: "info",
			logFunc: func() {
				logging.L().Debug("debug message")
			},
			expected: false, // debug filtered at info level
		},
		{
			level: "warn",
			logFunc: func() {
				logging.L().Info("info message")
			},
			expected: false, // info filtered at warn level
		},
		{
			level: "error",
			logFunc: func() {
				logging.L().Warn("warn message")
			},
			expected: false, // warn filtered at error level
		},
	}

	for _, tc := range testCases {
		t.Run(tc.level, func(t *testing.T) {
			logFile := tc.level + "-test.jsonl"
			cfg := &logging.Config{
				Level:         tc.level,
				LogDir:        tmpDir,
				LogFile:       logFile,
				MaxSizeMB:     1,
				MaxBackups:    1,
				EnableConsole: false,
				EnableFile:    true,
			}

			err := logging.Setup(cfg)
			if err != nil {
				t.Fatalf("Setup failed: %v", err)
			}

			tc.logFunc()
			logging.Sync()

			logPath := filepath.Join(tmpDir, logFile)
			content, err := os.ReadFile(logPath)
			if err != nil && !os.IsNotExist(err) {
				t.Fatalf("failed to read log file: %v", err)
			}

			hasContent := len(strings.TrimSpace(string(content))) > 0
			if hasContent != tc.expected {
				t.Errorf("at level %s, expected content=%v, got content=%v", tc.level, tc.expected, hasContent)
			}
		})
	}
}

func TestSugaredLogger(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := &logging.Config{
		Level:         "info",
		LogDir:        tmpDir,
		LogFile:       "sugar-test.jsonl",
		MaxSizeMB:     1,
		MaxBackups:    1,
		EnableConsole: false,
		EnableFile:    true,
	}

	err := logging.Setup(cfg)
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}

	sugar := logging.S()
	sugar.Infow("structured_log",
		"key1", "value1",
		"key2", 42,
	)
	logging.Sync()

	logPath := filepath.Join(tmpDir, "sugar-test.jsonl")
	content, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read log file: %v", err)
	}

	var entry map[string]interface{}
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(content))), &entry); err != nil {
		t.Fatalf("failed to parse log line: %v", err)
	}

	if entry["key1"] != "value1" {
		t.Errorf("expected key1='value1', got %v", entry["key1"])
	}
	if entry["key2"] != float64(42) {
		t.Errorf("expected key2=42, got %v", entry["key2"])
	}
}

func TestLogDirectoryCreation(t *testing.T) {
	tmpDir := t.TempDir()
	nestedDir := filepath.Join(tmpDir, "nested", "logs", "dir")

	cfg := &logging.Config{
		Level:         "info",
		LogDir:        nestedDir,
		LogFile:       "nested.jsonl",
		MaxSizeMB:     1,
		MaxBackups:    1,
		EnableConsole: false,
		EnableFile:    true,
	}

	err := logging.Setup(cfg)
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}

	logging.L().Info("test")
	logging.Sync()

	// Verify nested directory was created
	if _, err := os.Stat(nestedDir); os.IsNotExist(err) {
		t.Error("nested log directory was not created")
	}
}
