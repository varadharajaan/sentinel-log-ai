// Package logging provides structured JSONL logging for the sentinel-log-ai agent.
//
// Features:
// - JSONL (JSON Lines) format for AWS Athena and analytics pipelines
// - Rolling log files with configurable size and backup count
// - Structured fields for correlation and analysis
// - No ASCII art or decorations - clean machine-readable output
//
// Log Format:
// Each log entry is a single JSON object on its own line:
//
//	{"level":"info","ts":"2024-01-15T10:30:00.000Z","service":"sentinel-agent","msg":"ingestion_started","path":"/var/log/app.log"}
package logging

import (
	"os"
	"path/filepath"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/natefinch/lumberjack.v2"
)

// Config holds logging configuration.
type Config struct {
	// Level is the minimum log level (debug, info, warn, error)
	Level string
	// LogDir is the directory for log files
	LogDir string
	// LogFile is the log filename (not full path)
	LogFile string
	// MaxSizeMB is the maximum size in MB before rotation
	MaxSizeMB int
	// MaxBackups is the number of backup files to keep
	MaxBackups int
	// MaxAgeDays is the maximum age in days to retain logs
	MaxAgeDays int
	// EnableConsole enables console output
	EnableConsole bool
	// EnableFile enables file output
	EnableFile bool
	// ConsoleFormat is the console format (json, plain)
	ConsoleFormat string
}

// DefaultConfig returns the default logging configuration.
func DefaultConfig() *Config {
	return &Config{
		Level:         "info",
		LogDir:        "logs",
		LogFile:       "sentinel-agent.jsonl",
		MaxSizeMB:     10,
		MaxBackups:    5,
		MaxAgeDays:    30,
		EnableConsole: true,
		EnableFile:    true,
		ConsoleFormat: "plain",
	}
}

var (
	// globalLogger is the package-level logger instance
	globalLogger *zap.Logger
	// globalSugar is the sugared logger for convenience
	globalSugar *zap.SugaredLogger
	// fileWriter holds the rotating file writer for cleanup
	fileWriter *lumberjack.Logger
)

// Setup initializes the global logger with the given configuration.
func Setup(cfg *Config) error {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	// Parse log level
	level, err := parseLevel(cfg.Level)
	if err != nil {
		level = zapcore.InfoLevel
	}

	// Encoder config for JSONL output - optimized for Athena
	jsonEncoder := zapcore.EncoderConfig{
		TimeKey:        "timestamp",
		LevelKey:       "level",
		NameKey:        "logger",
		CallerKey:      "caller",
		FunctionKey:    zapcore.OmitKey,
		MessageKey:     "msg",
		StacktraceKey:  "stacktrace",
		LineEnding:     zapcore.DefaultLineEnding,
		EncodeLevel:    zapcore.LowercaseLevelEncoder,
		EncodeTime:     zapcore.ISO8601TimeEncoder,
		EncodeDuration: zapcore.MillisDurationEncoder,
		EncodeCaller:   zapcore.ShortCallerEncoder,
	}

	// Console encoder config
	consoleEncoder := zapcore.EncoderConfig{
		TimeKey:        "ts",
		LevelKey:       "level",
		NameKey:        "logger",
		CallerKey:      "caller",
		FunctionKey:    zapcore.OmitKey,
		MessageKey:     "msg",
		StacktraceKey:  "stacktrace",
		LineEnding:     zapcore.DefaultLineEnding,
		EncodeLevel:    zapcore.CapitalLevelEncoder,
		EncodeTime:     zapcore.TimeEncoderOfLayout("2006-01-02T15:04:05.000Z07:00"),
		EncodeDuration: zapcore.StringDurationEncoder,
		EncodeCaller:   zapcore.ShortCallerEncoder,
	}

	var cores []zapcore.Core

	// File core with rotation
	if cfg.EnableFile {
		// Ensure log directory exists
		logPath := filepath.Join(cfg.LogDir, cfg.LogFile)
		if err := os.MkdirAll(cfg.LogDir, 0755); err != nil {
			return err
		}

		// Rotating file writer using lumberjack
		fileWriter := &lumberjack.Logger{
			Filename:   logPath,
			MaxSize:    cfg.MaxSizeMB,
			MaxBackups: cfg.MaxBackups,
			MaxAge:     cfg.MaxAgeDays,
			Compress:   true,
			LocalTime:  false, // Use UTC for consistency
		}

		fileCore := zapcore.NewCore(
			zapcore.NewJSONEncoder(jsonEncoder),
			zapcore.AddSync(fileWriter),
			level,
		)
		cores = append(cores, fileCore)
	}

	// Console core
	if cfg.EnableConsole {
		var encoder zapcore.Encoder
		if cfg.ConsoleFormat == "json" {
			encoder = zapcore.NewJSONEncoder(jsonEncoder)
		} else {
			encoder = zapcore.NewConsoleEncoder(consoleEncoder)
		}

		consoleCore := zapcore.NewCore(
			encoder,
			zapcore.AddSync(os.Stdout),
			level,
		)
		cores = append(cores, consoleCore)
	}

	// Combine cores
	core := zapcore.NewTee(cores...)

	// Build logger with additional fields
	hostname, _ := os.Hostname()
	globalLogger = zap.New(core, zap.AddCaller(), zap.AddStacktrace(zapcore.ErrorLevel)).With(
		zap.String("service", "sentinel-agent"),
		zap.String("hostname", hostname),
		zap.Int("pid", os.Getpid()),
	)
	globalSugar = globalLogger.Sugar()

	return nil
}

// parseLevel converts a string level to zapcore.Level.
func parseLevel(level string) (zapcore.Level, error) {
	var l zapcore.Level
	err := l.UnmarshalText([]byte(level))
	return l, err
}

// L returns the global logger.
func L() *zap.Logger {
	if globalLogger == nil {
		_ = Setup(DefaultConfig())
	}
	return globalLogger
}

// S returns the global sugared logger.
func S() *zap.SugaredLogger {
	if globalSugar == nil {
		_ = Setup(DefaultConfig())
	}
	return globalSugar
}

// With creates a child logger with additional fields.
func With(fields ...zap.Field) *zap.Logger {
	return L().With(fields...)
}

// WithContext creates a child logger with context fields for request tracking.
func WithContext(requestID string, operationType string) *zap.Logger {
	return L().With(
		zap.String("request_id", requestID),
		zap.String("operation", operationType),
	)
}

// Sync flushes any buffered log entries.
func Sync() error {
	if globalLogger != nil {
		return globalLogger.Sync()
	}
	return nil
}

// Field constructors for common log fields

// Path returns a field for file/directory paths.
func Path(path string) zap.Field {
	return zap.String("path", path)
}

// Count returns a field for counts/quantities.
func Count(n int) zap.Field {
	return zap.Int("count", n)
}

// Duration returns a field for time durations.
func Duration(d time.Duration) zap.Field {
	return zap.Duration("duration", d)
}

// ErrorCode returns a field for sentinel error codes.
func ErrorCode(code string) zap.Field {
	return zap.String("error_code", code)
}

// BatchSize returns a field for batch sizes.
func BatchSize(size int) zap.Field {
	return zap.Int("batch_size", size)
}

// Source returns a field for log sources.
func Source(src string) zap.Field {
	return zap.String("source", src)
}

// ClusterID returns a field for cluster identifiers.
func ClusterID(id string) zap.Field {
	return zap.String("cluster_id", id)
}

// NoveltyScore returns a field for novelty scores.
func NoveltyScore(score float64) zap.Field {
	return zap.Float64("novelty_score", score)
}
