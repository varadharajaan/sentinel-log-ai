package grpcclient

import (
	"context"
	"testing"
	"time"

	"sentinel-log-ai/internal/logging"
	"sentinel-log-ai/internal/models"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// TestNewClient tests client creation with various configurations.
func TestNewClient(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	tests := []struct {
		name      string
		config    *Config
		wantErr   bool
		errSubstr string
	}{
		{
			name:    "default config",
			config:  nil,
			wantErr: false,
		},
		{
			name: "valid custom config",
			config: &Config{
				Address:        "localhost:50052",
				ConnectTimeout: 5 * time.Second,
				RequestTimeout: 15 * time.Second,
				MaxRetries:     5,
				RetryBackoff:   50 * time.Millisecond,
				MaxBackoff:     2 * time.Second,
				Logger:         zap.NewNop(),
			},
			wantErr: false,
		},
		{
			name: "empty address",
			config: &Config{
				Address:        "",
				ConnectTimeout: 5 * time.Second,
				RequestTimeout: 15 * time.Second,
			},
			wantErr:   true,
			errSubstr: "Address",
		},
		{
			name: "invalid ConnectTimeout",
			config: &Config{
				Address:        "localhost:50051",
				ConnectTimeout: 0,
				RequestTimeout: 15 * time.Second,
			},
			wantErr:   true,
			errSubstr: "ConnectTimeout",
		},
		{
			name: "invalid RequestTimeout",
			config: &Config{
				Address:        "localhost:50051",
				ConnectTimeout: 5 * time.Second,
				RequestTimeout: 0,
			},
			wantErr:   true,
			errSubstr: "RequestTimeout",
		},
		{
			name: "negative MaxRetries",
			config: &Config{
				Address:        "localhost:50051",
				ConnectTimeout: 5 * time.Second,
				RequestTimeout: 15 * time.Second,
				MaxRetries:     -1,
			},
			wantErr:   true,
			errSubstr: "MaxRetries",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.config)
			if tt.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.errSubstr)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, client)
			defer func() { _ = client.Close() }()
		})
	}
}

// TestDefaultConfig tests the default configuration values.
func TestDefaultConfig(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	cfg := DefaultConfig()

	assert.Equal(t, "localhost:50051", cfg.Address)
	assert.Equal(t, 10*time.Second, cfg.ConnectTimeout)
	assert.Equal(t, 30*time.Second, cfg.RequestTimeout)
	assert.Equal(t, 3, cfg.MaxRetries)
	assert.Equal(t, 100*time.Millisecond, cfg.RetryBackoff)
	assert.Equal(t, 5*time.Second, cfg.MaxBackoff)
	assert.True(t, cfg.EnableCompression)
	assert.Equal(t, 16*1024*1024, cfg.MaxMessageSize)
}

// TestConfigValidate tests configuration validation.
func TestConfigValidate(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	tests := []struct {
		name    string
		config  *Config
		wantErr bool
	}{
		{
			name:    "valid default config",
			config:  DefaultConfig(),
			wantErr: false,
		},
		{
			name: "valid minimal config",
			config: &Config{
				Address:        "localhost:50051",
				ConnectTimeout: 1 * time.Second,
				RequestTimeout: 1 * time.Second,
				MaxRetries:     0,
			},
			wantErr: false,
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

// TestClientClose tests client closure.
func TestClientClose(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	cfg := &Config{
		Address:        "localhost:50051",
		ConnectTimeout: 5 * time.Second,
		RequestTimeout: 15 * time.Second,
		Logger:         zap.NewNop(),
	}

	client, err := NewClient(cfg)
	require.NoError(t, err)

	// First close should succeed
	err = client.Close()
	assert.NoError(t, err)

	// Second close should be idempotent
	err = client.Close()
	assert.NoError(t, err)
}

// TestClientIsConnected tests connection status check.
func TestClientIsConnected(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	cfg := &Config{
		Address:        "localhost:50051",
		ConnectTimeout: 5 * time.Second,
		RequestTimeout: 15 * time.Second,
		Logger:         zap.NewNop(),
	}

	client, err := NewClient(cfg)
	require.NoError(t, err)
	defer func() { _ = client.Close() }()

	// Should not be connected initially (lazy connection)
	assert.False(t, client.IsConnected())
}

// TestConvertToProtoRecord tests log record conversion.
func TestConvertToProtoRecord(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	now := time.Now()

	tests := []struct {
		name     string
		record   *models.LogRecord
		expected *LogRecord
	}{
		{
			name:     "nil record",
			record:   nil,
			expected: nil,
		},
		{
			name: "full record",
			record: &models.LogRecord{
				Message:    "test message",
				Normalized: "normalized message",
				Level:      "ERROR",
				Source:     "/var/log/test.log",
				Timestamp:  &now,
				Attrs: map[string]any{
					"key": "value",
				},
			},
			expected: &LogRecord{
				Message:    "test message",
				Normalized: "normalized message",
				Level:      "ERROR",
				Source:     "/var/log/test.log",
				Timestamp:  &now,
				AttrsJSON:  `{"key":"value"}`,
			},
		},
		{
			name: "minimal record",
			record: &models.LogRecord{
				Message: "simple message",
				Source:  "stdin",
			},
			expected: &LogRecord{
				Message: "simple message",
				Source:  "stdin",
			},
		},
		{
			name: "record without attrs",
			record: &models.LogRecord{
				Message: "no attrs",
				Source:  "test",
				Attrs:   nil,
			},
			expected: &LogRecord{
				Message: "no attrs",
				Source:  "test",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertToProtoRecord(tt.record)

			if tt.expected == nil {
				assert.Nil(t, result)
				return
			}

			require.NotNil(t, result)
			assert.Equal(t, tt.expected.Message, result.Message)
			assert.Equal(t, tt.expected.Normalized, result.Normalized)
			assert.Equal(t, tt.expected.Level, result.Level)
			assert.Equal(t, tt.expected.Source, result.Source)

			if tt.expected.Timestamp != nil {
				require.NotNil(t, result.Timestamp)
				assert.Equal(t, *tt.expected.Timestamp, *result.Timestamp)
			}

			if tt.expected.AttrsJSON != "" {
				assert.JSONEq(t, tt.expected.AttrsJSON, result.AttrsJSON)
			}
		})
	}
}

// TestConvertToTimestamp tests timestamp conversion.
func TestConvertToTimestamp(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	now := time.Now()
	zero := time.Time{}

	tests := []struct {
		name  string
		input *time.Time
		isNil bool
	}{
		{
			name:  "nil time",
			input: nil,
			isNil: true,
		},
		{
			name:  "zero time",
			input: &zero,
			isNil: true,
		},
		{
			name:  "valid time",
			input: &now,
			isNil: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertToTimestamp(tt.input)
			if tt.isNil {
				assert.Nil(t, result)
			} else {
				require.NotNil(t, result)
				assert.Equal(t, tt.input.Unix(), result.AsTime().Unix())
			}
		})
	}
}

// TestConvertFromTimestamp tests protobuf to time.Time conversion.
func TestConvertFromTimestamp(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	tests := []struct {
		name  string
		input interface{}
		isNil bool
	}{
		{
			name:  "nil timestamp",
			input: nil,
			isNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertFromTimestamp(nil)
			if tt.isNil {
				assert.Nil(t, result)
			}
		})
	}
}

// TestBatchHandlerCreation tests batch handler creation.
func TestBatchHandlerCreation(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	cfg := &Config{
		Address:        "localhost:50051",
		ConnectTimeout: 5 * time.Second,
		RequestTimeout: 15 * time.Second,
		Logger:         zap.NewNop(),
	}

	client, err := NewClient(cfg)
	require.NoError(t, err)
	defer func() { _ = client.Close() }()

	handler := NewBatchHandler(client)
	require.NotNil(t, handler)
}

// TestBatchHandlerEmptyBatch tests handling of empty batch.
func TestBatchHandlerEmptyBatch(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	cfg := &Config{
		Address:        "localhost:50051",
		ConnectTimeout: 5 * time.Second,
		RequestTimeout: 15 * time.Second,
		Logger:         zap.NewNop(),
	}

	client, err := NewClient(cfg)
	require.NoError(t, err)
	defer func() { _ = client.Close() }()

	handler := NewBatchHandler(client)

	processed, err := handler.HandleBatch(context.Background(), nil)
	assert.NoError(t, err)
	assert.Equal(t, 0, processed)

	processed, err = handler.HandleBatch(context.Background(), []*models.LogRecord{})
	assert.NoError(t, err)
	assert.Equal(t, 0, processed)
}

// TestIsRetryableError tests error classification for retry logic.
func TestIsRetryableError(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	tests := []struct {
		name        string
		err         error
		isRetryable bool
	}{
		{
			name:        "nil error",
			err:         nil,
			isRetryable: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRetryableError(tt.err)
			assert.Equal(t, tt.isRetryable, result)
		})
	}
}

// TestClientGetConnection tests getting the underlying connection.
func TestClientGetConnection(t *testing.T) {
	t.Cleanup(func() { _ = logging.Close() })

	cfg := &Config{
		Address:        "localhost:50051",
		ConnectTimeout: 5 * time.Second,
		RequestTimeout: 15 * time.Second,
		Logger:         zap.NewNop(),
	}

	client, err := NewClient(cfg)
	require.NoError(t, err)
	defer func() { _ = client.Close() }()

	// Connection should be nil before Connect()
	conn := client.GetConnection()
	assert.Nil(t, conn)
}
