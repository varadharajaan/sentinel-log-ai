package models

import (
	"encoding/json"
	"testing"
	"time"
)

func TestLogRecord_ToJSON(t *testing.T) {
	tests := []struct {
		name    string
		record  LogRecord
		wantErr bool
	}{
		{
			name: "minimal record",
			record: LogRecord{
				Message: "test message",
				Source:  "/var/log/test.log",
				Raw:     "raw log line",
			},
			wantErr: false,
		},
		{
			name: "full record",
			record: LogRecord{
				Timestamp:  timePtr(time.Date(2024, 1, 15, 10, 30, 0, 0, time.UTC)),
				Level:      "ERROR",
				Message:    "connection failed",
				Source:     "/var/log/app.log",
				Raw:        "2024-01-15 ERROR connection failed",
				Attrs:      map[string]any{"host": "server-01", "port": 5432},
				Normalized: "connection failed",
			},
			wantErr: false,
		},
		{
			name: "record with empty message",
			record: LogRecord{
				Message: "",
				Source:  "stdin",
				Raw:     "",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.record.ToJSON()
			if (err != nil) != tt.wantErr {
				t.Errorf("LogRecord.ToJSON() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && len(got) == 0 {
				t.Error("LogRecord.ToJSON() returned empty bytes")
			}
		})
	}
}

func TestLogRecord_FromJSON(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		want    *LogRecord
		wantErr bool
	}{
		{
			name: "valid minimal JSON",
			json: `{"message":"test","source":"test.log","raw":"test"}`,
			want: &LogRecord{
				Message: "test",
				Source:  "test.log",
				Raw:     "test",
			},
			wantErr: false,
		},
		{
			name: "valid full JSON",
			json: `{
				"timestamp":"2024-01-15T10:30:00Z",
				"level":"ERROR",
				"message":"connection failed",
				"source":"/var/log/app.log",
				"raw":"2024-01-15 ERROR connection failed",
				"normalized":"connection failed",
				"attrs":{"host":"server-01"}
			}`,
			want: &LogRecord{
				Level:      "ERROR",
				Message:    "connection failed",
				Source:     "/var/log/app.log",
				Raw:        "2024-01-15 ERROR connection failed",
				Normalized: "connection failed",
			},
			wantErr: false,
		},
		{
			name:    "invalid JSON",
			json:    `{invalid`,
			want:    nil,
			wantErr: true,
		},
		{
			name:    "empty JSON",
			json:    `{}`,
			want:    &LogRecord{},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := FromJSON([]byte(tt.json))
			if (err != nil) != tt.wantErr {
				t.Errorf("FromJSON() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if got.Message != tt.want.Message {
					t.Errorf("FromJSON() Message = %v, want %v", got.Message, tt.want.Message)
				}
				if got.Source != tt.want.Source {
					t.Errorf("FromJSON() Source = %v, want %v", got.Source, tt.want.Source)
				}
			}
		})
	}
}

func TestLogRecord_RoundTrip(t *testing.T) {
	original := LogRecord{
		Timestamp:  timePtr(time.Date(2024, 1, 15, 10, 30, 0, 0, time.UTC)),
		Level:      "WARN",
		Message:    "disk space low",
		Source:     "/var/log/system.log",
		Raw:        "Jan 15 10:30:00 WARN disk space low",
		Attrs:      map[string]any{"disk": "/dev/sda1", "percent": 95.5},
		Normalized: "disk space low",
	}

	// Serialize
	jsonBytes, err := original.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON() failed: %v", err)
	}

	// Deserialize
	restored, err := FromJSON(jsonBytes)
	if err != nil {
		t.Fatalf("FromJSON() failed: %v", err)
	}

	// Compare key fields
	if restored.Level != original.Level {
		t.Errorf("Level mismatch: got %v, want %v", restored.Level, original.Level)
	}
	if restored.Message != original.Message {
		t.Errorf("Message mismatch: got %v, want %v", restored.Message, original.Message)
	}
	if restored.Source != original.Source {
		t.Errorf("Source mismatch: got %v, want %v", restored.Source, original.Source)
	}
	if restored.Raw != original.Raw {
		t.Errorf("Raw mismatch: got %v, want %v", restored.Raw, original.Raw)
	}
	if restored.Normalized != original.Normalized {
		t.Errorf("Normalized mismatch: got %v, want %v", restored.Normalized, original.Normalized)
	}
}

func TestClusterSummary_JSON(t *testing.T) {
	now := time.Now()
	cluster := ClusterSummary{
		ClusterID:      "cluster-001",
		Size:           150,
		Representative: "Connection timeout to database",
		Keywords:       []string{"connection", "timeout", "database"},
		NoveltyScore:   0.2,
		FirstSeen:      now,
		LastSeen:       now,
	}

	jsonBytes, err := json.Marshal(cluster)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var restored ClusterSummary
	if err := json.Unmarshal(jsonBytes, &restored); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if restored.ClusterID != cluster.ClusterID {
		t.Errorf("ClusterID mismatch: got %v, want %v", restored.ClusterID, cluster.ClusterID)
	}
	if restored.Size != cluster.Size {
		t.Errorf("Size mismatch: got %v, want %v", restored.Size, cluster.Size)
	}
	if len(restored.Keywords) != len(cluster.Keywords) {
		t.Errorf("Keywords length mismatch: got %v, want %v", len(restored.Keywords), len(cluster.Keywords))
	}
}

func TestExplanation_JSON(t *testing.T) {
	now := time.Now()
	explanation := Explanation{
		ClusterID:       "cluster-001",
		RootCause:       "Database connection pool exhausted",
		NextSteps:       []string{"Check connection pool size", "Review slow queries"},
		Remediation:     "Increase pool size in config",
		Confidence:      "HIGH",
		ConfidenceScore: 0.92,
		Reasoning:       "High cluster cohesion and matching historical patterns",
		GeneratedAt:     now,
	}

	jsonBytes, err := json.Marshal(explanation)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var restored Explanation
	if err := json.Unmarshal(jsonBytes, &restored); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if restored.ClusterID != explanation.ClusterID {
		t.Errorf("ClusterID mismatch: got %v, want %v", restored.ClusterID, explanation.ClusterID)
	}
	if restored.RootCause != explanation.RootCause {
		t.Errorf("RootCause mismatch: got %v, want %v", restored.RootCause, explanation.RootCause)
	}
	if restored.Confidence != explanation.Confidence {
		t.Errorf("Confidence mismatch: got %v, want %v", restored.Confidence, explanation.Confidence)
	}
	if restored.ConfidenceScore != explanation.ConfidenceScore {
		t.Errorf("ConfidenceScore mismatch: got %v, want %v", restored.ConfidenceScore, explanation.ConfidenceScore)
	}
	if len(restored.NextSteps) != len(explanation.NextSteps) {
		t.Errorf("NextSteps length mismatch: got %v, want %v", len(restored.NextSteps), len(explanation.NextSteps))
	}
}

// Helper function
func timePtr(t time.Time) *time.Time {
	return &t
}
