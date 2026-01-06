// Package models defines the core data structures used across the agent.
package models

import (
	"encoding/json"
	"time"
)

// LogRecord represents a canonical log entry used across ingestion, ML, and storage.
// This is the Go equivalent of the Python LogRecord model.
type LogRecord struct {
	// Timestamp of the log entry (optional, may be parsed from message)
	Timestamp *time.Time `json:"timestamp,omitempty"`

	// Level is the log level (e.g., INFO, WARN, ERROR)
	Level string `json:"level,omitempty"`

	// Message is the main log message content
	Message string `json:"message"`

	// Source identifies where this log came from (file path, journald unit, etc.)
	Source string `json:"source"`

	// Raw is the original unparsed log line
	Raw string `json:"raw"`

	// Attrs contains additional structured attributes
	Attrs map[string]any `json:"attrs,omitempty"`

	// Normalized is the message after masking/normalization (for ML)
	Normalized string `json:"normalized,omitempty"`
}

// ToJSON serializes the LogRecord to JSON bytes.
func (l *LogRecord) ToJSON() ([]byte, error) {
	return json.Marshal(l)
}

// FromJSON deserializes a LogRecord from JSON bytes.
func FromJSON(data []byte) (*LogRecord, error) {
	var record LogRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, err
	}
	return &record, nil
}

// ClusterSummary represents a cluster of similar log messages.
type ClusterSummary struct {
	// ClusterID is a stable identifier for this cluster
	ClusterID string `json:"cluster_id"`

	// Size is the number of logs in this cluster
	Size int `json:"size"`

	// Representative is the most representative log message
	Representative string `json:"representative"`

	// Keywords are the top keywords/tokens in this cluster
	Keywords []string `json:"keywords,omitempty"`

	// NoveltyScore indicates how novel this cluster is (0.0 = seen before, 1.0 = completely new)
	NoveltyScore float64 `json:"novelty_score"`

	// FirstSeen is when this cluster was first observed
	FirstSeen time.Time `json:"first_seen"`

	// LastSeen is when this cluster was last observed
	LastSeen time.Time `json:"last_seen"`
}

// Explanation represents an LLM-generated explanation for a cluster.
type Explanation struct {
	// ClusterID this explanation is for
	ClusterID string `json:"cluster_id"`

	// RootCause is the probable root cause
	RootCause string `json:"root_cause"`

	// NextSteps are suggested actions to investigate
	NextSteps []string `json:"next_steps"`

	// Remediation is a suggested fix if applicable
	Remediation string `json:"remediation,omitempty"`

	// Confidence is the confidence level (Low, Medium, High)
	Confidence string `json:"confidence"`

	// ConfidenceScore is the numeric confidence (0.0-1.0)
	ConfidenceScore float64 `json:"confidence_score"`

	// Reasoning explains why this confidence was assigned
	Reasoning string `json:"reasoning,omitempty"`

	// GeneratedAt is when this explanation was created
	GeneratedAt time.Time `json:"generated_at"`
}
