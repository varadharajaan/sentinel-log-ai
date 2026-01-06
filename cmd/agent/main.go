// Package main provides the entry point for the sentinel-log-ai agent.
// The agent is responsible for log ingestion, streaming, and orchestrating
// ML/LLM analysis via gRPC communication with the Python ML service.
package main

import (
	"os"

	"github.com/sentinel-log-ai/sentinel-log-ai/cmd/agent/cmd"
)

func main() {
	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
