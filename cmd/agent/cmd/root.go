// Package cmd provides the CLI commands for the sentinel-log-ai agent.
package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

var (
	cfgFile string
	verbose bool
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "sentinel-log-ai",
	Short: "AI-powered log intelligence engine",
	Long: `Sentinel Log AI is an intelligent log analysis agent that:
  - Ingests logs from files, journald, or stdin
  - Streams logs to the ML engine for embedding and clustering
  - Detects novel/unseen error patterns in real-time
  - Provides LLM-powered explanations for log clusters

The agent (Go) handles high-performance log ingestion while the ML engine
(Python) handles embeddings, clustering, and LLM integration via gRPC.`,
	Version: "0.1.0",
}

// Execute adds all child commands to the root command and sets flags appropriately.
func Execute() error {
	return rootCmd.Execute()
}

func init() {
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is ./sentinel-log-ai.yaml)")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "enable verbose output")
	
	// Add subcommands
	rootCmd.AddCommand(ingestCmd)
	rootCmd.AddCommand(analyzeCmd)
	rootCmd.AddCommand(novelCmd)
	rootCmd.AddCommand(explainCmd)
	rootCmd.AddCommand(serveCmd)
}

// ingestCmd represents the ingest command
var ingestCmd = &cobra.Command{
	Use:   "ingest [path]",
	Short: "Ingest logs from a file or directory",
	Long: `Ingest logs from the specified file or directory.
Supports batch mode (read entire file) and tail mode (follow new lines).

Examples:
  sentinel-log-ai ingest /var/log/syslog
  sentinel-log-ai ingest /var/log/nginx/ --pattern "*.log"
  sentinel-log-ai ingest /var/log/app.log --tail`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		path := args[0]
		tail, _ := cmd.Flags().GetBool("tail")
		fmt.Printf("Ingesting logs from: %s (tail=%v)\n", path, tail)
		// TODO: Implement ingestion logic
		return nil
	},
}

// analyzeCmd represents the analyze command
var analyzeCmd = &cobra.Command{
	Use:   "analyze",
	Short: "Analyze ingested logs for patterns and clusters",
	Long: `Run clustering analysis on ingested logs to identify patterns.
Shows cluster summaries with representative log messages.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		last, _ := cmd.Flags().GetString("last")
		fmt.Printf("Analyzing logs from last: %s\n", last)
		// TODO: Implement analysis logic
		return nil
	},
}

// novelCmd represents the novel command
var novelCmd = &cobra.Command{
	Use:   "novel",
	Short: "Detect novel/unseen log patterns",
	Long: `Monitor for novel log patterns that don't match known clusters.
Use --follow to continuously watch for new anomalies.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		follow, _ := cmd.Flags().GetBool("follow")
		fmt.Printf("Detecting novel patterns (follow=%v)\n", follow)
		// TODO: Implement novelty detection
		return nil
	},
}

// explainCmd represents the explain command
var explainCmd = &cobra.Command{
	Use:   "explain [cluster-id]",
	Short: "Get LLM explanation for a log cluster",
	Long: `Use LLM to generate a human-readable explanation for a log cluster.
Includes probable root cause, suggested next steps, and confidence score.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		clusterID := args[0]
		fmt.Printf("Explaining cluster: %s\n", clusterID)
		// TODO: Implement LLM explanation
		return nil
	},
}

// serveCmd represents the serve command
var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the agent in server mode",
	Long: `Start the agent as a long-running service that:
  - Watches configured log sources
  - Streams to ML engine via gRPC
  - Exposes metrics and health endpoints`,
	RunE: func(cmd *cobra.Command, args []string) error {
		port, _ := cmd.Flags().GetInt("port")
		fmt.Printf("Starting agent server on port %d\n", port)
		// TODO: Implement server mode
		return nil
	},
}

func init() {
	ingestCmd.Flags().Bool("tail", false, "follow file for new lines (like tail -f)")
	ingestCmd.Flags().String("pattern", "*", "glob pattern for files in directory mode")
	
	analyzeCmd.Flags().String("last", "1h", "analyze logs from last duration (e.g., 1h, 30m, 24h)")
	
	novelCmd.Flags().Bool("follow", false, "continuously monitor for novel patterns")
	novelCmd.Flags().Float64("threshold", 0.7, "novelty score threshold (0.0-1.0)")
	
	serveCmd.Flags().Int("port", 8080, "HTTP server port for metrics/health")
	serveCmd.Flags().String("ml-addr", "localhost:50051", "ML engine gRPC address")
}
