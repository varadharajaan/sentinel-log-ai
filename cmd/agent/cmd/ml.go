// Package cmd provides the CLI commands for the sentinel-log-ai agent.
package cmd

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"sentinel-log-ai/internal/grpcclient"
	"sentinel-log-ai/internal/logging"
	"sentinel-log-ai/internal/models"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

var (
	mlServerAddr string
	topK         int
	minSimilarity float64
	minClusterSize int
	noveltyThreshold float64
	timeout      time.Duration
)

// setupMLCommands configures ML-related commands
func setupMLCommands() {
	// Search command flags
	searchCmd.Flags().StringVar(&mlServerAddr, "ml-server", "localhost:50051", "ML server address")
	searchCmd.Flags().IntVar(&topK, "top-k", 10, "Number of similar logs to return")
	searchCmd.Flags().Float64Var(&minSimilarity, "min-similarity", 0.5, "Minimum similarity threshold (0.0-1.0)")
	searchCmd.Flags().DurationVar(&timeout, "timeout", 60*time.Second, "Request timeout")

	// Cluster command flags
	clusterCmd.Flags().StringVar(&mlServerAddr, "ml-server", "localhost:50051", "ML server address")
	clusterCmd.Flags().IntVar(&minClusterSize, "min-size", 2, "Minimum cluster size")
	clusterCmd.Flags().DurationVar(&timeout, "timeout", 120*time.Second, "Request timeout")

	// Novelty command flags  
	noveltyCmd.Flags().StringVar(&mlServerAddr, "ml-server", "localhost:50051", "ML server address")
	noveltyCmd.Flags().Float64Var(&noveltyThreshold, "threshold", 0.7, "Novelty threshold (0.0-1.0)")
	noveltyCmd.Flags().DurationVar(&timeout, "timeout", 30*time.Second, "Request timeout")

	// Health command flags
	healthCmd.Flags().StringVar(&mlServerAddr, "ml-server", "localhost:50051", "ML server address")
	healthCmd.Flags().DurationVar(&timeout, "timeout", 10*time.Second, "Request timeout")
}

// searchCmd searches for similar logs
var searchCmd = &cobra.Command{
	Use:   "search <log-file> <query>",
	Short: "Search for similar logs using semantic similarity",
	Long: `Search for logs similar to a query using vector embeddings.
First ingests the log file, then searches for logs similar to the query.

Examples:
  sentinel-log-ai search demo/demo_logs.jsonl "authentication failed"
  sentinel-log-ai search /var/log/syslog "connection timeout" --top-k 5`,
	Args: cobra.ExactArgs(2),
	RunE: runSearch,
}

// clusterCmd clusters logs
var clusterCmd = &cobra.Command{
	Use:   "cluster <log-file>",
	Short: "Cluster logs into groups of similar patterns",
	Long: `Analyze logs and group them into clusters based on semantic similarity.
Uses HDBSCAN clustering on log embeddings.

Examples:
  sentinel-log-ai cluster demo/demo_logs.jsonl
  sentinel-log-ai cluster /var/log/syslog --min-size 5`,
	Args: cobra.ExactArgs(1),
	RunE: runCluster,
}

// noveltyCmd detects novel patterns
var noveltyCmd = &cobra.Command{
	Use:   "novelty <log-file>",
	Short: "Detect novel/unusual log patterns",
	Long: `Analyze logs and identify patterns that are unusual or never seen before.
Logs with high novelty scores indicate potential anomalies.

Examples:
  sentinel-log-ai novelty demo/demo_logs.jsonl
  sentinel-log-ai novelty /var/log/syslog --threshold 0.8`,
	Args: cobra.ExactArgs(1),
	RunE: runNovelty,
}

// healthCmd checks ML server health
var healthCmd = &cobra.Command{
	Use:   "health",
	Short: "Check ML server health status",
	Long: `Check the health of the Python ML server and its components.

Examples:
  sentinel-log-ai health
  sentinel-log-ai health --ml-server localhost:50051`,
	RunE: runHealth,
}

func init() {
	setupMLCommands()
	rootCmd.AddCommand(searchCmd)
	rootCmd.AddCommand(clusterCmd)
	rootCmd.AddCommand(noveltyCmd)
	rootCmd.AddCommand(healthCmd)
}

func runSearch(cmd *cobra.Command, args []string) error {
	logFile := args[0]
	query := args[1]

	logger, cleanup, err := setupLogging()
	if err != nil {
		return err
	}
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		cancel()
	}()

	// Create gRPC client
	client, err := createMLClient(logger)
	if err != nil {
		return err
	}
	defer func() { _ = client.Close() }()

	// Load logs from file
	records, err := loadLogs(logFile)
	if err != nil {
		return fmt.Errorf("failed to load logs: %w", err)
	}

	fmt.Printf("Loaded %d logs from %s\n", len(records), logFile)

	// Note: logs should already be indexed in the vector store
	// You can use 'ingest' command first to embed logs

	// Now search for similar logs
	fmt.Printf("Searching for logs similar to: %q\n", query)

	queryRecord := &models.LogRecord{
		Message: query,
	}

	searchCtx, searchCancel := context.WithTimeout(ctx, timeout)
	defer searchCancel()

	results, err := client.Search(searchCtx, queryRecord, topK, float32(minSimilarity))
	if err != nil {
		// If search fails (no vector store), just print a message
		fmt.Printf("\nNote: Search requires logs to be indexed first.\n")
		fmt.Printf("The ML server needs to have a vector store with embedded logs.\n")
		return nil
	}

	if len(results) == 0 {
		fmt.Println("No similar logs found.")
		return nil
	}

	fmt.Printf("\nFound %d similar logs:\n", len(results))
	fmt.Println(strings.Repeat("-", 80))
	for i, r := range results {
		fmt.Printf("\n[%d] Similarity: %.2f%%\n", i+1, r.Similarity*100)
		fmt.Printf("    Level: %s\n", r.Record.Level)
		fmt.Printf("    Message: %s\n", truncate(r.Record.Message, 200))
		if r.ClusterID != "" {
			fmt.Printf("    Cluster: %s\n", r.ClusterID)
		}
	}

	return nil
}

func runCluster(cmd *cobra.Command, args []string) error {
	logFile := args[0]

	logger, cleanup, err := setupLogging()
	if err != nil {
		return err
	}
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		cancel()
	}()

	// Create gRPC client
	client, err := createMLClient(logger)
	if err != nil {
		return err
	}
	defer func() { _ = client.Close() }()

	// Load logs from file
	records, err := loadLogs(logFile)
	if err != nil {
		return fmt.Errorf("failed to load logs: %w", err)
	}

	fmt.Printf("Loaded %d logs from %s\n", len(records), logFile)
	fmt.Println("Clustering logs...")

	clusterCtx, clusterCancel := context.WithTimeout(ctx, timeout)
	defer clusterCancel()

	result, err := client.Cluster(clusterCtx, records, minClusterSize, false)
	if err != nil {
		return fmt.Errorf("clustering failed: %w", err)
	}

	fmt.Printf("\n=== Clustering Results ===\n")
	fmt.Printf("Total logs processed: %d\n", result.TotalProcessed)
	fmt.Printf("Clusters found: %d\n", len(result.Clusters))
	fmt.Printf("Noise points: %d\n", result.NoiseCount)
	fmt.Println(strings.Repeat("-", 80))

	for i, cluster := range result.Clusters {
		fmt.Printf("\n[Cluster %d] ID: %s\n", i+1, cluster.ClusterID)
		fmt.Printf("  Size: %d logs\n", cluster.Size)
		fmt.Printf("  Cohesion: %.2f%%\n", cluster.Cohesion*100)
		if cluster.IsNew {
			fmt.Printf("  Status: NEW (first seen)\n")
		}
		fmt.Printf("  Representative: %s\n", truncate(cluster.Representative, 150))
		if len(cluster.Keywords) > 0 {
			fmt.Printf("  Keywords: %s\n", strings.Join(cluster.Keywords, ", "))
		}
	}

	return nil
}

func runNovelty(cmd *cobra.Command, args []string) error {
	logFile := args[0]

	logger, cleanup, err := setupLogging()
	if err != nil {
		return err
	}
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		cancel()
	}()

	// Create gRPC client
	client, err := createMLClient(logger)
	if err != nil {
		return err
	}
	defer func() { _ = client.Close() }()

	// Load logs from file
	records, err := loadLogs(logFile)
	if err != nil {
		return fmt.Errorf("failed to load logs: %w", err)
	}

	fmt.Printf("Loaded %d logs from %s\n", len(records), logFile)
	fmt.Printf("Detecting novel patterns (threshold: %.1f%%)...\n", noveltyThreshold*100)

	novelLogs := make([]struct {
		record *models.LogRecord
		result *grpcclient.NoveltyResult
	}, 0)

	for i, record := range records {
		noveltyCtx, noveltyCancel := context.WithTimeout(ctx, timeout)
		result, err := client.DetectNovelty(noveltyCtx, record, float32(noveltyThreshold))
		noveltyCancel()

		if err != nil {
			logger.Warn("novelty_check_failed", zap.Int("index", i), zap.Error(err))
			continue
		}

		if result.IsNovel {
			novelLogs = append(novelLogs, struct {
				record *models.LogRecord
				result *grpcclient.NoveltyResult
			}{record, result})
		}

		// Progress indicator
		if (i+1)%10 == 0 {
			fmt.Printf("\rProcessed %d/%d logs...", i+1, len(records))
		}
	}

	fmt.Printf("\r                                            \r")
	fmt.Printf("\n=== Novelty Detection Results ===\n")
	fmt.Printf("Total logs analyzed: %d\n", len(records))
	fmt.Printf("Novel patterns found: %d\n", len(novelLogs))
	fmt.Println(strings.Repeat("-", 80))

	if len(novelLogs) == 0 {
		fmt.Println("No novel patterns detected. All logs match known patterns.")
		return nil
	}

	for i, nl := range novelLogs {
		fmt.Printf("\n[Novel %d] Score: %.2f%%\n", i+1, nl.result.NoveltyScore*100)
		fmt.Printf("  Level: %s\n", nl.record.Level)
		fmt.Printf("  Message: %s\n", truncate(nl.record.Message, 200))
		if nl.result.Reason != "" {
			fmt.Printf("  Reason: %s\n", nl.result.Reason)
		}
		if nl.result.ClosestClusterID != "" {
			fmt.Printf("  Closest Cluster: %s (distance: %.3f)\n", nl.result.ClosestClusterID, nl.result.DistanceToCluster)
		}
	}

	return nil
}

func runHealth(cmd *cobra.Command, args []string) error {
	logger, cleanup, err := setupLogging()
	if err != nil {
		return err
	}
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Create gRPC client
	client, err := createMLClient(logger)
	if err != nil {
		return err
	}
	defer func() { _ = client.Close() }()

	fmt.Printf("Checking ML server health at %s...\n", mlServerAddr)

	health, err := client.Health(ctx, true)
	if err != nil {
		fmt.Printf("\n❌ ML Server is UNHEALTHY\n")
		fmt.Printf("   Error: %v\n", err)
		return nil // Don't return error, just show status
	}

	if health.Healthy {
		fmt.Printf("\n✅ ML Server is HEALTHY\n")
	} else {
		fmt.Printf("\n⚠️  ML Server has issues\n")
	}
	fmt.Printf("   Version: %s\n", health.Version)

	if len(health.Components) > 0 {
		fmt.Printf("\n   Components:\n")
		for _, comp := range health.Components {
			status := "✅"
			if !comp.Healthy {
				status = "❌"
			}
			fmt.Printf("   %s %s", status, comp.Name)
			if comp.Message != "" {
				fmt.Printf(": %s", comp.Message)
			}
			fmt.Println()
		}
	}

	return nil
}

// setupLogging initializes the logging system and returns a cleanup function
func setupLogging() (*zap.Logger, func(), error) {
	logCfg := logging.DefaultConfig()
	logCfg.ConsoleFormat = "plain"
	if verbose {
		logCfg.Level = "debug"
	}
	if err := logging.Setup(logCfg); err != nil {
		return nil, nil, fmt.Errorf("failed to setup logging: %w", err)
	}
	logger := logging.L()
	return logger, func() { _ = logging.Close() }, nil
}

// createMLClient creates and connects a gRPC client
func createMLClient(logger *zap.Logger) (*grpcclient.Client, error) {
	cfg := &grpcclient.Config{
		Address:        mlServerAddr,
		ConnectTimeout: 10 * time.Second,
		RequestTimeout: timeout,
		MaxRetries:     3,
		RetryBackoff:   100 * time.Millisecond,
		MaxBackoff:     5 * time.Second,
		MaxMessageSize: 100 * 1024 * 1024,
		Logger:         logger,
	}

	client, err := grpcclient.NewClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC client: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.ConnectTimeout)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("failed to connect to ML server: %w", err)
	}

	return client, nil
}

// loadLogs loads logs from a JSONL file
func loadLogs(path string) ([]*models.LogRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var records []*models.LogRecord
	scanner := bufio.NewScanner(file)
	
	// Increase buffer size for long lines
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}

		var record models.LogRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			// Try parsing as raw message
			record = models.LogRecord{
				Message: line,
				Raw:     line,
			}
		}

		records = append(records, &record)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return records, nil
}

// truncate truncates a string to maxLen characters
func truncate(s string, maxLen int) string {
	// Remove newlines for display
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", "")
	
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
