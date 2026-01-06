// Package parser provides log line parsing for common log formats.
package parser

import (
	"encoding/json"
	"regexp"
	"strings"
	"time"

	"github.com/sentinel-log-ai/sentinel-log-ai/internal/models"
)

// Parser defines the interface for log format parsers.
type Parser interface {
	// Name returns the parser name.
	Name() string

	// CanParse returns true if this parser can handle the log line.
	CanParse(line string) bool

	// Parse parses a log line into a LogRecord.
	// Returns the record and true if parsing succeeded, or a raw record and false if not.
	Parse(line string, source string) (*models.LogRecord, bool)
}

// Registry holds registered parsers and routes log lines to appropriate parsers.
type Registry struct {
	parsers []Parser
}

// NewRegistry creates a new parser registry with default parsers.
func NewRegistry() *Registry {
	return &Registry{
		parsers: []Parser{
			NewJSONParser(),
			NewSyslogParser(),
			NewNginxParser(),
			NewPythonTracebackParser(),
			NewCommonLogParser(),
		},
	}
}

// Register adds a parser to the registry.
func (r *Registry) Register(p Parser) {
	r.parsers = append([]Parser{p}, r.parsers...) // prepend for priority
}

// Parse tries each parser until one succeeds.
func (r *Registry) Parse(line string, source string) *models.LogRecord {
	for _, p := range r.parsers {
		if p.CanParse(line) {
			if record, ok := p.Parse(line, source); ok {
				return record
			}
		}
	}
	// Fallback to raw record
	return &models.LogRecord{
		Message: line,
		Source:  source,
		Raw:     line,
	}
}

// JSONParser parses JSON-formatted log lines.
type JSONParser struct{}

// NewJSONParser creates a new JSON parser.
func NewJSONParser() *JSONParser {
	return &JSONParser{}
}

// Name returns the parser name.
func (p *JSONParser) Name() string {
	return "json"
}

// CanParse checks if the line looks like JSON.
func (p *JSONParser) CanParse(line string) bool {
	trimmed := strings.TrimSpace(line)
	return strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}")
}

// Parse parses a JSON log line.
func (p *JSONParser) Parse(line string, source string) (*models.LogRecord, bool) {
	// First parse the raw JSON to get all fields
	var jsonData map[string]any
	if err := json.Unmarshal([]byte(line), &jsonData); err != nil {
		return nil, false
	}

	record := &models.LogRecord{
		Source: source,
		Raw:    line,
		Attrs:  jsonData,
	}

	// Extract message from common fields
	if msg, ok := jsonData["message"].(string); ok {
		record.Message = msg
	} else if msg, ok := jsonData["msg"].(string); ok {
		record.Message = msg
	}

	// Extract level
	if level, ok := jsonData["level"].(string); ok {
		record.Level = level
	}

	// Extract timestamp
	if ts, ok := jsonData["timestamp"].(string); ok {
		if t, err := parseFlexibleTimestamp(ts); err == nil {
			record.Timestamp = &t
		}
	}

	return record, true
}

// SyslogParser parses syslog-formatted log lines.
type SyslogParser struct {
	// Jan 15 10:30:00 hostname program[pid]: message
	pattern *regexp.Regexp
}

// NewSyslogParser creates a new syslog parser.
func NewSyslogParser() *SyslogParser {
	return &SyslogParser{
		pattern: regexp.MustCompile(
			`^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+?)(?:\[(\d+)\])?:\s*(.*)$`,
		),
	}
}

// Name returns the parser name.
func (p *SyslogParser) Name() string {
	return "syslog"
}

// CanParse checks if the line looks like syslog.
func (p *SyslogParser) CanParse(line string) bool {
	return p.pattern.MatchString(line)
}

// Parse parses a syslog line.
func (p *SyslogParser) Parse(line string, source string) (*models.LogRecord, bool) {
	matches := p.pattern.FindStringSubmatch(line)
	if matches == nil {
		return nil, false
	}

	// Parse timestamp (add current year since syslog doesn't include it)
	ts, err := time.Parse("Jan 2 15:04:05", matches[1])
	if err == nil {
		// Set to current year
		now := time.Now()
		ts = ts.AddDate(now.Year(), 0, 0)
	}

	attrs := map[string]any{
		"hostname": matches[2],
		"program":  matches[3],
	}
	if matches[4] != "" {
		attrs["pid"] = matches[4]
	}

	record := &models.LogRecord{
		Timestamp: &ts,
		Message:   matches[5],
		Source:    source,
		Raw:       line,
		Attrs:     attrs,
	}

	// Try to extract level from message
	record.Level = extractLevel(matches[5])

	return record, true
}

// NginxParser parses nginx access/error log lines.
type NginxParser struct {
	accessPattern *regexp.Regexp
	errorPattern  *regexp.Regexp
}

// NewNginxParser creates a new nginx parser.
func NewNginxParser() *NginxParser {
	return &NginxParser{
		// Combined log format: 127.0.0.1 - - [15/Jan/2024:10:30:00 +0000] "GET /path HTTP/1.1" 200 1234 "-" "Mozilla/5.0"
		accessPattern: regexp.MustCompile(
			`^(\S+)\s+\S+\s+(\S+)\s+\[([^\]]+)\]\s+"([^"]+)"\s+(\d+)\s+(\d+)\s+"([^"]*)"\s+"([^"]*)"`,
		),
		// Error log: 2024/01/15 10:30:00 [error] 1234#1234: *5678 message
		errorPattern: regexp.MustCompile(
			`^(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[(\w+)\]\s+(\d+)#(\d+):\s*(?:\*\d+\s+)?(.*)$`,
		),
	}
}

// Name returns the parser name.
func (p *NginxParser) Name() string {
	return "nginx"
}

// CanParse checks if the line looks like nginx log.
func (p *NginxParser) CanParse(line string) bool {
	return p.accessPattern.MatchString(line) || p.errorPattern.MatchString(line)
}

// Parse parses an nginx log line.
func (p *NginxParser) Parse(line string, source string) (*models.LogRecord, bool) {
	// Try access log first
	if matches := p.accessPattern.FindStringSubmatch(line); matches != nil {
		ts, _ := time.Parse("02/Jan/2006:15:04:05 -0700", matches[3])
		return &models.LogRecord{
			Timestamp: &ts,
			Level:     httpStatusToLevel(matches[5]),
			Message:   matches[4],
			Source:    source,
			Raw:       line,
			Attrs: map[string]any{
				"client_ip":   matches[1],
				"user":        matches[2],
				"status_code": matches[5],
				"bytes":       matches[6],
				"referer":     matches[7],
				"user_agent":  matches[8],
			},
		}, true
	}

	// Try error log
	if matches := p.errorPattern.FindStringSubmatch(line); matches != nil {
		ts, _ := time.Parse("2006/01/02 15:04:05", matches[1])
		return &models.LogRecord{
			Timestamp: &ts,
			Level:     strings.ToUpper(matches[2]),
			Message:   matches[5],
			Source:    source,
			Raw:       line,
			Attrs: map[string]any{
				"pid": matches[3],
				"tid": matches[4],
			},
		}, true
	}

	return nil, false
}

// PythonTracebackParser parses Python traceback lines.
type PythonTracebackParser struct {
	errorPattern *regexp.Regexp
}

// NewPythonTracebackParser creates a new Python traceback parser.
func NewPythonTracebackParser() *PythonTracebackParser {
	return &PythonTracebackParser{
		// Match: ExceptionType: message or Traceback (most recent call last):
		errorPattern: regexp.MustCompile(
			`^(\w+Error|\w+Exception|Traceback \(most recent call last\)):?\s*(.*)$`,
		),
	}
}

// Name returns the parser name.
func (p *PythonTracebackParser) Name() string {
	return "python_traceback"
}

// CanParse checks if the line looks like a Python error.
func (p *PythonTracebackParser) CanParse(line string) bool {
	trimmed := strings.TrimSpace(line)
	return p.errorPattern.MatchString(line) ||
		strings.HasPrefix(trimmed, "File \"") ||
		(strings.HasPrefix(line, "    ") && len(trimmed) > 0)
}

// Parse parses a Python traceback line.
func (p *PythonTracebackParser) Parse(line string, source string) (*models.LogRecord, bool) {
	matches := p.errorPattern.FindStringSubmatch(line)
	if matches == nil {
		// It's a traceback context line, still valid
		return &models.LogRecord{
			Level:   "ERROR",
			Message: line,
			Source:  source,
			Raw:     line,
			Attrs: map[string]any{
				"parser": "python_traceback",
			},
		}, true
	}

	return &models.LogRecord{
		Level:   "ERROR",
		Message: line,
		Source:  source,
		Raw:     line,
		Attrs: map[string]any{
			"exception_type": matches[1],
			"exception_msg":  matches[2],
			"parser":         "python_traceback",
		},
	}, true
}

// CommonLogParser handles generic log patterns.
type CommonLogParser struct {
	// Generic: LEVEL message or [LEVEL] message or timestamp LEVEL message
	pattern *regexp.Regexp
}

// NewCommonLogParser creates a new common log parser.
func NewCommonLogParser() *CommonLogParser {
	return &CommonLogParser{
		pattern: regexp.MustCompile(
			`(?i)^(?:(\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+)?` +
				`(?:\[?(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\]?[:\s]+)?` +
				`(.+)$`,
		),
	}
}

// Name returns the parser name.
func (p *CommonLogParser) Name() string {
	return "common"
}

// CanParse always returns true as fallback.
func (p *CommonLogParser) CanParse(line string) bool {
	return true
}

// Parse parses a generic log line.
func (p *CommonLogParser) Parse(line string, source string) (*models.LogRecord, bool) {
	matches := p.pattern.FindStringSubmatch(line)
	if matches == nil {
		return nil, false
	}

	record := &models.LogRecord{
		Message: matches[3],
		Source:  source,
		Raw:     line,
	}

	// Parse timestamp if present
	if matches[1] != "" {
		if ts, err := parseFlexibleTimestamp(matches[1]); err == nil {
			record.Timestamp = &ts
		}
	}

	// Set level if present
	if matches[2] != "" {
		record.Level = normalizeLevel(matches[2])
	} else {
		record.Level = extractLevel(line)
	}

	return record, true
}

// Helper functions

func extractLevel(message string) string {
	upper := strings.ToUpper(message)
	switch {
	case strings.Contains(upper, "ERROR") || strings.Contains(upper, "FAIL"):
		return "ERROR"
	case strings.Contains(upper, "WARN"):
		return "WARN"
	case strings.Contains(upper, "DEBUG"):
		return "DEBUG"
	case strings.Contains(upper, "INFO"):
		return "INFO"
	case strings.Contains(upper, "FATAL") || strings.Contains(upper, "CRITICAL"):
		return "FATAL"
	default:
		return ""
	}
}

func normalizeLevel(level string) string {
	upper := strings.ToUpper(level)
	switch upper {
	case "WARNING":
		return "WARN"
	case "CRITICAL":
		return "FATAL"
	default:
		return upper
	}
}

func httpStatusToLevel(status string) string {
	if len(status) == 0 {
		return ""
	}
	switch status[0] {
	case '2':
		return "INFO"
	case '3':
		return "INFO"
	case '4':
		return "WARN"
	case '5':
		return "ERROR"
	default:
		return ""
	}
}

func parseFlexibleTimestamp(s string) (time.Time, error) {
	formats := []string{
		time.RFC3339,
		time.RFC3339Nano,
		"2006-01-02T15:04:05",
		"2006-01-02 15:04:05",
		"2006/01/02 15:04:05",
		"2006-01-02T15:04:05.000",
		"2006-01-02 15:04:05.000",
	}
	for _, format := range formats {
		if t, err := time.Parse(format, s); err == nil {
			return t, nil
		}
	}
	return time.Time{}, &time.ParseError{}
}
