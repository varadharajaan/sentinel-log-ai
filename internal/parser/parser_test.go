package parser

import (
	"testing"
	"time"
)

func TestJSONParser_CanParse(t *testing.T) {
	p := NewJSONParser()

	tests := []struct {
		name string
		line string
		want bool
	}{
		{"valid JSON object", `{"message": "test"}`, true},
		{"valid JSON with whitespace", `  {"message": "test"}  `, true},
		{"plain text", "just a plain log line", false},
		{"incomplete JSON", `{"message": "test"`, false},
		{"JSON array", `["item1", "item2"]`, false},
		{"empty line", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := p.CanParse(tt.line); got != tt.want {
				t.Errorf("JSONParser.CanParse() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestJSONParser_Parse(t *testing.T) {
	p := NewJSONParser()
	source := "/var/log/app.log"

	tests := []struct {
		name        string
		line        string
		wantOK      bool
		wantMessage string
		wantLevel   string
	}{
		{
			name:        "standard message field",
			line:        `{"message": "connection failed", "level": "ERROR"}`,
			wantOK:      true,
			wantMessage: "connection failed",
			wantLevel:   "ERROR",
		},
		{
			name:        "msg field instead of message",
			line:        `{"msg": "request processed", "level": "INFO"}`,
			wantOK:      true,
			wantMessage: "request processed",
			wantLevel:   "INFO",
		},
		{
			name:   "invalid JSON",
			line:   `{"message": broken}`,
			wantOK: false,
		},
		{
			name:        "nested attrs",
			line:        `{"message": "test", "attrs": {"host": "server-01"}}`,
			wantOK:      true,
			wantMessage: "test",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			record, ok := p.Parse(tt.line, source)
			if ok != tt.wantOK {
				t.Errorf("JSONParser.Parse() ok = %v, want %v", ok, tt.wantOK)
				return
			}
			if !tt.wantOK {
				return
			}
			if record.Message != tt.wantMessage {
				t.Errorf("JSONParser.Parse() message = %v, want %v", record.Message, tt.wantMessage)
			}
			if record.Level != tt.wantLevel {
				t.Errorf("JSONParser.Parse() level = %v, want %v", record.Level, tt.wantLevel)
			}
			if record.Source != source {
				t.Errorf("JSONParser.Parse() source = %v, want %v", record.Source, source)
			}
			if record.Raw != tt.line {
				t.Errorf("JSONParser.Parse() raw = %v, want %v", record.Raw, tt.line)
			}
		})
	}
}

func TestSyslogParser_CanParse(t *testing.T) {
	p := NewSyslogParser()

	tests := []struct {
		name string
		line string
		want bool
	}{
		{
			name: "standard syslog with PID",
			line: "Jan 15 10:30:00 myhost sshd[1234]: Accepted password for user",
			want: true,
		},
		{
			name: "syslog without PID",
			line: "Jan 15 10:30:00 myhost kernel: CPU0: Core temperature above threshold",
			want: true,
		},
		{
			name: "single digit day",
			line: "Jan  5 10:30:00 myhost app: message",
			want: true,
		},
		{
			name: "not syslog",
			line: "2024-01-15 10:30:00 ERROR something failed",
			want: false,
		},
		{
			name: "JSON",
			line: `{"message": "test"}`,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := p.CanParse(tt.line); got != tt.want {
				t.Errorf("SyslogParser.CanParse() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSyslogParser_Parse(t *testing.T) {
	p := NewSyslogParser()
	source := "/var/log/syslog"

	tests := []struct {
		name         string
		line         string
		wantOK       bool
		wantMessage  string
		wantHostname string
		wantProgram  string
		wantPID      string
	}{
		{
			name:         "full syslog line",
			line:         "Jan 15 10:30:00 myhost sshd[1234]: Accepted password for user from 192.168.1.1",
			wantOK:       true,
			wantMessage:  "Accepted password for user from 192.168.1.1",
			wantHostname: "myhost",
			wantProgram:  "sshd",
			wantPID:      "1234",
		},
		{
			name:         "syslog without PID",
			line:         "Jan 15 10:30:00 server kernel: Out of memory",
			wantOK:       true,
			wantMessage:  "Out of memory",
			wantHostname: "server",
			wantProgram:  "kernel",
			wantPID:      "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			record, ok := p.Parse(tt.line, source)
			if ok != tt.wantOK {
				t.Errorf("SyslogParser.Parse() ok = %v, want %v", ok, tt.wantOK)
				return
			}
			if !tt.wantOK {
				return
			}
			if record.Message != tt.wantMessage {
				t.Errorf("Message = %v, want %v", record.Message, tt.wantMessage)
			}
			if record.Attrs["hostname"] != tt.wantHostname {
				t.Errorf("hostname = %v, want %v", record.Attrs["hostname"], tt.wantHostname)
			}
			if record.Attrs["program"] != tt.wantProgram {
				t.Errorf("program = %v, want %v", record.Attrs["program"], tt.wantProgram)
			}
			if tt.wantPID != "" && record.Attrs["pid"] != tt.wantPID {
				t.Errorf("pid = %v, want %v", record.Attrs["pid"], tt.wantPID)
			}
			if record.Timestamp == nil {
				t.Error("Timestamp should not be nil")
			}
		})
	}
}

func TestNginxParser_CanParse(t *testing.T) {
	p := NewNginxParser()

	tests := []struct {
		name string
		line string
		want bool
	}{
		{
			name: "nginx access log",
			line: `127.0.0.1 - frank [15/Jan/2024:10:30:00 +0000] "GET /api/users HTTP/1.1" 200 1234 "-" "curl/7.68.0"`,
			want: true,
		},
		{
			name: "nginx error log",
			line: `2024/01/15 10:30:00 [error] 1234#5678: *9999 connect() failed`,
			want: true,
		},
		{
			name: "not nginx",
			line: "just a regular log line",
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := p.CanParse(tt.line); got != tt.want {
				t.Errorf("NginxParser.CanParse() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNginxParser_Parse(t *testing.T) {
	p := NewNginxParser()
	source := "/var/log/nginx/access.log"

	tests := []struct {
		name       string
		line       string
		wantOK     bool
		wantLevel  string
		wantStatus string
	}{
		{
			name:       "200 OK",
			line:       `127.0.0.1 - - [15/Jan/2024:10:30:00 +0000] "GET /api/health HTTP/1.1" 200 15 "-" "curl/7.68.0"`,
			wantOK:     true,
			wantLevel:  "INFO",
			wantStatus: "200",
		},
		{
			name:       "404 Not Found",
			line:       `10.0.0.1 - admin [15/Jan/2024:10:30:00 +0000] "GET /missing HTTP/1.1" 404 162 "-" "Mozilla/5.0"`,
			wantOK:     true,
			wantLevel:  "WARN",
			wantStatus: "404",
		},
		{
			name:       "500 Internal Error",
			line:       `10.0.0.1 - - [15/Jan/2024:10:30:00 +0000] "POST /api/crash HTTP/1.1" 500 0 "-" "curl/7.68.0"`,
			wantOK:     true,
			wantLevel:  "ERROR",
			wantStatus: "500",
		},
		{
			name:      "error log",
			line:      `2024/01/15 10:30:00 [error] 1234#5678: *9999 upstream timed out`,
			wantOK:    true,
			wantLevel: "ERROR",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			record, ok := p.Parse(tt.line, source)
			if ok != tt.wantOK {
				t.Errorf("NginxParser.Parse() ok = %v, want %v", ok, tt.wantOK)
				return
			}
			if !tt.wantOK {
				return
			}
			if record.Level != tt.wantLevel {
				t.Errorf("Level = %v, want %v", record.Level, tt.wantLevel)
			}
			if tt.wantStatus != "" {
				if record.Attrs["status_code"] != tt.wantStatus {
					t.Errorf("status_code = %v, want %v", record.Attrs["status_code"], tt.wantStatus)
				}
			}
		})
	}
}

func TestPythonTracebackParser_CanParse(t *testing.T) {
	p := NewPythonTracebackParser()

	tests := []struct {
		name string
		line string
		want bool
	}{
		{
			name: "exception line",
			line: "ValueError: invalid literal for int()",
			want: true,
		},
		{
			name: "traceback header",
			line: "Traceback (most recent call last):",
			want: true,
		},
		{
			name: "file reference",
			line: `  File "/app/main.py", line 42, in process`,
			want: true,
		},
		{
			name: "code context",
			line: "    result = int(value)",
			want: true,
		},
		{
			name: "not python",
			line: "2024-01-15 ERROR database connection failed",
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := p.CanParse(tt.line); got != tt.want {
				t.Errorf("PythonTracebackParser.CanParse() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPythonTracebackParser_Parse(t *testing.T) {
	p := NewPythonTracebackParser()
	source := "stderr"

	tests := []struct {
		name              string
		line              string
		wantOK            bool
		wantExceptionType string
	}{
		{
			name:              "ValueError",
			line:              "ValueError: invalid literal for int() with base 10: 'abc'",
			wantOK:            true,
			wantExceptionType: "ValueError",
		},
		{
			name:              "KeyError",
			line:              "KeyError: 'missing_key'",
			wantOK:            true,
			wantExceptionType: "KeyError",
		},
		{
			name:              "custom exception",
			line:              "CustomException: something went wrong",
			wantOK:            true,
			wantExceptionType: "CustomException",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			record, ok := p.Parse(tt.line, source)
			if ok != tt.wantOK {
				t.Errorf("Parse() ok = %v, want %v", ok, tt.wantOK)
				return
			}
			if !tt.wantOK {
				return
			}
			if record.Level != "ERROR" {
				t.Errorf("Level = %v, want ERROR", record.Level)
			}
			if tt.wantExceptionType != "" {
				if record.Attrs["exception_type"] != tt.wantExceptionType {
					t.Errorf("exception_type = %v, want %v", record.Attrs["exception_type"], tt.wantExceptionType)
				}
			}
		})
	}
}

func TestCommonLogParser_Parse(t *testing.T) {
	p := NewCommonLogParser()
	source := "/var/log/app.log"

	tests := []struct {
		name        string
		line        string
		wantOK      bool
		wantLevel   string
		wantMessage string
		wantHasTime bool
	}{
		{
			name:        "with timestamp and level",
			line:        "2024-01-15T10:30:00Z ERROR connection refused",
			wantOK:      true,
			wantLevel:   "ERROR",
			wantMessage: "connection refused",
			wantHasTime: true,
		},
		{
			name:        "level only",
			line:        "INFO Starting application",
			wantOK:      true,
			wantLevel:   "INFO",
			wantMessage: "Starting application",
			wantHasTime: false,
		},
		{
			name:        "bracketed level",
			line:        "[WARN] Memory usage high",
			wantOK:      true,
			wantLevel:   "WARN",
			wantMessage: "Memory usage high",
			wantHasTime: false,
		},
		{
			name:        "WARNING normalized to WARN",
			line:        "WARNING deprecated function used",
			wantOK:      true,
			wantLevel:   "WARN",
			wantMessage: "deprecated function used",
			wantHasTime: false,
		},
		{
			name:        "plain message - infer level from content",
			line:        "Database error occurred",
			wantOK:      true,
			wantLevel:   "ERROR",
			wantMessage: "Database error occurred",
			wantHasTime: false,
		},
		{
			name:        "plain message no level",
			line:        "Application started successfully",
			wantOK:      true,
			wantLevel:   "",
			wantMessage: "Application started successfully",
			wantHasTime: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			record, ok := p.Parse(tt.line, source)
			if ok != tt.wantOK {
				t.Errorf("Parse() ok = %v, want %v", ok, tt.wantOK)
				return
			}
			if !tt.wantOK {
				return
			}
			if record.Level != tt.wantLevel {
				t.Errorf("Level = %v, want %v", record.Level, tt.wantLevel)
			}
			if record.Message != tt.wantMessage {
				t.Errorf("Message = %v, want %v", record.Message, tt.wantMessage)
			}
			if tt.wantHasTime && record.Timestamp == nil {
				t.Error("Expected timestamp but got nil")
			}
			if !tt.wantHasTime && record.Timestamp != nil {
				t.Error("Expected no timestamp but got one")
			}
		})
	}
}

func TestRegistry_Parse(t *testing.T) {
	r := NewRegistry()
	source := "/var/log/test.log"

	tests := []struct {
		name       string
		line       string
		wantParser string // Expected parser based on attrs or behavior
	}{
		{
			name:       "JSON",
			line:       `{"message": "test", "level": "INFO"}`,
			wantParser: "json",
		},
		{
			name:       "syslog",
			line:       "Jan 15 10:30:00 myhost sshd[1234]: login attempt",
			wantParser: "syslog",
		},
		{
			name:       "nginx access",
			line:       `127.0.0.1 - - [15/Jan/2024:10:30:00 +0000] "GET / HTTP/1.1" 200 1234 "-" "curl"`,
			wantParser: "nginx",
		},
		{
			name:       "python error",
			line:       "ValueError: bad value",
			wantParser: "python_traceback",
		},
		{
			name:       "common log",
			line:       "2024-01-15 ERROR something failed",
			wantParser: "common",
		},
		{
			name:       "plain fallback",
			line:       "just some text",
			wantParser: "fallback",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			record := r.Parse(tt.line, source)
			if record == nil {
				t.Fatal("Parse() returned nil")
			}
			if record.Raw != tt.line {
				t.Errorf("Raw = %v, want %v", record.Raw, tt.line)
			}
			if record.Source != source {
				t.Errorf("Source = %v, want %v", record.Source, source)
			}
			// Verify message is populated
			if record.Message == "" && tt.line != "" {
				t.Error("Message should not be empty")
			}
		})
	}
}

func TestExtractLevel(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"Error: something failed", "ERROR"},
		{"WARNING: disk full", "WARN"},
		{"Debug: entering function", "DEBUG"},
		{"Info: started", "INFO"},
		{"FATAL: crash", "FATAL"},
		{"failure occurred", "ERROR"},
		{"plain message", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			if got := extractLevel(tt.input); got != tt.want {
				t.Errorf("extractLevel(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestHttpStatusToLevel(t *testing.T) {
	tests := []struct {
		status string
		want   string
	}{
		{"200", "INFO"},
		{"201", "INFO"},
		{"301", "INFO"},
		{"400", "WARN"},
		{"404", "WARN"},
		{"500", "ERROR"},
		{"503", "ERROR"},
		{"", ""},
	}

	for _, tt := range tests {
		t.Run(tt.status, func(t *testing.T) {
			if got := httpStatusToLevel(tt.status); got != tt.want {
				t.Errorf("httpStatusToLevel(%q) = %v, want %v", tt.status, got, tt.want)
			}
		})
	}
}

func TestParseFlexibleTimestamp(t *testing.T) {
	tests := []struct {
		input   string
		wantErr bool
	}{
		{"2024-01-15T10:30:00Z", false},
		{"2024-01-15T10:30:00+00:00", false},
		{"2024-01-15 10:30:00", false},
		{"2024/01/15 10:30:00", false},
		{"not a timestamp", true},
		{"", true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			_, err := parseFlexibleTimestamp(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseFlexibleTimestamp(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			}
		})
	}
}

// Benchmarks

func BenchmarkJSONParser_Parse(b *testing.B) {
	p := NewJSONParser()
	line := `{"timestamp":"2024-01-15T10:30:00Z","level":"ERROR","message":"connection failed","host":"server-01"}`
	source := "/var/log/app.log"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Parse(line, source)
	}
}

func BenchmarkSyslogParser_Parse(b *testing.B) {
	p := NewSyslogParser()
	line := "Jan 15 10:30:00 myhost sshd[1234]: Accepted password for user from 192.168.1.1"
	source := "/var/log/syslog"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Parse(line, source)
	}
}

func BenchmarkRegistry_Parse(b *testing.B) {
	r := NewRegistry()
	lines := []string{
		`{"message": "test"}`,
		"Jan 15 10:30:00 host app: message",
		`127.0.0.1 - - [15/Jan/2024:10:30:00 +0000] "GET / HTTP/1.1" 200 1234 "-" "curl"`,
		"2024-01-15 ERROR something failed",
	}
	source := "/var/log/test.log"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Parse(lines[i%len(lines)], source)
	}
}

// Test time parsing edge cases
func TestSyslogParser_TimestampYear(t *testing.T) {
	p := NewSyslogParser()
	source := "/var/log/syslog"
	line := "Jan 15 10:30:00 myhost app: test"

	record, ok := p.Parse(line, source)
	if !ok {
		t.Fatal("Parse failed")
	}

	if record.Timestamp == nil {
		t.Fatal("Timestamp is nil")
	}

	// Should have current year
	currentYear := time.Now().Year()
	if record.Timestamp.Year() != currentYear {
		t.Errorf("Year = %v, want %v", record.Timestamp.Year(), currentYear)
	}
}
