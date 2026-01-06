"""Tests for Python log parser module."""


import pytest

from sentinel_ml.parser import (
    CommonLogParser,
    JSONParser,
    NginxParser,
    Parser,
    ParserRegistry,
    PythonTracebackParser,
    SyslogParser,
    get_parser_registry,
    parse_log_line,
)


class TestJSONParser:
    """Test JSON parser."""

    @pytest.fixture
    def parser(self):
        return JSONParser()

    def test_name(self, parser):
        assert parser.name == "json"

    def test_can_parse_valid_json(self, parser):
        assert parser.can_parse('{"message": "test"}')
        assert parser.can_parse('  {"message": "test"}  ')

    def test_can_parse_invalid(self, parser):
        assert not parser.can_parse("plain text")
        assert not parser.can_parse('{"incomplete":')
        assert not parser.can_parse("")

    def test_parse_message_field(self, parser):
        line = '{"message": "connection failed", "level": "ERROR"}'
        record = parser.parse(line, "/var/log/app.log")

        assert record is not None
        assert record.message == "connection failed"
        assert record.level == "ERROR"

    def test_parse_msg_field(self, parser):
        line = '{"msg": "request processed", "level": "info"}'
        record = parser.parse(line, "/var/log/app.log")

        assert record is not None
        assert record.message == "request processed"
        assert record.level == "INFO"

    def test_parse_with_timestamp(self, parser):
        line = '{"message": "test", "timestamp": "2024-01-15T10:30:00Z"}'
        record = parser.parse(line, "test.log")

        assert record is not None
        assert record.timestamp is not None
        assert record.timestamp.year == 2024

    def test_parse_invalid_json(self, parser):
        result = parser.parse('{"broken: json}', "test.log")
        assert result is None


class TestSyslogParser:
    """Test syslog parser."""

    @pytest.fixture
    def parser(self):
        return SyslogParser()

    def test_name(self, parser):
        assert parser.name == "syslog"

    def test_can_parse_syslog(self, parser):
        assert parser.can_parse(
            "Jan 15 10:30:00 myhost sshd[1234]: login attempt"
        )
        assert parser.can_parse("Jan 15 10:30:00 host kernel: message")
        assert parser.can_parse("Jan  5 10:30:00 host app: test")

    def test_can_parse_not_syslog(self, parser):
        assert not parser.can_parse("2024-01-15 ERROR something")
        assert not parser.can_parse('{"message": "test"}')

    def test_parse_with_pid(self, parser):
        line = "Jan 15 10:30:00 myhost sshd[1234]: Accepted password"
        record = parser.parse(line, "/var/log/syslog")

        assert record is not None
        assert record.message == "Accepted password"
        assert record.attrs["hostname"] == "myhost"
        assert record.attrs["program"] == "sshd"
        assert record.attrs["pid"] == 1234

    def test_parse_without_pid(self, parser):
        line = "Jan 15 10:30:00 server kernel: Out of memory"
        record = parser.parse(line, "/var/log/syslog")

        assert record is not None
        assert record.message == "Out of memory"
        assert record.attrs["program"] == "kernel"
        assert "pid" not in record.attrs


class TestNginxParser:
    """Test nginx parser."""

    @pytest.fixture
    def parser(self):
        return NginxParser()

    def test_name(self, parser):
        assert parser.name == "nginx"

    def test_can_parse_access_log(self, parser):
        line = '127.0.0.1 - - [15/Jan/2024:10:30:00 +0000] "GET / HTTP/1.1" 200 1234 "-" "curl"'
        assert parser.can_parse(line)

    def test_can_parse_error_log(self, parser):
        line = "2024/01/15 10:30:00 [error] 1234#5678: connect() failed"
        assert parser.can_parse(line)

    def test_can_parse_not_nginx(self, parser):
        assert not parser.can_parse("regular log line")

    def test_parse_access_log(self, parser):
        line = '127.0.0.1 - admin [15/Jan/2024:10:30:00 +0000] "GET /api HTTP/1.1" 200 512 "-" "curl"'
        record = parser.parse(line, "/var/log/nginx/access.log")

        assert record is not None
        assert record.message == "GET /api HTTP/1.1"
        assert record.level == "INFO"
        assert record.attrs["status_code"] == 200
        assert record.attrs["client_ip"] == "127.0.0.1"

    def test_parse_error_log(self, parser):
        line = "2024/01/15 10:30:00 [error] 1234#5678: upstream timed out"
        record = parser.parse(line, "/var/log/nginx/error.log")

        assert record is not None
        assert record.level == "ERROR"
        assert "upstream timed out" in record.message

    def test_status_code_levels(self, parser):
        # 200 = INFO
        line = '127.0.0.1 - - [15/Jan/2024:10:30:00 +0000] "GET / HTTP/1.1" 200 0 "-" "test"'
        assert parser.parse(line, "test").level == "INFO"

        # 404 = WARN
        line = '127.0.0.1 - - [15/Jan/2024:10:30:00 +0000] "GET / HTTP/1.1" 404 0 "-" "test"'
        assert parser.parse(line, "test").level == "WARN"

        # 500 = ERROR
        line = '127.0.0.1 - - [15/Jan/2024:10:30:00 +0000] "GET / HTTP/1.1" 500 0 "-" "test"'
        assert parser.parse(line, "test").level == "ERROR"


class TestPythonTracebackParser:
    """Test Python traceback parser."""

    @pytest.fixture
    def parser(self):
        return PythonTracebackParser()

    def test_name(self, parser):
        assert parser.name == "python_traceback"

    def test_can_parse_exception(self, parser):
        assert parser.can_parse("ValueError: invalid literal")
        assert parser.can_parse("KeyError: 'missing'")
        assert parser.can_parse("Traceback (most recent call last):")

    def test_can_parse_file_line(self, parser):
        assert parser.can_parse('  File "/app/main.py", line 42')

    def test_can_parse_code_context(self, parser):
        assert parser.can_parse("    result = int(value)")

    def test_can_parse_not_traceback(self, parser):
        assert not parser.can_parse("2024-01-15 ERROR database error")

    def test_parse_exception(self, parser):
        line = "ValueError: invalid literal for int()"
        record = parser.parse(line, "stderr")

        assert record is not None
        assert record.level == "ERROR"
        assert record.attrs["exception_type"] == "ValueError"


class TestCommonLogParser:
    """Test common log parser."""

    @pytest.fixture
    def parser(self):
        return CommonLogParser()

    def test_name(self, parser):
        assert parser.name == "common"

    def test_can_parse_always_true(self, parser):
        assert parser.can_parse("anything")
        assert parser.can_parse("")

    def test_parse_with_timestamp_and_level(self, parser):
        line = "2024-01-15T10:30:00Z ERROR connection refused"
        record = parser.parse(line, "test.log")

        assert record is not None
        assert record.level == "ERROR"
        assert "connection refused" in record.message
        assert record.timestamp is not None

    def test_parse_bracketed_level(self, parser):
        line = "[WARN] Memory usage high"
        record = parser.parse(line, "test.log")

        assert record is not None
        assert record.level == "WARN"

    def test_parse_infer_level(self, parser):
        line = "Database error occurred"
        record = parser.parse(line, "test.log")

        assert record is not None
        assert record.level == "ERROR"


class TestParserRegistry:
    """Test parser registry."""

    def test_default_parsers(self):
        registry = ParserRegistry()
        assert len(registry.parsers) == 5

    def test_parse_json(self):
        registry = ParserRegistry()
        record = registry.parse('{"message": "test"}', "test.log")

        assert record.message == "test"

    def test_parse_syslog(self):
        registry = ParserRegistry()
        record = registry.parse(
            "Jan 15 10:30:00 host app: message", "/var/log/syslog"
        )

        assert record.message == "message"
        assert record.attrs["hostname"] == "host"

    def test_parse_fallback(self):
        registry = ParserRegistry()
        record = registry.parse("plain text line", "test.log")

        assert record is not None
        assert record.message == "plain text line"
        assert record.raw == "plain text line"

    def test_register_custom_parser(self):
        registry = ParserRegistry()

        class CustomParser(Parser):
            @property
            def name(self):
                return "custom"

            def can_parse(self, line):
                return line.startswith("CUSTOM:")

            def parse(self, line, source):
                from sentinel_ml.models import LogRecord

                return LogRecord(
                    message=line[7:], source=source, raw=line, level="CUSTOM"
                )

        registry.register(CustomParser(), priority=0)

        record = registry.parse("CUSTOM: test message", "test.log")
        assert record.level == "CUSTOM"


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_parser_registry(self):
        registry = get_parser_registry()
        assert isinstance(registry, ParserRegistry)

        # Should return same instance
        registry2 = get_parser_registry()
        assert registry is registry2

    def test_parse_log_line(self):
        record = parse_log_line('{"message": "test", "level": "INFO"}')
        assert record.message == "test"
        assert record.level == "INFO"

    def test_parse_log_line_with_source(self):
        record = parse_log_line("plain line", source="/var/log/app.log")
        assert record.source == "/var/log/app.log"


class TestParserEdgeCases:
    """Test edge cases in parsing."""

    def test_empty_line(self):
        record = parse_log_line("")
        assert record.message == ""

    def test_unicode_content(self):
        record = parse_log_line("ERROR: 日本語 émoji 🔥")
        assert "日本語" in record.message

    def test_very_long_line(self):
        long_line = "A" * 10000
        record = parse_log_line(long_line)
        assert len(record.raw) == 10000

    def test_multiline_content(self):
        # Single line input
        line = "Traceback (most recent call last):"
        record = parse_log_line(line)
        assert record.level == "ERROR"
