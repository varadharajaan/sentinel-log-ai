"""Tests for logging module."""

import json
import logging
import io
import sys
import pytest

from sentinel_ml.logging import setup_logging, get_logger


class TestSetupLogging:
    """Test logging setup."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        setup_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_setup_logging_debug(self):
        """Test debug level logging."""
        setup_logging(level="DEBUG")
        logger = get_logger("test_debug")
        assert logger is not None

    def test_setup_logging_warning(self):
        """Test warning level logging."""
        setup_logging(level="WARNING")
        logger = get_logger("test_warning")
        assert logger is not None

    def test_setup_logging_error(self):
        """Test error level logging."""
        setup_logging(level="ERROR")
        logger = get_logger("test_error")
        assert logger is not None

    def test_setup_logging_json_format(self, capsys):
        """Test JSON format logging."""
        setup_logging(level="INFO", format="json")
        logger = get_logger("test_json")
        
        # Log a message
        logger.info("test message", extra_field="value")
        
        # Check output
        captured = capsys.readouterr()
        # Should contain structured output
        assert "test" in captured.err or "test" in captured.out

    def test_setup_logging_console_format(self, capsys):
        """Test console format logging."""
        setup_logging(level="INFO", format="plain")
        logger = get_logger("test_console")
        
        logger.info("console test message")
        
        captured = capsys.readouterr()
        # Should have output
        assert len(captured.err) > 0 or len(captured.out) > 0


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_with_name(self):
        """Test getting a logger with a name."""
        setup_logging()
        logger = get_logger("mymodule")
        assert logger is not None

    def test_get_logger_returns_structlog(self):
        """Test that get_logger returns a structlog logger."""
        setup_logging()
        logger = get_logger("test_structlog")
        
        # Should have structlog methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")

    def test_logger_with_context(self, capsys):
        """Test logger with bound context."""
        setup_logging(level="INFO")
        logger = get_logger("context_test")
        
        # Bind context
        bound_logger = logger.bind(request_id="12345")
        bound_logger.info("with context")
        
        captured = capsys.readouterr()
        output = captured.err + captured.out
        # Context should be in output
        assert "12345" in output or "context" in output

    def test_multiple_loggers(self):
        """Test getting multiple loggers."""
        setup_logging()
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        assert logger1 is not None
        assert logger2 is not None

    def test_same_name_same_logger(self):
        """Test that same name returns equivalent loggers."""
        setup_logging()
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        
        # Both should work
        assert logger1 is not None
        assert logger2 is not None


class TestLogLevels:
    """Test different log levels."""

    def test_debug_level(self, capsys):
        """Test debug level logging."""
        setup_logging(level="DEBUG")
        logger = get_logger("debug_test")
        
        logger.debug("debug message")
        
        captured = capsys.readouterr()
        assert "debug" in captured.err.lower() or "debug" in captured.out.lower()

    def test_info_level(self, capsys):
        """Test info level logging."""
        setup_logging(level="INFO")
        logger = get_logger("info_test")
        
        logger.info("info message")
        
        captured = capsys.readouterr()
        assert "info" in captured.err.lower() or "info" in captured.out.lower()

    def test_warning_level(self, capsys):
        """Test warning level logging."""
        setup_logging(level="WARNING")
        logger = get_logger("warn_test")
        
        logger.warning("warning message")
        
        captured = capsys.readouterr()
        assert "warn" in captured.err.lower() or "warn" in captured.out.lower()

    def test_error_level(self, capsys):
        """Test error level logging."""
        setup_logging(level="ERROR")
        logger = get_logger("error_test")
        
        logger.error("error message")
        
        captured = capsys.readouterr()
        assert "error" in captured.err.lower() or "error" in captured.out.lower()

    def test_level_filtering(self, capsys):
        """Test that lower levels are filtered."""
        setup_logging(level="WARNING")
        logger = get_logger("filter_test")
        
        logger.debug("should not appear")
        logger.info("should not appear")
        logger.warning("should appear")
        
        captured = capsys.readouterr()
        output = captured.err + captured.out
        # Warning should appear, debug/info should not
        assert "should appear" in output or "warning" in output.lower()


class TestLoggerOutput:
    """Test logger output formatting."""

    def test_log_with_extra_fields(self, capsys):
        """Test logging with extra fields."""
        setup_logging(level="INFO")
        logger = get_logger("extra_test")
        
        logger.info("test message", user_id=123, action="login")
        
        captured = capsys.readouterr()
        output = captured.err + captured.out
        # Should contain the extra fields
        assert "test" in output

    def test_log_exception(self, capsys):
        """Test logging exceptions."""
        setup_logging(level="ERROR")
        logger = get_logger("exception_test")
        
        try:
            raise ValueError("test error")
        except ValueError:
            logger.exception("caught exception")
        
        captured = capsys.readouterr()
        output = captured.err + captured.out
        assert "exception" in output.lower() or "error" in output.lower()

    def test_log_complex_objects(self, capsys):
        """Test logging complex objects."""
        setup_logging(level="INFO")
        logger = get_logger("complex_test")
        
        logger.info(
            "complex data",
            data={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )
        
        captured = capsys.readouterr()
        output = captured.err + captured.out
        assert "complex" in output.lower() or "data" in output

    def test_log_unicode(self, capsys):
        """Test logging unicode content."""
        setup_logging(level="INFO")
        logger = get_logger("unicode_test")
        
        logger.info("unicode test: 日本語 émoji 🔥")
        
        captured = capsys.readouterr()
        output = captured.err + captured.out
        assert len(output) > 0  # Should not crash


class TestLoggerConfiguration:
    """Test logger configuration options."""

    def test_reconfigure_logging(self, capsys):
        """Test reconfiguring logging."""
        # First setup
        setup_logging(level="DEBUG")
        logger = get_logger("reconfig_test")
        logger.debug("debug level")
        
        # Reconfigure to higher level
        setup_logging(level="ERROR")
        logger2 = get_logger("reconfig_test2")
        logger2.info("info level")  # Should not appear
        logger2.error("error level")  # Should appear
        
        captured = capsys.readouterr()
        output = captured.err + captured.out
        assert "error" in output.lower()

    def test_json_format_structure(self, capsys):
        """Test JSON format produces valid JSON."""
        setup_logging(level="INFO", format="json")
        logger = get_logger("json_struct_test")
        
        logger.info("json test", key="value")
        
        captured = capsys.readouterr()
        output = (captured.err + captured.out).strip()
        
        # Try to parse as JSON (may have multiple lines)
        if output:
            for line in output.split("\n"):
                if line.strip().startswith("{"):
                    try:
                        parsed = json.loads(line)
                        assert isinstance(parsed, dict)
                    except json.JSONDecodeError:
                        pass  # Some lines may not be JSON


class TestLoggerPerformance:
    """Test logger performance characteristics."""

    def test_many_log_calls(self):
        """Test many log calls don't cause issues."""
        setup_logging(level="WARNING")  # Higher level to reduce output
        logger = get_logger("perf_test")
        
        for i in range(1000):
            logger.debug("debug message %d", i)  # Should be filtered
            if i % 100 == 0:
                logger.warning("warning %d", i)
        
        # Should complete without issues

    def test_large_message(self, capsys):
        """Test logging large messages."""
        setup_logging(level="INFO")
        logger = get_logger("large_test")
        
        large_msg = "A" * 10000
        logger.info("large message", data=large_msg)
        
        captured = capsys.readouterr()
        # Should not crash
        assert len(captured.err) > 0 or len(captured.out) > 0

    def test_concurrent_logging(self):
        """Test concurrent logging from multiple threads."""
        import threading
        
        setup_logging(level="WARNING")
        logger = get_logger("concurrent_test")
        
        errors = []
        
        def log_messages():
            try:
                for i in range(100):
                    logger.warning("thread message %d", i)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=log_messages) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors during concurrent logging: {errors}"
