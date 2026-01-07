"""
Unit tests for CLI module.

Tests cover:
- Themes and color utilities
- Formatters (JSON, Table, Cluster, Novelty, Explanation)
- Console output and capture
- Progress tracking and spinners
- Report generation (Markdown, HTML)
- Profiling and timing
- Configuration management
"""

from __future__ import annotations

import io
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

from sentinel_ml.cli import (
    ClusterFormatter,
    Console,
    ConsoleConfig,
    FormatOptions,
    HTMLReporter,
    JSONFormatter,
    MarkdownReporter,
    OutputFormat,
    ProfileFormatter,
    Profiler,
    ProfileTiming,
    ProgressTracker,
    ReportConfig,
    ReportData,
    SpinnerContext,
    SpinnerType,
    TableColumn,
    TableData,
    TableFormatter,
    TaskProgress,
    Theme,
    ThemeColors,
    TimingEntry,
    disable_profiling,
    enable_profiling,
    generate_config,
    get_confidence_color,
    get_log_level_color,
    get_novelty_color,
    get_profiler,
    get_severity_color,
    get_theme,
    load_config,
    measure,
    profile,
    show_config,
    spinner,
    timed_operation,
    validate_config,
)
from sentinel_ml.models import ClusterSummary, LogRecord

# =============================================================================
# Theme Tests
# =============================================================================


class TestThemes:
    """Tests for theme system."""

    def test_get_theme_dark(self) -> None:
        """Test getting dark theme."""
        colors = get_theme(Theme.DARK)
        assert isinstance(colors, ThemeColors)
        assert colors.primary == "cyan"
        assert colors.error == "red"

    def test_get_theme_light(self) -> None:
        """Test getting light theme."""
        colors = get_theme(Theme.LIGHT)
        assert isinstance(colors, ThemeColors)
        assert colors.primary == "blue"

    def test_get_theme_minimal(self) -> None:
        """Test getting minimal theme."""
        colors = get_theme(Theme.MINIMAL)
        assert isinstance(colors, ThemeColors)
        assert colors.primary == "bold"

    def test_get_theme_colorblind(self) -> None:
        """Test getting colorblind-friendly theme."""
        colors = get_theme(Theme.COLORBLIND)
        assert isinstance(colors, ThemeColors)
        assert "blue" in colors.primary.lower() or "dodger" in colors.primary.lower()

    def test_get_theme_none(self) -> None:
        """Test getting no-color theme."""
        colors = get_theme(Theme.NONE)
        assert isinstance(colors, ThemeColors)
        assert colors.primary == ""
        assert colors.error == ""

    def test_get_theme_by_string(self) -> None:
        """Test getting theme by string name."""
        colors = get_theme("dark")
        assert colors.primary == "cyan"

    def test_get_theme_invalid(self) -> None:
        """Test getting invalid theme raises error."""
        with pytest.raises(ValueError, match="Unknown theme"):
            get_theme("invalid_theme")

    def test_get_severity_color_high(self) -> None:
        """Test severity color for HIGH."""
        colors = get_theme(Theme.DARK)
        color = get_severity_color("HIGH", colors)
        assert color == colors.cluster_high

    def test_get_severity_color_critical(self) -> None:
        """Test severity color for CRITICAL."""
        colors = get_theme(Theme.DARK)
        color = get_severity_color("CRITICAL", colors)
        assert color == colors.cluster_high

    def test_get_severity_color_medium(self) -> None:
        """Test severity color for MEDIUM."""
        colors = get_theme(Theme.DARK)
        color = get_severity_color("MEDIUM", colors)
        assert color == colors.cluster_medium

    def test_get_severity_color_low(self) -> None:
        """Test severity color for LOW."""
        colors = get_theme(Theme.DARK)
        color = get_severity_color("LOW", colors)
        assert color == colors.cluster_low

    def test_get_novelty_color_high(self) -> None:
        """Test novelty color for high score."""
        colors = get_theme(Theme.DARK)
        color = get_novelty_color(0.8, colors)
        assert color == colors.novelty_high

    def test_get_novelty_color_medium(self) -> None:
        """Test novelty color for medium score."""
        colors = get_theme(Theme.DARK)
        color = get_novelty_color(0.5, colors)
        assert color == colors.novelty_medium

    def test_get_novelty_color_low(self) -> None:
        """Test novelty color for low score."""
        colors = get_theme(Theme.DARK)
        color = get_novelty_color(0.2, colors)
        assert color == colors.novelty_low

    def test_get_confidence_color_string_high(self) -> None:
        """Test confidence color for HIGH string."""
        colors = get_theme(Theme.DARK)
        color = get_confidence_color("HIGH", colors)
        assert color == colors.confidence_high

    def test_get_confidence_color_numeric(self) -> None:
        """Test confidence color for numeric value."""
        colors = get_theme(Theme.DARK)
        assert get_confidence_color(0.8, colors) == colors.confidence_high
        assert get_confidence_color(0.5, colors) == colors.confidence_medium
        assert get_confidence_color(0.2, colors) == colors.confidence_low

    def test_get_log_level_color_error(self) -> None:
        """Test log level color for ERROR."""
        colors = get_theme(Theme.DARK)
        color = get_log_level_color("ERROR", colors)
        assert color == colors.level_error

    def test_get_log_level_color_warn(self) -> None:
        """Test log level color for WARN."""
        colors = get_theme(Theme.DARK)
        color = get_log_level_color("WARN", colors)
        assert color == colors.level_warn

    def test_get_log_level_color_info(self) -> None:
        """Test log level color for INFO."""
        colors = get_theme(Theme.DARK)
        color = get_log_level_color("INFO", colors)
        assert color == colors.level_info

    def test_get_log_level_color_debug(self) -> None:
        """Test log level color for DEBUG."""
        colors = get_theme(Theme.DARK)
        color = get_log_level_color("DEBUG", colors)
        assert color == colors.level_debug


# =============================================================================
# Formatter Tests
# =============================================================================


class TestFormatOptions:
    """Tests for FormatOptions."""

    def test_default_options(self) -> None:
        """Test default format options."""
        opts = FormatOptions()
        assert opts.colors is True
        assert opts.compact is False
        assert opts.max_width == 120
        assert opts.truncate_messages == 80

    def test_custom_options(self) -> None:
        """Test custom format options."""
        opts = FormatOptions(
            colors=False,
            compact=True,
            max_width=80,
            truncate_messages=50,
        )
        assert opts.colors is False
        assert opts.compact is True
        assert opts.max_width == 80
        assert opts.truncate_messages == 50


class TestJSONFormatter:
    """Tests for JSON formatter."""

    def test_format_dict(self) -> None:
        """Test formatting a dictionary."""
        formatter = JSONFormatter()
        data = {"key": "value", "number": 42}
        result = formatter.format(data)
        parsed = json.loads(result)
        assert parsed == data

    def test_format_list(self) -> None:
        """Test formatting a list."""
        formatter = JSONFormatter()
        data = [{"a": 1}, {"b": 2}]
        result = formatter.format(data)
        parsed = json.loads(result)
        assert parsed == data

    def test_format_pydantic_model(self) -> None:
        """Test formatting a Pydantic model."""
        formatter = JSONFormatter()
        record = LogRecord(
            message="Test message",
            raw="Test raw",
            source="test",
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed["message"] == "Test message"
        assert parsed["source"] == "test"

    def test_format_compact(self) -> None:
        """Test compact JSON output."""
        formatter = JSONFormatter(pretty=False)
        data = {"key": "value"}
        result = formatter.format(data)
        assert "\n" not in result

    def test_format_datetime(self) -> None:
        """Test formatting datetime objects."""
        formatter = JSONFormatter()
        data = {"timestamp": datetime(2024, 1, 15, 10, 30, 0)}
        result = formatter.format(data)
        assert "2024" in result


class TestTableFormatter:
    """Tests for table formatter."""

    def test_format_simple_table(self) -> None:
        """Test formatting a simple table."""
        columns = [
            TableColumn(key="name", header="Name"),
            TableColumn(key="value", header="Value"),
        ]
        formatter = TableFormatter(columns=columns, title="Test Table")
        rows = [
            {"name": "foo", "value": "bar"},
            {"name": "baz", "value": "qux"},
        ]
        result = formatter.format(rows)
        assert "Test Table" in result
        assert "Name" in result
        assert "foo" in result
        assert "bar" in result

    def test_format_table_data(self) -> None:
        """Test formatting TableData object."""
        columns = [TableColumn(key="id", header="ID")]
        data = TableData(
            columns=columns,
            rows=[{"id": "1"}, {"id": "2"}],
            title="IDs",
            footer="2 items",
        )
        formatter = TableFormatter()
        result = formatter.format(data)
        assert "IDs" in result
        assert "1" in result
        assert "2" in result
        assert "2 items" in result

    def test_format_with_alignment(self) -> None:
        """Test column alignment."""
        columns = [
            TableColumn(key="num", header="Number", align="right"),
            TableColumn(key="txt", header="Text", align="left"),
        ]
        formatter = TableFormatter(columns=columns)
        rows = [{"num": "42", "txt": "hello"}]
        result = formatter.format(rows)
        assert "Number" in result
        assert "42" in result

    def test_truncate_long_values(self) -> None:
        """Test truncation of long values."""
        columns = [TableColumn(key="text", header="Text")]
        opts = FormatOptions(truncate_messages=10)
        formatter = TableFormatter(columns=columns, options=opts)
        rows = [{"text": "This is a very long text that should be truncated"}]
        result = formatter.format(rows)
        assert "..." in result


class TestClusterFormatter:
    """Tests for cluster formatter."""

    def test_format_single_cluster(self) -> None:
        """Test formatting a single cluster."""
        formatter = ClusterFormatter()
        cluster = ClusterSummary(
            cluster_id="test-cluster-123",
            size=50,
            representative="Error: Connection timeout",
            keywords=["error", "connection", "timeout"],
            cohesion=0.85,
            novelty_score=0.3,
            is_new=False,
        )
        result = formatter.format(cluster)
        assert "test-cluster" in result
        assert "50" in result
        assert "Error: Connection" in result

    def test_format_cluster_list(self) -> None:
        """Test formatting multiple clusters."""
        formatter = ClusterFormatter()
        clusters = [
            ClusterSummary(
                cluster_id="c1",
                size=10,
                representative="Log 1",
                cohesion=0.8,
            ),
            ClusterSummary(
                cluster_id="c2",
                size=20,
                representative="Log 2",
                cohesion=0.9,
            ),
        ]
        result = formatter.format(clusters)
        assert "c1" in result
        assert "c2" in result
        assert "2 found" in result

    def test_format_new_cluster(self) -> None:
        """Test formatting a new cluster."""
        formatter = ClusterFormatter()
        cluster = ClusterSummary(
            cluster_id="new-cluster",
            size=5,
            representative="New pattern",
            cohesion=0.7,
            is_new=True,
        )
        result = formatter.format(cluster)
        assert "NEW" in result

    def test_format_empty_list(self) -> None:
        """Test formatting empty cluster list."""
        formatter = ClusterFormatter()
        result = formatter.format([])
        assert "No clusters found" in result


class TestProfileFormatter:
    """Tests for profile formatter."""

    def test_format_timings(self) -> None:
        """Test formatting timing data."""
        formatter = ProfileFormatter()
        timings = [
            ProfileTiming(name="loading", duration_ms=100.0, percentage=40.0),
            ProfileTiming(name="processing", duration_ms=150.0, percentage=60.0),
        ]
        result = formatter.format(timings)
        assert "Timing Breakdown" in result
        assert "loading" in result
        assert "100" in result
        assert "processing" in result

    def test_format_nested_timings(self) -> None:
        """Test formatting nested timings."""
        formatter = ProfileFormatter()
        child = ProfileTiming(name="parsing", duration_ms=50.0, percentage=20.0)
        parent = ProfileTiming(
            name="loading",
            duration_ms=100.0,
            percentage=40.0,
            children=[child],
        )
        result = formatter.format([parent])
        assert "loading" in result
        assert "parsing" in result

    def test_format_empty(self) -> None:
        """Test formatting empty timings."""
        formatter = ProfileFormatter()
        result = formatter.format([])
        assert "No profiling data" in result


# =============================================================================
# Console Tests
# =============================================================================


class TestConsoleConfig:
    """Tests for console configuration."""

    def test_default_config(self) -> None:
        """Test default console config."""
        config = ConsoleConfig()
        assert config.theme == Theme.DARK
        assert config.format == OutputFormat.TEXT
        # Colors are disabled in non-TTY environments (like pytest)
        # The default is True but __post_init__ sets to False if not TTY
        assert config.colors is False  # Non-TTY during test

    def test_custom_config(self) -> None:
        """Test custom console config."""
        config = ConsoleConfig(
            theme=Theme.LIGHT,
            format=OutputFormat.JSON,
            colors=False,
            verbose=True,
        )
        assert config.theme == Theme.LIGHT
        assert config.format == OutputFormat.JSON
        assert config.colors is False
        assert config.verbose is True


class TestConsole:
    """Tests for Console class."""

    def test_create_console(self) -> None:
        """Test creating a console."""
        console = Console()
        assert console.config.theme == Theme.DARK

    def test_create_console_with_theme(self) -> None:
        """Test creating console with specific theme."""
        console = Console(theme=Theme.LIGHT)
        assert console.config.theme == Theme.LIGHT

    def test_print_output(self) -> None:
        """Test printing to output."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False)
        console = Console(config=config)
        console.print("Hello, World!")
        assert "Hello, World!" in output.getvalue()

    def test_info_message(self) -> None:
        """Test info message."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False)
        console = Console(config=config)
        console.info("Info message")
        assert "Info message" in output.getvalue()

    def test_success_message(self) -> None:
        """Test success message."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False)
        console = Console(config=config)
        console.success("Success!")
        assert "Success!" in output.getvalue()

    def test_warning_message(self) -> None:
        """Test warning message."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False)
        console = Console(config=config)
        console.warning("Warning!")
        assert "Warning!" in output.getvalue()

    def test_error_message(self) -> None:
        """Test error message."""
        error_output = io.StringIO()
        config = ConsoleConfig(error_output=error_output, colors=False)
        console = Console(config=config)
        console.error("Error!")
        assert "Error!" in error_output.getvalue()

    def test_debug_message_verbose(self) -> None:
        """Test debug message in verbose mode."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False, verbose=True)
        console = Console(config=config)
        console.debug("Debug info")
        assert "Debug info" in output.getvalue()

    def test_debug_message_not_verbose(self) -> None:
        """Test debug message not shown without verbose."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False, verbose=False)
        console = Console(config=config)
        console.debug("Debug info")
        assert "Debug info" not in output.getvalue()

    def test_quiet_mode(self) -> None:
        """Test quiet mode suppresses info."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False, quiet=True)
        console = Console(config=config)
        console.info("Should be suppressed")
        assert output.getvalue() == ""

    def test_header(self) -> None:
        """Test header output."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False)
        console = Console(config=config)
        console.header("Test Header")
        result = output.getvalue()
        assert "Test Header" in result
        assert "=" in result

    def test_divider(self) -> None:
        """Test divider output."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False)
        console = Console(config=config)
        console.divider()
        assert "─" in output.getvalue()

    def test_print_json(self) -> None:
        """Test JSON output."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False)
        console = Console(config=config)
        console.print_json({"key": "value"})
        parsed = json.loads(output.getvalue())
        assert parsed["key"] == "value"

    def test_print_table(self) -> None:
        """Test table output."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False)
        console = Console(config=config)
        console.print_table(
            rows=[{"name": "test", "value": 42}],
            title="Test Table",
        )
        result = output.getvalue()
        assert "test" in result
        assert "42" in result

    def test_print_table_empty(self) -> None:
        """Test empty table."""
        output = io.StringIO()
        config = ConsoleConfig(output=output, colors=False)
        console = Console(config=config)
        console.print_table([])
        assert "No data" in output.getvalue()

    def test_capture_output(self) -> None:
        """Test output capture."""
        console = Console()
        with console.capture() as captured:
            console.print("Captured text")
        assert "Captured text" in captured.output


class TestCapturedOutput:
    """Tests for CapturedOutput."""

    def test_capture_stdout(self) -> None:
        """Test capturing stdout."""
        console = Console()
        with console.capture() as captured:
            console.print("stdout text")
        assert "stdout text" in captured.output

    def test_capture_stderr(self) -> None:
        """Test capturing stderr."""
        console = Console()
        with console.capture() as captured:
            console.error("stderr text")
        assert "stderr text" in captured.error


# =============================================================================
# Progress Tests
# =============================================================================


class TestTaskProgress:
    """Tests for TaskProgress."""

    def test_create_task(self) -> None:
        """Test creating a task."""
        task = TaskProgress(name="test", total=100)
        assert task.name == "test"
        assert task.total == 100
        assert task.completed == 0

    def test_percentage(self) -> None:
        """Test percentage calculation."""
        task = TaskProgress(name="test", total=100, completed=50)
        assert task.percentage == 50.0

    def test_percentage_zero_total(self) -> None:
        """Test percentage with zero total."""
        task = TaskProgress(name="test", total=0)
        assert task.percentage == 0.0

    def test_advance(self) -> None:
        """Test advancing progress."""
        task = TaskProgress(name="test", total=100)
        task.advance(10)
        assert task.completed == 10
        task.advance()
        assert task.completed == 11

    def test_update(self) -> None:
        """Test updating progress."""
        task = TaskProgress(name="test", total=100)
        task.update(completed=50, status="halfway")
        assert task.completed == 50
        assert task.status == "halfway"

    def test_finish(self) -> None:
        """Test finishing a task."""
        task = TaskProgress(name="test", total=100)
        task.finish(status="Done")
        assert task.is_complete is True
        assert task.status == "Done"

    def test_elapsed_seconds(self) -> None:
        """Test elapsed time calculation."""
        task = TaskProgress(name="test")
        time.sleep(0.01)
        assert task.elapsed_seconds > 0

    def test_items_per_second(self) -> None:
        """Test rate calculation."""
        task = TaskProgress(name="test", total=100, completed=50)
        time.sleep(0.01)
        rate = task.items_per_second
        assert rate > 0


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_create_tracker(self) -> None:
        """Test creating a tracker."""
        tracker = ProgressTracker()
        assert tracker.show_eta is True
        assert tracker.show_rate is True

    def test_task_context(self) -> None:
        """Test task context manager."""
        output = io.StringIO()
        tracker = ProgressTracker(output=output)
        with tracker.task("Test Task", total=10) as task:
            for _ in range(10):
                task.advance()
        # Check output contains task info
        result = output.getvalue()
        assert "Test Task" in result

    def test_create_progress_bar(self) -> None:
        """Test creating progress bar."""
        tracker = ProgressTracker()
        bar = tracker.create_progress_bar(50, 100, width=20)
        assert "[" in bar
        assert "]" in bar
        assert "50.0%" in bar

    def test_format_duration(self) -> None:
        """Test duration formatting."""
        assert ProgressTracker.format_duration(0.5) == "500ms"
        assert ProgressTracker.format_duration(30) == "30.0s"
        assert ProgressTracker.format_duration(120) == "2m 0s"
        assert ProgressTracker.format_duration(3700) == "1h 1m"

    def test_format_rate(self) -> None:
        """Test rate formatting."""
        assert "items/s" in ProgressTracker.format_rate(100)
        assert "K" in ProgressTracker.format_rate(1500)
        assert "M" in ProgressTracker.format_rate(1500000)


class TestSpinnerContext:
    """Tests for SpinnerContext."""

    def test_spinner_context(self) -> None:
        """Test spinner context manager."""
        output = io.StringIO()
        with SpinnerContext("Loading", output=output):
            pass
        result = output.getvalue()
        assert "Loading" in result

    def test_spinner_update(self) -> None:
        """Test updating spinner message."""
        output = io.StringIO()
        with SpinnerContext("Loading", output=output) as s:
            s.update("Processing")
        # Just verify no exception

    def test_spinner_types(self) -> None:
        """Test different spinner types."""
        for spinner_type in SpinnerType:
            output = io.StringIO()
            with SpinnerContext("Test", spinner_type=spinner_type, output=output):
                pass


class TestSpinnerFunction:
    """Tests for spinner convenience function."""

    def test_spinner_function(self) -> None:
        """Test spinner as function."""
        with spinner("Test") as s:
            s.update("Updated")


class TestTimedOperation:
    """Tests for timed_operation context."""

    def test_timed_operation(self) -> None:
        """Test timed operation context."""
        with timed_operation("test_op") as timing:
            time.sleep(0.01)
        assert "duration_ms" in timing
        assert timing["duration_ms"] > 0
        assert timing["name"] == "test_op"


# =============================================================================
# Profiler Tests
# =============================================================================


class TestProfiler:
    """Tests for Profiler class."""

    def test_create_profiler(self) -> None:
        """Test creating a profiler."""
        profiler = Profiler()
        assert profiler.enabled is True

    def test_create_disabled_profiler(self) -> None:
        """Test creating a disabled profiler."""
        profiler = Profiler(enabled=False)
        assert profiler.enabled is False

    def test_measure_operation(self) -> None:
        """Test measuring an operation."""
        profiler = Profiler()
        with profiler.measure("test"):
            time.sleep(0.01)
        timings = profiler.get_timings()
        assert len(timings) == 1
        assert timings[0].name == "test"
        assert timings[0].duration_ms > 0

    def test_measure_nested(self) -> None:
        """Test measuring nested operations."""
        profiler = Profiler()
        with profiler.measure("outer"), profiler.measure("inner"):
            time.sleep(0.01)
        timings = profiler.get_timings()
        # Only root operations in top-level
        assert len(timings) == 1
        assert timings[0].name == "outer"

    def test_profile_decorator(self) -> None:
        """Test profile decorator."""
        profiler = Profiler()

        @profiler.profile()
        def test_func() -> int:
            time.sleep(0.01)
            return 42

        result = test_func()
        assert result == 42
        timings = profiler.get_timings()
        assert len(timings) == 1
        assert timings[0].name == "test_func"

    def test_get_summary(self) -> None:
        """Test getting summary stats."""
        profiler = Profiler()
        with profiler.measure("op1"):
            time.sleep(0.01)
        with profiler.measure("op2"):
            time.sleep(0.01)
        summary = profiler.get_summary()
        assert summary["operations"] == 2
        assert summary["total_ms"] > 0

    def test_format_report(self) -> None:
        """Test formatting report."""
        profiler = Profiler()
        with profiler.measure("test"):
            pass
        report = profiler.format_report()
        assert "test" in report

    def test_clear(self) -> None:
        """Test clearing profiler."""
        profiler = Profiler()
        with profiler.measure("test"):
            pass
        profiler.clear()
        assert len(profiler.get_timings()) == 0

    def test_to_dict(self) -> None:
        """Test exporting to dict."""
        profiler = Profiler()
        with profiler.measure("test"):
            pass
        data = profiler.to_dict()
        assert "entries" in data
        assert "summary" in data

    def test_threshold(self) -> None:
        """Test threshold filtering."""
        profiler = Profiler(threshold_ms=100)  # 100ms threshold
        with profiler.measure("fast"):  # Should be filtered
            pass
        timings = profiler.get_timings()
        # Fast operation should be filtered out
        assert len(timings) == 0


class TestGlobalProfiler:
    """Tests for global profiler functions."""

    def test_get_profiler(self) -> None:
        """Test getting global profiler."""
        profiler = get_profiler()
        assert isinstance(profiler, Profiler)

    def test_enable_profiling(self) -> None:
        """Test enabling profiling."""
        profiler = enable_profiling()
        assert profiler.enabled is True

    def test_disable_profiling(self) -> None:
        """Test disabling profiling."""
        enable_profiling()
        disable_profiling()
        profiler = get_profiler()
        assert profiler.enabled is False

    def test_measure_function(self) -> None:
        """Test measure convenience function."""
        enable_profiling()
        with measure("test"):
            pass
        profiler = get_profiler()
        # Clean up
        profiler.clear()
        disable_profiling()

    def test_profile_decorator_function(self) -> None:
        """Test profile decorator as function."""
        enable_profiling()

        @profile("decorated")
        def test_fn() -> str:
            return "result"

        result = test_fn()
        assert result == "result"
        # Clean up
        get_profiler().clear()
        disable_profiling()


class TestTimingEntry:
    """Tests for TimingEntry."""

    def test_duration_ms(self) -> None:
        """Test duration calculation."""
        entry = TimingEntry(
            name="test",
            start_time=0.0,
            end_time=0.1,
        )
        assert entry.duration_ms == 100.0

    def test_duration_ms_no_end(self) -> None:
        """Test duration with no end time."""
        entry = TimingEntry(name="test", start_time=0.0)
        assert entry.duration_ms == 0.0


# =============================================================================
# Report Tests
# =============================================================================


class TestReportConfig:
    """Tests for ReportConfig."""

    def test_default_config(self) -> None:
        """Test default report config."""
        config = ReportConfig()
        assert config.title == "Log Analysis Report"
        assert config.include_timestamp is True
        assert config.include_summary is True

    def test_custom_config(self) -> None:
        """Test custom report config."""
        config = ReportConfig(
            title="Custom Report",
            include_toc=False,
            max_clusters=10,
        )
        assert config.title == "Custom Report"
        assert config.include_toc is False
        assert config.max_clusters == 10


class TestReportData:
    """Tests for ReportData."""

    def test_empty_report_data(self) -> None:
        """Test empty report data."""
        data = ReportData()
        assert data.clusters == []
        assert data.novelty_scores == []
        assert data.explanations == []


class TestMarkdownReporter:
    """Tests for Markdown reporter."""

    def test_generate_empty_report(self) -> None:
        """Test generating empty report."""
        reporter = MarkdownReporter()
        data = ReportData()
        report = reporter.generate(data)
        assert "# Log Analysis Report" in report
        assert "Generated:" in report

    def test_generate_with_clusters(self) -> None:
        """Test generating report with clusters."""
        reporter = MarkdownReporter()
        clusters = [
            ClusterSummary(
                cluster_id="c1",
                size=10,
                representative="Error log",
                cohesion=0.8,
                keywords=["error"],
            )
        ]
        data = ReportData(clusters=clusters)
        report = reporter.generate(data)
        assert "Cluster Analysis" in report
        assert "c1" in report

    def test_save_report(self) -> None:
        """Test saving report to file."""
        reporter = MarkdownReporter()
        data = ReportData()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            saved_path = reporter.save(data, path)
            assert saved_path.exists()
            content = saved_path.read_text()
            assert "Log Analysis Report" in content


class TestHTMLReporter:
    """Tests for HTML reporter."""

    def test_generate_empty_report(self) -> None:
        """Test generating empty HTML report."""
        reporter = HTMLReporter()
        data = ReportData()
        report = reporter.generate(data)
        assert "<!DOCTYPE html>" in report
        assert "Log Analysis Report" in report
        assert "</html>" in report

    def test_generate_with_clusters(self) -> None:
        """Test generating HTML with clusters."""
        reporter = HTMLReporter()
        clusters = [
            ClusterSummary(
                cluster_id="c1",
                size=10,
                representative="Error log",
                cohesion=0.8,
            )
        ]
        data = ReportData(clusters=clusters)
        report = reporter.generate(data)
        assert "Cluster Analysis" in report
        assert "c1" in report

    def test_html_escaping(self) -> None:
        """Test HTML escaping."""
        reporter = HTMLReporter()
        clusters = [
            ClusterSummary(
                cluster_id="test",
                size=1,
                representative="<script>alert('xss')</script>",
                cohesion=0.8,
            )
        ]
        data = ReportData(clusters=clusters)
        report = reporter.generate(data)
        assert "<script>" not in report
        assert "&lt;script&gt;" in report


# =============================================================================
# Config Tests
# =============================================================================


class TestConfigGeneration:
    """Tests for config generation."""

    def test_generate_config_default(self) -> None:
        """Test generating default config."""
        content = generate_config()
        assert "server:" in content
        assert "embedding:" in content
        assert "clustering:" in content
        assert "llm:" in content

    def test_generate_config_minimal(self) -> None:
        """Test generating minimal config."""
        content = generate_config(minimal=True)
        # Minimal config is shorter
        assert len(content) < 500

    def test_generate_config_to_file(self) -> None:
        """Test generating config to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            generate_config(output_path=path)
            assert path.exists()
            content = path.read_text()
            assert "server:" in content


class TestConfigValidation:
    """Tests for config validation."""

    def test_validate_valid_config(self) -> None:
        """Test validating a valid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text("""
server:
  port: 50051
embedding:
  dimension: 384
            """)
            is_valid, errors = validate_config(path)
            assert is_valid is True
            assert len(errors) == 0

    def test_validate_invalid_port(self) -> None:
        """Test validating invalid port."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text("""
server:
  port: 99999
            """)
            is_valid, errors = validate_config(path)
            assert is_valid is False
            assert any("port" in e for e in errors)

    def test_validate_missing_file(self) -> None:
        """Test validating missing file."""
        is_valid, errors = validate_config("/nonexistent/config.yaml")
        assert is_valid is False
        assert any("not found" in e for e in errors)

    def test_validate_invalid_yaml(self) -> None:
        """Test validating invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text("invalid: yaml: content:")
            is_valid, _errors = validate_config(path)
            assert is_valid is False


class TestConfigLoading:
    """Tests for config loading."""

    def test_load_valid_config(self) -> None:
        """Test loading valid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text("""
server:
  port: 50052
embedding:
  model_name: test-model
  batch_size: 64
            """)
            config = load_config(path)
            assert config.server.port == 50052
            assert config.embedding.model_name == "test-model"
            assert config.embedding.batch_size == 64

    def test_load_invalid_config_raises(self) -> None:
        """Test loading invalid config raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text("""
server:
  port: -1
            """)
            with pytest.raises(ValueError, match="Invalid configuration"):
                load_config(path)


class TestShowConfig:
    """Tests for show_config."""

    def test_show_config(self) -> None:
        """Test showing config."""
        from sentinel_ml.config import Config

        config = Config()
        output = show_config(config)
        assert "Server:" in output
        assert "Embedding:" in output
        assert "Clustering:" in output
