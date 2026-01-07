"""
CLI module for Sentinel Log AI.

This module provides rich terminal output capabilities including:
- Formatted tables with colors
- Progress bars and spinners
- Interactive menus
- Report generation (Markdown, HTML)
- Profiling and timing breakdown
- Configuration management

Design Patterns:
- Strategy Pattern: Pluggable output formatters
- Template Method: Report generation templates
- Facade Pattern: Simple CLI interface to complex formatting
- Observer Pattern: Progress tracking callbacks
- Decorator Pattern: Transparent timing via profiler

SOLID Principles:
- Single Responsibility: Each component has one job
- Open/Closed: Easy to add new formatters/themes
- Liskov Substitution: All formatters are interchangeable
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions

Usage:
    from sentinel_ml.cli import Console, Theme
    console = Console(theme=Theme.DARK)
    console.print_table(data, title="Results")

    # With profiling
    from sentinel_ml.cli import Profiler
    profiler = Profiler()
    with profiler.measure("operation"):
        do_work()
    profiler.print_report()
"""

from sentinel_ml.cli.config_cmd import (
    generate_config,
    load_config,
    show_config,
    validate_config,
)
from sentinel_ml.cli.console import CapturedOutput, Console, ConsoleConfig, OutputFormat
from sentinel_ml.cli.formatters import (
    ClusterFormatter,
    ExplanationFormatter,
    FormatOptions,
    Formatter,
    JSONFormatter,
    LogRecordFormatter,
    NoveltyFormatter,
    ProfileFormatter,
    ProfileTiming,
    TableColumn,
    TableData,
    TableFormatter,
)
from sentinel_ml.cli.profiler import (
    Profiler,
    TimingEntry,
    disable_profiling,
    enable_profiling,
    get_profiler,
    measure,
    profile,
)
from sentinel_ml.cli.progress import (
    ProgressTracker,
    SpinnerContext,
    SpinnerType,
    TaskProgress,
    spinner,
    timed_operation,
)
from sentinel_ml.cli.report import (
    HTMLReporter,
    MarkdownReporter,
    ReportConfig,
    ReportData,
    Reporter,
)
from sentinel_ml.cli.themes import (
    Theme,
    ThemeColors,
    get_confidence_color,
    get_log_level_color,
    get_novelty_color,
    get_severity_color,
    get_theme,
)

__all__ = [
    "CapturedOutput",
    "ClusterFormatter",
    # Console
    "Console",
    "ConsoleConfig",
    "ExplanationFormatter",
    "FormatOptions",
    # Formatters
    "Formatter",
    "HTMLReporter",
    "JSONFormatter",
    "LogRecordFormatter",
    "MarkdownReporter",
    "NoveltyFormatter",
    "OutputFormat",
    "ProfileFormatter",
    "ProfileTiming",
    # Profiler
    "Profiler",
    # Progress
    "ProgressTracker",
    "ReportConfig",
    "ReportData",
    # Reports
    "Reporter",
    "SpinnerContext",
    "SpinnerType",
    "TableColumn",
    "TableData",
    "TableFormatter",
    "TaskProgress",
    # Themes
    "Theme",
    "ThemeColors",
    "TimingEntry",
    "disable_profiling",
    "enable_profiling",
    # Config
    "generate_config",
    "get_confidence_color",
    "get_log_level_color",
    "get_novelty_color",
    "get_profiler",
    "get_severity_color",
    "get_theme",
    "load_config",
    "measure",
    "profile",
    "show_config",
    "spinner",
    "timed_operation",
    "validate_config",
]
