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
"""

from sentinel_ml.cli.console import Console, ConsoleConfig
from sentinel_ml.cli.formatters import (
    ClusterFormatter,
    ExplanationFormatter,
    Formatter,
    JSONFormatter,
    LogRecordFormatter,
    NoveltyFormatter,
    TableFormatter,
)
from sentinel_ml.cli.progress import ProgressTracker, SpinnerContext, TaskProgress
from sentinel_ml.cli.report import HTMLReporter, MarkdownReporter, ReportConfig, Reporter
from sentinel_ml.cli.themes import Theme, ThemeColors, get_theme

__all__ = [
    # Console
    "Console",
    "ConsoleConfig",
    # Formatters
    "Formatter",
    "TableFormatter",
    "JSONFormatter",
    "ClusterFormatter",
    "NoveltyFormatter",
    "ExplanationFormatter",
    "LogRecordFormatter",
    # Progress
    "ProgressTracker",
    "TaskProgress",
    "SpinnerContext",
    # Themes
    "Theme",
    "ThemeColors",
    "get_theme",
    # Reports
    "Reporter",
    "MarkdownReporter",
    "HTMLReporter",
    "ReportConfig",
]
