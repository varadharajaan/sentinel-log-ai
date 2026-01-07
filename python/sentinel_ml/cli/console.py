"""
Console output manager for CLI.

Provides a unified interface for all CLI output with theme support,
formatting, and output redirection.

Design Pattern: Facade Pattern - simple interface to complex formatting.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TextIO

from sentinel_ml.cli.formatters import (
    ClusterFormatter,
    ExplanationFormatter,
    FormatOptions,
    JSONFormatter,
    LogRecordFormatter,
    NoveltyFormatter,
    ProfileFormatter,
    ProfileTiming,
    TableColumn,
    TableFormatter,
)
from sentinel_ml.cli.progress import ProgressTracker, SpinnerContext, SpinnerType
from sentinel_ml.cli.themes import Theme, get_theme
from sentinel_ml.logging import get_logger
from sentinel_ml.models import ClusterSummary, Explanation, LogRecord

if TYPE_CHECKING:
    from sentinel_ml.novelty import NoveltyScore

logger = get_logger(__name__)


class OutputFormat(str, Enum):
    """Output format options."""

    TEXT = "text"
    JSON = "json"
    TABLE = "table"
    COMPACT = "compact"


@dataclass
class ConsoleConfig:
    """
    Configuration for console output.

    Attributes:
        theme: Color theme to use.
        format: Default output format.
        max_width: Maximum output width.
        colors: Whether to use colors.
        timestamps: Whether to show timestamps.
        verbose: Enable verbose output.
        quiet: Suppress non-essential output.
        output: Output stream.
        error_output: Error output stream.
    """

    theme: Theme = Theme.DARK
    format: OutputFormat = OutputFormat.TEXT
    max_width: int = 120
    colors: bool = True
    timestamps: bool = True
    verbose: bool = False
    quiet: bool = False
    output: TextIO = field(default_factory=lambda: sys.stdout)
    error_output: TextIO = field(default_factory=lambda: sys.stderr)

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Disable colors if output is not a TTY
        if not hasattr(self.output, "isatty") or not self.output.isatty():
            self.colors = False


class Console:
    """
    Main console output manager.

    Provides a unified interface for all CLI output including:
    - Formatted messages (info, success, warning, error)
    - Tables with automatic column sizing
    - Progress bars and spinners
    - Cluster, novelty, and explanation formatting
    - JSON output for scripting

    Example:
        console = Console()
        console.info("Starting analysis...")
        console.print_clusters(clusters)
        console.success("Done!")
    """

    def __init__(
        self,
        config: ConsoleConfig | None = None,
        theme: Theme | str | None = None,
    ) -> None:
        """
        Initialize console.

        Args:
            config: Console configuration.
            theme: Theme override (overrides config.theme).
        """
        self.config = config or ConsoleConfig()

        # Override theme if specified
        if theme:
            if isinstance(theme, str):
                self.config.theme = Theme(theme)
            else:
                self.config.theme = theme

        # Get theme colors
        self._theme_colors = get_theme(self.config.theme)

        # Create format options
        self._format_options = FormatOptions(
            colors=self.config.colors,
            compact=self.config.format == OutputFormat.COMPACT,
            max_width=self.config.max_width,
            show_timestamps=self.config.timestamps,
        )

        # Initialize formatters
        self._init_formatters()

        # Progress tracker
        self._progress_tracker = ProgressTracker(
            use_colors=self.config.colors,
            output=self.config.error_output,
        )

        logger.debug(
            "console_initialized",
            theme=self.config.theme.value,
            format=self.config.format.value,
            colors=self.config.colors,
        )

    def _init_formatters(self) -> None:
        """Initialize output formatters."""
        self._json_formatter = JSONFormatter(
            theme=self._theme_colors,
            options=self._format_options,
        )
        self._table_formatter = TableFormatter(
            theme=self._theme_colors,
            options=self._format_options,
        )
        self._cluster_formatter = ClusterFormatter(
            theme=self._theme_colors,
            options=self._format_options,
        )
        self._novelty_formatter = NoveltyFormatter(
            theme=self._theme_colors,
            options=self._format_options,
        )
        self._explanation_formatter = ExplanationFormatter(
            theme=self._theme_colors,
            options=self._format_options,
        )
        self._log_formatter = LogRecordFormatter(
            theme=self._theme_colors,
            options=self._format_options,
        )
        self._profile_formatter = ProfileFormatter(
            theme=self._theme_colors,
            options=self._format_options,
        )

    # =========================================================================
    # Basic Output Methods
    # =========================================================================

    def print(self, message: str, *, end: str = "\n") -> None:
        """
        Print a message to output.

        Args:
            message: Message to print.
            end: Line ending.
        """
        self.config.output.write(message + end)
        self.config.output.flush()

    def print_error(self, message: str) -> None:
        """
        Print a message to error output.

        Args:
            message: Error message.
        """
        self.config.error_output.write(message + "\n")
        self.config.error_output.flush()

    def _colorize(self, text: str, color: str) -> str:
        """Wrap text in color markup."""
        if not self.config.colors or not color:
            return text
        return f"[{color}]{text}[/{color}]"

    # =========================================================================
    # Semantic Message Methods
    # =========================================================================

    def info(self, message: str) -> None:
        """Print an info message."""
        if self.config.quiet:
            return
        icon = "i" if self.config.colors else "[INFO]"
        formatted = f"{self._colorize(icon, self._theme_colors.info)} {message}"
        self.print(formatted)

    def success(self, message: str) -> None:
        """Print a success message."""
        icon = "✓" if self.config.colors else "[OK]"
        formatted = f"{self._colorize(icon, self._theme_colors.success)} {message}"
        self.print(formatted)

    def warning(self, message: str) -> None:
        """Print a warning message."""
        icon = "⚠" if self.config.colors else "[WARN]"
        formatted = f"{self._colorize(icon, self._theme_colors.warning)} {message}"
        self.print(formatted)

    def error(self, message: str) -> None:
        """Print an error message."""
        icon = "✗" if self.config.colors else "[ERROR]"
        formatted = f"{self._colorize(icon, self._theme_colors.error)} {message}"
        self.print_error(formatted)

    def debug(self, message: str) -> None:
        """Print a debug message (only in verbose mode)."""
        if not self.config.verbose:
            return
        icon = "🔍" if self.config.colors else "[DEBUG]"
        formatted = f"{self._colorize(icon, self._theme_colors.muted)} {message}"
        self.print(formatted)

    def header(self, message: str, *, char: str = "=") -> None:
        """
        Print a header with decorative line.

        Args:
            message: Header text.
            char: Character for the underline.
        """
        formatted = self._colorize(message, self._theme_colors.header)
        self.print(formatted)
        underline = char * min(len(message), self.config.max_width)
        self.print(self._colorize(underline, self._theme_colors.border))

    def divider(self, char: str = "─") -> None:
        """Print a horizontal divider."""
        line = char * self.config.max_width
        self.print(self._colorize(line, self._theme_colors.border))

    def blank(self) -> None:
        """Print a blank line."""
        self.print("")

    # =========================================================================
    # Structured Output Methods
    # =========================================================================

    def print_json(self, data: Any) -> None:
        """
        Print data as JSON.

        Args:
            data: Data to serialize as JSON.
        """
        formatted = self._json_formatter.format(data)
        self.print(formatted)

    def print_table(
        self,
        rows: list[dict[str, Any]],
        columns: list[TableColumn] | None = None,
        title: str | None = None,
    ) -> None:
        """
        Print data as a table.

        Args:
            rows: List of row dictionaries.
            columns: Column definitions (auto-detected if None).
            title: Table title.
        """
        if not rows:
            self.info("No data to display.")
            return

        # Auto-detect columns if not provided
        if not columns:
            all_keys: set[str] = set()
            for row in rows:
                all_keys.update(row.keys())
            columns = [
                TableColumn(key=k, header=k.replace("_", " ").title()) for k in sorted(all_keys)
            ]

        formatter = TableFormatter(
            theme=self._theme_colors,
            options=self._format_options,
            columns=columns,
            title=title,
        )
        formatted = formatter.format(rows)
        self.print(formatted)

    def print_clusters(
        self,
        clusters: ClusterSummary | Sequence[ClusterSummary],
        *,
        format: OutputFormat | None = None,
    ) -> None:
        """
        Print cluster summaries.

        Args:
            clusters: Cluster(s) to display.
            format: Override output format.
        """
        output_format = format or self.config.format

        if output_format == OutputFormat.JSON:
            if isinstance(clusters, Sequence) and not isinstance(clusters, ClusterSummary):
                data: list[Any] = [c.model_dump() for c in clusters]
            else:
                # Single cluster
                single = clusters  # type: ClusterSummary
                data = [single.model_dump()]
            self.print_json(data)
        else:
            formatted = self._cluster_formatter.format(clusters)  # type: ignore[arg-type]
            self.print(formatted)

    def print_novelty(
        self,
        scores: NoveltyScore | Sequence[NoveltyScore],
        *,
        format: OutputFormat | None = None,
    ) -> None:
        """
        Print novelty scores.

        Args:
            scores: Novelty score(s) to display.
            format: Override output format.
        """
        output_format = format or self.config.format

        if output_format == OutputFormat.JSON:
            if isinstance(scores, Sequence):
                data: list[Any] = [s.to_dict() for s in scores]
            else:
                data = [scores.to_dict()]
            self.print_json(data)
        else:
            formatted = self._novelty_formatter.format(scores)  # type: ignore[arg-type]
            self.print(formatted)

    def print_explanation(
        self,
        explanation: Explanation | Sequence[Explanation],
        *,
        format: OutputFormat | None = None,
    ) -> None:
        """
        Print LLM explanation.

        Args:
            explanation: Explanation(s) to display.
            format: Override output format.
        """
        output_format = format or self.config.format

        if output_format == OutputFormat.JSON:
            if isinstance(explanation, Sequence):
                data: list[Any] = [e.model_dump() for e in explanation]
            else:
                data = [explanation.model_dump()]
            self.print_json(data)
        else:
            formatted = self._explanation_formatter.format(explanation)  # type: ignore[arg-type]
            self.print(formatted)

    def print_logs(
        self,
        records: LogRecord | Sequence[LogRecord],
        *,
        format: OutputFormat | None = None,
    ) -> None:
        """
        Print log records.

        Args:
            records: Log record(s) to display.
            format: Override output format.
        """
        output_format = format or self.config.format

        if output_format == OutputFormat.JSON:
            if isinstance(records, Sequence) and not isinstance(records, LogRecord):
                data: list[Any] = [r.model_dump() for r in records]
            else:
                # Single record
                single = records  # type: LogRecord
                data = [single.model_dump()]
            self.print_json(data)
        else:
            formatted = self._log_formatter.format(records)  # type: ignore[arg-type]
            self.print(formatted)

    def print_profile(self, timings: list[ProfileTiming]) -> None:
        """
        Print profiling information.

        Args:
            timings: List of timing data.
        """
        formatted = self._profile_formatter.format(timings)
        self.print(formatted)

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    def progress(
        self,
        name: str,
        total: int = 0,
        status: str = "",
    ) -> Any:
        """
        Create a progress tracking context.

        Args:
            name: Task name.
            total: Total items (0 = indeterminate).
            status: Initial status.

        Yields:
            TaskProgress for updating progress.

        Example:
            with console.progress("Loading", total=100) as task:
                for item in items:
                    process(item)
                    task.advance()
        """
        return self._progress_tracker.task(name, total, status)

    def spinner(
        self,
        message: str,
        spinner_type: SpinnerType = SpinnerType.DOTS,
    ) -> SpinnerContext:
        """
        Create a spinner for indeterminate operations.

        Args:
            message: Message to display.
            spinner_type: Type of spinner.

        Returns:
            SpinnerContext for use as context manager.

        Example:
            with console.spinner("Processing..."):
                do_something()
        """
        return SpinnerContext(
            message=message,
            spinner_type=spinner_type,
            output=self.config.error_output,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def confirm(self, message: str, default: bool = False) -> bool:
        """
        Ask for user confirmation.

        Args:
            message: Confirmation message.
            default: Default value if user presses Enter.

        Returns:
            True if confirmed, False otherwise.
        """
        suffix = " [Y/n]" if default else " [y/N]"
        prompt = f"{message}{suffix}: "

        try:
            response = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.print("")
            return False

        if not response:
            return default

        return response in ("y", "yes", "1", "true")

    def prompt(
        self,
        message: str,
        default: str | None = None,
    ) -> str:
        """
        Prompt for user input.

        Args:
            message: Prompt message.
            default: Default value.

        Returns:
            User input or default.
        """
        prompt = f"{message} [{default}]: " if default else f"{message}: "

        try:
            response = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            self.print("")
            return default or ""

        return response or default or ""

    def select(
        self,
        message: str,
        options: list[str],
        default: int = 0,
    ) -> int:
        """
        Display a selection menu.

        Args:
            message: Selection prompt.
            options: List of options.
            default: Default option index.

        Returns:
            Selected option index.
        """
        self.print(message)
        for i, option in enumerate(options):
            marker = ">" if i == default else " "
            self.print(f"  {marker} {i + 1}. {option}")

        while True:
            try:
                response = input(f"Select [1-{len(options)}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.print("")
                return default

            if not response:
                return default

            try:
                idx = int(response) - 1
                if 0 <= idx < len(options):
                    return idx
            except ValueError:
                pass

            self.warning(f"Please enter a number between 1 and {len(options)}")

    def capture(self) -> CapturedOutput:
        """
        Capture output for testing or redirection.

        Returns:
            CapturedOutput context manager.

        Example:
            with console.capture() as captured:
                console.info("Hello")
            print(captured.output)
        """
        return CapturedOutput(self)


class CapturedOutput:
    """Context manager for capturing console output."""

    def __init__(self, console: Console) -> None:
        """Initialize capture."""
        self.console = console
        self._original_output: TextIO | None = None
        self._original_error: TextIO | None = None
        self._captured_output = io.StringIO()
        self._captured_error = io.StringIO()

    def __enter__(self) -> CapturedOutput:
        """Start capturing."""
        self._original_output = self.console.config.output
        self._original_error = self.console.config.error_output
        self.console.config.output = self._captured_output
        self.console.config.error_output = self._captured_error
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop capturing."""
        if self._original_output:
            self.console.config.output = self._original_output
        if self._original_error:
            self.console.config.error_output = self._original_error

    @property
    def output(self) -> str:
        """Get captured stdout."""
        return self._captured_output.getvalue()

    @property
    def error(self) -> str:
        """Get captured stderr."""
        return self._captured_error.getvalue()
