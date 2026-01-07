"""
Formatters for CLI output.

Provides various output formatters following the Strategy Pattern.
Each formatter implements a consistent interface for rendering data.

Design Pattern: Strategy Pattern - interchangeable formatting algorithms.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from sentinel_ml.cli.themes import (
    ThemeColors,
    get_confidence_color,
    get_log_level_color,
    get_novelty_color,
    get_severity_color,
)
from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    from sentinel_ml.models import ClusterSummary, Explanation, LogRecord
    from sentinel_ml.novelty import NoveltyScore

logger = get_logger(__name__)


@runtime_checkable
class FormattableRecord(Protocol):
    """Protocol for objects that can be formatted."""

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary."""
        ...


@dataclass
class FormatOptions:
    """
    Options for formatting output.

    Attributes:
        colors: Whether to use colors.
        compact: Whether to use compact output.
        max_width: Maximum output width.
        truncate_messages: Maximum message length (0 = no truncation).
        show_timestamps: Whether to show timestamps.
        show_ids: Whether to show IDs.
        time_format: Format string for timestamps.
        indent: Indentation for nested content.
    """

    colors: bool = True
    compact: bool = False
    max_width: int = 120
    truncate_messages: int = 80
    show_timestamps: bool = True
    show_ids: bool = True
    time_format: str = "%Y-%m-%d %H:%M:%S"
    indent: int = 2


class Formatter(ABC):
    """
    Abstract base class for output formatters.

    Implements Strategy Pattern for different output formats.
    All formatters produce string output that can be written to console or file.
    """

    def __init__(
        self,
        theme: ThemeColors | None = None,
        options: FormatOptions | None = None,
    ) -> None:
        """
        Initialize formatter.

        Args:
            theme: Color theme to use (None = no colors).
            options: Formatting options.
        """
        from sentinel_ml.cli.themes import DARK_THEME

        self.theme = theme or DARK_THEME
        self.options = options or FormatOptions()
        logger.debug(
            "formatter_initialized",
            formatter=self.__class__.__name__,
            colors=self.options.colors,
        )

    @abstractmethod
    def format(self, data: Any) -> str:
        """
        Format data for output.

        Args:
            data: Data to format.

        Returns:
            Formatted string output.
        """
        ...

    def _truncate(self, text: str, max_length: int | None = None) -> str:
        """Truncate text to max length."""
        max_len = max_length or self.options.truncate_messages
        if max_len <= 0 or len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _format_timestamp(self, ts: datetime | None) -> str:
        """Format timestamp for display."""
        if ts is None:
            return "-"
        return ts.strftime(self.options.time_format)

    def _colorize(self, text: str, color: str) -> str:
        """Wrap text in Rich color markup."""
        if not self.options.colors or not color:
            return text
        return f"[{color}]{text}[/{color}]"


class JSONFormatter(Formatter):
    """
    JSON output formatter.

    Produces JSON output for piping to other tools or storage.
    """

    def __init__(
        self,
        theme: ThemeColors | None = None,
        options: FormatOptions | None = None,
        pretty: bool = True,
    ) -> None:
        """
        Initialize JSON formatter.

        Args:
            theme: Color theme (not used for JSON).
            options: Formatting options.
            pretty: Whether to pretty-print JSON.
        """
        super().__init__(theme, options)
        self.pretty = pretty

    def format(self, data: Any) -> str:
        """
        Format data as JSON.

        Args:
            data: Data to format (must be JSON-serializable or have model_dump).

        Returns:
            JSON string.
        """
        if isinstance(data, FormattableRecord):
            data = data.model_dump()
        elif isinstance(data, list):
            data = [
                item.model_dump() if isinstance(item, FormattableRecord) else item for item in data
            ]

        return json.dumps(
            data,
            indent=2 if self.pretty else None,
            default=str,
            ensure_ascii=False,
        )


@dataclass
class TableColumn:
    """Definition of a table column."""

    key: str
    header: str
    width: int | None = None
    align: str = "left"  # left, center, right
    color_func: Any | None = None  # Function to get color for value


@dataclass
class TableData:
    """Data structure for table rendering."""

    columns: list[TableColumn]
    rows: list[dict[str, Any]]
    title: str | None = None
    footer: str | None = None
    caption: str | None = None


class TableFormatter(Formatter):
    """
    Rich table formatter.

    Creates beautiful ASCII/Unicode tables with colors and borders.
    Uses Rich library for rendering when available.
    """

    def __init__(
        self,
        theme: ThemeColors | None = None,
        options: FormatOptions | None = None,
        columns: list[TableColumn] | None = None,
        title: str | None = None,
        show_lines: bool = False,
    ) -> None:
        """
        Initialize table formatter.

        Args:
            theme: Color theme.
            options: Formatting options.
            columns: Column definitions.
            title: Table title.
            show_lines: Show row separator lines.
        """
        super().__init__(theme, options)
        self.columns = columns or []
        self.title = title
        self.show_lines = show_lines

    def format(self, data: list[dict[str, Any]] | TableData) -> str:
        """
        Format data as a table.

        Args:
            data: List of row dictionaries or TableData.

        Returns:
            Formatted table string (Rich markup).
        """
        if isinstance(data, TableData):
            table_data = data
        else:
            table_data = TableData(
                columns=self.columns,
                rows=data,
                title=self.title,
            )

        return self._build_table_markup(table_data)

    def _build_table_markup(self, table_data: TableData) -> str:
        """Build Rich-compatible table markup."""
        lines: list[str] = []

        # Determine column widths
        col_widths = self._calculate_widths(table_data)

        # Title
        if table_data.title:
            title_color = self.theme.header if self.options.colors else ""
            lines.append(self._colorize(table_data.title, title_color))
            lines.append("")

        # Header row
        header_parts = []
        for col in table_data.columns:
            width = col_widths.get(col.key, 10)
            header_text = (
                col.header.ljust(width) if col.align == "left" else col.header.rjust(width)
            )
            header_color = self.theme.header if self.options.colors else ""
            header_parts.append(self._colorize(header_text, header_color))
        lines.append("  ".join(header_parts))

        # Separator
        sep_parts = ["-" * col_widths.get(col.key, 10) for col in table_data.columns]
        sep_color = self.theme.border if self.options.colors else ""
        lines.append(self._colorize("  ".join(sep_parts), sep_color))

        # Data rows
        for row in table_data.rows:
            row_parts = []
            for col in table_data.columns:
                width = col_widths.get(col.key, 10)
                value = str(row.get(col.key, "-"))
                value = self._truncate(value, width)

                if col.align == "right":
                    cell_text = value.rjust(width)
                elif col.align == "center":
                    cell_text = value.center(width)
                else:
                    cell_text = value.ljust(width)

                # Apply color if color function provided
                if col.color_func and self.options.colors:
                    color = col.color_func(row.get(col.key), self.theme)
                    cell_text = self._colorize(cell_text, color)

                row_parts.append(cell_text)

            lines.append("  ".join(row_parts))

        # Footer
        if table_data.footer:
            lines.append("")
            muted_color = self.theme.muted if self.options.colors else ""
            lines.append(self._colorize(table_data.footer, muted_color))

        return "\n".join(lines)

    def _calculate_widths(self, table_data: TableData) -> dict[str, int]:
        """Calculate column widths based on content."""
        widths: dict[str, int] = {}

        for col in table_data.columns:
            if col.width:
                widths[col.key] = col.width
            else:
                # Calculate based on content
                max_content = len(col.header)
                for row in table_data.rows:
                    value = str(row.get(col.key, ""))
                    max_content = max(max_content, len(value))

                # Apply truncation limit
                if self.options.truncate_messages > 0:
                    max_content = min(max_content, self.options.truncate_messages)

                widths[col.key] = max_content

        return widths


class ClusterFormatter(Formatter):
    """
    Formatter for cluster summaries.

    Provides specialized formatting for log cluster data including
    severity coloring, representative samples, and statistics.
    """

    @dataclass
    class ClusterDisplayData:
        """Processed cluster data for display."""

        cluster_id: str
        size: int
        representative: str
        keywords: list[str]
        cohesion: float
        novelty_score: float
        common_level: str
        is_new: bool
        first_seen: str
        last_seen: str

    def format(self, data: ClusterSummary | list[ClusterSummary]) -> str:  # type: ignore[override]
        """
        Format cluster summary for display.

        Args:
            data: Single cluster or list of clusters.

        Returns:
            Formatted string output.
        """
        if not isinstance(data, list):
            data = [data]

        if not data:
            return self._colorize("No clusters found.", self.theme.muted)

        lines: list[str] = []

        # Header
        header = f"📊 Log Clusters ({len(data)} found)"
        lines.append(self._colorize(header, self.theme.header))
        lines.append("")

        for cluster in data:
            lines.extend(self._format_single_cluster(cluster))
            lines.append("")

        return "\n".join(lines)

    def _format_single_cluster(self, cluster: ClusterSummary) -> list[str]:
        """Format a single cluster."""
        lines: list[str] = []

        # Cluster header with severity color
        severity_color = get_severity_color(
            "HIGH"
            if cluster.novelty_score > 0.7
            else "MEDIUM"
            if cluster.novelty_score > 0.3
            else "LOW",
            self.theme,
        )

        cluster_id_display = (
            cluster.cluster_id[:12] if len(cluster.cluster_id) > 12 else cluster.cluster_id
        )
        header_line = f"┌─ Cluster {cluster_id_display}"
        if cluster.is_new:
            header_line += " " + self._colorize("[NEW]", self.theme.warning)
        lines.append(self._colorize(header_line, severity_color))

        # Stats row
        size_text = f"Size: {cluster.size}"
        cohesion_text = f"Cohesion: {cluster.cohesion:.2f}"
        novelty_text = f"Novelty: {cluster.novelty_score:.2f}"

        novelty_color = get_novelty_color(cluster.novelty_score, self.theme)
        lines.append(
            f"│ {size_text}  │  {cohesion_text}  │  " + self._colorize(novelty_text, novelty_color)
        )

        # Representative sample
        rep_text = self._truncate(cluster.representative, 100)
        lines.append(f"│ Representative: {self._colorize(rep_text, self.theme.info)}")

        # Keywords
        if cluster.keywords:
            keywords_str = ", ".join(cluster.keywords[:5])
            lines.append(f"│ Keywords: {self._colorize(keywords_str, self.theme.secondary)}")

        # Time range
        first_seen = self._format_timestamp(cluster.first_seen)
        last_seen = self._format_timestamp(cluster.last_seen)
        lines.append(f"└─ First: {first_seen}  │  Last: {last_seen}")

        return lines


class NoveltyFormatter(Formatter):
    """
    Formatter for novelty detection results.

    Provides specialized formatting for novelty scores with
    visual indicators and explanations.
    """

    def format(self, data: NoveltyScore | list[NoveltyScore]) -> str:  # type: ignore[override]
        """
        Format novelty scores for display.

        Args:
            data: Single score or list of scores.

        Returns:
            Formatted string output.
        """
        if not isinstance(data, list):
            data = [data]

        if not data:
            return self._colorize("No novelty scores.", self.theme.muted)

        lines: list[str] = []

        # Header
        header = f"🔍 Novelty Detection ({len(data)} samples)"
        lines.append(self._colorize(header, self.theme.header))
        lines.append("")

        # Summary stats
        scores = [s.score for s in data]
        novel_count = sum(1 for s in data if s.is_novel)
        avg_score = sum(scores) / len(scores) if scores else 0

        lines.append(f"Total: {len(data)}  │  Novel: {novel_count}  │  Avg Score: {avg_score:.3f}")
        lines.append("")

        # Individual scores (top N most novel)
        sorted_data = sorted(data, key=lambda x: x.score, reverse=True)
        for item in sorted_data[:10]:  # Show top 10
            lines.extend(self._format_single_score(item))

        if len(data) > 10:
            lines.append(self._colorize(f"  ... and {len(data) - 10} more", self.theme.muted))

        return "\n".join(lines)

    def _format_single_score(self, score: NoveltyScore) -> list[str]:
        """Format a single novelty score."""
        lines: list[str] = []

        # Score bar visualization
        bar = self._score_bar(score.score)
        score_color = get_novelty_color(score.score, self.theme)

        # Status indicator
        if score.is_novel:
            status = self._colorize("⚠ NOVEL", self.theme.warning)
        else:
            status = self._colorize("✓ known", self.theme.success)

        lines.append(f"  {bar} {self._colorize(f'{score.score:.3f}', score_color)}  {status}")

        # Explanation if available
        if score.explanation:
            exp_text = self._truncate(score.explanation, 60)
            lines.append(f"    └─ {self._colorize(exp_text, self.theme.muted)}")

        return lines

    def _score_bar(self, score: float, width: int = 20) -> str:
        """Create a visual score bar."""
        filled = int(score * width)
        empty = width - filled
        bar = "█" * filled + "░" * empty
        return f"[{bar}]"


class ExplanationFormatter(Formatter):
    """
    Formatter for LLM explanations.

    Provides specialized formatting for explanation data including
    root cause, severity, actions, and confidence.
    """

    def format(self, data: Explanation | list[Explanation]) -> str:  # type: ignore[override]
        """
        Format explanation for display.

        Args:
            data: Single explanation or list.

        Returns:
            Formatted string output.
        """
        if not isinstance(data, list):
            data = [data]

        if not data:
            return self._colorize("No explanations available.", self.theme.muted)

        lines: list[str] = []

        for explanation in data:
            lines.extend(self._format_single_explanation(explanation))
            lines.append("")

        return "\n".join(lines)

    def _format_single_explanation(self, explanation: Explanation) -> list[str]:
        """Format a single explanation."""
        lines: list[str] = []

        # Header with severity
        severity_color = get_severity_color(explanation.severity, self.theme)
        header = f"💡 Explanation for Cluster {explanation.cluster_id}"
        lines.append(self._colorize(header, self.theme.header))

        # Severity badge
        severity_badge = f"[{explanation.severity}]"
        lines.append(f"   Severity: {self._colorize(severity_badge, severity_color)}")

        # Confidence
        conf_color = get_confidence_color(explanation.confidence, self.theme)
        conf_text = f"{explanation.confidence} ({explanation.confidence_score:.0%})"
        lines.append(f"   Confidence: {self._colorize(conf_text, conf_color)}")

        lines.append("")

        # Root cause
        lines.append(self._colorize("Root Cause:", self.theme.primary))
        for line in explanation.root_cause.split("\n"):
            lines.append(f"   {line}")

        # Suggested actions
        if explanation.suggested_actions:
            lines.append("")
            lines.append(self._colorize("Suggested Actions:", self.theme.primary))
            for i, action in enumerate(explanation.suggested_actions, 1):
                lines.append(f"   {i}. {action}")

        # Summary
        if explanation.summary:
            lines.append("")
            lines.append(self._colorize("Summary:", self.theme.primary))
            lines.append(f"   {explanation.summary}")

        return lines


class LogRecordFormatter(Formatter):
    """
    Formatter for log records.

    Provides formatting for individual log records with
    level coloring and structured output.
    """

    def format(self, data: LogRecord | list[LogRecord]) -> str:  # type: ignore[override]
        """
        Format log records for display.

        Args:
            data: Single record or list.

        Returns:
            Formatted string output.
        """
        if not isinstance(data, list):
            data = [data]

        if not data:
            return self._colorize("No log records.", self.theme.muted)

        lines: list[str] = []

        for record in data:
            lines.append(self._format_single_record(record))

        return "\n".join(lines)

    def _format_single_record(self, record: LogRecord) -> str:
        """Format a single log record."""
        parts: list[str] = []

        # Timestamp
        if self.options.show_timestamps and record.timestamp:
            ts = self._format_timestamp(record.timestamp)
            parts.append(self._colorize(ts, self.theme.muted))

        # Level
        if record.level:
            level_color = get_log_level_color(record.level, self.theme)
            level_text = record.level.upper().ljust(5)
            parts.append(self._colorize(level_text, level_color))

        # Source
        parts.append(self._colorize(f"[{record.source}]", self.theme.secondary))

        # Message
        message = self._truncate(record.message)
        parts.append(message)

        return "  ".join(parts)


@dataclass
class ProfileTiming:
    """Timing information for a profiled operation."""

    name: str
    duration_ms: float
    percentage: float = 0.0
    children: list[ProfileTiming] = field(default_factory=list)


class ProfileFormatter(Formatter):
    """
    Formatter for profiling and timing data.

    Creates visual timing breakdowns for performance analysis.
    """

    def format(self, data: list[ProfileTiming]) -> str:  # type: ignore[override]
        """
        Format profiling data for display.

        Args:
            data: List of timing entries.

        Returns:
            Formatted string with timing breakdown.
        """
        if not data:
            return self._colorize("No profiling data.", self.theme.muted)

        lines: list[str] = []

        # Header
        lines.append(self._colorize("⏱ Timing Breakdown", self.theme.header))
        lines.append("")

        # Calculate total
        total_ms = sum(t.duration_ms for t in data)

        # Update percentages
        for timing in data:
            timing.percentage = (timing.duration_ms / total_ms * 100) if total_ms > 0 else 0

        # Sort by duration
        sorted_data = sorted(data, key=lambda x: x.duration_ms, reverse=True)

        # Format each entry
        for timing in sorted_data:
            lines.extend(self._format_timing(timing, 0))

        # Total
        lines.append("")
        lines.append(f"Total: {self._colorize(f'{total_ms:.2f}ms', self.theme.primary)}")

        return "\n".join(lines)

    def _format_timing(self, timing: ProfileTiming, indent: int) -> list[str]:
        """Format a single timing entry with bar chart."""
        lines: list[str] = []
        prefix = "  " * indent

        # Bar visualization
        bar_width = 30
        filled = int(timing.percentage / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Color based on percentage
        if timing.percentage > 50:
            color = self.theme.error
        elif timing.percentage > 25:
            color = self.theme.warning
        else:
            color = self.theme.success

        duration_str = f"{timing.duration_ms:>8.2f}ms"
        pct_str = f"({timing.percentage:>5.1f}%)"

        lines.append(
            f"{prefix}{timing.name.ljust(30)}  {self._colorize(bar, color)}  "
            f"{duration_str}  {pct_str}"
        )

        # Children
        for child in timing.children:
            lines.extend(self._format_timing(child, indent + 1))

        return lines
