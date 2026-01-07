"""
Evaluation report generation for clustering quality.

This module provides automated report generation with metrics summary,
trend analysis, and multiple output formats (JSON, Markdown, HTML).

Design Patterns:
- Template Method: Report generation workflow
- Strategy Pattern: Pluggable output formatters
- Builder Pattern: Report configuration and construction
- Observer Pattern: Report event tracking

SOLID Principles:
- Single Responsibility: Each component handles one concern
- Open/Closed: Extensible via new formatters
- Liskov Substitution: All formatters implement same interface
- Interface Segregation: Minimal formatter interface
- Dependency Inversion: Depends on abstractions
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from sentinel_ml.evaluation.metrics import ClusteringQualityResult, MetricType
from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from sentinel_ml.evaluation.golden_dataset import RegressionResult

logger = get_logger(__name__)


class TrendDirection(str, Enum):
    """Direction of metric trend."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


class ReportFormat(str, Enum):
    """Output format for reports."""

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class TrendAnalysis:
    """
    Trend analysis for a metric over time.

    Attributes:
        metric_name: Name of the metric.
        current_value: Most recent metric value.
        previous_value: Previous metric value.
        direction: Trend direction.
        change_percent: Percentage change.
        history: Historical values for trend visualization.
        samples_count: Number of samples in history.
    """

    metric_name: str
    current_value: float
    previous_value: float | None = None
    direction: TrendDirection = TrendDirection.UNKNOWN
    change_percent: float = 0.0
    history: list[tuple[datetime, float]] = field(default_factory=list)
    samples_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "current_value": round(self.current_value, 6),
            "previous_value": (round(self.previous_value, 6) if self.previous_value else None),
            "direction": self.direction.value,
            "change_percent": round(self.change_percent, 2),
            "samples_count": self.samples_count,
        }


@dataclass
class EvaluationReportConfig:
    """
    Configuration for evaluation report generation.

    Attributes:
        title: Report title.
        description: Report description.
        include_trends: Whether to include trend analysis.
        include_recommendations: Whether to include recommendations.
        history_window: Number of historical samples for trends.
        output_format: Output format for the report.
        output_path: Path to save the report.
    """

    title: str = "Clustering Quality Evaluation Report"
    description: str = ""
    include_trends: bool = True
    include_recommendations: bool = True
    history_window: int = 10
    output_format: ReportFormat = ReportFormat.MARKDOWN
    output_path: Path | None = None


@dataclass
class EvaluationReport:
    """
    A complete evaluation report.

    Attributes:
        config: Report configuration.
        quality_result: Clustering quality evaluation result.
        trends: Trend analysis for each metric.
        recommendations: Generated recommendations.
        regression_result: Optional regression test result.
        summary: Executive summary.
        generated_at: When report was generated.
        metadata: Additional report metadata.
    """

    config: EvaluationReportConfig
    quality_result: ClusteringQualityResult | None = None
    trends: list[TrendAnalysis] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    regression_result: RegressionResult | None = None
    summary: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "title": self.config.title,
            "description": self.config.description,
            "summary": self.summary,
            "generated_at": self.generated_at.isoformat(),
            "trends": [t.to_dict() for t in self.trends],
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

        if self.quality_result:
            data["quality"] = self.quality_result.to_dict()

        if self.regression_result:
            data["regression"] = self.regression_result.to_dict()

        return data


class ReportFormatter(ABC):
    """Abstract base class for report formatters."""

    @abstractmethod
    def format(self, report: EvaluationReport) -> str:
        """
        Format the report as a string.

        Args:
            report: Report to format.

        Returns:
            Formatted report string.
        """


class JSONReportFormatter(ReportFormatter):
    """Format report as JSON."""

    def __init__(self, indent: int = 2) -> None:
        """Initialize formatter with indentation level."""
        self._indent = indent

    def format(self, report: EvaluationReport) -> str:
        """Format report as JSON string."""
        return json.dumps(report.to_dict(), indent=self._indent)


class MarkdownReportFormatter(ReportFormatter):
    """Format report as Markdown."""

    def format(self, report: EvaluationReport) -> str:
        """Format report as Markdown string."""
        lines = []

        # Title and description
        lines.append(f"# {report.config.title}")
        lines.append("")
        if report.config.description:
            lines.append(report.config.description)
            lines.append("")

        # Summary
        if report.summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(report.summary)
            lines.append("")

        # Quality Metrics
        if report.quality_result:
            lines.extend(self._format_quality_section(report.quality_result))

        # Trends
        if report.trends:
            lines.extend(self._format_trends_section(report.trends))

        # Regression Results
        if report.regression_result:
            lines.extend(self._format_regression_section(report.regression_result))

        # Recommendations
        if report.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Metadata
        lines.append("---")
        lines.append("")
        lines.append(f"*Generated at: {report.generated_at.isoformat()}*")

        return "\n".join(lines)

    def _format_quality_section(self, quality: ClusteringQualityResult) -> list[str]:
        """Format quality metrics section."""
        lines = []
        lines.append("## Clustering Quality Metrics")
        lines.append("")

        # Summary stats
        lines.append(f"- **Overall Quality Score**: {quality.overall_quality:.2%}")
        lines.append(f"- **Samples**: {quality.n_samples}")
        lines.append(f"- **Clusters**: {quality.n_clusters}")
        lines.append(f"- **Noise Points**: {quality.n_noise}")
        lines.append("")

        # Metrics table
        lines.append("| Metric | Value | Interpretation |")
        lines.append("|--------|-------|----------------|")

        for metric in quality.metrics:
            value_str = f"{metric.value:.4f}" if not self._is_nan(metric.value) else "N/A"
            lines.append(f"| {metric.metric_type.value} | {value_str} | {metric.interpretation} |")

        lines.append("")

        # Warnings
        if quality.warnings:
            lines.append("### Warnings")
            lines.append("")
            for warning in quality.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        return lines

    def _format_trends_section(self, trends: list[TrendAnalysis]) -> list[str]:
        """Format trends section."""
        lines = []
        lines.append("## Metric Trends")
        lines.append("")

        lines.append("| Metric | Current | Previous | Change | Trend |")
        lines.append("|--------|---------|----------|--------|-------|")

        for trend in trends:
            prev_str = f"{trend.previous_value:.4f}" if trend.previous_value else "N/A"
            change_str = f"{trend.change_percent:+.1f}%" if trend.previous_value else "N/A"
            direction_symbol = self._get_trend_symbol(trend.direction)
            lines.append(
                f"| {trend.metric_name} | {trend.current_value:.4f} | "
                f"{prev_str} | {change_str} | {direction_symbol} |"
            )

        lines.append("")
        return lines

    def _format_regression_section(self, result: RegressionResult) -> list[str]:
        """Format regression test section."""
        lines = []
        lines.append("## Regression Test Results")
        lines.append("")
        lines.append(f"- **Dataset**: {result.dataset_name} v{result.dataset_version}")
        lines.append(f"- **Status**: {result.status.value.upper()}")
        lines.append(f"- **Accuracy**: {result.accuracy:.2%}")
        lines.append(f"- **Adjusted Rand Index**: {result.adjusted_rand_index:.4f}")
        lines.append(f"- **Normalized Mutual Info**: {result.normalized_mutual_info:.4f}")
        lines.append("")

        if result.misclassified_records:
            lines.append(f"### Misclassified Records ({len(result.misclassified_records)})")
            lines.append("")
            for record_id in result.misclassified_records[:10]:  # Limit to first 10
                lines.append(f"- {record_id}")
            if len(result.misclassified_records) > 10:
                lines.append(f"- ... and {len(result.misclassified_records) - 10} more")
            lines.append("")

        return lines

    def _get_trend_symbol(self, direction: TrendDirection) -> str:
        """Get symbol for trend direction."""
        symbols = {
            TrendDirection.IMPROVING: "Improving",
            TrendDirection.STABLE: "Stable",
            TrendDirection.DEGRADING: "Degrading",
            TrendDirection.UNKNOWN: "Unknown",
        }
        return symbols.get(direction, "?")

    def _is_nan(self, value: float) -> bool:
        """Check if value is NaN."""
        import math

        return math.isnan(value)


class HTMLReportFormatter(ReportFormatter):
    """Format report as HTML."""

    def format(self, report: EvaluationReport) -> str:
        """Format report as HTML string."""
        md_formatter = MarkdownReportFormatter()
        markdown_content = md_formatter.format(report)

        # Basic HTML wrapper with CSS
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{report.config.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .improving {{ color: #27ae60; }}
        .degrading {{ color: #e74c3c; }}
        .stable {{ color: #7f8c8d; }}
        hr {{ border: none; border-top: 1px solid #eee; margin: 30px 0; }}
    </style>
</head>
<body>
{self._markdown_to_html(markdown_content)}
</body>
</html>"""
        return html

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert markdown to HTML (basic conversion)."""
        import re

        html = markdown

        # Headers
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)

        # Bold
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)

        # Italic
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

        # Lists
        html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        html = re.sub(r"(<li>.*</li>)", r"<ul>\1</ul>", html)

        # Tables (basic)
        lines = html.split("\n")
        in_table = False
        new_lines = []
        for line in lines:
            if line.startswith("|"):
                if not in_table:
                    new_lines.append("<table>")
                    in_table = True
                if line.startswith("|--"):
                    continue  # Skip separator
                cells = [c.strip() for c in line.split("|")[1:-1]]
                row_tag = "th" if new_lines[-1] == "<table>" else "td"
                row = "".join(f"<{row_tag}>{c}</{row_tag}>" for c in cells)
                new_lines.append(f"<tr>{row}</tr>")
            else:
                if in_table:
                    new_lines.append("</table>")
                    in_table = False
                new_lines.append(line)

        if in_table:
            new_lines.append("</table>")

        # Horizontal rules
        html = "\n".join(new_lines)
        html = re.sub(r"^---$", r"<hr>", html, flags=re.MULTILINE)

        # Paragraphs
        html = re.sub(r"\n\n", r"</p><p>", html)
        html = f"<p>{html}</p>"

        return html


class EvaluationReportGenerator:
    """
    Generator for evaluation reports.

    Coordinates report generation with quality metrics,
    trend analysis, and recommendations.

    Usage:
        generator = EvaluationReportGenerator()
        report = generator.generate(quality_result, history)
        generator.save(report, Path("report.md"))
    """

    # Formatters for each output format
    FORMATTERS: ClassVar[dict[ReportFormat, type[ReportFormatter]]] = {
        ReportFormat.JSON: JSONReportFormatter,
        ReportFormat.MARKDOWN: MarkdownReportFormatter,
        ReportFormat.HTML: HTMLReportFormatter,
    }

    def __init__(
        self,
        config: EvaluationReportConfig | None = None,
    ) -> None:
        """
        Initialize the report generator.

        Args:
            config: Report configuration. Uses defaults if not provided.
        """
        self._config = config or EvaluationReportConfig()
        self._history: list[ClusteringQualityResult] = []

        logger.info(
            "evaluation_report_generator_initialized",
            format=self._config.output_format.value,
            include_trends=self._config.include_trends,
        )

    def generate(
        self,
        quality_result: ClusteringQualityResult,
        regression_result: RegressionResult | None = None,
        history: list[ClusteringQualityResult] | None = None,
    ) -> EvaluationReport:
        """
        Generate an evaluation report.

        Args:
            quality_result: Current quality evaluation result.
            regression_result: Optional regression test result.
            history: Optional historical results for trend analysis.

        Returns:
            Generated EvaluationReport.
        """
        logger.info(
            "generating_evaluation_report",
            title=self._config.title,
        )

        # Update internal history
        if history:
            self._history = history[-self._config.history_window :]

        # Compute trends
        trends = []
        if self._config.include_trends and self._history:
            trends = self._compute_trends(quality_result)

        # Generate recommendations
        recommendations = []
        if self._config.include_recommendations:
            recommendations = self._generate_recommendations(quality_result, regression_result)

        # Generate summary
        summary = self._generate_summary(quality_result, regression_result)

        report = EvaluationReport(
            config=self._config,
            quality_result=quality_result,
            trends=trends,
            recommendations=recommendations,
            regression_result=regression_result,
            summary=summary,
        )

        logger.info(
            "evaluation_report_generated",
            n_trends=len(trends),
            n_recommendations=len(recommendations),
        )

        return report

    def _compute_trends(
        self,
        current: ClusteringQualityResult,
    ) -> list[TrendAnalysis]:
        """Compute trend analysis for each metric."""
        trends = []

        for metric in current.metrics:
            history_values = self._get_metric_history(metric.metric_type)

            if not history_values:
                trends.append(
                    TrendAnalysis(
                        metric_name=metric.metric_type.value,
                        current_value=metric.value,
                    )
                )
                continue

            previous_value = history_values[-1][1] if history_values else None
            direction = self._determine_trend_direction(
                metric.value,
                previous_value,
                metric.optimal_direction,
            )

            change_percent = 0.0
            if previous_value and previous_value != 0:
                change_percent = ((metric.value - previous_value) / abs(previous_value)) * 100

            trends.append(
                TrendAnalysis(
                    metric_name=metric.metric_type.value,
                    current_value=metric.value,
                    previous_value=previous_value,
                    direction=direction,
                    change_percent=change_percent,
                    history=history_values,
                    samples_count=len(history_values),
                )
            )

        return trends

    def _get_metric_history(
        self,
        metric_type: MetricType,
    ) -> list[tuple[datetime, float]]:
        """Get historical values for a metric."""
        history = []

        for result in self._history:
            metric = result.get_metric(metric_type)
            if metric and not self._is_nan(metric.value):
                history.append((result.timestamp, metric.value))

        return history

    def _determine_trend_direction(
        self,
        current: float,
        previous: float | None,
        optimal_direction: str,
    ) -> TrendDirection:
        """Determine trend direction based on value change."""
        if previous is None or self._is_nan(current):
            return TrendDirection.UNKNOWN

        threshold = 0.05  # 5% change threshold

        change = (current - previous) / abs(previous) if previous != 0 else 0

        if abs(change) < threshold:
            return TrendDirection.STABLE

        if optimal_direction == "higher":
            return TrendDirection.IMPROVING if change > 0 else TrendDirection.DEGRADING
        else:  # lower is better
            return TrendDirection.IMPROVING if change < 0 else TrendDirection.DEGRADING

    def _generate_recommendations(
        self,
        quality: ClusteringQualityResult,
        regression: RegressionResult | None,
    ) -> list[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        # Check silhouette score
        silhouette = quality.get_metric(MetricType.SILHOUETTE)
        if silhouette and not self._is_nan(silhouette.value) and silhouette.value < 0.25:
            recommendations.append(
                "Low silhouette score indicates poor cluster separation. "
                "Consider adjusting min_cluster_size or min_samples parameters."
            )

        # Check Davies-Bouldin index
        db_index = quality.get_metric(MetricType.DAVIES_BOULDIN)
        if db_index and not self._is_nan(db_index.value) and db_index.value > 2.0:
            recommendations.append(
                "High Davies-Bouldin index suggests clusters may be overlapping. "
                "Try increasing cluster_selection_epsilon."
            )

        # Check noise ratio
        if quality.n_samples > 0:
            noise_ratio = quality.n_noise / quality.n_samples
            if noise_ratio > 0.3:
                recommendations.append(
                    f"High noise ratio ({noise_ratio:.1%}). "
                    "Consider lowering min_cluster_size to capture more patterns."
                )

        # Check regression results
        if regression:
            if regression.status.value in ("failed", "degraded"):
                recommendations.append(
                    f"Regression test {regression.status.value}. "
                    "Review misclassified records and consider retraining."
                )
            if len(regression.new_clusters) > 0:
                recommendations.append(
                    f"Found {len(regression.new_clusters)} new clusters. "
                    "Update golden dataset if these represent valid patterns."
                )

        return recommendations

    def _generate_summary(
        self,
        quality: ClusteringQualityResult,
        regression: RegressionResult | None,
    ) -> str:
        """Generate executive summary."""
        parts = []

        # Quality summary
        quality_grade = self._get_quality_grade(quality.overall_quality)
        parts.append(
            f"Clustering quality is **{quality_grade}** (score: {quality.overall_quality:.2%})."
        )

        # Cluster stats
        parts.append(
            f"Found {quality.n_clusters} clusters from {quality.n_samples} samples "
            f"with {quality.n_noise} noise points."
        )

        # Regression summary
        if regression:
            parts.append(
                f"Regression test **{regression.status.value}** "
                f"with {regression.accuracy:.1%} accuracy."
            )

        # Warnings summary
        if quality.warnings:
            parts.append(f"Generated {len(quality.warnings)} warnings during evaluation.")

        return " ".join(parts)

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade."""
        if score >= 0.8:
            return "excellent"
        if score >= 0.6:
            return "good"
        if score >= 0.4:
            return "acceptable"
        if score >= 0.2:
            return "poor"
        return "very poor"

    def _is_nan(self, value: float) -> bool:
        """Check if value is NaN."""
        import math

        return math.isnan(value)

    def format(self, report: EvaluationReport) -> str:
        """
        Format a report to string.

        Args:
            report: Report to format.

        Returns:
            Formatted report string.
        """
        formatter_cls = self.FORMATTERS.get(self._config.output_format, MarkdownReportFormatter)
        formatter = formatter_cls()
        return formatter.format(report)

    def save(
        self,
        report: EvaluationReport,
        path: Path | None = None,
    ) -> Path:
        """
        Save report to file.

        Args:
            report: Report to save.
            path: Optional path. Uses config path if not provided.

        Returns:
            Path where report was saved.

        Raises:
            ValueError: If no path provided and no config path set.
        """
        save_path = path or self._config.output_path
        if not save_path:
            msg = "No output path provided"
            raise ValueError(msg)

        content = self.format(report)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            f.write(content)

        logger.info(
            "evaluation_report_saved",
            path=str(save_path),
            format=self._config.output_format.value,
        )

        return save_path

    def add_to_history(self, result: ClusteringQualityResult) -> None:
        """
        Add a result to the internal history.

        Args:
            result: Quality result to add.
        """
        self._history.append(result)
        if len(self._history) > self._config.history_window:
            self._history = self._history[-self._config.history_window :]

    @property
    def history(self) -> list[ClusteringQualityResult]:
        """Return the current history."""
        return self._history.copy()
