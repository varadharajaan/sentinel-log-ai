"""
Report generation for CLI.

Generates Markdown and HTML reports from analysis results.
Supports export to files and clipboard.

Design Patterns:
- Template Method: Base report structure with customizable sections
- Strategy Pattern: Different output formats (Markdown, HTML)
"""

from __future__ import annotations

import html
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from sentinel_ml.models import ClusterSummary, Explanation
    from sentinel_ml.novelty import NoveltyScore

logger = get_logger(__name__)


@dataclass
class ReportConfig:
    """
    Configuration for report generation.

    Attributes:
        title: Report title.
        include_timestamp: Add generation timestamp.
        include_summary: Add executive summary section.
        include_toc: Add table of contents.
        max_clusters: Maximum clusters to include.
        max_novel: Maximum novelty items to include.
        include_raw_data: Include raw data appendix.
        author: Report author.
        custom_css: Custom CSS for HTML reports.
    """

    title: str = "Log Analysis Report"
    include_timestamp: bool = True
    include_summary: bool = True
    include_toc: bool = True
    max_clusters: int = 20
    max_novel: int = 10
    include_raw_data: bool = False
    author: str = ""
    custom_css: str = ""


@dataclass
class ReportData:
    """
    Data container for report generation.

    Attributes:
        clusters: Cluster analysis results.
        novelty_scores: Novelty detection results.
        explanations: LLM explanations.
        stats: Summary statistics.
        metadata: Additional metadata.
    """

    clusters: list[ClusterSummary] = field(default_factory=list)
    novelty_scores: list[NoveltyScore] = field(default_factory=list)
    explanations: list[Explanation] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Reporter(ABC):
    """
    Abstract base class for report generators.

    Implements Template Method pattern with customizable sections.
    """

    def __init__(self, config: ReportConfig | None = None) -> None:
        """
        Initialize reporter.

        Args:
            config: Report configuration.
        """
        self.config = config or ReportConfig()
        logger.debug(
            "reporter_initialized",
            reporter=self.__class__.__name__,
            title=self.config.title,
        )

    def generate(self, data: ReportData) -> str:
        """
        Generate complete report.

        Template method that calls hook methods for each section.

        Args:
            data: Report data.

        Returns:
            Complete report as string.
        """
        sections: list[str] = []

        # Header
        sections.append(self._render_header())

        # Table of contents
        if self.config.include_toc:
            sections.append(self._render_toc(data))

        # Executive summary
        if self.config.include_summary:
            sections.append(self._render_summary(data))

        # Cluster analysis
        if data.clusters:
            sections.append(self._render_clusters(data.clusters))

        # Novelty detection
        if data.novelty_scores:
            sections.append(self._render_novelty(data.novelty_scores))

        # LLM explanations
        if data.explanations:
            sections.append(self._render_explanations(data.explanations))

        # Statistics
        if data.stats:
            sections.append(self._render_stats(data.stats))

        # Footer
        sections.append(self._render_footer())

        return self._join_sections(sections)

    def save(self, data: ReportData, path: str | Path) -> Path:
        """
        Generate and save report to file.

        Args:
            data: Report data.
            path: Output file path.

        Returns:
            Path to saved file.
        """
        content = self.generate(data)
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

        logger.info(
            "report_saved",
            path=str(file_path),
            size_bytes=len(content),
        )

        return file_path

    @abstractmethod
    def _render_header(self) -> str:
        """Render report header."""
        ...

    @abstractmethod
    def _render_toc(self, data: ReportData) -> str:
        """Render table of contents."""
        ...

    @abstractmethod
    def _render_summary(self, data: ReportData) -> str:
        """Render executive summary."""
        ...

    @abstractmethod
    def _render_clusters(self, clusters: list[ClusterSummary]) -> str:
        """Render cluster analysis section."""
        ...

    @abstractmethod
    def _render_novelty(self, scores: list[NoveltyScore]) -> str:
        """Render novelty detection section."""
        ...

    @abstractmethod
    def _render_explanations(self, explanations: list[Explanation]) -> str:
        """Render LLM explanations section."""
        ...

    @abstractmethod
    def _render_stats(self, stats: dict[str, Any]) -> str:
        """Render statistics section."""
        ...

    @abstractmethod
    def _render_footer(self) -> str:
        """Render report footer."""
        ...

    @abstractmethod
    def _join_sections(self, sections: list[str]) -> str:
        """Join sections into final report."""
        ...


class MarkdownReporter(Reporter):
    """
    Generate reports in Markdown format.

    Produces clean, readable Markdown that can be:
    - Viewed in GitHub/GitLab
    - Converted to PDF
    - Used in documentation
    """

    def _render_header(self) -> str:
        """Render Markdown header."""
        lines = [
            f"# {self.config.title}",
            "",
        ]

        if self.config.include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"*Generated: {timestamp}*")
            lines.append("")

        if self.config.author:
            lines.append(f"*Author: {self.config.author}*")
            lines.append("")

        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def _render_toc(self, data: ReportData) -> str:
        """Render Markdown table of contents."""
        lines = [
            "## Table of Contents",
            "",
        ]

        toc_items = []
        if self.config.include_summary:
            toc_items.append("- [Executive Summary](#executive-summary)")
        if data.clusters:
            toc_items.append("- [Cluster Analysis](#cluster-analysis)")
        if data.novelty_scores:
            toc_items.append("- [Novelty Detection](#novelty-detection)")
        if data.explanations:
            toc_items.append("- [LLM Explanations](#llm-explanations)")
        if data.stats:
            toc_items.append("- [Statistics](#statistics)")

        lines.extend(toc_items)
        lines.append("")

        return "\n".join(lines)

    def _render_summary(self, data: ReportData) -> str:
        """Render executive summary."""
        lines = [
            "## Executive Summary",
            "",
        ]

        # Calculate summary stats
        cluster_count = len(data.clusters)
        novel_count = sum(1 for s in data.novelty_scores if s.is_novel)
        high_severity = sum(
            1
            for e in data.explanations
            if getattr(e, "severity", "").upper() in ("HIGH", "CRITICAL")
        )

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Clusters | {cluster_count} |")
        lines.append(f"| Novel Patterns | {novel_count} |")
        lines.append(f"| High Severity Items | {high_severity} |")
        lines.append(f"| Explanations Generated | {len(data.explanations)} |")
        lines.append("")

        # Key findings
        if data.clusters:
            lines.append("### Key Findings")
            lines.append("")

            # Top clusters by size
            sorted_clusters = sorted(data.clusters, key=lambda c: c.size, reverse=True)
            for i, cluster in enumerate(sorted_clusters[:3], 1):
                lines.append(f"{i}. **Cluster {cluster.cluster_id[:8]}**: {cluster.size} logs")
                lines.append(f"   - Representative: `{cluster.representative[:60]}...`")
            lines.append("")

        return "\n".join(lines)

    def _render_clusters(self, clusters: list[ClusterSummary]) -> str:
        """Render cluster analysis section."""
        lines = [
            "## Cluster Analysis",
            "",
            f"Found **{len(clusters)} clusters** in the analyzed logs.",
            "",
        ]

        # Cluster table
        lines.append("| Cluster ID | Size | Cohesion | Novelty | Keywords |")
        lines.append("|------------|------|----------|---------|----------|")

        for cluster in clusters[: self.config.max_clusters]:
            cluster_id = cluster.cluster_id[:12]
            keywords = ", ".join(cluster.keywords[:3]) if cluster.keywords else "-"
            lines.append(
                f"| {cluster_id} | {cluster.size} | "
                f"{cluster.cohesion:.2f} | {cluster.novelty_score:.2f} | {keywords} |"
            )

        if len(clusters) > self.config.max_clusters:
            lines.append(f"\n*... and {len(clusters) - self.config.max_clusters} more clusters*")

        lines.append("")

        # Detailed cluster info
        lines.append("### Cluster Details")
        lines.append("")

        for cluster in clusters[: self.config.max_clusters]:
            lines.append(f"#### Cluster {cluster.cluster_id[:12]}")
            lines.append("")
            lines.append(f"- **Size**: {cluster.size} logs")
            lines.append(f"- **Cohesion**: {cluster.cohesion:.3f}")
            lines.append(f"- **Novelty Score**: {cluster.novelty_score:.3f}")
            if cluster.is_new:
                lines.append("- **Status**: 🆕 New cluster")
            lines.append("")
            lines.append("**Representative Log:**")
            lines.append("```")
            lines.append(cluster.representative[:200])
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def _render_novelty(self, scores: list[NoveltyScore]) -> str:
        """Render novelty detection section."""
        novel_items = [s for s in scores if s.is_novel]

        lines = [
            "## Novelty Detection",
            "",
            f"Detected **{len(novel_items)} novel patterns** out of {len(scores)} samples.",
            "",
        ]

        if novel_items:
            lines.append("### Novel Patterns")
            lines.append("")
            lines.append("| Score | Status | Explanation |")
            lines.append("|-------|--------|-------------|")

            sorted_novel = sorted(novel_items, key=lambda x: x.score, reverse=True)
            for score in sorted_novel[: self.config.max_novel]:
                exp = score.explanation[:50] + "..." if score.explanation else "-"
                lines.append(f"| {score.score:.3f} | ⚠️ Novel | {exp} |")

            lines.append("")

        return "\n".join(lines)

    def _render_explanations(self, explanations: list[Explanation]) -> str:
        """Render LLM explanations section."""
        lines = [
            "## LLM Explanations",
            "",
            f"Generated **{len(explanations)} explanations** for log patterns.",
            "",
        ]

        for explanation in explanations:
            cluster_id = explanation.cluster_id[:12] if len(explanation.cluster_id) > 12 else explanation.cluster_id
            lines.append(f"### Cluster {cluster_id}")
            lines.append("")

            # Confidence
            conf = f"{explanation.confidence} ({explanation.confidence_score:.0%})"
            lines.append(f"**Confidence**: {conf}")
            lines.append("")

            # Root cause
            lines.append("**Root Cause:**")
            lines.append(f"> {explanation.root_cause}")
            lines.append("")

            # Next steps
            if explanation.next_steps:
                lines.append("**Next Steps:**")
                for i, action in enumerate(explanation.next_steps, 1):
                    lines.append(f"{i}. {action}")
                lines.append("")

            # Remediation
            if explanation.remediation:
                lines.append("**Remediation:**")
                lines.append(f"> {explanation.remediation}")
                lines.append("")

        return "\n".join(lines)

    def _render_stats(self, stats: dict[str, Any]) -> str:
        """Render statistics section."""
        lines = [
            "## Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]

        for key, value in stats.items():
            formatted_key = key.replace("_", " ").title()
            lines.append(f"| {formatted_key} | {value} |")

        lines.append("")

        return "\n".join(lines)

    def _render_footer(self) -> str:
        """Render Markdown footer."""
        lines = [
            "---",
            "",
            "*Report generated by Sentinel Log AI*",
            "",
        ]
        return "\n".join(lines)

    def _join_sections(self, sections: list[str]) -> str:
        """Join sections with blank lines."""
        return "\n".join(sections)


class HTMLReporter(Reporter):
    """
    Generate reports in HTML format.

    Produces self-contained HTML reports with embedded CSS
    for viewing in web browsers.
    """

    DEFAULT_CSS = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
        line-height: 1.6;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background: #f5f5f5;
        color: #333;
    }
    .report {
        background: white;
        padding: 40px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 30px; }
    h3 { color: #7f8c8d; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #3498db; color: white; }
    tr:hover { background: #f5f5f5; }
    code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
    pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }
    .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .badge-high { background: #e74c3c; color: white; }
    .badge-medium { background: #f39c12; color: white; }
    .badge-low { background: #27ae60; color: white; }
    .badge-novel { background: #9b59b6; color: white; }
    .cluster-card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 15px 0; }
    .cluster-card h4 { margin-top: 0; color: #2c3e50; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
    .stat-card { background: #3498db; color: white; padding: 20px; border-radius: 8px; text-align: center; }
    .stat-value { font-size: 36px; font-weight: bold; }
    .stat-label { font-size: 14px; opacity: 0.9; }
    .toc { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .toc ul { list-style: none; padding-left: 20px; }
    .toc a { color: #3498db; text-decoration: none; }
    .toc a:hover { text-decoration: underline; }
    .timestamp { color: #7f8c8d; font-size: 14px; }
    blockquote { border-left: 4px solid #3498db; margin: 0; padding-left: 20px; color: #555; }
    """

    def _render_header(self) -> str:
        """Render HTML header."""
        css = self.config.custom_css or self.DEFAULT_CSS
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self.config.title)}</title>
    <style>{css}</style>
</head>
<body>
<div class="report">
    <h1>{html.escape(self.config.title)}</h1>
    <p class="timestamp">Generated: {timestamp}</p>
"""

    def _render_toc(self, data: ReportData) -> str:
        """Render HTML table of contents."""
        items = []
        if self.config.include_summary:
            items.append('<li><a href="#summary">Executive Summary</a></li>')
        if data.clusters:
            items.append('<li><a href="#clusters">Cluster Analysis</a></li>')
        if data.novelty_scores:
            items.append('<li><a href="#novelty">Novelty Detection</a></li>')
        if data.explanations:
            items.append('<li><a href="#explanations">LLM Explanations</a></li>')
        if data.stats:
            items.append('<li><a href="#stats">Statistics</a></li>')

        return f"""
    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>
            {"".join(items)}
        </ul>
    </div>
"""

    def _render_summary(self, data: ReportData) -> str:
        """Render HTML executive summary."""
        cluster_count = len(data.clusters)
        novel_count = sum(1 for s in data.novelty_scores if s.is_novel)
        high_severity = sum(
            1
            for e in data.explanations
            if getattr(e, "severity", "").upper() in ("HIGH", "CRITICAL")
        )

        return f"""
    <section id="summary">
        <h2>Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{cluster_count}</div>
                <div class="stat-label">Total Clusters</div>
            </div>
            <div class="stat-card" style="background: #9b59b6;">
                <div class="stat-value">{novel_count}</div>
                <div class="stat-label">Novel Patterns</div>
            </div>
            <div class="stat-card" style="background: #e74c3c;">
                <div class="stat-value">{high_severity}</div>
                <div class="stat-label">High Severity</div>
            </div>
            <div class="stat-card" style="background: #27ae60;">
                <div class="stat-value">{len(data.explanations)}</div>
                <div class="stat-label">Explanations</div>
            </div>
        </div>
    </section>
"""

    def _render_clusters(self, clusters: list[ClusterSummary]) -> str:
        """Render HTML cluster analysis section."""
        rows = []
        for cluster in clusters[: self.config.max_clusters]:
            cluster_id = html.escape(cluster.cluster_id[:12])
            keywords = ", ".join(cluster.keywords[:3]) if cluster.keywords else "-"
            rows.append(
                f"""<tr>
                    <td><code>{cluster_id}</code></td>
                    <td>{cluster.size}</td>
                    <td>{cluster.cohesion:.2f}</td>
                    <td>{cluster.novelty_score:.2f}</td>
                    <td>{html.escape(keywords)}</td>
                </tr>"""
            )

        cards = []
        for cluster in clusters[: self.config.max_clusters]:
            new_badge = '<span class="badge badge-novel">NEW</span>' if cluster.is_new else ""
            cards.append(
                f"""<div class="cluster-card">
                    <h4>Cluster {html.escape(cluster.cluster_id[:12])} {new_badge}</h4>
                    <p><strong>Size:</strong> {cluster.size} logs |
                       <strong>Cohesion:</strong> {cluster.cohesion:.3f} |
                       <strong>Novelty:</strong> {cluster.novelty_score:.3f}</p>
                    <p><strong>Representative:</strong></p>
                    <pre>{html.escape(cluster.representative[:200])}</pre>
                </div>"""
            )

        return f"""
    <section id="clusters">
        <h2>Cluster Analysis</h2>
        <p>Found <strong>{len(clusters)} clusters</strong> in the analyzed logs.</p>
        <table>
            <thead>
                <tr>
                    <th>Cluster ID</th>
                    <th>Size</th>
                    <th>Cohesion</th>
                    <th>Novelty</th>
                    <th>Keywords</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        <h3>Cluster Details</h3>
        {"".join(cards)}
    </section>
"""

    def _render_novelty(self, scores: list[NoveltyScore]) -> str:
        """Render HTML novelty detection section."""
        novel_items = [s for s in scores if s.is_novel]

        rows = []
        sorted_novel = sorted(novel_items, key=lambda x: x.score, reverse=True)
        for score in sorted_novel[: self.config.max_novel]:
            exp = html.escape(score.explanation[:50] + "..." if score.explanation else "-")
            rows.append(
                f"""<tr>
                    <td>{score.score:.3f}</td>
                    <td><span class="badge badge-novel">Novel</span></td>
                    <td>{exp}</td>
                </tr>"""
            )

        return f"""
    <section id="novelty">
        <h2>Novelty Detection</h2>
        <p>Detected <strong>{len(novel_items)} novel patterns</strong> out of {len(scores)} samples.</p>
        <table>
            <thead>
                <tr><th>Score</th><th>Status</th><th>Explanation</th></tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </section>
"""

    def _render_explanations(self, explanations: list[Explanation]) -> str:
        """Render HTML LLM explanations section."""
        cards = []
        for explanation in explanations:
            cluster_id = explanation.cluster_id[:12] if len(explanation.cluster_id) > 12 else explanation.cluster_id
            conf = f"{explanation.confidence} ({explanation.confidence_score:.0%})"

            actions_html = ""
            if explanation.next_steps:
                action_items = "".join(f"<li>{html.escape(a)}</li>" for a in explanation.next_steps)
                actions_html = f"<p><strong>Next Steps:</strong></p><ol>{action_items}</ol>"

            remediation_html = ""
            if explanation.remediation:
                remediation_html = f"<p><strong>Remediation:</strong></p><blockquote>{html.escape(explanation.remediation)}</blockquote>"

            root_cause_escaped = html.escape(explanation.root_cause) if explanation.root_cause else "N/A"
            cards.append(
                f"""<div class="cluster-card">
                    <h4>Cluster {html.escape(cluster_id)}</h4>
                    <p><strong>Confidence:</strong> {html.escape(conf)}</p>
                    <p><strong>Root Cause:</strong></p>
                    <blockquote>{root_cause_escaped}</blockquote>
                    {actions_html}
                    {remediation_html}
                </div>"""
            )

        return f"""
    <section id="explanations">
        <h2>LLM Explanations</h2>
        <p>Generated <strong>{len(explanations)} explanations</strong> for log patterns.</p>
        {"".join(cards)}
    </section>
"""

    def _render_stats(self, stats: dict[str, Any]) -> str:
        """Render HTML statistics section."""
        rows = []
        for key, value in stats.items():
            formatted_key = key.replace("_", " ").title()
            rows.append(
                f"<tr><td>{html.escape(formatted_key)}</td><td>{html.escape(str(value))}</td></tr>"
            )

        return f"""
    <section id="stats">
        <h2>Statistics</h2>
        <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </section>
"""

    def _render_footer(self) -> str:
        """Render HTML footer."""
        return """
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center;">
        <p>Report generated by <strong>Sentinel Log AI</strong></p>
    </footer>
</div>
</body>
</html>
"""

    def _join_sections(self, sections: list[str]) -> str:
        """Join HTML sections."""
        return "\n".join(sections)
