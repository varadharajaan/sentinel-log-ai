"""
Theme system for CLI output.

Provides consistent color schemes and styling for terminal output.
Supports multiple themes (dark, light, minimal, colorblind-friendly).

Design Pattern: Strategy Pattern for theme selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Theme(str, Enum):
    """Available CLI themes."""

    DARK = "dark"
    LIGHT = "light"
    MINIMAL = "minimal"
    COLORBLIND = "colorblind"  # Colorblind-friendly palette
    NONE = "none"  # No colors (for piping to files)


@dataclass(frozen=True)
class ThemeColors:
    """
    Color scheme for a theme.

    All colors use Rich markup syntax.
    See: https://rich.readthedocs.io/en/stable/appendix/colors.html

    Attributes:
        primary: Primary accent color for headers and highlights.
        secondary: Secondary color for less important text.
        success: Color for success states and positive indicators.
        warning: Color for warning states and caution indicators.
        error: Color for error states and negative indicators.
        info: Color for informational text.
        muted: Color for dimmed/secondary text.
        highlight: Color for emphasized text.
        border: Color for table borders and dividers.
        header: Color for table headers.
        cluster_high: Color for high-severity clusters.
        cluster_medium: Color for medium-severity clusters.
        cluster_low: Color for low-severity clusters.
        novelty_high: Color for high novelty scores.
        novelty_medium: Color for medium novelty scores.
        novelty_low: Color for low novelty scores.
        confidence_high: Color for high confidence.
        confidence_medium: Color for medium confidence.
        confidence_low: Color for low confidence.
    """

    # Base colors
    primary: str
    secondary: str
    success: str
    warning: str
    error: str
    info: str
    muted: str
    highlight: str
    border: str
    header: str

    # Semantic colors for log analysis
    cluster_high: str
    cluster_medium: str
    cluster_low: str

    novelty_high: str
    novelty_medium: str
    novelty_low: str

    confidence_high: str
    confidence_medium: str
    confidence_low: str

    # Log level colors
    level_error: str
    level_warn: str
    level_info: str
    level_debug: str
    level_trace: str


# Theme definitions
DARK_THEME = ThemeColors(
    # Base colors
    primary="cyan",
    secondary="bright_blue",
    success="green",
    warning="yellow",
    error="red",
    info="blue",
    muted="dim",
    highlight="bold bright_white",
    border="bright_black",
    header="bold cyan",
    # Cluster severity
    cluster_high="bold red",
    cluster_medium="yellow",
    cluster_low="green",
    # Novelty scores
    novelty_high="bold magenta",
    novelty_medium="bright_yellow",
    novelty_low="dim green",
    # Confidence
    confidence_high="bold green",
    confidence_medium="yellow",
    confidence_low="dim red",
    # Log levels
    level_error="bold red",
    level_warn="yellow",
    level_info="cyan",
    level_debug="dim white",
    level_trace="dim",
)

LIGHT_THEME = ThemeColors(
    # Base colors
    primary="blue",
    secondary="dark_blue",
    success="dark_green",
    warning="dark_orange",
    error="dark_red",
    info="navy_blue",
    muted="grey50",
    highlight="bold black",
    border="grey70",
    header="bold blue",
    # Cluster severity
    cluster_high="bold dark_red",
    cluster_medium="dark_orange",
    cluster_low="dark_green",
    # Novelty scores
    novelty_high="bold purple",
    novelty_medium="orange3",
    novelty_low="grey50",
    # Confidence
    confidence_high="bold dark_green",
    confidence_medium="dark_orange",
    confidence_low="grey50",
    # Log levels
    level_error="bold dark_red",
    level_warn="dark_orange",
    level_info="blue",
    level_debug="grey50",
    level_trace="grey70",
)

MINIMAL_THEME = ThemeColors(
    # Base colors - mostly white/black
    primary="bold",
    secondary="",
    success="bold",
    warning="bold",
    error="bold",
    info="",
    muted="dim",
    highlight="bold",
    border="",
    header="bold",
    # Cluster severity
    cluster_high="bold",
    cluster_medium="",
    cluster_low="dim",
    # Novelty scores
    novelty_high="bold",
    novelty_medium="",
    novelty_low="dim",
    # Confidence
    confidence_high="bold",
    confidence_medium="",
    confidence_low="dim",
    # Log levels
    level_error="bold",
    level_warn="",
    level_info="",
    level_debug="dim",
    level_trace="dim",
)

# Colorblind-friendly theme using blue-orange diverging palette
COLORBLIND_THEME = ThemeColors(
    # Base colors - blue/orange safe for most colorblind types
    primary="dodger_blue2",
    secondary="steel_blue",
    success="dodger_blue1",
    warning="orange1",
    error="dark_orange",
    info="deep_sky_blue1",
    muted="grey62",
    highlight="bold white",
    border="grey50",
    header="bold dodger_blue2",
    # Cluster severity - using brightness instead of hue
    cluster_high="bold orange1",
    cluster_medium="grey74",
    cluster_low="dodger_blue1",
    # Novelty scores
    novelty_high="bold orange1",
    novelty_medium="grey74",
    novelty_low="dodger_blue1",
    # Confidence
    confidence_high="bold dodger_blue1",
    confidence_medium="grey74",
    confidence_low="dim orange1",
    # Log levels - use patterns + brightness
    level_error="bold orange1",
    level_warn="orange3",
    level_info="dodger_blue1",
    level_debug="grey62",
    level_trace="grey50",
)

# No-color theme for piping to files
NONE_THEME = ThemeColors(
    primary="",
    secondary="",
    success="",
    warning="",
    error="",
    info="",
    muted="",
    highlight="",
    border="",
    header="",
    cluster_high="",
    cluster_medium="",
    cluster_low="",
    novelty_high="",
    novelty_medium="",
    novelty_low="",
    confidence_high="",
    confidence_medium="",
    confidence_low="",
    level_error="",
    level_warn="",
    level_info="",
    level_debug="",
    level_trace="",
)

# Theme registry
_THEMES: dict[Theme, ThemeColors] = {
    Theme.DARK: DARK_THEME,
    Theme.LIGHT: LIGHT_THEME,
    Theme.MINIMAL: MINIMAL_THEME,
    Theme.COLORBLIND: COLORBLIND_THEME,
    Theme.NONE: NONE_THEME,
}


def get_theme(theme: Theme | str) -> ThemeColors:
    """
    Get the color scheme for a theme.

    Args:
        theme: Theme name or Theme enum value.

    Returns:
        ThemeColors for the specified theme.

    Raises:
        ValueError: If theme is not recognized.

    Example:
        >>> colors = get_theme(Theme.DARK)
        >>> print(colors.primary)
        'cyan'
    """
    if isinstance(theme, str):
        try:
            theme = Theme(theme.lower())
        except ValueError as e:
            valid = ", ".join(t.value for t in Theme)
            msg = f"Unknown theme '{theme}'. Valid themes: {valid}"
            raise ValueError(msg) from e

    if theme not in _THEMES:
        msg = f"Theme {theme} not found in registry"
        raise ValueError(msg)

    return _THEMES[theme]


def get_severity_color(
    severity: str,
    theme_colors: ThemeColors,
) -> str:
    """
    Get color for a severity level.

    Args:
        severity: Severity level (HIGH, MEDIUM, LOW, or CRITICAL).
        theme_colors: Theme colors to use.

    Returns:
        Color string for the severity level.
    """
    severity_upper = severity.upper()
    if severity_upper in ("HIGH", "CRITICAL"):
        return theme_colors.cluster_high
    if severity_upper == "MEDIUM":
        return theme_colors.cluster_medium
    return theme_colors.cluster_low


def get_novelty_color(
    score: float,
    theme_colors: ThemeColors,
    high_threshold: float = 0.7,
    medium_threshold: float = 0.4,
) -> str:
    """
    Get color for a novelty score.

    Args:
        score: Novelty score (0.0 to 1.0).
        theme_colors: Theme colors to use.
        high_threshold: Score above this is "high".
        medium_threshold: Score above this is "medium".

    Returns:
        Color string for the novelty score.
    """
    if score >= high_threshold:
        return theme_colors.novelty_high
    if score >= medium_threshold:
        return theme_colors.novelty_medium
    return theme_colors.novelty_low


def get_confidence_color(
    confidence: float | str,
    theme_colors: ThemeColors,
) -> str:
    """
    Get color for a confidence level.

    Args:
        confidence: Confidence value (0.0-1.0) or level (HIGH/MEDIUM/LOW).
        theme_colors: Theme colors to use.

    Returns:
        Color string for the confidence level.
    """
    if isinstance(confidence, str):
        conf_upper = confidence.upper()
        if conf_upper == "HIGH":
            return theme_colors.confidence_high
        if conf_upper == "MEDIUM":
            return theme_colors.confidence_medium
        return theme_colors.confidence_low

    # Numeric confidence
    if confidence >= 0.7:
        return theme_colors.confidence_high
    if confidence >= 0.4:
        return theme_colors.confidence_medium
    return theme_colors.confidence_low


def get_log_level_color(
    level: str,
    theme_colors: ThemeColors,
) -> str:
    """
    Get color for a log level.

    Args:
        level: Log level (ERROR, WARN, INFO, DEBUG, TRACE).
        theme_colors: Theme colors to use.

    Returns:
        Color string for the log level.
    """
    level_upper = level.upper()
    if level_upper in ("ERROR", "FATAL", "CRITICAL", "PANIC"):
        return theme_colors.level_error
    if level_upper in ("WARN", "WARNING"):
        return theme_colors.level_warn
    if level_upper in ("INFO", "NOTICE"):
        return theme_colors.level_info
    if level_upper == "DEBUG":
        return theme_colors.level_debug
    if level_upper in ("TRACE", "VERBOSE"):
        return theme_colors.level_trace
    # Default to info
    return theme_colors.level_info
