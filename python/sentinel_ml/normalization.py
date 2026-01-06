"""
Log normalization and masking pipeline.

Normalizes log messages to reduce noise and improve clustering by:
- Masking IP addresses, UUIDs, hex tokens, numbers, timestamps
- Removing variable content that doesn't affect log semantics
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from sentinel_ml.models import LogRecord

logger = get_logger(__name__)


@dataclass
class MaskingRule:
    """A single masking rule with pattern and replacement."""

    name: str
    pattern: re.Pattern[str]
    replacement: str
    enabled: bool = True


@dataclass
class NormalizationPipeline:
    """
    Pipeline for normalizing and masking log messages.

    The pipeline applies a series of regex-based masking rules to
    convert variable content (IPs, UUIDs, etc.) to stable placeholders.
    """

    rules: list[MaskingRule] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.rules:
            self.rules = self._default_rules()

    @staticmethod
    def _default_rules() -> list[MaskingRule]:
        """Create the default set of masking rules.

        Order matters! More specific patterns should come before general ones.
        """
        return [
            # URLs (must be before paths to avoid partial matching)
            MaskingRule(
                name="url",
                pattern=re.compile(r"https?://[^\s]+"),
                replacement="<url>",
            ),
            # ISO timestamps: 2024-01-15T10:30:00.123Z
            MaskingRule(
                name="iso_timestamp",
                pattern=re.compile(
                    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
                ),
                replacement="<ts>",
            ),
            # Common log timestamps: Jan 15 10:30:00, 15/Jan/2024:10:30:00
            MaskingRule(
                name="common_timestamp",
                pattern=re.compile(
                    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
                ),
                replacement="<ts>",
            ),
            # Nginx/Apache timestamps: 15/Jan/2024:10:30:00 +0000
            MaskingRule(
                name="nginx_timestamp",
                pattern=re.compile(
                    r"\d{2}/(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/\d{4}:\d{2}:\d{2}:\d{2}\s*[+-]?\d{4}?"
                ),
                replacement="<ts>",
            ),
            # UUID: 550e8400-e29b-41d4-a716-446655440000
            MaskingRule(
                name="uuid",
                pattern=re.compile(
                    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
                ),
                replacement="<uuid>",
            ),
            # IPv4: 192.168.1.1
            MaskingRule(
                name="ipv4",
                pattern=re.compile(
                    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
                ),
                replacement="<ip>",
            ),
            # IPv6 (simplified)
            MaskingRule(
                name="ipv6",
                pattern=re.compile(
                    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|"
                    r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|"
                    r"\b::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}\b"
                ),
                replacement="<ip>",
            ),
            # Email addresses
            MaskingRule(
                name="email",
                pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
                replacement="<email>",
            ),
            # File paths (Unix-style)
            MaskingRule(
                name="unix_path",
                pattern=re.compile(r"(?:/[a-zA-Z0-9._-]+){2,}"),
                replacement="<path>",
            ),
            # Shorter numbers with context (port numbers, PIDs, etc.)
            MaskingRule(
                name="port_number",
                pattern=re.compile(r"(?<=:)\d{2,5}\b"),
                replacement="<port>",
            ),
            # PID patterns: pid=1234, PID: 1234, [1234]
            MaskingRule(
                name="pid",
                pattern=re.compile(r"(?:pid[=: ]+|PID[=: ]+|\[)\d+(?:\])?"),
                replacement="<pid>",
            ),
            # Hex tokens with 0x prefix (must contain letters to distinguish from pure numbers)
            MaskingRule(
                name="hex_token",
                pattern=re.compile(r"\b0x[0-9a-fA-F]{4,}\b|\b[0-9a-fA-F]*[a-fA-F][0-9a-fA-F]*\b(?=.*[0-9])"),
                replacement="<hex>",
            ),
            # Long numbers (5+ digits): 123456, 1234567890
            MaskingRule(
                name="long_number",
                pattern=re.compile(r"\b\d{5,}\b"),
                replacement="<num>",
            ),
        ]

    def normalize(self, message: str) -> str:
        """
        Apply all enabled masking rules to a log message.

        Args:
            message: The original log message

        Returns:
            The normalized message with variables masked
        """
        result = message
        for rule in self.rules:
            if rule.enabled:
                result = rule.pattern.sub(rule.replacement, result)

        # Collapse multiple spaces
        result = re.sub(r"\s+", " ", result).strip()

        return result

    def normalize_record(self, record: LogRecord) -> LogRecord:
        """
        Normalize a LogRecord, setting the normalized field.

        Args:
            record: The log record to normalize

        Returns:
            The same record with normalized field set
        """
        record.normalized = self.normalize(record.message)
        return record

    def add_rule(
        self,
        name: str,
        pattern: str,
        replacement: str,
        enabled: bool = True,
    ) -> None:
        """Add a custom masking rule."""
        self.rules.append(
            MaskingRule(
                name=name,
                pattern=re.compile(pattern),
                replacement=replacement,
                enabled=enabled,
            )
        )

    def disable_rule(self, name: str) -> None:
        """Disable a masking rule by name."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = False
                return

    def enable_rule(self, name: str) -> None:
        """Enable a masking rule by name."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = True
                return


# Global pipeline instance
_pipeline: NormalizationPipeline | None = None


def get_normalizer() -> NormalizationPipeline:
    """Get the global normalization pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = NormalizationPipeline()
    return _pipeline


def normalize(message: str) -> str:
    """Convenience function to normalize a message using the global pipeline."""
    return get_normalizer().normalize(message)
