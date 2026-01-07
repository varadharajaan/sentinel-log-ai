"""
PII Redaction Module for Sentinel Log AI.

This module provides configurable PII (Personally Identifiable Information)
redaction with support for multiple pattern types and redaction strategies.

Design Patterns:
- Strategy Pattern: Different redaction strategies (mask, hash, remove)
- Chain of Responsibility: Pipeline of redactors
- Factory Pattern: RedactorFactory for creating redactor instances
- Composite Pattern: CompositeRedactor for combining multiple redactors
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

import structlog

logger = structlog.get_logger(__name__)


class PIIType(Enum):
    """
    Types of PII that can be detected and redacted.

    Attributes:
        EMAIL: Email addresses.
        PHONE: Phone numbers (various formats).
        SSN: Social Security Numbers.
        CREDIT_CARD: Credit card numbers.
        IP_ADDRESS: IPv4 and IPv6 addresses.
        API_KEY: API keys and tokens.
        PASSWORD: Passwords in log messages.
        USERNAME: Usernames.
        NAME: Personal names.
        ADDRESS: Physical addresses.
        DATE_OF_BIRTH: Birth dates.
        CUSTOM: User-defined patterns.
    """

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    PASSWORD = "password"
    USERNAME = "username"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    CUSTOM = "custom"


class RedactionLevel(Enum):
    """
    Level of redaction to apply.

    Attributes:
        NONE: No redaction (passthrough).
        MINIMAL: Redact only high-risk PII (SSN, credit cards).
        STANDARD: Redact common PII (emails, phones, IPs).
        STRICT: Redact all detected PII.
        PARANOID: Maximum redaction including potential false positives.
    """

    NONE = "none"
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass(frozen=True)
class CustomPattern:
    """
    A custom redaction pattern.

    Attributes:
        name: Pattern name for identification.
        pattern: Regular expression pattern.
        pii_type: Type of PII this pattern detects.
        description: Human-readable description.
        replacement: Custom replacement string (optional).
    """

    name: str
    pattern: str
    pii_type: PIIType = PIIType.CUSTOM
    description: str = ""
    replacement: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "pattern": self.pattern,
            "pii_type": self.pii_type.value,
            "description": self.description,
            "replacement": self.replacement,
        }


@dataclass(frozen=True)
class RedactionResult:
    """
    Result of a redaction operation.

    Attributes:
        original_text: Original text before redaction.
        redacted_text: Text after redaction.
        redactions: List of redactions performed.
        pii_types_found: Set of PII types detected.
        redaction_count: Total number of redactions.
    """

    original_text: str
    redacted_text: str
    redactions: tuple[dict[str, Any], ...]
    pii_types_found: frozenset[PIIType]
    redaction_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "original_length": len(self.original_text),
            "redacted_length": len(self.redacted_text),
            "redactions": list(self.redactions),
            "pii_types_found": [t.value for t in self.pii_types_found],
            "redaction_count": self.redaction_count,
        }


@dataclass
class RedactionStats:
    """
    Statistics for redaction operations.

    Attributes:
        total_processed: Total texts processed.
        total_redactions: Total redactions performed.
        redactions_by_type: Redactions grouped by PII type.
        processing_time_ms: Total processing time.
        last_updated: Last update timestamp.
    """

    total_processed: int = 0
    total_redactions: int = 0
    redactions_by_type: dict[PIIType, int] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def record(self, result: RedactionResult, time_ms: float) -> None:
        """Record a redaction result."""
        self.total_processed += 1
        self.total_redactions += result.redaction_count
        self.processing_time_ms += time_ms
        self.last_updated = datetime.now(tz=timezone.utc)

        for pii_type in result.pii_types_found:
            self.redactions_by_type[pii_type] = self.redactions_by_type.get(pii_type, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_processed": self.total_processed,
            "total_redactions": self.total_redactions,
            "redactions_by_type": {k.value: v for k, v in self.redactions_by_type.items()},
            "processing_time_ms": self.processing_time_ms,
            "avg_time_per_text_ms": (
                self.processing_time_ms / self.total_processed if self.total_processed > 0 else 0.0
            ),
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class RedactionConfig:
    """
    Configuration for redaction behavior.

    Attributes:
        level: Redaction level to apply.
        enabled_types: Set of PII types to redact.
        custom_patterns: Additional custom patterns.
        replacement_format: Format for replacement text.
        hash_sensitive: Hash instead of mask for traceability.
        preserve_format: Preserve original format where possible.
        log_redactions: Log redaction events.
    """

    level: RedactionLevel = RedactionLevel.STANDARD
    enabled_types: set[PIIType] = field(default_factory=set)
    custom_patterns: list[CustomPattern] = field(default_factory=list)
    replacement_format: str = "[REDACTED:{type}]"
    hash_sensitive: bool = False
    preserve_format: bool = False
    log_redactions: bool = True

    def __post_init__(self) -> None:
        """Set default enabled types based on level."""
        if not self.enabled_types:
            self.enabled_types = self._get_default_types()

    def _get_default_types(self) -> set[PIIType]:
        """Get default PII types for the configured level."""
        level_types: dict[RedactionLevel, set[PIIType]] = {
            RedactionLevel.NONE: set(),
            RedactionLevel.MINIMAL: {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PASSWORD},
            RedactionLevel.STANDARD: {
                PIIType.EMAIL,
                PIIType.PHONE,
                PIIType.SSN,
                PIIType.CREDIT_CARD,
                PIIType.IP_ADDRESS,
                PIIType.API_KEY,
                PIIType.PASSWORD,
            },
            RedactionLevel.STRICT: {
                PIIType.EMAIL,
                PIIType.PHONE,
                PIIType.SSN,
                PIIType.CREDIT_CARD,
                PIIType.IP_ADDRESS,
                PIIType.API_KEY,
                PIIType.PASSWORD,
                PIIType.USERNAME,
                PIIType.NAME,
            },
            RedactionLevel.PARANOID: set(PIIType),
        }
        return level_types.get(self.level, set())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "level": self.level.value,
            "enabled_types": [t.value for t in self.enabled_types],
            "custom_patterns": [p.to_dict() for p in self.custom_patterns],
            "replacement_format": self.replacement_format,
            "hash_sensitive": self.hash_sensitive,
            "preserve_format": self.preserve_format,
            "log_redactions": self.log_redactions,
        }


class Redactor(ABC):
    """
    Abstract base class for redactors.

    All redactor implementations must inherit from this class
    and implement the redact method.
    """

    @abstractmethod
    def redact(self, text: str) -> RedactionResult:
        """
        Redact PII from text.

        Args:
            text: Input text to redact.

        Returns:
            RedactionResult with redacted text and metadata.
        """
        pass

    @abstractmethod
    def get_pii_types(self) -> set[PIIType]:
        """Get the PII types this redactor handles."""
        pass


class RegexRedactor(Redactor):
    """
    Regex-based PII redactor.

    Uses regular expressions to detect and redact PII patterns.
    """

    PATTERNS: ClassVar[dict[PIIType, list[tuple[str, str]]]] = {
        PIIType.EMAIL: [
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email address"),
        ],
        PIIType.PHONE: [
            (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "US phone"),
            (r"\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b", "international phone"),
            (r"\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b", "formatted phone"),
        ],
        PIIType.SSN: [
            (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "SSN"),
        ],
        PIIType.CREDIT_CARD: [
            (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "credit card"),
            (r"\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b", "AMEX card"),
        ],
        PIIType.IP_ADDRESS: [
            (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "IPv4"),
            (r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b", "IPv6 full"),
            (r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b", "IPv6 compressed"),
        ],
        PIIType.API_KEY: [
            (r"\b[A-Za-z0-9]{32,}\b", "API key"),
            (r"\bsk[-_][A-Za-z0-9]{24,}\b", "secret key"),
            (r"\bapi[-_]?key[=:]\s*['\"]?[A-Za-z0-9_-]+['\"]?", "API key assignment"),
            (r"\btoken[=:]\s*['\"]?[A-Za-z0-9_.-]+['\"]?", "token assignment"),
            (r"\bbearer\s+[A-Za-z0-9_.-]+", "bearer token"),
        ],
        PIIType.PASSWORD: [
            (r"\bpassword[=:]\s*['\"]?[^\s'\"]+['\"]?", "password field"),
            (r"\bpwd[=:]\s*['\"]?[^\s'\"]+['\"]?", "pwd field"),
            (r"\bsecret[=:]\s*['\"]?[^\s'\"]+['\"]?", "secret field"),
        ],
        PIIType.USERNAME: [
            (r"\buser(?:name)?[=:]\s*['\"]?[A-Za-z0-9_.-]+['\"]?", "username field"),
            (r"\blogin[=:]\s*['\"]?[A-Za-z0-9_.-]+['\"]?", "login field"),
        ],
        PIIType.DATE_OF_BIRTH: [
            (r"\b(?:dob|birth[-_]?date)[=:]\s*\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", "DOB field"),
            (r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b", "date format"),
        ],
    }

    def __init__(self, config: RedactionConfig) -> None:
        """
        Initialize the regex redactor.

        Args:
            config: Redaction configuration.
        """
        self.config = config
        self._compiled_patterns: dict[PIIType, list[tuple[re.Pattern[str], str]]] = {}
        self._compile_patterns()
        self.stats = RedactionStats()

        logger.info(
            "regex_redactor_initialized",
            level=config.level.value,
            enabled_types=[t.value for t in config.enabled_types],
        )

    def _compile_patterns(self) -> None:
        """Compile regex patterns for enabled PII types."""
        for pii_type in self.config.enabled_types:
            if pii_type in self.PATTERNS:
                self._compiled_patterns[pii_type] = [
                    (re.compile(pattern, re.IGNORECASE), desc)
                    for pattern, desc in self.PATTERNS[pii_type]
                ]

        for custom in self.config.custom_patterns:
            if custom.pii_type not in self._compiled_patterns:
                self._compiled_patterns[custom.pii_type] = []
            self._compiled_patterns[custom.pii_type].append(
                (re.compile(custom.pattern, re.IGNORECASE), custom.name)
            )

    def redact(self, text: str) -> RedactionResult:
        """
        Redact PII from text using regex patterns.

        Args:
            text: Input text to redact.

        Returns:
            RedactionResult with redacted text and metadata.
        """
        import time

        start_time = time.perf_counter()

        if self.config.level == RedactionLevel.NONE:
            return RedactionResult(
                original_text=text,
                redacted_text=text,
                redactions=(),
                pii_types_found=frozenset(),
                redaction_count=0,
            )

        redacted_text = text
        redactions: list[dict[str, Any]] = []
        pii_types_found: set[PIIType] = set()

        for pii_type, patterns in self._compiled_patterns.items():
            for pattern, desc in patterns:
                matches = list(pattern.finditer(redacted_text))
                for match in reversed(matches):
                    original_value = match.group()
                    replacement = self._get_replacement(pii_type, original_value)

                    redactions.append(
                        {
                            "type": pii_type.value,
                            "description": desc,
                            "start": match.start(),
                            "end": match.end(),
                            "replacement": replacement,
                        }
                    )
                    pii_types_found.add(pii_type)

                    redacted_text = (
                        redacted_text[: match.start()] + replacement + redacted_text[match.end() :]
                    )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            redactions=tuple(redactions),
            pii_types_found=frozenset(pii_types_found),
            redaction_count=len(redactions),
        )

        self.stats.record(result, elapsed_ms)

        if self.config.log_redactions and redactions:
            logger.debug(
                "pii_redacted",
                redaction_count=len(redactions),
                pii_types=[t.value for t in pii_types_found],
            )

        return result

    def _get_replacement(self, pii_type: PIIType, original: str) -> str:
        """Generate replacement text for redacted content."""
        if self.config.hash_sensitive:
            hash_value = hashlib.sha256(original.encode()).hexdigest()[:8]
            return self.config.replacement_format.format(type=pii_type.value) + f":{hash_value}"

        if self.config.preserve_format:
            return self._preserve_format_replacement(pii_type, original)

        return self.config.replacement_format.format(type=pii_type.value)

    def _preserve_format_replacement(self, pii_type: PIIType, original: str) -> str:
        """Generate format-preserving replacement."""
        if pii_type == PIIType.EMAIL:
            parts = original.split("@")
            if len(parts) == 2:
                return f"{'*' * len(parts[0])}@{'*' * len(parts[1])}"

        if pii_type == PIIType.PHONE:
            return re.sub(r"\d", "*", original)

        if pii_type == PIIType.CREDIT_CARD:
            return re.sub(r"\d(?=\d{4})", "*", original)

        if pii_type == PIIType.IP_ADDRESS:
            return "xxx.xxx.xxx.xxx"

        return "*" * len(original)

    def get_pii_types(self) -> set[PIIType]:
        """Get the PII types this redactor handles."""
        return set(self._compiled_patterns.keys())


class CompositeRedactor(Redactor):
    """
    Composite redactor that chains multiple redactors.

    Implements the Composite pattern to allow combining
    multiple redaction strategies.
    """

    def __init__(self, redactors: list[Redactor]) -> None:
        """
        Initialize the composite redactor.

        Args:
            redactors: List of redactors to chain.
        """
        self.redactors = redactors
        self.stats = RedactionStats()

        logger.info(
            "composite_redactor_initialized",
            redactor_count=len(redactors),
        )

    def redact(self, text: str) -> RedactionResult:
        """
        Redact PII using all configured redactors.

        Args:
            text: Input text to redact.

        Returns:
            Combined RedactionResult from all redactors.
        """
        import time

        start_time = time.perf_counter()

        current_text = text
        all_redactions: list[dict[str, Any]] = []
        all_pii_types: set[PIIType] = set()

        for redactor in self.redactors:
            result = redactor.redact(current_text)
            current_text = result.redacted_text
            all_redactions.extend(result.redactions)
            all_pii_types.update(result.pii_types_found)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = RedactionResult(
            original_text=text,
            redacted_text=current_text,
            redactions=tuple(all_redactions),
            pii_types_found=frozenset(all_pii_types),
            redaction_count=len(all_redactions),
        )

        self.stats.record(result, elapsed_ms)

        return result

    def get_pii_types(self) -> set[PIIType]:
        """Get all PII types handled by child redactors."""
        types: set[PIIType] = set()
        for redactor in self.redactors:
            types.update(redactor.get_pii_types())
        return types


class RedactorFactory:
    """
    Factory for creating redactor instances.

    Implements the Factory pattern to centralize
    redactor creation logic.
    """

    @staticmethod
    def create(config: RedactionConfig) -> Redactor:
        """
        Create a redactor based on configuration.

        Args:
            config: Redaction configuration.

        Returns:
            Configured Redactor instance.
        """
        logger.info(
            "creating_redactor",
            level=config.level.value,
            custom_patterns=len(config.custom_patterns),
        )

        return RegexRedactor(config)

    @staticmethod
    def create_composite(configs: list[RedactionConfig]) -> CompositeRedactor:
        """
        Create a composite redactor from multiple configurations.

        Args:
            configs: List of redaction configurations.

        Returns:
            CompositeRedactor combining all configurations.
        """
        redactors = [RedactorFactory.create(config) for config in configs]
        return CompositeRedactor(redactors)

    @staticmethod
    def create_default(level: RedactionLevel = RedactionLevel.STANDARD) -> Redactor:
        """
        Create a redactor with default configuration.

        Args:
            level: Redaction level to apply.

        Returns:
            Redactor with default settings.
        """
        config = RedactionConfig(level=level)
        return RedactorFactory.create(config)
