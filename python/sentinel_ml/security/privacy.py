"""
Privacy Management Module for Sentinel Log AI.

This module provides privacy controls including:
- Never-store-raw-logs mode
- Privacy levels and policies
- Log sanitization
- Privacy reporting

Design Patterns:
- Strategy Pattern: Different privacy modes
- Template Method: Sanitization pipeline
- Observer Pattern: Privacy event notifications
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from sentinel_ml.security.redaction import (
    PIIType,
    RedactionConfig,
    RedactionLevel,
    RedactorFactory,
)

logger = structlog.get_logger(__name__)


class PrivacyLevel(Enum):
    """
    Privacy protection levels.

    Attributes:
        BASIC: Minimal privacy controls.
        ENHANCED: Standard privacy with PII redaction.
        MAXIMUM: Maximum privacy, never store raw logs.
    """

    BASIC = "basic"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


class PrivacyMode(Enum):
    """
    Log storage privacy modes.

    Attributes:
        STORE_ALL: Store all logs (least private).
        STORE_REDACTED: Store only redacted logs.
        STORE_EMBEDDINGS_ONLY: Store only embeddings, not text.
        NEVER_STORE: Never persist any log data.
    """

    STORE_ALL = "store_all"
    STORE_REDACTED = "store_redacted"
    STORE_EMBEDDINGS_ONLY = "store_embeddings_only"
    NEVER_STORE = "never_store"


class RawLogPolicy(Enum):
    """
    Policy for handling raw log data.

    Attributes:
        ALLOW: Allow raw log storage.
        REDACT_THEN_STORE: Redact PII before storage.
        HASH_ONLY: Store only hash of raw logs.
        DISCARD: Immediately discard raw logs.
    """

    ALLOW = "allow"
    REDACT_THEN_STORE = "redact_then_store"
    HASH_ONLY = "hash_only"
    DISCARD = "discard"


@dataclass(frozen=True)
class SanitizedLog:
    """
    A sanitized log entry.

    Attributes:
        original_hash: SHA-256 hash of original log.
        sanitized_text: Sanitized log text.
        timestamp: Original log timestamp.
        metadata: Preserved metadata.
        pii_detected: Whether PII was detected.
        pii_types: Types of PII detected.
    """

    original_hash: str
    sanitized_text: str
    timestamp: datetime
    metadata: dict[str, Any]
    pii_detected: bool
    pii_types: frozenset[PIIType]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "original_hash": self.original_hash,
            "sanitized_text": self.sanitized_text,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "pii_detected": self.pii_detected,
            "pii_types": [t.value for t in self.pii_types],
        }


@dataclass
class SanitizationResult:
    """
    Result of log sanitization.

    Attributes:
        sanitized_logs: List of sanitized log entries.
        total_processed: Total logs processed.
        pii_detected_count: Logs with PII detected.
        redaction_count: Total redactions performed.
        processing_time_ms: Processing time in milliseconds.
    """

    sanitized_logs: list[SanitizedLog]
    total_processed: int
    pii_detected_count: int
    redaction_count: int
    processing_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_processed": self.total_processed,
            "pii_detected_count": self.pii_detected_count,
            "pii_detection_rate": (
                self.pii_detected_count / self.total_processed if self.total_processed > 0 else 0.0
            ),
            "redaction_count": self.redaction_count,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class PrivacyConfig:
    """
    Configuration for privacy controls.

    Attributes:
        level: Privacy protection level.
        mode: Log storage mode.
        raw_log_policy: Policy for raw log handling.
        redaction_config: PII redaction configuration.
        retain_metadata: Preserve log metadata.
        encrypt_at_rest: Enable encryption for stored data.
        audit_access: Enable access auditing.
        data_retention_days: Days to retain data (0 = forever).
    """

    level: PrivacyLevel = PrivacyLevel.ENHANCED
    mode: PrivacyMode = PrivacyMode.STORE_REDACTED
    raw_log_policy: RawLogPolicy = RawLogPolicy.REDACT_THEN_STORE
    redaction_config: RedactionConfig = field(default_factory=RedactionConfig)
    retain_metadata: bool = True
    encrypt_at_rest: bool = False
    audit_access: bool = True
    data_retention_days: int = 90

    def __post_init__(self) -> None:
        """Apply privacy level defaults."""
        if self.level == PrivacyLevel.MAXIMUM:
            self.mode = PrivacyMode.NEVER_STORE
            self.raw_log_policy = RawLogPolicy.DISCARD
            self.redaction_config = RedactionConfig(level=RedactionLevel.PARANOID)
            self.encrypt_at_rest = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "level": self.level.value,
            "mode": self.mode.value,
            "raw_log_policy": self.raw_log_policy.value,
            "redaction_config": self.redaction_config.to_dict(),
            "retain_metadata": self.retain_metadata,
            "encrypt_at_rest": self.encrypt_at_rest,
            "audit_access": self.audit_access,
            "data_retention_days": self.data_retention_days,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrivacyConfig:
        """Create from dictionary representation."""
        return cls(
            level=PrivacyLevel(data.get("level", "enhanced")),
            mode=PrivacyMode(data.get("mode", "store_redacted")),
            raw_log_policy=RawLogPolicy(data.get("raw_log_policy", "redact_then_store")),
            retain_metadata=data.get("retain_metadata", True),
            encrypt_at_rest=data.get("encrypt_at_rest", False),
            audit_access=data.get("audit_access", True),
            data_retention_days=data.get("data_retention_days", 90),
        )


@dataclass
class PrivacyReport:
    """
    Privacy compliance report.

    Attributes:
        generated_at: Report generation timestamp.
        config: Current privacy configuration.
        total_logs_processed: Total logs processed.
        pii_detections: PII detection statistics.
        storage_mode: Current storage mode.
        encryption_enabled: Whether encryption is enabled.
        data_retention_policy: Current retention policy.
        recommendations: Privacy recommendations.
    """

    generated_at: datetime
    config: PrivacyConfig
    total_logs_processed: int
    pii_detections: dict[str, int]
    storage_mode: PrivacyMode
    encryption_enabled: bool
    data_retention_policy: str
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "config": self.config.to_dict(),
            "total_logs_processed": self.total_logs_processed,
            "pii_detections": self.pii_detections,
            "storage_mode": self.storage_mode.value,
            "encryption_enabled": self.encryption_enabled,
            "data_retention_policy": self.data_retention_policy,
            "recommendations": self.recommendations,
        }


class PrivacyObserver(ABC):
    """Abstract observer for privacy events."""

    @abstractmethod
    def on_pii_detected(self, log_hash: str, pii_types: set[PIIType]) -> None:
        """Called when PII is detected in a log."""
        pass

    @abstractmethod
    def on_log_sanitized(self, log_hash: str, redaction_count: int) -> None:
        """Called when a log is sanitized."""
        pass

    @abstractmethod
    def on_storage_denied(self, log_hash: str, reason: str) -> None:
        """Called when log storage is denied by policy."""
        pass


class PrivacyManager:
    """
    Central manager for privacy controls.

    Coordinates redaction, sanitization, and privacy policy
    enforcement across the system.
    """

    def __init__(self, config: PrivacyConfig) -> None:
        """
        Initialize the privacy manager.

        Args:
            config: Privacy configuration.
        """
        self.config = config
        self.redactor = RedactorFactory.create(config.redaction_config)
        self._observers: list[PrivacyObserver] = []
        self._total_processed: int = 0
        self._pii_detected: int = 0
        self._logs_stored: int = 0
        self._logs_discarded: int = 0
        self._redactions_by_type: dict[str, int] = {}

        logger.info(
            "privacy_manager_initialized",
            level=config.level.value,
            mode=config.mode.value,
            raw_log_policy=config.raw_log_policy.value,
        )

    def add_observer(self, observer: PrivacyObserver) -> None:
        """Add a privacy event observer."""
        self._observers.append(observer)

    def remove_observer(self, observer: PrivacyObserver) -> None:
        """Remove a privacy event observer."""
        self._observers.remove(observer)

    def sanitize(self, text: str, metadata: dict[str, Any] | None = None) -> SanitizedLog:
        """
        Sanitize a log entry.

        Args:
            text: Raw log text.
            metadata: Optional log metadata.

        Returns:
            Sanitized log entry.
        """
        import hashlib
        import time

        start_time = time.perf_counter()

        original_hash = hashlib.sha256(text.encode()).hexdigest()
        metadata = metadata or {}

        if self.config.raw_log_policy == RawLogPolicy.DISCARD:
            sanitized = SanitizedLog(
                original_hash=original_hash,
                sanitized_text="",
                timestamp=datetime.now(tz=timezone.utc),
                metadata=metadata if self.config.retain_metadata else {},
                pii_detected=False,
                pii_types=frozenset(),
            )
            self._notify_storage_denied(original_hash, "raw_log_policy=discard")
            return sanitized

        if self.config.raw_log_policy == RawLogPolicy.HASH_ONLY:
            sanitized = SanitizedLog(
                original_hash=original_hash,
                sanitized_text=f"[HASH:{original_hash[:16]}]",
                timestamp=datetime.now(tz=timezone.utc),
                metadata=metadata if self.config.retain_metadata else {},
                pii_detected=False,
                pii_types=frozenset(),
            )
            return sanitized

        result = self.redactor.redact(text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        self._total_processed += 1

        if result.pii_types_found:
            self._pii_detected += 1
            for pii_type in result.pii_types_found:
                type_key = pii_type.value
                self._redactions_by_type[type_key] = (
                    self._redactions_by_type.get(type_key, 0) + 1
                )
            self._notify_pii_detected(original_hash, set(result.pii_types_found))

        sanitized = SanitizedLog(
            original_hash=original_hash,
            sanitized_text=result.redacted_text,
            timestamp=datetime.now(tz=timezone.utc),
            metadata=metadata if self.config.retain_metadata else {},
            pii_detected=bool(result.pii_types_found),
            pii_types=result.pii_types_found,
        )

        self._notify_log_sanitized(original_hash, result.redaction_count)

        logger.debug(
            "log_sanitized",
            original_hash=original_hash[:16],
            pii_detected=sanitized.pii_detected,
            redaction_count=result.redaction_count,
            elapsed_ms=elapsed_ms,
        )

        return sanitized

    def sanitize_batch(self, logs: list[tuple[str, dict[str, Any] | None]]) -> SanitizationResult:
        """
        Sanitize a batch of log entries.

        Args:
            logs: List of (text, metadata) tuples.

        Returns:
            Sanitization result with all sanitized logs.
        """
        import time

        start_time = time.perf_counter()

        sanitized_logs: list[SanitizedLog] = []
        pii_detected_count = 0
        total_redactions = 0

        for text, metadata in logs:
            sanitized = self.sanitize(text, metadata)
            sanitized_logs.append(sanitized)

            if sanitized.pii_detected:
                pii_detected_count += 1
                total_redactions += len(sanitized.pii_types)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = SanitizationResult(
            sanitized_logs=sanitized_logs,
            total_processed=len(logs),
            pii_detected_count=pii_detected_count,
            redaction_count=total_redactions,
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            "batch_sanitization_completed",
            total_processed=result.total_processed,
            pii_detected_count=result.pii_detected_count,
            elapsed_ms=elapsed_ms,
        )

        return result

    def can_store(self, sanitized_log: SanitizedLog) -> bool:
        """
        Check if a sanitized log can be stored.

        Args:
            sanitized_log: The sanitized log to check.

        Returns:
            True if storage is allowed.
        """
        _ = sanitized_log  # May be used for future policy checks

        if self.config.mode == PrivacyMode.NEVER_STORE:
            return False

        if self.config.mode == PrivacyMode.STORE_EMBEDDINGS_ONLY:
            return False

        if self.config.mode == PrivacyMode.STORE_REDACTED:
            return True

        return True

    def generate_report(self) -> PrivacyReport:
        """
        Generate a privacy compliance report.

        Returns:
            Privacy report with current statistics and recommendations.
        """
        recommendations = self._generate_recommendations()

        retention_policy = (
            f"{self.config.data_retention_days} days"
            if self.config.data_retention_days > 0
            else "indefinite"
        )

        report = PrivacyReport(
            generated_at=datetime.now(tz=timezone.utc),
            config=self.config,
            total_logs_processed=self._total_processed,
            pii_detections=self._redactions_by_type.copy(),
            storage_mode=self.config.mode,
            encryption_enabled=self.config.encrypt_at_rest,
            data_retention_policy=retention_policy,
            recommendations=recommendations,
        )

        logger.info(
            "privacy_report_generated",
            total_processed=report.total_logs_processed,
            pii_detection_count=sum(report.pii_detections.values()),
        )

        return report

    def _generate_recommendations(self) -> list[str]:
        """Generate privacy improvement recommendations."""
        recommendations: list[str] = []

        if self.config.level != PrivacyLevel.MAXIMUM:
            recommendations.append(
                "Consider using MAXIMUM privacy level for sensitive environments"
            )

        if not self.config.encrypt_at_rest:
            recommendations.append("Enable at-rest encryption to protect stored data")

        if self.config.data_retention_days == 0:
            recommendations.append("Set a data retention policy to automatically delete old data")

        if self.config.mode == PrivacyMode.STORE_ALL:
            recommendations.append("Switch to STORE_REDACTED mode to avoid storing PII")

        pii_rate = (
            self._pii_detected / self._total_processed
            if self._total_processed > 0
            else 0.0
        )

        if pii_rate > 0.5:
            recommendations.append(
                "High PII detection rate detected. Review log sources for PII leakage"
            )

        return recommendations

    def _notify_pii_detected(self, log_hash: str, pii_types: set[PIIType]) -> None:
        """Notify observers of PII detection."""
        for observer in self._observers:
            try:
                observer.on_pii_detected(log_hash, pii_types)
            except Exception as e:
                logger.warning("observer_notification_failed", error=str(e))

    def _notify_log_sanitized(self, log_hash: str, redaction_count: int) -> None:
        """Notify observers of log sanitization."""
        for observer in self._observers:
            try:
                observer.on_log_sanitized(log_hash, redaction_count)
            except Exception as e:
                logger.warning("observer_notification_failed", error=str(e))

    def _notify_storage_denied(self, log_hash: str, reason: str) -> None:
        """Notify observers of storage denial."""
        for observer in self._observers:
            try:
                observer.on_storage_denied(log_hash, reason)
            except Exception as e:
                logger.warning("observer_notification_failed", error=str(e))

    def get_stats(self) -> dict[str, Any]:
        """Get privacy statistics."""
        stats: dict[str, Any] = {
            "total_processed": self._total_processed,
            "pii_detected": self._pii_detected,
            "logs_stored": self._logs_stored,
            "logs_discarded": self._logs_discarded,
            "redactions_by_type": self._redactions_by_type.copy(),
        }
        if hasattr(self.redactor, "stats"):
            stats["redactor_stats"] = self.redactor.stats.to_dict()
        return stats

    def reset_stats(self) -> None:
        """Reset privacy statistics."""
        self._total_processed = 0
        self._pii_detected = 0
        self._logs_stored = 0
        self._logs_discarded = 0
        self._redactions_by_type = {}
        logger.info("privacy_stats_reset")
