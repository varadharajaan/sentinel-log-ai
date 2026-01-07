"""
Security and Privacy module for Sentinel Log AI.

This module provides comprehensive security controls including:
- PII redaction with configurable patterns
- Never-store-raw-logs mode for privacy-first operation
- At-rest encryption for sensitive data
- Secure defaults and privacy controls

Design Patterns:
- Strategy Pattern: Pluggable redaction strategies
- Chain of Responsibility: Redaction pipeline
- Decorator Pattern: Encryption wrappers
- Factory Pattern: Redactor creation
- Observer Pattern: Security event notifications

SOLID Principles:
- Single Responsibility: Each class has one security concern
- Open/Closed: Extensible via custom redaction patterns
- Liskov Substitution: All redactors implement same interface
- Interface Segregation: Separate read/write/encrypt interfaces
- Dependency Inversion: Depends on abstractions
"""

from sentinel_ml.security.encryption import (
    DecryptionError,
    EncryptedData,
    EncryptedStore,
    EncryptionConfig,
    EncryptionError,
    EncryptionKey,
    EncryptionProvider,
    FernetEncryptionProvider,
    KeyDerivationConfig,
    KeyManager,
)
from sentinel_ml.security.privacy import (
    PrivacyConfig,
    PrivacyLevel,
    PrivacyManager,
    PrivacyMode,
    PrivacyReport,
    RawLogPolicy,
    SanitizationResult,
    SanitizedLog,
)
from sentinel_ml.security.redaction import (
    CompositeRedactor,
    CustomPattern,
    PIIType,
    RedactionConfig,
    RedactionLevel,
    RedactionResult,
    RedactionStats,
    Redactor,
    RedactorFactory,
    RegexRedactor,
)

__all__ = [
    "CompositeRedactor",
    "CustomPattern",
    "DecryptionError",
    "EncryptedData",
    "EncryptedStore",
    "EncryptionConfig",
    "EncryptionError",
    "EncryptionKey",
    "EncryptionProvider",
    "FernetEncryptionProvider",
    "KeyDerivationConfig",
    "KeyManager",
    "PIIType",
    "PrivacyConfig",
    "PrivacyLevel",
    "PrivacyManager",
    "PrivacyMode",
    "PrivacyReport",
    "RawLogPolicy",
    "RedactionConfig",
    "RedactionLevel",
    "RedactionResult",
    "RedactionStats",
    "Redactor",
    "RedactorFactory",
    "RegexRedactor",
    "SanitizationResult",
    "SanitizedLog",
]
