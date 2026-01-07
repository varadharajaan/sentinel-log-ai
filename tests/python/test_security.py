"""
Comprehensive tests for the security module.

Tests cover:
- PII redaction (email, phone, SSN, credit cards, API keys, passwords)
- Privacy management (modes, policies, sanitization)
- Encryption (Fernet, key management, encrypted store)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from sentinel_ml.security import (
    CompositeRedactor,
    CustomPattern,
    DecryptionError,
    EncryptedData,
    EncryptedStore,
    EncryptionConfig,
    EncryptionError,
    EncryptionKey,
    FernetEncryptionProvider,
    KeyDerivationConfig,
    KeyManager,
    PIIType,
    PrivacyConfig,
    PrivacyLevel,
    PrivacyManager,
    PrivacyMode,
    RawLogPolicy,
    RedactionConfig,
    RedactionLevel,
    RedactionResult,
    RedactionStats,
    Redactor,
    RedactorFactory,
    RegexRedactor,
    SanitizedLog,
)
from sentinel_ml.security.encryption import EncryptionAlgorithm

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_log_with_pii() -> str:
    """Sample log containing various PII types."""
    return (
        "User john.doe@example.com logged in from 192.168.1.100. "
        "Contact: 555-123-4567. SSN: 123-45-6789. "
        "Card: 4111-1111-1111-1111. API key: sk-test0000000000000000000"  # noqa: S105
    )


@pytest.fixture
def sample_log_without_pii() -> str:
    """Sample log without PII."""
    return "Application started successfully. Server listening on port 8080."


# ==============================================================================
# PIIType Tests
# ==============================================================================


class TestPIIType:
    """Tests for PIIType enum."""

    def test_all_pii_types_exist(self) -> None:
        """Test all expected PII types are defined."""
        expected_types = [
            "EMAIL",
            "PHONE",
            "SSN",
            "CREDIT_CARD",
            "IP_ADDRESS",
            "API_KEY",
            "PASSWORD",
            "USERNAME",
            "NAME",
            "ADDRESS",
            "DATE_OF_BIRTH",
            "CUSTOM",
        ]
        for type_name in expected_types:
            assert hasattr(PIIType, type_name)

    def test_pii_type_values(self) -> None:
        """Test PII type values are lowercase strings."""
        assert PIIType.EMAIL.value == "email"
        assert PIIType.CREDIT_CARD.value == "credit_card"


# ==============================================================================
# RedactionLevel Tests
# ==============================================================================


class TestRedactionLevel:
    """Tests for RedactionLevel enum."""

    def test_redaction_levels(self) -> None:
        """Test all redaction levels exist."""
        assert RedactionLevel.NONE is not None
        assert RedactionLevel.MINIMAL is not None
        assert RedactionLevel.STANDARD is not None
        assert RedactionLevel.STRICT is not None
        assert RedactionLevel.PARANOID is not None


# ==============================================================================
# CustomPattern Tests
# ==============================================================================


class TestCustomPattern:
    """Tests for CustomPattern dataclass."""

    def test_create_custom_pattern(self) -> None:
        """Test creating a custom pattern."""
        pattern = CustomPattern(
            name="employee_id",
            pattern=r"\bEMP\d{6}\b",
            pii_type=PIIType.CUSTOM,
            description="Employee ID",
        )
        assert pattern.name == "employee_id"
        assert pattern.pii_type == PIIType.CUSTOM

    def test_custom_pattern_to_dict(self) -> None:
        """Test converting custom pattern to dict."""
        pattern = CustomPattern(
            name="test",
            pattern=r"\d+",
        )
        data = pattern.to_dict()
        assert data["name"] == "test"
        assert data["pattern"] == r"\d+"


# ==============================================================================
# RedactionResult Tests
# ==============================================================================


class TestRedactionResult:
    """Tests for RedactionResult dataclass."""

    def test_create_redaction_result(self) -> None:
        """Test creating a redaction result."""
        result = RedactionResult(
            original_text="test@example.com",
            redacted_text="[REDACTED:email]",
            redactions=({"type": "email", "start": 0, "end": 16},),
            pii_types_found=frozenset({PIIType.EMAIL}),
            redaction_count=1,
        )
        assert result.redaction_count == 1
        assert PIIType.EMAIL in result.pii_types_found

    def test_redaction_result_to_dict(self) -> None:
        """Test converting result to dict."""
        result = RedactionResult(
            original_text="test",
            redacted_text="test",
            redactions=(),
            pii_types_found=frozenset(),
            redaction_count=0,
        )
        data = result.to_dict()
        assert data["original_length"] == 4
        assert data["redaction_count"] == 0


# ==============================================================================
# RedactionStats Tests
# ==============================================================================


class TestRedactionStats:
    """Tests for RedactionStats class."""

    def test_create_stats(self) -> None:
        """Test creating redaction stats."""
        stats = RedactionStats()
        assert stats.total_processed == 0
        assert stats.total_redactions == 0

    def test_record_result(self) -> None:
        """Test recording a redaction result."""
        stats = RedactionStats()
        result = RedactionResult(
            original_text="test@example.com",
            redacted_text="[REDACTED]",
            redactions=({"type": "email"},),
            pii_types_found=frozenset({PIIType.EMAIL}),
            redaction_count=1,
        )
        stats.record(result, 1.5)
        assert stats.total_processed == 1
        assert stats.total_redactions == 1
        assert stats.redactions_by_type[PIIType.EMAIL] == 1


# ==============================================================================
# RedactionConfig Tests
# ==============================================================================


class TestRedactionConfig:
    """Tests for RedactionConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default redaction configuration."""
        config = RedactionConfig()
        assert config.level == RedactionLevel.STANDARD
        assert len(config.enabled_types) > 0

    def test_minimal_level_types(self) -> None:
        """Test minimal level enables only high-risk PII."""
        config = RedactionConfig(level=RedactionLevel.MINIMAL)
        assert PIIType.SSN in config.enabled_types
        assert PIIType.CREDIT_CARD in config.enabled_types

    def test_none_level_disables_all(self) -> None:
        """Test NONE level disables all redaction."""
        config = RedactionConfig(level=RedactionLevel.NONE)
        assert len(config.enabled_types) == 0

    def test_config_to_dict(self) -> None:
        """Test converting config to dict."""
        config = RedactionConfig(level=RedactionLevel.STRICT)
        data = config.to_dict()
        assert data["level"] == "strict"


# ==============================================================================
# RegexRedactor Tests
# ==============================================================================


class TestRegexRedactor:
    """Tests for RegexRedactor class."""

    def test_redact_email(self) -> None:
        """Test email redaction."""
        config = RedactionConfig(
            level=RedactionLevel.STANDARD,
            enabled_types={PIIType.EMAIL},
        )
        redactor = RegexRedactor(config)
        result = redactor.redact("Contact: user@example.com")
        assert "user@example.com" not in result.redacted_text
        assert PIIType.EMAIL in result.pii_types_found

    def test_redact_phone(self) -> None:
        """Test phone number redaction."""
        config = RedactionConfig(
            level=RedactionLevel.STANDARD,
            enabled_types={PIIType.PHONE},
        )
        redactor = RegexRedactor(config)
        result = redactor.redact("Call me at 555-123-4567")
        assert "555-123-4567" not in result.redacted_text
        assert PIIType.PHONE in result.pii_types_found

    def test_redact_ssn(self) -> None:
        """Test SSN redaction."""
        config = RedactionConfig(
            level=RedactionLevel.STANDARD,
            enabled_types={PIIType.SSN},
        )
        redactor = RegexRedactor(config)
        result = redactor.redact("SSN: 123-45-6789")
        assert "123-45-6789" not in result.redacted_text
        assert PIIType.SSN in result.pii_types_found

    def test_redact_credit_card(self) -> None:
        """Test credit card redaction."""
        config = RedactionConfig(
            level=RedactionLevel.STANDARD,
            enabled_types={PIIType.CREDIT_CARD},
        )
        redactor = RegexRedactor(config)
        result = redactor.redact("Card: 4111-1111-1111-1111")
        assert "4111-1111-1111-1111" not in result.redacted_text
        assert PIIType.CREDIT_CARD in result.pii_types_found

    def test_redact_ip_address(self) -> None:
        """Test IP address redaction."""
        config = RedactionConfig(
            level=RedactionLevel.STANDARD,
            enabled_types={PIIType.IP_ADDRESS},
        )
        redactor = RegexRedactor(config)
        result = redactor.redact("From IP: 192.168.1.100")
        assert "192.168.1.100" not in result.redacted_text
        assert PIIType.IP_ADDRESS in result.pii_types_found

    def test_redact_api_key(self) -> None:
        """Test API key redaction."""
        config = RedactionConfig(
            level=RedactionLevel.STANDARD,
            enabled_types={PIIType.API_KEY},
        )
        redactor = RegexRedactor(config)
        result = redactor.redact("token=abc123def456ghi789jkl012mno345")
        assert "abc123def456" not in result.redacted_text

    def test_redact_password(self) -> None:
        """Test password redaction."""
        config = RedactionConfig(
            level=RedactionLevel.STANDARD,
            enabled_types={PIIType.PASSWORD},
        )
        redactor = RegexRedactor(config)
        result = redactor.redact("password=mysecretpassword123")
        assert "mysecretpassword123" not in result.redacted_text
        assert PIIType.PASSWORD in result.pii_types_found

    def test_no_redaction_when_disabled(self) -> None:
        """Test no redaction when level is NONE."""
        config = RedactionConfig(level=RedactionLevel.NONE)
        redactor = RegexRedactor(config)
        original = "user@example.com"
        result = redactor.redact(original)
        assert result.redacted_text == original
        assert result.redaction_count == 0

    def test_hash_sensitive_replacement(self) -> None:
        """Test hash-based replacement."""
        config = RedactionConfig(
            level=RedactionLevel.STANDARD,
            enabled_types={PIIType.EMAIL},
            hash_sensitive=True,
        )
        redactor = RegexRedactor(config)
        result = redactor.redact("user@example.com")
        assert "[REDACTED:email]:" in result.redacted_text

    def test_preserve_format_email(self) -> None:
        """Test format-preserving redaction for email."""
        config = RedactionConfig(
            level=RedactionLevel.STANDARD,
            enabled_types={PIIType.EMAIL},
            preserve_format=True,
        )
        redactor = RegexRedactor(config)
        result = redactor.redact("user@example.com")
        assert "@" in result.redacted_text
        assert "*" in result.redacted_text

    def test_custom_patterns(self) -> None:
        """Test custom pattern redaction."""
        custom = CustomPattern(
            name="employee_id",
            pattern=r"\bEMP\d{6}\b",
            pii_type=PIIType.CUSTOM,
        )
        config = RedactionConfig(
            level=RedactionLevel.STANDARD,
            custom_patterns=[custom],
        )
        redactor = RegexRedactor(config)
        result = redactor.redact("Employee: EMP123456")
        assert "EMP123456" not in result.redacted_text

    def test_multiple_pii_types(self, sample_log_with_pii: str) -> None:
        """Test redacting multiple PII types."""
        config = RedactionConfig(level=RedactionLevel.STRICT)
        redactor = RegexRedactor(config)
        result = redactor.redact(sample_log_with_pii)
        assert "john.doe@example.com" not in result.redacted_text
        assert "555-123-4567" not in result.redacted_text
        assert "123-45-6789" not in result.redacted_text
        assert result.redaction_count > 0

    def test_get_pii_types(self) -> None:
        """Test getting handled PII types."""
        config = RedactionConfig(
            level=RedactionLevel.STANDARD,
            enabled_types={PIIType.EMAIL, PIIType.PHONE},
        )
        redactor = RegexRedactor(config)
        types = redactor.get_pii_types()
        assert PIIType.EMAIL in types
        assert PIIType.PHONE in types


# ==============================================================================
# CompositeRedactor Tests
# ==============================================================================


class TestCompositeRedactor:
    """Tests for CompositeRedactor class."""

    def test_composite_redaction(self) -> None:
        """Test combining multiple redactors."""
        config1 = RedactionConfig(
            level=RedactionLevel.MINIMAL,
            enabled_types={PIIType.EMAIL},
        )
        config2 = RedactionConfig(
            level=RedactionLevel.MINIMAL,
            enabled_types={PIIType.PHONE},
        )
        redactor1 = RegexRedactor(config1)
        redactor2 = RegexRedactor(config2)

        composite = CompositeRedactor([redactor1, redactor2])
        result = composite.redact("Email: test@test.com Phone: 555-123-4567")

        assert "test@test.com" not in result.redacted_text
        assert "555-123-4567" not in result.redacted_text

    def test_composite_get_pii_types(self) -> None:
        """Test getting all PII types from composite."""
        config1 = RedactionConfig(enabled_types={PIIType.EMAIL})
        config2 = RedactionConfig(enabled_types={PIIType.PHONE})

        composite = CompositeRedactor(
            [
                RegexRedactor(config1),
                RegexRedactor(config2),
            ]
        )

        types = composite.get_pii_types()
        assert PIIType.EMAIL in types
        assert PIIType.PHONE in types


# ==============================================================================
# RedactorFactory Tests
# ==============================================================================


class TestRedactorFactory:
    """Tests for RedactorFactory class."""

    def test_create_default(self) -> None:
        """Test creating default redactor."""
        redactor = RedactorFactory.create_default()
        assert isinstance(redactor, Redactor)

    def test_create_with_config(self) -> None:
        """Test creating redactor with config."""
        config = RedactionConfig(level=RedactionLevel.STRICT)
        redactor = RedactorFactory.create(config)
        assert isinstance(redactor, RegexRedactor)

    def test_create_composite(self) -> None:
        """Test creating composite redactor."""
        configs = [
            RedactionConfig(enabled_types={PIIType.EMAIL}),
            RedactionConfig(enabled_types={PIIType.PHONE}),
        ]
        composite = RedactorFactory.create_composite(configs)
        assert isinstance(composite, CompositeRedactor)


# ==============================================================================
# PrivacyLevel Tests
# ==============================================================================


class TestPrivacyLevel:
    """Tests for PrivacyLevel enum."""

    def test_privacy_levels(self) -> None:
        """Test all privacy levels exist."""
        assert PrivacyLevel.BASIC is not None
        assert PrivacyLevel.ENHANCED is not None
        assert PrivacyLevel.MAXIMUM is not None


# ==============================================================================
# PrivacyMode Tests
# ==============================================================================


class TestPrivacyMode:
    """Tests for PrivacyMode enum."""

    def test_privacy_modes(self) -> None:
        """Test all privacy modes exist."""
        assert PrivacyMode.STORE_ALL is not None
        assert PrivacyMode.STORE_REDACTED is not None
        assert PrivacyMode.STORE_EMBEDDINGS_ONLY is not None
        assert PrivacyMode.NEVER_STORE is not None


# ==============================================================================
# RawLogPolicy Tests
# ==============================================================================


class TestRawLogPolicy:
    """Tests for RawLogPolicy enum."""

    def test_raw_log_policies(self) -> None:
        """Test all raw log policies exist."""
        assert RawLogPolicy.ALLOW is not None
        assert RawLogPolicy.REDACT_THEN_STORE is not None
        assert RawLogPolicy.HASH_ONLY is not None
        assert RawLogPolicy.DISCARD is not None


# ==============================================================================
# SanitizedLog Tests
# ==============================================================================


class TestSanitizedLog:
    """Tests for SanitizedLog dataclass."""

    def test_create_sanitized_log(self) -> None:
        """Test creating a sanitized log."""
        log = SanitizedLog(
            original_hash="abc123",
            sanitized_text="Sanitized content",
            timestamp=datetime.now(tz=timezone.utc),
            metadata={"source": "test"},
            pii_detected=True,
            pii_types=frozenset({PIIType.EMAIL}),
        )
        assert log.pii_detected is True
        assert PIIType.EMAIL in log.pii_types

    def test_sanitized_log_to_dict(self) -> None:
        """Test converting sanitized log to dict."""
        log = SanitizedLog(
            original_hash="abc123",
            sanitized_text="test",
            timestamp=datetime.now(tz=timezone.utc),
            metadata={},
            pii_detected=False,
            pii_types=frozenset(),
        )
        data = log.to_dict()
        assert data["original_hash"] == "abc123"


# ==============================================================================
# PrivacyConfig Tests
# ==============================================================================


class TestPrivacyConfig:
    """Tests for PrivacyConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default privacy configuration."""
        config = PrivacyConfig()
        assert config.level == PrivacyLevel.ENHANCED
        assert config.mode == PrivacyMode.STORE_REDACTED

    def test_maximum_privacy_defaults(self) -> None:
        """Test MAXIMUM level applies strict defaults."""
        config = PrivacyConfig(level=PrivacyLevel.MAXIMUM)
        assert config.mode == PrivacyMode.NEVER_STORE
        assert config.raw_log_policy == RawLogPolicy.DISCARD
        assert config.encrypt_at_rest is True

    def test_config_to_dict(self) -> None:
        """Test converting config to dict."""
        config = PrivacyConfig()
        data = config.to_dict()
        assert data["level"] == "enhanced"

    def test_config_from_dict(self) -> None:
        """Test creating config from dict."""
        data = {"level": "basic", "mode": "store_all"}
        config = PrivacyConfig.from_dict(data)
        assert config.level == PrivacyLevel.BASIC


# ==============================================================================
# PrivacyManager Tests
# ==============================================================================


class TestPrivacyManager:
    """Tests for PrivacyManager class."""

    def test_sanitize_with_pii(self, sample_log_with_pii: str) -> None:
        """Test sanitizing log with PII."""
        config = PrivacyConfig(level=PrivacyLevel.ENHANCED)
        manager = PrivacyManager(config)
        sanitized = manager.sanitize(sample_log_with_pii)
        assert sanitized.pii_detected is True
        assert "john.doe@example.com" not in sanitized.sanitized_text

    def test_sanitize_without_pii(self, sample_log_without_pii: str) -> None:
        """Test sanitizing log without PII."""
        config = PrivacyConfig(level=PrivacyLevel.ENHANCED)
        manager = PrivacyManager(config)
        sanitized = manager.sanitize(sample_log_without_pii)
        assert sanitized.pii_detected is False

    def test_sanitize_with_metadata(self) -> None:
        """Test sanitizing with metadata preservation."""
        config = PrivacyConfig(retain_metadata=True)
        manager = PrivacyManager(config)
        metadata = {"source": "test", "level": "INFO"}
        sanitized = manager.sanitize("Test log", metadata)
        assert sanitized.metadata == metadata

    def test_sanitize_discard_policy(self) -> None:
        """Test DISCARD policy returns empty text."""
        config = PrivacyConfig(raw_log_policy=RawLogPolicy.DISCARD)
        manager = PrivacyManager(config)
        sanitized = manager.sanitize("Sensitive data")
        assert sanitized.sanitized_text == ""

    def test_sanitize_hash_only_policy(self) -> None:
        """Test HASH_ONLY policy returns hash."""
        config = PrivacyConfig(raw_log_policy=RawLogPolicy.HASH_ONLY)
        manager = PrivacyManager(config)
        sanitized = manager.sanitize("Sensitive data")
        assert "[HASH:" in sanitized.sanitized_text

    def test_sanitize_batch(self) -> None:
        """Test batch sanitization."""
        config = PrivacyConfig()
        manager = PrivacyManager(config)
        logs = [
            ("User test@test.com logged in", {"level": "INFO"}),
            ("Application started", None),
        ]
        result = manager.sanitize_batch(logs)
        assert result.total_processed == 2
        assert len(result.sanitized_logs) == 2

    def test_can_store_never_store_mode(self) -> None:
        """Test storage check with NEVER_STORE mode."""
        config = PrivacyConfig(mode=PrivacyMode.NEVER_STORE)
        manager = PrivacyManager(config)
        sanitized = manager.sanitize("Test")
        assert manager.can_store(sanitized) is False

    def test_can_store_redacted_mode(self) -> None:
        """Test storage check with STORE_REDACTED mode."""
        config = PrivacyConfig(mode=PrivacyMode.STORE_REDACTED)
        manager = PrivacyManager(config)
        sanitized = manager.sanitize("Test")
        assert manager.can_store(sanitized) is True

    def test_generate_report(self, sample_log_with_pii: str) -> None:
        """Test generating privacy report."""
        config = PrivacyConfig()
        manager = PrivacyManager(config)
        manager.sanitize(sample_log_with_pii)
        report = manager.generate_report()
        assert report.total_logs_processed == 1

    def test_get_stats(self) -> None:
        """Test getting privacy statistics."""
        config = PrivacyConfig()
        manager = PrivacyManager(config)
        stats = manager.get_stats()
        assert "total_processed" in stats

    def test_reset_stats(self, sample_log_with_pii: str) -> None:
        """Test resetting privacy statistics."""
        config = PrivacyConfig()
        manager = PrivacyManager(config)
        manager.sanitize(sample_log_with_pii)
        manager.reset_stats()
        stats = manager.get_stats()
        assert stats["total_processed"] == 0


# ==============================================================================
# EncryptionAlgorithm Tests
# ==============================================================================


class TestEncryptionAlgorithm:
    """Tests for EncryptionAlgorithm enum."""

    def test_algorithms(self) -> None:
        """Test encryption algorithms exist."""
        assert EncryptionAlgorithm.FERNET is not None
        assert EncryptionAlgorithm.AES_GCM is not None


# ==============================================================================
# EncryptionKey Tests
# ==============================================================================


class TestEncryptionKey:
    """Tests for EncryptionKey dataclass."""

    def test_create_key(self) -> None:
        """Test creating an encryption key."""
        key = EncryptionKey(
            key_id="test-key-123",
            key_data="dGVzdGtleWRhdGE=",
            algorithm=EncryptionAlgorithm.FERNET,
            created_at=datetime.now(tz=timezone.utc),
        )
        assert key.key_id == "test-key-123"

    def test_key_not_expired(self) -> None:
        """Test key expiration check (not expired)."""
        key = EncryptionKey(
            key_id="test",
            key_data="data",
            algorithm=EncryptionAlgorithm.FERNET,
            created_at=datetime.now(tz=timezone.utc),
            expires_at=None,
        )
        assert key.is_expired() is False

    def test_key_to_dict(self) -> None:
        """Test converting key to dict (excluding sensitive data)."""
        key = EncryptionKey(
            key_id="test",
            key_data="sensitive",
            algorithm=EncryptionAlgorithm.FERNET,
            created_at=datetime.now(tz=timezone.utc),
        )
        data = key.to_dict()
        assert "key_data" not in data
        assert data["key_id"] == "test"


# ==============================================================================
# EncryptedData Tests
# ==============================================================================


class TestEncryptedData:
    """Tests for EncryptedData dataclass."""

    def test_create_encrypted_data(self) -> None:
        """Test creating encrypted data."""
        encrypted = EncryptedData(
            ciphertext="encrypted_content",
            key_id="key-123",
            algorithm=EncryptionAlgorithm.FERNET,
        )
        assert encrypted.ciphertext == "encrypted_content"

    def test_encrypted_data_to_dict(self) -> None:
        """Test converting encrypted data to dict."""
        encrypted = EncryptedData(
            ciphertext="test",
            key_id="key-123",
            algorithm=EncryptionAlgorithm.FERNET,
        )
        data = encrypted.to_dict()
        assert data["ciphertext"] == "test"
        assert data["algorithm"] == "fernet"

    def test_encrypted_data_from_dict(self) -> None:
        """Test creating encrypted data from dict."""
        data = {
            "ciphertext": "test",
            "key_id": "key-123",
            "algorithm": "fernet",
            "nonce": None,
            "encrypted_at": "2024-01-01T00:00:00+00:00",
        }
        encrypted = EncryptedData.from_dict(data)
        assert encrypted.ciphertext == "test"


# ==============================================================================
# KeyDerivationConfig Tests
# ==============================================================================


class TestKeyDerivationConfig:
    """Tests for KeyDerivationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default key derivation config."""
        config = KeyDerivationConfig()
        assert config.iterations == 600000
        assert config.key_length == 32

    def test_config_to_dict(self) -> None:
        """Test converting config to dict."""
        config = KeyDerivationConfig()
        data = config.to_dict()
        assert data["iterations"] == 600000


# ==============================================================================
# EncryptionConfig Tests
# ==============================================================================


class TestEncryptionConfig:
    """Tests for EncryptionConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default encryption config."""
        config = EncryptionConfig()
        assert config.algorithm == EncryptionAlgorithm.FERNET

    def test_config_to_dict(self) -> None:
        """Test converting config to dict."""
        config = EncryptionConfig()
        data = config.to_dict()
        assert data["algorithm"] == "fernet"


# ==============================================================================
# FernetEncryptionProvider Tests
# ==============================================================================


class TestFernetEncryptionProvider:
    """Tests for FernetEncryptionProvider class."""

    @pytest.fixture
    def provider(self) -> FernetEncryptionProvider:
        """Create a Fernet provider."""
        return FernetEncryptionProvider()

    def test_generate_key(self, provider: FernetEncryptionProvider) -> None:
        """Test generating a Fernet key."""
        try:
            key = provider.generate_key()
            assert key.algorithm == EncryptionAlgorithm.FERNET
            assert len(key.key_id) > 0
        except EncryptionError:
            pytest.skip("cryptography package not installed")

    def test_encrypt_decrypt(self, provider: FernetEncryptionProvider) -> None:
        """Test encrypt and decrypt cycle."""
        try:
            key = provider.generate_key()
            plaintext = b"Hello, World!"
            encrypted = provider.encrypt(plaintext, key)
            decrypted = provider.decrypt(encrypted, key)
            assert decrypted == plaintext
        except EncryptionError:
            pytest.skip("cryptography package not installed")

    def test_decrypt_wrong_key_fails(self, provider: FernetEncryptionProvider) -> None:
        """Test decryption with wrong key fails."""
        try:
            key1 = provider.generate_key()
            key2 = provider.generate_key()
            plaintext = b"Secret data"
            encrypted = provider.encrypt(plaintext, key1)

            with pytest.raises(DecryptionError):
                provider.decrypt(encrypted, key2)
        except EncryptionError:
            pytest.skip("cryptography package not installed")


# ==============================================================================
# KeyManager Tests
# ==============================================================================


class TestKeyManager:
    """Tests for KeyManager class."""

    @pytest.fixture
    def key_manager(self, temp_dir: Path) -> KeyManager:
        """Create a key manager."""
        config = EncryptionConfig(store_path=temp_dir / "keys")
        return KeyManager(config)

    def test_generate_key(self, key_manager: KeyManager) -> None:
        """Test generating a key."""
        try:
            key = key_manager.generate_key()
            assert key is not None
        except EncryptionError:
            pytest.skip("cryptography package not installed")

    def test_get_active_key(self, key_manager: KeyManager) -> None:
        """Test getting active key."""
        try:
            key = key_manager.get_active_key()
            assert key is not None
        except EncryptionError:
            pytest.skip("cryptography package not installed")

    def test_rotate_key(self, key_manager: KeyManager) -> None:
        """Test key rotation."""
        try:
            old_key = key_manager.get_active_key()
            new_key = key_manager.rotate_key()
            assert old_key.key_id != new_key.key_id
        except EncryptionError:
            pytest.skip("cryptography package not installed")

    def test_get_stats(self, key_manager: KeyManager) -> None:
        """Test getting key manager stats."""
        stats = key_manager.get_stats()
        assert "total_keys" in stats


# ==============================================================================
# EncryptedStore Tests
# ==============================================================================


class TestEncryptedStore:
    """Tests for EncryptedStore class."""

    @pytest.fixture
    def encrypted_store(self, temp_dir: Path) -> EncryptedStore:
        """Create an encrypted store."""
        config = EncryptionConfig(store_path=temp_dir / "keys")
        key_manager = KeyManager(config)
        return EncryptedStore(key_manager)

    def test_encrypt_string(self, encrypted_store: EncryptedStore) -> None:
        """Test encrypting a string."""
        try:
            encrypted = encrypted_store.encrypt("Hello, World!")
            assert encrypted.ciphertext is not None
        except EncryptionError:
            pytest.skip("cryptography package not installed")

    def test_encrypt_dict(self, encrypted_store: EncryptedStore) -> None:
        """Test encrypting a dictionary."""
        try:
            data = {"key": "value", "number": 42}
            encrypted = encrypted_store.encrypt(data)
            assert encrypted.ciphertext is not None
        except EncryptionError:
            pytest.skip("cryptography package not installed")

    def test_decrypt_to_string(self, encrypted_store: EncryptedStore) -> None:
        """Test decrypting to string."""
        try:
            original = "Secret message"
            encrypted = encrypted_store.encrypt(original)
            decrypted = encrypted_store.decrypt_to_string(encrypted)
            assert decrypted == original
        except EncryptionError:
            pytest.skip("cryptography package not installed")

    def test_decrypt_to_dict(self, encrypted_store: EncryptedStore) -> None:
        """Test decrypting to dictionary."""
        try:
            original = {"key": "value"}
            encrypted = encrypted_store.encrypt(original)
            decrypted = encrypted_store.decrypt_to_dict(encrypted)
            assert decrypted == original
        except EncryptionError:
            pytest.skip("cryptography package not installed")

    def test_get_stats(self, encrypted_store: EncryptedStore) -> None:
        """Test getting store stats."""
        stats = encrypted_store.get_stats()
        assert "encryptions" in stats
        assert "decryptions" in stats


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestSecurityIntegration:
    """Integration tests for security module."""

    def test_full_privacy_pipeline(self, sample_log_with_pii: str) -> None:
        """Test full privacy pipeline from raw log to sanitized output."""
        config = PrivacyConfig(
            level=PrivacyLevel.ENHANCED,
            mode=PrivacyMode.STORE_REDACTED,
        )
        manager = PrivacyManager(config)

        sanitized = manager.sanitize(sample_log_with_pii, {"source": "test"})

        assert sanitized.pii_detected is True
        assert "john.doe@example.com" not in sanitized.sanitized_text
        assert manager.can_store(sanitized) is True

        report = manager.generate_report()
        assert report.total_logs_processed == 1

    def test_maximum_privacy_mode(self, sample_log_with_pii: str) -> None:
        """Test maximum privacy mode discards all data."""
        config = PrivacyConfig(level=PrivacyLevel.MAXIMUM)
        manager = PrivacyManager(config)

        sanitized = manager.sanitize(sample_log_with_pii)

        assert sanitized.sanitized_text == ""
        assert manager.can_store(sanitized) is False

    def test_encrypted_storage_pipeline(self, temp_dir: Path) -> None:
        """Test encrypted storage pipeline."""
        try:
            encryption_config = EncryptionConfig(store_path=temp_dir / "keys")
            key_manager = KeyManager(encryption_config)
            store = EncryptedStore(key_manager)

            privacy_config = PrivacyConfig(encrypt_at_rest=True)
            privacy_manager = PrivacyManager(privacy_config)

            sanitized = privacy_manager.sanitize("User test@test.com logged in")

            encrypted = store.encrypt(sanitized.to_dict())

            decrypted = store.decrypt_to_dict(encrypted)
            assert decrypted["sanitized_text"] == sanitized.sanitized_text

        except EncryptionError:
            pytest.skip("cryptography package not installed")
