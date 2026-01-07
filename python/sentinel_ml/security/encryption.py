"""
Encryption Module for Sentinel Log AI.

This module provides at-rest encryption for sensitive data using
industry-standard cryptographic algorithms.

Design Patterns:
- Strategy Pattern: Pluggable encryption providers
- Factory Pattern: Key generation and management
- Decorator Pattern: Encryption wrappers for storage
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class EncryptionError(Exception):
    """Base exception for encryption errors."""

    pass


class DecryptionError(EncryptionError):
    """Exception raised when decryption fails."""

    pass


class KeyDerivationError(EncryptionError):
    """Exception raised when key derivation fails."""

    pass


class EncryptionAlgorithm(Enum):
    """
    Supported encryption algorithms.

    Attributes:
        FERNET: Fernet symmetric encryption (AES-128-CBC + HMAC).
        AES_GCM: AES-256 in GCM mode (authenticated encryption).
    """

    FERNET = "fernet"
    AES_GCM = "aes-gcm"


@dataclass(frozen=True)
class EncryptionKey:
    """
    An encryption key with metadata.

    Attributes:
        key_id: Unique key identifier.
        key_data: The actual key bytes (base64 encoded for storage).
        algorithm: Encryption algorithm.
        created_at: Key creation timestamp.
        expires_at: Key expiration timestamp (optional).
        version: Key version for rotation.
    """

    key_id: str
    key_data: str
    algorithm: EncryptionAlgorithm
    created_at: datetime
    expires_at: datetime | None = None
    version: int = 1

    def is_expired(self) -> bool:
        """Check if the key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(tz=timezone.utc) > self.expires_at

    def get_key_bytes(self) -> bytes:
        """Get the raw key bytes."""
        return base64.urlsafe_b64decode(self.key_data.encode())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding sensitive key data)."""
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "version": self.version,
        }


@dataclass(frozen=True)
class EncryptedData:
    """
    Encrypted data with metadata.

    Attributes:
        ciphertext: The encrypted data (base64 encoded).
        key_id: ID of the key used for encryption.
        algorithm: Encryption algorithm used.
        nonce: Nonce/IV used (base64 encoded, for AES-GCM).
        encrypted_at: Encryption timestamp.
    """

    ciphertext: str
    key_id: str
    algorithm: EncryptionAlgorithm
    nonce: str | None = None
    encrypted_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "ciphertext": self.ciphertext,
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "nonce": self.nonce,
            "encrypted_at": self.encrypted_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncryptedData:
        """Create from dictionary representation."""
        return cls(
            ciphertext=data["ciphertext"],
            key_id=data["key_id"],
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            nonce=data.get("nonce"),
            encrypted_at=datetime.fromisoformat(data["encrypted_at"]),
        )


@dataclass
class KeyDerivationConfig:
    """
    Configuration for key derivation.

    Attributes:
        iterations: Number of PBKDF2 iterations.
        salt_length: Length of salt in bytes.
        key_length: Derived key length in bytes.
        hash_algorithm: Hash algorithm for PBKDF2.
    """

    iterations: int = 600000
    salt_length: int = 32
    key_length: int = 32
    hash_algorithm: str = "sha256"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "iterations": self.iterations,
            "salt_length": self.salt_length,
            "key_length": self.key_length,
            "hash_algorithm": self.hash_algorithm,
        }


@dataclass
class EncryptionConfig:
    """
    Configuration for encryption.

    Attributes:
        algorithm: Encryption algorithm to use.
        key_rotation_days: Days between key rotations.
        key_derivation: Key derivation configuration.
        store_path: Path to store encrypted keys.
    """

    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET
    key_rotation_days: int = 90
    key_derivation: KeyDerivationConfig = field(default_factory=KeyDerivationConfig)
    store_path: Path = field(default_factory=lambda: Path(".data/keys"))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "algorithm": self.algorithm.value,
            "key_rotation_days": self.key_rotation_days,
            "key_derivation": self.key_derivation.to_dict(),
            "store_path": str(self.store_path),
        }


class EncryptionProvider(ABC):
    """
    Abstract base class for encryption providers.

    All encryption implementations must inherit from this class.
    """

    @abstractmethod
    def encrypt(self, plaintext: bytes, key: EncryptionKey) -> EncryptedData:
        """
        Encrypt data.

        Args:
            plaintext: Data to encrypt.
            key: Encryption key to use.

        Returns:
            EncryptedData with ciphertext and metadata.

        Raises:
            EncryptionError: If encryption fails.
        """
        pass

    @abstractmethod
    def decrypt(self, encrypted: EncryptedData, key: EncryptionKey) -> bytes:
        """
        Decrypt data.

        Args:
            encrypted: Encrypted data.
            key: Encryption key to use.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            DecryptionError: If decryption fails.
        """
        pass

    @abstractmethod
    def generate_key(self) -> EncryptionKey:
        """
        Generate a new encryption key.

        Returns:
            New EncryptionKey.
        """
        pass


class FernetEncryptionProvider(EncryptionProvider):
    """
    Fernet-based encryption provider.

    Uses the cryptography library's Fernet implementation
    for symmetric authenticated encryption.
    """

    def __init__(self) -> None:
        """Initialize the Fernet encryption provider."""
        try:
            from cryptography.fernet import Fernet

            self._fernet_class = Fernet
        except ImportError:
            self._fernet_class = None
            logger.warning(
                "cryptography_not_installed",
                message="Install cryptography package for encryption support",
            )

        logger.info("fernet_encryption_provider_initialized")

    def _ensure_cryptography(self) -> None:
        """Ensure cryptography package is available."""
        if self._fernet_class is None:
            raise EncryptionError(
                "cryptography package not installed. Run: pip install cryptography"
            )

    def encrypt(self, plaintext: bytes, key: EncryptionKey) -> EncryptedData:
        """
        Encrypt data using Fernet.

        Args:
            plaintext: Data to encrypt.
            key: Encryption key to use.

        Returns:
            EncryptedData with ciphertext.

        Raises:
            EncryptionError: If encryption fails.
        """
        self._ensure_cryptography()

        if key.algorithm != EncryptionAlgorithm.FERNET:
            raise EncryptionError(
                f"Key algorithm mismatch: expected FERNET, got {key.algorithm.value}"
            )

        try:
            fernet = self._fernet_class(key.get_key_bytes())
            ciphertext = fernet.encrypt(plaintext)

            encrypted = EncryptedData(
                ciphertext=base64.urlsafe_b64encode(ciphertext).decode(),
                key_id=key.key_id,
                algorithm=EncryptionAlgorithm.FERNET,
            )

            logger.debug(
                "data_encrypted",
                key_id=key.key_id,
                plaintext_length=len(plaintext),
                ciphertext_length=len(ciphertext),
            )

            return encrypted

        except Exception as e:
            logger.error("encryption_failed", error=str(e))
            raise EncryptionError(f"Encryption failed: {e}") from e

    def decrypt(self, encrypted: EncryptedData, key: EncryptionKey) -> bytes:
        """
        Decrypt data using Fernet.

        Args:
            encrypted: Encrypted data.
            key: Encryption key to use.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            DecryptionError: If decryption fails.
        """
        self._ensure_cryptography()

        if key.key_id != encrypted.key_id:
            raise DecryptionError(f"Key ID mismatch: expected {encrypted.key_id}, got {key.key_id}")

        if key.is_expired():
            logger.warning("decrypting_with_expired_key", key_id=key.key_id)

        try:
            fernet = self._fernet_class(key.get_key_bytes())
            ciphertext = base64.urlsafe_b64decode(encrypted.ciphertext.encode())
            plaintext = fernet.decrypt(ciphertext)

            logger.debug(
                "data_decrypted",
                key_id=key.key_id,
                ciphertext_length=len(ciphertext),
                plaintext_length=len(plaintext),
            )

            return plaintext

        except Exception as e:
            logger.error("decryption_failed", error=str(e))
            raise DecryptionError(f"Decryption failed: {e}") from e

    def generate_key(self) -> EncryptionKey:
        """
        Generate a new Fernet key.

        Returns:
            New EncryptionKey for Fernet encryption.
        """
        self._ensure_cryptography()

        from cryptography.fernet import Fernet

        key_bytes = Fernet.generate_key()
        key_id = secrets.token_hex(16)

        key = EncryptionKey(
            key_id=key_id,
            key_data=key_bytes.decode(),
            algorithm=EncryptionAlgorithm.FERNET,
            created_at=datetime.now(tz=timezone.utc),
        )

        logger.info("encryption_key_generated", key_id=key_id)

        return key


class KeyManager:
    """
    Manages encryption keys.

    Handles key generation, storage, rotation, and retrieval.
    """

    def __init__(self, config: EncryptionConfig) -> None:
        """
        Initialize the key manager.

        Args:
            config: Encryption configuration.
        """
        self.config = config
        self._keys: dict[str, EncryptionKey] = {}
        self._active_key_id: str | None = None
        self._provider = self._create_provider()

        config.store_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "key_manager_initialized",
            algorithm=config.algorithm.value,
            store_path=str(config.store_path),
        )

    def _create_provider(self) -> EncryptionProvider:
        """Create encryption provider based on config."""
        if self.config.algorithm == EncryptionAlgorithm.FERNET:
            return FernetEncryptionProvider()

        raise EncryptionError(f"Unsupported algorithm: {self.config.algorithm.value}")

    def generate_key(self) -> EncryptionKey:
        """
        Generate and store a new key.

        Returns:
            Newly generated EncryptionKey.
        """
        key = self._provider.generate_key()
        self._keys[key.key_id] = key
        self._active_key_id = key.key_id

        logger.info(
            "new_key_generated_and_stored",
            key_id=key.key_id,
            is_active=True,
        )

        return key

    def get_active_key(self) -> EncryptionKey:
        """
        Get the currently active encryption key.

        Returns:
            Active EncryptionKey.

        Raises:
            EncryptionError: If no active key exists.
        """
        if self._active_key_id is None:
            key = self.generate_key()
            return key

        key = self._keys.get(self._active_key_id)
        if key is None:
            raise EncryptionError(f"Active key not found: {self._active_key_id}")

        if key.is_expired():
            logger.warning("active_key_expired", key_id=key.key_id)
            key = self.rotate_key()

        return key

    def get_key(self, key_id: str) -> EncryptionKey:
        """
        Get a key by ID.

        Args:
            key_id: Key identifier.

        Returns:
            EncryptionKey with matching ID.

        Raises:
            EncryptionError: If key not found.
        """
        key = self._keys.get(key_id)
        if key is None:
            raise EncryptionError(f"Key not found: {key_id}")
        return key

    def rotate_key(self) -> EncryptionKey:
        """
        Rotate to a new active key.

        Returns:
            New active EncryptionKey.
        """
        old_key_id = self._active_key_id
        new_key = self.generate_key()

        logger.info(
            "key_rotated",
            old_key_id=old_key_id,
            new_key_id=new_key.key_id,
        )

        return new_key

    def derive_key_from_password(
        self, password: str, salt: bytes | None = None
    ) -> tuple[EncryptionKey, bytes]:
        """
        Derive an encryption key from a password.

        Args:
            password: Password to derive key from.
            salt: Optional salt (generated if not provided).

        Returns:
            Tuple of (derived EncryptionKey, salt used).
        """
        if salt is None:
            salt = os.urandom(self.config.key_derivation.salt_length)

        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.config.key_derivation.key_length,
                salt=salt,
                iterations=self.config.key_derivation.iterations,
            )

            key_bytes = kdf.derive(password.encode())

            if self.config.algorithm == EncryptionAlgorithm.FERNET:
                fernet_key = base64.urlsafe_b64encode(key_bytes)
                key_data = fernet_key.decode()
            else:
                key_data = base64.urlsafe_b64encode(key_bytes).decode()

            key_id = hashlib.sha256(salt + password.encode()).hexdigest()[:32]

            key = EncryptionKey(
                key_id=key_id,
                key_data=key_data,
                algorithm=self.config.algorithm,
                created_at=datetime.now(tz=timezone.utc),
            )

            logger.info(
                "key_derived_from_password",
                key_id=key_id,
                iterations=self.config.key_derivation.iterations,
            )

            return key, salt

        except ImportError as e:
            raise EncryptionError("cryptography package required for key derivation") from e

    def export_keys_metadata(self) -> list[dict[str, Any]]:
        """
        Export key metadata (not the actual keys).

        Returns:
            List of key metadata dictionaries.
        """
        return [key.to_dict() for key in self._keys.values()]

    def get_stats(self) -> dict[str, Any]:
        """Get key manager statistics."""
        return {
            "total_keys": len(self._keys),
            "active_key_id": self._active_key_id,
            "algorithm": self.config.algorithm.value,
            "expired_keys": sum(1 for k in self._keys.values() if k.is_expired()),
        }


class EncryptedStore:
    """
    Encrypted data store wrapper.

    Provides transparent encryption/decryption for stored data.
    """

    def __init__(self, key_manager: KeyManager) -> None:
        """
        Initialize the encrypted store.

        Args:
            key_manager: Key manager for encryption keys.
        """
        self.key_manager = key_manager
        self._provider = key_manager._provider
        self._stats = {
            "encryptions": 0,
            "decryptions": 0,
            "bytes_encrypted": 0,
            "bytes_decrypted": 0,
        }

        logger.info("encrypted_store_initialized")

    def encrypt(self, data: str | bytes | dict[str, Any]) -> EncryptedData:
        """
        Encrypt data.

        Args:
            data: Data to encrypt (string, bytes, or dict).

        Returns:
            EncryptedData object.
        """
        if isinstance(data, dict):
            plaintext = json.dumps(data).encode()
        elif isinstance(data, str):
            plaintext = data.encode()
        else:
            plaintext = data

        key = self.key_manager.get_active_key()
        encrypted = self._provider.encrypt(plaintext, key)

        self._stats["encryptions"] += 1
        self._stats["bytes_encrypted"] += len(plaintext)

        return encrypted

    def decrypt(self, encrypted: EncryptedData) -> bytes:
        """
        Decrypt data.

        Args:
            encrypted: Encrypted data.

        Returns:
            Decrypted bytes.
        """
        key = self.key_manager.get_key(encrypted.key_id)
        plaintext = self._provider.decrypt(encrypted, key)

        self._stats["decryptions"] += 1
        self._stats["bytes_decrypted"] += len(plaintext)

        return plaintext

    def decrypt_to_string(self, encrypted: EncryptedData) -> str:
        """Decrypt data and return as string."""
        return self.decrypt(encrypted).decode()

    def decrypt_to_dict(self, encrypted: EncryptedData) -> dict[str, Any]:
        """Decrypt data and return as dictionary."""
        return json.loads(self.decrypt(encrypted))

    def get_stats(self) -> dict[str, Any]:
        """Get encryption statistics."""
        return {
            **self._stats,
            "key_manager_stats": self.key_manager.get_stats(),
        }
