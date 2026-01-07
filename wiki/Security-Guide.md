# Security Guide

This guide covers security features, privacy controls, and best practices for Sentinel Log AI.

## Overview

Sentinel Log AI includes comprehensive security features:

- **PII Redaction**: Detect and redact 12+ types of personally identifiable information
- **Privacy Modes**: Control how logs are stored (or not stored)
- **At-Rest Encryption**: Encrypt sensitive data using Fernet (AES-128-CBC)
- **Safe Defaults**: Privacy-first configuration out of the box

## Quick Start

### Basic PII Redaction

```python
from sentinel_ml.security import RedactorFactory, RedactionLevel

# Create a redactor with standard protection
redactor = RedactorFactory.create_default(level=RedactionLevel.STANDARD)

# Redact PII from log text
log = "User john@example.com logged in from 192.168.1.100"
result = redactor.redact(log)

print(result.redacted_text)
# Output: "User [EMAIL_REDACTED] logged in from [IP_REDACTED]"
```

### Privacy Manager

```python
from sentinel_ml.security import PrivacyManager, PrivacyConfig, PrivacyMode

config = PrivacyConfig(mode=PrivacyMode.STORE_REDACTED)
manager = PrivacyManager(config)

result = manager.sanitize("Error for user admin@corp.com: auth failed")
print(result.sanitized_log.content)
# Output: "Error for user [EMAIL_REDACTED]: auth failed"
```

### Encrypted Storage

```python
from sentinel_ml.security import EncryptedStore, KeyManager

key_manager = KeyManager()
store = EncryptedStore(key_manager=key_manager)

# Encrypt sensitive data
encrypted = store.encrypt_dict({"user": "admin", "action": "login"})

# Decrypt when needed
data = store.decrypt_to_dict(encrypted)
```

## PII Detection

### Supported PII Types

| Type | Description | Example Pattern |
|------|-------------|-----------------|
| EMAIL | Email addresses | user@example.com |
| PHONE | Phone numbers | +1-555-123-4567 |
| SSN | Social Security Numbers | 123-45-6789 |
| CREDIT_CARD | Credit card numbers | 4111-1111-1111-1111 |
| IP_ADDRESS | IPv4 and IPv6 addresses | 192.168.1.1 |
| MAC_ADDRESS | MAC addresses | 00:1A:2B:3C:4D:5E |
| API_KEY | API keys and tokens | sk-abc123... |
| PASSWORD | Password patterns | password=secret |
| SECRET | Generic secrets | secret_key=... |
| USERNAME | Usernames | username=admin |
| DOB | Dates of birth | DOB: 1990-01-15 |
| ADDRESS | Physical addresses | 123 Main St... |

### Redaction Levels

| Level | Description | Types Redacted |
|-------|-------------|----------------|
| NONE | No redaction | None |
| MINIMAL | Critical PII only | SSN, CREDIT_CARD |
| STANDARD | Common PII | EMAIL, PHONE, SSN, CREDIT_CARD, IP, API_KEY |
| STRICT | Extended PII | All except ADDRESS |
| PARANOID | Maximum protection | All PII types |

### Custom Patterns

```python
from sentinel_ml.security import (
    RegexRedactor,
    RedactionConfig,
    CustomPattern,
    PIIType,
)

# Define custom pattern
custom = CustomPattern(
    name="employee_id",
    pattern=r"EMP-\d{6}",
    replacement="[EMPLOYEE_ID]",
    pii_type=PIIType.OTHER,
)

config = RedactionConfig(custom_patterns=[custom])
redactor = RegexRedactor(config)

result = redactor.redact("Employee EMP-123456 logged in")
# Output: "Employee [EMPLOYEE_ID] logged in"
```

## Privacy Modes

### Mode Options

| Mode | Raw Logs | Embeddings | Use Case |
|------|----------|------------|----------|
| STORE_ALL | Stored | Stored | Development, debugging |
| STORE_REDACTED | Redacted | Stored | Standard production |
| STORE_EMBEDDINGS_ONLY | Not stored | Stored | Privacy-conscious |
| NEVER_STORE | Not stored | Not stored | Maximum privacy |

### Raw Log Policy

| Policy | Behavior |
|--------|----------|
| ALLOW | Store logs as-is (only with STORE_ALL mode) |
| REDACT_THEN_STORE | Apply redaction before storage |
| HASH_ONLY | Store only a hash of the log |
| DISCARD | Do not store raw logs |

### Configuration Example

```python
from sentinel_ml.security import (
    PrivacyManager,
    PrivacyConfig,
    PrivacyLevel,
    PrivacyMode,
    RawLogPolicy,
)

config = PrivacyConfig(
    level=PrivacyLevel.ENHANCED,
    mode=PrivacyMode.STORE_EMBEDDINGS_ONLY,
    raw_log_policy=RawLogPolicy.DISCARD,
    retain_metadata=True,  # Keep timestamp, level, etc.
)

manager = PrivacyManager(config)
```

## Encryption

### Requirements

Encryption requires the `cryptography` package:

```bash
pip install sentinel-ml[encryption]
# or
pip install cryptography
```

### Key Management

```python
from sentinel_ml.security import KeyManager

key_manager = KeyManager()

# Generate a new key
key = key_manager.generate_key()
print(f"Key ID: {key.key_id}")

# Rotate to a new key (old keys kept for decryption)
new_key = key_manager.rotate_key()

# Get key statistics
stats = key_manager.get_stats()
print(f"Total keys: {stats['total_keys']}")
```

### Password-Based Keys

```python
from sentinel_ml.security import KeyManager

key_manager = KeyManager()

# Derive key from password
key, salt = key_manager.derive_key_from_password("your-secure-password")

# Later, recreate key with same salt
key2, _ = key_manager.derive_key_from_password(
    "your-secure-password",
    salt=salt,
)
```

### Encrypted Store

```python
from sentinel_ml.security import EncryptedStore, KeyManager

key_manager = KeyManager()
store = EncryptedStore(key_manager=key_manager)

# Encrypt various data types
encrypted_str = store.encrypt_string("sensitive log data")
encrypted_bytes = store.encrypt(b"binary data")
encrypted_dict = store.encrypt_dict({"key": "value"})

# Decrypt
decrypted_str = store.decrypt_to_string(encrypted_str)
decrypted_bytes = store.decrypt(encrypted_bytes)
decrypted_dict = store.decrypt_to_dict(encrypted_dict)
```

## Safe Defaults

### Default Configuration

Sentinel Log AI uses privacy-conscious defaults:

```python
# Default redaction level
RedactionLevel.STANDARD

# Default privacy mode
PrivacyMode.STORE_REDACTED

# Default raw log policy
RawLogPolicy.REDACT_THEN_STORE

# PII types redacted by default
{EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS, API_KEY}
```

### Production Recommendations

```python
from sentinel_ml.security import (
    PrivacyManager,
    PrivacyConfig,
    PrivacyLevel,
    PrivacyMode,
    RawLogPolicy,
    RedactionLevel,
)

# Recommended production configuration
config = PrivacyConfig(
    level=PrivacyLevel.ENHANCED,
    mode=PrivacyMode.STORE_REDACTED,
    raw_log_policy=RawLogPolicy.DISCARD,
    redaction_level=RedactionLevel.STRICT,
    retain_metadata=True,
)

manager = PrivacyManager(config)
```

## Frequently Asked Questions

### General Security

**Q: Is my data encrypted in transit?**

A: The gRPC connection between the Go Agent and Python ML Server can be configured with TLS. See the deployment guide for TLS configuration.

**Q: Where are encryption keys stored?**

A: By default, keys are stored in memory. For production, configure a secure key storage backend or use environment variables.

**Q: How often should I rotate encryption keys?**

A: We recommend rotating keys every 90 days. Use `key_manager.rotate_key()` to generate new keys while keeping old keys for decryption.

### PII Redaction

**Q: Can I add custom PII patterns?**

A: Yes, use `CustomPattern` to define regex patterns for organization-specific data:

```python
custom = CustomPattern(
    name="order_id",
    pattern=r"ORD-[A-Z0-9]{10}",
    replacement="[ORDER_ID]",
)
```

**Q: Does redaction affect log analysis accuracy?**

A: Minimal impact. ML models work with embeddings of redacted text, which preserves semantic meaning while protecting sensitive data.

**Q: Can redacted data be recovered?**

A: No. Redaction is one-way. When `hash_sensitive=True`, a consistent hash is used for correlation, but the original value cannot be recovered.

### Privacy Modes

**Q: What mode should I use for GDPR compliance?**

A: Use `STORE_EMBEDDINGS_ONLY` or `NEVER_STORE` mode. These modes ensure raw logs with potential PII are not persisted.

**Q: Can I query logs in NEVER_STORE mode?**

A: Only during the session. Real-time anomaly detection works, but historical queries are not available.

**Q: How do I handle the right to erasure?**

A: With embedding-only storage, there is no recoverable personal data. For full compliance, implement data retention policies.

### Encryption

**Q: What encryption algorithm is used?**

A: Fernet (AES-128-CBC with HMAC for authentication). This provides both confidentiality and integrity.

**Q: Can I use my own encryption keys?**

A: Yes, you can import existing Fernet keys:

```python
from sentinel_ml.security import EncryptionKey
import base64

key_material = base64.urlsafe_b64decode(your_fernet_key)
key = EncryptionKey.from_bytes(key_material)
```

**Q: Is hardware security module (HSM) supported?**

A: Not currently. For HSM integration, you can implement a custom `EncryptionProvider`.

### Compliance

**Q: Does Sentinel Log AI support HIPAA?**

A: The security features (encryption, redaction, audit logging) support HIPAA requirements. Consult your compliance team for full assessment.

**Q: Is there an audit log?**

A: Yes, all security operations are logged via structlog. Configure appropriate log retention for audit trails.

**Q: Can I disable encryption for performance?**

A: Yes, but not recommended for production. Set `encryption.enabled=false` in configuration.

## Troubleshooting

### Common Issues

**Redaction not working**

1. Check redaction level is not `NONE`
2. Verify PII types are enabled in config
3. Test patterns with sample data

**Encryption errors**

1. Ensure `cryptography` package is installed
2. Verify key is valid Fernet key
3. Check key has not expired

**Performance concerns**

1. Use `RedactionLevel.STANDARD` instead of `PARANOID`
2. Consider batch processing for large volumes
3. Cache redactor instances

### Debug Mode

```python
import logging
logging.getLogger("sentinel_ml.security").setLevel(logging.DEBUG)
```

## Related Documentation

- [Threat Model](../docs/threat-model.md) - Security threat analysis
- [Architecture Overview](Architecture-Overview.md) - System design
- [Configuration Reference](Configuration-Reference.md) - All settings
