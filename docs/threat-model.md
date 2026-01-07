# Threat Model

This document describes the security threat model for Sentinel Log AI, including trust boundaries, data flow, potential attack vectors, and mitigations.

## 1. System Overview

Sentinel Log AI is a log analysis system that ingests, processes, and stores log data. The system uses machine learning for anomaly detection and can integrate with external services for alerting.

### 1.1 Components

| Component | Description | Trust Level |
|-----------|-------------|-------------|
| Log Source | External systems generating logs | Untrusted |
| Go Agent | Log ingestion and forwarding | Semi-trusted |
| Python ML Server | ML processing, embedding, storage | Trusted |
| Vector Store | Embedding storage (ChromaDB/FAISS) | Trusted |
| LLM Provider | External AI service (OpenAI, etc.) | External |
| Alert Destinations | Slack, Email, GitHub | External |

### 1.2 Data Classification

| Data Type | Sensitivity | Handling |
|-----------|-------------|----------|
| Raw Logs | High | Redact PII before processing |
| Log Embeddings | Medium | No PII reconstruction possible |
| Cluster Labels | Low | Anonymized descriptions |
| Alert Content | Medium | May contain sanitized snippets |

## 2. Trust Boundaries

```
+------------------------------------------------------------------+
|                        EXTERNAL ZONE                              |
|  +------------+  +------------+  +-----------+  +-------------+   |
|  | Log Source |  |  Alerting  |  |    LLM    |  |   Vector    |   |
|  |  Systems   |  | (Slack/    |  | Provider  |  |   Store     |   |
|  +------------+  |   Email)   |  +-----------+  +-------------+   |
+------------------------------------------------------------------+
        |                |              |                |
        v                ^              ^                ^
========|================|==============|================|==========
        |     TRUST BOUNDARY (Network/Auth)              |
========|================|==============|================|==========
        v                |              |                |
+------------------------------------------------------------------+
|                        INTERNAL ZONE                              |
|  +------------+       +----------------------------------+        |
|  | Go Agent   | ----> |        Python ML Server          |        |
|  | (Ingestion)|       |  +------------+ +-------------+  |        |
|  +------------+       |  | Redaction  | | Encryption  |  |        |
|                       |  +------------+ +-------------+  |        |
|                       |  +------------+ +-------------+  |        |
|                       |  | Privacy    | | ML Pipeline |  |        |
|                       |  | Manager    | |             |  |        |
|                       |  +------------+ +-------------+  |        |
|                       +----------------------------------+        |
+------------------------------------------------------------------+
```

## 3. Data Flow Security

### 3.1 Ingestion Flow

```
Log Source --> Go Agent --> Python Server --> Storage
    |             |              |              |
    v             v              v              v
[Untrusted]  [Validate]    [Redact PII]   [Encrypt]
             [Parse]       [Sanitize]     [Store]
```

### 3.2 What Each Component Can See

| Component | Raw Logs | Redacted Logs | Embeddings | Decrypted Data |
|-----------|----------|---------------|------------|----------------|
| Log Source | Yes | No | No | No |
| Go Agent | Yes | No | No | No |
| Redaction Module | Yes | Yes | No | No |
| Privacy Manager | No | Yes | No | No |
| Embedding Module | No | Yes | Yes | No |
| Vector Store | No | No | Yes | No |
| LLM Provider | No | Partial | No | No |
| Encryption Module | No | No | No | Yes |
| Alerting | No | Partial | No | No |

### 3.3 Data Minimization

The system follows data minimization principles:

1. **PII Redaction**: Applied before any storage or external transmission
2. **Embedding-Only Mode**: Store only embeddings, not original text
3. **Never Store Mode**: Process logs without any persistent storage
4. **Automatic Expiration**: Keys and data can be configured to expire

## 4. Threat Analysis

### 4.1 STRIDE Analysis

| Threat | Category | Risk | Mitigation |
|--------|----------|------|------------|
| Log injection attack | Spoofing | Medium | Input validation, parsing |
| PII in logs leaked | Information Disclosure | High | PII redaction, encryption |
| Unauthorized access to stored logs | Information Disclosure | High | Encryption at rest |
| Man-in-the-middle on log transport | Tampering | Medium | TLS, gRPC encryption |
| Malicious log patterns | Denial of Service | Medium | Rate limiting, validation |
| Embedding reconstruction | Information Disclosure | Low | Use high-dim embeddings |
| Key theft | Information Disclosure | High | Key rotation, secure storage |

### 4.2 Attack Vectors

#### 4.2.1 Log Injection

**Threat**: Attacker injects malicious log entries to exploit parsing or storage.

**Mitigations**:
- Input validation in Go Agent
- Structured parsing with error handling
- Size limits on log entries
- Special character sanitization

#### 4.2.2 PII Exposure

**Threat**: Sensitive data in logs exposed to unauthorized parties.

**Mitigations**:
- Multiple redaction levels (MINIMAL to PARANOID)
- 12+ PII pattern types detected
- Hash-based replacements for correlation
- Never-store mode for maximum privacy

#### 4.2.3 Data Breach

**Threat**: Stored data accessed by unauthorized parties.

**Mitigations**:
- Fernet encryption (AES-128-CBC)
- Key rotation support
- Password-based key derivation (PBKDF2)
- Encrypted storage with authenticated encryption

#### 4.2.4 External Service Exposure

**Threat**: Sensitive data sent to external services (LLM, alerting).

**Mitigations**:
- PII redaction before LLM calls
- Sanitized content in alerts
- Configurable alert content levels
- No raw logs to external services

## 5. Security Controls

### 5.1 PII Redaction

```python
from sentinel_ml.security import RedactorFactory, RedactionConfig

config = RedactionConfig(
    level=RedactionLevel.STRICT,
    pii_types={PIIType.EMAIL, PIIType.PHONE, PIIType.SSN}
)
redactor = RedactorFactory.create(config)
result = redactor.redact(log_text)
```

Supported PII Types:
- EMAIL, PHONE, SSN, CREDIT_CARD
- IP_ADDRESS, MAC_ADDRESS
- API_KEY, PASSWORD, SECRET
- USERNAME, DOB, ADDRESS

### 5.2 Privacy Modes

| Mode | Raw Log Storage | Embedding Storage | Use Case |
|------|-----------------|-------------------|----------|
| STORE_ALL | Yes | Yes | Development |
| STORE_REDACTED | No (redacted) | Yes | Production |
| STORE_EMBEDDINGS_ONLY | No | Yes | Privacy-first |
| NEVER_STORE | No | No | Maximum privacy |

### 5.3 Encryption

```python
from sentinel_ml.security import EncryptedStore, KeyManager

key_manager = KeyManager()
store = EncryptedStore(key_manager=key_manager)

# Encrypt sensitive data
encrypted = store.encrypt_string(sensitive_log)

# Decrypt when needed
decrypted = store.decrypt_to_string(encrypted)
```

Features:
- Fernet symmetric encryption
- Automatic key generation
- Key rotation support
- Password-based key derivation

## 6. Security Best Practices

### 6.1 Configuration

```yaml
# Recommended production settings
security:
  redaction:
    level: STRICT
    hash_sensitive: true
    preserve_format: false
  
  privacy:
    level: ENHANCED
    mode: STORE_REDACTED
    raw_log_policy: DISCARD
  
  encryption:
    enabled: true
    algorithm: FERNET
    key_rotation_days: 90
```

### 6.2 Operational Security

1. **Key Management**
   - Store encryption keys separately from encrypted data
   - Use environment variables or secret managers for keys
   - Rotate keys regularly (recommended: 90 days)

2. **Access Control**
   - Limit access to ML server endpoints
   - Use authentication for gRPC connections
   - Audit access to stored data

3. **Monitoring**
   - Log security events (key rotation, access)
   - Monitor for unusual patterns
   - Alert on decryption failures

4. **Data Retention**
   - Define retention policies
   - Implement automatic data expiration
   - Secure deletion procedures

## 7. Compliance Considerations

### 7.1 GDPR

- PII redaction supports right to erasure
- Never-store mode prevents data retention
- Hash-based replacements maintain auditability

### 7.2 HIPAA

- Encryption at rest satisfies ePHI requirements
- Access logging for audit trails
- Data minimization through redaction

### 7.3 SOC 2

- Encryption controls for data protection
- Key management procedures
- Security monitoring and logging

## 8. Incident Response

### 8.1 Key Compromise

1. Generate new encryption keys immediately
2. Re-encrypt all stored data with new keys
3. Revoke compromised keys
4. Audit access logs for unauthorized access
5. Notify affected parties if required

### 8.2 Data Breach

1. Identify scope of exposed data
2. If encrypted, verify key was not compromised
3. If PII exposed, assess redaction effectiveness
4. Follow organizational incident response procedures
5. Document lessons learned

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-21 | Initial threat model |
