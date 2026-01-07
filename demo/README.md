# 🎯 Demo: ML vs Regex for Log Security Detection

> **The definitive demo showing why ML detection beats regex for security logs.**

## The Problem

When teams ask "Why can't we just use regex?", this demo provides the answer.

| Approach | How It Works | Catches Known Attacks | Catches Novel Attacks |
|----------|--------------|----------------------|----------------------|
| **Regex** | Pattern match on TEXT | ✅ 4/4 (100%) | ❌ 0/17 (0%) |
| **ML** | Semantic match on MEANING | ✅ 4/4 (100%) | ✅ 14/17 (82%) |

**Attackers don't reuse the same patterns. They innovate. Your detection must too.**

## Quick Start

```bash
# From the demo folder
python demo_ml_vs_regex.py
```

## What the Demo Shows

### Step 1: Baseline Learning
ML builds understanding of "normal" from 75 production logs:
- Kubernetes pods, PostgreSQL queries, Kafka consumers
- Redis cache, OAuth tokens, GraphQL operations
- Cron jobs, gRPC calls, nginx access, Prometheus metrics

### Step 2: Known Attack Detection
Both regex and ML detect 4/4 classic attacks:
- SQL injection (`' OR 1=1`)
- XSS (`<script>alert`)
- Path traversal (`../../../etc/passwd`)

### Step 3: Novel Attack Detection
The key difference - attacks regex has NEVER seen:

| Attack Category | Example | Regex | ML |
|-----------------|---------|-------|-----|
| **Supply Chain** | Malicious package install | ❌ | ✅ |
| **Credential Stuffing** | Distributed login attempts | ❌ | ✅ |
| **Container Escape** | Docker socket access | ❌ | ✅ |
| **SSRF** | Internal service probe | ❌ | ✅ |
| **Privilege Escalation** | Unexpected sudo/setuid | ❌ | ✅ |
| **DNS Exfiltration** | Base64 encoded subdomains | ❌ | ✅ |

## Files

| File | Description |
|------|-------------|
| `demo_ml_vs_regex.py` | Interactive demo script with colorful output |
| `demo_logs.jsonl` | 96 production logs (75 normal, 4 known attacks, 17 novel attacks) |

## The Dataset

### Normal Logs (75)
Real-world patterns from production systems:
- Kubernetes: Pod lifecycle, probe failures, scaling events
- Databases: PostgreSQL query times, connection pools
- Message Queues: Kafka consumer lag, partition rebalance
- Security: OAuth token refresh, session management
- APIs: GraphQL queries, gRPC calls, REST endpoints

### Known Attacks (4)
Classic patterns that regex easily detects:
- SQL injection with `' OR 1=1`
- XSS with `<script>` tags
- Path traversal with `../`

### Novel Attacks (17)
Real attack patterns that regex misses:
- **Supply chain**: `pip install cryptominer-helper-v2.1.0`
- **Credential stuffing**: 847 unique IPs, 12,000 attempts
- **Container escape**: `/var/run/docker.sock` access
- **SSRF**: `http://169.254.169.254/metadata`
- **Privilege escalation**: Unexpected setuid binary
- **DNS exfiltration**: Base64 in DNS queries

## Key Insight

> Regex matches **TEXT** — the exact characters in a pattern.
> 
> ML matches **MEANING** — what the log semantically represents.

When attackers change their text but keep the same intent, regex fails. ML catches them because the meaning hasn't changed.

## Requirements

```bash
pip install sentence-transformers numpy scikit-learn
```

## Related Documentation

- [docs/demo.md](../docs/demo.md) - Full demo documentation
- [wiki/FAQ.md](../wiki/FAQ.md) - "Why ML instead of Regex?" FAQ
- [Architecture Overview](../wiki/Architecture-Overview.md) - How the ML stack works

---

*Crafted with ❤️ by [Varad](https://github.com/varadharajaan)*
