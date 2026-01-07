# Sentinel Log AI Wiki

<div align="center">

**🔍 AI-Powered Log Intelligence for On-Call Engineers**

*Crafted with ❤️ by [Varad](https://github.com/varadharajaan)*

</div>

---

## Welcome

Welcome to the **Sentinel Log AI** wiki! This documentation provides comprehensive coverage of the system architecture, design decisions, usage patterns, and development guidelines.

Sentinel Log AI is a polyglot log intelligence system that combines:
- **Go's performance** for high-throughput log ingestion
- **Python's ML ecosystem** for intelligent pattern analysis
- **Local LLM inference** for human-readable explanations

## 📚 Table of Contents

### Getting Started
- [[Quick Start Guide|Quick-Start]]
- [[Installation|Installation]]
- [[Configuration|Configuration]]
- [[First Analysis|First-Analysis]]

### Architecture
- [[System Overview|Architecture-Overview]]
- [[Go Agent|Go-Agent]]
- [[Python ML Engine|Python-ML-Engine]]
- [[gRPC Communication|gRPC-Protocol]]
- [[Design Patterns|Design-Patterns]]

### Core Features
- [[Log Ingestion|Log-Ingestion]]
- [[Embeddings & Vector Store|Embeddings]]
- [[Clustering & Patterns|Clustering]]
- [[Novelty Detection|Novelty-Detection]]
- [[LLM Explanations|LLM-Explanations]]
- [[CLI & UX|CLI-UX]]

### Development
- [[Development Setup|Development-Setup]]
- [[Testing Guide|Testing-Guide]]
- [[Code Style|Code-Style]]
- [[Contributing|Contributing]]

### Reference
- [[API Reference|API-Reference]]
- [[Configuration Reference|Configuration-Reference]]
- [[Error Codes|Error-Codes]]
- [[FAQ|FAQ]]

---

## 🎯 Project Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| M0 | ✅ Complete | Project scaffolding & DevX |
| M1 | ✅ Complete | Ingestion & preprocessing pipeline |
| M2 | ✅ Complete | Embeddings & FAISS vector store |
| M3 | ✅ Complete | HDBSCAN clustering & patterns |
| M4 | ✅ Complete | Novelty detection (k-NN) |
| M5 | ✅ Complete | LLM explanations (Ollama) |
| M6 | ✅ Complete | CLI polish & rich output |
| M7 | 🔄 Planned | Performance & docs |
| M8 | 🔄 Planned | Storage & retention |
| M9 | 🔄 Planned | Alerting integrations |

## 🏆 Key Capabilities

### Pattern Discovery
Automatically groups similar log messages using semantic embeddings and density-based clustering. No manual regex rules required.

### Novel Error Detection
Identifies log patterns that don't match any known cluster — the errors you haven't seen before and need to investigate.

### Root Cause Analysis
LLM-powered explanations provide:
- Root cause hypothesis
- Suggested investigation steps
- Confidence scores
- Severity assessment

### Local-First Privacy
Everything runs on your machine:
- No cloud API calls for embeddings
- Local Ollama for LLM inference
- Your logs never leave your infrastructure

---

## 🚀 Quick Links

- [GitHub Repository](https://github.com/varadharajaan/sentinel-log-ai)
- [Technical Docs (in repo)](https://github.com/varadharajaan/sentinel-log-ai/tree/main/docs)
- [Issues & Bug Reports](https://github.com/varadharajaan/sentinel-log-ai/issues)

---

## 📁 Technical Documentation

For in-depth technical specifications, see the [`docs/`](https://github.com/varadharajaan/sentinel-log-ai/tree/main/docs) directory:

| Document | Description |
|----------|-------------|
| [Architecture](https://github.com/varadharajaan/sentinel-log-ai/blob/main/docs/architecture.md) | System architecture with Mermaid diagrams |
| [Data Flow](https://github.com/varadharajaan/sentinel-log-ai/blob/main/docs/data-flow.md) | End-to-end data flow and sequence diagrams |
| [Error Handling](https://github.com/varadharajaan/sentinel-log-ai/blob/main/docs/error-handling.md) | Error codes, exceptions, recovery strategies |
| [Logging](https://github.com/varadharajaan/sentinel-log-ai/blob/main/docs/logging.md) | JSONL format, Athena integration |
- [Pull Requests](https://github.com/varadharajaan/sentinel-log-ai/pulls)
- [Releases](https://github.com/varadharajaan/sentinel-log-ai/releases)

---

*Last updated: January 2026*
