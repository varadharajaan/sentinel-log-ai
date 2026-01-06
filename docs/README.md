# Sentinel Log AI - Technical Documentation

This directory contains technical documentation for the Sentinel Log AI system.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System architecture, component overview, and design decisions |
| [Data Flow](data-flow.md) | End-to-end data flow diagrams and processing pipelines |
| [API Reference](api-reference.md) | gRPC API specification and usage |
| [Error Handling](error-handling.md) | Error codes, exception hierarchy, and recovery strategies |
| [Logging](logging.md) | JSONL logging format, Athena integration, and observability |
| [Configuration](configuration.md) | Configuration options and environment variables |

## Quick Links

- [Go Agent Documentation](go-agent.md)
- [Python ML Engine Documentation](python-ml-engine.md)
- [Development Guide](development.md)
- [Deployment Guide](deployment.md)

## Architecture Overview

Sentinel Log AI is a polyglot log intelligence system:

```
+------------------+     gRPC      +------------------+
|   Go Agent       | <-----------> |  Python ML       |
|   (Ingestion)    |               |  Engine          |
+------------------+               +------------------+
        |                                  |
        v                                  v
  +------------+                   +---------------+
  | Log Sources|                   | Vector Store  |
  | - Files    |                   | (FAISS)       |
  | - Stdin    |                   +---------------+
  | - Journald |                           |
  +------------+                           v
                                   +---------------+
                                   | LLM Provider  |
                                   | (Ollama/API)  |
                                   +---------------+
```

## Key Design Principles

1. **Polyglot Architecture**: Go for high-performance ingestion, Python for ML/AI
2. **SOLID Design Patterns**: Single responsibility, dependency injection
3. **Structured Logging**: JSONL format for analytics pipeline integration
4. **Error Isolation**: Typed errors with retry logic and graceful degradation
5. **TDD Approach**: Comprehensive test coverage for all components
