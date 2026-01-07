# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Packaging module for version management, changelog automation, and release workflows
- Dockerfile for containerized deployment with multi-stage build
- PyInstaller spec for Windows executable generation
- Installation verification and dependency checking utilities
- Comprehensive wiki documentation for packaging and release

### Changed
- Updated build configuration for hatchling dynamic versioning

## [0.10.0] - 2024-01-15

### Added
- Evaluation and quality framework (M10)
- Labeling utilities for ground truth annotation
- Golden dataset creation and validation
- Ablation study framework for systematic evaluation
- Agreement metrics for inter-annotator reliability
- Stratified labeling strategies

### Changed
- Improved benchmark metrics with comprehensive reporting
- Enhanced test coverage for evaluation components

### Fixed
- Flaky test in agreement metric computation
- Type hints for mypy strict mode compliance

## [0.9.0] - 2024-01-10

### Added
- Alerting and notification system (M9)
- Slack, Email, Webhook, and GitHub issue integrations
- Alert routing with priority-based delivery
- Health monitoring for alert channels
- Watch mode for real-time log processing

### Changed
- Improved configuration management with pydantic-settings
- Enhanced error handling across all modules

## [0.8.0] - 2024-01-05

### Added
- Storage backends for vector persistence (M8)
- ChromaDB integration for vector similarity search
- FAISS support for high-performance indexing
- Configurable caching layer for embeddings

### Changed
- Optimized embedding pipeline for batch processing
- Reduced memory footprint for large log datasets

## [0.7.0] - 2024-01-01

### Added
- LLM integration for log analysis (M7)
- OpenAI API support with configurable models
- Prompt templates for log summarization
- Context window management for large log batches

### Changed
- Improved clustering explanations with LLM assistance
- Enhanced novelty detection with semantic understanding

## [0.6.0] - 2023-12-28

### Added
- Vector store for semantic log indexing (M6)
- Similarity search across log embeddings
- Configurable distance metrics

### Changed
- Optimized embedding model loading
- Improved batch processing throughput

## [0.5.0] - 2023-12-25

### Added
- Novelty detection for anomaly identification (M5)
- Isolation Forest and LOF implementations
- Configurable anomaly thresholds

### Changed
- Enhanced preprocessing pipeline
- Improved log normalization accuracy

## [0.4.0] - 2023-12-22

### Added
- Clustering module for log pattern grouping (M4)
- DBSCAN and K-Means implementations
- Cluster quality metrics

### Changed
- Improved embedding dimensionality reduction
- Optimized clustering performance

## [0.3.0] - 2023-12-19

### Added
- Embedding generation with sentence transformers (M3)
- Configurable embedding models
- Batch embedding support

### Changed
- Improved log preprocessing
- Enhanced text normalization

## [0.2.0] - 2023-12-16

### Added
- Log parsing and preprocessing module (M2)
- Support for JSONL, JSON, and plain text formats
- Log normalization utilities

### Changed
- Improved project structure
- Enhanced configuration management

## [0.1.0] - 2023-12-13

### Added
- Initial project structure (M0/M1)
- Go agent for log ingestion
- Python ML engine foundation
- gRPC communication layer
- Basic CLI interface
- Structured logging with structlog
- Configuration management with pydantic

[Unreleased]: https://github.com/varadharajaan/sentinel-log-ai/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/varadharajaan/sentinel-log-ai/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/varadharajaan/sentinel-log-ai/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/varadharajaan/sentinel-log-ai/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/varadharajaan/sentinel-log-ai/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/varadharajaan/sentinel-log-ai/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/varadharajaan/sentinel-log-ai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/varadharajaan/sentinel-log-ai/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/varadharajaan/sentinel-log-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/varadharajaan/sentinel-log-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/varadharajaan/sentinel-log-ai/releases/tag/v0.1.0
