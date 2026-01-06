"""
Core data models for the ML engine.

These models mirror the Go models and protobuf definitions to ensure
consistent data structures across the polyglot architecture.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LogRecord(BaseModel):
    """Canonical log record used across ingestion, ML, storage, and explanation."""

    model_config = {"ser_json_timedelta": "iso8601"}

    id: str | None = Field(default=None, description="Unique identifier for this record")
    message: str = Field(..., description="Main log message content")
    normalized: str | None = Field(default=None, description="Normalized/masked message for ML")
    level: str | None = Field(default=None, description="Log level (INFO, WARN, ERROR, etc.)")
    source: str = Field(..., description="Source of the log (file path, journald unit, etc.)")
    timestamp: datetime | None = Field(default=None, description="Timestamp of the log entry")
    raw: str = Field(..., description="Original unparsed log line")
    attrs: dict[str, Any] = Field(default_factory=dict, description="Additional structured attributes")


class ClusterSummary(BaseModel):
    """Summary of a log cluster."""

    cluster_id: str = Field(..., description="Stable identifier for this cluster")
    size: int = Field(..., description="Number of logs in this cluster")
    representative: str = Field(..., description="Most representative log message")
    keywords: list[str] = Field(default_factory=list, description="Top keywords/tokens in this cluster")
    cohesion: float = Field(default=0.0, description="Cluster cohesion score (0.0 - 1.0)")
    novelty_score: float = Field(default=0.0, description="How novel this cluster is (0.0 - 1.0)")
    first_seen: datetime | None = Field(default=None, description="When this cluster was first observed")
    last_seen: datetime | None = Field(default=None, description="When this cluster was last observed")
    is_new: bool = Field(default=False, description="Whether this is a new cluster")


class ConfidenceLevel(str, Enum):
    """Confidence levels for LLM explanations."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Explanation(BaseModel):
    """LLM-generated explanation for a log cluster."""

    cluster_id: str = Field(..., description="Cluster ID this explanation is for")
    root_cause: str = Field(..., description="Probable root cause")
    next_steps: list[str] = Field(default_factory=list, description="Suggested actions to investigate")
    remediation: str | None = Field(default=None, description="Suggested fix if applicable")
    confidence: ConfidenceLevel = Field(..., description="Confidence level")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Numeric confidence (0.0-1.0)")
    confidence_reasoning: str | None = Field(default=None, description="Why this confidence was assigned")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=None), description="When this explanation was created")
    raw_response: str | None = Field(default=None, description="Raw LLM response for debugging")


class NoveltyResult(BaseModel):
    """Result of novelty detection for a log."""

    is_novel: bool = Field(..., description="Whether this log pattern is novel")
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Novelty score (0.0 = seen, 1.0 = new)")
    closest_cluster_id: str | None = Field(default=None, description="Closest cluster if any")
    distance_to_cluster: float | None = Field(default=None, description="Distance to closest cluster")
    reason: str | None = Field(default=None, description="Explanation of novelty assessment")


class SearchResult(BaseModel):
    """A single similar log search result."""

    record: LogRecord = Field(..., description="The matching log record")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score (higher is more similar)")
    distance: float = Field(..., description="Distance in embedding space")
    cluster_id: str | None = Field(default=None, description="Cluster ID this log belongs to")
