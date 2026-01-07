"""
Clustering quality metrics for evaluation.

This module provides implementations of standard clustering quality metrics
including Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.

Design Patterns:
- Strategy Pattern: Each metric is a pluggable strategy
- Factory Pattern: MetricFactory for creating metrics
- Observer Pattern: Trend tracking for metric changes

SOLID Principles:
- Single Responsibility: Each metric class computes one metric
- Open/Closed: Add new metrics without modifying existing code
- Liskov Substitution: All metrics implement MetricStrategy
- Interface Segregation: Minimal MetricStrategy interface
- Dependency Inversion: QualityEvaluator depends on abstractions
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Types of clustering quality metrics."""

    SILHOUETTE = "silhouette"
    DAVIES_BOULDIN = "davies_bouldin"
    CALINSKI_HARABASZ = "calinski_harabasz"


@dataclass
class MetricResult:
    """
    Result from a single metric computation.

    Attributes:
        metric_type: Type of the metric computed.
        value: The computed metric value.
        interpretation: Human-readable interpretation.
        optimal_direction: Whether higher or lower is better.
        computation_time_seconds: Time taken to compute.
        metadata: Additional metric-specific data.
    """

    metric_type: MetricType
    value: float
    interpretation: str
    optimal_direction: str  # "higher" or "lower"
    computation_time_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_type": self.metric_type.value,
            "value": round(self.value, 6),
            "interpretation": self.interpretation,
            "optimal_direction": self.optimal_direction,
            "computation_time_seconds": round(self.computation_time_seconds, 6),
            "metadata": self.metadata,
        }


@dataclass
class ClusteringQualityResult:
    """
    Aggregated clustering quality results.

    Attributes:
        metrics: Individual metric results.
        overall_quality: Normalized overall quality score (0-1).
        n_samples: Number of samples evaluated.
        n_clusters: Number of clusters found.
        n_noise: Number of noise points.
        evaluation_time_seconds: Total evaluation time.
        timestamp: When evaluation was performed.
        warnings: Any warnings generated during evaluation.
    """

    metrics: list[MetricResult] = field(default_factory=list)
    overall_quality: float = 0.0
    n_samples: int = 0
    n_clusters: int = 0
    n_noise: int = 0
    evaluation_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    warnings: list[str] = field(default_factory=list)

    def get_metric(self, metric_type: MetricType) -> MetricResult | None:
        """Get a specific metric result by type."""
        for metric in self.metrics:
            if metric.metric_type == metric_type:
                return metric
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "overall_quality": round(self.overall_quality, 4),
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "evaluation_time_seconds": round(self.evaluation_time_seconds, 6),
            "timestamp": self.timestamp.isoformat(),
            "warnings": self.warnings,
        }


class MetricStrategy(ABC):
    """
    Abstract base class for clustering quality metrics.

    Implements the Strategy pattern for pluggable metric computation.
    """

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Return the metric type."""

    @property
    @abstractmethod
    def optimal_direction(self) -> str:
        """Return whether higher or lower is better."""

    @abstractmethod
    def compute(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> MetricResult:
        """
        Compute the metric for the given clustering.

        Args:
            embeddings: Array of embeddings with shape (n_samples, n_features).
            labels: Array of cluster labels. -1 indicates noise.

        Returns:
            MetricResult with computed value and interpretation.
        """

    def _validate_inputs(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], int]:
        """
        Validate and filter inputs for metric computation.

        Args:
            embeddings: Raw embeddings array.
            labels: Raw labels array.

        Returns:
            Tuple of (filtered_embeddings, filtered_labels, n_clusters).

        Raises:
            ValueError: If inputs are invalid for metric computation.
        """
        if len(embeddings) != len(labels):
            msg = f"Embeddings ({len(embeddings)}) and labels ({len(labels)}) must have same length"
            raise ValueError(msg)

        # Filter out noise points (label == -1)
        mask = labels >= 0
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]

        n_clusters = len(set(filtered_labels))

        if len(filtered_embeddings) < 2:
            msg = "Need at least 2 non-noise samples for metric computation"
            raise ValueError(msg)

        if n_clusters < 2:
            msg = f"Need at least 2 clusters for metric computation, got {n_clusters}"
            raise ValueError(msg)

        return filtered_embeddings, filtered_labels, n_clusters


class SilhouetteMetric(MetricStrategy):
    """
    Silhouette Score metric for clustering quality.

    The Silhouette Score measures how similar an object is to its own cluster
    compared to other clusters. Values range from -1 to 1:
    - 1: Clusters are well separated
    - 0: Clusters are overlapping
    - -1: Samples may be assigned to wrong clusters
    """

    @property
    def metric_type(self) -> MetricType:
        """Return the metric type."""
        return MetricType.SILHOUETTE

    @property
    def optimal_direction(self) -> str:
        """Higher silhouette score is better."""
        return "higher"

    def compute(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> MetricResult:
        """
        Compute Silhouette Score.

        Args:
            embeddings: Array of embeddings.
            labels: Array of cluster labels.

        Returns:
            MetricResult with silhouette score.
        """
        start_time = time.perf_counter()

        try:
            filtered_emb, filtered_labels, n_clusters = self._validate_inputs(
                embeddings, labels
            )

            from sklearn.metrics import silhouette_score

            score = float(silhouette_score(filtered_emb, filtered_labels))

            interpretation = self._interpret_score(score)

            logger.debug(
                "silhouette_score_computed",
                score=round(score, 4),
                n_samples=len(filtered_emb),
                n_clusters=n_clusters,
            )

            return MetricResult(
                metric_type=self.metric_type,
                value=score,
                interpretation=interpretation,
                optimal_direction=self.optimal_direction,
                computation_time_seconds=time.perf_counter() - start_time,
                metadata={
                    "n_samples": len(filtered_emb),
                    "n_clusters": n_clusters,
                    "score_range": {"min": -1.0, "max": 1.0},
                },
            )

        except ValueError as e:
            logger.warning("silhouette_score_failed", error=str(e))
            return MetricResult(
                metric_type=self.metric_type,
                value=float("nan"),
                interpretation=f"Could not compute: {e}",
                optimal_direction=self.optimal_direction,
                computation_time_seconds=time.perf_counter() - start_time,
                metadata={"error": str(e)},
            )

    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation of the score."""
        if score >= 0.7:
            return "Excellent cluster separation"
        if score >= 0.5:
            return "Good cluster separation"
        if score >= 0.25:
            return "Moderate cluster separation"
        if score >= 0.0:
            return "Weak cluster separation, clusters may overlap"
        return "Poor clustering, samples may be misassigned"


class DaviesBouldinMetric(MetricStrategy):
    """
    Davies-Bouldin Index for clustering quality.

    The Davies-Bouldin Index measures the average similarity between clusters,
    where similarity is the ratio of within-cluster distances to between-cluster
    distances. Lower values indicate better clustering.

    A value of 0 would indicate perfect clustering (clusters infinitely far
    apart and infinitely compact).
    """

    @property
    def metric_type(self) -> MetricType:
        """Return the metric type."""
        return MetricType.DAVIES_BOULDIN

    @property
    def optimal_direction(self) -> str:
        """Lower Davies-Bouldin index is better."""
        return "lower"

    def compute(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> MetricResult:
        """
        Compute Davies-Bouldin Index.

        Args:
            embeddings: Array of embeddings.
            labels: Array of cluster labels.

        Returns:
            MetricResult with Davies-Bouldin index.
        """
        start_time = time.perf_counter()

        try:
            filtered_emb, filtered_labels, n_clusters = self._validate_inputs(
                embeddings, labels
            )

            from sklearn.metrics import davies_bouldin_score

            score = float(davies_bouldin_score(filtered_emb, filtered_labels))

            interpretation = self._interpret_score(score)

            logger.debug(
                "davies_bouldin_score_computed",
                score=round(score, 4),
                n_samples=len(filtered_emb),
                n_clusters=n_clusters,
            )

            return MetricResult(
                metric_type=self.metric_type,
                value=score,
                interpretation=interpretation,
                optimal_direction=self.optimal_direction,
                computation_time_seconds=time.perf_counter() - start_time,
                metadata={
                    "n_samples": len(filtered_emb),
                    "n_clusters": n_clusters,
                    "score_range": {"min": 0.0, "max": "unbounded"},
                },
            )

        except ValueError as e:
            logger.warning("davies_bouldin_score_failed", error=str(e))
            return MetricResult(
                metric_type=self.metric_type,
                value=float("nan"),
                interpretation=f"Could not compute: {e}",
                optimal_direction=self.optimal_direction,
                computation_time_seconds=time.perf_counter() - start_time,
                metadata={"error": str(e)},
            )

    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation of the score."""
        if score <= 0.5:
            return "Excellent cluster compactness and separation"
        if score <= 1.0:
            return "Good cluster quality"
        if score <= 2.0:
            return "Moderate cluster quality"
        return "Poor cluster quality, consider adjusting parameters"


class CalinskiHarabaszMetric(MetricStrategy):
    """
    Calinski-Harabasz Index (Variance Ratio Criterion).

    This metric evaluates cluster quality based on the ratio of between-cluster
    dispersion to within-cluster dispersion. Higher values indicate better
    defined clusters.
    """

    @property
    def metric_type(self) -> MetricType:
        """Return the metric type."""
        return MetricType.CALINSKI_HARABASZ

    @property
    def optimal_direction(self) -> str:
        """Higher Calinski-Harabasz index is better."""
        return "higher"

    def compute(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> MetricResult:
        """
        Compute Calinski-Harabasz Index.

        Args:
            embeddings: Array of embeddings.
            labels: Array of cluster labels.

        Returns:
            MetricResult with Calinski-Harabasz index.
        """
        start_time = time.perf_counter()

        try:
            filtered_emb, filtered_labels, n_clusters = self._validate_inputs(
                embeddings, labels
            )

            from sklearn.metrics import calinski_harabasz_score

            score = float(calinski_harabasz_score(filtered_emb, filtered_labels))

            interpretation = self._interpret_score(score, n_clusters)

            logger.debug(
                "calinski_harabasz_score_computed",
                score=round(score, 4),
                n_samples=len(filtered_emb),
                n_clusters=n_clusters,
            )

            return MetricResult(
                metric_type=self.metric_type,
                value=score,
                interpretation=interpretation,
                optimal_direction=self.optimal_direction,
                computation_time_seconds=time.perf_counter() - start_time,
                metadata={
                    "n_samples": len(filtered_emb),
                    "n_clusters": n_clusters,
                    "score_range": {"min": 0.0, "max": "unbounded"},
                },
            )

        except ValueError as e:
            logger.warning("calinski_harabasz_score_failed", error=str(e))
            return MetricResult(
                metric_type=self.metric_type,
                value=float("nan"),
                interpretation=f"Could not compute: {e}",
                optimal_direction=self.optimal_direction,
                computation_time_seconds=time.perf_counter() - start_time,
                metadata={"error": str(e)},
            )

    def _interpret_score(self, score: float, n_clusters: int) -> str:
        """Provide human-readable interpretation of the score."""
        # CH index is relative to number of clusters and samples
        # Higher is generally better, but absolute thresholds are domain-specific
        if score >= 500:
            return "Excellent cluster definition"
        if score >= 200:
            return "Good cluster definition"
        if score >= 50:
            return "Moderate cluster definition"
        return "Weak cluster definition"


class QualityEvaluator:
    """
    Evaluator for clustering quality using multiple metrics.

    Coordinates metric computation and aggregates results.
    Implements the Facade pattern for simplified quality evaluation.

    Usage:
        evaluator = QualityEvaluator()
        result = evaluator.evaluate(embeddings, labels)
        print(f"Overall quality: {result.overall_quality}")
    """

    # Default metrics to compute
    DEFAULT_METRICS: list[type[MetricStrategy]] = [
        SilhouetteMetric,
        DaviesBouldinMetric,
        CalinskiHarabaszMetric,
    ]

    def __init__(
        self,
        metrics: list[MetricStrategy] | None = None,
    ) -> None:
        """
        Initialize the quality evaluator.

        Args:
            metrics: List of metric strategies to use. If None, uses defaults.
        """
        if metrics is None:
            self._metrics = [cls() for cls in self.DEFAULT_METRICS]
        else:
            self._metrics = metrics

        logger.info(
            "quality_evaluator_initialized",
            metrics=[m.metric_type.value for m in self._metrics],
        )

    def evaluate(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> ClusteringQualityResult:
        """
        Evaluate clustering quality using all configured metrics.

        Args:
            embeddings: Array of embeddings with shape (n_samples, n_features).
            labels: Array of cluster labels. -1 indicates noise.

        Returns:
            ClusteringQualityResult with all metric values and overall score.
        """
        start_time = time.perf_counter()
        warnings: list[str] = []

        # Basic statistics
        n_samples = len(labels)
        n_noise = int(np.sum(labels == -1))
        unique_labels = set(labels)
        unique_labels.discard(-1)
        n_clusters = len(unique_labels)

        logger.info(
            "clustering_quality_evaluation_started",
            n_samples=n_samples,
            n_clusters=n_clusters,
            n_noise=n_noise,
        )

        # Compute all metrics
        metric_results: list[MetricResult] = []
        for metric in self._metrics:
            try:
                result = metric.compute(embeddings, labels)
                metric_results.append(result)

                if np.isnan(result.value):
                    warnings.append(
                        f"{result.metric_type.value}: {result.interpretation}"
                    )

            except Exception as e:
                logger.error(
                    "metric_computation_failed",
                    metric=metric.metric_type.value,
                    error=str(e),
                )
                warnings.append(f"{metric.metric_type.value}: computation failed - {e}")

        # Compute overall quality score
        overall_quality = self._compute_overall_quality(metric_results)

        total_time = time.perf_counter() - start_time

        result = ClusteringQualityResult(
            metrics=metric_results,
            overall_quality=overall_quality,
            n_samples=n_samples,
            n_clusters=n_clusters,
            n_noise=n_noise,
            evaluation_time_seconds=total_time,
            warnings=warnings,
        )

        logger.info(
            "clustering_quality_evaluation_completed",
            overall_quality=round(overall_quality, 4),
            n_metrics=len(metric_results),
            evaluation_time_seconds=round(total_time, 4),
        )

        return result

    def _compute_overall_quality(
        self,
        metrics: list[MetricResult],
    ) -> float:
        """
        Compute a normalized overall quality score from individual metrics.

        Uses weighted normalization based on metric characteristics:
        - Silhouette: normalized to 0-1 (already -1 to 1)
        - Davies-Bouldin: inverted and normalized (lower is better)
        - Calinski-Harabasz: logarithmically scaled

        Args:
            metrics: List of computed metric results.

        Returns:
            Overall quality score from 0.0 to 1.0.
        """
        if not metrics:
            return 0.0

        valid_scores: list[float] = []

        for metric in metrics:
            if np.isnan(metric.value):
                continue

            if metric.metric_type == MetricType.SILHOUETTE:
                # Silhouette: -1 to 1, normalize to 0 to 1
                normalized = (metric.value + 1.0) / 2.0
                valid_scores.append(normalized)

            elif metric.metric_type == MetricType.DAVIES_BOULDIN:
                # Davies-Bouldin: 0 to infinity, lower is better
                # Use inverse with clipping
                normalized = max(0.0, min(1.0, 1.0 / (1.0 + metric.value)))
                valid_scores.append(normalized)

            elif metric.metric_type == MetricType.CALINSKI_HARABASZ:
                # Calinski-Harabasz: 0 to infinity, higher is better
                # Use logarithmic scaling
                import math

                if metric.value > 0:
                    normalized = min(1.0, math.log10(metric.value + 1) / 3.0)
                    valid_scores.append(normalized)

        if not valid_scores:
            return 0.0

        return sum(valid_scores) / len(valid_scores)

    def add_metric(self, metric: MetricStrategy) -> None:
        """
        Add a metric strategy to the evaluator.

        Args:
            metric: Metric strategy to add.
        """
        self._metrics.append(metric)
        logger.debug("metric_added", metric_type=metric.metric_type.value)

    def remove_metric(self, metric_type: MetricType) -> bool:
        """
        Remove a metric strategy by type.

        Args:
            metric_type: Type of metric to remove.

        Returns:
            True if metric was found and removed.
        """
        for i, metric in enumerate(self._metrics):
            if metric.metric_type == metric_type:
                self._metrics.pop(i)
                logger.debug("metric_removed", metric_type=metric_type.value)
                return True
        return False

    @property
    def metrics(self) -> list[MetricStrategy]:
        """Return list of configured metrics."""
        return self._metrics.copy()
