"""
HDBSCAN-based clustering for log pattern discovery.

This module provides clustering functionality for grouping similar log messages
based on their embedding vectors, enabling pattern discovery and summarization.

Design Patterns:
- Strategy Pattern: Pluggable clustering algorithms (HDBSCAN, KMeans, etc.)
- Factory Pattern: Cluster service creation with configuration
- Template Method: Common clustering workflow with customizable steps
- Observer Pattern: Hooks for cluster lifecycle events

SOLID Principles:
- Single Responsibility: Each class handles one concern
- Open/Closed: Extensible via ClusteringAlgorithm interface
- Liskov Substitution: All algorithms implement same interface
- Interface Segregation: Minimal interfaces for specific capabilities
- Dependency Inversion: Depends on abstractions not implementations
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from sentinel_ml.config import ClusteringConfig, get_config
from sentinel_ml.exceptions import ProcessingError
from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sentinel_ml.models import LogRecord

logger = get_logger(__name__)

# Type aliases
EmbeddingArray: TypeAlias = NDArray[np.float32]
ClusterLabels: TypeAlias = NDArray[np.int32]


class ClusteringAlgorithmType(str, Enum):
    """Supported clustering algorithms."""

    HDBSCAN = "hdbscan"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"


@dataclass
class ClusterStats:
    """Statistics for clustering operations."""

    total_clustered: int = 0
    n_clusters_found: int = 0
    n_noise_points: int = 0
    clustering_time_seconds: float = 0.0
    silhouette_score: float | None = None
    last_cluster_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging."""
        return {
            "total_clustered": self.total_clustered,
            "n_clusters_found": self.n_clusters_found,
            "n_noise_points": self.n_noise_points,
            "clustering_time_seconds": round(self.clustering_time_seconds, 3),
            "silhouette_score": (
                round(self.silhouette_score, 3) if self.silhouette_score else None
            ),
            "last_cluster_time": (
                self.last_cluster_time.isoformat() if self.last_cluster_time else None
            ),
        }


@dataclass
class ClusterSummary:
    """
    Summary information for a single cluster.

    Contains representative samples, statistics, and metadata
    for understanding the cluster's content.
    """

    id: str
    label: int
    size: int
    representative_messages: list[str]
    representative_indices: list[int]
    centroid: EmbeddingArray | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    # Pattern analysis
    common_level: str | None = None
    common_source: str | None = None
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "size": self.size,
            "representative_messages": self.representative_messages,
            "representative_indices": self.representative_indices,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "common_level": self.common_level,
            "common_source": self.common_source,
            "time_range_start": (
                self.time_range_start.isoformat() if self.time_range_start else None
            ),
            "time_range_end": (self.time_range_end.isoformat() if self.time_range_end else None),
        }


@dataclass
class ClusteringResult:
    """Result of a clustering operation."""

    labels: ClusterLabels
    n_clusters: int
    n_noise: int
    summaries: list[ClusterSummary]
    clustering_time_seconds: float
    algorithm: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "summaries": [s.to_dict() for s in self.summaries],
            "clustering_time_seconds": round(self.clustering_time_seconds, 3),
            "algorithm": self.algorithm,
            "parameters": self.parameters,
        }


class ClusteringAlgorithm(ABC):
    """Abstract base class for clustering algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name."""
        pass

    @abstractmethod
    def fit_predict(self, embeddings: EmbeddingArray) -> ClusterLabels:
        """
        Fit the clustering model and predict cluster labels.

        Args:
            embeddings: Array of embeddings with shape (n_samples, n_features).

        Returns:
            Array of cluster labels. -1 indicates noise/outliers.
        """
        pass

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return the algorithm parameters."""
        pass


class HDBSCANAlgorithm(ClusteringAlgorithm):
    """
    HDBSCAN (Hierarchical Density-Based Spatial Clustering).

    HDBSCAN is particularly well-suited for log clustering because:
    - Automatically determines the number of clusters
    - Handles variable density clusters
    - Robust to noise and outliers
    - No need to pre-specify cluster count
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
    ) -> None:
        """
        Initialize HDBSCAN algorithm.

        Args:
            min_cluster_size: Minimum number of samples in a cluster.
            min_samples: Number of samples in a neighborhood for core points.
            cluster_selection_epsilon: Distance threshold for cluster selection.
            metric: Distance metric to use.
        """
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples or min_cluster_size
        self._cluster_selection_epsilon = cluster_selection_epsilon
        self._metric = metric
        self._clusterer: Any = None

        logger.info(
            "hdbscan_algorithm_initialized",
            min_cluster_size=min_cluster_size,
            min_samples=self._min_samples,
            metric=metric,
        )

    @property
    def name(self) -> str:
        """Return the algorithm name."""
        return "hdbscan"

    def fit_predict(self, embeddings: EmbeddingArray) -> ClusterLabels:
        """Fit HDBSCAN and predict cluster labels."""
        try:
            import hdbscan
        except ImportError as e:
            msg = "hdbscan not installed. Install with: pip install hdbscan"
            logger.error("hdbscan_import_error", error=msg)
            raise ProcessingError.clustering_failed(len(embeddings), msg) from e

        if embeddings.shape[0] < self._min_cluster_size:
            logger.warning(
                "insufficient_samples_for_clustering",
                n_samples=embeddings.shape[0],
                min_cluster_size=self._min_cluster_size,
            )
            # Return all as noise
            return np.full(embeddings.shape[0], -1, dtype=np.int32)

        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
            cluster_selection_epsilon=self._cluster_selection_epsilon,
            metric=self._metric,
            core_dist_n_jobs=-1,  # Use all cores
        )

        labels = self._clusterer.fit_predict(embeddings)

        logger.debug(
            "hdbscan_clustering_complete",
            n_samples=embeddings.shape[0],
            n_clusters=len(set(labels)) - (1 if -1 in labels else 0),
            n_noise=int(np.sum(labels == -1)),
        )

        return np.asarray(labels, dtype=np.int32)

    def get_parameters(self) -> dict[str, Any]:
        """Return algorithm parameters."""
        return {
            "min_cluster_size": self._min_cluster_size,
            "min_samples": self._min_samples,
            "cluster_selection_epsilon": self._cluster_selection_epsilon,
            "metric": self._metric,
        }


class MockClusteringAlgorithm(ClusteringAlgorithm):
    """
    Mock clustering algorithm for testing.

    Assigns labels based on simple vector quantization.
    """

    def __init__(self, n_clusters: int = 5, noise_ratio: float = 0.1) -> None:
        """
        Initialize mock clustering.

        Args:
            n_clusters: Number of clusters to create.
            noise_ratio: Ratio of points to mark as noise.
        """
        self._n_clusters = n_clusters
        self._noise_ratio = noise_ratio

        logger.debug(
            "mock_clustering_initialized",
            n_clusters=n_clusters,
            noise_ratio=noise_ratio,
        )

    @property
    def name(self) -> str:
        """Return the algorithm name."""
        return "mock"

    def fit_predict(self, embeddings: EmbeddingArray) -> ClusterLabels:
        """Generate mock cluster labels."""
        n_samples = embeddings.shape[0]

        if n_samples == 0:
            return np.array([], dtype=np.int32)

        # Use first component of embedding for deterministic assignment
        first_component = embeddings[:, 0] if embeddings.shape[1] > 0 else np.zeros(n_samples)

        # Quantize to cluster labels
        labels = (
            np.digitize(
                first_component,
                np.linspace(first_component.min(), first_component.max(), self._n_clusters),
            )
            - 1
        )
        labels = np.clip(labels, 0, self._n_clusters - 1)

        # Add noise
        n_noise = int(n_samples * self._noise_ratio)
        if n_noise > 0:
            noise_indices = np.random.choice(n_samples, size=n_noise, replace=False)
            labels[noise_indices] = -1

        return np.asarray(labels, dtype=np.int32)

    def get_parameters(self) -> dict[str, Any]:
        """Return algorithm parameters."""
        return {
            "n_clusters": self._n_clusters,
            "noise_ratio": self._noise_ratio,
        }


@dataclass
class ClusteringService:
    """
    High-level service for log clustering operations.

    Provides a unified interface for clustering log embeddings with:
    - Configurable clustering algorithms
    - Cluster summary generation
    - Representative sample selection
    - Statistics tracking

    Usage:
        service = ClusteringService.from_config()
        result = service.cluster(embeddings, records=log_records)
        for summary in result.summaries:
            print(f"Cluster {summary.label}: {summary.size} logs")
    """

    algorithm: ClusteringAlgorithm
    stats: ClusterStats = field(default_factory=ClusterStats)
    n_representative_samples: int = 5

    @classmethod
    def from_config(
        cls,
        config: ClusteringConfig | None = None,
        use_mock: bool = False,
    ) -> ClusteringService:
        """
        Create a clustering service from configuration.

        Args:
            config: Clustering configuration. If None, loads from global config.
            use_mock: If True, use mock algorithm for testing.

        Returns:
            Configured ClusteringService instance.
        """
        if config is None:
            config = get_config().clustering

        if use_mock:
            algorithm: ClusteringAlgorithm = MockClusteringAlgorithm()
        else:
            algorithm = HDBSCANAlgorithm(
                min_cluster_size=config.min_cluster_size,
                min_samples=config.min_samples,
                cluster_selection_epsilon=config.cluster_selection_epsilon,
                metric=config.metric,
            )

        logger.info(
            "clustering_service_created",
            algorithm=algorithm.name,
            config=config.model_dump(),
        )

        return cls(algorithm=algorithm)

    def cluster(
        self,
        embeddings: EmbeddingArray,
        records: Sequence[LogRecord] | None = None,
        messages: list[str] | None = None,
    ) -> ClusteringResult:
        """
        Cluster embeddings and generate summaries.

        Args:
            embeddings: Array of embeddings with shape (n_samples, n_features).
            records: Optional log records for enhanced summaries.
            messages: Optional list of messages (used if records not provided).

        Returns:
            ClusteringResult with labels and cluster summaries.

        Raises:
            ProcessingError: If clustering fails.
        """
        start_time = time.time()

        if embeddings.shape[0] == 0:
            logger.warning("clustering_empty_input")
            return ClusteringResult(
                labels=np.array([], dtype=np.int32),
                n_clusters=0,
                n_noise=0,
                summaries=[],
                clustering_time_seconds=0.0,
                algorithm=self.algorithm.name,
                parameters=self.algorithm.get_parameters(),
            )

        try:
            # Perform clustering
            labels = self.algorithm.fit_predict(embeddings)

            # Calculate cluster statistics
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = int(np.sum(labels == -1))

            # Generate summaries for each cluster
            summaries = self._generate_summaries(
                embeddings=embeddings,
                labels=labels,
                records=records,
                messages=messages,
            )

            clustering_time = time.time() - start_time

            # Update stats
            self.stats.total_clustered += embeddings.shape[0]
            self.stats.n_clusters_found = n_clusters
            self.stats.n_noise_points = n_noise
            self.stats.clustering_time_seconds += clustering_time
            self.stats.last_cluster_time = datetime.now(timezone.utc)

            logger.info(
                "clustering_complete",
                n_samples=embeddings.shape[0],
                n_clusters=n_clusters,
                n_noise=n_noise,
                time_seconds=round(clustering_time, 3),
            )

            return ClusteringResult(
                labels=labels,
                n_clusters=n_clusters,
                n_noise=n_noise,
                summaries=summaries,
                clustering_time_seconds=clustering_time,
                algorithm=self.algorithm.name,
                parameters=self.algorithm.get_parameters(),
            )

        except Exception as e:
            logger.error(
                "clustering_failed",
                n_samples=embeddings.shape[0],
                error=str(e),
            )
            raise ProcessingError.clustering_failed(embeddings.shape[0], str(e)) from e

    def _generate_summaries(
        self,
        embeddings: EmbeddingArray,
        labels: ClusterLabels,
        records: Sequence[LogRecord] | None = None,
        messages: list[str] | None = None,
    ) -> list[ClusterSummary]:
        """Generate cluster summaries with representative samples."""
        summaries = []
        unique_labels = sorted(set(labels))

        for label in unique_labels:
            if label == -1:
                # Skip noise cluster
                continue

            # Get indices of samples in this cluster
            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]
            cluster_size = len(cluster_indices)

            # Get cluster embeddings
            cluster_embeddings = embeddings[cluster_mask]

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0).astype(np.float32)

            # Select representative samples (closest to centroid)
            representative_indices = self._select_representatives(
                cluster_embeddings,
                centroid,
                cluster_indices,
            )

            # Get representative messages
            representative_messages = []
            if records is not None:
                for idx in representative_indices:
                    if idx < len(records):
                        msg = records[idx].normalized or records[idx].message
                        representative_messages.append(msg)
            elif messages is not None:
                for idx in representative_indices:
                    if idx < len(messages):
                        representative_messages.append(messages[idx])

            # Extract metadata from records
            common_level = None
            common_source = None
            time_range_start = None
            time_range_end = None

            if records is not None:
                cluster_records = [records[i] for i in cluster_indices if i < len(records)]
                if cluster_records:
                    # Find most common level
                    levels = [r.level for r in cluster_records if r.level]
                    if levels:
                        from collections import Counter

                        level_counts = Counter(levels)
                        common_level = level_counts.most_common(1)[0][0]

                    # Find most common source
                    sources = [r.source for r in cluster_records if r.source]
                    if sources:
                        from collections import Counter

                        source_counts = Counter(sources)
                        common_source = source_counts.most_common(1)[0][0]

                    # Get time range
                    timestamps = [r.timestamp for r in cluster_records if r.timestamp]
                    if timestamps:
                        time_range_start = min(timestamps)
                        time_range_end = max(timestamps)

            summary = ClusterSummary(
                id=str(uuid.uuid4()),
                label=int(label),
                size=cluster_size,
                representative_messages=representative_messages,
                representative_indices=[int(i) for i in representative_indices],
                centroid=centroid,
                common_level=common_level,
                common_source=common_source,
                time_range_start=time_range_start,
                time_range_end=time_range_end,
            )
            summaries.append(summary)

            logger.debug(
                "cluster_summary_generated",
                cluster_label=label,
                size=cluster_size,
                n_representatives=len(representative_messages),
            )

        return summaries

    def _select_representatives(
        self,
        cluster_embeddings: EmbeddingArray,
        centroid: EmbeddingArray,
        cluster_indices: NDArray[np.int64],
    ) -> list[int]:
        """Select representative samples closest to centroid."""
        # Calculate distances to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

        # Get indices of closest samples
        n_samples = min(self.n_representative_samples, len(cluster_indices))
        closest_local_indices = np.argsort(distances)[:n_samples]

        # Map back to original indices
        return [int(cluster_indices[i]) for i in closest_local_indices]

    def recluster_with_params(
        self,
        embeddings: EmbeddingArray,
        min_cluster_size: int,
        min_samples: int | None = None,
        records: Sequence[LogRecord] | None = None,
    ) -> ClusteringResult:
        """
        Re-cluster with custom parameters.

        Useful for exploring different clustering granularities.

        Args:
            embeddings: Array of embeddings.
            min_cluster_size: Minimum cluster size.
            min_samples: Minimum samples for core points.
            records: Optional log records.

        Returns:
            ClusteringResult with new clustering.
        """
        # Create temporary algorithm with custom params
        temp_algorithm = HDBSCANAlgorithm(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        # Swap algorithm temporarily
        original_algorithm = self.algorithm
        self.algorithm = temp_algorithm

        try:
            result = self.cluster(embeddings, records=records)
        finally:
            # Restore original algorithm
            self.algorithm = original_algorithm

        return result


# Module-level singleton
_clustering_service: ClusteringService | None = None


def get_clustering_service() -> ClusteringService:
    """Get the global clustering service instance."""
    global _clustering_service
    if _clustering_service is None:
        _clustering_service = ClusteringService.from_config()
    return _clustering_service


def set_clustering_service(service: ClusteringService) -> None:
    """Set the global clustering service instance."""
    global _clustering_service
    _clustering_service = service


def reset_clustering_service() -> None:
    """Reset the global clustering service instance."""
    global _clustering_service
    _clustering_service = None
