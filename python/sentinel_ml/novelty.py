"""
Novelty detection for identifying unusual log patterns.

This module provides k-NN density-based novelty detection for identifying
log messages that significantly deviate from established patterns.

Design Patterns:
- Strategy Pattern: Pluggable novelty detection algorithms (k-NN, LOF, Isolation Forest)
- Factory Pattern: Service creation with configuration
- Template Method: Common detection workflow with customizable steps
- Observer Pattern: Hooks for novelty detection events

SOLID Principles:
- Single Responsibility: Each class handles one concern
- Open/Closed: Extensible via NoveltyDetector interface
- Liskov Substitution: All detectors implement same interface
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

from sentinel_ml.config import NoveltyConfig, get_config
from sentinel_ml.exceptions import ProcessingError
from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sentinel_ml.models import LogRecord

logger = get_logger(__name__)

# Type aliases for clarity
EmbeddingArray: TypeAlias = NDArray[np.float32]
NoveltyScores: TypeAlias = NDArray[np.float32]


class NoveltyAlgorithmType(str, Enum):
    """Supported novelty detection algorithms."""

    KNN_DENSITY = "knn_density"
    LOCAL_OUTLIER_FACTOR = "lof"
    ISOLATION_FOREST = "isolation_forest"


@dataclass
class NoveltyStats:
    """Statistics for novelty detection operations."""

    total_analyzed: int = 0
    total_novel_detected: int = 0
    total_normal_detected: int = 0
    avg_novelty_score: float = 0.0
    max_novelty_score: float = 0.0
    min_novelty_score: float = 1.0
    detection_time_seconds: float = 0.0
    last_detection_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging and serialization."""
        return {
            "total_analyzed": self.total_analyzed,
            "total_novel_detected": self.total_novel_detected,
            "total_normal_detected": self.total_normal_detected,
            "avg_novelty_score": round(self.avg_novelty_score, 4),
            "max_novelty_score": round(self.max_novelty_score, 4),
            "min_novelty_score": round(self.min_novelty_score, 4),
            "detection_time_seconds": round(self.detection_time_seconds, 3),
            "last_detection_time": (
                self.last_detection_time.isoformat() if self.last_detection_time else None
            ),
            "novelty_rate": (
                round(self.total_novel_detected / self.total_analyzed, 4)
                if self.total_analyzed > 0
                else 0.0
            ),
        }


@dataclass
class NoveltyScore:
    """
    Novelty score for a single log entry.

    Contains the novelty score, whether it's flagged as novel,
    and additional context for explanation.
    """

    score: float
    is_novel: bool
    index: int
    k_neighbor_distances: list[float] | None = None
    density: float | None = None
    message: str | None = None
    record_id: str | None = None
    explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "score": round(self.score, 4),
            "is_novel": self.is_novel,
            "index": self.index,
        }
        if self.k_neighbor_distances:
            result["k_neighbor_distances"] = [round(d, 4) for d in self.k_neighbor_distances]
        if self.density is not None:
            result["density"] = round(self.density, 4)
        if self.message:
            result["message"] = self.message
        if self.record_id:
            result["record_id"] = self.record_id
        if self.explanation:
            result["explanation"] = self.explanation
        return result


@dataclass
class NoveltyResult:
    """Result of a novelty detection operation."""

    scores: NoveltyScores
    is_novel: NDArray[np.bool_]
    novel_scores: list[NoveltyScore]
    n_novel: int
    n_normal: int
    threshold: float
    detection_time_seconds: float
    algorithm: str
    parameters: dict[str, Any]
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_id": self.batch_id,
            "n_samples": len(self.scores),
            "n_novel": self.n_novel,
            "n_normal": self.n_normal,
            "novelty_rate": round(self.n_novel / len(self.scores), 4) if len(self.scores) > 0 else 0.0,
            "threshold": self.threshold,
            "detection_time_seconds": round(self.detection_time_seconds, 3),
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "novel_samples": [s.to_dict() for s in self.novel_scores],
            "score_stats": {
                "mean": round(float(np.mean(self.scores)), 4) if len(self.scores) > 0 else 0.0,
                "std": round(float(np.std(self.scores)), 4) if len(self.scores) > 0 else 0.0,
                "min": round(float(np.min(self.scores)), 4) if len(self.scores) > 0 else 0.0,
                "max": round(float(np.max(self.scores)), 4) if len(self.scores) > 0 else 0.0,
            },
        }

    def get_novel_indices(self) -> list[int]:
        """Get indices of novel samples."""
        return [int(i) for i in np.where(self.is_novel)[0]]


class NoveltyDetector(ABC):
    """
    Abstract base class for novelty detection algorithms.

    Implements the Strategy pattern for pluggable detection algorithms.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name."""
        pass

    @abstractmethod
    def fit(self, embeddings: EmbeddingArray) -> None:
        """
        Fit the detector on reference embeddings.

        Args:
            embeddings: Reference embeddings with shape (n_samples, n_features).
        """
        pass

    @abstractmethod
    def score(self, embeddings: EmbeddingArray) -> NoveltyScores:
        """
        Compute novelty scores for embeddings.

        Args:
            embeddings: Embeddings to score with shape (n_samples, n_features).

        Returns:
            Array of novelty scores in range [0, 1].
            Higher scores indicate more novel/unusual samples.
        """
        pass

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return the algorithm parameters."""
        pass

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        pass


class KNNNoveltyDetector(NoveltyDetector):
    """
    k-Nearest Neighbors density-based novelty detector.

    Novelty is determined by measuring the density around each sample
    using k-NN distances. Samples in low-density regions (far from
    their neighbors) are considered novel.

    Algorithm:
    1. Compute k-NN distances for each sample
    2. Calculate local density estimate (inverse of mean k-NN distance)
    3. Compare density to reference distribution
    4. Score = 1 - (normalized density), higher = more novel

    This approach is particularly effective for log data because:
    - No assumption about cluster shape or distribution
    - Robust to varying densities in embedding space
    - Interpretable: novel = far from known patterns
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        use_density: bool = True,
        distance_metric: str = "euclidean",
    ) -> None:
        """
        Initialize k-NN novelty detector.

        Args:
            k_neighbors: Number of neighbors for density estimation.
            use_density: If True, use density-based scoring. Otherwise, use raw distances.
            distance_metric: Distance metric for neighbor search.
        """
        self._k_neighbors = k_neighbors
        self._use_density = use_density
        self._distance_metric = distance_metric
        self._reference_embeddings: EmbeddingArray | None = None
        self._reference_densities: NDArray[np.float32] | None = None
        self._density_mean: float = 0.0
        self._density_std: float = 1.0
        self._fitted = False

        logger.info(
            "knn_novelty_detector_initialized",
            k_neighbors=k_neighbors,
            use_density=use_density,
            distance_metric=distance_metric,
        )

    @property
    def name(self) -> str:
        """Return the algorithm name."""
        return "knn_density"

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._fitted

    def fit(self, embeddings: EmbeddingArray) -> None:
        """
        Fit the detector on reference embeddings.

        Computes the density distribution of the reference set for
        comparison during scoring.

        Args:
            embeddings: Reference embeddings with shape (n_samples, n_features).

        Raises:
            ProcessingError: If fitting fails.
        """
        if embeddings.shape[0] == 0:
            logger.warning("knn_novelty_fit_empty_input")
            self._fitted = False
            return

        if embeddings.shape[0] < self._k_neighbors:
            logger.warning(
                "knn_novelty_insufficient_samples",
                n_samples=embeddings.shape[0],
                k_neighbors=self._k_neighbors,
            )
            # Adjust k to available samples
            effective_k = max(1, embeddings.shape[0] - 1)
            logger.info(
                "knn_novelty_adjusted_k",
                original_k=self._k_neighbors,
                effective_k=effective_k,
            )
        else:
            effective_k = self._k_neighbors

        try:
            self._reference_embeddings = embeddings.copy()

            # Compute k-NN distances for reference set
            distances = self._compute_knn_distances(embeddings, effective_k)

            if self._use_density:
                # Compute densities (inverse of mean k-NN distance)
                mean_distances = np.mean(distances, axis=1)
                # Avoid division by zero
                self._reference_densities = 1.0 / (mean_distances + 1e-10)
                self._reference_densities = self._reference_densities.astype(np.float32)
                self._density_mean = float(np.mean(self._reference_densities))
                self._density_std = float(np.std(self._reference_densities))
                if self._density_std < 1e-10:
                    self._density_std = 1.0
            else:
                self._reference_densities = np.mean(distances, axis=1).astype(np.float32)
                self._density_mean = float(np.mean(self._reference_densities))
                self._density_std = float(np.std(self._reference_densities))
                if self._density_std < 1e-10:
                    self._density_std = 1.0

            self._fitted = True

            logger.info(
                "knn_novelty_detector_fitted",
                n_samples=embeddings.shape[0],
                n_features=embeddings.shape[1],
                density_mean=round(self._density_mean, 4),
                density_std=round(self._density_std, 4),
            )

        except Exception as e:
            logger.error(
                "knn_novelty_fit_failed",
                n_samples=embeddings.shape[0],
                error=str(e),
            )
            raise ProcessingError(
                message=f"Failed to fit k-NN novelty detector: {e}",
                context={"n_samples": embeddings.shape[0], "error": str(e)},
            ) from e

    def score(self, embeddings: EmbeddingArray) -> NoveltyScores:
        """
        Compute novelty scores for embeddings.

        Scores are normalized to [0, 1] range where:
        - 0.0 = completely normal (high density, close to reference)
        - 1.0 = highly novel (low density, far from reference)

        Args:
            embeddings: Embeddings to score with shape (n_samples, n_features).

        Returns:
            Array of novelty scores.

        Raises:
            ProcessingError: If detector not fitted or scoring fails.
        """
        if not self._fitted or self._reference_embeddings is None:
            msg = "Detector must be fitted before scoring"
            logger.error("knn_novelty_not_fitted")
            raise ProcessingError(message=msg, context={"fitted": self._fitted})

        if embeddings.shape[0] == 0:
            return np.array([], dtype=np.float32)

        try:
            effective_k = min(self._k_neighbors, self._reference_embeddings.shape[0])

            # Compute distances to reference embeddings
            distances = self._compute_cross_knn_distances(
                embeddings, self._reference_embeddings, effective_k
            )

            if self._use_density:
                # Compute density for new samples
                mean_distances = np.mean(distances, axis=1)
                densities = 1.0 / (mean_distances + 1e-10)

                # Normalize using reference distribution
                z_scores = (densities - self._density_mean) / self._density_std

                # Convert to novelty score: low density = high novelty
                # Use sigmoid to map z-scores to [0, 1]
                scores = 1.0 / (1.0 + np.exp(z_scores))
            else:
                # Use raw distance as novelty score
                mean_distances = np.mean(distances, axis=1)
                z_scores = (mean_distances - self._density_mean) / self._density_std

                # High distance = high novelty
                scores = 1.0 / (1.0 + np.exp(-z_scores))

            # Clip to valid range
            scores = np.clip(scores, 0.0, 1.0).astype(np.float32)

            logger.debug(
                "knn_novelty_scores_computed",
                n_samples=embeddings.shape[0],
                mean_score=round(float(np.mean(scores)), 4),
                max_score=round(float(np.max(scores)), 4),
            )

            return scores

        except Exception as e:
            logger.error(
                "knn_novelty_scoring_failed",
                n_samples=embeddings.shape[0],
                error=str(e),
            )
            raise ProcessingError(
                message=f"Failed to compute novelty scores: {e}",
                context={"n_samples": embeddings.shape[0], "error": str(e)},
            ) from e

    def _compute_knn_distances(
        self, embeddings: EmbeddingArray, k: int
    ) -> NDArray[np.float32]:
        """
        Compute k-NN distances within a set of embeddings.

        Args:
            embeddings: Embeddings with shape (n_samples, n_features).
            k: Number of neighbors.

        Returns:
            Array of k-NN distances with shape (n_samples, k).
        """
        n_samples = embeddings.shape[0]

        # Compute all pairwise distances
        # Using squared Euclidean for efficiency
        diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
        sq_distances = np.sum(diff**2, axis=2)

        # Set diagonal to infinity to exclude self
        np.fill_diagonal(sq_distances, np.inf)

        # Get k smallest distances
        k_neighbors = min(k, n_samples - 1)
        if k_neighbors <= 0:
            return np.zeros((n_samples, 1), dtype=np.float32)

        # Partition to find k smallest
        indices = np.argpartition(sq_distances, k_neighbors, axis=1)[:, :k_neighbors]
        knn_sq_distances = np.take_along_axis(sq_distances, indices, axis=1)

        # Return actual distances (not squared)
        return np.sqrt(knn_sq_distances).astype(np.float32)

    def _compute_cross_knn_distances(
        self,
        query_embeddings: EmbeddingArray,
        reference_embeddings: EmbeddingArray,
        k: int,
    ) -> NDArray[np.float32]:
        """
        Compute k-NN distances from query to reference embeddings.

        Args:
            query_embeddings: Query embeddings with shape (n_queries, n_features).
            reference_embeddings: Reference embeddings with shape (n_ref, n_features).
            k: Number of neighbors.

        Returns:
            Array of k-NN distances with shape (n_queries, k).
        """
        n_queries = query_embeddings.shape[0]
        n_ref = reference_embeddings.shape[0]

        # Compute all pairwise distances
        diff = query_embeddings[:, np.newaxis, :] - reference_embeddings[np.newaxis, :, :]
        sq_distances = np.sum(diff**2, axis=2)

        # Get k smallest distances
        k_neighbors = min(k, n_ref)
        if k_neighbors <= 0:
            return np.zeros((n_queries, 1), dtype=np.float32)

        # Partition to find k smallest
        indices = np.argpartition(sq_distances, k_neighbors - 1, axis=1)[:, :k_neighbors]
        knn_sq_distances = np.take_along_axis(sq_distances, indices, axis=1)

        return np.sqrt(knn_sq_distances).astype(np.float32)

    def get_parameters(self) -> dict[str, Any]:
        """Return algorithm parameters."""
        return {
            "k_neighbors": self._k_neighbors,
            "use_density": self._use_density,
            "distance_metric": self._distance_metric,
            "is_fitted": self._fitted,
            "n_reference_samples": (
                self._reference_embeddings.shape[0]
                if self._reference_embeddings is not None
                else 0
            ),
        }


class MockNoveltyDetector(NoveltyDetector):
    """
    Mock novelty detector for testing.

    Produces deterministic scores based on embedding values.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        novelty_ratio: float = 0.1,
    ) -> None:
        """
        Initialize mock novelty detector.

        Args:
            threshold: Threshold for novelty classification.
            novelty_ratio: Expected ratio of novel samples.
        """
        self._threshold = threshold
        self._novelty_ratio = novelty_ratio
        self._fitted = False

        logger.debug(
            "mock_novelty_detector_initialized",
            threshold=threshold,
            novelty_ratio=novelty_ratio,
        )

    @property
    def name(self) -> str:
        """Return the algorithm name."""
        return "mock"

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._fitted

    def fit(self, embeddings: EmbeddingArray) -> None:
        """Fit the mock detector (no-op for mock)."""
        self._fitted = True
        logger.debug(
            "mock_novelty_detector_fitted",
            n_samples=embeddings.shape[0] if embeddings.shape[0] > 0 else 0,
        )

    def score(self, embeddings: EmbeddingArray) -> NoveltyScores:
        """
        Generate deterministic mock novelty scores.

        Uses the L2 norm of embeddings to generate consistent scores.
        """
        if not self._fitted:
            msg = "Detector must be fitted before scoring"
            logger.error("mock_novelty_not_fitted")
            raise ProcessingError(message=msg, context={"fitted": self._fitted})

        if embeddings.shape[0] == 0:
            return np.array([], dtype=np.float32)

        # Use L2 norm for deterministic scoring
        norms = np.linalg.norm(embeddings, axis=1)

        # Normalize to [0, 1] range
        if norms.max() > norms.min():
            scores = (norms - norms.min()) / (norms.max() - norms.min())
        else:
            scores = np.full(len(norms), 0.5)

        return scores.astype(np.float32)

    def get_parameters(self) -> dict[str, Any]:
        """Return algorithm parameters."""
        return {
            "threshold": self._threshold,
            "novelty_ratio": self._novelty_ratio,
            "is_fitted": self._fitted,
        }


@dataclass
class NoveltyService:
    """
    High-level service for novelty detection operations.

    Provides a unified interface for detecting novel log patterns with:
    - Configurable detection algorithms
    - Automatic threshold-based classification
    - Detailed novelty explanations
    - Statistics tracking

    Usage:
        service = NoveltyService.from_config()
        service.fit(reference_embeddings)
        result = service.detect(new_embeddings, messages=log_messages)
        for novel in result.novel_scores:
            print(f"Novel log: {novel.message} (score: {novel.score})")
    """

    detector: NoveltyDetector
    threshold: float
    stats: NoveltyStats = field(default_factory=NoveltyStats)

    @classmethod
    def from_config(
        cls,
        config: NoveltyConfig | None = None,
        use_mock: bool = False,
    ) -> NoveltyService:
        """
        Create a novelty service from configuration.

        Args:
            config: Novelty configuration. If None, loads from global config.
            use_mock: If True, use mock detector for testing.

        Returns:
            Configured NoveltyService instance.
        """
        if config is None:
            config = get_config().novelty

        if use_mock:
            detector: NoveltyDetector = MockNoveltyDetector(
                threshold=config.threshold,
            )
        else:
            detector = KNNNoveltyDetector(
                k_neighbors=config.k_neighbors,
                use_density=config.use_density,
            )

        logger.info(
            "novelty_service_created",
            algorithm=detector.name,
            threshold=config.threshold,
            config=config.model_dump(),
        )

        return cls(
            detector=detector,
            threshold=config.threshold,
        )

    def fit(
        self,
        embeddings: EmbeddingArray,
        records: Sequence[LogRecord] | None = None,
    ) -> None:
        """
        Fit the novelty detector on reference embeddings.

        Args:
            embeddings: Reference embeddings representing normal patterns.
            records: Optional log records for context.
        """
        start_time = time.time()

        if embeddings.shape[0] == 0:
            logger.warning("novelty_fit_empty_input")
            return

        self.detector.fit(embeddings)

        fit_time = time.time() - start_time

        logger.info(
            "novelty_detector_fitted",
            n_samples=embeddings.shape[0],
            algorithm=self.detector.name,
            fit_time_seconds=round(fit_time, 3),
        )

    def detect(
        self,
        embeddings: EmbeddingArray,
        records: Sequence[LogRecord] | None = None,
        messages: list[str] | None = None,
        threshold: float | None = None,
    ) -> NoveltyResult:
        """
        Detect novel samples in embeddings.

        Args:
            embeddings: Embeddings to analyze with shape (n_samples, n_features).
            records: Optional log records for enhanced results.
            messages: Optional list of messages (used if records not provided).
            threshold: Override default threshold. None uses service threshold.

        Returns:
            NoveltyResult with scores and novel sample details.

        Raises:
            ProcessingError: If detection fails.
        """
        start_time = time.time()
        effective_threshold = threshold if threshold is not None else self.threshold

        if embeddings.shape[0] == 0:
            logger.warning("novelty_detect_empty_input")
            return NoveltyResult(
                scores=np.array([], dtype=np.float32),
                is_novel=np.array([], dtype=np.bool_),
                novel_scores=[],
                n_novel=0,
                n_normal=0,
                threshold=effective_threshold,
                detection_time_seconds=0.0,
                algorithm=self.detector.name,
                parameters=self.detector.get_parameters(),
            )

        if not self.detector.is_fitted:
            msg = "Detector must be fitted before detection"
            logger.error("novelty_detector_not_fitted")
            raise ProcessingError(
                message=msg,
                context={"fitted": self.detector.is_fitted},
            )

        try:
            # Compute novelty scores
            scores = self.detector.score(embeddings)

            # Classify based on threshold
            is_novel = scores >= effective_threshold

            # Build detailed scores for novel samples
            novel_scores = self._build_novel_scores(
                scores=scores,
                is_novel=is_novel,
                records=records,
                messages=messages,
                threshold=effective_threshold,
            )

            n_novel = int(np.sum(is_novel))
            n_normal = len(scores) - n_novel

            detection_time = time.time() - start_time

            # Update statistics
            self._update_stats(scores, n_novel, n_normal, detection_time)

            logger.info(
                "novelty_detection_complete",
                n_samples=len(scores),
                n_novel=n_novel,
                n_normal=n_normal,
                novelty_rate=round(n_novel / len(scores), 4) if len(scores) > 0 else 0.0,
                threshold=effective_threshold,
                time_seconds=round(detection_time, 3),
            )

            return NoveltyResult(
                scores=scores,
                is_novel=is_novel,
                novel_scores=novel_scores,
                n_novel=n_novel,
                n_normal=n_normal,
                threshold=effective_threshold,
                detection_time_seconds=detection_time,
                algorithm=self.detector.name,
                parameters=self.detector.get_parameters(),
            )

        except ProcessingError:
            raise
        except Exception as e:
            logger.error(
                "novelty_detection_failed",
                n_samples=embeddings.shape[0],
                error=str(e),
            )
            raise ProcessingError(
                message=f"Novelty detection failed: {e}",
                context={"n_samples": embeddings.shape[0], "error": str(e)},
            ) from e

    def score_batch(
        self,
        embeddings: EmbeddingArray,
    ) -> NoveltyScores:
        """
        Compute novelty scores without classification.

        Useful when you need raw scores for custom thresholding
        or visualization.

        Args:
            embeddings: Embeddings to score.

        Returns:
            Array of novelty scores.
        """
        if not self.detector.is_fitted:
            msg = "Detector must be fitted before scoring"
            logger.error("novelty_detector_not_fitted_for_batch")
            raise ProcessingError(
                message=msg,
                context={"fitted": self.detector.is_fitted},
            )

        return self.detector.score(embeddings)

    def _build_novel_scores(
        self,
        scores: NoveltyScores,
        is_novel: NDArray[np.bool_],
        records: Sequence[LogRecord] | None = None,
        messages: list[str] | None = None,
        threshold: float = 0.7,
    ) -> list[NoveltyScore]:
        """Build detailed NoveltyScore objects for novel samples."""
        novel_scores = []
        novel_indices = np.where(is_novel)[0]

        for idx in novel_indices:
            score = float(scores[idx])

            # Get message
            message = None
            record_id = None
            if records is not None and idx < len(records):
                message = records[idx].normalized or records[idx].message
                record_id = records[idx].id
            elif messages is not None and idx < len(messages):
                message = messages[idx]

            # Generate explanation
            explanation = self._generate_explanation(score, threshold)

            novel_scores.append(
                NoveltyScore(
                    score=score,
                    is_novel=True,
                    index=int(idx),
                    message=message,
                    record_id=record_id,
                    explanation=explanation,
                )
            )

        # Sort by score descending (most novel first)
        novel_scores.sort(key=lambda x: x.score, reverse=True)

        return novel_scores

    def _generate_explanation(self, score: float, threshold: float) -> str:
        """Generate human-readable explanation for novelty score."""
        margin = score - threshold

        if score >= 0.9:
            severity = "highly unusual"
        elif score >= 0.8:
            severity = "unusual"
        elif score >= 0.7:
            severity = "moderately unusual"
        else:
            severity = "slightly unusual"

        return (
            f"This log pattern is {severity} (score: {score:.3f}, "
            f"threshold: {threshold:.3f}, margin: {margin:.3f}). "
            f"It appears in a low-density region of the embedding space, "
            f"indicating deviation from established patterns."
        )

    def _update_stats(
        self,
        scores: NoveltyScores,
        n_novel: int,
        n_normal: int,
        detection_time: float,
    ) -> None:
        """Update service statistics."""
        n_samples = len(scores)

        # Incremental average update
        total_before = self.stats.total_analyzed
        total_after = total_before + n_samples

        if total_after > 0:
            self.stats.avg_novelty_score = (
                (self.stats.avg_novelty_score * total_before + float(np.sum(scores)))
                / total_after
            )

        self.stats.total_analyzed = total_after
        self.stats.total_novel_detected += n_novel
        self.stats.total_normal_detected += n_normal
        self.stats.max_novelty_score = max(
            self.stats.max_novelty_score,
            float(np.max(scores)) if len(scores) > 0 else 0.0,
        )
        self.stats.min_novelty_score = min(
            self.stats.min_novelty_score,
            float(np.min(scores)) if len(scores) > 0 else 1.0,
        )
        self.stats.detection_time_seconds += detection_time
        self.stats.last_detection_time = datetime.now(timezone.utc)

    def get_stats(self) -> NoveltyStats:
        """Get current statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset statistics to initial values."""
        self.stats = NoveltyStats()
        logger.debug("novelty_stats_reset")
