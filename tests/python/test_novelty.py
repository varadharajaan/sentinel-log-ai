"""
Comprehensive unit tests for novelty detection module.

Tests cover:
- NoveltyStats data class
- NoveltyScore data class
- NoveltyResult data class
- NoveltyDetector ABC and implementations
- KNNNoveltyDetector algorithm
- MockNoveltyDetector for testing
- NoveltyService high-level API
- Edge cases and error handling
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from sentinel_ml.config import NoveltyConfig
from sentinel_ml.exceptions import ProcessingError
from sentinel_ml.novelty import (
    KNNNoveltyDetector,
    MockNoveltyDetector,
    NoveltyAlgorithmType,
    NoveltyResult,
    NoveltyScore,
    NoveltyService,
    NoveltyStats,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(100, 64).astype(np.float32)


@pytest.fixture
def small_embeddings() -> np.ndarray:
    """Generate small set of embeddings for edge case testing."""
    np.random.seed(42)
    return np.random.randn(5, 64).astype(np.float32)


@pytest.fixture
def clustered_embeddings() -> np.ndarray:
    """Generate embeddings with clear clusters and outliers."""
    np.random.seed(42)

    # Create 3 dense clusters
    cluster1 = np.random.randn(30, 64).astype(np.float32) * 0.1 + np.array([1.0] * 64)
    cluster2 = np.random.randn(30, 64).astype(np.float32) * 0.1 + np.array([-1.0] * 64)
    cluster3 = np.random.randn(30, 64).astype(np.float32) * 0.1 + np.array([0.0] * 64)

    # Create outliers far from clusters
    outliers = np.random.randn(10, 64).astype(np.float32) * 0.1 + np.array([5.0] * 64)

    return np.vstack([cluster1, cluster2, cluster3, outliers]).astype(np.float32)


@pytest.fixture
def sample_messages() -> list[str]:
    """Sample log messages for testing."""
    return [
        "User login successful",
        "Database connection established",
        "API request processed",
        "Error: Connection timeout",
        "Warning: Memory usage high",
    ]


@pytest.fixture
def mock_log_records(sample_messages: list[str]) -> list[MagicMock]:
    """Create mock log records for testing."""
    records = []
    for i, msg in enumerate(sample_messages):
        record = MagicMock()
        record.id = f"record_{i}"
        record.message = msg
        record.normalized = f"normalized: {msg}"
        records.append(record)
    return records


@pytest.fixture
def novelty_config() -> NoveltyConfig:
    """Create a test novelty configuration."""
    return NoveltyConfig(
        threshold=0.7,
        k_neighbors=5,
        use_density=True,
    )


# ============================================================================
# NoveltyStats Tests
# ============================================================================


class TestNoveltyStats:
    """Tests for NoveltyStats data class."""

    def test_default_values(self) -> None:
        """Test default values are correctly initialized."""
        stats = NoveltyStats()

        assert stats.total_analyzed == 0
        assert stats.total_novel_detected == 0
        assert stats.total_normal_detected == 0
        assert stats.avg_novelty_score == 0.0
        assert stats.max_novelty_score == 0.0
        assert stats.min_novelty_score == 1.0
        assert stats.detection_time_seconds == 0.0
        assert stats.last_detection_time is None

    def test_custom_values(self) -> None:
        """Test custom values initialization."""
        now = datetime.now(timezone.utc)
        stats = NoveltyStats(
            total_analyzed=100,
            total_novel_detected=10,
            total_normal_detected=90,
            avg_novelty_score=0.45,
            max_novelty_score=0.95,
            min_novelty_score=0.05,
            detection_time_seconds=1.5,
            last_detection_time=now,
        )

        assert stats.total_analyzed == 100
        assert stats.total_novel_detected == 10
        assert stats.total_normal_detected == 90
        assert stats.avg_novelty_score == 0.45
        assert stats.max_novelty_score == 0.95
        assert stats.min_novelty_score == 0.05
        assert stats.detection_time_seconds == 1.5
        assert stats.last_detection_time == now

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        now = datetime.now(timezone.utc)
        stats = NoveltyStats(
            total_analyzed=100,
            total_novel_detected=10,
            total_normal_detected=90,
            avg_novelty_score=0.45678,
            max_novelty_score=0.95123,
            min_novelty_score=0.05456,
            detection_time_seconds=1.5678,
            last_detection_time=now,
        )

        result = stats.to_dict()

        assert result["total_analyzed"] == 100
        assert result["total_novel_detected"] == 10
        assert result["total_normal_detected"] == 90
        assert result["avg_novelty_score"] == 0.4568  # Rounded
        assert result["max_novelty_score"] == 0.9512  # Rounded
        assert result["min_novelty_score"] == 0.0546  # Rounded
        assert result["detection_time_seconds"] == 1.568  # Rounded
        assert result["last_detection_time"] == now.isoformat()
        assert result["novelty_rate"] == 0.1  # 10/100

    def test_novelty_rate_zero_total(self) -> None:
        """Test novelty rate when no samples analyzed."""
        stats = NoveltyStats()
        result = stats.to_dict()

        assert result["novelty_rate"] == 0.0

    def test_to_dict_no_last_detection_time(self) -> None:
        """Test dictionary conversion without last detection time."""
        stats = NoveltyStats(total_analyzed=50)
        result = stats.to_dict()

        assert result["last_detection_time"] is None


# ============================================================================
# NoveltyScore Tests
# ============================================================================


class TestNoveltyScore:
    """Tests for NoveltyScore data class."""

    def test_minimal_creation(self) -> None:
        """Test creating score with minimal required fields."""
        score = NoveltyScore(
            score=0.85,
            is_novel=True,
            index=42,
        )

        assert score.score == 0.85
        assert score.is_novel is True
        assert score.index == 42
        assert score.k_neighbor_distances is None
        assert score.density is None
        assert score.message is None
        assert score.record_id is None
        assert score.explanation is None

    def test_full_creation(self) -> None:
        """Test creating score with all fields."""
        score = NoveltyScore(
            score=0.85,
            is_novel=True,
            index=42,
            k_neighbor_distances=[0.1, 0.2, 0.3],
            density=0.5,
            message="Test log message",
            record_id="record_123",
            explanation="This is unusual",
        )

        assert score.score == 0.85
        assert score.k_neighbor_distances == [0.1, 0.2, 0.3]
        assert score.density == 0.5
        assert score.message == "Test log message"
        assert score.record_id == "record_123"
        assert score.explanation == "This is unusual"

    def test_to_dict_minimal(self) -> None:
        """Test dictionary conversion with minimal fields."""
        score = NoveltyScore(
            score=0.85678,
            is_novel=True,
            index=42,
        )

        result = score.to_dict()

        assert result["score"] == 0.8568  # Rounded
        assert result["is_novel"] is True
        assert result["index"] == 42
        assert "k_neighbor_distances" not in result
        assert "density" not in result
        assert "message" not in result

    def test_to_dict_full(self) -> None:
        """Test dictionary conversion with all fields."""
        score = NoveltyScore(
            score=0.85,
            is_novel=True,
            index=42,
            k_neighbor_distances=[0.12340, 0.23456, 0.34567],
            density=0.56789,
            message="Test message",
            record_id="rec_1",
            explanation="Unusual pattern",
        )

        result = score.to_dict()

        # Check distances are rounded to 4 decimal places
        assert len(result["k_neighbor_distances"]) == 3
        assert result["density"] == 0.5679  # Rounded
        assert result["message"] == "Test message"
        assert result["record_id"] == "rec_1"
        assert result["explanation"] == "Unusual pattern"


# ============================================================================
# NoveltyResult Tests
# ============================================================================


class TestNoveltyResult:
    """Tests for NoveltyResult data class."""

    def test_creation(self) -> None:
        """Test creating a novelty result."""
        scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        is_novel = np.array([False, False, False, True, True], dtype=np.bool_)
        novel_scores = [
            NoveltyScore(score=0.8, is_novel=True, index=3),
            NoveltyScore(score=0.9, is_novel=True, index=4),
        ]

        result = NoveltyResult(
            scores=scores,
            is_novel=is_novel,
            novel_scores=novel_scores,
            n_novel=2,
            n_normal=3,
            threshold=0.75,
            detection_time_seconds=0.5,
            algorithm="knn_density",
            parameters={"k_neighbors": 5},
        )

        assert len(result.scores) == 5
        assert result.n_novel == 2
        assert result.n_normal == 3
        assert result.threshold == 0.75
        assert result.algorithm == "knn_density"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        scores = np.array([0.5, 0.6, 0.8, 0.9], dtype=np.float32)
        is_novel = np.array([False, False, True, True], dtype=np.bool_)
        novel_scores = [
            NoveltyScore(score=0.8, is_novel=True, index=2),
        ]

        result = NoveltyResult(
            scores=scores,
            is_novel=is_novel,
            novel_scores=novel_scores,
            n_novel=2,
            n_normal=2,
            threshold=0.75,
            detection_time_seconds=0.5,
            algorithm="knn_density",
            parameters={"k_neighbors": 5},
        )

        result_dict = result.to_dict()

        assert result_dict["n_samples"] == 4
        assert result_dict["n_novel"] == 2
        assert result_dict["n_normal"] == 2
        assert result_dict["novelty_rate"] == 0.5
        assert result_dict["algorithm"] == "knn_density"
        assert "score_stats" in result_dict
        assert "novel_samples" in result_dict

    def test_get_novel_indices(self) -> None:
        """Test getting indices of novel samples."""
        scores = np.array([0.5, 0.6, 0.8, 0.9], dtype=np.float32)
        is_novel = np.array([False, False, True, True], dtype=np.bool_)

        result = NoveltyResult(
            scores=scores,
            is_novel=is_novel,
            novel_scores=[],
            n_novel=2,
            n_normal=2,
            threshold=0.75,
            detection_time_seconds=0.5,
            algorithm="knn_density",
            parameters={},
        )

        novel_indices = result.get_novel_indices()

        assert novel_indices == [2, 3]

    def test_empty_result(self) -> None:
        """Test empty result handling."""
        result = NoveltyResult(
            scores=np.array([], dtype=np.float32),
            is_novel=np.array([], dtype=np.bool_),
            novel_scores=[],
            n_novel=0,
            n_normal=0,
            threshold=0.75,
            detection_time_seconds=0.0,
            algorithm="knn_density",
            parameters={},
        )

        result_dict = result.to_dict()

        assert result_dict["n_samples"] == 0
        assert result_dict["novelty_rate"] == 0.0


# ============================================================================
# KNNNoveltyDetector Tests
# ============================================================================


class TestKNNNoveltyDetector:
    """Tests for k-NN novelty detector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = KNNNoveltyDetector(
            k_neighbors=10,
            use_density=True,
            distance_metric="euclidean",
        )

        assert detector.name == "knn_density"
        assert detector.is_fitted is False

    def test_fit(self, sample_embeddings: NDArray[np.float32]) -> None:
        """Test fitting the detector."""
        detector = KNNNoveltyDetector(k_neighbors=5)

        detector.fit(sample_embeddings)

        assert detector.is_fitted is True

    def test_fit_empty_input(self) -> None:
        """Test fitting with empty input."""
        detector = KNNNoveltyDetector(k_neighbors=5)
        empty = np.array([], dtype=np.float32).reshape(0, 64)

        detector.fit(empty)

        assert detector.is_fitted is False

    def test_fit_insufficient_samples(self) -> None:
        """Test fitting with fewer samples than k."""
        detector = KNNNoveltyDetector(k_neighbors=10)
        small = np.random.randn(3, 64).astype(np.float32)

        # Should not raise, but adjust k internally
        detector.fit(small)

        assert detector.is_fitted is True

    def test_score_not_fitted(self) -> None:
        """Test scoring without fitting raises error."""
        detector = KNNNoveltyDetector(k_neighbors=5)
        embeddings = np.random.randn(10, 64).astype(np.float32)

        with pytest.raises(ProcessingError) as exc_info:
            detector.score(embeddings)

        assert "fitted" in str(exc_info.value).lower()

    def test_score(self, sample_embeddings: NDArray[np.float32]) -> None:
        """Test scoring embeddings."""
        detector = KNNNoveltyDetector(k_neighbors=5)
        detector.fit(sample_embeddings[:80])

        scores = detector.score(sample_embeddings[80:])

        assert len(scores) == 20
        assert all(0 <= s <= 1 for s in scores)

    def test_score_empty_input(self, sample_embeddings: NDArray[np.float32]) -> None:
        """Test scoring with empty input."""
        detector = KNNNoveltyDetector(k_neighbors=5)
        detector.fit(sample_embeddings)
        empty = np.array([], dtype=np.float32).reshape(0, 64)

        scores = detector.score(empty)

        assert len(scores) == 0

    def test_score_outliers_higher(self, clustered_embeddings: NDArray[np.float32]) -> None:
        """Test that outliers receive higher novelty scores."""
        detector = KNNNoveltyDetector(k_neighbors=5, use_density=True)

        # Fit on the clustered data (excluding outliers)
        reference = clustered_embeddings[:90]  # 3 clusters
        detector.fit(reference)

        # Score the outliers
        outliers = clustered_embeddings[90:]  # 10 outliers
        outlier_scores = detector.score(outliers)

        # Score some normal points
        normal = clustered_embeddings[:10]
        normal_scores = detector.score(normal)

        # Outliers should have higher average score
        avg_outlier_score = np.mean(outlier_scores)
        avg_normal_score = np.mean(normal_scores)

        assert avg_outlier_score > avg_normal_score

    def test_get_parameters(self) -> None:
        """Test getting algorithm parameters."""
        detector = KNNNoveltyDetector(
            k_neighbors=10,
            use_density=True,
            distance_metric="euclidean",
        )

        params = detector.get_parameters()

        assert params["k_neighbors"] == 10
        assert params["use_density"] is True
        assert params["distance_metric"] == "euclidean"
        assert params["is_fitted"] is False
        assert params["n_reference_samples"] == 0

    def test_get_parameters_after_fit(self, sample_embeddings: NDArray[np.float32]) -> None:
        """Test parameters after fitting."""
        detector = KNNNoveltyDetector(k_neighbors=5)
        detector.fit(sample_embeddings)

        params = detector.get_parameters()

        assert params["is_fitted"] is True
        assert params["n_reference_samples"] == 100

    def test_use_density_false(self, sample_embeddings: NDArray[np.float32]) -> None:
        """Test detector with density disabled."""
        detector = KNNNoveltyDetector(k_neighbors=5, use_density=False)
        detector.fit(sample_embeddings[:80])

        scores = detector.score(sample_embeddings[80:])

        assert len(scores) == 20
        assert all(0 <= s <= 1 for s in scores)


# ============================================================================
# MockNoveltyDetector Tests
# ============================================================================


class TestMockNoveltyDetector:
    """Tests for mock novelty detector."""

    def test_initialization(self) -> None:
        """Test mock detector initialization."""
        detector = MockNoveltyDetector(threshold=0.8, novelty_ratio=0.2)

        assert detector.name == "mock"
        assert detector.is_fitted is False

    def test_fit(self, sample_embeddings: NDArray[np.float32]) -> None:
        """Test fitting mock detector."""
        detector = MockNoveltyDetector()

        detector.fit(sample_embeddings)

        assert detector.is_fitted is True

    def test_score(self, sample_embeddings: NDArray[np.float32]) -> None:
        """Test mock scoring."""
        detector = MockNoveltyDetector()
        detector.fit(sample_embeddings)

        scores = detector.score(sample_embeddings)

        assert len(scores) == 100
        assert all(0 <= s <= 1 for s in scores)

    def test_score_deterministic(self, sample_embeddings: NDArray[np.float32]) -> None:
        """Test that mock scores are deterministic."""
        detector = MockNoveltyDetector()
        detector.fit(sample_embeddings)

        scores1 = detector.score(sample_embeddings)
        scores2 = detector.score(sample_embeddings)

        np.testing.assert_array_equal(scores1, scores2)

    def test_score_not_fitted(self) -> None:
        """Test scoring without fitting raises error."""
        detector = MockNoveltyDetector()
        embeddings = np.random.randn(10, 64).astype(np.float32)

        with pytest.raises(ProcessingError):
            detector.score(embeddings)

    def test_score_empty_input(self) -> None:
        """Test scoring empty input."""
        detector = MockNoveltyDetector()
        detector.fit(np.random.randn(10, 64).astype(np.float32))
        empty = np.array([], dtype=np.float32).reshape(0, 64)

        scores = detector.score(empty)

        assert len(scores) == 0

    def test_get_parameters(self) -> None:
        """Test getting mock parameters."""
        detector = MockNoveltyDetector(threshold=0.8, novelty_ratio=0.2)

        params = detector.get_parameters()

        assert params["threshold"] == 0.8
        assert params["novelty_ratio"] == 0.2
        assert params["is_fitted"] is False


# ============================================================================
# NoveltyService Tests
# ============================================================================


class TestNoveltyService:
    """Tests for NoveltyService high-level API."""

    def test_from_config(self, novelty_config: NoveltyConfig) -> None:
        """Test creating service from configuration."""
        service = NoveltyService.from_config(config=novelty_config)

        assert isinstance(service.detector, KNNNoveltyDetector)
        assert service.threshold == 0.7

    def test_from_config_mock(self, novelty_config: NoveltyConfig) -> None:
        """Test creating service with mock detector."""
        service = NoveltyService.from_config(
            config=novelty_config,
            use_mock=True,
        )

        assert isinstance(service.detector, MockNoveltyDetector)

    def test_from_config_default(self) -> None:
        """Test creating service with default configuration."""
        service = NoveltyService.from_config()

        assert service.detector is not None
        assert service.threshold > 0

    def test_fit(
        self,
        novelty_config: NoveltyConfig,
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Test fitting the service."""
        service = NoveltyService.from_config(config=novelty_config)

        service.fit(sample_embeddings)

        assert service.detector.is_fitted is True

    def test_fit_empty(self, novelty_config: NoveltyConfig) -> None:
        """Test fitting with empty input."""
        service = NoveltyService.from_config(config=novelty_config)
        empty = np.array([], dtype=np.float32).reshape(0, 64)

        # Should not raise
        service.fit(empty)

    def test_detect(
        self,
        novelty_config: NoveltyConfig,
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Test novelty detection."""
        service = NoveltyService.from_config(config=novelty_config)
        service.fit(sample_embeddings[:80])

        result = service.detect(sample_embeddings[80:])

        assert isinstance(result, NoveltyResult)
        assert len(result.scores) == 20
        assert result.n_novel + result.n_normal == 20
        assert result.threshold == 0.7
        assert result.algorithm == "knn_density"

    def test_detect_empty(
        self,
        novelty_config: NoveltyConfig,
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Test detection with empty input."""
        service = NoveltyService.from_config(config=novelty_config)
        service.fit(sample_embeddings)
        empty = np.array([], dtype=np.float32).reshape(0, 64)

        result = service.detect(empty)

        assert result.n_novel == 0
        assert result.n_normal == 0
        assert len(result.scores) == 0

    def test_detect_not_fitted(self, novelty_config: NoveltyConfig) -> None:
        """Test detection without fitting raises error."""
        service = NoveltyService.from_config(config=novelty_config)
        embeddings = np.random.randn(10, 64).astype(np.float32)

        with pytest.raises(ProcessingError) as exc_info:
            service.detect(embeddings)

        assert "fitted" in str(exc_info.value).lower()

    def test_detect_with_messages(
        self,
        novelty_config: NoveltyConfig,
        sample_embeddings: NDArray[np.float32],
        sample_messages: list[str],
    ) -> None:
        """Test detection with message context."""
        service = NoveltyService.from_config(config=novelty_config, use_mock=True)
        # Use small embeddings matching message count
        small_embeddings = sample_embeddings[:5]
        service.fit(small_embeddings)

        result = service.detect(small_embeddings, messages=sample_messages)

        # Check that novel scores include messages
        for novel in result.novel_scores:
            if novel.index < len(sample_messages):
                assert novel.message is not None

    def test_detect_with_records(
        self,
        novelty_config: NoveltyConfig,
        sample_embeddings: NDArray[np.float32],
        mock_log_records: list[MagicMock],
    ) -> None:
        """Test detection with log record context."""
        service = NoveltyService.from_config(config=novelty_config, use_mock=True)
        small_embeddings = sample_embeddings[:5]
        service.fit(small_embeddings)

        result = service.detect(small_embeddings, records=mock_log_records)

        for novel in result.novel_scores:
            if novel.index < len(mock_log_records):
                assert novel.record_id is not None

    def test_detect_custom_threshold(
        self,
        novelty_config: NoveltyConfig,
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Test detection with custom threshold."""
        service = NoveltyService.from_config(config=novelty_config)
        service.fit(sample_embeddings[:80])

        # Very high threshold - fewer novels
        high_result = service.detect(sample_embeddings[80:], threshold=0.99)
        # Very low threshold - more novels
        low_result = service.detect(sample_embeddings[80:], threshold=0.1)

        assert low_result.n_novel >= high_result.n_novel

    def test_score_batch(
        self,
        novelty_config: NoveltyConfig,
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Test batch scoring without classification."""
        service = NoveltyService.from_config(config=novelty_config)
        service.fit(sample_embeddings[:80])

        scores = service.score_batch(sample_embeddings[80:])

        assert len(scores) == 20
        assert all(0 <= s <= 1 for s in scores)

    def test_score_batch_not_fitted(self, novelty_config: NoveltyConfig) -> None:
        """Test batch scoring without fitting."""
        service = NoveltyService.from_config(config=novelty_config)
        embeddings = np.random.randn(10, 64).astype(np.float32)

        with pytest.raises(ProcessingError):
            service.score_batch(embeddings)

    def test_stats_tracking(
        self,
        novelty_config: NoveltyConfig,
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Test that statistics are tracked."""
        service = NoveltyService.from_config(config=novelty_config)
        service.fit(sample_embeddings[:80])

        # Initial stats
        initial_stats = service.get_stats()
        assert initial_stats.total_analyzed == 0

        # Detect
        service.detect(sample_embeddings[80:])

        # Updated stats
        stats = service.get_stats()
        assert stats.total_analyzed == 20
        assert stats.total_novel_detected + stats.total_normal_detected == 20
        assert stats.last_detection_time is not None

    def test_stats_accumulate(
        self,
        novelty_config: NoveltyConfig,
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Test that statistics accumulate across detections."""
        service = NoveltyService.from_config(config=novelty_config)
        service.fit(sample_embeddings[:70])

        # First detection
        service.detect(sample_embeddings[70:85])

        # Second detection
        service.detect(sample_embeddings[85:])

        stats = service.get_stats()
        assert stats.total_analyzed == 30  # 15 + 15

    def test_reset_stats(
        self,
        novelty_config: NoveltyConfig,
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Test resetting statistics."""
        service = NoveltyService.from_config(config=novelty_config)
        service.fit(sample_embeddings[:80])
        service.detect(sample_embeddings[80:])

        # Stats should have data
        assert service.get_stats().total_analyzed > 0

        # Reset
        service.reset_stats()

        # Stats should be zeroed
        stats = service.get_stats()
        assert stats.total_analyzed == 0
        assert stats.total_novel_detected == 0

    def test_explanation_generation(
        self,
        novelty_config: NoveltyConfig,
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Test that explanations are generated for novel samples."""
        # Use mock with controlled output
        service = NoveltyService.from_config(config=novelty_config, use_mock=True)
        service.fit(sample_embeddings)

        # Force some high scores to be detected as novel
        result = service.detect(sample_embeddings[:10], threshold=0.3)

        for novel in result.novel_scores:
            assert novel.explanation is not None
            assert "unusual" in novel.explanation.lower()


# ============================================================================
# NoveltyAlgorithmType Tests
# ============================================================================


class TestNoveltyAlgorithmType:
    """Tests for NoveltyAlgorithmType enum."""

    def test_enum_values(self) -> None:
        """Test enum values exist."""
        assert NoveltyAlgorithmType.KNN_DENSITY.value == "knn_density"
        assert NoveltyAlgorithmType.LOCAL_OUTLIER_FACTOR.value == "lof"
        assert NoveltyAlgorithmType.ISOLATION_FOREST.value == "isolation_forest"


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestNoveltyEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample_fit(self) -> None:
        """Test fitting with single sample."""
        detector = KNNNoveltyDetector(k_neighbors=5)
        single = np.random.randn(1, 64).astype(np.float32)

        # Should handle gracefully
        detector.fit(single)

        assert detector.is_fitted is True

    def test_identical_embeddings(self) -> None:
        """Test with all identical embeddings."""
        detector = KNNNoveltyDetector(k_neighbors=5)
        identical = np.ones((20, 64), dtype=np.float32)

        detector.fit(identical)
        scores = detector.score(identical)

        # All scores should be similar (around 0.5)
        assert np.std(scores) < 0.1

    def test_high_dimensional_embeddings(self) -> None:
        """Test with high-dimensional embeddings."""
        detector = KNNNoveltyDetector(k_neighbors=5)
        high_dim = np.random.randn(50, 1024).astype(np.float32)

        detector.fit(high_dim[:40])
        scores = detector.score(high_dim[40:])

        assert len(scores) == 10
        assert all(0 <= s <= 1 for s in scores)

    def test_large_batch(self) -> None:
        """Test with large batch of embeddings."""
        detector = KNNNoveltyDetector(k_neighbors=10)
        large = np.random.randn(1000, 64).astype(np.float32)

        detector.fit(large[:800])
        scores = detector.score(large[800:])

        assert len(scores) == 200

    def test_k_larger_than_reference(self) -> None:
        """Test when k is larger than reference set."""
        detector = KNNNoveltyDetector(k_neighbors=100)
        small_ref = np.random.randn(10, 64).astype(np.float32)
        query = np.random.randn(5, 64).astype(np.float32)

        detector.fit(small_ref)
        scores = detector.score(query)

        assert len(scores) == 5


# ============================================================================
# Integration Tests
# ============================================================================


class TestNoveltyIntegration:
    """Integration tests for novelty detection pipeline."""

    def test_full_pipeline_with_outliers(self, clustered_embeddings: NDArray[np.float32]) -> None:
        """Test full pipeline correctly identifies outliers."""
        service = NoveltyService.from_config(
            config=NoveltyConfig(
                threshold=0.6,
                k_neighbors=5,
                use_density=True,
            )
        )

        # Fit on clustered data
        reference = clustered_embeddings[:90]
        service.fit(reference)

        # Test detection on mixed data
        result = service.detect(clustered_embeddings)

        # Should detect the outliers (indices 90-99)
        novel_indices = result.get_novel_indices()

        # At least some of the outliers should be detected
        outlier_indices_detected = [i for i in novel_indices if i >= 90]
        assert len(outlier_indices_detected) > 0

    def test_pipeline_with_messages(self, sample_embeddings: NDArray[np.float32]) -> None:
        """Test pipeline with message context."""
        messages = [f"Log message {i}" for i in range(20)]

        service = NoveltyService.from_config(use_mock=True)
        service.fit(sample_embeddings)

        result = service.detect(sample_embeddings[:20], messages=messages)

        # Verify result structure
        assert isinstance(result, NoveltyResult)
        assert result.batch_id is not None

    def test_serialization_roundtrip(self, sample_embeddings: NDArray[np.float32]) -> None:
        """Test that results can be serialized and contain expected data."""
        service = NoveltyService.from_config(use_mock=True)
        service.fit(sample_embeddings)

        result = service.detect(sample_embeddings[:10])
        result_dict = result.to_dict()

        # Verify all expected keys
        assert "batch_id" in result_dict
        assert "n_samples" in result_dict
        assert "n_novel" in result_dict
        assert "n_normal" in result_dict
        assert "novelty_rate" in result_dict
        assert "score_stats" in result_dict
        assert "novel_samples" in result_dict
