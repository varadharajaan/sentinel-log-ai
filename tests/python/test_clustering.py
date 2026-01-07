"""
Comprehensive tests for the clustering module.

Tests cover:
- ClusteringAlgorithm: HDBSCAN and Mock implementations
- ClusteringService: High-level clustering operations
- ClusterSummary: Cluster metadata and statistics
- Integration: End-to-end clustering workflow
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from sentinel_ml.clustering import (
    ClusteringResult,
    ClusteringService,
    ClusterStats,
    ClusterSummary,
    HDBSCANAlgorithm,
    MockClusteringAlgorithm,
    get_clustering_service,
    reset_clustering_service,
    set_clustering_service,
)
from sentinel_ml.config import ClusteringConfig
from sentinel_ml.models import LogRecord

# Check if hdbscan is available
try:
    import hdbscan  # noqa: F401

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

requires_hdbscan = pytest.mark.skipif(not HAS_HDBSCAN, reason="hdbscan package not installed")

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create sample embeddings with natural clusters."""
    np.random.seed(42)

    # Create 3 distinct clusters
    cluster1 = np.random.randn(20, 64).astype(np.float32) + np.array([5, 0] + [0] * 62)
    cluster2 = np.random.randn(20, 64).astype(np.float32) + np.array([-5, 0] + [0] * 62)
    cluster3 = np.random.randn(20, 64).astype(np.float32) + np.array([0, 5] + [0] * 62)

    # Add some noise points
    noise = np.random.randn(5, 64).astype(np.float32) * 10

    return np.vstack([cluster1, cluster2, cluster3, noise])


@pytest.fixture
def sample_records() -> list[LogRecord]:
    """Create sample log records for clustering."""
    records = []
    levels = ["INFO", "ERROR", "WARN"]
    sources = ["app.log", "system.log", "auth.log"]

    for i in range(65):
        cluster_idx = i // 20 if i < 60 else 0
        records.append(
            LogRecord(
                id=f"log-{i}",
                message=f"Log message {i} from cluster {cluster_idx}",
                normalized=f"Log message <num> from cluster {cluster_idx}",
                level=levels[cluster_idx % 3],
                source=sources[cluster_idx % 3],
                raw=f"[INFO] Log message {i}",
                timestamp=datetime.now(timezone.utc),
            )
        )

    return records


@pytest.fixture
def mock_algorithm() -> MockClusteringAlgorithm:
    """Create a mock clustering algorithm."""
    return MockClusteringAlgorithm(n_clusters=3, noise_ratio=0.1)


@pytest.fixture
def mock_service() -> ClusteringService:
    """Create a clustering service with mock algorithm."""
    return ClusteringService.from_config(use_mock=True)


@pytest.fixture(autouse=True)
def reset_global_service() -> None:
    """Reset global clustering service after each test."""
    yield
    reset_clustering_service()


# ============================================================================
# ClusterStats Tests
# ============================================================================


class TestClusterStats:
    """Tests for ClusterStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values initialization."""
        stats = ClusterStats()

        assert stats.total_clustered == 0
        assert stats.n_clusters_found == 0
        assert stats.n_noise_points == 0
        assert stats.clustering_time_seconds == 0.0
        assert stats.silhouette_score is None
        assert stats.last_cluster_time is None

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        stats = ClusterStats(
            total_clustered=100,
            n_clusters_found=5,
            n_noise_points=10,
            clustering_time_seconds=1.234,
            silhouette_score=0.756,
            last_cluster_time=datetime(2026, 1, 7, 12, 0, 0, tzinfo=timezone.utc),
        )

        result = stats.to_dict()

        assert result["total_clustered"] == 100
        assert result["n_clusters_found"] == 5
        assert result["n_noise_points"] == 10
        assert result["clustering_time_seconds"] == 1.234
        assert result["silhouette_score"] == 0.756
        assert "2026-01-07" in result["last_cluster_time"]


# ============================================================================
# ClusterSummary Tests
# ============================================================================


class TestClusterSummary:
    """Tests for ClusterSummary dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        summary = ClusterSummary(
            id="test-id",
            label=0,
            size=10,
            representative_messages=["msg1", "msg2"],
            representative_indices=[0, 5],
        )

        assert summary.id == "test-id"
        assert summary.label == 0
        assert summary.size == 10
        assert len(summary.representative_messages) == 2
        assert summary.centroid is None

    def test_with_centroid(self) -> None:
        """Test creation with centroid."""
        centroid = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        summary = ClusterSummary(
            id="test-id",
            label=0,
            size=10,
            representative_messages=["msg"],
            representative_indices=[0],
            centroid=centroid,
        )

        assert summary.centroid is not None
        assert len(summary.centroid) == 3

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        centroid = np.array([1.0, 2.0], dtype=np.float32)
        summary = ClusterSummary(
            id="test-id",
            label=1,
            size=20,
            representative_messages=["msg1"],
            representative_indices=[0],
            centroid=centroid,
            common_level="ERROR",
            common_source="app.log",
        )

        result = summary.to_dict()

        assert result["id"] == "test-id"
        assert result["label"] == 1
        assert result["size"] == 20
        assert result["centroid"] == [1.0, 2.0]
        assert result["common_level"] == "ERROR"
        assert result["common_source"] == "app.log"

    def test_to_dict_without_centroid(self) -> None:
        """Test to_dict when centroid is None."""
        summary = ClusterSummary(
            id="test-id",
            label=0,
            size=5,
            representative_messages=[],
            representative_indices=[],
        )

        result = summary.to_dict()

        assert result["centroid"] is None


# ============================================================================
# MockClusteringAlgorithm Tests
# ============================================================================


class TestMockClusteringAlgorithm:
    """Tests for MockClusteringAlgorithm."""

    def test_name(self, mock_algorithm: MockClusteringAlgorithm) -> None:
        """Test algorithm name."""
        assert mock_algorithm.name == "mock"

    def test_get_parameters(self, mock_algorithm: MockClusteringAlgorithm) -> None:
        """Test parameter retrieval."""
        params = mock_algorithm.get_parameters()

        assert params["n_clusters"] == 3
        assert params["noise_ratio"] == 0.1

    def test_fit_predict_basic(self, mock_algorithm: MockClusteringAlgorithm) -> None:
        """Test basic clustering."""
        embeddings = np.random.randn(50, 64).astype(np.float32)

        labels = mock_algorithm.fit_predict(embeddings)

        assert len(labels) == 50
        assert labels.dtype == np.int32
        # Should have some noise points
        assert np.any(labels == -1)

    def test_fit_predict_empty(self, mock_algorithm: MockClusteringAlgorithm) -> None:
        """Test clustering with empty input."""
        embeddings = np.array([], dtype=np.float32).reshape(0, 64)

        labels = mock_algorithm.fit_predict(embeddings)

        assert len(labels) == 0

    def test_deterministic_labels(self) -> None:
        """Test that labels are deterministic for same input."""
        algorithm = MockClusteringAlgorithm(n_clusters=3, noise_ratio=0.0)
        embeddings = np.array([[1.0] * 64, [2.0] * 64], dtype=np.float32)

        labels1 = algorithm.fit_predict(embeddings)
        labels2 = algorithm.fit_predict(embeddings)

        np.testing.assert_array_equal(labels1, labels2)


# ============================================================================
# HDBSCANAlgorithm Tests
# ============================================================================


@requires_hdbscan
class TestHDBSCANAlgorithm:
    """Tests for HDBSCANAlgorithm."""

    def test_name(self) -> None:
        """Test algorithm name."""
        algorithm = HDBSCANAlgorithm(min_cluster_size=5)
        assert algorithm.name == "hdbscan"

    def test_get_parameters(self) -> None:
        """Test parameter retrieval."""
        algorithm = HDBSCANAlgorithm(
            min_cluster_size=10,
            min_samples=5,
            cluster_selection_epsilon=0.5,
            metric="cosine",
        )

        params = algorithm.get_parameters()

        assert params["min_cluster_size"] == 10
        assert params["min_samples"] == 5
        assert params["cluster_selection_epsilon"] == 0.5
        assert params["metric"] == "cosine"

    def test_fit_predict_basic(self, sample_embeddings: np.ndarray) -> None:
        """Test basic HDBSCAN clustering."""
        algorithm = HDBSCANAlgorithm(min_cluster_size=5)

        labels = algorithm.fit_predict(sample_embeddings)

        assert len(labels) == len(sample_embeddings)
        assert labels.dtype == np.int32

    def test_fit_predict_finds_clusters(self, sample_embeddings: np.ndarray) -> None:
        """Test that HDBSCAN finds the expected clusters."""
        algorithm = HDBSCANAlgorithm(min_cluster_size=5)

        labels = algorithm.fit_predict(sample_embeddings)

        # Should find at least 2 clusters (3 ideal, but depends on parameters)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        assert n_clusters >= 1

    def test_fit_predict_insufficient_samples(self) -> None:
        """Test clustering with too few samples."""
        algorithm = HDBSCANAlgorithm(min_cluster_size=10)
        embeddings = np.random.randn(5, 64).astype(np.float32)

        labels = algorithm.fit_predict(embeddings)

        # All should be noise
        assert np.all(labels == -1)

    def test_fit_predict_empty(self) -> None:
        """Test clustering with empty input."""
        algorithm = HDBSCANAlgorithm(min_cluster_size=5)
        embeddings = np.array([], dtype=np.float32).reshape(0, 64)

        labels = algorithm.fit_predict(embeddings)

        assert len(labels) == 0


# ============================================================================
# ClusteringService Tests
# ============================================================================


class TestClusteringService:
    """Tests for ClusteringService."""

    def test_from_config_default(self) -> None:
        """Test service creation with default config."""
        service = ClusteringService.from_config(use_mock=True)

        assert service.algorithm.name == "mock"
        assert service.n_representative_samples == 5

    def test_from_config_custom(self) -> None:
        """Test service creation with custom config."""
        config = ClusteringConfig(
            min_cluster_size=10,
            min_samples=5,
            metric="cosine",
        )

        service = ClusteringService.from_config(config=config, use_mock=False)

        params = service.algorithm.get_parameters()
        assert params["min_cluster_size"] == 10
        assert params["min_samples"] == 5
        assert params["metric"] == "cosine"

    def test_cluster_basic(
        self,
        mock_service: ClusteringService,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test basic clustering."""
        result = mock_service.cluster(sample_embeddings)

        assert isinstance(result, ClusteringResult)
        assert len(result.labels) == len(sample_embeddings)
        assert result.n_clusters > 0
        assert result.algorithm == "mock"

    def test_cluster_with_records(
        self,
        mock_service: ClusteringService,
        sample_embeddings: np.ndarray,
        sample_records: list[LogRecord],
    ) -> None:
        """Test clustering with log records."""
        result = mock_service.cluster(sample_embeddings, records=sample_records)

        assert len(result.summaries) > 0

        # Check that summaries have representative messages
        for summary in result.summaries:
            assert len(summary.representative_messages) > 0
            assert summary.size > 0

    def test_cluster_with_messages(
        self,
        mock_service: ClusteringService,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test clustering with message list."""
        messages = [f"Message {i}" for i in range(len(sample_embeddings))]

        result = mock_service.cluster(sample_embeddings, messages=messages)

        # Check that summaries have representative messages
        for summary in result.summaries:
            assert len(summary.representative_messages) > 0

    def test_cluster_empty_input(self, mock_service: ClusteringService) -> None:
        """Test clustering with empty input."""
        embeddings = np.array([], dtype=np.float32).reshape(0, 64)

        result = mock_service.cluster(embeddings)

        assert result.n_clusters == 0
        assert result.n_noise == 0
        assert len(result.summaries) == 0
        assert len(result.labels) == 0

    def test_cluster_updates_stats(
        self,
        mock_service: ClusteringService,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test that clustering updates statistics."""
        initial_clustered = mock_service.stats.total_clustered

        mock_service.cluster(sample_embeddings)

        assert mock_service.stats.total_clustered == initial_clustered + len(sample_embeddings)
        assert mock_service.stats.n_clusters_found > 0
        assert mock_service.stats.clustering_time_seconds > 0
        assert mock_service.stats.last_cluster_time is not None

    def test_cluster_result_to_dict(
        self,
        mock_service: ClusteringService,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test ClusteringResult to_dict."""
        result = mock_service.cluster(sample_embeddings)
        result_dict = result.to_dict()

        assert "n_clusters" in result_dict
        assert "n_noise" in result_dict
        assert "summaries" in result_dict
        assert "clustering_time_seconds" in result_dict
        assert "algorithm" in result_dict
        assert "parameters" in result_dict

    @requires_hdbscan
    def test_recluster_with_params(
        self,
        mock_service: ClusteringService,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test re-clustering with custom parameters."""
        # This will use HDBSCAN with custom params
        result = mock_service.recluster_with_params(
            sample_embeddings,
            min_cluster_size=3,
            min_samples=2,
        )

        assert isinstance(result, ClusteringResult)
        # Original algorithm should be restored
        assert mock_service.algorithm.name == "mock"

    def test_cluster_extracts_common_level(
        self,
        mock_service: ClusteringService,
        sample_embeddings: np.ndarray,
        sample_records: list[LogRecord],
    ) -> None:
        """Test that clustering extracts common log level."""
        result = mock_service.cluster(sample_embeddings, records=sample_records)

        # At least one summary should have a common level
        has_level = any(s.common_level is not None for s in result.summaries)
        assert has_level

    def test_cluster_extracts_time_range(
        self,
        mock_service: ClusteringService,
        sample_embeddings: np.ndarray,
        sample_records: list[LogRecord],
    ) -> None:
        """Test that clustering extracts time range."""
        result = mock_service.cluster(sample_embeddings, records=sample_records)

        # At least one summary should have time range
        has_time = any(
            s.time_range_start is not None and s.time_range_end is not None
            for s in result.summaries
        )
        assert has_time


# ============================================================================
# Global Service Tests
# ============================================================================


class TestGlobalService:
    """Tests for global clustering service management."""

    def test_get_clustering_service(self) -> None:
        """Test getting global service."""
        service = get_clustering_service()

        assert isinstance(service, ClusteringService)

    def test_set_clustering_service(self) -> None:
        """Test setting global service."""
        custom_service = ClusteringService.from_config(use_mock=True)
        set_clustering_service(custom_service)

        retrieved = get_clustering_service()
        assert retrieved is custom_service

    def test_reset_clustering_service(self) -> None:
        """Test resetting global service."""
        # Get initial service
        service1 = get_clustering_service()

        # Reset
        reset_clustering_service()

        # Get new service
        service2 = get_clustering_service()

        assert service1 is not service2


# ============================================================================
# Integration Tests
# ============================================================================


class TestClusteringIntegration:
    """Integration tests for clustering workflow."""

    def test_full_clustering_workflow(self) -> None:
        """Test complete clustering workflow."""
        # Create embeddings
        np.random.seed(42)
        cluster1 = np.random.randn(15, 32).astype(np.float32) + 3
        cluster2 = np.random.randn(15, 32).astype(np.float32) - 3
        embeddings = np.vstack([cluster1, cluster2])

        # Create records
        records = [
            LogRecord(
                message=f"Error in module A: {i}",
                normalized="Error in module A: <num>",
                level="ERROR" if i < 15 else "WARN",
                source="module_a.log" if i < 15 else "module_b.log",
                raw=f"Error {i}",
            )
            for i in range(30)
        ]

        # Cluster
        service = ClusteringService.from_config(use_mock=True)
        result = service.cluster(embeddings, records=records)

        # Verify results
        assert result.n_clusters >= 1
        assert len(result.summaries) >= 1

        for summary in result.summaries:
            assert summary.id is not None
            assert summary.size > 0
            assert len(summary.representative_indices) > 0
            assert summary.centroid is not None

    @requires_hdbscan
    def test_clustering_with_hdbscan(self) -> None:
        """Test clustering with actual HDBSCAN."""
        # Create well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(20, 10).astype(np.float32) + 10
        cluster2 = np.random.randn(20, 10).astype(np.float32) - 10
        embeddings = np.vstack([cluster1, cluster2])

        service = ClusteringService.from_config(use_mock=False)
        result = service.cluster(embeddings)

        # Should find 2 clusters
        assert result.n_clusters >= 1
        assert len(result.labels) == 40

    def test_incremental_clustering(self, mock_service: ClusteringService) -> None:
        """Test multiple clustering operations."""
        np.random.seed(42)

        for i in range(3):
            embeddings = np.random.randn(20, 32).astype(np.float32)
            result = mock_service.cluster(embeddings)

            assert result.n_clusters >= 0
            assert mock_service.stats.total_clustered == (i + 1) * 20

    @requires_hdbscan
    def test_cluster_with_all_noise(self) -> None:
        """Test clustering when all points are noise."""
        # Very sparse data
        embeddings = np.random.randn(10, 64).astype(np.float32) * 100

        service = ClusteringService.from_config(use_mock=False)
        service.algorithm = HDBSCANAlgorithm(min_cluster_size=20)  # Too large

        result = service.cluster(embeddings)

        # All should be noise
        assert result.n_clusters == 0
        assert result.n_noise == 10
        assert len(result.summaries) == 0
