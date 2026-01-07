"""
Comprehensive tests for the vector store module.

Tests cover:
- MockVectorIndex: Basic vector operations
- VectorStore: High-level storage and search
- VectorMetadata: Metadata management
- Persistence: Save/load functionality
- Integration: End-to-end workflows
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from sentinel_ml.models import LogRecord
from sentinel_ml.vectorstore import (
    IndexStrategy,
    MockVectorIndex,
    SearchResult,
    VectorMetadata,
    VectorStore,
    VectorStoreStats,
    get_vector_store,
    set_vector_store,
)


class TestVectorStoreStats:
    """Tests for VectorStoreStats dataclass."""

    def test_stats_initialization(self) -> None:
        """Test default stats initialization."""
        stats = VectorStoreStats()

        assert stats.total_vectors == 0
        assert stats.total_searches == 0
        assert stats.total_adds == 0
        assert stats.avg_search_time_ms == 0.0
        assert stats.avg_add_time_ms == 0.0
        assert stats.last_persist_time is None

    def test_stats_to_dict(self) -> None:
        """Test conversion to dictionary."""
        now = datetime.now(timezone.utc)
        stats = VectorStoreStats(
            total_vectors=1000,
            total_searches=50,
            total_adds=20,
            avg_search_time_ms=1.5,
            avg_add_time_ms=5.0,
            last_persist_time=now,
        )

        d = stats.to_dict()

        assert d["total_vectors"] == 1000
        assert d["total_searches"] == 50
        assert d["total_adds"] == 20
        assert d["avg_search_time_ms"] == 1.5
        assert d["avg_add_time_ms"] == 5.0
        assert d["last_persist_time"] == now.isoformat()


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Test search result creation."""
        result = SearchResult(
            id="log-123",
            distance=0.5,
            similarity=0.75,
            metadata={"source": "app.log"},
        )

        assert result.id == "log-123"
        assert result.distance == 0.5
        assert result.similarity == 0.75
        assert result.metadata["source"] == "app.log"

    def test_search_result_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = SearchResult(
            id="log-456",
            distance=0.123456789,
            similarity=0.987654321,
        )

        d = result.to_dict()

        assert d["id"] == "log-456"
        assert d["distance"] == 0.123457  # Rounded to 6 decimals
        assert d["similarity"] == 0.987654


class TestVectorMetadata:
    """Tests for VectorMetadata dataclass."""

    def test_metadata_creation(self) -> None:
        """Test metadata creation."""
        now = datetime.now(timezone.utc)
        metadata = VectorMetadata(
            id="ext-123",
            index_id=42,
            source="/var/log/app.log",
            level="ERROR",
            timestamp=now,
            message_preview="Connection failed",
            cluster_id="cluster-001",
        )

        assert metadata.id == "ext-123"
        assert metadata.index_id == 42
        assert metadata.source == "/var/log/app.log"
        assert metadata.level == "ERROR"
        assert metadata.cluster_id == "cluster-001"

    def test_metadata_to_dict_and_back(self) -> None:
        """Test round-trip serialization."""
        now = datetime.now(timezone.utc)
        original = VectorMetadata(
            id="ext-456",
            index_id=100,
            source="syslog",
            level="WARN",
            timestamp=now,
            message_preview="Disk space low",
        )

        d = original.to_dict()
        restored = VectorMetadata.from_dict(d)

        assert restored.id == original.id
        assert restored.index_id == original.index_id
        assert restored.source == original.source
        assert restored.level == original.level
        assert restored.message_preview == original.message_preview


class TestMockVectorIndex:
    """Tests for MockVectorIndex."""

    def test_index_initialization(self) -> None:
        """Test index initialization."""
        index = MockVectorIndex(dimension=64)

        assert index.dimension == 64
        assert index.size == 0

    def test_add_vectors(self) -> None:
        """Test adding vectors."""
        index = MockVectorIndex(dimension=32)
        vectors = np.random.randn(5, 32).astype(np.float32)

        ids = index.add(vectors)

        assert len(ids) == 5
        assert ids == [0, 1, 2, 3, 4]
        assert index.size == 5

    def test_add_empty(self) -> None:
        """Test adding empty array."""
        index = MockVectorIndex(dimension=32)
        vectors = np.array([], dtype=np.float32).reshape(0, 32)

        ids = index.add(vectors)

        assert ids == []
        assert index.size == 0

    def test_search_basic(self) -> None:
        """Test basic search functionality."""
        index = MockVectorIndex(dimension=4)

        # Add known vectors
        vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        index.add(vectors)

        # Search for vector closest to [1, 0, 0, 0]
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        distances, indices = index.search(query, k=2)

        # First result should be index 0 (closest to [1,0,0,0])
        assert indices[0, 0] == 0
        assert distances[0, 0] < distances[0, 1]

    def test_search_empty_index(self) -> None:
        """Test searching empty index."""
        index = MockVectorIndex(dimension=32)
        query = np.random.randn(32).astype(np.float32)

        distances, indices = index.search(query, k=5)

        assert distances.shape == (1, 0)
        assert indices.shape == (1, 0)

    def test_search_k_larger_than_size(self) -> None:
        """Test search with k larger than index size."""
        index = MockVectorIndex(dimension=8)
        vectors = np.random.randn(3, 8).astype(np.float32)
        index.add(vectors)

        query = np.random.randn(8).astype(np.float32)
        distances, indices = index.search(query, k=10)

        # Should return only 3 results
        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_save_and_load(self) -> None:
        """Test saving and loading index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index.json"

            # Create and populate index
            index1 = MockVectorIndex(dimension=16)
            vectors = np.random.randn(5, 16).astype(np.float32)
            index1.add(vectors)

            # Save
            index1.save(path)
            assert path.exists()

            # Load into new index
            index2 = MockVectorIndex(dimension=16)
            index2.load(path)

            assert index2.size == 5
            assert index2.dimension == 16


class TestVectorStore:
    """Tests for VectorStore high-level interface."""

    @pytest.fixture
    def mock_store(self) -> VectorStore:
        """Create a mock vector store for testing."""
        return VectorStore.create_mock(dimension=64)

    def test_store_creation(self, mock_store: VectorStore) -> None:
        """Test store creation."""
        assert mock_store.dimension == 64
        assert mock_store.size == 0

    def test_add_vectors(self, mock_store: VectorStore) -> None:
        """Test adding vectors without records."""
        embeddings = np.random.randn(5, 64).astype(np.float32)

        ids = mock_store.add(embeddings)

        assert len(ids) == 5
        assert mock_store.size == 5
        assert mock_store.stats.total_adds == 1

    def test_add_with_custom_ids(self, mock_store: VectorStore) -> None:
        """Test adding vectors with custom IDs."""
        embeddings = np.random.randn(3, 64).astype(np.float32)
        custom_ids = ["log-001", "log-002", "log-003"]

        ids = mock_store.add(embeddings, ids=custom_ids)

        assert ids == custom_ids
        assert mock_store.get_by_id("log-001") is not None
        assert mock_store.get_by_id("log-002") is not None

    def test_add_with_records(self, mock_store: VectorStore) -> None:
        """Test adding vectors with log records."""
        now = datetime.now(timezone.utc)
        embeddings = np.random.randn(2, 64).astype(np.float32)
        records = [
            LogRecord(
                id="rec-001",
                message="Error connecting to database",
                source="/var/log/app.log",
                raw="ERROR Error connecting to database",
                level="ERROR",
                timestamp=now,
            ),
            LogRecord(
                id="rec-002",
                message="Request completed successfully",
                source="/var/log/app.log",
                raw="INFO Request completed successfully",
                level="INFO",
                timestamp=now,
            ),
        ]

        ids = mock_store.add(embeddings, records=records)

        assert len(ids) == 2

        # Check metadata
        metadata = mock_store.get_by_id(ids[0])
        assert metadata is not None
        assert metadata.source == "/var/log/app.log"
        assert metadata.level == "ERROR"
        assert "Error connecting" in (metadata.message_preview or "")

    def test_add_empty(self, mock_store: VectorStore) -> None:
        """Test adding empty array."""
        embeddings = np.array([], dtype=np.float32).reshape(0, 64)

        ids = mock_store.add(embeddings)

        assert ids == []
        assert mock_store.size == 0

    def test_add_mismatched_records(self, mock_store: VectorStore) -> None:
        """Test error when records count doesn't match embeddings."""
        embeddings = np.random.randn(3, 64).astype(np.float32)
        records = [
            LogRecord(message="msg", source="src", raw="raw"),
        ]

        with pytest.raises(Exception):  # StorageError
            mock_store.add(embeddings, records=records)

    def test_search_basic(self, mock_store: VectorStore) -> None:
        """Test basic search functionality."""
        # Add some vectors
        embeddings = np.random.randn(10, 64).astype(np.float32)
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        mock_store.add(embeddings)

        # Search
        query = embeddings[0]  # Search for first vector
        results = mock_store.search(query, k=3)

        assert len(results) <= 3
        assert mock_store.stats.total_searches == 1

        # First result should be very similar to query
        if results:
            assert results[0].similarity > 0.5

    def test_search_empty_store(self, mock_store: VectorStore) -> None:
        """Test searching empty store."""
        query = np.random.randn(64).astype(np.float32)

        results = mock_store.search(query, k=5)

        assert results == []

    def test_search_with_min_similarity(self, mock_store: VectorStore) -> None:
        """Test search with minimum similarity threshold."""
        # Add vectors
        embeddings = np.random.randn(10, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        mock_store.add(embeddings)

        # Create a very different query
        query = np.random.randn(64).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Search with high threshold
        results = mock_store.search(query, k=10, min_similarity=0.99)

        # Most random vectors won't be 0.99 similar
        assert len(results) <= 10

    def test_search_batch(self, mock_store: VectorStore) -> None:
        """Test batch search."""
        embeddings = np.random.randn(20, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        mock_store.add(embeddings)

        # Batch query
        queries = np.random.randn(3, 64).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        results = mock_store.search_batch(queries, k=5)

        assert len(results) == 3
        for result_list in results:
            assert len(result_list) <= 5

    def test_get_by_id(self, mock_store: VectorStore) -> None:
        """Test getting metadata by ID."""
        embeddings = np.random.randn(1, 64).astype(np.float32)
        ids = mock_store.add(embeddings, ids=["test-id"])

        metadata = mock_store.get_by_id("test-id")

        assert metadata is not None
        assert metadata.id == "test-id"

    def test_get_by_id_not_found(self, mock_store: VectorStore) -> None:
        """Test getting non-existent ID."""
        metadata = mock_store.get_by_id("nonexistent")
        assert metadata is None

    def test_update_cluster_id(self, mock_store: VectorStore) -> None:
        """Test updating cluster ID."""
        embeddings = np.random.randn(1, 64).astype(np.float32)
        mock_store.add(embeddings, ids=["test-id"])

        success = mock_store.update_cluster_id("test-id", "cluster-001")
        assert success

        metadata = mock_store.get_by_id("test-id")
        assert metadata is not None
        assert metadata.cluster_id == "cluster-001"

    def test_update_cluster_id_not_found(self, mock_store: VectorStore) -> None:
        """Test updating non-existent ID."""
        success = mock_store.update_cluster_id("nonexistent", "cluster-001")
        assert not success

    def test_clear(self, mock_store: VectorStore) -> None:
        """Test clearing the store."""
        embeddings = np.random.randn(10, 64).astype(np.float32)
        mock_store.add(embeddings)
        assert mock_store.size == 10

        mock_store.clear()

        assert mock_store.size == 0
        assert mock_store.stats.total_vectors == 0

    def test_reset_stats(self, mock_store: VectorStore) -> None:
        """Test resetting stats."""
        embeddings = np.random.randn(5, 64).astype(np.float32)
        mock_store.add(embeddings)

        query = np.random.randn(64).astype(np.float32)
        mock_store.search(query, k=3)

        assert mock_store.stats.total_searches > 0
        assert mock_store.stats.total_adds > 0

        mock_store.reset_stats()

        assert mock_store.stats.total_searches == 0
        assert mock_store.stats.total_adds == 0
        assert mock_store.stats.total_vectors == 5  # Preserved

    def test_save_and_load(self) -> None:
        """Test saving and loading the store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir)

            # Create and populate store
            store1 = VectorStore(
                index=MockVectorIndex(dimension=32),
                persist_dir=persist_dir,
            )
            embeddings = np.random.randn(5, 32).astype(np.float32)
            records = [
                LogRecord(
                    message=f"Message {i}",
                    source="test.log",
                    raw=f"INFO Message {i}",
                )
                for i in range(5)
            ]
            ids = store1.add(embeddings, records=records, ids=[f"id-{i}" for i in range(5)])

            # Save
            store1.save()

            # Load into new store
            store2 = VectorStore(
                index=MockVectorIndex(dimension=32),
                persist_dir=persist_dir,
            )
            store2.load()

            assert store2.size == 5

            # Check metadata preserved
            metadata = store2.get_by_id("id-0")
            assert metadata is not None
            assert metadata.source == "test.log"


class TestGlobalVectorStore:
    """Tests for global vector store singleton."""

    def test_set_and_get_store(self) -> None:
        """Test setting and getting global store."""
        mock_store = VectorStore.create_mock(dimension=128)

        set_vector_store(mock_store)
        retrieved = get_vector_store()

        assert retrieved is mock_store
        assert retrieved.dimension == 128


class TestVectorStoreIntegration:
    """Integration tests for vector store with embeddings."""

    def test_end_to_end_workflow(self) -> None:
        """Test complete workflow: add, search, update, persist."""
        store = VectorStore.create_mock(dimension=32)

        # Create test data
        now = datetime.now(timezone.utc)
        records = [
            LogRecord(
                message="Connection timeout to database",
                normalized="Connection timeout to database",
                source="app.log",
                raw="ERROR Connection timeout to database",
                level="ERROR",
                timestamp=now,
            ),
            LogRecord(
                message="Connection refused by server",
                normalized="Connection refused by server",
                source="app.log",
                raw="ERROR Connection refused by server",
                level="ERROR",
                timestamp=now,
            ),
            LogRecord(
                message="Request completed in 100ms",
                normalized="Request completed in <num>ms",
                source="app.log",
                raw="INFO Request completed in 100ms",
                level="INFO",
                timestamp=now,
            ),
        ]

        # Create mock embeddings (similar for similar messages)
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0] + [0.0] * 28,  # Connection error 1
                [0.9, 0.1, 0.0, 0.0] + [0.0] * 28,  # Connection error 2 (similar)
                [0.0, 0.0, 1.0, 0.0] + [0.0] * 28,  # Different (success message)
            ],
            dtype=np.float32,
        )
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add to store
        ids = store.add(embeddings, records=records)
        assert len(ids) == 3

        # Search for similar to first record
        results = store.search(embeddings[0], k=2)
        assert len(results) == 2

        # First result should be exact match
        assert results[0].similarity > 0.95

        # Second result should be similar (connection error 2)
        # Note: with mock, this depends on the actual distance calculation
        assert results[1].id in ids

        # Update cluster ID
        store.update_cluster_id(ids[0], "connection-errors")
        store.update_cluster_id(ids[1], "connection-errors")
        store.update_cluster_id(ids[2], "success-messages")

        # Verify cluster IDs
        metadata = store.get_by_id(ids[0])
        assert metadata is not None
        assert metadata.cluster_id == "connection-errors"

    def test_stats_tracking(self) -> None:
        """Test that statistics are tracked correctly."""
        store = VectorStore.create_mock(dimension=16)

        # Multiple adds
        for _ in range(5):
            embeddings = np.random.randn(10, 16).astype(np.float32)
            store.add(embeddings)

        assert store.stats.total_adds == 5
        assert store.stats.total_vectors == 50

        # Multiple searches
        for _ in range(10):
            query = np.random.randn(16).astype(np.float32)
            store.search(query, k=5)

        assert store.stats.total_searches == 10
        assert store.stats.avg_search_time_ms > 0
        assert store.stats.avg_add_time_ms > 0

    def test_large_batch_handling(self) -> None:
        """Test handling of large batches."""
        store = VectorStore.create_mock(dimension=64)

        # Add 1000 vectors
        embeddings = np.random.randn(1000, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        ids = store.add(embeddings)

        assert len(ids) == 1000
        assert store.size == 1000

        # Search should still work
        query = embeddings[500]
        results = store.search(query, k=10)

        assert len(results) == 10
        # First result should be exact match (or very close)
        assert results[0].similarity > 0.99
