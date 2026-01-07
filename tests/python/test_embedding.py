"""
Comprehensive tests for the embedding module.

Tests cover:
- EmbeddingCache: LRU caching behavior
- MockEmbeddingProvider: Deterministic mock embeddings
- EmbeddingService: High-level embedding operations
- Integration with log records
"""

import hashlib
from datetime import datetime, timezone

import numpy as np
import pytest

from sentinel_ml.embedding import (
    EmbeddingCache,
    EmbeddingService,
    EmbeddingStats,
    MockEmbeddingProvider,
    get_embedding_service,
    set_embedding_service,
)
from sentinel_ml.models import LogRecord


class TestEmbeddingStats:
    """Tests for EmbeddingStats dataclass."""

    def test_stats_initialization(self) -> None:
        """Test default stats initialization."""
        stats = EmbeddingStats()

        assert stats.total_embedded == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.batch_count == 0
        assert stats.total_time_seconds == 0.0

    def test_cache_hit_rate_empty(self) -> None:
        """Test cache hit rate when no operations."""
        stats = EmbeddingStats()
        assert stats.cache_hit_rate == 0.0

    def test_cache_hit_rate_calculation(self) -> None:
        """Test cache hit rate calculation."""
        stats = EmbeddingStats(cache_hits=75, cache_misses=25)
        assert stats.cache_hit_rate == 0.75

    def test_cache_hit_rate_all_hits(self) -> None:
        """Test cache hit rate when all hits."""
        stats = EmbeddingStats(cache_hits=100, cache_misses=0)
        assert stats.cache_hit_rate == 1.0

    def test_cache_hit_rate_all_misses(self) -> None:
        """Test cache hit rate when all misses."""
        stats = EmbeddingStats(cache_hits=0, cache_misses=100)
        assert stats.cache_hit_rate == 0.0

    def test_avg_time_per_batch_empty(self) -> None:
        """Test average time when no batches."""
        stats = EmbeddingStats()
        assert stats.avg_time_per_batch == 0.0

    def test_avg_time_per_batch_calculation(self) -> None:
        """Test average time per batch calculation."""
        stats = EmbeddingStats(total_time_seconds=10.0, batch_count=5)
        assert stats.avg_time_per_batch == 2.0

    def test_throughput_empty(self) -> None:
        """Test throughput when no time elapsed."""
        stats = EmbeddingStats()
        assert stats.throughput == 0.0

    def test_throughput_calculation(self) -> None:
        """Test throughput calculation."""
        stats = EmbeddingStats(total_embedded=1000, total_time_seconds=10.0)
        assert stats.throughput == 100.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        stats = EmbeddingStats(
            total_embedded=500,
            cache_hits=400,
            cache_misses=100,
            total_time_seconds=5.0,
            batch_count=10,
            model_load_time_seconds=2.5,
        )

        d = stats.to_dict()

        assert d["total_embedded"] == 500
        assert d["cache_hits"] == 400
        assert d["cache_misses"] == 100
        assert d["cache_hit_rate"] == 0.8
        assert d["total_time_seconds"] == 5.0
        assert d["batch_count"] == 10
        assert d["avg_time_per_batch"] == 0.5
        assert d["throughput"] == 100.0
        assert d["model_load_time_seconds"] == 2.5


class TestEmbeddingCache:
    """Tests for EmbeddingCache LRU implementation."""

    def test_cache_initialization(self) -> None:
        """Test cache initialization."""
        cache = EmbeddingCache(max_size=100)
        assert cache.size == 0

    def test_cache_put_and_get(self) -> None:
        """Test basic put and get operations."""
        cache = EmbeddingCache()
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        cache.put("test message", embedding)
        assert cache.size == 1

        cached = cache.get("test message")
        assert cached is not None
        np.testing.assert_array_almost_equal(cached, embedding)

    def test_cache_miss(self) -> None:
        """Test cache miss returns None."""
        cache = EmbeddingCache()
        assert cache.get("nonexistent") is None

    def test_cache_eviction_lru(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(max_size=3)

        # Fill cache
        for i in range(3):
            cache.put(f"msg_{i}", np.array([float(i)], dtype=np.float32))

        assert cache.size == 3

        # Add one more, should evict msg_0 (oldest)
        cache.put("msg_new", np.array([99.0], dtype=np.float32))

        assert cache.size == 3
        assert cache.get("msg_0") is None  # Evicted
        assert cache.get("msg_1") is not None
        assert cache.get("msg_2") is not None
        assert cache.get("msg_new") is not None

    def test_cache_lru_access_update(self) -> None:
        """Test that accessing updates LRU order."""
        cache = EmbeddingCache(max_size=3)

        # Fill cache
        for i in range(3):
            cache.put(f"msg_{i}", np.array([float(i)], dtype=np.float32))

        # Access msg_0 to make it recently used
        cache.get("msg_0")

        # Add new item, should evict msg_1 (now oldest)
        cache.put("msg_new", np.array([99.0], dtype=np.float32))

        assert cache.get("msg_0") is not None  # Still present
        assert cache.get("msg_1") is None  # Evicted
        assert cache.get("msg_2") is not None
        assert cache.get("msg_new") is not None

    def test_cache_update_existing(self) -> None:
        """Test updating an existing cache entry."""
        cache = EmbeddingCache()
        embedding1 = np.array([0.1, 0.2], dtype=np.float32)
        embedding2 = np.array([0.9, 0.8], dtype=np.float32)

        cache.put("test", embedding1)
        cache.put("test", embedding2)

        assert cache.size == 1
        cached = cache.get("test")
        assert cached is not None
        np.testing.assert_array_almost_equal(cached, embedding2)

    def test_cache_get_batch(self) -> None:
        """Test batch retrieval with partial hits."""
        cache = EmbeddingCache()

        # Add some embeddings
        cache.put("msg_1", np.array([1.0], dtype=np.float32))
        cache.put("msg_3", np.array([3.0], dtype=np.float32))

        texts = ["msg_1", "msg_2", "msg_3", "msg_4"]
        embeddings, miss_indices = cache.get_batch(texts)

        assert len(embeddings) == 4
        assert embeddings[0] is not None
        assert embeddings[1] is None
        assert embeddings[2] is not None
        assert embeddings[3] is None
        assert miss_indices == [1, 3]

    def test_cache_put_batch(self) -> None:
        """Test batch storage."""
        cache = EmbeddingCache()
        texts = ["msg_a", "msg_b", "msg_c"]
        embeddings = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

        cache.put_batch(texts, embeddings)

        assert cache.size == 3
        for i, text in enumerate(texts):
            cached = cache.get(text)
            assert cached is not None
            np.testing.assert_array_almost_equal(cached, embeddings[i])

    def test_cache_clear(self) -> None:
        """Test cache clearing."""
        cache = EmbeddingCache()
        cache.put("msg", np.array([1.0], dtype=np.float32))
        assert cache.size == 1

        cache.clear()

        assert cache.size == 0
        assert cache.get("msg") is None

    def test_cache_embedding_copy(self) -> None:
        """Test that cached embeddings are copied."""
        cache = EmbeddingCache()
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        cache.put("test", embedding)

        # Modify original
        embedding[0] = 999.0

        # Cached should be unchanged
        cached = cache.get("test")
        assert cached is not None
        assert cached[0] == 1.0

    def test_cache_deterministic_hashing(self) -> None:
        """Test that same text always gets same hash."""
        cache = EmbeddingCache()
        text = "test message for hashing"

        hash1 = cache._compute_hash(text)
        hash2 = cache._compute_hash(text)

        assert hash1 == hash2


class TestMockEmbeddingProvider:
    """Tests for MockEmbeddingProvider."""

    def test_mock_initialization(self) -> None:
        """Test mock provider initialization."""
        provider = MockEmbeddingProvider(embedding_dim=128)

        assert provider.embedding_dim == 128
        assert provider.model_name == "mock-model"

    def test_mock_embed_single(self) -> None:
        """Test single text embedding."""
        provider = MockEmbeddingProvider(embedding_dim=64)

        embedding = provider.embed_single("test message")

        assert embedding.shape == (64,)
        assert embedding.dtype == np.float32

    def test_mock_embed_batch(self) -> None:
        """Test batch text embedding."""
        provider = MockEmbeddingProvider(embedding_dim=32)
        texts = ["message 1", "message 2", "message 3"]

        embeddings = provider.embed(texts)

        assert embeddings.shape == (3, 32)
        assert embeddings.dtype == np.float32

    def test_mock_embed_empty(self) -> None:
        """Test embedding empty list."""
        provider = MockEmbeddingProvider(embedding_dim=16)

        embeddings = provider.embed([])

        assert embeddings.shape == (0, 16)

    def test_mock_deterministic(self) -> None:
        """Test that same text always gives same embedding."""
        provider = MockEmbeddingProvider(embedding_dim=64)

        embedding1 = provider.embed_single("test message")
        embedding2 = provider.embed_single("test message")

        np.testing.assert_array_equal(embedding1, embedding2)

    def test_mock_different_texts_different_embeddings(self) -> None:
        """Test that different texts give different embeddings."""
        provider = MockEmbeddingProvider(embedding_dim=64)

        embedding1 = provider.embed_single("message one")
        embedding2 = provider.embed_single("message two")

        assert not np.allclose(embedding1, embedding2)

    def test_mock_l2_normalized(self) -> None:
        """Test that mock embeddings are L2 normalized."""
        provider = MockEmbeddingProvider(embedding_dim=128)

        embedding = provider.embed_single("test message")
        norm = np.linalg.norm(embedding)

        np.testing.assert_almost_equal(norm, 1.0, decimal=5)


class TestEmbeddingService:
    """Tests for EmbeddingService high-level interface."""

    @pytest.fixture
    def mock_service(self) -> EmbeddingService:
        """Create a mock embedding service for testing."""
        return EmbeddingService.create_mock(embedding_dim=64)

    def test_service_creation(self, mock_service: EmbeddingService) -> None:
        """Test service creation with mock provider."""
        assert mock_service.embedding_dim == 64
        assert mock_service.cache is not None
        assert mock_service.stats.total_embedded == 0

    def test_embed_texts(self, mock_service: EmbeddingService) -> None:
        """Test embedding text list."""
        texts = ["message 1", "message 2", "message 3"]

        embeddings = mock_service.embed_texts(texts)

        assert embeddings.shape == (3, 64)
        assert mock_service.stats.total_embedded == 3
        assert mock_service.stats.batch_count == 1

    def test_embed_texts_empty(self, mock_service: EmbeddingService) -> None:
        """Test embedding empty list."""
        embeddings = mock_service.embed_texts([])

        assert embeddings.shape == (0, 64)
        assert mock_service.stats.total_embedded == 0

    def test_embed_single(self, mock_service: EmbeddingService) -> None:
        """Test embedding single text."""
        embedding = mock_service.embed_single("test message")

        assert embedding.shape == (64,)
        assert mock_service.stats.total_embedded == 1

    def test_embed_records(self, mock_service: EmbeddingService) -> None:
        """Test embedding log records."""
        records = [
            LogRecord(
                message="Error connecting to database",
                normalized="Error connecting to database",
                source="app.log",
                raw="ERROR Error connecting to database",
            ),
            LogRecord(
                message="Request timeout after 30s",
                normalized="Request timeout after <num>s",
                source="app.log",
                raw="WARN Request timeout after 30s",
            ),
        ]

        embeddings = mock_service.embed_records(records)

        assert embeddings.shape == (2, 64)
        assert mock_service.stats.total_embedded == 2

    def test_embed_records_uses_normalized(self, mock_service: EmbeddingService) -> None:
        """Test that normalized message is preferred."""
        record = LogRecord(
            message="Connection to 192.168.1.1 failed",
            normalized="Connection to <ip> failed",
            source="app.log",
            raw="ERROR Connection to 192.168.1.1 failed",
        )

        # Embed the record
        embedding1 = mock_service.embed_records([record])[0]

        # Embed the normalized text directly
        embedding2 = mock_service.embed_single("Connection to <ip> failed")

        # Should be the same since we use normalized
        np.testing.assert_array_equal(embedding1, embedding2)

    def test_embed_records_fallback_to_message(
        self, mock_service: EmbeddingService
    ) -> None:
        """Test fallback to message when normalized is None."""
        record = LogRecord(
            message="Plain message without normalization",
            source="app.log",
            raw="INFO Plain message without normalization",
        )

        embedding1 = mock_service.embed_records([record])[0]
        embedding2 = mock_service.embed_single("Plain message without normalization")

        np.testing.assert_array_equal(embedding1, embedding2)

    def test_embed_records_empty(self, mock_service: EmbeddingService) -> None:
        """Test embedding empty record list."""
        embeddings = mock_service.embed_records([])

        assert embeddings.shape == (0, 64)

    def test_cache_hit(self, mock_service: EmbeddingService) -> None:
        """Test that cache hits are counted."""
        text = "repeated message"

        # First call - cache miss
        mock_service.embed_single(text)
        assert mock_service.stats.cache_misses == 1
        assert mock_service.stats.cache_hits == 0

        # Second call - cache hit
        mock_service.embed_single(text)
        assert mock_service.stats.cache_hits == 1

    def test_cache_disabled(self) -> None:
        """Test embedding with cache disabled."""
        service = EmbeddingService(
            provider=MockEmbeddingProvider(embedding_dim=64),
            cache=None,
        )

        # Embed same text twice
        text = "test message"
        service.embed_single(text)
        service.embed_single(text)

        # Both should be cache misses (no caching)
        assert service.stats.cache_misses == 2
        assert service.stats.cache_hits == 0

    def test_no_cache_flag(self, mock_service: EmbeddingService) -> None:
        """Test bypassing cache with use_cache=False."""
        text = "test message"

        # First call with cache
        mock_service.embed_single(text, use_cache=True)
        assert mock_service.cache is not None
        assert mock_service.cache.size == 1

        # Second call without cache - still hits because cache exists
        mock_service.embed_single(text, use_cache=False)
        # This should be a miss since we're not using cache
        assert mock_service.stats.cache_misses == 2

    def test_batch_partial_cache_hits(self, mock_service: EmbeddingService) -> None:
        """Test batch embedding with partial cache hits."""
        # Cache some embeddings
        mock_service.embed_texts(["msg_1", "msg_3"])

        # Reset stats
        mock_service.reset_stats()

        # Embed batch with some cached
        mock_service.embed_texts(["msg_1", "msg_2", "msg_3", "msg_4"])

        assert mock_service.stats.cache_hits == 2  # msg_1, msg_3
        assert mock_service.stats.cache_misses == 2  # msg_2, msg_4
        assert mock_service.stats.total_embedded == 4

    def test_clear_cache(self, mock_service: EmbeddingService) -> None:
        """Test cache clearing."""
        mock_service.embed_texts(["msg_1", "msg_2"])
        assert mock_service.cache is not None
        assert mock_service.cache.size == 2

        mock_service.clear_cache()

        assert mock_service.cache.size == 0

    def test_reset_stats(self, mock_service: EmbeddingService) -> None:
        """Test stats reset."""
        mock_service.embed_texts(["msg_1", "msg_2"])
        assert mock_service.stats.total_embedded == 2

        mock_service.reset_stats()

        assert mock_service.stats.total_embedded == 0
        assert mock_service.stats.batch_count == 0


class TestGlobalEmbeddingService:
    """Tests for global embedding service singleton."""

    def test_set_and_get_service(self) -> None:
        """Test setting and getting global service."""
        mock_service = EmbeddingService.create_mock(embedding_dim=128)

        set_embedding_service(mock_service)
        retrieved = get_embedding_service()

        assert retrieved is mock_service
        assert retrieved.embedding_dim == 128


class TestEmbeddingIntegration:
    """Integration tests for embedding with preprocessing."""

    @pytest.fixture
    def service(self) -> EmbeddingService:
        """Create a mock embedding service."""
        return EmbeddingService.create_mock(embedding_dim=64)

    def test_embed_realistic_logs(self, service: EmbeddingService) -> None:
        """Test embedding realistic log messages."""
        now = datetime.now(timezone.utc)
        records = [
            LogRecord(
                message="Connection refused to host db-primary.internal:5432",
                normalized="Connection refused to host <path>",
                level="ERROR",
                source="/var/log/app.log",
                raw="2024-01-15 10:30:00 ERROR Connection refused to host db-primary.internal:5432",
                timestamp=now,
            ),
            LogRecord(
                message="Request completed in 245ms",
                normalized="Request completed in <num>ms",
                level="INFO",
                source="/var/log/app.log",
                raw="2024-01-15 10:30:01 INFO Request completed in 245ms",
                timestamp=now,
            ),
            LogRecord(
                message="Out of memory: Kill process 12345",
                normalized="Out of memory: Kill process <num>",
                level="ERROR",
                source="/var/log/system.log",
                raw="Jan 15 10:30:02 kernel: Out of memory: Kill process 12345",
                timestamp=now,
            ),
        ]

        embeddings = service.embed_records(records)

        assert embeddings.shape == (3, 64)
        # All embeddings should be normalized (L2 norm ≈ 1)
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_similar_logs_similar_embeddings(self, service: EmbeddingService) -> None:
        """Test that similar logs have similar embeddings."""
        # Note: With mock provider, similarity is based on hash
        # In real scenario, similar messages would have similar embeddings
        text1 = "Connection timeout error"
        text2 = "Connection timeout error"

        embedding1 = service.embed_single(text1)
        embedding2 = service.embed_single(text2)

        # Identical texts should have identical embeddings
        np.testing.assert_array_equal(embedding1, embedding2)

    def test_embedding_throughput_tracking(self, service: EmbeddingService) -> None:
        """Test that throughput is tracked correctly."""
        # Generate many embeddings
        texts = [f"message_{i}" for i in range(100)]
        service.embed_texts(texts)

        assert service.stats.total_embedded == 100
        assert service.stats.throughput > 0
        assert service.stats.total_time_seconds > 0

    def test_embedding_batch_efficiency(self, service: EmbeddingService) -> None:
        """Test that batching is more efficient than single calls."""
        texts = [f"message_{i}" for i in range(10)]

        # Single batch
        service.embed_texts(texts)
        batch_stats = service.stats.to_dict()

        service.reset_stats()

        # Individual calls
        for text in texts:
            service.embed_single(text)

        single_stats = service.stats.to_dict()

        # Batch should have fewer batches
        assert batch_stats["batch_count"] < single_stats["batch_count"]
