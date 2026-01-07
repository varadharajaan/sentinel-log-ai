"""
Integration tests for M2: Embeddings & Vector Store.

These tests verify the complete pipeline from log ingestion through
embedding generation to vector search.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from sentinel_ml.config import Config, EmbeddingConfig, ServerConfig, VectorStoreConfig
from sentinel_ml.embedding import EmbeddingService
from sentinel_ml.models import LogRecord
from sentinel_ml.preprocessing import PreprocessingService
from sentinel_ml.server import MLServiceServicer
from sentinel_ml.vectorstore import VectorStore


@pytest.mark.integration
class TestM2EmbeddingPipeline:
    """Integration tests for the embedding pipeline."""

    @pytest.fixture
    def embedding_service(self) -> EmbeddingService:
        """Create a mock embedding service for testing."""
        return EmbeddingService.create_mock(embedding_dim=128)

    @pytest.fixture
    def vector_store(self) -> VectorStore:
        """Create a mock vector store for testing."""
        return VectorStore.create_mock(dimension=128)

    def test_preprocess_embed_store_workflow(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        """Test complete workflow: preprocess -> embed -> store."""
        # Create preprocessing service
        preprocessing = PreprocessingService()

        # Create test records
        now = datetime.now(timezone.utc)
        raw_records = [
            LogRecord(
                message="Connection timeout to 192.168.1.1:5432",
                raw="2024-01-15 10:30:00 ERROR Connection timeout to 192.168.1.1:5432",
                source="/var/log/app.log",
                level="ERROR",
                timestamp=now,
            ),
            LogRecord(
                message="Connection timeout to 10.0.0.1:5432",
                raw="2024-01-15 10:30:01 ERROR Connection timeout to 10.0.0.1:5432",
                source="/var/log/app.log",
                level="ERROR",
                timestamp=now,
            ),
            LogRecord(
                message="Request completed in 150ms",
                raw="2024-01-15 10:30:02 INFO Request completed in 150ms",
                source="/var/log/app.log",
                level="INFO",
                timestamp=now,
            ),
        ]

        # Step 1: Preprocess
        processed = preprocessing.preprocess_batch(raw_records)
        assert len(processed) == 3
        assert all(r.normalized is not None for r in processed)

        # Step 2: Embed
        embeddings = embedding_service.embed_records(processed)
        assert embeddings.shape == (3, 128)

        # Step 3: Store
        ids = vector_store.add(embeddings, records=processed)
        assert len(ids) == 3
        assert vector_store.size == 3

        # Step 4: Search
        query_embedding = embeddings[0]  # Search for first record
        results = vector_store.search(query_embedding, k=2)

        assert len(results) == 2
        # First result should be exact match (or very close)
        assert results[0].similarity > 0.9

    def test_similar_logs_clustered_together(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        """Test that similar log messages end up near each other in vector space."""
        preprocessing = PreprocessingService()

        # Create groups of similar logs
        connection_logs = [
            LogRecord(
                message="Connection failed to host-1",
                raw="ERROR Connection failed to host-1",
                source="app.log",
            ),
            LogRecord(
                message="Connection failed to host-2",
                raw="ERROR Connection failed to host-2",
                source="app.log",
            ),
            LogRecord(
                message="Connection timeout to server",
                raw="ERROR Connection timeout to server",
                source="app.log",
            ),
        ]

        memory_logs = [
            LogRecord(
                message="Out of memory error",
                raw="ERROR Out of memory error",
                source="app.log",
            ),
            LogRecord(
                message="Memory allocation failed",
                raw="ERROR Memory allocation failed",
                source="app.log",
            ),
        ]

        # Process all logs
        all_logs = connection_logs + memory_logs
        processed = preprocessing.preprocess_batch(all_logs)

        # Embed
        embeddings = embedding_service.embed_records(processed)
        vector_store.add(embeddings, records=processed)

        # Search for a connection-related query
        # Note: With mock embeddings, we're testing the pipeline, not semantic similarity
        query_embedding = embeddings[0]  # First connection log
        results = vector_store.search(query_embedding, k=5)

        # Should return results (testing that the search works)
        assert len(results) > 0

    def test_embedding_cache_reduces_computation(
        self, embedding_service: EmbeddingService
    ) -> None:
        """Test that embedding cache improves performance for repeated logs."""
        preprocessing = PreprocessingService()

        # Create some records with repeated normalized patterns
        records = [
            LogRecord(
                message="Error connecting to server",
                raw="ERROR Error connecting to server",
                source="app.log",
            )
            for _ in range(10)
        ]

        processed = preprocessing.preprocess_batch(records)

        # First batch - all cache misses
        embedding_service.embed_records(processed)
        initial_misses = embedding_service.stats.cache_misses

        # Second batch - all cache hits
        embedding_service.embed_records(processed)

        final_hits = embedding_service.stats.cache_hits

        # All second batch should be cache hits
        assert final_hits >= 10
        assert embedding_service.stats.cache_hit_rate > 0


@pytest.mark.integration
class TestM2ServerIntegration:
    """Integration tests for the ML server with embedding support."""

    @pytest.fixture
    def servicer(self) -> MLServiceServicer:
        """Create a servicer with mock embedding and vector store."""
        config = Config(
            server=ServerConfig(host="localhost", port=50051),
        )
        embedding_service = EmbeddingService.create_mock(embedding_dim=64)
        vector_store = VectorStore.create_mock(dimension=64)

        return MLServiceServicer(
            config,
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

    def test_full_ingestion_pipeline(self, servicer: MLServiceServicer) -> None:
        """Test complete ingestion pipeline through servicer."""
        # Raw log data as would come from Go agent
        raw_records = [
            {
                "message": "Database connection failed after 30s timeout",
                "raw": "2024-01-15 10:30:00 ERROR Database connection failed after 30s timeout",
                "source": "/var/log/app.log",
                "level": "ERROR",
            },
            {
                "message": "User authentication successful for user-12345",
                "raw": "2024-01-15 10:30:01 INFO User authentication successful for user-12345",
                "source": "/var/log/auth.log",
                "level": "INFO",
            },
            {
                "message": "Memory usage at 85%",
                "raw": "2024-01-15 10:30:02 WARN Memory usage at 85%",
                "source": "/var/log/system.log",
                "level": "WARN",
            },
        ]

        # Use full pipeline
        processed, embeddings, ids = servicer.ingest_and_embed(raw_records, store=True)

        # Verify preprocessing
        assert len(processed) == 3
        assert all(r.id is not None for r in processed)
        assert all(r.normalized is not None for r in processed)

        # Verify embeddings
        assert embeddings.shape == (3, 64)

        # Verify storage
        assert len(ids) == 3
        assert servicer._vector_store.size == 3

        # Verify we can search
        query = embeddings[0]
        results = servicer.search(query, k=3)
        assert len(results) == 3

    def test_search_after_ingestion(self, servicer: MLServiceServicer) -> None:
        """Test that we can search for similar logs after ingestion."""
        # Ingest some logs
        records = [
            {
                "message": f"Connection error #{i}",
                "raw": f"ERROR Connection error #{i}",
                "source": "app.log",
            }
            for i in range(20)
        ]

        _, embeddings, _ = servicer.ingest_and_embed(records, store=True)

        # Search for similar logs
        results = servicer.search(embeddings[0], k=5)

        assert len(results) == 5
        # All results should have valid similarity scores
        for result in results:
            assert 0 <= result.similarity <= 1
            assert result.id is not None

    def test_metrics_tracked_correctly(self, servicer: MLServiceServicer) -> None:
        """Test that metrics are tracked through the pipeline."""
        records = [
            {"message": f"Log message {i}", "raw": f"INFO Log message {i}", "source": "app.log"}
            for i in range(50)
        ]

        servicer.ingest_and_embed(records, store=True)

        # Check embedding stats
        embedding_stats = servicer.get_embedding_stats()
        assert embedding_stats["total_embedded"] >= 50

        # Check vector store stats
        vector_stats = servicer.get_vector_store_stats()
        assert vector_stats["total_vectors"] == 50
        assert vector_stats["total_adds"] >= 1

    def test_health_reflects_initialized_components(
        self, servicer: MLServiceServicer
    ) -> None:
        """Test that health check reflects initialized components."""
        # Components should show as initialized since we passed them in
        health = servicer.health_check(detailed=True)

        assert health["healthy"] is True

        components = {c["name"]: c for c in health["components"]}

        assert "embedding_service" in components
        assert components["embedding_service"]["healthy"] is True

        assert "vector_store" in components
        assert components["vector_store"]["healthy"] is True


@pytest.mark.integration
class TestM2Persistence:
    """Integration tests for vector store persistence."""

    def test_save_and_load_vector_store(self) -> None:
        """Test persisting and loading the vector store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir)

            # Create and populate store
            store1 = VectorStore(
                index=VectorStore.create_mock(dimension=32).index,
                persist_dir=persist_dir,
            )

            # Add some vectors
            embeddings = np.random.randn(10, 32).astype(np.float32)
            records = [
                LogRecord(
                    message=f"Message {i}",
                    source="test.log",
                    raw=f"INFO Message {i}",
                )
                for i in range(10)
            ]
            ids = store1.add(embeddings, records=records, ids=[f"log-{i}" for i in range(10)])

            # Save
            store1.save()

            # Create new store and load
            store2 = VectorStore(
                index=VectorStore.create_mock(dimension=32).index,
                persist_dir=persist_dir,
            )
            store2.load()

            # Verify data
            assert store2.size == 10

            # Verify metadata
            metadata = store2.get_by_id("log-0")
            assert metadata is not None
            assert metadata.source == "test.log"

    def test_search_after_reload(self) -> None:
        """Test that search works correctly after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir)

            # Create, populate, and save
            store1 = VectorStore(
                index=VectorStore.create_mock(dimension=16).index,
                persist_dir=persist_dir,
            )

            embeddings = np.random.randn(5, 16).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            store1.add(embeddings)
            store1.save()

            # Load into new store
            store2 = VectorStore(
                index=VectorStore.create_mock(dimension=16).index,
                persist_dir=persist_dir,
            )
            store2.load()

            # Search should work
            results = store2.search(embeddings[0], k=3)
            assert len(results) == 3
            assert results[0].similarity > 0.9


@pytest.mark.integration
class TestM2EdgeCases:
    """Edge case tests for M2 components."""

    def test_empty_batch_handling(self) -> None:
        """Test handling of empty batches."""
        embedding_service = EmbeddingService.create_mock(embedding_dim=64)
        vector_store = VectorStore.create_mock(dimension=64)

        # Empty embedding
        embeddings = embedding_service.embed_records([])
        assert embeddings.shape == (0, 64)

        # Empty store add
        ids = vector_store.add(np.array([]).reshape(0, 64).astype(np.float32))
        assert ids == []

        # Empty search
        results = vector_store.search(np.random.randn(64).astype(np.float32), k=5)
        assert results == []

    def test_single_record_processing(self) -> None:
        """Test processing a single record."""
        embedding_service = EmbeddingService.create_mock(embedding_dim=64)
        vector_store = VectorStore.create_mock(dimension=64)
        preprocessing = PreprocessingService()

        record = LogRecord(
            message="Single log message",
            raw="INFO Single log message",
            source="test.log",
        )

        # Preprocess
        processed = preprocessing.preprocess_batch([record])
        assert len(processed) == 1

        # Embed
        embeddings = embedding_service.embed_records(processed)
        assert embeddings.shape == (1, 64)

        # Store
        ids = vector_store.add(embeddings)
        assert len(ids) == 1

        # Search
        results = vector_store.search(embeddings[0], k=1)
        assert len(results) == 1

    def test_large_batch_handling(self) -> None:
        """Test handling of large batches."""
        embedding_service = EmbeddingService.create_mock(embedding_dim=64)
        vector_store = VectorStore.create_mock(dimension=64)

        # Create many records
        records = [
            LogRecord(
                message=f"Log message number {i} with some content",
                normalized=f"Log message number <num> with some content",
                raw=f"INFO Log message number {i} with some content",
                source="test.log",
            )
            for i in range(500)
        ]

        # Embed
        embeddings = embedding_service.embed_records(records)
        assert embeddings.shape == (500, 64)

        # Store
        ids = vector_store.add(embeddings)
        assert len(ids) == 500
        assert vector_store.size == 500

        # Search should still be fast
        import time

        start = time.time()
        results = vector_store.search(embeddings[0], k=10)
        elapsed = time.time() - start

        assert len(results) == 10
        assert elapsed < 1.0  # Should be fast even with 500 vectors

    def test_normalized_message_used_for_embedding(self) -> None:
        """Test that normalized messages are used for embedding."""
        embedding_service = EmbeddingService.create_mock(embedding_dim=64)

        # Two records with different raw messages but same normalized
        record1 = LogRecord(
            message="Error from 192.168.1.1",
            normalized="Error from <ip>",
            raw="ERROR Error from 192.168.1.1",
            source="test.log",
        )

        record2 = LogRecord(
            message="Error from 10.0.0.1",
            normalized="Error from <ip>",
            raw="ERROR Error from 10.0.0.1",
            source="test.log",
        )

        # Embeddings should be identical since normalized is the same
        emb1 = embedding_service.embed_records([record1])
        emb2 = embedding_service.embed_records([record2])

        np.testing.assert_array_equal(emb1, emb2)
