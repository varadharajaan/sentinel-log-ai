"""Tests for the gRPC server module."""

from unittest.mock import patch

import numpy as np
import pytest

from sentinel_ml.config import Config, ServerConfig
from sentinel_ml.embedding import EmbeddingService
from sentinel_ml.server import (
    ComponentHealth,
    GRPCServer,
    MLServiceServicer,
    ServerMetrics,
    create_server,
)
from sentinel_ml.vectorstore import VectorStore


class TestServerMetrics:
    """Test ServerMetrics dataclass."""

    def test_initial_values(self) -> None:
        """Test initial metric values."""
        metrics = ServerMetrics()

        assert metrics.requests_total == 0
        assert metrics.requests_success == 0
        assert metrics.requests_failed == 0
        assert metrics.records_processed == 0
        assert metrics.last_request_time is None

    def test_record_successful_request(self) -> None:
        """Test recording a successful request."""
        metrics = ServerMetrics()

        metrics.record_request(success=True, record_count=10)

        assert metrics.requests_total == 1
        assert metrics.requests_success == 1
        assert metrics.requests_failed == 0
        assert metrics.records_processed == 10
        assert metrics.last_request_time is not None

    def test_record_failed_request(self) -> None:
        """Test recording a failed request."""
        metrics = ServerMetrics()

        metrics.record_request(success=False)

        assert metrics.requests_total == 1
        assert metrics.requests_success == 0
        assert metrics.requests_failed == 1
        assert metrics.records_processed == 0

    def test_uptime_seconds(self) -> None:
        """Test uptime calculation."""
        metrics = ServerMetrics()

        # Uptime should be positive
        assert metrics.uptime_seconds >= 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = ServerMetrics()
        metrics.record_request(success=True, record_count=5)

        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["requests_total"] == 1
        assert d["requests_success"] == 1
        assert d["records_processed"] == 5
        assert "uptime_seconds" in d


class TestComponentHealth:
    """Test ComponentHealth dataclass."""

    def test_creation(self) -> None:
        """Test creating a component health."""
        health = ComponentHealth(
            name="preprocessing",
            healthy=True,
            message="Ready",
        )

        assert health.name == "preprocessing"
        assert health.healthy is True
        assert health.message == "Ready"

    def test_unhealthy_component(self) -> None:
        """Test unhealthy component."""
        health = ComponentHealth(
            name="embedding_model",
            healthy=False,
            message="Model failed to load",
        )

        assert health.healthy is False


class TestMLServiceServicer:
    """Test MLServiceServicer class."""

    @pytest.fixture
    def config(self) -> Config:
        """Create a test config."""
        return Config(
            server=ServerConfig(host="localhost", port=50051),
        )

    def test_initialization(self, config: Config) -> None:
        """Test servicer initialization."""
        servicer = MLServiceServicer(config)

        assert servicer.config == config
        assert servicer._preprocessing is not None

    def test_preprocess_records(self, config: Config) -> None:
        """Test preprocessing records."""
        servicer = MLServiceServicer(config)
        records = [
            {
                "message": "Connection failed",
                "raw": "ERROR Connection failed",
                "source": "app.log",
            },
            {
                "message": "Request completed",
                "raw": "INFO Request completed",
                "source": "app.log",
            },
        ]

        results = servicer.preprocess_records(records)

        assert len(results) == 2
        assert all(r.id is not None for r in results)
        assert all(r.normalized is not None for r in results)

    def test_preprocess_records_with_attrs(self, config: Config) -> None:
        """Test preprocessing records with attributes."""
        servicer = MLServiceServicer(config)
        records = [
            {
                "message": "User login",
                "raw": "User login",
                "source": "auth.log",
                "level": "INFO",
                "attrs": {"user_id": "12345"},
            },
        ]

        results = servicer.preprocess_records(records)

        assert len(results) == 1
        assert results[0].attrs.get("user_id") == "12345"

    def test_health_check_basic(self, config: Config) -> None:
        """Test basic health check."""
        servicer = MLServiceServicer(config)

        health = servicer.health_check(detailed=False)

        assert health["healthy"] is True
        assert health["version"] == MLServiceServicer.VERSION
        assert "components" not in health

    def test_health_check_detailed(self, config: Config) -> None:
        """Test detailed health check."""
        servicer = MLServiceServicer(config)

        health = servicer.health_check(detailed=True)

        assert health["healthy"] is True
        assert "components" in health
        assert "metrics" in health
        assert len(health["components"]) > 0

    def test_get_metrics(self, config: Config) -> None:
        """Test getting metrics."""
        servicer = MLServiceServicer(config)

        # Process some records to generate metrics
        servicer.preprocess_records([{"message": "test", "raw": "test", "source": "test"}])

        metrics = servicer.get_metrics()

        assert isinstance(metrics, dict)
        assert metrics["requests_total"] >= 1
        assert metrics["records_processed"] >= 1


class TestGRPCServer:
    """Test GRPCServer class."""

    @pytest.fixture
    def config(self) -> Config:
        """Create a test config."""
        return Config(
            server=ServerConfig(host="localhost", port=50099),  # Use non-standard port
        )

    def test_creation(self, config: Config) -> None:
        """Test server creation."""
        server = GRPCServer(config)

        assert server.config == config
        assert server.servicer is not None
        assert not server.is_running()

    def test_creation_with_custom_servicer(self, config: Config) -> None:
        """Test server creation with custom servicer."""
        servicer = MLServiceServicer(config)
        server = GRPCServer(config, servicer=servicer)

        assert server.servicer is servicer

    def test_start_stop(self, config: Config) -> None:
        """Test starting and stopping the server."""
        server = GRPCServer(config)

        server.start()
        assert server.is_running()

        server.stop(grace=0.1)
        assert not server.is_running()

    def test_double_start(self, config: Config) -> None:
        """Test that double start is handled gracefully."""
        server = GRPCServer(config)

        server.start()
        server.start()  # Should not raise

        assert server.is_running()
        server.stop(grace=0.1)

    def test_health_check(self, config: Config) -> None:
        """Test health check through server."""
        server = GRPCServer(config)
        server.start()

        try:
            health = server.health_check(detailed=True)
            assert health["healthy"] is True
        finally:
            server.stop(grace=0.1)

    def test_wait_for_termination_with_timeout(self, config: Config) -> None:
        """Test waiting for termination with timeout."""
        server = GRPCServer(config)
        server.start()

        # Should timeout immediately since server is still running
        result = server.wait_for_termination(timeout=0.1)
        assert result is False

        server.stop(grace=0.1)


class TestCreateServer:
    """Test create_server function."""

    def test_create_with_default_config(self) -> None:
        """Test creating server with default config."""
        with patch("sentinel_ml.server.get_config") as mock_get_config:
            mock_config = Config(
                server=ServerConfig(host="localhost", port=50100),
            )
            mock_get_config.return_value = mock_config

            server = create_server()

            assert server is not None
            assert isinstance(server, GRPCServer)

    def test_create_with_custom_config(self) -> None:
        """Test creating server with custom config."""
        config = Config(
            server=ServerConfig(host="0.0.0.0", port=50101),
        )

        server = create_server(config)

        assert server.config == config


class TestServerIntegration:
    """Integration tests for the server."""

    @pytest.fixture
    def server(self) -> GRPCServer:
        """Create and start a test server."""
        config = Config(
            server=ServerConfig(host="localhost", port=50102),
        )
        server = GRPCServer(config)
        server.start()
        yield server
        server.stop(grace=0.1)

    def test_preprocess_through_servicer(self, server: GRPCServer) -> None:
        """Test preprocessing through the servicer."""
        records = [
            {
                "message": "Error from 192.168.1.1",
                "raw": "2024-01-15 ERROR Error from 192.168.1.1",
                "source": "app.log",
            },
        ]

        results = server.servicer.preprocess_records(records)

        assert len(results) == 1
        assert results[0].normalized is not None
        # IP should be masked in normalized
        assert "192.168.1.1" not in results[0].normalized

    def test_metrics_after_processing(self, server: GRPCServer) -> None:
        """Test that metrics are updated after processing."""
        records = [{"message": "test", "raw": "test", "source": "test"} for _ in range(10)]

        server.servicer.preprocess_records(records)

        metrics = server.servicer.get_metrics()
        assert metrics["records_processed"] >= 10

    def test_health_during_operation(self, server: GRPCServer) -> None:
        """Test health check during operation."""
        # Process some records
        server.servicer.preprocess_records([{"message": "test", "raw": "test", "source": "test"}])

        health = server.health_check(detailed=True)

        assert health["healthy"] is True
        assert health["metrics"]["records_processed"] >= 1


class TestMLServiceServicerEmbedding:
    """Tests for MLServiceServicer embedding functionality."""

    @pytest.fixture
    def config(self) -> Config:
        """Create a test config."""
        return Config(
            server=ServerConfig(host="localhost", port=50051),
        )

    @pytest.fixture
    def mock_embedding_service(self) -> EmbeddingService:
        """Create a mock embedding service."""
        return EmbeddingService.create_mock(embedding_dim=64)

    @pytest.fixture
    def mock_vector_store(self) -> VectorStore:
        """Create a mock vector store."""
        return VectorStore.create_mock(dimension=64)

    @pytest.fixture
    def servicer_with_mocks(
        self,
        config: Config,
        mock_embedding_service: EmbeddingService,
        mock_vector_store: VectorStore,
    ) -> MLServiceServicer:
        """Create a servicer with mock embedding and vector store."""
        return MLServiceServicer(
            config,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )

    def test_embed_records(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test embedding records."""
        from sentinel_ml.models import LogRecord

        records = [
            LogRecord(
                message="Connection timeout",
                normalized="Connection timeout",
                source="app.log",
                raw="ERROR Connection timeout",
            ),
            LogRecord(
                message="Request completed",
                normalized="Request completed",
                source="app.log",
                raw="INFO Request completed",
            ),
        ]

        embeddings, cache_hits = servicer_with_mocks.embed_records(records)

        assert embeddings.shape == (2, 64)
        assert cache_hits == 0  # First call, no cache hits

    def test_embed_records_with_cache(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test embedding with cache hits."""
        from sentinel_ml.models import LogRecord

        record = LogRecord(
            message="Same message",
            normalized="Same message",
            source="app.log",
            raw="INFO Same message",
        )

        # First call - no cache
        servicer_with_mocks.embed_records([record])

        # Second call - should hit cache
        _, cache_hits = servicer_with_mocks.embed_records([record])
        assert cache_hits == 1

    def test_add_to_store(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test adding embeddings to vector store."""
        embeddings = np.random.randn(3, 64).astype(np.float32)

        ids = servicer_with_mocks.add_to_store(embeddings)

        assert len(ids) == 3
        assert servicer_with_mocks._vector_store is not None
        assert servicer_with_mocks._vector_store.size == 3

    def test_add_to_store_with_records(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test adding embeddings with metadata from records."""
        from sentinel_ml.models import LogRecord

        records = [
            LogRecord(
                message="Error occurred",
                source="app.log",
                raw="ERROR Error occurred",
                level="ERROR",
            ),
        ]
        embeddings = np.random.randn(1, 64).astype(np.float32)

        ids = servicer_with_mocks.add_to_store(embeddings, records=records)

        assert len(ids) == 1
        metadata = servicer_with_mocks._vector_store.get_by_id(ids[0])
        assert metadata is not None
        assert metadata.source == "app.log"
        assert metadata.level == "ERROR"

    def test_search(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test vector search."""
        # Add some vectors
        embeddings = np.random.randn(10, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        servicer_with_mocks.add_to_store(embeddings)

        # Search
        query = embeddings[0]
        results = servicer_with_mocks.search(query, k=3)

        assert len(results) <= 3
        assert all(0 <= r.similarity <= 1 for r in results)

    def test_search_empty_store(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test search on empty store."""
        query = np.random.randn(64).astype(np.float32)

        results = servicer_with_mocks.search(query, k=5)

        assert results == []

    def test_ingest_and_embed_pipeline(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test full ingest and embed pipeline."""
        records = [
            {
                "message": "Connection to 192.168.1.1 failed",
                "raw": "ERROR Connection to 192.168.1.1 failed",
                "source": "app.log",
            },
            {
                "message": "Request completed in 100ms",
                "raw": "INFO Request completed in 100ms",
                "source": "app.log",
            },
        ]

        processed, embeddings, ids = servicer_with_mocks.ingest_and_embed(records, store=True)

        assert len(processed) == 2
        assert embeddings.shape == (2, 64)
        assert len(ids) == 2
        assert servicer_with_mocks._vector_store.size == 2

    def test_ingest_and_embed_no_store(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test pipeline without storing."""
        records = [
            {"message": "test", "raw": "test", "source": "test"},
        ]

        processed, embeddings, ids = servicer_with_mocks.ingest_and_embed(records, store=False)

        assert len(processed) == 1
        assert embeddings.shape[0] == 1
        assert ids == []  # Not stored
        assert servicer_with_mocks._vector_store.size == 0

    def test_get_embedding_stats(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test getting embedding stats."""
        from sentinel_ml.models import LogRecord

        records = [LogRecord(message="test", normalized="test", source="test", raw="test")]
        servicer_with_mocks.embed_records(records)

        stats = servicer_with_mocks.get_embedding_stats()

        assert stats["total_embedded"] >= 1
        assert "cache_hit_rate" in stats

    def test_get_vector_store_stats(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test getting vector store stats."""
        embeddings = np.random.randn(5, 64).astype(np.float32)
        servicer_with_mocks.add_to_store(embeddings)

        stats = servicer_with_mocks.get_vector_store_stats()

        assert stats["total_vectors"] == 5
        assert stats["total_adds"] >= 1

    def test_health_check_with_embedding(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test health check includes embedding service."""
        health = servicer_with_mocks.health_check(detailed=True)

        assert health["healthy"] is True
        assert "embedding_stats" in health
        assert "vector_store_stats" in health

        # Find embedding component
        embedding_component = next(
            (c for c in health["components"] if "embedding" in c["name"].lower()),
            None,
        )
        assert embedding_component is not None
        assert embedding_component["healthy"] is True

    def test_health_check_with_vectors(self, servicer_with_mocks: MLServiceServicer) -> None:
        """Test health check shows vector count."""
        embeddings = np.random.randn(10, 64).astype(np.float32)
        servicer_with_mocks.add_to_store(embeddings)

        health = servicer_with_mocks.health_check(detailed=True)

        vector_component = next(
            (c for c in health["components"] if "vector" in c["name"].lower()),
            None,
        )
        assert vector_component is not None
        assert "10 vectors" in vector_component["message"]
