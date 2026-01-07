"""
gRPC server for the ML engine.

This server exposes the ML capabilities to the Go agent via gRPC.
Implements the MLService as defined in proto/ml/v1/ml_service.proto.

Design Patterns:
- Facade Pattern: Simple interface to complex ML subsystems
- Service Locator: Lazy initialization of ML components
- Observer: Health monitoring

SOLID Principles:
- Single Responsibility: Server handles only gRPC concerns
- Dependency Inversion: Components injected via config
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import grpc
import numpy as np

from sentinel_ml.clustering import ClusteringResult, ClusteringService, ClusterStats
from sentinel_ml.config import Config, get_config
from sentinel_ml.embedding import EmbeddingService, EmbeddingStats
from sentinel_ml.logging import get_logger, setup_logging
from sentinel_ml.models import LogRecord
from sentinel_ml.novelty import NoveltyResult, NoveltyService, NoveltyStats
from sentinel_ml.preprocessing import PreprocessingService
from sentinel_ml.vectorstore import SearchResult, VectorStore, VectorStoreStats

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass
class ServerMetrics:
    """Metrics for the gRPC server."""

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    records_processed: int = 0
    start_time: float = field(default_factory=time.time)
    last_request_time: float | None = None

    def record_request(self, success: bool, record_count: int = 0) -> None:
        """Record a request."""
        self.requests_total += 1
        if success:
            self.requests_success += 1
            self.records_processed += record_count
        else:
            self.requests_failed += 1
        self.last_request_time = time.time()

    @property
    def uptime_seconds(self) -> float:
        """Get server uptime."""
        return time.time() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requests_total": self.requests_total,
            "requests_success": self.requests_success,
            "requests_failed": self.requests_failed,
            "records_processed": self.records_processed,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "last_request_time": self.last_request_time,
        }


@dataclass
class ComponentHealth:
    """Health status of a component."""

    name: str
    healthy: bool
    message: str = ""


class MLServiceServicer:
    """
    Implementation of the ML gRPC service.

    This class handles all ML-related requests from the Go agent.
    Implements preprocessing, embedding, vector search (M2), clustering (M3),
    and novelty detection (M4).
    LLM-powered explanations will be added in subsequent milestones.
    """

    VERSION = "0.4.0"

    def __init__(
        self,
        config: Config,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        clustering_service: ClusteringService | None = None,
        novelty_service: NoveltyService | None = None,
    ) -> None:
        """
        Initialize the servicer.

        Args:
            config: Server configuration.
            embedding_service: Optional embedding service (for testing).
            vector_store: Optional vector store (for testing).
            clustering_service: Optional clustering service (for testing).
            novelty_service: Optional novelty service (for testing).
        """
        self.config = config
        self._preprocessing = PreprocessingService()
        self._metrics = ServerMetrics()
        self._lock = threading.Lock()

        # Embedding service (lazy loaded if not provided)
        self._embedding_service = embedding_service
        self._embedding_initialized = embedding_service is not None

        # Vector store (lazy loaded if not provided)
        self._vector_store = vector_store
        self._vector_store_initialized = vector_store is not None

        # Clustering service (lazy loaded if not provided)
        self._clustering_service = clustering_service
        self._clustering_initialized = clustering_service is not None

        # Novelty detection service (lazy loaded if not provided)
        self._novelty_service = novelty_service
        self._novelty_initialized = novelty_service is not None
        self._novelty_fitted = False

        # Future components
        self._llm_client = None

        logger.info(
            "ml_servicer_initialized",
            version=self.VERSION,
            embedding_initialized=self._embedding_initialized,
            vector_store_initialized=self._vector_store_initialized,
            clustering_initialized=self._clustering_initialized,
            novelty_initialized=self._novelty_initialized,
        )

    def _ensure_embedding_service(self) -> EmbeddingService:
        """Lazy initialize embedding service."""
        if not self._embedding_initialized:
            logger.info("initializing_embedding_service")
            self._embedding_service = EmbeddingService.from_config(self.config.embedding)
            self._embedding_initialized = True
        assert self._embedding_service is not None
        return self._embedding_service

    def _ensure_vector_store(self) -> VectorStore:
        """Lazy initialize vector store."""
        if not self._vector_store_initialized:
            logger.info("initializing_vector_store")
            self._vector_store = VectorStore.from_config(self.config.vector_store)
            self._vector_store_initialized = True
        assert self._vector_store is not None
        return self._vector_store

    def _ensure_clustering_service(self) -> ClusteringService:
        """Lazy initialize clustering service."""
        if not self._clustering_initialized:
            logger.info("initializing_clustering_service")
            self._clustering_service = ClusteringService.from_config(
                config=self.config.clustering,
                use_mock=False,
            )
            self._clustering_initialized = True
        assert self._clustering_service is not None
        return self._clustering_service

    def _ensure_novelty_service(self) -> NoveltyService:
        """Lazy initialize novelty detection service."""
        if not self._novelty_initialized:
            logger.info("initializing_novelty_service")
            self._novelty_service = NoveltyService.from_config(
                config=self.config.novelty,
                use_mock=False,
            )
            self._novelty_initialized = True
        assert self._novelty_service is not None
        return self._novelty_service

    def preprocess_records(
        self,
        records: list[dict[str, Any]],
    ) -> list[LogRecord]:
        """
        Preprocess raw log records.

        Args:
            records: List of raw record dictionaries.

        Returns:
            List of preprocessed LogRecord objects.
        """
        log_records = []
        for data in records:
            record = LogRecord(
                id=data.get("id"),
                message=data.get("message", data.get("raw", "")),
                raw=data.get("raw", data.get("message", "")),
                normalized=data.get("normalized"),
                level=data.get("level"),
                source=data.get("source", "unknown"),
                timestamp=data.get("timestamp"),
                attrs=data.get("attrs", {}),
            )
            log_records.append(record)

        processed = self._preprocessing.preprocess_batch(log_records)

        with self._lock:
            self._metrics.record_request(True, len(processed))

        logger.info(
            "records_preprocessed",
            input_count=len(records),
            output_count=len(processed),
        )

        return processed

    def embed_records(
        self,
        records: list[LogRecord],
        use_cache: bool = True,
    ) -> tuple[np.ndarray, int]:
        """
        Embed log records.

        Args:
            records: List of preprocessed LogRecords.
            use_cache: Whether to use embedding cache.

        Returns:
            Tuple of (embeddings array, cache_hits count).
        """
        embedding_service = self._ensure_embedding_service()

        # Get initial cache hits
        initial_hits = embedding_service.stats.cache_hits

        embeddings = embedding_service.embed_records(records, use_cache=use_cache)
        cache_hits = embedding_service.stats.cache_hits - initial_hits

        with self._lock:
            self._metrics.record_request(True, len(records))

        logger.info(
            "records_embedded",
            count=len(records),
            embedding_dim=embeddings.shape[1] if embeddings.size > 0 else 0,
            cache_hits=cache_hits,
        )

        return embeddings, cache_hits

    def add_to_store(
        self,
        embeddings: np.ndarray,
        records: list[LogRecord] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Add embeddings to vector store.

        Args:
            embeddings: Embedding vectors.
            records: Optional log records for metadata.
            ids: Optional external IDs.

        Returns:
            List of assigned IDs.
        """
        vector_store = self._ensure_vector_store()

        added_ids = vector_store.add(embeddings, records=records, ids=ids)

        logger.info(
            "embeddings_added_to_store",
            count=len(added_ids),
            total_vectors=vector_store.size,
        )

        return added_ids

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query embedding vector.
            k: Number of results to return.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of SearchResult objects.
        """
        vector_store = self._ensure_vector_store()

        results = vector_store.search(query_embedding, k=k, min_similarity=min_similarity)

        logger.debug(
            "vector_search_completed",
            k=k,
            results_count=len(results),
            min_similarity=min_similarity,
        )

        return results

    def ingest_and_embed(
        self,
        records: list[dict[str, Any]],
        store: bool = True,
        use_cache: bool = True,
    ) -> tuple[list[LogRecord], np.ndarray, list[str]]:
        """
        Full pipeline: preprocess, embed, and optionally store.

        Args:
            records: Raw log record dictionaries.
            store: Whether to add to vector store.
            use_cache: Whether to use embedding cache.

        Returns:
            Tuple of (processed records, embeddings, vector IDs).
        """
        # Preprocess
        processed = self.preprocess_records(records)

        if not processed:
            return [], np.array([]), []

        # Embed
        embeddings, _ = self.embed_records(processed, use_cache=use_cache)

        # Store
        ids: list[str] = []
        if store:
            ids = self.add_to_store(embeddings, records=processed)

        logger.info(
            "ingest_and_embed_completed",
            input_count=len(records),
            processed_count=len(processed),
            stored=store,
        )

        return processed, embeddings, ids

    def cluster_logs(
        self,
        embeddings: np.ndarray,
        records: list[LogRecord] | None = None,
        messages: list[str] | None = None,
    ) -> ClusteringResult:
        """
        Cluster log embeddings using HDBSCAN.

        Args:
            embeddings: Embedding vectors to cluster.
            records: Optional log records for metadata extraction.
            messages: Optional message list for representative selection.

        Returns:
            ClusteringResult with labels, summaries, and statistics.
        """
        clustering_service = self._ensure_clustering_service()

        result = clustering_service.cluster(
            embeddings=embeddings,
            records=records,
            messages=messages,
        )

        with self._lock:
            self._metrics.record_request(True, len(embeddings))

        logger.info(
            "logs_clustered",
            n_samples=len(embeddings),
            n_clusters=result.n_clusters,
            n_noise=result.n_noise,
            clustering_time_seconds=result.clustering_time_seconds,
        )

        return result

    def ingest_embed_and_cluster(
        self,
        records: list[dict[str, Any]],
        store: bool = True,
        use_cache: bool = True,
    ) -> tuple[list[LogRecord], np.ndarray, list[str], ClusteringResult]:
        """
        Full pipeline: preprocess, embed, store, and cluster.

        Args:
            records: Raw log record dictionaries.
            store: Whether to add to vector store.
            use_cache: Whether to use embedding cache.

        Returns:
            Tuple of (processed records, embeddings, vector IDs, clustering result).
        """
        # Use existing ingest_and_embed
        processed, embeddings, ids = self.ingest_and_embed(
            records=records,
            store=store,
            use_cache=use_cache,
        )

        if len(processed) == 0:
            return (
                [],
                np.array([]),
                [],
                ClusteringResult(
                    labels=np.array([], dtype=np.int32),
                    n_clusters=0,
                    n_noise=0,
                    summaries=[],
                    algorithm="hdbscan",
                    parameters={},
                    clustering_time_seconds=0.0,
                ),
            )

        # Cluster
        clustering_result = self.cluster_logs(embeddings, records=processed)

        # Update vector store with cluster IDs
        vector_store = self._ensure_vector_store()
        for idx, record_id in enumerate(ids):
            cluster_label = int(clustering_result.labels[idx])
            if cluster_label >= 0:
                vector_store.update_cluster_id(record_id, str(cluster_label))

        logger.info(
            "ingest_embed_and_cluster_completed",
            input_count=len(records),
            processed_count=len(processed),
            n_clusters=clustering_result.n_clusters,
            stored=store,
        )

        return processed, embeddings, ids, clustering_result

    def fit_novelty_detector(
        self,
        embeddings: np.ndarray,
        records: list[LogRecord] | None = None,
    ) -> None:
        """
        Fit the novelty detector on reference embeddings.

        This establishes the baseline of "normal" log patterns.
        Subsequent calls to detect_novelty will compare against this baseline.

        Args:
            embeddings: Reference embeddings representing normal patterns.
            records: Optional log records for context.
        """
        novelty_service = self._ensure_novelty_service()

        novelty_service.fit(embeddings, records=records)
        self._novelty_fitted = True

        logger.info(
            "novelty_detector_fitted",
            n_reference_samples=len(embeddings),
            embedding_dim=embeddings.shape[1] if embeddings.size > 0 else 0,
        )

    def detect_novelty(
        self,
        embeddings: np.ndarray,
        records: list[LogRecord] | None = None,
        messages: list[str] | None = None,
        threshold: float | None = None,
    ) -> NoveltyResult:
        """
        Detect novel log patterns in embeddings.

        Must call fit_novelty_detector first to establish baseline.

        Args:
            embeddings: Embedding vectors to analyze.
            records: Optional log records for metadata extraction.
            messages: Optional message list for context.
            threshold: Override default novelty threshold.

        Returns:
            NoveltyResult with scores and novel sample details.
        """
        novelty_service = self._ensure_novelty_service()

        result = novelty_service.detect(
            embeddings=embeddings,
            records=records,
            messages=messages,
            threshold=threshold,
        )

        with self._lock:
            self._metrics.record_request(True, len(embeddings))

        logger.info(
            "novelty_detection_completed",
            n_samples=len(embeddings),
            n_novel=result.n_novel,
            n_normal=result.n_normal,
            novelty_rate=round(result.n_novel / len(embeddings), 4)
            if len(embeddings) > 0
            else 0.0,
            threshold=result.threshold,
        )

        return result

    def ingest_embed_and_detect_novelty(
        self,
        records: list[dict[str, Any]],
        reference_embeddings: np.ndarray | None = None,
        store: bool = True,
        use_cache: bool = True,
        threshold: float | None = None,
    ) -> tuple[list[LogRecord], np.ndarray, list[str], NoveltyResult]:
        """
        Full pipeline: preprocess, embed, store, and detect novelty.

        If reference_embeddings is not provided, uses the vector store
        contents as reference (requires store to have data).

        Args:
            records: Raw log record dictionaries.
            reference_embeddings: Optional baseline embeddings for novelty.
            store: Whether to add to vector store.
            use_cache: Whether to use embedding cache.
            threshold: Override default novelty threshold.

        Returns:
            Tuple of (processed records, embeddings, vector IDs, novelty result).
        """
        # Use existing ingest_and_embed
        processed, embeddings, ids = self.ingest_and_embed(
            records=records,
            store=store,
            use_cache=use_cache,
        )

        if len(processed) == 0:
            return (
                [],
                np.array([]),
                [],
                NoveltyResult(
                    scores=np.array([], dtype=np.float32),
                    is_novel=np.array([], dtype=np.bool_),
                    novel_scores=[],
                    n_novel=0,
                    n_normal=0,
                    threshold=threshold or self.config.novelty.threshold,
                    detection_time_seconds=0.0,
                    algorithm="knn_density",
                    parameters={},
                ),
            )

        # Fit novelty detector if reference provided and not fitted
        self._ensure_novelty_service()
        if reference_embeddings is not None and not self._novelty_fitted:
            self.fit_novelty_detector(reference_embeddings)
        elif not self._novelty_fitted:
            # Use first batch as reference
            self.fit_novelty_detector(embeddings)

        # Detect novelty
        novelty_result = self.detect_novelty(
            embeddings,
            records=processed,
            threshold=threshold,
        )

        logger.info(
            "ingest_embed_and_detect_novelty_completed",
            input_count=len(records),
            processed_count=len(processed),
            n_novel=novelty_result.n_novel,
            stored=store,
        )

        return processed, embeddings, ids, novelty_result

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get embedding service statistics."""
        if not self._embedding_initialized or self._embedding_service is None:
            return EmbeddingStats().to_dict()
        return self._embedding_service.stats.to_dict()

    def get_vector_store_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        if not self._vector_store_initialized or self._vector_store is None:
            return VectorStoreStats().to_dict()
        return self._vector_store.stats.to_dict()

    def get_clustering_stats(self) -> dict[str, Any]:
        """Get clustering service statistics."""
        if not self._clustering_initialized or self._clustering_service is None:
            return ClusterStats().to_dict()
        return self._clustering_service.stats.to_dict()

    def get_novelty_stats(self) -> dict[str, Any]:
        """Get novelty detection service statistics."""
        if not self._novelty_initialized or self._novelty_service is None:
            return NoveltyStats().to_dict()
        return self._novelty_service.stats.to_dict()

    def health_check(self, detailed: bool = False) -> dict[str, Any]:
        """
        Check service health.

        Args:
            detailed: Include component-level health.

        Returns:
            Health status dictionary.
        """
        components = []
        overall_healthy = True

        # Check preprocessing service
        components.append(
            ComponentHealth(
                name="preprocessing",
                healthy=True,
                message="Ready",
            )
        )

        # Check embedding model (future)
        if self._embedding_initialized and self._embedding_service is not None:
            components.append(
                ComponentHealth(
                    name="embedding_service",
                    healthy=True,
                    message=f"Model loaded ({self._embedding_service.provider.model_name})",
                )
            )
        else:
            components.append(
                ComponentHealth(
                    name="embedding_service",
                    healthy=True,  # Not loaded is OK - lazy initialization
                    message="Ready (lazy load)",
                )
            )

        # Check vector store
        if self._vector_store_initialized and self._vector_store is not None:
            components.append(
                ComponentHealth(
                    name="vector_store",
                    healthy=True,
                    message=f"Store ready ({self._vector_store.size} vectors)",
                )
            )
        else:
            components.append(
                ComponentHealth(
                    name="vector_store",
                    healthy=True,  # Not loaded is OK - lazy initialization
                    message="Ready (lazy load)",
                )
            )

        # Check clustering service
        if self._clustering_initialized and self._clustering_service is not None:
            stats = self._clustering_service.stats
            components.append(
                ComponentHealth(
                    name="clustering_service",
                    healthy=True,
                    message=f"Ready ({stats.n_clusters_found} clusters found)",
                )
            )
        else:
            components.append(
                ComponentHealth(
                    name="clustering_service",
                    healthy=True,  # Not loaded is OK - lazy initialization
                    message="Ready (lazy load)",
                )
            )

        # Check novelty detection service
        if self._novelty_initialized and self._novelty_service is not None:
            stats = self._novelty_service.stats
            fitted_msg = "fitted" if self._novelty_fitted else "not fitted"
            components.append(
                ComponentHealth(
                    name="novelty_service",
                    healthy=True,
                    message=f"Ready ({fitted_msg}, {stats.total_analyzed} analyzed)",
                )
            )
        else:
            components.append(
                ComponentHealth(
                    name="novelty_service",
                    healthy=True,  # Not loaded is OK - lazy initialization
                    message="Ready (lazy load)",
                )
            )

        result: dict[str, Any] = {
            "healthy": overall_healthy,
            "version": self.VERSION,
        }

        if detailed:
            result["components"] = [
                {"name": c.name, "healthy": c.healthy, "message": c.message} for c in components
            ]
            result["metrics"] = self._metrics.to_dict()
            result["embedding_stats"] = self.get_embedding_stats()
            result["vector_store_stats"] = self.get_vector_store_stats()
            result["clustering_stats"] = self.get_clustering_stats()
            result["novelty_stats"] = self.get_novelty_stats()

        return result

    def get_metrics(self) -> dict[str, Any]:
        """Get server metrics."""
        with self._lock:
            return self._metrics.to_dict()


class GRPCServer:
    """
    High-level gRPC server wrapper.

    Provides lifecycle management, graceful shutdown, and health monitoring.
    """

    def __init__(
        self,
        config: Config | None = None,
        servicer: MLServiceServicer | None = None,
    ) -> None:
        """
        Initialize the server.

        Args:
            config: Server configuration.
            servicer: Custom servicer instance.
        """
        self.config = config or get_config()
        self.servicer = servicer or MLServiceServicer(self.config)
        self._server: grpc.Server | None = None
        self._shutdown_event = threading.Event()
        self._started = False

        logger.info(
            "grpc_server_created",
            host=self.config.server.host,
            port=self.config.server.port,
        )

    def _create_server(self) -> grpc.Server:
        """Create the underlying gRPC server."""
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.config.server.max_workers),
            options=[
                ("grpc.max_send_message_length", self.config.server.max_message_size),
                ("grpc.max_receive_message_length", self.config.server.max_message_size),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
            ],
        )

        # TODO: Add servicer once proto is compiled
        # ml_pb2_grpc.add_MLServiceServicer_to_server(self.servicer, server)

        address = f"{self.config.server.host}:{self.config.server.port}"
        server.add_insecure_port(address)

        return server

    def start(self) -> None:
        """Start the server."""
        if self._started:
            logger.warning("server_already_started")
            return

        self._server = self._create_server()
        self._server.start()
        self._started = True

        logger.info(
            "grpc_server_started",
            host=self.config.server.host,
            port=self.config.server.port,
            max_workers=self.config.server.max_workers,
        )

    def stop(self, grace: float = 5.0) -> None:
        """
        Stop the server gracefully.

        Args:
            grace: Grace period in seconds.
        """
        if not self._started or self._server is None:
            return

        logger.info("grpc_server_stopping", grace_period=grace)
        self._server.stop(grace=grace)
        self._started = False
        self._shutdown_event.set()
        logger.info("grpc_server_stopped")

    def wait_for_termination(self, timeout: float | None = None) -> bool:
        """
        Wait for the server to terminate.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if terminated, False if timeout.
        """
        if self._server is None:
            return True
        return self._shutdown_event.wait(timeout=timeout)

    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._started

    def health_check(self, detailed: bool = False) -> dict[str, Any]:
        """Get health status."""
        return self.servicer.health_check(detailed)


def create_server(config: Config | None = None) -> GRPCServer:
    """
    Create a new gRPC server instance.

    Args:
        config: Server configuration.

    Returns:
        Configured GRPCServer instance.
    """
    return GRPCServer(config)


def serve(
    config: Config | None = None,
    shutdown_handler: Callable[[], None] | None = None,
) -> None:
    """
    Start the gRPC server and block until shutdown.

    Args:
        config: Server configuration.
        shutdown_handler: Optional callback on shutdown.
    """
    config = config or get_config()
    setup_logging()

    server = create_server(config)
    server.start()

    # Handle graceful shutdown
    def signal_handler(_signum: int, _frame: object) -> None:
        logger.info("shutdown_signal_received")
        server.stop(grace=5.0)
        if shutdown_handler:
            shutdown_handler()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Block until shutdown
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")
        server.stop(grace=5.0)


def main() -> None:
    """Entry point for the ML server."""
    logger.info("starting_ml_server")
    try:
        serve()
    except Exception as e:
        logger.error("server_failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
