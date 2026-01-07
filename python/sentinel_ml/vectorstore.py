"""
FAISS-based vector store for log embeddings.

This module provides a persistent vector store for log embeddings using FAISS,
enabling efficient similarity search and k-nearest neighbor queries.

Design Patterns:
- Repository Pattern: Abstract storage operations
- Strategy Pattern: Pluggable index types (Flat, IVF, HNSW)
- Factory Pattern: Index creation based on configuration
- Observer Pattern: Hooks for persistence events

SOLID Principles:
- Single Responsibility: VectorStore handles storage, EmbeddingService handles embedding
- Open/Closed: Extensible via custom index strategies
- Liskov Substitution: All index types implement same interface
- Interface Segregation: Separate interfaces for read/write operations
- Dependency Inversion: Depends on abstractions for embedding provider
"""

from __future__ import annotations

import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from sentinel_ml.config import VectorStoreConfig, get_config
from sentinel_ml.exceptions import StorageError
from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sentinel_ml.models import LogRecord

logger = get_logger(__name__)

# Type aliases
EmbeddingArray = NDArray[np.float32]
IndexType = Any  # FAISS index type


class IndexStrategy(str, Enum):
    """FAISS index strategies."""

    FLAT = "Flat"  # Exact search, O(n) but accurate
    IVF_FLAT = "IVFFlat"  # Inverted file, faster but approximate
    HNSW = "HNSW"  # Hierarchical Navigable Small World


@dataclass
class SearchResult:
    """Result of a vector similarity search."""

    id: str
    distance: float
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "distance": round(self.distance, 6),
            "similarity": round(self.similarity, 6),
            "metadata": self.metadata,
        }


@dataclass
class VectorStoreStats:
    """Statistics for vector store operations."""

    total_vectors: int = 0
    total_searches: int = 0
    total_adds: int = 0
    avg_search_time_ms: float = 0.0
    avg_add_time_ms: float = 0.0
    index_size_bytes: int = 0
    last_persist_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging."""
        return {
            "total_vectors": self.total_vectors,
            "total_searches": self.total_searches,
            "total_adds": self.total_adds,
            "avg_search_time_ms": round(self.avg_search_time_ms, 3),
            "avg_add_time_ms": round(self.avg_add_time_ms, 3),
            "index_size_bytes": self.index_size_bytes,
            "last_persist_time": (
                self.last_persist_time.isoformat() if self.last_persist_time else None
            ),
        }


class VectorIndex(ABC):
    """Abstract base class for vector indices."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of vectors in the index."""
        pass

    @abstractmethod
    def add(self, vectors: EmbeddingArray) -> list[int]:
        """
        Add vectors to the index.

        Args:
            vectors: Array of vectors with shape (n, dimension).

        Returns:
            List of assigned vector IDs.
        """
        pass

    @abstractmethod
    def search(
        self, query: EmbeddingArray, k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """
        Search for nearest neighbors.

        Args:
            query: Query vector(s) with shape (n_queries, dimension).
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (distances, indices) arrays.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the index to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load the index from disk."""
        pass


class FAISSIndex(VectorIndex):
    """
    FAISS-based vector index implementation.

    Supports multiple index types:
    - Flat: Exact L2 search, best for small datasets
    - IVFFlat: Approximate search with inverted file
    - HNSW: Graph-based approximate search
    """

    def __init__(
        self,
        dimension: int,
        strategy: IndexStrategy = IndexStrategy.FLAT,
        nlist: int = 100,
        nprobe: int = 10,
    ) -> None:
        """
        Initialize the FAISS index.

        Args:
            dimension: Vector dimensionality.
            strategy: Index strategy to use.
            nlist: Number of clusters for IVF index.
            nprobe: Number of clusters to probe during search.
        """
        self._dimension = dimension
        self._strategy = strategy
        self._nlist = nlist
        self._nprobe = nprobe
        self._index: IndexType = None
        self._is_trained = False

        self._create_index()

        logger.info(
            "faiss_index_initialized",
            dimension=dimension,
            strategy=strategy.value,
            nlist=nlist if strategy == IndexStrategy.IVF_FLAT else None,
        )

    def _create_index(self) -> None:
        """Create the FAISS index based on strategy."""
        try:
            import faiss
        except ImportError as e:
            msg = "faiss-cpu not installed. Install with: pip install sentinel-log-ai-ml[ml]"
            logger.error("faiss_import_error", error=msg)
            raise StorageError.write_failed("faiss_index", msg) from e

        if self._strategy == IndexStrategy.FLAT:
            # Simple flat index - exact search
            self._index = faiss.IndexFlatL2(self._dimension)
            self._is_trained = True

        elif self._strategy == IndexStrategy.IVF_FLAT:
            # IVF with flat quantizer
            quantizer = faiss.IndexFlatL2(self._dimension)
            self._index = faiss.IndexIVFFlat(
                quantizer, self._dimension, self._nlist, faiss.METRIC_L2
            )
            self._index.nprobe = self._nprobe
            self._is_trained = False

        elif self._strategy == IndexStrategy.HNSW:
            # HNSW graph-based index
            self._index = faiss.IndexHNSWFlat(self._dimension, 32)  # 32 neighbors
            self._is_trained = True

        else:
            msg = f"Unknown index strategy: {self._strategy}"
            raise StorageError.write_failed("faiss_index", msg)

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return self._dimension

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    @property
    def is_trained(self) -> bool:
        """Return whether the index is trained."""
        return self._is_trained

    def train(self, vectors: EmbeddingArray) -> None:
        """
        Train the index (required for IVF).

        Args:
            vectors: Training vectors with shape (n, dimension).
        """
        if self._is_trained:
            return

        if vectors.shape[0] < self._nlist:
            logger.warning(
                "insufficient_training_data",
                n_vectors=vectors.shape[0],
                nlist=self._nlist,
            )
            # Use what we have
            if vectors.shape[0] > 0:
                self._index.train(vectors)
                self._is_trained = True
        else:
            self._index.train(vectors)
            self._is_trained = True

        logger.info("faiss_index_trained", n_training_vectors=vectors.shape[0])

    def add(self, vectors: EmbeddingArray) -> list[int]:
        """
        Add vectors to the index.

        Args:
            vectors: Array of vectors with shape (n, dimension).

        Returns:
            List of assigned vector IDs (indices in the index).
        """
        if vectors.shape[0] == 0:
            return []

        if not self._is_trained:
            self.train(vectors)

        start_idx = self.size
        self._index.add(vectors.astype(np.float32))
        end_idx = self.size

        ids = list(range(start_idx, end_idx))

        logger.debug(
            "vectors_added_to_index",
            count=len(ids),
            total_size=self.size,
        )

        return ids

    def search(
        self, query: EmbeddingArray, k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """
        Search for nearest neighbors.

        Args:
            query: Query vector(s) with shape (n_queries, dimension) or (dimension,).
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (distances, indices) arrays.
        """
        if self.size == 0:
            n_queries = 1 if query.ndim == 1 else query.shape[0]
            return (
                np.array([[]], dtype=np.float32).reshape(n_queries, 0),
                np.array([[]], dtype=np.int64).reshape(n_queries, 0),
            )

        # Handle single query
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Limit k to actual size
        k = min(k, self.size)

        distances, indices = self._index.search(query.astype(np.float32), k)

        return distances.astype(np.float32), indices.astype(np.int64)

    def save(self, path: Path) -> None:
        """Save the index to disk."""
        import faiss

        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

        logger.info("faiss_index_saved", path=str(path), size=self.size)

    def load(self, path: Path) -> None:
        """Load the index from disk."""
        import faiss

        if not path.exists():
            raise StorageError.read_failed(str(path), "Index file not found")

        self._index = faiss.read_index(str(path))
        self._is_trained = True

        logger.info("faiss_index_loaded", path=str(path), size=self.size)


class MockVectorIndex(VectorIndex):
    """
    Mock vector index for testing.

    Uses numpy for exact nearest neighbor search without FAISS dependency.
    """

    def __init__(self, dimension: int) -> None:
        """
        Initialize the mock index.

        Args:
            dimension: Vector dimensionality.
        """
        self._dimension = dimension
        self._vectors: list[EmbeddingArray] = []

        logger.debug("mock_vector_index_initialized", dimension=dimension)

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return self._dimension

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._vectors)

    def add(self, vectors: EmbeddingArray) -> list[int]:
        """Add vectors to the index."""
        start_idx = self.size
        for i in range(vectors.shape[0]):
            self._vectors.append(vectors[i].copy())
        return list(range(start_idx, self.size))

    def search(
        self, query: EmbeddingArray, k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Search for nearest neighbors using brute force."""
        if self.size == 0:
            n_queries = 1 if query.ndim == 1 else query.shape[0]
            return (
                np.array([[]], dtype=np.float32).reshape(n_queries, 0),
                np.array([[]], dtype=np.int64).reshape(n_queries, 0),
            )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        k = min(k, self.size)
        all_vectors = np.array(self._vectors, dtype=np.float32)

        # Compute L2 distances
        n_queries = query.shape[0]
        distances = np.zeros((n_queries, k), dtype=np.float32)
        indices = np.zeros((n_queries, k), dtype=np.int64)

        for i in range(n_queries):
            # L2 distance
            dists = np.sum((all_vectors - query[i]) ** 2, axis=1)
            nearest_idx = np.argsort(dists)[:k]
            distances[i] = dists[nearest_idx]
            indices[i] = nearest_idx

        return distances, indices

    def save(self, path: Path) -> None:
        """Save the index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dimension": self._dimension,
            "vectors": [v.tolist() for v in self._vectors],
        }
        with path.open("w") as f:
            json.dump(data, f)

    def load(self, path: Path) -> None:
        """Load the index from disk."""
        if not path.exists():
            raise StorageError.read_failed(str(path), "Index file not found")

        with path.open() as f:
            data = json.load(f)

        self._dimension = data["dimension"]
        self._vectors = [np.array(v, dtype=np.float32) for v in data["vectors"]]


@dataclass
class VectorMetadata:
    """Metadata for a stored vector."""

    id: str  # External ID (log record ID)
    index_id: int  # Internal FAISS index ID
    source: str | None = None
    level: str | None = None
    timestamp: datetime | None = None
    message_preview: str | None = None
    cluster_id: str | None = None
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "index_id": self.index_id,
            "source": self.source,
            "level": self.level,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "message_preview": self.message_preview,
            "cluster_id": self.cluster_id,
            "added_at": self.added_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorMetadata:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            index_id=data["index_id"],
            source=data.get("source"),
            level=data.get("level"),
            timestamp=(
                datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
            ),
            message_preview=data.get("message_preview"),
            cluster_id=data.get("cluster_id"),
            added_at=(
                datetime.fromisoformat(data["added_at"])
                if data.get("added_at")
                else datetime.now(timezone.utc)
            ),
        )


@dataclass
class VectorStore:
    """
    High-level vector store with metadata management.

    Combines FAISS index with metadata storage for log records,
    providing a complete solution for similarity search.

    Usage:
        store = VectorStore.from_config()
        store.add(embeddings, records)
        results = store.search(query_embedding, k=10)
    """

    index: VectorIndex
    persist_dir: Path | None = None
    stats: VectorStoreStats = field(default_factory=VectorStoreStats)
    _metadata: dict[int, VectorMetadata] = field(default_factory=dict)
    _id_to_index: dict[str, int] = field(default_factory=dict)
    _search_times: list[float] = field(default_factory=list)
    _add_times: list[float] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: VectorStoreConfig | None = None) -> VectorStore:
        """
        Create a vector store from configuration.

        Args:
            config: Vector store configuration. Uses global config if None.

        Returns:
            Configured VectorStore instance.
        """
        if config is None:
            config = get_config().vector_store

        # Get embedding dimension from config
        # Default dimension for all-MiniLM-L6-v2
        dimension = 384

        # Select index strategy
        strategy = IndexStrategy(config.index_type)

        index = FAISSIndex(
            dimension=dimension,
            strategy=strategy,
            nlist=config.nlist,
            nprobe=config.nprobe,
        )

        persist_dir = Path(config.persist_dir) if config.persist_dir else None

        store = cls(index=index, persist_dir=persist_dir)

        # Try to load existing index
        if persist_dir and (persist_dir / "index.faiss").exists():
            try:
                store.load()
                logger.info(
                    "vector_store_loaded",
                    persist_dir=str(persist_dir),
                    size=store.size,
                )
            except Exception as e:
                logger.warning("vector_store_load_failed", error=str(e))

        logger.info(
            "vector_store_created",
            strategy=strategy.value,
            persist_dir=str(persist_dir) if persist_dir else None,
        )

        return store

    @classmethod
    def create_mock(cls, dimension: int = 384) -> VectorStore:
        """
        Create a mock vector store for testing.

        Args:
            dimension: Vector dimensionality.

        Returns:
            VectorStore with mock index.
        """
        index = MockVectorIndex(dimension=dimension)
        return cls(index=index)

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return self.index.dimension

    @property
    def size(self) -> int:
        """Return the number of vectors in the store."""
        return self.index.size

    def add(
        self,
        embeddings: EmbeddingArray,
        records: Sequence[LogRecord] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Add embeddings with optional metadata from log records.

        Args:
            embeddings: Array of embeddings with shape (n, dimension).
            records: Optional list of log records for metadata.
            ids: Optional list of external IDs. Generated if not provided.

        Returns:
            List of external IDs for the added vectors.
        """
        if embeddings.shape[0] == 0:
            return []

        start_time = time.time()

        # Generate IDs if not provided
        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in range(embeddings.shape[0])]

        # Validate inputs
        if records is not None and len(records) != embeddings.shape[0]:
            raise StorageError.write_failed(
                "vector_store",
                f"Number of records ({len(records)}) must match embeddings ({embeddings.shape[0]})",
            )

        if len(ids) != embeddings.shape[0]:
            raise StorageError.write_failed(
                "vector_store",
                f"Number of IDs ({len(ids)}) must match embeddings ({embeddings.shape[0]})",
            )

        # Add to index
        index_ids = self.index.add(embeddings)

        # Store metadata
        for i, (external_id, index_id) in enumerate(zip(ids, index_ids, strict=True)):
            record = records[i] if records else None
            metadata = VectorMetadata(
                id=external_id,
                index_id=index_id,
                source=record.source if record else None,
                level=record.level if record else None,
                timestamp=record.timestamp if record else None,
                message_preview=(record.message[:200] if record and record.message else None),
            )
            self._metadata[index_id] = metadata
            self._id_to_index[external_id] = index_id

        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self._add_times.append(elapsed_ms)
        self.stats.total_adds += 1
        self.stats.total_vectors = self.size
        self.stats.avg_add_time_ms = sum(self._add_times) / len(self._add_times)

        logger.debug(
            "vectors_added",
            count=len(ids),
            total_size=self.size,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return ids

    def search(
        self,
        query: EmbeddingArray,
        k: int = 10,
        min_similarity: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query: Query embedding with shape (dimension,) or (1, dimension).
            k: Number of results to return.
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            List of SearchResult objects.
        """
        start_time = time.time()

        if self.size == 0:
            return []

        # Ensure query is 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)

        distances, indices = self.index.search(query, k)

        # Convert distances to similarities (assuming L2 normalized vectors)
        # For L2 normalized vectors: similarity = 1 - distance/2
        # For cosine: similarity = 1 - distance^2/2
        results: list[SearchResult] = []

        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0:  # FAISS uses -1 for no result
                continue

            # Convert L2 distance to cosine similarity for normalized vectors
            # d^2 = 2(1 - cos_sim) => cos_sim = 1 - d^2/2
            similarity = max(0.0, 1.0 - float(dist) / 2.0)

            if similarity < min_similarity:
                continue

            metadata = self._metadata.get(int(idx), VectorMetadata(id=str(idx), index_id=int(idx)))

            results.append(
                SearchResult(
                    id=metadata.id,
                    distance=float(dist),
                    similarity=similarity,
                    metadata=metadata.to_dict(),
                )
            )

        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self._search_times.append(elapsed_ms)
        self.stats.total_searches += 1
        self.stats.avg_search_time_ms = sum(self._search_times) / len(self._search_times)

        logger.debug(
            "vector_search_completed",
            k=k,
            results=len(results),
            elapsed_ms=round(elapsed_ms, 2),
        )

        return results

    def search_batch(
        self,
        queries: EmbeddingArray,
        k: int = 10,
        min_similarity: float = 0.0,
    ) -> list[list[SearchResult]]:
        """
        Search for similar vectors for multiple queries.

        Args:
            queries: Query embeddings with shape (n_queries, dimension).
            k: Number of results per query.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of result lists, one per query.
        """
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        results: list[list[SearchResult]] = []
        for query in queries:
            results.append(self.search(query, k=k, min_similarity=min_similarity))

        return results

    def get_by_id(self, external_id: str) -> VectorMetadata | None:
        """
        Get metadata for a vector by external ID.

        Args:
            external_id: The external ID.

        Returns:
            VectorMetadata or None if not found.
        """
        index_id = self._id_to_index.get(external_id)
        if index_id is None:
            return None
        return self._metadata.get(index_id)

    def update_cluster_id(self, external_id: str, cluster_id: str) -> bool:
        """
        Update the cluster ID for a vector.

        Args:
            external_id: The external ID.
            cluster_id: The new cluster ID.

        Returns:
            True if updated, False if not found.
        """
        index_id = self._id_to_index.get(external_id)
        if index_id is None:
            return False

        metadata = self._metadata.get(index_id)
        if metadata is None:
            return False

        metadata.cluster_id = cluster_id
        return True

    def save(self) -> None:
        """Save the vector store to disk."""
        if self.persist_dir is None:
            logger.warning("vector_store_save_skipped", reason="no persist_dir configured")
            return

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Save index
        index_path = self.persist_dir / "index.faiss"
        self.index.save(index_path)

        # Save metadata
        metadata_path = self.persist_dir / "metadata.pkl"
        with metadata_path.open("wb") as f:
            pickle.dump(
                {
                    "metadata": {k: v.to_dict() for k, v in self._metadata.items()},
                    "id_to_index": self._id_to_index,
                    "stats": self.stats.to_dict(),
                },
                f,
            )

        self.stats.last_persist_time = datetime.now(timezone.utc)

        logger.info(
            "vector_store_saved",
            persist_dir=str(self.persist_dir),
            size=self.size,
        )

    def load(self) -> None:
        """Load the vector store from disk."""
        if self.persist_dir is None:
            raise StorageError.read_failed("vector_store", "no persist_dir configured")

        # Load index
        index_path = self.persist_dir / "index.faiss"
        self.index.load(index_path)

        # Load metadata
        metadata_path = self.persist_dir / "metadata.pkl"
        if metadata_path.exists():
            with metadata_path.open("rb") as f:
                data = pickle.load(f)

            self._metadata = {
                int(k): VectorMetadata.from_dict(v) for k, v in data["metadata"].items()
            }
            self._id_to_index = data["id_to_index"]

        self.stats.total_vectors = self.size

        logger.info(
            "vector_store_loaded",
            persist_dir=str(self.persist_dir),
            size=self.size,
        )

    def clear(self) -> None:
        """Clear all vectors and metadata."""
        # Recreate index
        if isinstance(self.index, FAISSIndex):
            self.index._create_index()
        elif isinstance(self.index, MockVectorIndex):
            self.index._vectors.clear()

        self._metadata.clear()
        self._id_to_index.clear()
        self.stats = VectorStoreStats()
        self._search_times.clear()
        self._add_times.clear()

        logger.info("vector_store_cleared")

    def reset_stats(self) -> None:
        """Reset statistics without clearing vectors."""
        self.stats = VectorStoreStats()
        self.stats.total_vectors = self.size
        self._search_times.clear()
        self._add_times.clear()


# Module-level singleton
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """
    Get the global vector store instance.

    Returns:
        The singleton VectorStore instance.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore.from_config()
    return _vector_store


def set_vector_store(store: VectorStore) -> None:
    """
    Set the global vector store instance.

    Args:
        store: The VectorStore to use globally.
    """
    global _vector_store
    _vector_store = store
