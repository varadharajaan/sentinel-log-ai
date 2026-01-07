"""
Embedding generation for log messages.

This module provides embedding functionality using sentence-transformers
for converting log messages into dense vector representations suitable
for similarity search and clustering.

Design Patterns:
- Strategy Pattern: Pluggable embedding providers (local, remote)
- Factory Pattern: Embedding service creation with configuration
- Facade Pattern: Simple interface to complex model operations
- Singleton Pattern: Model caching for efficiency

SOLID Principles:
- Single Responsibility: Each class handles one concern
- Open/Closed: Extensible via EmbeddingProvider interface
- Dependency Inversion: Depends on abstractions not implementations
- Interface Segregation: Minimal interfaces for specific capabilities
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from sentinel_ml.config import EmbeddingConfig, get_config
from sentinel_ml.exceptions import ProcessingError
from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sentinel_ml.models import LogRecord

logger = get_logger(__name__)

# Type alias for embeddings
EmbeddingArray: TypeAlias = NDArray[np.float32]


@dataclass
class EmbeddingStats:
    """Statistics for embedding operations."""

    total_embedded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_time_seconds: float = 0.0
    batch_count: int = 0
    model_load_time_seconds: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def avg_time_per_batch(self) -> float:
        """Calculate average time per batch."""
        if self.batch_count == 0:
            return 0.0
        return self.total_time_seconds / self.batch_count

    @property
    def throughput(self) -> float:
        """Calculate embeddings per second."""
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_embedded / self.total_time_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging."""
        return {
            "total_embedded": self.total_embedded,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "total_time_seconds": round(self.total_time_seconds, 3),
            "batch_count": self.batch_count,
            "avg_time_per_batch": round(self.avg_time_per_batch, 3),
            "throughput": round(self.throughput, 2),
            "model_load_time_seconds": round(self.model_load_time_seconds, 3),
        }


class EmbeddingCache:
    """
    LRU cache for embeddings keyed by message hash.

    Uses SHA-256 hash of normalized messages to cache embeddings,
    avoiding redundant computation for duplicate log patterns.
    """

    def __init__(self, max_size: int = 10000) -> None:
        """
        Initialize the embedding cache.

        Args:
            max_size: Maximum number of cached embeddings.
        """
        self._cache: dict[str, EmbeddingArray] = {}
        self._access_order: list[str] = []
        self._max_size = max_size
        logger.debug("embedding_cache_initialized", max_size=max_size)

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute SHA-256 hash of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get(self, text: str) -> EmbeddingArray | None:
        """
        Retrieve embedding from cache.

        Args:
            text: The normalized message text.

        Returns:
            The cached embedding or None if not found.
        """
        key = self._compute_hash(text)
        if key in self._cache:
            # Update access order for LRU
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, text: str, embedding: EmbeddingArray) -> None:
        """
        Store embedding in cache.

        Args:
            text: The normalized message text.
            embedding: The embedding vector.
        """
        key = self._compute_hash(text)

        # Evict if at capacity
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
            logger.debug("embedding_cache_eviction", evicted_key=oldest)

        self._cache[key] = embedding.copy()
        if key not in self._access_order:
            self._access_order.append(key)

    def get_batch(self, texts: list[str]) -> tuple[list[EmbeddingArray | None], list[int]]:
        """
        Get cached embeddings for a batch of texts.

        Args:
            texts: List of normalized message texts.

        Returns:
            Tuple of (embeddings list with None for misses, list of miss indices).
        """
        embeddings: list[EmbeddingArray | None] = []
        miss_indices: list[int] = []

        for i, text in enumerate(texts):
            cached = self.get(text)
            embeddings.append(cached)
            if cached is None:
                miss_indices.append(i)

        return embeddings, miss_indices

    def put_batch(self, texts: list[str], embeddings: EmbeddingArray) -> None:
        """
        Store batch of embeddings in cache.

        Args:
            texts: List of normalized message texts.
            embeddings: Array of embeddings with shape (n, dim).
        """
        for i, text in enumerate(texts):
            self.put(text, embeddings[i])

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        logger.debug("embedding_cache_cleared")

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbeddingArray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).

        Raises:
            ProcessingError: If embedding generation fails.
        """
        pass

    @abstractmethod
    def embed_single(self, text: str) -> EmbeddingArray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector with shape (embedding_dim,).

        Raises:
            ProcessingError: If embedding generation fails.
        """
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Embedding provider using sentence-transformers.

    Lazy loads the model on first use to avoid import-time overhead.
    Supports CPU, CUDA, and MPS devices.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        """
        Initialize the sentence transformer provider.

        Args:
            model_name: HuggingFace model name or path.
            device: Device to use (cpu, cuda, mps).
        """
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._embedding_dim: int | None = None
        self._load_time: float = 0.0

        logger.info(
            "sentence_transformer_provider_initialized",
            model_name=model_name,
            device=device,
        )

    def _ensure_model_loaded(self) -> None:
        """Load the model if not already loaded."""
        if self._model is not None:
            return

        logger.info("loading_embedding_model", model_name=self._model_name)
        start_time = time.time()

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            self._load_time = time.time() - start_time

            logger.info(
                "embedding_model_loaded",
                model_name=self._model_name,
                embedding_dim=self._embedding_dim,
                device=self._device,
                load_time_seconds=round(self._load_time, 2),
            )
        except ImportError as e:
            msg = "sentence-transformers not installed. Install with: pip install sentinel-log-ai-ml[ml]"
            logger.error("embedding_model_import_error", error=msg)
            raise ProcessingError.model_load_failed(self._model_name, msg) from e
        except Exception as e:
            logger.error(
                "embedding_model_load_failed",
                model_name=self._model_name,
                error=str(e),
            )
            raise ProcessingError.model_load_failed(self._model_name, str(e)) from e

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        self._ensure_model_loaded()
        assert self._embedding_dim is not None
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def load_time(self) -> float:
        """Return the model load time in seconds."""
        return self._load_time

    def embed(self, texts: list[str]) -> EmbeddingArray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).

        Raises:
            ProcessingError: If embedding generation fails.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        self._ensure_model_loaded()

        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(
                "embedding_generation_failed",
                batch_size=len(texts),
                error=str(e),
            )
            raise ProcessingError.embedding_failed(len(texts), str(e)) from e

    def embed_single(self, text: str) -> EmbeddingArray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector with shape (embedding_dim,).

        Raises:
            ProcessingError: If embedding generation fails.
        """
        embeddings = self.embed([text])
        return np.asarray(embeddings[0], dtype=np.float32)


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock embedding provider for testing.

    Generates deterministic pseudo-random embeddings based on text hash.
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        """
        Initialize the mock provider.

        Args:
            embedding_dim: Dimensionality of mock embeddings.
        """
        self._embedding_dim = embedding_dim
        self._model_name = "mock-model"
        logger.debug("mock_embedding_provider_initialized", embedding_dim=embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def embed(self, texts: list[str]) -> EmbeddingArray:
        """Generate deterministic mock embeddings."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        embeddings = np.zeros((len(texts), self._embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Use hash to generate deterministic embedding
            hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
            # Convert first bytes to seed
            seed = int.from_bytes(hash_bytes[:4], "big")
            rng = np.random.default_rng(seed)
            embedding = rng.standard_normal(self._embedding_dim).astype(np.float32)
            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings[i] = embedding

        return embeddings

    def embed_single(self, text: str) -> EmbeddingArray:
        """Generate deterministic mock embedding for single text."""
        return np.asarray(self.embed([text])[0], dtype=np.float32)


@dataclass
class EmbeddingService:
    """
    High-level service for embedding generation with caching.

    Provides a unified interface for embedding log messages with:
    - Automatic caching for duplicate messages
    - Batch processing for efficiency
    - Statistics tracking for monitoring

    Usage:
        service = EmbeddingService.from_config()
        embeddings = service.embed_records(log_records)
    """

    provider: EmbeddingProvider
    cache: EmbeddingCache | None = None
    stats: EmbeddingStats = field(default_factory=EmbeddingStats)

    @classmethod
    def from_config(cls, config: EmbeddingConfig | None = None) -> EmbeddingService:
        """
        Create an embedding service from configuration.

        Args:
            config: Embedding configuration. Uses global config if None.

        Returns:
            Configured EmbeddingService instance.
        """
        if config is None:
            config = get_config().embedding

        provider = SentenceTransformerProvider(
            model_name=config.model_name,
            device=config.device,
        )

        cache = EmbeddingCache() if config.cache_enabled else None

        logger.info(
            "embedding_service_created",
            model_name=config.model_name,
            device=config.device,
            cache_enabled=config.cache_enabled,
        )

        return cls(provider=provider, cache=cache)

    @classmethod
    def create_mock(cls, embedding_dim: int = 384) -> EmbeddingService:
        """
        Create a mock embedding service for testing.

        Args:
            embedding_dim: Dimensionality of mock embeddings.

        Returns:
            EmbeddingService with mock provider.
        """
        provider = MockEmbeddingProvider(embedding_dim=embedding_dim)
        cache = EmbeddingCache()
        return cls(provider=provider, cache=cache)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self.provider.embedding_dim

    def embed_texts(self, texts: list[str], use_cache: bool = True) -> EmbeddingArray:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed.
            use_cache: Whether to use the cache.

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        start_time = time.time()
        self.stats.batch_count += 1

        if use_cache and self.cache is not None:
            # Check cache for existing embeddings
            cached_embeddings, miss_indices = self.cache.get_batch(texts)

            self.stats.cache_hits += len(texts) - len(miss_indices)
            self.stats.cache_misses += len(miss_indices)

            if not miss_indices:
                # All hits - return cached
                embeddings = np.array(
                    [e for e in cached_embeddings if e is not None], dtype=np.float32
                )
                self.stats.total_embedded += len(texts)
                self.stats.total_time_seconds += time.time() - start_time
                return embeddings

            # Compute missing embeddings
            miss_texts = [texts[i] for i in miss_indices]
            new_embeddings = self.provider.embed(miss_texts)

            # Cache the new embeddings
            self.cache.put_batch(miss_texts, new_embeddings)

            # Merge cached and new embeddings
            all_embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
            new_idx = 0
            for i in range(len(texts)):
                if cached_embeddings[i] is not None:
                    all_embeddings[i] = cached_embeddings[i]
                else:
                    all_embeddings[i] = new_embeddings[new_idx]
                    new_idx += 1

            embeddings = all_embeddings
        else:
            # No caching - compute all
            embeddings = self.provider.embed(texts)
            self.stats.cache_misses += len(texts)

        self.stats.total_embedded += len(texts)
        self.stats.total_time_seconds += time.time() - start_time

        logger.debug(
            "texts_embedded",
            count=len(texts),
            cache_hits=self.stats.cache_hits,
            duration_ms=round((time.time() - start_time) * 1000, 2),
        )

        return embeddings

    def embed_records(
        self,
        records: Sequence[LogRecord],
        use_cache: bool = True,
    ) -> EmbeddingArray:
        """
        Embed a batch of log records.

        Uses the normalized message if available, otherwise the raw message.

        Args:
            records: List of log records to embed.
            use_cache: Whether to use the cache.

        Returns:
            Array of embeddings with shape (n_records, embedding_dim).
        """
        if not records:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        # Extract text from records, preferring normalized
        texts = []
        for record in records:
            if record.normalized:
                texts.append(record.normalized)
            else:
                texts.append(record.message)

        return self.embed_texts(texts, use_cache=use_cache)

    def embed_single(self, text: str, use_cache: bool = True) -> EmbeddingArray:
        """
        Embed a single text.

        Args:
            text: Text to embed.
            use_cache: Whether to use the cache.

        Returns:
            Embedding vector with shape (embedding_dim,).
        """
        embeddings = self.embed_texts([text], use_cache=use_cache)
        return np.asarray(embeddings[0], dtype=np.float32)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = EmbeddingStats()


# Module-level singleton for convenience
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """
    Get the global embedding service instance.

    Returns:
        The singleton EmbeddingService instance.
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService.from_config()
    return _embedding_service


def set_embedding_service(service: EmbeddingService) -> None:
    """
    Set the global embedding service instance.

    Args:
        service: The EmbeddingService to use globally.
    """
    global _embedding_service
    _embedding_service = service
