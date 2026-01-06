"""
Configuration management for the ML engine.

Supports YAML/TOML config files with environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model."""

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model name",
    )
    batch_size: int = Field(default=32, description="Batch size for embedding")
    device: str = Field(default="cpu", description="Device to use (cpu, cuda, mps)")
    cache_enabled: bool = Field(default=True, description="Enable embedding cache")
    cache_dir: str = Field(default=".cache/embeddings", description="Cache directory")


class VectorStoreConfig(BaseModel):
    """Configuration for the FAISS vector store."""

    index_type: str = Field(default="Flat", description="FAISS index type (Flat, IVF, HNSW)")
    persist_dir: str = Field(default=".data/faiss", description="Directory to persist index")
    nlist: int = Field(default=100, description="Number of clusters for IVF index")
    nprobe: int = Field(default=10, description="Number of clusters to search")


class ClusteringConfig(BaseModel):
    """Configuration for HDBSCAN clustering."""

    min_cluster_size: int = Field(default=5, description="Minimum cluster size")
    min_samples: int = Field(default=3, description="Minimum samples for core points")
    cluster_selection_epsilon: float = Field(default=0.0, description="Cluster selection epsilon")
    metric: str = Field(default="euclidean", description="Distance metric")


class NoveltyConfig(BaseModel):
    """Configuration for novelty detection."""

    threshold: float = Field(default=0.7, description="Novelty score threshold")
    k_neighbors: int = Field(default=5, description="Number of neighbors for kNN density")
    use_density: bool = Field(default=True, description="Use density-based novelty scoring")


class LLMConfig(BaseModel):
    """Configuration for LLM integration."""

    provider: str = Field(default="ollama", description="LLM provider (ollama, openai)")
    model: str = Field(default="llama3.2", description="Model name")
    base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    temperature: float = Field(default=0.1, description="Sampling temperature")


class ServerConfig(BaseModel):
    """Configuration for the gRPC server."""

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=50051, description="Server port")
    max_workers: int = Field(default=10, description="Maximum worker threads")
    max_message_size: int = Field(default=100 * 1024 * 1024, description="Max message size (100MB)")


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json, plain)")
    file: str | None = Field(default=None, description="Log file path (None for stdout)")


class Config(BaseSettings):
    """Main configuration for the ML engine."""

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_ML_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with path.open() as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    @classmethod
    def load(cls, config_path: str | None = None) -> Config:
        """
        Load configuration with precedence:
        1. Environment variables (highest)
        2. Config file
        3. Defaults (lowest)
        """
        # Try to find config file
        if config_path is None:
            config_path = os.getenv("SENTINEL_ML_CONFIG")

        if config_path is None:
            # Check common locations
            for candidate in [
                "sentinel-ml.yaml",
                "sentinel-ml.yml",
                "config/sentinel-ml.yaml",
                ".sentinel-ml.yaml",
            ]:
                if Path(candidate).exists():
                    config_path = candidate
                    break

        if config_path and Path(config_path).exists():
            return cls.from_yaml(config_path)

        return cls()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
