"""Tests for configuration module."""

import os
import pytest
import tempfile
from pathlib import Path

from sentinel_ml.config import (
    Config,
    EmbeddingConfig,
    VectorStoreConfig,
    ClusteringConfig,
    NoveltyConfig,
    LLMConfig,
    ServerConfig,
    LoggingConfig,
    get_config,
    set_config,
)


class TestEmbeddingConfig:
    """Test EmbeddingConfig model."""

    def test_default_values(self):
        """Test default embedding config values."""
        config = EmbeddingConfig()

        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.batch_size == 32
        assert config.device == "cpu"
        assert config.cache_enabled is True

    def test_custom_values(self):
        """Test custom embedding config values."""
        config = EmbeddingConfig(
            model_name="custom-model",
            batch_size=64,
            device="cuda",
            cache_enabled=False,
        )

        assert config.model_name == "custom-model"
        assert config.batch_size == 64
        assert config.device == "cuda"
        assert config.cache_enabled is False


class TestVectorStoreConfig:
    """Test VectorStoreConfig model."""

    def test_default_values(self):
        """Test default vector store config values."""
        config = VectorStoreConfig()

        assert config.index_type == "Flat"
        assert config.nlist == 100
        assert config.nprobe == 10

    def test_custom_values(self):
        """Test custom vector store config values."""
        config = VectorStoreConfig(
            index_type="IVF",
            nlist=200,
            nprobe=20,
        )

        assert config.index_type == "IVF"
        assert config.nlist == 200
        assert config.nprobe == 20


class TestClusteringConfig:
    """Test ClusteringConfig model."""

    def test_default_values(self):
        """Test default clustering config values."""
        config = ClusteringConfig()

        assert config.min_cluster_size == 5
        assert config.min_samples == 3
        assert config.metric == "euclidean"

    def test_custom_values(self):
        """Test custom clustering config values."""
        config = ClusteringConfig(
            min_cluster_size=10,
            min_samples=5,
            cluster_selection_epsilon=0.1,
            metric="cosine",
        )

        assert config.min_cluster_size == 10
        assert config.min_samples == 5
        assert config.cluster_selection_epsilon == 0.1
        assert config.metric == "cosine"


class TestNoveltyConfig:
    """Test NoveltyConfig model."""

    def test_default_values(self):
        """Test default novelty config values."""
        config = NoveltyConfig()

        assert config.threshold == 0.7
        assert config.k_neighbors == 5
        assert config.use_density is True

    def test_custom_values(self):
        """Test custom novelty config values."""
        config = NoveltyConfig(
            threshold=0.5,
            k_neighbors=10,
            use_density=False,
        )

        assert config.threshold == 0.5
        assert config.k_neighbors == 10
        assert config.use_density is False


class TestLLMConfig:
    """Test LLMConfig model."""

    def test_default_values(self):
        """Test default LLM config values."""
        config = LLMConfig()

        assert config.provider == "ollama"
        assert config.model == "llama3.2"
        assert config.base_url == "http://localhost:11434"
        assert config.timeout == 120
        assert config.temperature == 0.1

    def test_custom_values(self):
        """Test custom LLM config values."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            base_url="https://api.openai.com",
            temperature=0.5,
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5


class TestServerConfig:
    """Test ServerConfig model."""

    def test_default_values(self):
        """Test default server config values."""
        config = ServerConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 50051
        assert config.max_workers == 10

    def test_custom_values(self):
        """Test custom server config values."""
        config = ServerConfig(
            host="127.0.0.1",
            port=50052,
            max_workers=20,
        )

        assert config.host == "127.0.0.1"
        assert config.port == 50052
        assert config.max_workers == 20


class TestLoggingConfig:
    """Test LoggingConfig model."""

    def test_default_values(self):
        """Test default logging config values."""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.format == "json"
        assert config.file is None

    def test_custom_values(self):
        """Test custom logging config values."""
        config = LoggingConfig(
            level="DEBUG",
            format="plain",
            file="/var/log/sentinel.log",
        )

        assert config.level == "DEBUG"
        assert config.format == "plain"
        assert config.file == "/var/log/sentinel.log"


class TestMainConfig:
    """Test main Config model."""

    def test_default_config(self):
        """Test default config values."""
        config = Config()

        # Check nested configs
        assert config.embedding.model_name == "all-MiniLM-L6-v2"
        assert config.server.port == 50051
        assert config.novelty.threshold == 0.7
        assert config.llm.provider == "ollama"

    def test_custom_config(self):
        """Test custom config values."""
        config = Config(
            embedding=EmbeddingConfig(model_name="custom-model"),
            server=ServerConfig(port=60000),
        )

        assert config.embedding.model_name == "custom-model"
        assert config.server.port == 60000
        # Others should have defaults
        assert config.novelty.threshold == 0.7

    def test_config_from_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
embedding:
  model_name: "yaml-model"
  batch_size: 64

server:
  port: 55555

novelty:
  threshold: 0.9
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = Config.from_yaml(f.name)

                assert config.embedding.model_name == "yaml-model"
                assert config.embedding.batch_size == 64
                assert config.server.port == 55555
                assert config.novelty.threshold == 0.9
                # Defaults for unspecified
                assert config.llm.provider == "ollama"
            finally:
                os.unlink(f.name)

    def test_config_from_yaml_missing_file(self):
        """Test handling missing config file."""
        config = Config.from_yaml("/nonexistent/path/config.yaml")
        # Should return defaults
        assert config.embedding.model_name == "all-MiniLM-L6-v2"

    def test_config_to_yaml(self):
        """Test saving config to YAML file."""
        config = Config(
            embedding=EmbeddingConfig(model_name="saved-model"),
            server=ServerConfig(port=44444),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            # Load it back
            loaded = Config.from_yaml(yaml_path)

            assert loaded.embedding.model_name == "saved-model"
            assert loaded.server.port == 44444

    def test_config_model_dump(self):
        """Test converting config to dict."""
        config = Config()
        d = config.model_dump()

        assert isinstance(d, dict)
        assert "embedding" in d
        assert "server" in d
        assert d["embedding"]["model_name"] == "all-MiniLM-L6-v2"

    def test_config_model_dump_json(self):
        """Test converting config to JSON."""
        config = Config()
        json_str = config.model_dump_json()

        assert isinstance(json_str, str)
        assert "all-MiniLM-L6-v2" in json_str


class TestConfigLoad:
    """Test Config.load() method."""

    def test_load_default(self):
        """Test loading with defaults."""
        config = Config.load()
        assert isinstance(config, Config)
        assert config.embedding.model_name == "all-MiniLM-L6-v2"

    def test_load_from_path(self):
        """Test loading from explicit path."""
        yaml_content = """
server:
  port: 33333
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = Config.load(f.name)
                assert config.server.port == 33333
            finally:
                os.unlink(f.name)

    def test_load_from_env_var(self, monkeypatch):
        """Test loading config path from env var."""
        yaml_content = """
server:
  port: 22222
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            try:
                monkeypatch.setenv("SENTINEL_ML_CONFIG", f.name)
                config = Config.load()
                assert config.server.port == 22222
            finally:
                os.unlink(f.name)


class TestGlobalConfig:
    """Test global config functions."""

    def test_get_config(self):
        """Test get_config returns a Config instance."""
        # Reset global
        set_config(None)

        config = get_config()
        assert isinstance(config, Config)

    def test_set_config(self):
        """Test set_config sets the global instance."""
        custom = Config(server=ServerConfig(port=11111))
        set_config(custom)

        retrieved = get_config()
        assert retrieved.server.port == 11111

        # Clean up
        set_config(None)


class TestConfigValidation:
    """Test config validation."""

    def test_invalid_yaml_handling(self):
        """Test handling invalid YAML content."""
        yaml_content = """
this is not: valid: yaml:
  - [broken
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            try:
                # Should handle gracefully - either raise or return defaults
                try:
                    config = Config.from_yaml(f.name)
                    # If it doesn't raise, should have defaults
                    assert isinstance(config, Config)
                except Exception:
                    # It's acceptable to raise on invalid YAML
                    pass
            finally:
                os.unlink(f.name)

    def test_empty_yaml_handling(self):
        """Test handling empty YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()

            try:
                config = Config.from_yaml(f.name)
                assert isinstance(config, Config)
                # Should have defaults
                assert config.embedding.model_name == "all-MiniLM-L6-v2"
            finally:
                os.unlink(f.name)

    def test_partial_yaml_handling(self):
        """Test handling partial YAML config."""
        yaml_content = """
embedding:
  model_name: "partial-model"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = Config.from_yaml(f.name)

                # Specified value
                assert config.embedding.model_name == "partial-model"
                # Defaults for unspecified
                assert config.server.port == 50051
            finally:
                os.unlink(f.name)
