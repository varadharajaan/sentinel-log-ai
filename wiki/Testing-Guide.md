# Testing Guide

Comprehensive testing strategy for Sentinel Log AI, covering unit tests, integration tests, and testing patterns.

## Test Overview

| Language | Framework | Coverage Target |
|----------|-----------|-----------------|
| Python | pytest | 90%+ |
| Go | testing | 85%+ |

```
tests/
├── python/                 # Python test suite
│   ├── conftest.py         # Shared fixtures
│   ├── test_config.py      # Configuration tests
│   ├── test_exceptions.py  # Exception hierarchy
│   ├── test_logging.py     # Logging module
│   ├── test_models.py      # Pydantic models
│   ├── test_normalization.py # Log masking
│   ├── test_parser.py      # Log parsing
│   ├── test_preprocessing.py # Pipeline stages
│   ├── test_embedding.py   # Embedding service
│   ├── test_vectorstore.py # FAISS store
│   ├── test_clustering.py  # HDBSCAN clustering
│   ├── test_novelty.py     # Novelty detection
│   ├── test_llm.py         # LLM integration
│   ├── test_cli.py         # CLI module (112 tests)
│   └── test_server.py      # gRPC server
├── go/                     # Go test suite
│   └── ...
└── integration/            # Integration tests
    └── ...
```

## Running Tests

### Python Tests

```bash
# Run all tests
make test-python
# or
pytest tests/python/ -v

# Run with coverage
pytest tests/python/ --cov=sentinel_ml --cov-report=html

# Run specific module
pytest tests/python/test_cli.py -v

# Run specific test class
pytest tests/python/test_cli.py::TestConsole -v

# Run specific test
pytest tests/python/test_cli.py::TestConsole::test_info_output -v

# Run with markers
pytest tests/python/ -m "not slow"
pytest tests/python/ -m "integration"
```

### Go Tests

```bash
# Run all tests
make test-go
# or
go test ./...

# With coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Run specific package
go test ./internal/parser/...

# Verbose output
go test -v ./internal/parser/...
```

### All Tests

```bash
make test          # Run all Python and Go tests
make test-coverage # Run with coverage reports
```

---

## Test Patterns

### AAA Pattern (Arrange-Act-Assert)

```python
class TestEmbeddingService:
    def test_embed_single_text(self):
        # Arrange
        service = EmbeddingService(MockEmbeddingProvider())
        text = "User login successful"
        
        # Act
        embedding = service.embed([text])
        
        # Assert
        assert embedding.shape == (1, 384)
        assert isinstance(embedding, np.ndarray)
```

### Given-When-Then (BDD Style)

```python
class TestNoveltyDetection:
    def test_novel_pattern_detected(self):
        """
        Given: A trained detector with known patterns
        When: A novel log message is analyzed
        Then: It should be flagged as novel with high score
        """
        # Given
        detector = KNNNoveltyDetector(k=5)
        detector.fit(known_embeddings)
        
        # When
        novel_embedding = generate_novel_embedding()
        score = detector.score(novel_embedding)
        
        # Then
        assert score > 0.7
        assert detector.is_novel(novel_embedding)
```

### Fixture-Based Testing

```python
# conftest.py
import pytest
from sentinel_ml.config import Config
from sentinel_ml.models import LogRecord

@pytest.fixture
def sample_config():
    """Provide default test configuration."""
    return Config()

@pytest.fixture
def sample_log_record():
    """Provide a sample LogRecord for testing."""
    return LogRecord(
        id="test-123",
        message="User login successful",
        level="INFO",
        timestamp=datetime.now(),
        source="auth-service",
    )

@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings for testing."""
    return np.random.default_rng(42).random((100, 384)).astype(np.float32)

# test_clustering.py
class TestClusteringService:
    def test_cluster_with_fixtures(self, sample_embeddings, sample_config):
        service = ClusteringService(sample_config.clustering)
        result = service.cluster(sample_embeddings)
        assert len(result.labels) == 100
```

---

## Mock Strategies

### Mock Providers

Each service has a mock implementation for testing:

```python
# Production code
class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider | None = None):
        self._provider = provider or SentenceTransformerProvider()

# Test code
class TestEmbeddingService:
    def test_with_mock_provider(self):
        mock_provider = MockEmbeddingProvider()
        service = EmbeddingService(mock_provider)
        
        result = service.embed(["test"])
        
        # Mock provides deterministic output
        assert result.shape == (1, 384)
```

### unittest.mock

```python
from unittest.mock import Mock, patch, MagicMock

class TestLLMService:
    def test_explain_with_mock_response(self):
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"root_cause": "Test", "severity": "low"}'
        }
        
        with patch("requests.post", return_value=mock_response):
            service = LLMService(LLMConfig(provider="ollama"))
            explanation = service.explain_cluster(cluster)
            
            assert explanation.root_cause == "Test"
    
    def test_retry_on_failure(self):
        with patch("requests.post") as mock_post:
            # First two calls fail, third succeeds
            mock_post.side_effect = [
                requests.Timeout(),
                requests.Timeout(),
                Mock(json=lambda: {"response": '{"root_cause": "OK"}'}),
            ]
            
            service = LLMService(LLMConfig(max_retries=3))
            explanation = service.explain_cluster(cluster)
            
            assert mock_post.call_count == 3
```

### Dependency Injection

```python
# Design for testability
class ClusteringService:
    def __init__(
        self,
        config: ClusteringConfig,
        algorithm: ClusteringAlgorithm | None = None,
    ):
        self._algorithm = algorithm or HDBSCANAlgorithm(config)

# In tests
class TestClusteringService:
    def test_with_mock_algorithm(self):
        mock_algorithm = MockClusteringAlgorithm()
        service = ClusteringService(config, algorithm=mock_algorithm)
        
        result = service.cluster(embeddings)
        
        # Assertions on mock behavior
        assert mock_algorithm.cluster_called
```

---

## Test Categories

### Unit Tests

Test individual components in isolation:

```python
class TestNormalization:
    """Unit tests for log normalization."""
    
    def test_mask_ip_address(self):
        result = mask_ip("Connection from 192.168.1.100")
        assert result == "Connection from {IP}"
    
    def test_mask_email(self):
        result = mask_email("User user@example.com logged in")
        assert result == "User {EMAIL} logged in"
    
    def test_mask_multiple_patterns(self):
        text = "User user@test.com from 10.0.0.1 at 2026-01-07"
        result = normalize(text)
        assert "{EMAIL}" in result
        assert "{IP}" in result
        assert "{TIMESTAMP}" in result
```

### Integration Tests

Test component interactions:

```python
@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for the full analysis pipeline."""
    
    def test_full_analysis_pipeline(self, sample_logs):
        # Setup
        config = Config()
        embedding_service = EmbeddingService(config.embedding)
        vector_store = VectorStore(config.vector_store)
        clustering_service = ClusteringService(config.clustering)
        novelty_service = NoveltyService(config.novelty)
        
        # Embed
        embeddings = embedding_service.embed(sample_logs)
        
        # Store
        vector_store.add(ids, embeddings, records)
        
        # Cluster
        clustering_result = clustering_service.cluster(embeddings)
        
        # Detect novelty
        novelty_result = novelty_service.detect(
            embeddings,
            reference_embeddings=embeddings[:50],
        )
        
        # Assertions
        assert len(clustering_result.summaries) > 0
        assert len(novelty_result.scores) == len(sample_logs)
```

### Parameterized Tests

```python
class TestParser:
    @pytest.mark.parametrize("log_line,expected_level", [
        ('{"level": "INFO", "msg": "test"}', "INFO"),
        ('{"level": "ERROR", "msg": "fail"}', "ERROR"),
        ('{"level": "DEBUG", "msg": "trace"}', "DEBUG"),
    ])
    def test_json_parser_levels(self, log_line, expected_level):
        parser = JSONParser()
        record = parser.parse(log_line)
        assert record.level == expected_level
    
    @pytest.mark.parametrize("input_text,expected_masked", [
        ("192.168.1.1", "{IP}"),
        ("10.0.0.255", "{IP}"),
        ("256.1.1.1", "256.1.1.1"),  # Invalid IP, not masked
    ])
    def test_ip_masking(self, input_text, expected_masked):
        result = mask_ip(input_text)
        assert result == expected_masked
```

### Property-Based Tests

```python
from hypothesis import given, strategies as st

class TestVectorStore:
    @given(st.lists(st.floats(min_value=-1, max_value=1), min_size=384, max_size=384))
    def test_add_retrieve_vector(self, vector):
        store = VectorStore(VectorStoreConfig())
        vector_array = np.array([vector], dtype=np.float32)
        
        store.add(["test-id"], vector_array, [mock_record])
        
        result = store.search(vector_array, k=1)
        assert len(result) == 1
        assert result[0].id == "test-id"
```

---

## CLI Testing

### Output Capture

```python
class TestConsole:
    def test_info_output(self):
        console = Console(ConsoleConfig(colors=False))
        
        with console.capture() as captured:
            console.info("Test message")
        
        assert "[INFO] Test message" in captured.stdout
    
    def test_error_output(self):
        console = Console(ConsoleConfig(colors=False))
        
        with console.capture() as captured:
            console.error("Error occurred")
        
        assert "Error occurred" in captured.stderr
```

### Theme Testing

```python
class TestThemes:
    @pytest.mark.parametrize("theme", list(Theme))
    def test_all_themes_have_required_colors(self, theme):
        colors = get_theme(theme)
        
        assert colors.primary is not None
        assert colors.success is not None
        assert colors.warning is not None
        assert colors.error is not None
        assert colors.info is not None
```

### Formatter Testing

```python
class TestJSONFormatter:
    def test_format_dict(self):
        formatter = JSONFormatter()
        data = {"key": "value", "count": 42}
        
        output = formatter.format(data)
        parsed = json.loads(output)
        
        assert parsed == data
    
    def test_format_with_indent(self):
        formatter = JSONFormatter()
        data = {"nested": {"key": "value"}}
        
        output = formatter.format(data, FormatOptions(indent=2))
        
        assert "  " in output  # Has indentation
```

---

## Test Markers

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "requires_ollama: requires Ollama to be running",
    "requires_gpu: requires GPU for execution",
]

# Usage
@pytest.mark.slow
def test_large_dataset_clustering():
    ...

@pytest.mark.integration
def test_grpc_communication():
    ...

@pytest.mark.requires_ollama
def test_llm_explanation():
    ...

# Run without slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m "integration"
```

---

## Coverage Requirements

| Module | Minimum Coverage |
|--------|-----------------|
| `config.py` | 95% |
| `exceptions.py` | 100% |
| `models.py` | 95% |
| `normalization.py` | 90% |
| `parser.py` | 90% |
| `embedding.py` | 85% |
| `vectorstore.py` | 85% |
| `clustering.py` | 85% |
| `novelty.py` | 85% |
| `llm.py` | 80% |
| `cli/` | 85% |

### Checking Coverage

```bash
# Generate coverage report
pytest --cov=sentinel_ml --cov-report=term-missing

# HTML report
pytest --cov=sentinel_ml --cov-report=html
open htmlcov/index.html

# Fail if coverage below threshold
pytest --cov=sentinel_ml --cov-fail-under=85
```

---

## Continuous Integration

Tests run automatically on every push:

```yaml
# .github/workflows/ci.yml
jobs:
  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest tests/python/ --cov=sentinel_ml --cov-fail-under=85
  
  test-go:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - run: go test -race -coverprofile=coverage.out ./...
      - run: go tool cover -func=coverage.out
```

---

*See also: [[Contributing]], [[Code Style]]*
